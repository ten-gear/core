#include "tsgrad/ops.h"
#include "cudahelpers.hpp"
#include <format>
#include <cuda_runtime.h>

namespace tsgrad::cuda {
namespace {

constexpr int32_t LAYERNORM_THREADS = 256;

__device__ long long leading_offset(long long row, const Layout& layout) {
    long long offset = 0;
    for (int32_t dim = layout.rank - 2; dim >= 0; dim--) {
        const long long index = row % layout.shape[dim];
        row /= layout.shape[dim];
        offset += index * static_cast<long long>(layout.stride[dim]);
    }
    return offset;
}

__global__ void layernorm_kernel(
    float* out, Layout out_layout, const float* in, Layout in_layout,
    const float* gamma, Layout gamma_layout, const float* beta, Layout beta_layout,
    float* mean, Layout mean_layout, float* inv_std, Layout inv_std_layout,
    float epsilon, long long features
) {
    const long long row = blockIdx.x;
    const int32_t tid = threadIdx.x;
    const long long in_base = leading_offset(row, in_layout);
    const long long out_base = leading_offset(row, out_layout);
    const long long in_feature_stride = in_layout.stride[in_layout.rank - 1];
    const long long out_feature_stride = out_layout.stride[out_layout.rank - 1];
    __shared__ float partial[LAYERNORM_THREADS];

    float local_sum = 0.0f;
    for (long long col = tid; col < features; col += blockDim.x)
        local_sum += in[in_base + col * in_feature_stride];
    partial[tid] = local_sum;
    __syncthreads();
    for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    const float row_mean = partial[0] / static_cast<float>(features);

    float local_variance = 0.0f;
    for (long long col = tid; col < features; col += blockDim.x) {
        const float centered = in[in_base + col * in_feature_stride] - row_mean;
        local_variance += centered * centered;
    }
    partial[tid] = local_variance;
    __syncthreads();
    for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    const float row_inv_std = rsqrtf(partial[0] / static_cast<float>(features) + epsilon);

    for (long long col = tid; col < features; col += blockDim.x) {
        const float x_hat = (in[in_base + col * in_feature_stride] - row_mean) * row_inv_std;
        out[out_base + col * out_feature_stride] =
            gamma[col * gamma_layout.stride[0]] * x_hat + beta[col * beta_layout.stride[0]];
    }
    if (tid == 0) {
        mean[row * mean_layout.stride[0]] = row_mean;
        inv_std[row * inv_std_layout.stride[0]] = row_inv_std;
    }
}

__global__ void layernorm_backward_input_kernel(
    float* grad_in, Layout grad_in_layout, const float* grad_out, Layout grad_out_layout,
    const float* in, Layout in_layout, const float* gamma, Layout gamma_layout,
    const float* mean, Layout mean_layout, const float* inv_std, Layout inv_std_layout,
    long long features
) {
    const long long row = blockIdx.x;
    const int32_t tid = threadIdx.x;
    const long long grad_in_base = leading_offset(row, grad_in_layout);
    const long long grad_out_base = leading_offset(row, grad_out_layout);
    const long long in_base = leading_offset(row, in_layout);
    const long long grad_in_feature_stride = grad_in_layout.stride[grad_in_layout.rank - 1];
    const long long grad_out_feature_stride = grad_out_layout.stride[grad_out_layout.rank - 1];
    const long long in_feature_stride = in_layout.stride[in_layout.rank - 1];
    const float row_mean = mean[row * mean_layout.stride[0]];
    const float row_inv_std = inv_std[row * inv_std_layout.stride[0]];
    __shared__ float partial_q[LAYERNORM_THREADS];
    __shared__ float partial_q_x_hat[LAYERNORM_THREADS];

    float local_q = 0.0f;
    float local_q_x_hat = 0.0f;
    for (long long col = tid; col < features; col += blockDim.x) {
        const float x_hat = (in[in_base + col * in_feature_stride] - row_mean) * row_inv_std;
        const float q = grad_out[grad_out_base + col * grad_out_feature_stride]
            * gamma[col * gamma_layout.stride[0]];
        local_q += q;
        local_q_x_hat += q * x_hat;
    }
    partial_q[tid] = local_q;
    partial_q_x_hat[tid] = local_q_x_hat;
    __syncthreads();
    for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial_q[tid] += partial_q[tid + stride];
            partial_q_x_hat[tid] += partial_q_x_hat[tid + stride];
        }
        __syncthreads();
    }

    const float mean_q = partial_q[0] / static_cast<float>(features);
    const float mean_q_x_hat = partial_q_x_hat[0] / static_cast<float>(features);
    for (long long col = tid; col < features; col += blockDim.x) {
        const float x_hat = (in[in_base + col * in_feature_stride] - row_mean) * row_inv_std;
        const float q = grad_out[grad_out_base + col * grad_out_feature_stride]
            * gamma[col * gamma_layout.stride[0]];
        grad_in[grad_in_base + col * grad_in_feature_stride] =
            row_inv_std * (q - mean_q - x_hat * mean_q_x_hat);
    }
}

__global__ void layernorm_backward_params_kernel(
    float* grad_gamma, Layout grad_gamma_layout, float* grad_beta, Layout grad_beta_layout,
    const float* grad_out, Layout grad_out_layout, const float* in, Layout in_layout,
    const float* mean, Layout mean_layout, const float* inv_std, Layout inv_std_layout,
    long long rows, long long features
) {
    const long long col = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (col >= features) return;

    const long long grad_out_feature_stride = grad_out_layout.stride[grad_out_layout.rank - 1];
    const long long in_feature_stride = in_layout.stride[in_layout.rank - 1];
    float gamma_sum = 0.0f;
    float beta_sum = 0.0f;
    for (long long row = 0; row < rows; row++) {
        const long long grad_out_base = leading_offset(row, grad_out_layout);
        const long long in_base = leading_offset(row, in_layout);
        const float dy = grad_out[grad_out_base + col * grad_out_feature_stride];
        const float x_hat = (in[in_base + col * in_feature_stride]
            - mean[row * mean_layout.stride[0]]) * inv_std[row * inv_std_layout.stride[0]];
        gamma_sum += dy * x_hat;
        beta_sum += dy;
    }
    grad_gamma[col * grad_gamma_layout.stride[0]] = gamma_sum;
    grad_beta[col * grad_beta_layout.stride[0]] = beta_sum;
}

void require_cuda_rank(const View& view, const char* name) {
    if (view.rank > 16)
        throw std::runtime_error(std::format(
            "CUDA layernorm supports ranks up to 16, received {}.rank: {}", name, view.rank));
}

} // namespace

void layernorm(
    const View& out, const View& in, const View& gamma, const View& beta,
    const View& mean, const View& inv_std, const float epsilon
) {
    require_cuda_rank(out, "out");
    require_cuda_rank(in, "in");
    Context& ctx = Context::get();
    const long long features = in.shape[in.rank - 1];
    const long long rows = prod(in.shape, in.rank) / features;
    if (rows == 0) return;

    layernorm_kernel<<<static_cast<uint32_t>(rows), LAYERNORM_THREADS, 0,
        static_cast<cudaStream_t>(ctx.getStream())>>>(
        out.data, layout_from(out), in.data, layout_from(in),
        gamma.data, layout_from(gamma), beta.data, layout_from(beta),
        mean.data, layout_from(mean), inv_std.data, layout_from(inv_std), epsilon, features);
    ctx.checkLastError();
}

void layernorm_backward(
    const View& grad_in, const View& grad_gamma, const View& grad_beta,
    const View& grad_out, const View& in, const View& gamma,
    const View& mean, const View& inv_std
) {
    require_cuda_rank(grad_in, "grad_in");
    require_cuda_rank(grad_out, "grad_out");
    require_cuda_rank(in, "in");
    Context& ctx = Context::get();
    const long long features = in.shape[in.rank - 1];
    const long long rows = prod(in.shape, in.rank) / features;

    if (rows > 0) {
        layernorm_backward_input_kernel<<<static_cast<uint32_t>(rows), LAYERNORM_THREADS, 0,
            static_cast<cudaStream_t>(ctx.getStream())>>>(
            grad_in.data, layout_from(grad_in), grad_out.data, layout_from(grad_out),
            in.data, layout_from(in), gamma.data, layout_from(gamma),
            mean.data, layout_from(mean), inv_std.data, layout_from(inv_std), features);
        ctx.checkLastError();
    }

    const int32_t blocks = static_cast<int32_t>((features + LAYERNORM_THREADS - 1) / LAYERNORM_THREADS);
    layernorm_backward_params_kernel<<<blocks, LAYERNORM_THREADS, 0,
        static_cast<cudaStream_t>(ctx.getStream())>>>(
        grad_gamma.data, layout_from(grad_gamma), grad_beta.data, layout_from(grad_beta),
        grad_out.data, layout_from(grad_out), in.data, layout_from(in),
        mean.data, layout_from(mean), inv_std.data, layout_from(inv_std), rows, features);
    ctx.checkLastError();
}
}
