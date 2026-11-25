#include "tsgrad/ops.h"
#include "cudahelpers.hpp"
#include <format>
#include <cuda_runtime.h>
#include <limits>
#include <string>

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

// Same row split as the contiguous bias reduction. One thread per channel
// walking every row is a handful of blocks; each block instead keeps 256
// consecutive channels and a short run of rows.
struct RowChunkPlan {
    int32_t chunks;
    int32_t rows_per_chunk;
    int32_t blocks_x;
};

RowChunkPlan plan_row_chunks(int32_t reduce_len, int32_t width) {
    constexpr int32_t min_rows_per_chunk = 32;
    // Same 32-row chunks as the bias reduction, capped at a 64 MB partial buffer.
    constexpr int32_t max_partial_floats = 16 * 1024 * 1024;
    const int32_t blocks_x = static_cast<int32_t>((width + LAYERNORM_THREADS - 1) / LAYERNORM_THREADS);
    int32_t chunks = (reduce_len + min_rows_per_chunk - 1) / min_rows_per_chunk;
    if (chunks < 1) chunks = 1;
    const int32_t max_chunks = std::max(1, max_partial_floats / std::max(width, 1));
    if (chunks > max_chunks) chunks = max_chunks;
    const int32_t rows_per_chunk = (reduce_len + chunks - 1) / chunks;
    return {chunks, rows_per_chunk, blocks_x};
}

float* param_partial_scratch(size_t floats, cudaStream_t stream) {
    struct Buffer {
        float* data = nullptr;
        size_t floats = 0;
    };
    static Buffer buffer;
    if (buffer.floats >= floats) return buffer.data;

    if (buffer.data != nullptr) {
        cudaError_t synced = cudaStreamSynchronize(stream);
        if (synced != cudaSuccess)
            throw std::runtime_error(std::string("cudaStreamSynchronize: ") + cudaGetErrorString(synced));
        cudaError_t freed = cudaFree(buffer.data);
        buffer.data = nullptr;
        buffer.floats = 0;
        if (freed != cudaSuccess)
            throw std::runtime_error(std::string("cudaFree: ") + cudaGetErrorString(freed));
    }

    cudaError_t allocated = cudaMalloc(&buffer.data, floats * sizeof(float));
    if (allocated != cudaSuccess) {
        buffer.data = nullptr;
        buffer.floats = 0;
        throw std::runtime_error(std::string("CUDA layernorm parameter backward could not allocate partials: ") + cudaGetErrorString(allocated));
    }
    buffer.floats = floats;
    return buffer.data;
}

__global__ void layernorm_backward_params_partial_kernel(
    float* partial_gamma, float* partial_beta,
    const float* grad_out, const float* in, const float* mean, const float* inv_std,
    int32_t rows, int32_t features, int32_t rows_per_chunk
) {
    const int32_t col = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (col < 0 || col >= features) return;

    const int32_t chunk = static_cast<int32_t>(blockIdx.y);
    const long long row0 = static_cast<long long>(chunk) * rows_per_chunk;
    const long long slot = static_cast<long long>(chunk) * features + col;
    if (row0 >= rows) {
        partial_gamma[slot] = 0.0f;
        partial_beta[slot] = 0.0f;
        return;
    }
    long long row1 = row0 + rows_per_chunk;
    if (row1 > rows) row1 = rows;

    float g0 = 0.0f, g1 = 0.0f, g2 = 0.0f, g3 = 0.0f;
    float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f, b3 = 0.0f;
    long long row = row0;
    const long long tail = row0 + ((row1 - row0) & ~3LL);
    for (; row < tail; row += 4) {
        const long long o0 = row * features + col;
        const long long o1 = o0 + features;
        const long long o2 = o1 + features;
        const long long o3 = o2 + features;
        const float dy0 = grad_out[o0];
        const float dy1 = grad_out[o1];
        const float dy2 = grad_out[o2];
        const float dy3 = grad_out[o3];
        const float x0 = (in[o0] - mean[row]) * inv_std[row];
        const float x1 = (in[o1] - mean[row + 1]) * inv_std[row + 1];
        const float x2 = (in[o2] - mean[row + 2]) * inv_std[row + 2];
        const float x3 = (in[o3] - mean[row + 3]) * inv_std[row + 3];
        g0 += dy0 * x0;
        g1 += dy1 * x1;
        g2 += dy2 * x2;
        g3 += dy3 * x3;
        b0 += dy0;
        b1 += dy1;
        b2 += dy2;
        b3 += dy3;
    }
    float gamma_sum = (g0 + g1) + (g2 + g3);
    float beta_sum = (b0 + b1) + (b2 + b3);
    for (; row < row1; ++row) {
        const long long offset = row * features + col;
        const float dy = grad_out[offset];
        const float x_hat = (in[offset] - mean[row]) * inv_std[row];
        gamma_sum += dy * x_hat;
        beta_sum += dy;
    }
    partial_gamma[slot] = gamma_sum;
    partial_beta[slot] = beta_sum;
}

__global__ void layernorm_backward_params_finalize_kernel(
    float* grad_gamma, long long gamma_stride,
    float* grad_beta, long long beta_stride,
    const float* partial_gamma, const float* partial_beta,
    int32_t chunks, int32_t features
) {
    const int32_t col = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (col < 0 || col >= features) return;

    float gamma_sum = 0.0f;
    float beta_sum = 0.0f;
    for (int32_t chunk = 0; chunk < chunks; ++chunk) {
        const long long slot = static_cast<long long>(chunk) * features + col;
        gamma_sum += partial_gamma[slot];
        beta_sum += partial_beta[slot];
    }
    grad_gamma[static_cast<long long>(col) * gamma_stride] = gamma_sum;
    grad_beta[static_cast<long long>(col) * beta_stride] = beta_sum;
}

bool packed_last_stride(const View& view) {
    return view.rank >= 1 && view.stride[view.rank - 1] == 1;
}

bool launch_packed_param_grads(
    const View& grad_gamma, const View& grad_beta,
    const View& grad_out, const View& in, const View& mean, const View& inv_std,
    long long rows, long long features
) {
    if (rows < 256 || features < 32) return false;
    if (rows > std::numeric_limits<int32_t>::max() || features > std::numeric_limits<int32_t>::max())
        return false;
    if (grad_gamma.rank < 1 || grad_beta.rank < 1) return false;
    // Forward stores mean[row * stride[0]]. Rank-1 contiguous storage is mean[row].
    if (mean.rank != 1 || inv_std.rank != 1 || mean.stride[0] != 1 || inv_std.stride[0] != 1)
        return false;
    if (!is_contiguous(grad_out) || !is_contiguous(in) || !is_contiguous(mean) || !is_contiguous(inv_std))
        return false;
    if (!packed_last_stride(grad_out) || !packed_last_stride(in)) return false;
    if (in.rank < 1 || grad_out.rank < 1) return false;
    if (in.shape[in.rank - 1] != features || grad_out.shape[grad_out.rank - 1] != features)
        return false;
    if (prod(in.shape, in.rank) != rows * features || prod(grad_out.shape, grad_out.rank) != rows * features)
        return false;
    if (prod(mean.shape, mean.rank) != rows || prod(inv_std.shape, inv_std.rank) != rows)
        return false;

    const int32_t feature_count = static_cast<int32_t>(features);
    const RowChunkPlan plan = plan_row_chunks(static_cast<int32_t>(rows), feature_count);
    Context& ctx = Context::get();
    cudaStream_t stream = static_cast<cudaStream_t>(ctx.getStream());
    const size_t partial_floats = static_cast<size_t>(plan.chunks) * static_cast<size_t>(feature_count);
    float* scratch = param_partial_scratch(partial_floats * 2, stream);
    float* partial_gamma = scratch;
    float* partial_beta = scratch + partial_floats;

    layernorm_backward_params_partial_kernel<<<dim3(plan.blocks_x, plan.chunks), LAYERNORM_THREADS, 0, stream>>>(
        partial_gamma, partial_beta, grad_out.data, in.data, mean.data, inv_std.data,
        static_cast<int32_t>(rows), feature_count, plan.rows_per_chunk);
    ctx.checkLastError();
    layernorm_backward_params_finalize_kernel<<<plan.blocks_x, LAYERNORM_THREADS, 0, stream>>>(
        grad_gamma.data, grad_gamma.stride[0], grad_beta.data, grad_beta.stride[0],
        partial_gamma, partial_beta, plan.chunks, feature_count);
    ctx.checkLastError();
    return true;
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

    if (!launch_packed_param_grads(
            grad_gamma, grad_beta, grad_out, in, mean, inv_std, rows, features)) {
        const int32_t blocks = static_cast<int32_t>((features + LAYERNORM_THREADS - 1) / LAYERNORM_THREADS);
        layernorm_backward_params_kernel<<<blocks, LAYERNORM_THREADS, 0,
            static_cast<cudaStream_t>(ctx.getStream())>>>(
            grad_gamma.data, layout_from(grad_gamma), grad_beta.data, layout_from(grad_beta),
            grad_out.data, layout_from(grad_out), in.data, layout_from(in),
            mean.data, layout_from(mean), inv_std.data, layout_from(inv_std), rows, features);
        ctx.checkLastError();
    }
}
}
