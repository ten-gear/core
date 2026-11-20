#include "tsgrad/ops.h"
#include "cudahelpers.hpp"
#include <cuda_runtime.h>
#include <format>

namespace tsgrad::cuda {
namespace {

constexpr int32_t SOFTMAX_THREADS = 256;

__device__ long long slice_offset(long long row, const Layout& layout, int32_t dim) {
    long long offset = 0;
    for (int32_t axis = layout.rank - 1; axis >= 0; axis--) {
        if (axis == dim) continue;
        const long long index = row % layout.shape[axis];
        row /= layout.shape[axis];
        offset += index * static_cast<long long>(layout.stride[axis]);
    }
    return offset;
}

template <bool Log>
__global__ void softmax_kernel(
    float* out, Layout out_layout, const float* input, Layout input_layout,
    int32_t dim, long long dim_size
) {
    const long long row = blockIdx.x;
    const int32_t tid = threadIdx.x;
    const long long out_base = slice_offset(row, out_layout, dim);
    const long long input_base = slice_offset(row, input_layout, dim);
    __shared__ float partial[SOFTMAX_THREADS];

    float local_max = -CUDART_INF_F;
    for (long long col = tid; col < dim_size; col += blockDim.x)
        local_max = fmaxf(local_max, input[input_base + col * input_layout.stride[dim]]);
    partial[tid] = local_max;
    __syncthreads();
    for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] = fmaxf(partial[tid], partial[tid + stride]);
        __syncthreads();
    }
    const float maximum = partial[0];

    float local_sum = 0.0f;
    for (long long col = tid; col < dim_size; col += blockDim.x)
        local_sum += expf(input[input_base + col * input_layout.stride[dim]] - maximum);
    partial[tid] = local_sum;
    __syncthreads();
    for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }

    if constexpr (Log) {
        const float log_sum_exp = maximum + logf(partial[0]);
        for (long long col = tid; col < dim_size; col += blockDim.x)
            out[out_base + col * out_layout.stride[dim]] =
                input[input_base + col * input_layout.stride[dim]] - log_sum_exp;
    } else {
        const float inverse_sum = 1.0f / partial[0];
        for (long long col = tid; col < dim_size; col += blockDim.x)
            out[out_base + col * out_layout.stride[dim]] =
                expf(input[input_base + col * input_layout.stride[dim]] - maximum) * inverse_sum;
    }
}

template <bool Log>
__global__ void softmax_backward_kernel(
    float* grad_in, Layout grad_in_layout,
    const float* grad_out, Layout grad_out_layout,
    const float* out, Layout out_layout,
    int32_t dim, long long dim_size
) {
    const long long row = blockIdx.x;
    const int32_t tid = threadIdx.x;
    const long long grad_in_base = slice_offset(row, grad_in_layout, dim);
    const long long grad_out_base = slice_offset(row, grad_out_layout, dim);
    const long long out_base = slice_offset(row, out_layout, dim);
    __shared__ float partial[SOFTMAX_THREADS];

    float local_sum = 0.0f;
    for (long long col = tid; col < dim_size; col += blockDim.x) {
        const float dy = grad_out[grad_out_base + col * grad_out_layout.stride[dim]];
        if constexpr (Log)
            local_sum += dy;
        else
            local_sum += dy * out[out_base + col * out_layout.stride[dim]];
    }
    partial[tid] = local_sum;
    __syncthreads();
    for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    const float reduced = partial[0];

    for (long long col = tid; col < dim_size; col += blockDim.x) {
        const float dy = grad_out[grad_out_base + col * grad_out_layout.stride[dim]];
        const float y = out[out_base + col * out_layout.stride[dim]];
        grad_in[grad_in_base + col * grad_in_layout.stride[dim]] = Log ? dy - expf(y) * reduced : y * (dy - reduced);
    }
}

void require_rank(const View& view, const char* operation, const char* name) {
    if (view.rank > 16)
        throw std::runtime_error(std::format(
            "CUDA {} supports ranks up to 16, received {}.rank: {}", operation, name, view.rank));
}

template <bool Log>
void forward(const View& out, const View& input, int32_t dim, const char* operation) {
    require_rank(out, operation, "out");
    require_rank(input, operation, "input");
    const long long dim_size = input.shape[dim];
    const long long rows = prod(input.shape, input.rank) / dim_size;
    if (rows == 0) return;

    Context& ctx = Context::get();
    softmax_kernel<Log><<<static_cast<uint32_t>(rows), SOFTMAX_THREADS, 0,
        static_cast<cudaStream_t>(ctx.getStream())>>>(
        out.data, layout_from(out), input.data, layout_from(input), dim, dim_size);
    ctx.checkLastError();
}

template <bool Log>
void backward(
    const View& grad_in, const View& grad_out, const View& out,
    int32_t dim, const char* operation
) {
    require_rank(grad_in, operation, "grad_in");
    require_rank(grad_out, operation, "grad_out");
    require_rank(out, operation, "out");
    const long long dim_size = out.shape[dim];
    const long long rows = prod(out.shape, out.rank) / dim_size;
    if (rows == 0) return;

    Context& ctx = Context::get();
    softmax_backward_kernel<Log><<<static_cast<uint32_t>(rows), SOFTMAX_THREADS, 0,
        static_cast<cudaStream_t>(ctx.getStream())>>>(
        grad_in.data, layout_from(grad_in),
        grad_out.data, layout_from(grad_out),
        out.data, layout_from(out), dim, dim_size);
    ctx.checkLastError();
}

} // namespace

void softmax(const View& out, const View& input, int32_t dim) {
    forward<false>(out, input, dim, "softmax");
}

void softmax_backward(const View& grad_in, const View& grad_out, const View& out, int32_t dim) {
    backward<false>(grad_in, grad_out, out, dim, "softmax_backward");
}

void logsoftmax(const View& out, const View& input, int32_t dim) {
    forward<true>(out, input, dim, "logsoftmax");
}

void logsoftmax_backward(const View& grad_in, const View& grad_out, const View& out, int32_t dim) {
    backward<true>(grad_in, grad_out, out, dim, "logsoftmax_backward");
}

} // namespace tsgrad::cuda
