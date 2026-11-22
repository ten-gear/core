#include "tsgrad/ops.h"
#include "cudahelpers.hpp"
#include <cuda_runtime.h>
#include <format>

namespace tsgrad::cuda {
namespace {
__device__ long long leading_offset(long long row, const Layout& layout) {
    long long offset = 0;
    for (int32_t dim = layout.rank - 2; dim >= 0; dim--) {
        const long long index = row % layout.shape[dim];
        row /= layout.shape[dim];
        offset += index * static_cast<long long>(layout.stride[dim]);
    }
    return offset;
}

__global__ void rope_kernel(
    float* out, Layout out_layout, const float* in, Layout in_layout,
    const float* sin, Layout sin_layout, const float* cos, Layout cos_layout,
    long long features, long long half, int32_t offset, bool backward
) {
    const long long row = blockIdx.x;
    const int32_t tid = threadIdx.x;
    const long long position = (row % in_layout.shape[in_layout.rank - 2]) + offset;
    const long long in_base = leading_offset(row, in_layout);
    const long long out_base = leading_offset(row, out_layout);
    const long long in_feature_stride = in_layout.stride[in_layout.rank - 1];
    const long long out_feature_stride = out_layout.stride[out_layout.rank - 1];
    const long long sin_base = position * sin_layout.stride[0];
    const long long cos_base = position * cos_layout.stride[0];

    for (long long col = tid; col < features; col += blockDim.x) {
        const long long freq = col < half ? col : col - half;
        const long long partner = col < half ? col + half : col - half;
        const float sign = col < half
            ? (backward ? 1.0f : -1.0f)
            : (backward ? -1.0f : 1.0f);
        const float c = cos[cos_base + freq * cos_layout.stride[1]];
        const float s = sin[sin_base + freq * sin_layout.stride[1]];
        const float x0 = in[in_base + col * in_feature_stride];
        const float x1 = in[in_base + partner * in_feature_stride];
        out[out_base + col * out_feature_stride] = x0 * c + sign * x1 * s;
    }
}

void require_cuda_rank(const View& view, const char* name) {
    if (view.rank > 16)
        throw std::runtime_error(std::format("CUDA rope supports ranks up to 16, received {}.rank: {}", name, view.rank));
}

void launch_rope(
    const View& out, const View& in, const View& sin, const View& cos,
    const int32_t offset, const bool backward
) {
    require_cuda_rank(out, "out");
    require_cuda_rank(in, "in");
    require_cuda_rank(sin, "sin");
    require_cuda_rank(cos, "cos");

    const long long features = in.shape[in.rank - 1];
    const long long rows = prod(in.shape, in.rank) / features;
    if (rows == 0) return;

    Context& ctx = Context::get();
    rope_kernel<<<static_cast<uint32_t>(rows), 256, 0, static_cast<cudaStream_t>(ctx.getStream())>>>(
        out.data, layout_from(out), in.data, layout_from(in),
        sin.data, layout_from(sin), cos.data, layout_from(cos),
        features, features / 2, offset, backward);
    ctx.checkLastError();
}

} // namespace

void rope(const View& out, const View& in, const View& sin, const View& cos, const int32_t offset) {
    launch_rope(out, in, sin, cos, offset, false);
}

void rope_backward(
    const View& grad_in, const View& grad_out, const View& sin, const View& cos,
    const int32_t offset
) {
    launch_rope(grad_in, grad_out, sin, cos, offset, true);
}

} // namespace tsgrad::cuda
