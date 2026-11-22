#include "tsgrad/ops.h"
#include "helpers.hpp"
#include <format>
#include <span>

namespace tsgrad {
namespace {

// row counts head vectors, with the sequence axis varying fastest among the
// leading dimensions. The returned base does not include the last dimension.
long long leading_offset(long long row, const View& view) {
    long long offset = 0;
    for (int32_t dim = view.rank - 2; dim >= 0; dim--) {
        const long long index = row % view.shape[dim];
        row /= view.shape[dim];
        offset += index * static_cast<long long>(view.stride[dim]);
    }
    return offset;
}

void validate(const View& out, const View& in, const View& sin, const View& cos, int32_t offset) {
    if (!shapesAreEqual(out.shape, in.shape, out.rank, in.rank)) {
        std::span<const int32_t> out_shape{out.shape, static_cast<size_t>(out.rank)};
        std::span<const int32_t> in_shape{in.shape, static_cast<size_t>(in.rank)};
        throw std::runtime_error(std::format(
            "rope received out and in with different shapes, received out.shape: {:n}, in.shape: {:n}",
            out_shape, in_shape));
    }

    const bool shape_ok = in.rank >= 2 && in.shape[in.rank - 1] > 0 && (in.shape[in.rank - 1] % 2) == 0;
    if (!shape_ok) {
        std::span<const int32_t> in_shape{in.shape, static_cast<size_t>(in.rank)};
        throw std::runtime_error(std::format(
            "rope received bad in.shape, received in.shape: {:n}", in_shape));
    }
    if (offset < 0)
        throw std::runtime_error(std::format("rope received a negative offset: {}", offset));

    const long long half = static_cast<long long>(in.shape[in.rank - 1]) / 2;
    const long long needed = static_cast<long long>(in.shape[in.rank - 2]) + offset;
    const bool table_ok = sin.rank == 2 && cos.rank == 2
        && sin.shape[1] == half && cos.shape[1] == half
        && static_cast<long long>(sin.shape[0]) >= needed
        && static_cast<long long>(cos.shape[0]) >= needed;
    if (!table_ok) {
        std::span<const int32_t> sin_shape{sin.shape, static_cast<size_t>(sin.rank)};
        std::span<const int32_t> cos_shape{cos.shape, static_cast<size_t>(cos.rank)};
        throw std::runtime_error(std::format(
            "rope received invalid sin/cos shapes, expected at least [{}, {}] for both, received sin.shape: {:n}, cos.shape: {:n}",
            needed, half, sin_shape, cos_shape));
    }
}

void rope_impl(
    const View& out, const View& in, const View& sin, const View& cos,
    const int32_t offset, const bool backward
) {
    validate(out, in, sin, cos, offset);

    if constexpr (TSGRAD_HAS_CUDA) {
        if (backward)
            tsgrad::cuda::rope_backward(out, in, sin, cos, offset);
        else
            tsgrad::cuda::rope(out, in, sin, cos, offset);
        return;
    }

    const long long features = in.shape[in.rank - 1];
    const long long half = features / 2;
    const long long seq = in.shape[in.rank - 2];
    const long long rows = prod(in.shape, in.rank) / features;
    if (rows == 0) return;

    const long long in_feature_stride = in.stride[in.rank - 1];
    const long long out_feature_stride = out.stride[out.rank - 1];
    const long long sin_row_stride = sin.stride[0];
    const long long sin_freq_stride = sin.stride[1];
    const long long cos_row_stride = cos.stride[0];
    const long long cos_freq_stride = cos.stride[1];

    #pragma omp parallel for if (rows * features >= THRESHOLD)
    for (long long row = 0; row < rows; row++) {
        const long long position = (row % seq) + offset;
        const long long in_base = leading_offset(row, in);
        const long long out_base = leading_offset(row, out);
        const long long sin_base = position * sin_row_stride;
        const long long cos_base = position * cos_row_stride;

        for (long long col = 0; col < features; col++) {
            const long long freq = col < half ? col : col - half;
            const long long partner = col < half ? col + half : col - half;
            const float sign = col < half
                ? (backward ? 1.0f : -1.0f)
                : (backward ? -1.0f : 1.0f);
            const float c = cos.data[cos_base + freq * cos_freq_stride];
            const float s = sin.data[sin_base + freq * sin_freq_stride];
            const float x0 = in.data[in_base + col * in_feature_stride];
            const float x1 = in.data[in_base + partner * in_feature_stride];
            out.data[out_base + col * out_feature_stride] = x0 * c + sign * x1 * s;
        }
    }
}

} // namespace

void rope(const View& out, const View& in, const View& sin, const View& cos, const int32_t offset) {
    rope_impl(out, in, sin, cos, offset, false);
}

void rope_backward(
    const View& grad_in, const View& grad_out, const View& sin, const View& cos,
    const int32_t offset
) {
    rope_impl(grad_in, grad_out, sin, cos, offset, true);
}

} // namespace tsgrad
