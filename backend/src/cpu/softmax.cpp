#include "tsgrad/ops.h"
#include "helpers.hpp"
#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <span>
#include <utility>

namespace tsgrad {
namespace {

long long slice_offset(long long row, const View& view, int32_t dim) {
    long long offset = 0;
    for (int32_t axis = view.rank - 1; axis >= 0; axis--) {
        if (axis == dim) continue;
        const long long index = row % view.shape[axis];
        row /= view.shape[axis];
        offset += index * static_cast<long long>(view.stride[axis]);
    }
    return offset;
}

void require_same_shape(
    const View& a, const View& b, const char* operation,
    const char* a_name, const char* b_name
) {
    if (shapesAreEqual(a.shape, b.shape, a.rank, b.rank)) return;
    std::span<const int32_t> a_shape{a.shape, static_cast<size_t>(a.rank)};
    std::span<const int32_t> b_shape{b.shape, static_cast<size_t>(b.rank)};
    throw std::runtime_error(std::format(
        "{} received {} and {} with different shapes, received {}: {}, {}: {}",
        operation, a_name, b_name, a_name, shapeText(a_shape), b_name, shapeText(b_shape)));
}

std::pair<long long, long long> dimensions(const View& input, int32_t dim, const char* operation) {
    if (input.rank < 1)
        throw std::runtime_error(std::format("{} expects an input with rank at least 1", operation));
    if (dim < 0 || dim >= input.rank)
        throw std::runtime_error(std::format("{} received dim outside the input rank", operation));
    const long long dim_size = input.shape[dim];
    if (dim_size <= 0)
        throw std::runtime_error(std::format("{} requires a non-empty normalized dimension", operation));
    return {prod(input.shape, input.rank) / dim_size, dim_size};
}

template <bool Log>
void softmax_forward(const View& out, const View& input, int32_t dim, const char* operation) {
    require_same_shape(out, input, operation, "out.shape", "input.shape");
    const auto shape = dimensions(input, dim, operation);
    const long long rows = shape.first;
    const long long dim_size = shape.second;

    if constexpr (TSGRAD_HAS_CUDA) {
        if constexpr (Log)
            tsgrad::cuda::logsoftmax(out, input, dim);
        else
            tsgrad::cuda::softmax(out, input, dim);
        return;
    }

    #pragma omp parallel for if (rows * dim_size >= THRESHOLD)
    for (long long row = 0; row < rows; row++) {
        const long long input_base = slice_offset(row, input, dim);
        const long long out_base = slice_offset(row, out, dim);
        float maximum = -std::numeric_limits<float>::infinity();
        for (long long col = 0; col < dim_size; col++)
            maximum = std::max(maximum, input.data[input_base + col * input.stride[dim]]);

        float sum_exp = 0.0f;
        for (long long col = 0; col < dim_size; col++)
            sum_exp += std::exp(input.data[input_base + col * input.stride[dim]] - maximum);

        if constexpr (Log) {
            const float log_sum_exp = maximum + std::log(sum_exp);
            for (long long col = 0; col < dim_size; col++)
                out.data[out_base + col * out.stride[dim]] =
                    input.data[input_base + col * input.stride[dim]] - log_sum_exp;
        } else {
            const float inverse_sum = 1.0f / sum_exp;
            for (long long col = 0; col < dim_size; col++)
                out.data[out_base + col * out.stride[dim]] =
                    std::exp(input.data[input_base + col * input.stride[dim]] - maximum) * inverse_sum;
        }
    }
}

template <bool Log>
void softmax_backward_impl(
    const View& grad_in, const View& grad_out, const View& out,
    int32_t dim, const char* operation
) {
    require_same_shape(grad_in, out, operation, "grad_in.shape", "out.shape");
    require_same_shape(grad_out, out, operation, "grad_out.shape", "out.shape");
    const auto shape = dimensions(out, dim, operation);
    const long long rows = shape.first;
    const long long dim_size = shape.second;

    if constexpr (TSGRAD_HAS_CUDA) {
        if constexpr (Log)
            tsgrad::cuda::logsoftmax_backward(grad_in, grad_out, out, dim);
        else
            tsgrad::cuda::softmax_backward(grad_in, grad_out, out, dim);
        return;
    }

    #pragma omp parallel for if (rows * dim_size >= THRESHOLD)
    for (long long row = 0; row < rows; row++) {
        const long long grad_in_base = slice_offset(row, grad_in, dim);
        const long long grad_out_base = slice_offset(row, grad_out, dim);
        const long long out_base = slice_offset(row, out, dim);
        float reduced = 0.0f;
        for (long long col = 0; col < dim_size; col++) {
            const float dy = grad_out.data[grad_out_base + col * grad_out.stride[dim]];
            if constexpr (Log)
                reduced += dy;
            else
                reduced += dy * out.data[out_base + col * out.stride[dim]];
        }

        for (long long col = 0; col < dim_size; col++) {
            const float dy = grad_out.data[grad_out_base + col * grad_out.stride[dim]];
            const float y = out.data[out_base + col * out.stride[dim]];
            grad_in.data[grad_in_base + col * grad_in.stride[dim]] =
                Log ? dy - std::exp(y) * reduced : y * (dy - reduced);
        }
    }
}

} // namespace

void softmax(const View& out, const View& input, int32_t dim) {
    softmax_forward<false>(out, input, dim, "softmax");
}

void softmax_backward(const View& grad_in, const View& grad_out, const View& out, int32_t dim) {
    softmax_backward_impl<false>(grad_in, grad_out, out, dim, "softmax_backward");
}

void logsoftmax(const View& out, const View& input, int32_t dim) {
    softmax_forward<true>(out, input, dim, "logsoftmax");
}

void logsoftmax_backward(const View& grad_in, const View& grad_out, const View& out, int32_t dim) {
    softmax_backward_impl<true>(grad_in, grad_out, out, dim, "logsoftmax_backward");
}

} // namespace tsgrad
