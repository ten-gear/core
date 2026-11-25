#include "tsgrad/ops.h"
#include "helpers.hpp"
#include <format>
#include <span>
#include <cmath>
#include <utility>

namespace tsgrad{
namespace {

long long leading_offset(long long row, const View& view) {
    long long offset = 0;
    for (int32_t dim = view.rank - 2; dim >= 0; dim--) {
        const long long index = row % view.shape[dim];
        row /= view.shape[dim];
        offset += index * static_cast<long long>(view.stride[dim]);
    }
    return offset;
}

void require_same_shape(const View& a, const View& b, const char* operation, const char* a_name, const char* b_name) {
    if (shapesAreEqual(a.shape, b.shape, a.rank, b.rank)) return;
    std::span<const int32_t> a_shape{a.shape, static_cast<size_t>(a.rank)};
    std::span<const int32_t> b_shape{b.shape, static_cast<size_t>(b.rank)};
    throw std::runtime_error(std::format(
        "{} received {} and {} with different shapes, received {}: {}, {}: {}",
        operation, a_name, b_name, a_name, shapeText(a_shape), b_name, shapeText(b_shape)));
}

std::pair<long long, long long> validate_common(
    const View& in, const View& gamma, const View& mean, const View& inv_std, const char* operation
) {
    if (in.rank < 1)
        throw std::runtime_error(std::format("{} expects an input with rank at least 1", operation));
    if (gamma.rank != 1)
        throw std::runtime_error(std::format("{} expects gamma to have rank 1", operation));
    if (mean.rank != 1 || inv_std.rank != 1)
        throw std::runtime_error(std::format("{} expects mean and inv_std to have rank 1", operation));

    const long long features = in.shape[in.rank - 1];
    if (features <= 0)
        throw std::runtime_error(std::format("{} requires a non-empty normalized dimension", operation));
    if (gamma.shape[0] != features)
        throw std::runtime_error(std::format("{} gamma shape mismatch", operation));

    const long long rows = prod(in.shape, in.rank) / features;
    if (mean.shape[0] != rows || inv_std.shape[0] != rows)
        throw std::runtime_error(std::format("{} saved-stat shape mismatch", operation));
    return {rows, features};
}

} // namespace

void layernorm(
    const View& out, const View& in, const View& gamma, const View& beta,
    const View& mean, const View& inv_std, const float epsilon
) {
    require_same_shape(out, in, "layernorm", "out.shape", "in.shape");
    const auto shape = validate_common(in, gamma, mean, inv_std, "layernorm");
    const long long rows = shape.first;
    const long long features = shape.second;
    if (beta.rank != 1 || beta.shape[0] != features)
        throw std::runtime_error("layernorm beta shape mismatch");
    if (!std::isfinite(epsilon) || epsilon < 0.0f)
        throw std::runtime_error(std::format("layernorm expects a finite non-negative epsilon, received: {}", epsilon));

    if constexpr (TSGRAD_HAS_CUDA) {
        tsgrad::cuda::layernorm(out, in, gamma, beta, mean, inv_std, epsilon);
        return;
    }

    const long long in_feature_stride = in.stride[in.rank - 1];
    const long long out_feature_stride = out.stride[out.rank - 1];
    const long long gamma_stride = gamma.stride[0];
    const long long beta_stride = beta.stride[0];

    #pragma omp parallel for if (rows * features >= THRESHOLD)
    for (long long row = 0; row < rows; row++) {
        const long long in_base = leading_offset(row, in);
        const long long out_base = leading_offset(row, out);

        // Welford's algorithm keeps the variance stable for large offsets.
        float row_mean = 0.0f;
        float m2 = 0.0f;
        long long count = 0;
        for (long long col = 0; col < features; col++) {
            const float x = in.data[in_base + col * in_feature_stride];
            count++;
            const float delta = x - row_mean;
            row_mean += delta / static_cast<float>(count);
            m2 += delta * (x - row_mean);
        }

        const float row_inv_std = 1.0f / std::sqrt(m2 / static_cast<float>(features) + epsilon);
        for (long long col = 0; col < features; col++) {
            const float x_hat = (in.data[in_base + col * in_feature_stride] - row_mean) * row_inv_std;
            out.data[out_base + col * out_feature_stride] =
                gamma.data[col * gamma_stride] * x_hat + beta.data[col * beta_stride];
        }
        mean.data[row * mean.stride[0]] = row_mean;
        inv_std.data[row * inv_std.stride[0]] = row_inv_std;
    }
}

void layernorm_backward(
    const View& grad_in, const View& grad_gamma, const View& grad_beta,
    const View& grad_out, const View& in, const View& gamma,
    const View& mean, const View& inv_std
) {
    require_same_shape(grad_in, in, "layernorm_backward", "grad_in.shape", "in.shape");
    require_same_shape(grad_out, in, "layernorm_backward", "grad_out.shape", "in.shape");
    const auto shape = validate_common(in, gamma, mean, inv_std, "layernorm_backward");
    const long long rows = shape.first;
    const long long features = shape.second;
    if (grad_gamma.rank != 1 || grad_gamma.shape[0] != features ||
        grad_beta.rank != 1 || grad_beta.shape[0] != features)
        throw std::runtime_error("layernorm_backward parameter-gradient shape mismatch");

    if constexpr (TSGRAD_HAS_CUDA) {
        tsgrad::cuda::layernorm_backward(
            grad_in, grad_gamma, grad_beta, grad_out, in, gamma, mean, inv_std);
        return;
    }

    const long long in_feature_stride = in.stride[in.rank - 1];
    const long long grad_in_feature_stride = grad_in.stride[grad_in.rank - 1];
    const long long grad_out_feature_stride = grad_out.stride[grad_out.rank - 1];

    #pragma omp parallel for if (rows * features >= THRESHOLD)
    for (long long row = 0; row < rows; row++) {
        const long long in_base = leading_offset(row, in);
        const long long grad_in_base = leading_offset(row, grad_in);
        const long long grad_out_base = leading_offset(row, grad_out);
        const float row_mean = mean.data[row * mean.stride[0]];
        const float row_inv_std = inv_std.data[row * inv_std.stride[0]];

        float sum_q = 0.0f;
        float sum_q_x_hat = 0.0f;
        for (long long col = 0; col < features; col++) {
            const float x_hat = (in.data[in_base + col * in_feature_stride] - row_mean) * row_inv_std;
            const float q = grad_out.data[grad_out_base + col * grad_out_feature_stride]
                * gamma.data[col * gamma.stride[0]];
            sum_q += q;
            sum_q_x_hat += q * x_hat;
        }

        const float mean_q = sum_q / static_cast<float>(features);
        const float mean_q_x_hat = sum_q_x_hat / static_cast<float>(features);
        for (long long col = 0; col < features; col++) {
            const float x_hat = (in.data[in_base + col * in_feature_stride] - row_mean) * row_inv_std;
            const float q = grad_out.data[grad_out_base + col * grad_out_feature_stride]
                * gamma.data[col * gamma.stride[0]];
            grad_in.data[grad_in_base + col * grad_in_feature_stride] =
                row_inv_std * (q - mean_q - x_hat * mean_q_x_hat);
        }
    }

    #pragma omp parallel for if (rows * features >= THRESHOLD)
    for (long long col = 0; col < features; col++) {
        float gamma_sum = 0.0f;
        float beta_sum = 0.0f;
        for (long long row = 0; row < rows; row++) {
            const long long in_base = leading_offset(row, in);
            const long long grad_out_base = leading_offset(row, grad_out);
            const float dy = grad_out.data[grad_out_base + col * grad_out_feature_stride];
            const float x_hat = (in.data[in_base + col * in_feature_stride]
                - mean.data[row * mean.stride[0]]) * inv_std.data[row * inv_std.stride[0]];
            gamma_sum += dy * x_hat;
            beta_sum += dy;
        }
        grad_gamma.data[col * grad_gamma.stride[0]] = gamma_sum;
        grad_beta.data[col * grad_beta.stride[0]] = beta_sum;
    }
}

} //namespace
