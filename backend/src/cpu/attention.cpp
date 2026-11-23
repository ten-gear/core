#include "tsgrad/ops.h"
#include "helpers.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <span>
#include <vector>

namespace tsgrad {
namespace {

void require_shape(const View& actual, const View& expected, const char* operation, const char* name) {
    if (shapesAreEqual(actual.shape, expected.shape, actual.rank, expected.rank)) return;
    std::span<const int32_t> actual_shape{actual.shape, static_cast<size_t>(actual.rank)};
    std::span<const int32_t> expected_shape{expected.shape, static_cast<size_t>(expected.rank)};
    throw std::runtime_error(std::format(
        "{} received invalid {} shape, expected {:n}, received {:n}",
        operation, name, expected_shape, actual_shape));
}

void validate_forward(
    const View& out, const View& lse,
    const View& query, const View& key, const View& value,
    float scale, const char* operation
) {
    if (query.rank != 4)
        throw std::runtime_error(std::format("{} expects rank-4 [batch, heads, time, headDim] inputs", operation));
    require_shape(key, query, operation, "key");
    require_shape(value, query, operation, "value");
    require_shape(out, query, operation, "out");
    if (query.shape[0] <= 0 || query.shape[1] <= 0 || query.shape[2] <= 0)
        throw std::runtime_error(std::format("{} requires non-empty batch, head, and time dimensions", operation));
    const int32_t head_dim = query.shape[3];
    if (head_dim != 32 && head_dim != 64 && head_dim != 128)
        throw std::runtime_error(std::format(
            "{} supports head dimensions 32, 64, and 128, received {}", operation, head_dim));
    if (lse.rank != 3
        || lse.shape[0] != query.shape[0]
        || lse.shape[1] != query.shape[1]
        || lse.shape[2] != query.shape[2])
        throw std::runtime_error(std::format("{} expects lse shape [batch, heads, time]", operation));
    if (!std::isfinite(scale))
        throw std::runtime_error(std::format("{} expects a finite scale", operation));
}

long long position4(const View& view, int32_t batch, int32_t head, int32_t time, int32_t feature) {
    return static_cast<long long>(batch) * view.stride[0]
        + static_cast<long long>(head) * view.stride[1]
        + static_cast<long long>(time) * view.stride[2]
        + static_cast<long long>(feature) * view.stride[3];
}

long long position3(const View& view, int32_t batch, int32_t head, int32_t time) {
    return static_cast<long long>(batch) * view.stride[0]
        + static_cast<long long>(head) * view.stride[1]
        + static_cast<long long>(time) * view.stride[2];
}

float dot_rows(
    const View& a, const View& b,
    int32_t batch, int32_t head, int32_t a_time, int32_t b_time, int32_t head_dim
) {
    const long long a_base = position4(a, batch, head, a_time, 0);
    const long long b_base = position4(b, batch, head, b_time, 0);
    float total = 0.0f;
    for (int32_t feature = 0; feature < head_dim; feature++)
        total += a.data[a_base + static_cast<long long>(feature) * a.stride[3]]
            * b.data[b_base + static_cast<long long>(feature) * b.stride[3]];
    return total;
}

} // namespace

void attention(
    const View& out, const View& lse,
    const View& query, const View& key, const View& value,
    float scale, bool causal
) {
    validate_forward(out, lse, query, key, value, scale, "attention");

    if constexpr (TSGRAD_HAS_CUDA) {
        cuda::attention(out, lse, query, key, value, scale, causal);
        return;
    }

    const int32_t batches = query.shape[0];
    const int32_t heads = query.shape[1];
    const int32_t time = query.shape[2];
    const int32_t head_dim = query.shape[3];
    const long long rows = static_cast<long long>(batches) * heads * time;

    #pragma omp parallel for if (rows * time * head_dim >= THRESHOLD)
    for (long long row = 0; row < rows; row++) {
        const int32_t query_time = static_cast<int32_t>(row % time);
        const int32_t head = static_cast<int32_t>((row / time) % heads);
        const int32_t batch = static_cast<int32_t>(row / (static_cast<long long>(time) * heads));
        const int32_t key_limit = causal ? query_time + 1 : time;
        std::vector<float> accumulator(static_cast<size_t>(head_dim), 0.0f);
        float maximum = -std::numeric_limits<float>::infinity();
        float denominator = 0.0f;

        for (int32_t key_time = 0; key_time < key_limit; key_time++) {
            const float score = scale * dot_rows(
                query, key, batch, head, query_time, key_time, head_dim);
            const float next_maximum = std::max(maximum, score);
            const float previous_scale = std::exp(maximum - next_maximum);
            const float probability_scale = std::exp(score - next_maximum);
            denominator = denominator * previous_scale + probability_scale;
            const long long value_base = position4(value, batch, head, key_time, 0);
            for (int32_t feature = 0; feature < head_dim; feature++)
                accumulator[feature] = accumulator[feature] * previous_scale
                    + probability_scale * value.data[
                        value_base + static_cast<long long>(feature) * value.stride[3]];
            maximum = next_maximum;
        }

        const long long out_base = position4(out, batch, head, query_time, 0);
        for (int32_t feature = 0; feature < head_dim; feature++)
            out.data[out_base + static_cast<long long>(feature) * out.stride[3]]
                = accumulator[feature] / denominator;
        lse.data[position3(lse, batch, head, query_time)] = maximum + std::log(denominator);
    }
}

void attention_backward(
    const View& grad_query, const View& grad_key, const View& grad_value,
    const View& grad_out, const View& query, const View& key, const View& value,
    const View& out, const View& lse, float scale, bool causal
) {
    validate_forward(out, lse, query, key, value, scale, "attention_backward");
    require_shape(grad_query, query, "attention_backward", "gradQuery");
    require_shape(grad_key, query, "attention_backward", "gradKey");
    require_shape(grad_value, query, "attention_backward", "gradValue");
    require_shape(grad_out, query, "attention_backward", "gradOut");

    if constexpr (TSGRAD_HAS_CUDA) {
        cuda::attention_backward(
            grad_query, grad_key, grad_value, grad_out,
            query, key, value, out, lse, scale, causal);
        return;
    }

    const int32_t batches = query.shape[0];
    const int32_t heads = query.shape[1];
    const int32_t time = query.shape[2];
    const int32_t head_dim = query.shape[3];
    const long long rows = static_cast<long long>(batches) * heads * time;

    #pragma omp parallel for if (rows * time * head_dim >= THRESHOLD)
    for (long long row = 0; row < rows; row++) {
        const int32_t query_time = static_cast<int32_t>(row % time);
        const int32_t head = static_cast<int32_t>((row / time) % heads);
        const int32_t batch = static_cast<int32_t>(row / (static_cast<long long>(time) * heads));
        const int32_t key_limit = causal ? query_time + 1 : time;
        const long long grad_out_base = position4(grad_out, batch, head, query_time, 0);
        const long long out_base = position4(out, batch, head, query_time, 0);
        float delta = 0.0f;
        for (int32_t feature = 0; feature < head_dim; feature++)
            delta += out.data[out_base + static_cast<long long>(feature) * out.stride[3]]
                * grad_out.data[
                    grad_out_base + static_cast<long long>(feature) * grad_out.stride[3]];

        std::vector<float> accumulator(static_cast<size_t>(head_dim), 0.0f);
        const float row_lse = lse.data[position3(lse, batch, head, query_time)];
        for (int32_t key_time = 0; key_time < key_limit; key_time++) {
            const float probability = std::exp(
                scale * dot_rows(query, key, batch, head, query_time, key_time, head_dim)
                - row_lse);
            const float grad_value_dot = dot_rows(
                grad_out, value, batch, head, query_time, key_time, head_dim);
            const float grad_score = probability * (grad_value_dot - delta);
            const long long key_base = position4(key, batch, head, key_time, 0);
            for (int32_t feature = 0; feature < head_dim; feature++)
                accumulator[feature] += scale * grad_score * key.data[
                    key_base + static_cast<long long>(feature) * key.stride[3]];
        }
        const long long grad_query_base = position4(grad_query, batch, head, query_time, 0);
        for (int32_t feature = 0; feature < head_dim; feature++)
            grad_query.data[
                grad_query_base + static_cast<long long>(feature) * grad_query.stride[3]]
                = accumulator[feature];
    }

    #pragma omp parallel for if (rows * time * head_dim >= THRESHOLD)
    for (long long row = 0; row < rows; row++) {
        const int32_t key_time = static_cast<int32_t>(row % time);
        const int32_t head = static_cast<int32_t>((row / time) % heads);
        const int32_t batch = static_cast<int32_t>(row / (static_cast<long long>(time) * heads));
        const int32_t query_start = causal ? key_time : 0;
        std::vector<float> key_accumulator(static_cast<size_t>(head_dim), 0.0f);
        std::vector<float> value_accumulator(static_cast<size_t>(head_dim), 0.0f);

        for (int32_t query_time = query_start; query_time < time; query_time++) {
            const long long grad_out_base = position4(grad_out, batch, head, query_time, 0);
            const long long out_base = position4(out, batch, head, query_time, 0);
            float delta = 0.0f;
            for (int32_t feature = 0; feature < head_dim; feature++)
                delta += out.data[out_base + static_cast<long long>(feature) * out.stride[3]]
                    * grad_out.data[
                        grad_out_base + static_cast<long long>(feature) * grad_out.stride[3]];

            const float probability = std::exp(
                scale * dot_rows(query, key, batch, head, query_time, key_time, head_dim)
                - lse.data[position3(lse, batch, head, query_time)]);
            const float grad_value_dot = dot_rows(
                grad_out, value, batch, head, query_time, key_time, head_dim);
            const float grad_score = probability * (grad_value_dot - delta);
            const long long query_base = position4(query, batch, head, query_time, 0);
            for (int32_t feature = 0; feature < head_dim; feature++) {
                key_accumulator[feature] += scale * grad_score * query.data[
                    query_base + static_cast<long long>(feature) * query.stride[3]];
                value_accumulator[feature] += probability * grad_out.data[
                    grad_out_base + static_cast<long long>(feature) * grad_out.stride[3]];
            }
        }

        const long long grad_key_base = position4(grad_key, batch, head, key_time, 0);
        const long long grad_value_base = position4(grad_value, batch, head, key_time, 0);
        for (int32_t feature = 0; feature < head_dim; feature++) {
            grad_key.data[
                grad_key_base + static_cast<long long>(feature) * grad_key.stride[3]]
                = key_accumulator[feature];
            grad_value.data[
                grad_value_base + static_cast<long long>(feature) * grad_value.stride[3]]
                = value_accumulator[feature];
        }
    }
}

} // namespace tsgrad
