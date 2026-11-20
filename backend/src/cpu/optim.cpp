#include "tsgrad/ops.h"
#include "helpers.hpp"

#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>

namespace tsgrad {
namespace {

void check_same_shape(const View& a, const View& b, const char* what) {
    if (!shapesAreEqual(a.shape, b.shape, a.rank, b.rank)) {
        throw std::runtime_error(std::string(what) + " received mismatched shapes");
    }
}

} // namespace

void sgd(const View& param, const View& grad, float lr) {
    if constexpr (TSGRAD_HAS_CUDA) {
        cuda::sgd(param, grad, lr);
        return;
    }

    check_same_shape(param, grad, "sgd");
    const long long n = prod(param.shape, param.rank);
    if (n == 0) return;

    if (is_contiguous(param) && is_contiguous(grad)) {
        for (long long i = 0; i < n; i++)
            param.data[i] -= lr * grad.data[i];
        return;
    }

    std::vector<int32_t> index(param.rank, 0);
    for (long long i = 0; i < n; i++) {
        position_to_index(i, param.shape, index.data(), param.rank);
        const long long pos = index_to_position(index.data(), param.stride, param.rank);
        const long long gpos = index_to_position(index.data(), grad.stride, grad.rank);
        param.data[pos] -= lr * grad.data[gpos];
    }
}

void sgd_momentum(const View& param, const View& grad, const View& velocity, float lr, float beta) {
    if constexpr (TSGRAD_HAS_CUDA) {
        cuda::sgd_momentum(param, grad, velocity, lr, beta);
        return;
    }

    check_same_shape(param, grad, "sgd_momentum");
    check_same_shape(param, velocity, "sgd_momentum");
    const long long n = prod(param.shape, param.rank);
    if (n == 0) return;

    if (is_contiguous(param) && is_contiguous(grad) && is_contiguous(velocity)) {
        for (long long i = 0; i < n; i++) {
            const float updated = beta * velocity.data[i] + (1.0F - beta) * grad.data[i];
            velocity.data[i] = updated;
            param.data[i] -= lr * updated;
        }
        return;
    }

    std::vector<int32_t> index(param.rank, 0);
    for (long long i = 0; i < n; i++) {
        position_to_index(i, param.shape, index.data(), param.rank);
        const long long pos = index_to_position(index.data(), param.stride, param.rank);
        const long long gpos = index_to_position(index.data(), grad.stride, grad.rank);
        const long long vpos = index_to_position(index.data(), velocity.stride, velocity.rank);
        const float updated = beta * velocity.data[vpos] + (1.0F - beta) * grad.data[gpos];
        velocity.data[vpos] = updated;
        param.data[pos] -= lr * updated;
    }
}

void adamw(const View& param, const View& grad, const View& m, const View& v, float lr, float beta1, float beta2, float eps, float weight_decay, uint64_t t) {
    check_same_shape(param, grad, "adamw");
    check_same_shape(param, m, "adamw");
    check_same_shape(param, v, "adamw");
    if (t == 0)
        throw std::runtime_error("AdamW t must be at least 1");

    if (!is_contiguous(param) || !is_contiguous(grad) || !is_contiguous(v) || !is_contiguous(m))
        throw std::runtime_error("AdamW requires contiguous storage");

    if constexpr (TSGRAD_HAS_CUDA) {
        cuda::adamw(param, grad, m, v, lr, beta1, beta2, eps, weight_decay, t);
        return;
    }

    const long long n = prod(param.shape, param.rank);
    if (n == 0) return;

    float bias1 = 1 - std::pow(beta1, t);
    float bias2 = 1 - std::pow(beta2, t);

    #pragma omp parallel for if (n >= THRESHOLD)
    for (long long i = 0; i < n; i++) {
        const float g = grad.data[i];
        const float mi = beta1 * m.data[i] + (1.0f - beta1) * g;
        const float vi = beta2 * v.data[i] + (1.0f - beta2) * g * g;

        m.data[i] = mi;
        v.data[i] = vi;

        const float mhat = mi / bias1;
        const float vhat = vi / bias2;

        param.data[i] = param.data[i] * (1.0f - lr * weight_decay) - lr * mhat / (std::sqrt(vhat) + eps);
    }
}

} // namespace tsgrad
