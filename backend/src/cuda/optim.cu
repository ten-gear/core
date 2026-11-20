#include "tsgrad/ops.h"
#include "cudahelpers.hpp"

#include <cuda_runtime.h>
#include <stdexcept>
#include <cmath>

namespace tsgrad::cuda {

__global__ void sgd_kernel(float* param, const float* grad, long long n, float lr) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    param[i] -= lr * grad[i];
}

__global__ void sgd_momentum_kernel(float* param, const float* grad, float* velocity, long long n, float lr, float beta) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float updated = beta * velocity[i] + (1.0F - beta) * grad[i];
    velocity[i] = updated;
    param[i] -= lr * updated;
}

void sgd(const View& param, const View& grad, float lr) {
    if (!shapesAreEqual(param.shape, grad.shape, param.rank, grad.rank))
        throw std::runtime_error("sgd received mismatched shapes");
    if (!is_contiguous(param) || !is_contiguous(grad))
        throw std::runtime_error("sgd requires contiguous storage");

    const long long n = prod(param.shape, param.rank);
    if (n == 0) return;

    Context& ctx = Context::get();
    const int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    sgd_kernel<<<blocks, threads, 0, static_cast<cudaStream_t>(ctx.getStream())>>>(param.data, grad.data, n, lr);
    ctx.checkLastError();
}

void sgd_momentum(const View& param, const View& grad, const View& velocity, float lr, float beta) {
    if (!shapesAreEqual(param.shape, grad.shape, param.rank, grad.rank) || !shapesAreEqual(param.shape, velocity.shape, param.rank, velocity.rank))
        throw std::runtime_error("sgd_momentum received mismatched shapes");
    if (!is_contiguous(param) || !is_contiguous(grad) || !is_contiguous(velocity))
        throw std::runtime_error("sgd_momentum requires contiguous storage");

    const long long n = prod(param.shape, param.rank);
    if (n == 0) return;

    Context& ctx = Context::get();
    const int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    sgd_momentum_kernel<<<blocks, threads, 0, static_cast<cudaStream_t>(ctx.getStream())>>>(param.data, grad.data, velocity.data, n, lr, beta);
    ctx.checkLastError();
}

__global__ void adamw_kernel(float* param, const float* grad, float* m, float* v, long long n,
    float lr, float beta1, float beta2, float eps, float weight_decay,
    float bias1, float bias2){
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i >= n) return;

    const float g = grad[i];
    const float mi = beta1 * m[i] + (1.0f - beta1) * g;
    const float vi = beta2 * v[i] + (1.0f - beta2) * g * g;

    m[i] = mi;
    v[i] = vi;

    const float mhat = mi / bias1;
    const float vhat = vi / bias2;

    param[i] = param[i] * (1.0f - lr * weight_decay) - lr * mhat / (sqrtf(vhat) + eps);
}

void adamw(const View& param, const View& grad, const View& m, const View& v, float lr, float beta1, float beta2, float eps, float weight_decay, uint64_t t){
    const long long n = prod(param.shape, param.rank);
    if (n == 0) return;

    const float bias1 = 1 - std::pow(beta1, t);
    const float bias2 = 1 - std::pow(beta2, t);
    Context& ctx = Context::get();
    const int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    adamw_kernel<<<blocks, threads, 0, static_cast<cudaStream_t>(ctx.getStream())>>>(param.data, grad.data, m.data, v.data, n, lr, beta1, beta2, eps, weight_decay, bias1, bias2);
    ctx.checkLastError();
}

} // namespace tsgrad::cuda
