#pragma once
#include <cuda_runtime.h>

namespace tsgrad::cuda{
    __device__ float id(float a){ return a; }
    __device__ float neg(float a){ return -a; }
    __device__ float sigmoid(float a){ 
        if (a >= 0)
            return 1.0F/(1.0F + expf(-a));  // e^a
        else
            return expf(a) / (1.0F + expf(a));
    }
    __device__ float relu(float a){ if (a < 0) a = 0; return a; }
    __device__ float gelu(float a){ 
        constexpr float INV_SQRT_2 = 0.70710678118f;
        return 0.5f * a * (1.0f + erff(a * INV_SQRT_2)); 
    }

    __device__ float log(float a){ return logf(a); }
    __device__ float exp(float a){ return expf(a); }
    __device__ float inv(float a){ return 1/a; }
    __device__ float sqrt(float a){ return sqrtf(a); }
    
    __device__ float add(float a, float b){ return a + b; }
    __device__ float sub(float a, float b){ return a - b; }
    __device__ float mul(float a, float b){ return a * b; }
    __device__ float div(float a, float b){ return a / b; }
    __device__ float lt(float a, float b){ return a < b ? 1 : 0; }
    __device__ float eq(float a, float b){ return a == b; }
    __device__ float max(float a, float b){ return a > b ? a : b; }
    __device__ float is_close(float a, float b){ return fabsf(a - b) < 0.01; }
    __device__ float gelu_backward(float grad, float x) {
        constexpr float INV_SQRT_2 = 0.70710678118f;
        constexpr float INV_SQRT_2PI = 0.3989422804014327f;
        const float cdf = 0.5f * (1.0f + erff(x * INV_SQRT_2));
        const float pdf = INV_SQRT_2PI * expf(-0.5f * x * x);
    
        return grad * (cdf + x * pdf);
    }
};