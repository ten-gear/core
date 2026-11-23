#pragma once

#ifndef TSGRAD_HAS_CUDA
    #define TSGRAD_HAS_CUDA 0
#endif

#include <cstddef>
#include <cstdint>

// Host-side view of one tensor argument.
//
// data is Storage::data(): a host vector on a CPU build, a device pointer on
// a CUDA build. Do not free it. shape and stride are borrowed for this call.
// Do not keep these pointers after the function returns.
//
// out is written. Every other View is read. Shape and stride follow the same
// rules as indexToPosition / broadcastIndex in src/tensor_data.ts.
namespace tsgrad {

struct View {
    float* data;
    const int32_t* shape;
    const int32_t* stride;
    int32_t rank;
};

// One output element per input element. Same indexing as tensorMap.
void map_neg(const View& out, const View& in);
void map_inv(const View& out, const View& in);
void map_sigmoid(const View& out, const View& in); // stable split at 0, same as operators.sigmoid
void map_relu(const View& out, const View& in);
void map_gelu(const View& out, const View& in);
void map_silu(const View& out, const View& in);
void map_log(const View& out, const View& in);
void map_exp(const View& out, const View& in);
void map_sqrt(const View& out, const View& in);
void map_id(const View& out, const View& in);

// One output element from a broadcast pair. Same indexing as tensorZip.
// zip_is_close writes 1 when abs(a - b) < 1e-2, else 0. That tolerance is operators.isClose.
void zip_add(const View& out, const View& a, const View& b);
void zip_sub(const View& out, const View& a, const View& b);
void zip_mul(const View& out, const View& a, const View& b);
void zip_div(const View& out, const View& a, const View& b);
void zip_lt(const View& out, const View& a, const View& b);
void zip_eq(const View& out, const View& a, const View& b);
void zip_is_close(const View& out, const View& a, const View& b);
void zip_gelu_backward(const View& out, const View& grad, const View& in);
void zip_silu_backward(const View& out, const View& grad, const View& in);

// param and grad are the same shape. Both are updated through param's data.
void sgd(const View& param, const View& grad, float lr);
void sgd_momentum(const View& param, const View& grad, const View& velocity, float lr, float beta);
void adamw(const View& param, const View& grad, const View& m, const View& v, float lr, float beta1, float beta2, float eps, float weight_decay, uint64_t t);

// out's reduced dimension has length 1. base is the identity:
// max uses -infinity, sum uses 0, prod uses 1. Same walk as tensorReduce.
void reduce_max(const View& out, const View& in, int32_t dim);
void reduce_sum(const View& out, const View& in, int32_t dim);
void reduce_prod(const View& out, const View& in, int32_t dim);

// out[..., M, N] = a[..., M, K] @ b[..., K, N]. Batch dimensions must match.
// Strides included.
void matmul(const View& out, const View& a, const View& b);

// conv1d: out [batch, outChannels, outWidth], input [batch, inChannels, width],
// weight [outChannels, inChannels, kWidth].
// conv2d: out [batch, outChannels, outHeight, outWidth],
// input [batch, inChannels, height, width], weight [outChannels, inChannels, kHeight, kWidth].
// reverse walks the kernel backward. Samples outside the input contribute 0.
// Same indexing as fastConv1d / fastConv2d. Strides included.
void conv1d(const View& out, const View& input, const View& weight, bool reverse);
void conv2d(const View& out, const View& input, const View& weight, bool reverse);

void gather(const View& out, const View& indicies, const View& values, const int32_t dim);

// out[coordinate with dim replaced by indicies] += grad[coordinate].
// grad and indicies have the same rank. Adds into the values already in out.
void scatter(const View& out, const View& indicies, const View& grad, const int32_t dim);

// out[*indicies.shape, features] = table[indicies, :]. table is [vocabulary, features].
void embedding(const View& out, const View& indicies, const View& table, const int32_t features);

// table[indicies, :] += grad. grad is [*indicies.shape, features].
// Adds into the values already in table. Duplicate ids add.
void embedding_backward(const View& table, const View& indicies, const View& grad, const int32_t features);

// Normalize the last input dimension. out/in have the same rank and shape.
// gamma/beta are [features]. mean/inv_std are [product(leading dimensions)].
void layernorm(
    const View& out, const View& in, const View& gamma, const View& beta,
    const View& mean, const View& inv_std, float epsilon);

// Writes all three gradients; it does not add to their previous values.
void layernorm_backward(
    const View& grad_in, const View& grad_gamma, const View& grad_beta,
    const View& grad_out, const View& in, const View& gamma,
    const View& mean, const View& inv_std);

// Normalize along dim. Backward consumes the saved forward output.
void softmax(const View& out, const View& input, int32_t dim);
void softmax_backward(const View& grad_in, const View& grad_out, const View& out, int32_t dim);
void logsoftmax(const View& out, const View& input, int32_t dim);
void logsoftmax_backward(const View& grad_in, const View& grad_out, const View& out, int32_t dim);

// Rotate pairs in the last dimension. in and out are [..., seq, headDim], headDim even.
// sin and cos are [positions, headDim/2]. Token t reads table row t + offset.
// Leading dimensions are independent rows. Strides included. Writes out; it does not add.
void rope(const View& out, const View& in, const View& sin, const View& cos, int32_t offset);
void rope_backward(const View& grad_in, const View& grad_out, const View& sin, const View& cos, int32_t offset);

// Fused scaled dot-product attention for [batch, heads, time, headDim].
// lse is [batch, heads, time]. No time-by-time probability matrix is stored.
void attention(
    const View& out, const View& lse,
    const View& query, const View& key, const View& value,
    float scale, bool causal);
void attention_backward(
    const View& grad_query, const View& grad_key, const View& grad_value,
    const View& grad_out, const View& query, const View& key, const View& value,
    const View& out, const View& lse, float scale, bool causal);

// Same signatures as above. Defined in src/cuda/*.cu when TSGRAD_HAS_CUDA is set.
// On a CUDA build View::data is already a device pointer. The .cu function
// launches on the process-wide stream and returns without waiting. It does not
// free data. copyToHost and synchronize wait.
namespace cuda {

class Context {
public:
    static Context& get();
    void* getStream();
    void* getCublas();
    void synchronize();
    void checkLastError();
    int device() const;

private:
    Context();
    ~Context();
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    int device_;
    void* stream_; // cudaStream_t, stored without including cuda_runtime.h here
    void* cublas_; // cublasHandle_t, stored without including cublas_v2.h here
};

void map_neg(const View& out, const View& in);
void map_inv(const View& out, const View& in);
void map_sigmoid(const View& out, const View& in);
void map_relu(const View& out, const View& in);
void map_gelu(const View& out, const View& in);
void map_silu(const View& out, const View& in);
void map_log(const View& out, const View& in);
void map_exp(const View& out, const View& in);
void map_sqrt(const View& out, const View& in);
void map_id(const View& out, const View& in);

void zip_add(const View& out, const View& a, const View& b);
void zip_sub(const View& out, const View& a, const View& b);
void zip_mul(const View& out, const View& a, const View& b);
void zip_div(const View& out, const View& a, const View& b);
void zip_lt(const View& out, const View& a, const View& b);
void zip_eq(const View& out, const View& a, const View& b);
void zip_is_close(const View& out, const View& a, const View& b);
void zip_gelu_backward(const View& out, const View& grad, const View& in);
void zip_silu_backward(const View& out, const View& grad, const View& in);

void sgd(const View& param, const View& grad, float lr);
void sgd_momentum(const View& param, const View& grad, const View& velocity, float lr, float beta);
void adamw(const View& param, const View& grad, const View& m, const View& v, float lr, float beta1, float beta2, float eps, float weight_decay, uint64_t t);

void reduce_max(const View& out, const View& in, int32_t dim);
void reduce_sum(const View& out, const View& in, int32_t dim);
void reduce_prod(const View& out, const View& in, int32_t dim);

// Leading dimensions are a batch. A uniform stride goes to cuBLAS.
// The stride kernel covers every other layout.
void matmul(const View& out, const View& a, const View& b);

void conv1d(const View& out, const View& input, const View& weight, bool reverse);
void conv2d(const View& out, const View& input, const View& weight, bool reverse);

void gather(const View& out, const View& indicies, const View& values, const int32_t dim);
void scatter(const View& out, const View& indicies, const View& grad, const int32_t dim);
void embedding(const View& out, const View& indicies, const View& table, const int32_t features);
void embedding_backward(const View& table, const View& indicies, const View& grad, const int32_t features);

void layernorm(
    const View& out, const View& in, const View& gamma, const View& beta,
    const View& mean, const View& inv_std, float epsilon);
void layernorm_backward(
    const View& grad_in, const View& grad_gamma, const View& grad_beta,
    const View& grad_out, const View& in, const View& gamma,
    const View& mean, const View& inv_std);

void softmax(const View& out, const View& input, int32_t dim);
void softmax_backward(
    const View& grad_in, const View& grad_out, const View& out, int32_t dim);
void logsoftmax(const View& out, const View& input, int32_t dim);
void logsoftmax_backward(
    const View& grad_in, const View& grad_out, const View& out, int32_t dim);

void rope(const View& out, const View& in, const View& sin, const View& cos, int32_t offset);
void rope_backward(const View& grad_in, const View& grad_out, const View& sin, const View& cos, int32_t offset);

void attention(
    const View& out, const View& lse,
    const View& query, const View& key, const View& value,
    float scale, bool causal);
void attention_backward(
    const View& grad_query, const View& grad_key, const View& grad_value,
    const View& grad_out, const View& query, const View& key, const View& value,
    const View& out, const View& lse, float scale, bool causal);

// Non-zero storage fill. Queued on the process-wide stream. Does not wait.
void fill(float* data, size_t numel, float value);

} // namespace cuda
} // namespace tsgrad
