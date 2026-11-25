#include "tsgrad/ops.h"
#include "cudahelpers.hpp"
#include "cudaops.hpp"
#include <format>
#include <cuda_runtime.h>
#include <limits>
#include <string>

// Host functions here are tsgrad::cuda::*. The CPU functions in
// src/cpu/elementwise.cpp and src/cpu/reduce.cpp are what addon.cc calls.
// Those CPU functions pass the same host Views here.
//
// Each host function below is the same sequence:
//   1. Context::get()              current device and stream
//   2. kernel<<<grid, block, 0, stream>>> on View::data, already device memory
//   3. Context::checkLastError()   a bad launch is reported here; the stream is not waited on
//
// Shape and stride ride along as a Layout argument. Do not free View::data.

namespace tsgrad::cuda {
    
template <float (*Func)(float)>
__global__ void map_kernel(float* out, Layout out_layout, const float* in, Layout in_layout, long long n) {
    long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i >= n) return;

    int32_t out_index[16];
    int32_t in_index[16];

    position_to_index(i, out_layout.shape, out_index, out_layout.rank);
    long long outPos = index_to_position(out_index, out_layout.stride, out_layout.rank);

    broadcast_index(out_index, out_layout.shape, out_layout.rank, in_layout.shape, in_index, in_layout.rank);
    long long inPos = index_to_position(in_index, in_layout.stride, in_layout.rank);

    out[outPos] = Func(in[inPos]);
}

template <float (*Func)(float)>
__global__ void map_kernel_contiguous(float* out, const float* in, long long n){
    long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) return;

    out[i] = Func(in[i]);
}

template <float (*Func)(float, float)>
__global__ void zip_kernel(float* out, Layout out_layout, const float* a, Layout a_layout, const float* b, Layout b_layout, long long n) {
    long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) return;

    int32_t out_index[16];
    int32_t a_index[16];
    int32_t b_index[16];

    position_to_index(i, out_layout.shape, out_index, out_layout.rank);
    long long outPos = index_to_position(out_index, out_layout.stride, out_layout.rank);

    broadcast_index(out_index, out_layout.shape, out_layout.rank, a_layout.shape, a_index, a_layout.rank);
    long long aPos = index_to_position(a_index, a_layout.stride, a_layout.rank);

    broadcast_index(out_index, out_layout.shape, out_layout.rank, b_layout.shape, b_index, b_layout.rank);
    long long bPos = index_to_position(b_index, b_layout.stride, b_layout.rank);

    out[outPos] = Func(a[aPos], b[bPos]);
}

template <float (*Func)(float, float)>
__global__ void zip_kernel_contiguous(float* out, const float* a, const float* b, long long n) {
    long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) return;

    out[i] = Func(a[i], b[i]);
}

template <float (*Func)(float, float)>
__global__ void reduce_kernel(float* out, Layout out_layout, const float* in, Layout in_layout, int32_t dim, float base, long long n) {
    long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i >= n) return;

    const int32_t rank = in_layout.rank;
    int32_t dimSize = in_layout.shape[dim];
    int32_t out_index[16];

    position_to_index(i, out_layout.shape, out_index, rank);
    int32_t temp = out_index[dim];
    float cur = base;

    for (int32_t j = 0; j < dimSize; j++){
        out_index[dim] = j;
        cur = Func(cur, in[index_to_position(out_index, in_layout.stride, rank)]);
    }

    out_index[dim] = temp;
    out[index_to_position(out_index, out_layout.stride, rank)] = cur;
}

__global__ void fill_kernel(float* data, long long n, float value) {
    long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    data[i] = value;
}

template<float (*Func)(float)>
void map_op(const View& out, const View& in){
    if (out.rank > 16 || in.rank > 16)
        throw std::runtime_error(std::format("CUDA map_op received exceptionally large shape: {}", std::max(out.rank, in.rank)));
    
    Context& ctx = Context::get();
    long long outsize = prod(out.shape, out.rank);
    if (outsize == 0) return;

    int32_t threads = 256;
    int32_t blocks = static_cast<int32_t>((outsize + threads - 1) / threads);

    if (is_contiguous(out) && is_contiguous(in) && shapesAreEqual(out.shape, in.shape, out.rank, in.rank)){
        map_kernel_contiguous<Func><<<blocks, threads, 0, static_cast<cudaStream_t>(ctx.getStream())>>>(out.data, in.data, outsize);
        ctx.checkLastError();
        return;
    }

    map_kernel<Func><<<blocks, threads, 0, static_cast<cudaStream_t>(ctx.getStream())>>>(out.data, layout_from(out), in.data, layout_from(in), outsize);
    ctx.checkLastError();
}

void map_neg(const View& out, const View& in){ map_op<neg>(out, in); }

void map_inv(const View& out, const View& in) { map_op<inv>(out, in); }
void map_sigmoid(const View& out, const View& in) { map_op<sigmoid>(out, in); }
void map_relu(const View& out, const View& in) { map_op<relu>(out, in); }
void map_gelu(const View& out, const View& in) { map_op<gelu>(out, in); }
void map_silu(const View& out, const View& in) { map_op<silu>(out, in); }
void map_log(const View& out, const View& in) { map_op<log>(out, in); }
void map_exp(const View& out, const View& in) { map_op<exp>(out, in); }
void map_sqrt(const View& out, const View& in) { map_op<sqrt>(out, in); }
void map_id(const View& out, const View& in) { map_op<id>(out, in); }

template<float (*Func)(float, float)>
void zip_op(const View& out, const View& a, const View& b){
    if (out.rank > 16 || a.rank > 16 || b.rank > 16)
        throw std::runtime_error(std::format("CUDA zip_op received exceptionally large shape: {}", std::max(std::max(out.rank, a.rank), b.rank)));
    
    Context& ctx = Context::get();
    long long outsize = prod(out.shape, out.rank);
    if (outsize == 0) return;

    int32_t threads = 256;
    int32_t blocks = static_cast<int32_t>((outsize + threads - 1) / threads);

    if (is_contiguous(out) && is_contiguous(a) && is_contiguous(b) && shapesAreEqual(out.shape, a.shape, out.rank, a.rank) && shapesAreEqual(out.shape, b.shape, out.rank, b.rank)){
        zip_kernel_contiguous<Func><<<blocks, threads, 0, static_cast<cudaStream_t>(ctx.getStream())>>>(out.data, a.data, b.data, outsize);
        ctx.checkLastError();
        return;
    }

    if (!is_broadcastable(out.shape, a.shape, out.rank, a.rank) || !is_broadcastable(out.shape, b.shape, out.rank, b.rank))
        throw std::runtime_error("zip_op received unbroadcastable shapes");

    zip_kernel<Func><<<blocks, threads, 0, static_cast<cudaStream_t>(ctx.getStream())>>>(
        out.data, layout_from(out),
        a.data, layout_from(a),
        b.data, layout_from(b),
        outsize
    );
    ctx.checkLastError();
}

void zip_add(const View& out, const View& a, const View& b) { zip_op<add>(out, a, b); }
void zip_sub(const View& out, const View& a, const View& b) { zip_op<sub>(out, a, b); }

void zip_mul(const View& out, const View& a, const View& b) { zip_op<mul>(out, a, b); }
void zip_div(const View& out, const View& a, const View& b) { zip_op<div>(out, a, b); }
void zip_lt(const View& out, const View& a, const View& b) { zip_op<lt>(out, a, b); }
void zip_eq(const View& out, const View& a, const View& b) { zip_op<eq>(out, a, b); }
void zip_is_close(const View& out, const View& a, const View& b) { zip_op<is_close>(out, a, b); }
void zip_gelu_backward(const View& out, const View& grad,const View& in){ zip_op<gelu_backward>(out, grad, in); }
void zip_silu_backward(const View& out, const View& grad,const View& in){ zip_op<silu_backward>(out, grad, in); }


namespace {

// One thread per output walks the whole reduced axis, so a [4096, 768] bias
// gradient is 3 blocks and each thread issues 4096 dependent loads. Split the
// rows across blocks instead. A warp still owns consecutive columns: that is
// the contiguous inner run, and splitting a column inside the warp would jump
// `inner` floats between neighbors.
struct RowChunkPlan {
    int32_t chunks;
    int32_t rows_per_chunk;
    int32_t blocks_x;
};

RowChunkPlan plan_row_chunks(int32_t reduce_len, int32_t width) {
    constexpr int32_t threads = 256;
    constexpr int32_t min_rows_per_chunk = 32;
    // Keep 32-row chunks. Shrink only when the partial buffer would pass 64 MB.
    constexpr int32_t max_partial_floats = 16 * 1024 * 1024;
    const int32_t blocks_x = (width + threads - 1) / threads;
    int32_t chunks = (reduce_len + min_rows_per_chunk - 1) / min_rows_per_chunk;
    if (chunks < 1) chunks = 1;
    const int32_t max_chunks = std::max(1, max_partial_floats / std::max(width, 1));
    if (chunks > max_chunks) chunks = max_chunks;
    const int32_t rows_per_chunk = (reduce_len + chunks - 1) / chunks;
    return {chunks, rows_per_chunk, blocks_x};
}

float* sum_partial_scratch(size_t floats, cudaStream_t stream) {
    struct Buffer {
        float* data = nullptr;
        size_t floats = 0;
    };
    static Buffer buffer;
    if (buffer.floats >= floats) return buffer.data;

    // The previous generation may still be queued on this stream.
    if (buffer.data != nullptr) {
        cudaError_t synced = cudaStreamSynchronize(stream);
        if (synced != cudaSuccess)
            throw std::runtime_error(std::string("cudaStreamSynchronize: ") + cudaGetErrorString(synced));
        cudaError_t freed = cudaFree(buffer.data);
        buffer.data = nullptr;
        buffer.floats = 0;
        if (freed != cudaSuccess)
            throw std::runtime_error(std::string("cudaFree: ") + cudaGetErrorString(freed));
    }

    cudaError_t allocated = cudaMalloc(&buffer.data, floats * sizeof(float));
    if (allocated != cudaSuccess) {
        buffer.data = nullptr;
        buffer.floats = 0;
        throw std::runtime_error(std::string("CUDA reduce_sum could not allocate partials: ") + cudaGetErrorString(allocated));
    }
    buffer.floats = floats;
    return buffer.data;
}

__global__ void reduce_sum_partial_kernel(
    float* partial, const float* in,
    int32_t reduce_len, int32_t inner, int32_t width, int32_t rows_per_chunk
) {
    const int32_t col = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (col < 0 || col >= width) return;

    const int32_t chunk = static_cast<int32_t>(blockIdx.y);
    const long long row0 = static_cast<long long>(chunk) * rows_per_chunk;
    const long long partial_index = static_cast<long long>(chunk) * width + col;
    if (row0 >= reduce_len) {
        partial[partial_index] = 0.0f;
        return;
    }
    long long row1 = row0 + rows_per_chunk;
    if (row1 > reduce_len) row1 = reduce_len;

    const int32_t outer_i = col / inner;
    const int32_t inner_j = col - outer_i * inner;
    const float* col_ptr = in + (static_cast<long long>(outer_i) * reduce_len) * inner + inner_j;

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    long long row = row0;
    const long long tail = row0 + ((row1 - row0) & ~3LL);
    for (; row < tail; row += 4) {
        a0 += col_ptr[row * inner];
        a1 += col_ptr[(row + 1) * inner];
        a2 += col_ptr[(row + 2) * inner];
        a3 += col_ptr[(row + 3) * inner];
    }
    float acc = (a0 + a1) + (a2 + a3);
    for (; row < row1; ++row)
        acc += col_ptr[row * inner];
    partial[partial_index] = acc;
}

// The vocab bias is already ~200 column tiles. Splitting those rows across a
// second grid dimension did not make the reduction faster, so that shape
// streams one thread per column and skips the partial buffer.
__global__ void reduce_sum_columns_kernel(
    float* out, const float* in, int32_t reduce_len, int32_t inner, int32_t width
) {
    const int32_t col = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (col < 0 || col >= width) return;

    const int32_t outer_i = col / inner;
    const int32_t inner_j = col - outer_i * inner;
    const int32_t stride = inner;
    const float* ptr = in + (static_cast<long long>(outer_i) * reduce_len) * stride + inner_j;

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    int32_t row = 0;
    const int32_t tail = reduce_len & ~3;
    for (; row < tail; row += 4) {
        a0 += ptr[0];
        a1 += ptr[stride];
        a2 += ptr[static_cast<int32_t>(stride * 2)];
        a3 += ptr[static_cast<int32_t>(stride * 3)];
        ptr += stride * 4;
    }
    float acc = (a0 + a1) + (a2 + a3);
    for (; row < reduce_len; ++row) {
        acc += ptr[0];
        ptr += stride;
    }
    out[col] = acc;
}

__global__ void reduce_sum_finalize_kernel(float* out, const float* partial, int32_t chunks, int32_t width) {
    const int32_t col = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (col < 0 || col >= width) return;

    float acc = 0.0f;
    for (int32_t chunk = 0; chunk < chunks; ++chunk)
        acc += partial[static_cast<long long>(chunk) * width + col];
    out[col] = acc;
}

bool reduce_sum_contiguous(const View& out, const View& in, int32_t dim) {
    if (dim < 0 || dim >= in.rank || out.rank != in.rank || out.rank > 16) return false;
    if (!is_contiguous(out) || !is_contiguous(in)) return false;
    for (int32_t axis = 0; axis < in.rank; ++axis) {
        const int32_t expected = axis == dim ? 1 : in.shape[axis];
        if (out.shape[axis] != expected) return false;
    }

    const int32_t reduce_len = in.shape[dim];
    if (reduce_len < 256) return false;

    long long inner = 1;
    for (int32_t axis = dim + 1; axis < in.rank; ++axis) inner *= in.shape[axis];
    if (inner < 32 || inner > std::numeric_limits<int32_t>::max()) return false;

    const long long width = prod(out.shape, out.rank);
    if (width <= 0 || width > std::numeric_limits<int32_t>::max()) return false;
    if (prod(in.shape, in.rank) != width * reduce_len) return false;

    const RowChunkPlan plan = plan_row_chunks(reduce_len, static_cast<int32_t>(width));
    Context& ctx = Context::get();
    cudaStream_t stream = static_cast<cudaStream_t>(ctx.getStream());
    if (plan.blocks_x >= 128) {
        reduce_sum_columns_kernel<<<plan.blocks_x, 256, 0, stream>>>(
            out.data, in.data, reduce_len, static_cast<int32_t>(inner), static_cast<int32_t>(width));
        ctx.checkLastError();
        return true;
    }
    float* partial = sum_partial_scratch(
        static_cast<size_t>(plan.chunks) * static_cast<size_t>(width), stream);

    reduce_sum_partial_kernel<<<dim3(plan.blocks_x, plan.chunks), 256, 0, stream>>>(
        partial, in.data, reduce_len, static_cast<int32_t>(inner),
        static_cast<int32_t>(width), plan.rows_per_chunk);
    ctx.checkLastError();
    reduce_sum_finalize_kernel<<<plan.blocks_x, 256, 0, stream>>>(
        out.data, partial, plan.chunks, static_cast<int32_t>(width));
    ctx.checkLastError();
    return true;
}

} // namespace

template<float (*Func)(float, float)>
void reduce_op(const View& out, const View& in, int32_t dim, float base){
    if (dim < 0 || dim >= in.rank)
        throw std::runtime_error("reduce_op received dim outside the boundaries of in.shape");

    if (out.rank > 16 || in.rank > 16)
        throw std::runtime_error(std::format("CUDA reduce_op received exceptionally large shape: {}", std::max(out.rank, in.rank)));
    
    Context& ctx = Context::get();
    long long outsize = prod(out.shape, out.rank);
    if (outsize == 0) return;

    int32_t threads = 256;
    int32_t blocks = static_cast<int32_t>((outsize + threads - 1) / threads);

    reduce_kernel<Func><<<blocks, threads, 0, static_cast<cudaStream_t>(ctx.getStream())>>>(out.data, layout_from(out), in.data, layout_from(in), dim, base, outsize);
    ctx.checkLastError();
}

void fill(float* data, size_t numel, float value) {
    if (numel == 0 || data == nullptr) return;

    Context& ctx = Context::get();
    const int threads = 256;
    const int blocks = static_cast<int>((numel + threads - 1) / threads);
    fill_kernel<<<blocks, threads, 0, static_cast<cudaStream_t>(ctx.getStream())>>>(data, static_cast<long long>(numel), value);
    ctx.checkLastError();
}

void reduce_max(const View& out, const View& in, int32_t dim) { reduce_op<max>(out, in, dim, -std::numeric_limits<float>::infinity()); }
void reduce_sum(const View& out, const View& in, int32_t dim) {
    if (reduce_sum_contiguous(out, in, dim)) return;
    reduce_op<add>(out, in, dim, 0);
}
void reduce_prod(const View& out, const View& in, int32_t dim) { reduce_op<mul>(out, in, dim, 1); }

} // namespace tsgrad::cuda
