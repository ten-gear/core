#include "tsgrad/ops.h"
#include "cudahelpers.hpp"
#include <format>
#include <cuda_runtime.h>

namespace tsgrad::cuda {
    __global__ void embedding_contiguous_kernel(float* out, const float* indicies, const float* table, int32_t features, long long n){
        long long i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;

        const long long row = i / features;
        const long long col = i % features;
        out[i] = table[static_cast<long long>(indicies[row]) * features + col];
    }

    __global__ void embedding_kernel(float* out, Layout out_layout, const float* indicies, Layout indicies_layout, const float* table, Layout table_layout, long long n){
        long long i = blockIdx.x * blockDim.x + threadIdx.x;

        if (i >= n) return;

        int32_t out_index[16];
        position_to_index(i, out_layout.shape, out_index, out_layout.rank);
        const long long out_pos = index_to_position(out_index, out_layout.stride, out_layout.rank);
        const int32_t token = static_cast<int32_t>(indicies[index_to_position(out_index, indicies_layout.stride, indicies_layout.rank)]);
        const int32_t table_index[2] = {token, out_index[out_layout.rank - 1]};
        out[out_pos] = table[index_to_position(table_index, table_layout.stride, 2)];
    }

    void embedding(const View& out, const View& indicies, const View& table, const int32_t features){
        if (out.rank > 16 || indicies.rank > 16 || table.rank > 16)
            throw std::runtime_error(std::format("CUDA embedding received exceptionally large shape, received out.rank: {}, indicies.rank: {}, table.rank: {}", out.rank, indicies.rank, table.rank));

        Context& ctx = Context::get();
        long long out_size = prod(out.shape, out.rank);
        if (out_size == 0) return;

        int32_t threads = 256;
        int32_t blocks = static_cast<int32_t>((out_size + threads - 1) / threads);

        if (is_contiguous(out) && is_contiguous(indicies) && is_contiguous(table))
            embedding_contiguous_kernel<<<blocks, threads, 0, static_cast<cudaStream_t>(ctx.getStream())>>>(out.data, indicies.data, table.data, features, out_size);
        else
            embedding_kernel<<<blocks, threads, 0, static_cast<cudaStream_t>(ctx.getStream())>>>(out.data, layout_from(out), indicies.data, layout_from(indicies), table.data, layout_from(table), out_size);
        ctx.checkLastError();
    }

    __global__ void embedding_backward_contiguous_kernel(float* table, const float* indicies, const float* grad, int32_t features, long long n){
        long long i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;

        const long long row = i / features;
        const long long col = i % features;
        atomicAdd(&table[static_cast<long long>(indicies[row]) * features + col], grad[i]);
    }

    __global__ void embedding_backward_kernel(float* table, Layout table_layout, const float* indicies, Layout indicies_layout, const float* grad, Layout grad_layout, long long n){
        long long i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;

        int32_t grad_index[16];
        position_to_index(i, grad_layout.shape, grad_index, grad_layout.rank);
        const long long grad_pos = index_to_position(grad_index, grad_layout.stride, grad_layout.rank);
        const int32_t token = static_cast<int32_t>(indicies[index_to_position(grad_index, indicies_layout.stride, indicies_layout.rank)]);
        const int32_t table_index[2] = {token, grad_index[grad_layout.rank - 1]};
        atomicAdd(&table[index_to_position(table_index, table_layout.stride, 2)], grad[grad_pos]);
    }

    void embedding_backward(const View& table, const View& indicies, const View& grad, const int32_t features){
        if (grad.rank > 16 || indicies.rank > 16 || table.rank > 16)
            throw std::runtime_error(std::format("CUDA embedding_backward received exceptionally large shape, received table.rank: {}, indicies.rank: {}, grad.rank: {}", table.rank, indicies.rank, grad.rank));

        Context& ctx = Context::get();
        long long grad_size = prod(grad.shape, grad.rank);
        if (grad_size == 0) return;

        int32_t threads = 256;
        int32_t blocks = static_cast<int32_t>((grad_size + threads - 1) / threads);
        if (is_contiguous(grad) && is_contiguous(indicies) && is_contiguous(table))
            embedding_backward_contiguous_kernel<<<blocks, threads, 0, static_cast<cudaStream_t>(ctx.getStream())>>>(table.data, indicies.data, grad.data, features, grad_size);
        else
            embedding_backward_kernel<<<blocks, threads, 0, static_cast<cudaStream_t>(ctx.getStream())>>>(table.data, layout_from(table), indicies.data, layout_from(indicies), grad.data, layout_from(grad), grad_size);
        ctx.checkLastError();
    }
}
