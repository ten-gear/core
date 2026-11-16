#include "tsgrad/ops.h"
#include "helpers.hpp"
#include <vector>
#include <format>
#include <span>

namespace tsgrad{
void embedding(const View& out, const View& indicies, const View& table, const int32_t features){
    if (table.rank != 2)
        throw std::runtime_error(std::format("embedding received table with dimensions not equal to 2, received: {}", table.rank));

    if (indicies.rank + 1 != out.rank)
        throw std::runtime_error(std::format("embedding received indicies/out with bad dimensions, received indicies.rank: {}, out.rank: {}", indicies.rank, out.rank));

    for (int32_t i = 0; i < indicies.rank; i++){
        if (indicies.shape[i] != out.shape[i]){
            std::span<const int32_t> indiciesview{indicies.shape, static_cast<size_t>(indicies.rank)};
            std::span<const int32_t> outview{out.shape, static_cast<size_t>(out.rank)};
            throw std::runtime_error(std::format("embedding received indicies/out with bad shapes, received indicies.shape: {:n}, out.shape: {:n}", indiciesview, outview));
        }
    }

    if (table.shape[1] != features || out.shape[out.rank - 1] != features){
        std::span<const int32_t> tableview{table.shape, static_cast<size_t>(table.rank)};
        std::span<const int32_t> outview{out.shape, static_cast<size_t>(out.rank)};
        throw std::runtime_error(std::format("embedding received out with bad shape, received out.shape: {:n}, table.shape: {:n}, features: {}", outview, tableview, features));
    }

    if constexpr (TSGRAD_HAS_CUDA){
        tsgrad::cuda::embedding(out, indicies, table, features);
        return;
    }

    long long size = prod(out.shape, out.rank);
    if (size == 0) return;

    if (is_contiguous(out) && is_contiguous(indicies) && is_contiguous(table)){
        #pragma omp parallel for if (size >= THRESHOLD)
        for (long long ordinal = 0; ordinal < size; ordinal++){
            const long long row = ordinal / features;
            const long long col = ordinal % features;
            out.data[ordinal] = table.data[static_cast<long long>(indicies.data[row]) * features + col];
        }
        return;
    }

    #pragma omp parallel if (size >= THRESHOLD)
    {
        std::vector<int32_t> out_index(static_cast<size_t>(out.rank), 0);

        #pragma omp for
        for (long long ordinal = 0; ordinal < size; ordinal++){
            position_to_index(ordinal, out.shape, out_index.data(), out.rank);
            const long long out_pos = index_to_position(out_index.data(), out.stride, out.rank);
            const int32_t token = static_cast<int32_t>(indicies.data[index_to_position(out_index.data(), indicies.stride, indicies.rank)]);
            const int32_t table_index[2] = {token, out_index[out.rank - 1]};
            out.data[out_pos] = table.data[index_to_position(table_index, table.stride, 2)];
        }
    }
}

void embedding_backward(const View& table, const View& indicies, const View& grad, const int32_t features){
    if (table.rank != 2)
        throw std::runtime_error(std::format("embedding_backward received table with dimensions not equal to 2, received: {}", table.rank));

    if (indicies.rank + 1 != grad.rank)
        throw std::runtime_error(std::format("embedding_backward received indicies/grad with bad dimensions, received indicies.rank: {}, grad.rank: {}", indicies.rank, grad.rank));

    for (int32_t i = 0; i < indicies.rank; i++){
        if (indicies.shape[i] != grad.shape[i]){
            std::span<const int32_t> indiciesview{indicies.shape, static_cast<size_t>(indicies.rank)};
            std::span<const int32_t> gradview{grad.shape, static_cast<size_t>(grad.rank)};
            throw std::runtime_error(std::format("embedding_backward received indicies/grad with bad shapes, received indicies.shape: {:n}, grad.shape: {:n}", indiciesview, gradview));
        }
    }

    if (table.shape[1] != features || grad.shape[grad.rank - 1] != features){
        std::span<const int32_t> tableview{table.shape, static_cast<size_t>(table.rank)};
        std::span<const int32_t> gradview{grad.shape, static_cast<size_t>(grad.rank)};
        throw std::runtime_error(std::format("embedding_backward received grad with bad shape, received grad.shape: {:n}, table.shape: {:n}, features: {}", gradview, tableview, features));
    }

    if constexpr (TSGRAD_HAS_CUDA){
        tsgrad::cuda::embedding_backward(table, indicies, grad, features);
        return;
    }

    long long size = prod(grad.shape, grad.rank);
    if (size == 0) return;

    if (is_contiguous(grad) && is_contiguous(indicies) && is_contiguous(table)){
        #pragma omp parallel for if (size >= THRESHOLD)
        for (long long ordinal = 0; ordinal < size; ordinal++){
            const long long row = ordinal / features;
            const long long col = ordinal % features;
            #pragma omp atomic
            table.data[static_cast<long long>(indicies.data[row]) * features + col] += grad.data[ordinal];
        }
        return;
    }

    #pragma omp parallel if (size >= THRESHOLD)
    {
        std::vector<int32_t> grad_index(static_cast<size_t>(grad.rank), 0);

        #pragma omp for
        for (long long ordinal = 0; ordinal < size; ordinal++){
            position_to_index(ordinal, grad.shape, grad_index.data(), grad.rank);
            const long long grad_pos = index_to_position(grad_index.data(), grad.stride, grad.rank);
            const int32_t token = static_cast<int32_t>(indicies.data[index_to_position(grad_index.data(), indicies.stride, indicies.rank)]);
            const int32_t table_index[2] = {token, grad_index[grad.rank - 1]};

            #pragma omp atomic
            table.data[index_to_position(table_index, table.stride, 2)] += grad.data[grad_pos];
        }
    }

    
}

} //namespace
