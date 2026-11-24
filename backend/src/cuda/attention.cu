#include "tsgrad/ops.h"
#include "cudahelpers.hpp"

#include <cuda_runtime.h>
#include <math_constants.h>
#include <format>
#include <stdexcept>

namespace tsgrad::cuda {
namespace {

constexpr int32_t THREADS = 256;

__device__ long long position4(
    const Layout& layout, int32_t batch, int32_t head, int32_t time, int32_t feature
) {
    return static_cast<long long>(batch) * layout.stride[0]
        + static_cast<long long>(head) * layout.stride[1]
        + static_cast<long long>(time) * layout.stride[2]
        + static_cast<long long>(feature) * layout.stride[3];
}

__device__ long long position3(
    const Layout& layout, int32_t batch, int32_t head, int32_t time
) {
    return static_cast<long long>(batch) * layout.stride[0]
        + static_cast<long long>(head) * layout.stride[1]
        + static_cast<long long>(time) * layout.stride[2];
}

template <int32_t HeadDim>
__global__ void attention_forward_kernel(
    float* out, Layout out_layout, float* lse, Layout lse_layout,
    const float* query, Layout query_layout,
    const float* key, Layout key_layout,
    const float* value, Layout value_layout,
    float scale, bool causal
) {
    constexpr int32_t QueryTile = HeadDim <= 64 ? 32 : 16;
    constexpr int32_t KeyTile = HeadDim <= 64 ? 32 : 16;
    __shared__ float query_tile[QueryTile * HeadDim];
    __shared__ float key_tile[KeyTile * HeadDim];
    __shared__ float value_tile[KeyTile * HeadDim];
    __shared__ float score_tile[QueryTile * KeyTile];
    __shared__ float output_tile[QueryTile * HeadDim];
    __shared__ float row_maximum[QueryTile];
    __shared__ float row_denominator[QueryTile];

    const int32_t time = query_layout.shape[2];
    const int32_t heads = query_layout.shape[1];
    const int32_t query_tiles = (time + QueryTile - 1) / QueryTile;
    const int32_t query_tile_index = blockIdx.x % query_tiles;
    const int32_t head = (blockIdx.x / query_tiles) % heads;
    const int32_t batch = blockIdx.x / (query_tiles * heads);
    const int32_t query_start = query_tile_index * QueryTile;
    const int32_t tid = threadIdx.x;

    for (int32_t index = tid; index < QueryTile * HeadDim; index += blockDim.x) {
        const int32_t row = index / HeadDim;
        const int32_t feature = index % HeadDim;
        const int32_t query_time = query_start + row;
        query_tile[index] = query_time < time
            ? query[position4(query_layout, batch, head, query_time, feature)]
            : 0.0f;
        output_tile[index] = 0.0f;
    }
    if (tid < QueryTile) {
        row_maximum[tid] = -CUDART_INF_F;
        row_denominator[tid] = 0.0f;
    }
    __syncthreads();

    for (int32_t key_start = 0; key_start < time; key_start += KeyTile) {
        for (int32_t index = tid; index < KeyTile * HeadDim; index += blockDim.x) {
            const int32_t row = index / HeadDim;
            const int32_t feature = index % HeadDim;
            const int32_t key_time = key_start + row;
            key_tile[index] = key_time < time
                ? key[position4(key_layout, batch, head, key_time, feature)]
                : 0.0f;
            value_tile[index] = key_time < time
                ? value[position4(value_layout, batch, head, key_time, feature)]
                : 0.0f;
        }
        __syncthreads();

        for (int32_t index = tid; index < QueryTile * KeyTile; index += blockDim.x) {
            const int32_t query_row = index / KeyTile;
            const int32_t key_row = index % KeyTile;
            const int32_t query_time = query_start + query_row;
            const int32_t key_time = key_start + key_row;
            float score = -CUDART_INF_F;
            if (query_time < time && key_time < time && (!causal || key_time <= query_time)) {
                score = 0.0f;
                for (int32_t feature = 0; feature < HeadDim; feature++)
                    score += query_tile[query_row * HeadDim + feature]
                        * key_tile[key_row * HeadDim + feature];
                score *= scale;
            }
            score_tile[index] = score;
        }
        __syncthreads();

        if (tid < QueryTile && query_start + tid < time) {
            float tile_maximum = -CUDART_INF_F;
            for (int32_t key_row = 0; key_row < KeyTile; key_row++)
                tile_maximum = fmaxf(tile_maximum, score_tile[tid * KeyTile + key_row]);
            const float next_maximum = fmaxf(row_maximum[tid], tile_maximum);
            const float previous_scale = expf(row_maximum[tid] - next_maximum);
            float tile_denominator = 0.0f;
            for (int32_t feature = 0; feature < HeadDim; feature++)
                output_tile[tid * HeadDim + feature] *= previous_scale;
            for (int32_t key_row = 0; key_row < KeyTile; key_row++) {
                const float score = score_tile[tid * KeyTile + key_row];
                const float probability_scale = score == -CUDART_INF_F
                    ? 0.0f
                    : expf(score - next_maximum);
                tile_denominator += probability_scale;
                for (int32_t feature = 0; feature < HeadDim; feature++)
                    output_tile[tid * HeadDim + feature] += probability_scale
                        * value_tile[key_row * HeadDim + feature];
            }
            row_denominator[tid] = row_denominator[tid] * previous_scale + tile_denominator;
            row_maximum[tid] = next_maximum;
        }
        __syncthreads();
    }

    for (int32_t index = tid; index < QueryTile * HeadDim; index += blockDim.x) {
        const int32_t row = index / HeadDim;
        const int32_t feature = index % HeadDim;
        const int32_t query_time = query_start + row;
        if (query_time < time)
            out[position4(out_layout, batch, head, query_time, feature)]
                = output_tile[index] / row_denominator[row];
    }
    if (tid < QueryTile && query_start + tid < time)
        lse[position3(lse_layout, batch, head, query_start + tid)]
            = row_maximum[tid] + logf(row_denominator[tid]);
}

template <int32_t HeadDim>
__global__ void attention_backward_query_kernel(
    float* grad_query, Layout grad_query_layout,
    const float* grad_out, Layout grad_out_layout,
    const float* query, Layout query_layout,
    const float* key, Layout key_layout,
    const float* value, Layout value_layout,
    const float* out, Layout out_layout,
    const float* lse, Layout lse_layout,
    float scale, bool causal
) {
    constexpr int32_t QueryTile = HeadDim <= 64 ? 32 : 16;
    constexpr int32_t KeyTile = HeadDim <= 64 ? 32 : 16;
    __shared__ float query_tile[QueryTile * HeadDim];
    __shared__ float grad_out_tile[QueryTile * HeadDim];
    __shared__ float grad_query_tile[QueryTile * HeadDim];
    __shared__ float key_tile[KeyTile * HeadDim];
    __shared__ float value_tile[KeyTile * HeadDim];
    __shared__ float score_tile[QueryTile * KeyTile];
    __shared__ float delta[QueryTile];

    const int32_t time = query_layout.shape[2];
    const int32_t heads = query_layout.shape[1];
    const int32_t query_tiles = (time + QueryTile - 1) / QueryTile;
    const int32_t query_tile_index = blockIdx.x % query_tiles;
    const int32_t head = (blockIdx.x / query_tiles) % heads;
    const int32_t batch = blockIdx.x / (query_tiles * heads);
    const int32_t query_start = query_tile_index * QueryTile;
    const int32_t tid = threadIdx.x;

    for (int32_t index = tid; index < QueryTile * HeadDim; index += blockDim.x) {
        const int32_t row = index / HeadDim;
        const int32_t feature = index % HeadDim;
        const int32_t query_time = query_start + row;
        query_tile[index] = query_time < time
            ? query[position4(query_layout, batch, head, query_time, feature)]
            : 0.0f;
        grad_out_tile[index] = query_time < time
            ? grad_out[position4(grad_out_layout, batch, head, query_time, feature)]
            : 0.0f;
        grad_query_tile[index] = 0.0f;
    }
    for (int32_t row = tid; row < QueryTile; row += blockDim.x)
        delta[row] = 0.0f;
    __syncthreads();

    for (int32_t index = tid; index < QueryTile * HeadDim; index += blockDim.x) {
        const int32_t row = index / HeadDim;
        const int32_t feature = index % HeadDim;
        const int32_t query_time = query_start + row;
        if (query_time < time) {
            atomicAdd(
                &delta[row],
                out[position4(out_layout, batch, head, query_time, feature)]
                    * grad_out_tile[index]);
        }
    }
    __syncthreads();

    for (int32_t key_start = 0; key_start < time; key_start += KeyTile) {
        for (int32_t index = tid; index < KeyTile * HeadDim; index += blockDim.x) {
            const int32_t row = index / HeadDim;
            const int32_t feature = index % HeadDim;
            const int32_t key_time = key_start + row;
            key_tile[index] = key_time < time
                ? key[position4(key_layout, batch, head, key_time, feature)]
                : 0.0f;
            value_tile[index] = key_time < time
                ? value[position4(value_layout, batch, head, key_time, feature)]
                : 0.0f;
        }
        __syncthreads();

        for (int32_t index = tid; index < QueryTile * KeyTile; index += blockDim.x) {
            const int32_t query_row = index / KeyTile;
            const int32_t key_row = index % KeyTile;
            const int32_t query_time = query_start + query_row;
            const int32_t key_time = key_start + key_row;
            float probability = 0.0f;
            if (query_time < time && key_time < time && (!causal || key_time <= query_time)) {
                float query_key_dot = 0.0f;
                for (int32_t feature = 0; feature < HeadDim; feature++)
                    query_key_dot += query_tile[query_row * HeadDim + feature]
                        * key_tile[key_row * HeadDim + feature];
                probability = expf(
                    scale * query_key_dot
                    - lse[position3(lse_layout, batch, head, query_time)]);
            }
            score_tile[index] = probability;
        }
        __syncthreads();

        for (int32_t index = tid; index < QueryTile * KeyTile; index += blockDim.x) {
            const int32_t query_row = index / KeyTile;
            const int32_t key_row = index % KeyTile;
            const float probability = score_tile[index];
            float grad_value_dot = 0.0f;
            if (probability != 0.0f) {
                for (int32_t feature = 0; feature < HeadDim; feature++)
                    grad_value_dot += grad_out_tile[query_row * HeadDim + feature]
                        * value_tile[key_row * HeadDim + feature];
            }
            score_tile[index] = probability * (grad_value_dot - delta[query_row]);
        }
        __syncthreads();

        for (int32_t index = tid; index < QueryTile * HeadDim; index += blockDim.x) {
            const int32_t query_row = index / HeadDim;
            const int32_t feature = index % HeadDim;
            float accumulator = grad_query_tile[index];
            for (int32_t key_row = 0; key_row < KeyTile; key_row++)
                accumulator += scale * score_tile[query_row * KeyTile + key_row]
                    * key_tile[key_row * HeadDim + feature];
            grad_query_tile[index] = accumulator;
        }
        __syncthreads();
    }

    for (int32_t index = tid; index < QueryTile * HeadDim; index += blockDim.x) {
        const int32_t row = index / HeadDim;
        const int32_t feature = index % HeadDim;
        const int32_t query_time = query_start + row;
        if (query_time < time)
            grad_query[position4(grad_query_layout, batch, head, query_time, feature)]
                = grad_query_tile[index];
    }
}

template <int32_t HeadDim>
__global__ void attention_backward_key_value_kernel(
    float* grad_key, Layout grad_key_layout,
    float* grad_value, Layout grad_value_layout,
    const float* grad_out, Layout grad_out_layout,
    const float* query, Layout query_layout,
    const float* key, Layout key_layout,
    const float* value, Layout value_layout,
    const float* out, Layout out_layout,
    const float* lse, Layout lse_layout,
    float scale, bool causal
) {
    constexpr int32_t Tile = HeadDim <= 64 ? 16 : 8;
    __shared__ float key_tile[Tile * HeadDim];
    __shared__ float value_tile[Tile * HeadDim];
    __shared__ float grad_key_tile[Tile * HeadDim];
    __shared__ float grad_value_tile[Tile * HeadDim];
    __shared__ float query_tile[Tile * HeadDim];
    __shared__ float grad_out_tile[Tile * HeadDim];
    __shared__ float probability_tile[Tile * Tile];
    __shared__ float grad_score_tile[Tile * Tile];
    __shared__ float delta[Tile];

    const int32_t time = query_layout.shape[2];
    const int32_t heads = query_layout.shape[1];
    const int32_t key_tiles = (time + Tile - 1) / Tile;
    const int32_t key_tile_index = blockIdx.x % key_tiles;
    const int32_t head = (blockIdx.x / key_tiles) % heads;
    const int32_t batch = blockIdx.x / (key_tiles * heads);
    const int32_t key_start = key_tile_index * Tile;
    const int32_t tid = threadIdx.x;

    for (int32_t index = tid; index < Tile * HeadDim; index += blockDim.x) {
        const int32_t row = index / HeadDim;
        const int32_t feature = index % HeadDim;
        const int32_t key_time = key_start + row;
        key_tile[index] = key_time < time
            ? key[position4(key_layout, batch, head, key_time, feature)]
            : 0.0f;
        value_tile[index] = key_time < time
            ? value[position4(value_layout, batch, head, key_time, feature)]
            : 0.0f;
        grad_key_tile[index] = 0.0f;
        grad_value_tile[index] = 0.0f;
    }
    __syncthreads();

    for (int32_t query_start = 0; query_start < time; query_start += Tile) {
        for (int32_t index = tid; index < Tile * HeadDim; index += blockDim.x) {
            const int32_t row = index / HeadDim;
            const int32_t feature = index % HeadDim;
            const int32_t query_time = query_start + row;
            query_tile[index] = query_time < time
                ? query[position4(query_layout, batch, head, query_time, feature)]
                : 0.0f;
            grad_out_tile[index] = query_time < time
                ? grad_out[position4(grad_out_layout, batch, head, query_time, feature)]
                : 0.0f;
        }
        for (int32_t row = tid; row < Tile; row += blockDim.x)
            delta[row] = 0.0f;
        __syncthreads();

        for (int32_t index = tid; index < Tile * HeadDim; index += blockDim.x) {
            const int32_t row = index / HeadDim;
            const int32_t feature = index % HeadDim;
            const int32_t query_time = query_start + row;
            if (query_time < time) {
                atomicAdd(
                    &delta[row],
                    out[position4(out_layout, batch, head, query_time, feature)]
                        * grad_out_tile[index]);
            }
        }
        __syncthreads();

        for (int32_t index = tid; index < Tile * Tile; index += blockDim.x) {
            const int32_t query_row = index / Tile;
            const int32_t key_row = index % Tile;
            const int32_t query_time = query_start + query_row;
            const int32_t key_time = key_start + key_row;
            float probability = 0.0f;
            if (query_time < time && key_time < time && (!causal || key_time <= query_time)) {
                float query_key_dot = 0.0f;
                for (int32_t feature = 0; feature < HeadDim; feature++)
                    query_key_dot += query_tile[query_row * HeadDim + feature]
                        * key_tile[key_row * HeadDim + feature];
                probability = expf(
                    scale * query_key_dot
                    - lse[position3(lse_layout, batch, head, query_time)]);
            }
            probability_tile[index] = probability;
        }
        __syncthreads();

        for (int32_t index = tid; index < Tile * Tile; index += blockDim.x) {
            const int32_t query_row = index / Tile;
            const int32_t key_row = index % Tile;
            const float probability = probability_tile[index];
            float grad_value_dot = 0.0f;
            if (probability != 0.0f) {
                for (int32_t feature = 0; feature < HeadDim; feature++)
                    grad_value_dot += grad_out_tile[query_row * HeadDim + feature]
                        * value_tile[key_row * HeadDim + feature];
            }
            grad_score_tile[index] = probability * (grad_value_dot - delta[query_row]);
        }
        __syncthreads();

        for (int32_t index = tid; index < Tile * HeadDim; index += blockDim.x) {
            const int32_t key_row = index / HeadDim;
            const int32_t feature = index % HeadDim;
            float key_accumulator = grad_key_tile[index];
            float value_accumulator = grad_value_tile[index];
            for (int32_t query_row = 0; query_row < Tile; query_row++) {
                const float probability = probability_tile[query_row * Tile + key_row];
                const float grad_score = grad_score_tile[query_row * Tile + key_row];
                key_accumulator += scale * grad_score
                    * query_tile[query_row * HeadDim + feature];
                value_accumulator += probability
                    * grad_out_tile[query_row * HeadDim + feature];
            }
            grad_key_tile[index] = key_accumulator;
            grad_value_tile[index] = value_accumulator;
        }
        __syncthreads();
    }

    for (int32_t index = tid; index < Tile * HeadDim; index += blockDim.x) {
        const int32_t row = index / HeadDim;
        const int32_t feature = index % HeadDim;
        const int32_t key_time = key_start + row;
        if (key_time < time) {
            grad_key[position4(grad_key_layout, batch, head, key_time, feature)]
                = grad_key_tile[index];
            grad_value[position4(grad_value_layout, batch, head, key_time, feature)]
                = grad_value_tile[index];
        }
    }
}

void require_rank(const View& view, const char* operation, const char* name) {
    if (view.rank > 16)
        throw std::runtime_error(std::format(
            "CUDA {} supports ranks up to 16, received {}.rank: {}", operation, name, view.rank));
}

template <int32_t HeadDim>
void launch_forward(
    const View& out, const View& lse,
    const View& query, const View& key, const View& value,
    float scale, bool causal
) {
    constexpr int32_t QueryTile = HeadDim <= 64 ? 32 : 16;
    const int32_t blocks = query.shape[0] * query.shape[1]
        * ((query.shape[2] + QueryTile - 1) / QueryTile);
    Context& ctx = Context::get();
    attention_forward_kernel<HeadDim><<<blocks, THREADS, 0,
        static_cast<cudaStream_t>(ctx.getStream())>>>(
        out.data, layout_from(out), lse.data, layout_from(lse),
        query.data, layout_from(query), key.data, layout_from(key),
        value.data, layout_from(value), scale, causal);
    ctx.checkLastError();
}

template <int32_t HeadDim>
void launch_backward(
    const View& grad_query, const View& grad_key, const View& grad_value,
    const View& grad_out, const View& query, const View& key, const View& value,
    const View& out, const View& lse, float scale, bool causal
) {
    constexpr int32_t QueryTile = HeadDim <= 64 ? 32 : 16;
    constexpr int32_t KeyValueTile = HeadDim <= 64 ? 16 : 8;
    const int32_t query_blocks = query.shape[0] * query.shape[1]
        * ((query.shape[2] + QueryTile - 1) / QueryTile);
    const int32_t key_value_blocks = query.shape[0] * query.shape[1]
        * ((query.shape[2] + KeyValueTile - 1) / KeyValueTile);
    Context& ctx = Context::get();
    cudaStream_t stream = static_cast<cudaStream_t>(ctx.getStream());
    attention_backward_query_kernel<HeadDim><<<query_blocks, THREADS, 0, stream>>>(
        grad_query.data, layout_from(grad_query),
        grad_out.data, layout_from(grad_out),
        query.data, layout_from(query), key.data, layout_from(key),
        value.data, layout_from(value), out.data, layout_from(out),
        lse.data, layout_from(lse), scale, causal);
    ctx.checkLastError();
    attention_backward_key_value_kernel<HeadDim><<<key_value_blocks, THREADS, 0, stream>>>(
        grad_key.data, layout_from(grad_key),
        grad_value.data, layout_from(grad_value),
        grad_out.data, layout_from(grad_out),
        query.data, layout_from(query), key.data, layout_from(key),
        value.data, layout_from(value), out.data, layout_from(out),
        lse.data, layout_from(lse), scale, causal);
    ctx.checkLastError();
}

} // namespace

void attention(
    const View& out, const View& lse,
    const View& query, const View& key, const View& value,
    float scale, bool causal
) {
    require_rank(out, "attention", "out");
    require_rank(lse, "attention", "lse");
    require_rank(query, "attention", "query");
    require_rank(key, "attention", "key");
    require_rank(value, "attention", "value");
    switch (query.shape[3]) {
        case 32: launch_forward<32>(out, lse, query, key, value, scale, causal); break;
        case 64: launch_forward<64>(out, lse, query, key, value, scale, causal); break;
        case 128: launch_forward<128>(out, lse, query, key, value, scale, causal); break;
        default: throw std::runtime_error("CUDA attention received an unsupported head dimension");
    }
}

void attention_backward(
    const View& grad_query, const View& grad_key, const View& grad_value,
    const View& grad_out, const View& query, const View& key, const View& value,
    const View& out, const View& lse, float scale, bool causal
) {
    require_rank(grad_query, "attention_backward", "grad_query");
    require_rank(grad_key, "attention_backward", "grad_key");
    require_rank(grad_value, "attention_backward", "grad_value");
    require_rank(grad_out, "attention_backward", "grad_out");
    switch (query.shape[3]) {
        case 32:
            launch_backward<32>(
                grad_query, grad_key, grad_value, grad_out,
                query, key, value, out, lse, scale, causal);
            break;
        case 64:
            launch_backward<64>(
                grad_query, grad_key, grad_value, grad_out,
                query, key, value, out, lse, scale, causal);
            break;
        case 128:
            launch_backward<128>(
                grad_query, grad_key, grad_value, grad_out,
                query, key, value, out, lse, scale, causal);
            break;
        default: throw std::runtime_error("CUDA attention_backward received an unsupported head dimension");
    }
}

} // namespace tsgrad::cuda
