#include "tsgrad/ops.h"
#include "cudahelpers.hpp"

#include <cuda_runtime.h>
#include <math_constants.h>
#include <format>
#include <stdexcept>

// FlashAttention-2 schedule on CUDA cores, in FP32.
//
// One block owns a query tile and streams key/value tiles. The T×T score
// matrix is never stored. QKᵀ and PV are GEMMs done by the whole block:
// K is staged transposed so a warp's dot reads consecutive shared-memory
// columns, and each thread keeps its slice of the output in registers.
// The online row max / row sum rescales those registers between tiles.
// Causal tiles that sit entirely past the diagonal are skipped.
//
// Head dimensions 32, 64, and 128. Tile sizes stay inside the 48KB
// shared-memory default so the kernels launch without an opt-in carveout.
//
// Backward uses the same full-block GEMM. dQ keeps K in registers' shared
// tile (both layouts) instead of loading it twice, and D_i = sum(O * dO)
// is one warp reduction per row rather than an atomic on every tile.

namespace tsgrad::cuda {
namespace {

constexpr int32_t THREADS = 256;

template <int32_t HeadDim>
struct Tile {
    // Query rows per block, key columns per iteration.
    static constexpr int32_t BR = HeadDim == 128 ? 16 : HeadDim == 64 ? 32 : 64;
    static constexpr int32_t BC = HeadDim == 128 ? 16 : 32;
    // Pad the transposed leading dimension so it is not a multiple of 32 banks.
    static constexpr int32_t KT = BC + 4;
    static constexpr int32_t SCORES = (BR * BC) / THREADS;
    static constexpr int32_t OUTPUTS = (BR * HeadDim) / THREADS;
    static constexpr int32_t KEY_OUTPUTS = (BC * HeadDim) / THREADS;
};

static_assert(Tile<32>::SCORES > 0 && Tile<64>::SCORES > 0 && Tile<128>::SCORES > 0);
static_assert(Tile<32>::OUTPUTS > 0 && Tile<64>::OUTPUTS > 0 && Tile<128>::OUTPUTS > 0);
static_assert(Tile<32>::KEY_OUTPUTS > 0 && Tile<64>::KEY_OUTPUTS > 0 && Tile<128>::KEY_OUTPUTS > 0);

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

__device__ bool vector4_layout(const float* src, const Layout& layout, int32_t batch, int32_t head) {
    if (layout.stride[3] != 1 || (layout.stride[2] & 3) || (layout.stride[1] & 3) || (layout.stride[0] & 3))
        return false;
    const float* origin = src + position4(layout, batch, head, 0, 0);
    return (reinterpret_cast<uintptr_t>(origin) & 0xf) == 0;
}

template <int32_t HeadDim, int32_t Rows>
__device__ void load_rows(
    float* dst, const float* src, const Layout& layout,
    int32_t batch, int32_t head, int32_t start, int32_t time
) {
    const int32_t tid = threadIdx.x;
    if constexpr (HeadDim % 4 == 0) {
        if (vector4_layout(src, layout, batch, head)) {
            const int32_t vecs = (Rows * HeadDim) / 4;
            for (int32_t index = tid; index < vecs; index += THREADS) {
                const int32_t elem = index * 4;
                const int32_t row = elem / HeadDim;
                const int32_t feature = elem % HeadDim;
                const int32_t token = start + row;
                float4 value = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                if (token < time)
                    value = *reinterpret_cast<const float4*>(
                        src + position4(layout, batch, head, token, feature));
                *reinterpret_cast<float4*>(dst + elem) = value;
            }
            return;
        }
    }
    for (int32_t index = tid; index < Rows * HeadDim; index += THREADS) {
        const int32_t row = index / HeadDim;
        const int32_t feature = index % HeadDim;
        const int32_t token = start + row;
        dst[index] = token < time
            ? src[position4(layout, batch, head, token, feature)]
            : 0.0f;
    }
}

// dst[feature, column] with a padded column stride. Global reads stay contiguous
// in the head dimension; the transpose happens on the shared store.
template <int32_t HeadDim, int32_t Cols, int32_t KT>
__device__ void load_transposed(
    float* dst, const float* src, const Layout& layout,
    int32_t batch, int32_t head, int32_t start, int32_t time
) {
    const int32_t tid = threadIdx.x;
    if constexpr (HeadDim % 4 == 0) {
        if (vector4_layout(src, layout, batch, head)) {
            const int32_t vecs = (Cols * HeadDim) / 4;
            for (int32_t index = tid; index < vecs; index += THREADS) {
                const int32_t elem = index * 4;
                const int32_t column = elem / HeadDim;
                const int32_t feature = elem % HeadDim;
                const int32_t token = start + column;
                float4 value = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                if (token < time)
                    value = *reinterpret_cast<const float4*>(
                        src + position4(layout, batch, head, token, feature));
                dst[(feature + 0) * KT + column] = value.x;
                dst[(feature + 1) * KT + column] = value.y;
                dst[(feature + 2) * KT + column] = value.z;
                dst[(feature + 3) * KT + column] = value.w;
            }
            return;
        }
    }
    for (int32_t index = tid; index < Cols * HeadDim; index += THREADS) {
        const int32_t column = index / HeadDim;
        const int32_t feature = index % HeadDim;
        const int32_t token = start + column;
        const float value = token < time
            ? src[position4(layout, batch, head, token, feature)]
            : 0.0f;
        dst[feature * KT + column] = value;
    }
}

// One global read, then both layouts the backward GEMMs need:
// row-major [column, feature] for dS @ K, transposed [feature, column] for QKᵀ.
template <int32_t HeadDim, int32_t Cols, int32_t KT>
__device__ void load_key_both(
    float* row_major, float* transposed, const float* src, const Layout& layout,
    int32_t batch, int32_t head, int32_t start, int32_t time
) {
    const int32_t tid = threadIdx.x;
    if constexpr (HeadDim % 4 == 0) {
        if (vector4_layout(src, layout, batch, head)) {
            const int32_t vecs = (Cols * HeadDim) / 4;
            for (int32_t index = tid; index < vecs; index += THREADS) {
                const int32_t elem = index * 4;
                const int32_t column = elem / HeadDim;
                const int32_t feature = elem % HeadDim;
                const int32_t token = start + column;
                float4 value = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                if (token < time)
                    value = *reinterpret_cast<const float4*>(
                        src + position4(layout, batch, head, token, feature));
                *reinterpret_cast<float4*>(row_major + elem) = value;
                transposed[(feature + 0) * KT + column] = value.x;
                transposed[(feature + 1) * KT + column] = value.y;
                transposed[(feature + 2) * KT + column] = value.z;
                transposed[(feature + 3) * KT + column] = value.w;
            }
            return;
        }
    }
    for (int32_t index = tid; index < Cols * HeadDim; index += THREADS) {
        const int32_t column = index / HeadDim;
        const int32_t feature = index % HeadDim;
        const int32_t token = start + column;
        const float value = token < time
            ? src[position4(layout, batch, head, token, feature)]
            : 0.0f;
        row_major[index] = value;
        transposed[feature * KT + column] = value;
    }
}

template <int32_t HeadDim, int32_t Cols>
__device__ void load_row_major(
    float* dst, const float* src, const Layout& layout,
    int32_t batch, int32_t head, int32_t start, int32_t time
) {
    load_rows<HeadDim, Cols>(dst, src, layout, batch, head, start, time);
}

// Each thread owns SCORES locations of the [BR, BC] score tile, all in one
// key column. rightT is [HeadDim, KT]. The raw dot is written to scores.
template <int32_t HeadDim, int32_t BR, int32_t BC, int32_t KT>
__device__ void gemm_dots(
    const float* left, const float* rightT, float* scores
) {
    constexpr int32_t N = (BR * BC) / THREADS;
    const int32_t tid = threadIdx.x;
    const int32_t column = tid % BC;
    int32_t row[N];
    float acc[N];
    #pragma unroll
    for (int32_t i = 0; i < N; i++) {
        row[i] = (tid + i * THREADS) / BC;
        acc[i] = 0.0f;
    }

    #pragma unroll 4
    for (int32_t feature = 0; feature < HeadDim; feature++) {
        const float right = rightT[feature * KT + column];
        #pragma unroll
        for (int32_t i = 0; i < N; i++)
            acc[i] = fmaf(left[row[i] * HeadDim + feature], right, acc[i]);
    }

    #pragma unroll
    for (int32_t i = 0; i < N; i++)
        scores[row[i] * BC + column] = acc[i];
}

template <int32_t HeadDim, int32_t BR, int32_t N>
__device__ void output_coords(int32_t (&rows)[N], int32_t& feature) {
    const int32_t tid = threadIdx.x;
    feature = tid % HeadDim;
    #pragma unroll
    for (int32_t i = 0; i < N; i++)
        rows[i] = (tid + i * THREADS) / HeadDim;
}

// acc[i] belongs to rows[i]. coeff is [BR, BC], right is [BC, HeadDim].
template <int32_t HeadDim, int32_t BC, int32_t N>
__device__ void accumulate_outer(
    float (&acc)[N], const int32_t (&rows)[N], int32_t feature,
    const float* coeff, const float* right, float coeff_scale
) {
    static_assert(BC % 4 == 0, "key tile is a multiple of the float4 width");
    for (int32_t key = 0; key < BC; key += 4) {
        #pragma unroll
        for (int32_t i = 0; i < N; i++) {
            const float4 weight = *reinterpret_cast<const float4*>(coeff + rows[i] * BC + key);
            acc[i] = fmaf(coeff_scale * weight.x, right[(key + 0) * HeadDim + feature], acc[i]);
            acc[i] = fmaf(coeff_scale * weight.y, right[(key + 1) * HeadDim + feature], acc[i]);
            acc[i] = fmaf(coeff_scale * weight.z, right[(key + 2) * HeadDim + feature], acc[i]);
            acc[i] = fmaf(coeff_scale * weight.w, right[(key + 3) * HeadDim + feature], acc[i]);
        }
    }
}

// Threads that own a score tile. Four consecutive key columns when the tile
// divides that way, so QKᵀ reads float4 fragments instead of one column.
template <int32_t HeadDim, int32_t BR, int32_t BC>
struct ScoreThread {
    static constexpr int32_t N = (BR * BC) / THREADS;
    static constexpr bool VECTOR = (N % 4 == 0) && (BC % 4 == 0) && (HeadDim % 4 == 0);
    static constexpr int32_t COLS = VECTOR ? 4 : 1;
    static constexpr int32_t ROWS = N / COLS;
    static constexpr int32_t COL_GROUPS = BC / COLS;

    int32_t col;
    int32_t row[ROWS];

    __device__ ScoreThread() {
        const int32_t tid = threadIdx.x;
        col = (tid % COL_GROUPS) * COLS;
        const int32_t row_thread = tid / COL_GROUPS;
        constexpr int32_t ROW_THREADS = THREADS / COL_GROUPS;
        #pragma unroll
        for (int32_t i = 0; i < ROWS; i++)
            row[i] = row_thread + i * ROW_THREADS;
    }
};

template <int32_t HeadDim, int32_t BR, int32_t BC, int32_t KT>
__device__ void gemm_probabilities(
    const float* left, const float* rightT, float* scores,
    int32_t query_start, int32_t key_start, int32_t time, bool diagonal,
    const float* lse, const Layout& lse_layout, int32_t batch, int32_t head, float scale
) {
    ScoreThread<HeadDim, BR, BC> owned;
    constexpr int32_t ROWS = ScoreThread<HeadDim, BR, BC>::ROWS;
    constexpr int32_t COLS = ScoreThread<HeadDim, BR, BC>::COLS;
    float dot[ROWS][COLS];
    #pragma unroll
    for (int32_t r = 0; r < ROWS; r++)
        #pragma unroll
        for (int32_t c = 0; c < COLS; c++)
            dot[r][c] = 0.0f;

    if constexpr (ScoreThread<HeadDim, BR, BC>::VECTOR) {
        #pragma unroll 4
        for (int32_t feature = 0; feature < HeadDim; feature += 4) {
            float q[ROWS][4];
            #pragma unroll
            for (int32_t r = 0; r < ROWS; r++) {
                const float4 value = *reinterpret_cast<const float4*>(left + owned.row[r] * HeadDim + feature);
                q[r][0] = value.x;
                q[r][1] = value.y;
                q[r][2] = value.z;
                q[r][3] = value.w;
            }
            const float4 k0 = *reinterpret_cast<const float4*>(rightT + (feature + 0) * KT + owned.col);
            const float4 k1 = *reinterpret_cast<const float4*>(rightT + (feature + 1) * KT + owned.col);
            const float4 k2 = *reinterpret_cast<const float4*>(rightT + (feature + 2) * KT + owned.col);
            const float4 k3 = *reinterpret_cast<const float4*>(rightT + (feature + 3) * KT + owned.col);
            #pragma unroll
            for (int32_t r = 0; r < ROWS; r++) {
                dot[r][0] = fmaf(q[r][0], k0.x, dot[r][0]);
                dot[r][0] = fmaf(q[r][1], k1.x, dot[r][0]);
                dot[r][0] = fmaf(q[r][2], k2.x, dot[r][0]);
                dot[r][0] = fmaf(q[r][3], k3.x, dot[r][0]);
                dot[r][1] = fmaf(q[r][0], k0.y, dot[r][1]);
                dot[r][1] = fmaf(q[r][1], k1.y, dot[r][1]);
                dot[r][1] = fmaf(q[r][2], k2.y, dot[r][1]);
                dot[r][1] = fmaf(q[r][3], k3.y, dot[r][1]);
                dot[r][2] = fmaf(q[r][0], k0.z, dot[r][2]);
                dot[r][2] = fmaf(q[r][1], k1.z, dot[r][2]);
                dot[r][2] = fmaf(q[r][2], k2.z, dot[r][2]);
                dot[r][2] = fmaf(q[r][3], k3.z, dot[r][2]);
                dot[r][3] = fmaf(q[r][0], k0.w, dot[r][3]);
                dot[r][3] = fmaf(q[r][1], k1.w, dot[r][3]);
                dot[r][3] = fmaf(q[r][2], k2.w, dot[r][3]);
                dot[r][3] = fmaf(q[r][3], k3.w, dot[r][3]);
            }
        }
    } else {
        #pragma unroll 4
        for (int32_t feature = 0; feature < HeadDim; feature++) {
            const float right = rightT[feature * KT + owned.col];
            #pragma unroll
            for (int32_t r = 0; r < ROWS; r++)
                dot[r][0] = fmaf(left[owned.row[r] * HeadDim + feature], right, dot[r][0]);
        }
    }

    #pragma unroll
    for (int32_t r = 0; r < ROWS; r++) {
        const int32_t query_time = query_start + owned.row[r];
        const bool live = query_time < time;
        const float row_lse = live
            ? lse[position3(lse_layout, batch, head, query_time)]
            : 0.0f;
        #pragma unroll
        for (int32_t c = 0; c < COLS; c++) {
            const int32_t key_time = key_start + owned.col + c;
            float probability = 0.0f;
            if (live && key_time < time && (!diagonal || key_time <= query_time))
                probability = expf(scale * dot[r][c] - row_lse);
            scores[owned.row[r] * BC + owned.col + c] = probability;
        }
    }
}

// Overwrites probabilities with dS = P * (dO·V - delta). Same thread owns the same scores.
template <int32_t HeadDim, int32_t BR, int32_t BC, int32_t KT>
__device__ void gemm_score_gradient(
    const float* grad_out, const float* valueT, float* scores,
    const float* delta, int32_t query_start, int32_t key_start, int32_t time,
    int32_t batch, int32_t head, int32_t heads
) {
    ScoreThread<HeadDim, BR, BC> owned;
    constexpr int32_t ROWS = ScoreThread<HeadDim, BR, BC>::ROWS;
    constexpr int32_t COLS = ScoreThread<HeadDim, BR, BC>::COLS;
    float dot[ROWS][COLS];
    #pragma unroll
    for (int32_t r = 0; r < ROWS; r++)
        #pragma unroll
        for (int32_t c = 0; c < COLS; c++)
            dot[r][c] = 0.0f;

    if constexpr (ScoreThread<HeadDim, BR, BC>::VECTOR) {
        #pragma unroll 4
        for (int32_t feature = 0; feature < HeadDim; feature += 4) {
            float g[ROWS][4];
            #pragma unroll
            for (int32_t r = 0; r < ROWS; r++) {
                const float4 value = *reinterpret_cast<const float4*>(
                    grad_out + owned.row[r] * HeadDim + feature);
                g[r][0] = value.x;
                g[r][1] = value.y;
                g[r][2] = value.z;
                g[r][3] = value.w;
            }
            const float4 v0 = *reinterpret_cast<const float4*>(valueT + (feature + 0) * KT + owned.col);
            const float4 v1 = *reinterpret_cast<const float4*>(valueT + (feature + 1) * KT + owned.col);
            const float4 v2 = *reinterpret_cast<const float4*>(valueT + (feature + 2) * KT + owned.col);
            const float4 v3 = *reinterpret_cast<const float4*>(valueT + (feature + 3) * KT + owned.col);
            #pragma unroll
            for (int32_t r = 0; r < ROWS; r++) {
                dot[r][0] = fmaf(g[r][0], v0.x, dot[r][0]);
                dot[r][0] = fmaf(g[r][1], v1.x, dot[r][0]);
                dot[r][0] = fmaf(g[r][2], v2.x, dot[r][0]);
                dot[r][0] = fmaf(g[r][3], v3.x, dot[r][0]);
                dot[r][1] = fmaf(g[r][0], v0.y, dot[r][1]);
                dot[r][1] = fmaf(g[r][1], v1.y, dot[r][1]);
                dot[r][1] = fmaf(g[r][2], v2.y, dot[r][1]);
                dot[r][1] = fmaf(g[r][3], v3.y, dot[r][1]);
                dot[r][2] = fmaf(g[r][0], v0.z, dot[r][2]);
                dot[r][2] = fmaf(g[r][1], v1.z, dot[r][2]);
                dot[r][2] = fmaf(g[r][2], v2.z, dot[r][2]);
                dot[r][2] = fmaf(g[r][3], v3.z, dot[r][2]);
                dot[r][3] = fmaf(g[r][0], v0.w, dot[r][3]);
                dot[r][3] = fmaf(g[r][1], v1.w, dot[r][3]);
                dot[r][3] = fmaf(g[r][2], v2.w, dot[r][3]);
                dot[r][3] = fmaf(g[r][3], v3.w, dot[r][3]);
            }
        }
    } else {
        #pragma unroll 4
        for (int32_t feature = 0; feature < HeadDim; feature++) {
            const float right = valueT[feature * KT + owned.col];
            #pragma unroll
            for (int32_t r = 0; r < ROWS; r++)
                dot[r][0] = fmaf(grad_out[owned.row[r] * HeadDim + feature], right, dot[r][0]);
        }
    }

    #pragma unroll
    for (int32_t r = 0; r < ROWS; r++) {
        const int32_t query_time = query_start + owned.row[r];
        const float row_delta = query_time < time
            ? delta[(static_cast<long long>(batch) * heads + head) * time + query_time]
            : 0.0f;
        #pragma unroll
        for (int32_t c = 0; c < COLS; c++) {
            const int32_t index = owned.row[r] * BC + owned.col + c;
            scores[index] = scores[index] * (dot[r][c] - row_delta);
        }
    }
}

// dV[k, d] += P[q, k] * dO[q, d], and the same shape for dK with dS and Q.
// Each thread owns VEC consecutive features so the row vector is one float4.
template <int32_t HeadDim, int32_t BC>
struct KeyThread {
    static constexpr int32_t KEYS = Tile<HeadDim>::KEY_OUTPUTS / 4;
    static constexpr int32_t FEAT_THREADS = HeadDim / 4;

    int32_t feature;
    int32_t key[KEYS];

    __device__ KeyThread() {
        const int32_t tid = threadIdx.x;
        feature = (tid % FEAT_THREADS) * 4;
        const int32_t group = tid / FEAT_THREADS;
        #pragma unroll
        for (int32_t i = 0; i < KEYS; i++)
            key[i] = group + i * (BC / KEYS);
    }
};

template <int32_t HeadDim, int32_t BC, int32_t BR>
__device__ void accumulate_key_outer(
    float (&acc)[KeyThread<HeadDim, BC>::KEYS][4],
    const float* coeff, const float* rows, float coeff_scale,
    int32_t query_start, int32_t time
) {
    KeyThread<HeadDim, BC> owned;
    constexpr int32_t KEYS = KeyThread<HeadDim, BC>::KEYS;
    const int32_t live = query_start >= time ? 0
        : (query_start + BR <= time ? BR : time - query_start);
    for (int32_t row = 0; row < live; row++) {
        const float4 sample = *reinterpret_cast<const float4*>(rows + row * HeadDim + owned.feature);
        #pragma unroll
        for (int32_t i = 0; i < KEYS; i++) {
            const float weight = coeff_scale * coeff[row * BC + owned.key[i]];
            acc[i][0] = fmaf(weight, sample.x, acc[i][0]);
            acc[i][1] = fmaf(weight, sample.y, acc[i][1]);
            acc[i][2] = fmaf(weight, sample.z, acc[i][2]);
            acc[i][3] = fmaf(weight, sample.w, acc[i][3]);
        }
    }
}

__device__ bool vector4_store(const float* dst, const Layout& layout) {
    if (layout.stride[3] != 1 || (layout.stride[2] & 3) || (layout.stride[1] & 3) || (layout.stride[0] & 3))
        return false;
    return (reinterpret_cast<uintptr_t>(dst) & 0xf) == 0;
}

template <int32_t HeadDim>
__global__ void attention_delta_kernel(
    float* delta, const float* out, Layout out_layout, const float* grad_out, Layout grad_out_layout
) {
    const int32_t time = out_layout.shape[2];
    const int32_t heads = out_layout.shape[1];
    const int32_t rows = out_layout.shape[0] * heads * time;
    constexpr int32_t WARPS = THREADS / 32;
    const int32_t row = blockIdx.x * WARPS + (threadIdx.x >> 5);
    const int32_t lane = threadIdx.x & 31;
    if (row >= rows) return;

    const int32_t token = row % time;
    const int32_t head = (row / time) % heads;
    const int32_t batch = row / (time * heads);
    float sum = 0.0f;
    for (int32_t feature = lane; feature < HeadDim; feature += 32) {
        sum = fmaf(
            out[position4(out_layout, batch, head, token, feature)],
            grad_out[position4(grad_out_layout, batch, head, token, feature)],
            sum);
    }
    #pragma unroll
    for (int32_t offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, offset);
    if (lane == 0)
        delta[row] = sum;
}

template <int32_t HeadDim>
__global__ void attention_forward_kernel(
    float* out, Layout out_layout, float* lse, Layout lse_layout,
    const float* query, Layout query_layout,
    const float* key, Layout key_layout,
    const float* value, Layout value_layout,
    float scale, bool causal
) {
    constexpr int32_t BR = Tile<HeadDim>::BR;
    constexpr int32_t BC = Tile<HeadDim>::BC;
    constexpr int32_t KT = Tile<HeadDim>::KT;
    constexpr int32_t N = Tile<HeadDim>::OUTPUTS;
    constexpr int32_t SCORE_N = Tile<HeadDim>::SCORES;

    __shared__ __align__(16) float query_tile[BR * HeadDim];
    __shared__ __align__(16) float scores[BR * BC];
    __shared__ __align__(16) float pack[HeadDim * KT];
    __shared__ float row_maximum[BR];
    __shared__ float row_sum[BR];
    __shared__ float row_rescale[BR];

    const int32_t time = query_layout.shape[2];
    const int32_t heads = query_layout.shape[1];
    const int32_t query_tiles = (time + BR - 1) / BR;
    const int32_t query_tile_index = blockIdx.x % query_tiles;
    const int32_t head = (blockIdx.x / query_tiles) % heads;
    const int32_t batch = blockIdx.x / (query_tiles * heads);
    const int32_t query_start = query_tile_index * BR;
    const int32_t query_last = query_start + BR - 1;
    const int32_t tid = threadIdx.x;

    load_rows<HeadDim, BR>(query_tile, query, query_layout, batch, head, query_start, time);
    if (tid < BR) {
        row_maximum[tid] = -CUDART_INF_F;
        row_sum[tid] = 0.0f;
    }

    int32_t out_row[N];
    int32_t feature = 0;
    output_coords<HeadDim, BR, N>(out_row, feature);
    float acc[N];
    #pragma unroll
    for (int32_t i = 0; i < N; i++)
        acc[i] = 0.0f;
    __syncthreads();

    for (int32_t key_start = 0; key_start < time; key_start += BC) {
        if (causal && key_start > query_last)
            break;
        const bool diagonal = causal && (key_start + BC - 1 >= query_start);

        load_transposed<HeadDim, BC, KT>(pack, key, key_layout, batch, head, key_start, time);
        __syncthreads();
        gemm_dots<HeadDim, BR, BC, KT>(query_tile, pack, scores);

        const int32_t column = tid % BC;
        #pragma unroll
        for (int32_t i = 0; i < SCORE_N; i++) {
            const int32_t row = (tid + i * THREADS) / BC;
            const int32_t query_time = query_start + row;
            const int32_t key_time = key_start + column;
            float score = scores[row * BC + column] * scale;
            if (query_time >= time || key_time >= time || (diagonal && key_time > query_time))
                score = -CUDART_INF_F;
            scores[row * BC + column] = score;
        }
        __syncthreads();

        if (tid < BR) {
            const int32_t query_time = query_start + tid;
            float tile_maximum = -CUDART_INF_F;
            if (query_time < time) {
                for (int32_t key = 0; key < BC; key++)
                    tile_maximum = fmaxf(tile_maximum, scores[tid * BC + key]);
            }
            const float previous = row_maximum[tid];
            const float next = fmaxf(previous, tile_maximum);
            const float rescale = previous == -CUDART_INF_F ? 0.0f : expf(previous - next);
            row_maximum[tid] = next;
            row_rescale[tid] = rescale;

            float tile_sum = 0.0f;
            if (query_time < time) {
                for (int32_t key = 0; key < BC; key++) {
                    const float score = scores[tid * BC + key];
                    const float probability = score == -CUDART_INF_F ? 0.0f : expf(score - next);
                    scores[tid * BC + key] = probability;
                    tile_sum += probability;
                }
            }
            row_sum[tid] = row_sum[tid] * rescale + tile_sum;
        }
        __syncthreads();

        #pragma unroll
        for (int32_t i = 0; i < N; i++)
            acc[i] *= row_rescale[out_row[i]];

        // pack's transposed key is consumed. Reuse it as row-major V.
        load_row_major<HeadDim, BC>(pack, value, value_layout, batch, head, key_start, time);
        __syncthreads();
        accumulate_outer<HeadDim, BC, N>(acc, out_row, feature, scores, pack, 1.0f);
        __syncthreads();
    }

    #pragma unroll
    for (int32_t i = 0; i < N; i++) {
        const int32_t query_time = query_start + out_row[i];
        if (query_time < time) {
            const float denominator = row_sum[out_row[i]];
            out[position4(out_layout, batch, head, query_time, feature)] =
                denominator == 0.0f ? 0.0f : acc[i] / denominator;
        }
    }
    if (tid < BR) {
        const int32_t query_time = query_start + tid;
        if (query_time < time)
            lse[position3(lse_layout, batch, head, query_time)] =
                row_maximum[tid] + logf(row_sum[tid]);
    }
}

template <int32_t HeadDim>
__global__ void attention_backward_query_kernel(
    float* grad_query, Layout grad_query_layout,
    const float* grad_out, Layout grad_out_layout,
    const float* query, Layout query_layout,
    const float* key, Layout key_layout,
    const float* value, Layout value_layout,
    const float* lse, Layout lse_layout,
    const float* delta, float scale, bool causal
) {
    constexpr int32_t BR = Tile<HeadDim>::BR;
    constexpr int32_t BC = Tile<HeadDim>::BC;
    constexpr int32_t KT = Tile<HeadDim>::KT;
    constexpr int32_t N = Tile<HeadDim>::OUTPUTS;

    __shared__ __align__(16) float query_tile[BR * HeadDim];
    __shared__ __align__(16) float grad_out_tile[BR * HeadDim];
    __shared__ __align__(16) float scores[BR * BC];
    __shared__ __align__(16) float pack[HeadDim * KT];
    __shared__ __align__(16) float key_rows[BC * HeadDim];

    const int32_t time = query_layout.shape[2];
    const int32_t heads = query_layout.shape[1];
    const int32_t query_tiles = (time + BR - 1) / BR;
    const int32_t query_tile_index = blockIdx.x % query_tiles;
    const int32_t head = (blockIdx.x / query_tiles) % heads;
    const int32_t batch = blockIdx.x / (query_tiles * heads);
    const int32_t query_start = query_tile_index * BR;
    const int32_t query_last = query_start + BR - 1;

    load_rows<HeadDim, BR>(query_tile, query, query_layout, batch, head, query_start, time);
    load_rows<HeadDim, BR>(grad_out_tile, grad_out, grad_out_layout, batch, head, query_start, time);

    int32_t out_row[N];
    int32_t feature = 0;
    output_coords<HeadDim, BR, N>(out_row, feature);
    float acc[N];
    #pragma unroll
    for (int32_t i = 0; i < N; i++)
        acc[i] = 0.0f;
    __syncthreads();

    for (int32_t key_start = 0; key_start < time; key_start += BC) {
        if (causal && key_start > query_last)
            break;
        const bool diagonal = causal && (key_start + BC - 1 >= query_start);

        load_key_both<HeadDim, BC, KT>(
            key_rows, pack, key, key_layout, batch, head, key_start, time);
        __syncthreads();
        gemm_probabilities<HeadDim, BR, BC, KT>(
            query_tile, pack, scores, query_start, key_start, time, diagonal,
            lse, lse_layout, batch, head, scale);
        __syncthreads();

        load_transposed<HeadDim, BC, KT>(pack, value, value_layout, batch, head, key_start, time);
        __syncthreads();
        gemm_score_gradient<HeadDim, BR, BC, KT>(
            grad_out_tile, pack, scores, delta, query_start, key_start, time, batch, head, heads);
        __syncthreads();
        accumulate_outer<HeadDim, BC, N>(acc, out_row, feature, scores, key_rows, scale);
        __syncthreads();
    }

    #pragma unroll
    for (int32_t i = 0; i < N; i++) {
        const int32_t query_time = query_start + out_row[i];
        if (query_time < time)
            grad_query[position4(grad_query_layout, batch, head, query_time, feature)] = acc[i];
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
    const float* lse, Layout lse_layout,
    const float* delta, float scale, bool causal
) {
    constexpr int32_t BR = Tile<HeadDim>::BR;
    constexpr int32_t BC = Tile<HeadDim>::BC;
    constexpr int32_t KT = Tile<HeadDim>::KT;
    constexpr int32_t KEYS = KeyThread<HeadDim, BC>::KEYS;
    static_assert(Tile<HeadDim>::KEY_OUTPUTS % 4 == 0, "key outputs group into float4s");
    static_assert(THREADS % KeyThread<HeadDim, BC>::FEAT_THREADS == 0, "feature groups fill the block");
    static_assert((THREADS / KeyThread<HeadDim, BC>::FEAT_THREADS) * KEYS == BC, "key rows are covered once");

    __shared__ __align__(16) float key_tile[HeadDim * KT];
    __shared__ __align__(16) float value_tile[HeadDim * KT];
    __shared__ __align__(16) float query_tile[BR * HeadDim];
    __shared__ __align__(16) float grad_out_tile[BR * HeadDim];
    __shared__ __align__(16) float scores[BR * BC];

    const int32_t time = query_layout.shape[2];
    const int32_t heads = query_layout.shape[1];
    const int32_t key_tiles = (time + BC - 1) / BC;
    const int32_t key_tile_index = blockIdx.x % key_tiles;
    const int32_t head = (blockIdx.x / key_tiles) % heads;
    const int32_t batch = blockIdx.x / (key_tiles * heads);
    const int32_t key_start = key_tile_index * BC;

    load_transposed<HeadDim, BC, KT>(key_tile, key, key_layout, batch, head, key_start, time);
    load_transposed<HeadDim, BC, KT>(value_tile, value, value_layout, batch, head, key_start, time);

    KeyThread<HeadDim, BC> owned;
    float acc_key[KEYS][4];
    float acc_value[KEYS][4];
    #pragma unroll
    for (int32_t i = 0; i < KEYS; i++) {
        #pragma unroll
        for (int32_t lane = 0; lane < 4; lane++) {
            acc_key[i][lane] = 0.0f;
            acc_value[i][lane] = 0.0f;
        }
    }
    __syncthreads();

    const int32_t first_query = causal ? (key_start / BR) * BR : 0;
    for (int32_t query_start = first_query; query_start < time; query_start += BR) {
        const bool diagonal = causal && (query_start + BR - 1 >= key_start);

        load_rows<HeadDim, BR>(query_tile, query, query_layout, batch, head, query_start, time);
        load_rows<HeadDim, BR>(grad_out_tile, grad_out, grad_out_layout, batch, head, query_start, time);
        __syncthreads();

        gemm_probabilities<HeadDim, BR, BC, KT>(
            query_tile, key_tile, scores, query_start, key_start, time, diagonal,
            lse, lse_layout, batch, head, scale);
        __syncthreads();
        // dV consumes every probability. gemm_score_gradient then overwrites them with dS.
        accumulate_key_outer<HeadDim, BC, BR>(acc_value, scores, grad_out_tile, 1.0f, query_start, time);
        __syncthreads();
        gemm_score_gradient<HeadDim, BR, BC, KT>(
            grad_out_tile, value_tile, scores, delta, query_start, key_start, time, batch, head, heads);
        __syncthreads();
        accumulate_key_outer<HeadDim, BC, BR>(acc_key, scores, query_tile, scale, query_start, time);
        __syncthreads();
    }

    const bool vector_key = vector4_store(grad_key, grad_key_layout);
    const bool vector_value = vector4_store(grad_value, grad_value_layout);
    #pragma unroll
    for (int32_t i = 0; i < KEYS; i++) {
        const int32_t key_time = key_start + owned.key[i];
        if (key_time >= time) continue;
        const float4 key_grad = make_float4(acc_key[i][0], acc_key[i][1], acc_key[i][2], acc_key[i][3]);
        const float4 value_grad = make_float4(acc_value[i][0], acc_value[i][1], acc_value[i][2], acc_value[i][3]);
        if (vector_key) {
            *reinterpret_cast<float4*>(
                grad_key + position4(grad_key_layout, batch, head, key_time, owned.feature)) = key_grad;
        } else {
            #pragma unroll
            for (int32_t lane = 0; lane < 4; lane++)
                grad_key[position4(grad_key_layout, batch, head, key_time, owned.feature + lane)] = acc_key[i][lane];
        }
        if (vector_value) {
            *reinterpret_cast<float4*>(
                grad_value + position4(grad_value_layout, batch, head, key_time, owned.feature)) = value_grad;
        } else {
            #pragma unroll
            for (int32_t lane = 0; lane < 4; lane++)
                grad_value[position4(grad_value_layout, batch, head, key_time, owned.feature + lane)] =
                    acc_value[i][lane];
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
    constexpr int32_t BR = Tile<HeadDim>::BR;
    const int32_t tiles = (query.shape[2] + BR - 1) / BR;
    const int32_t blocks = query.shape[0] * query.shape[1] * tiles;
    if (blocks <= 0) return;
    Context& ctx = Context::get();
    attention_forward_kernel<HeadDim><<<blocks, THREADS, 0,
        static_cast<cudaStream_t>(ctx.getStream())>>>(
        out.data, layout_from(out), lse.data, layout_from(lse),
        query.data, layout_from(query), key.data, layout_from(key),
        value.data, layout_from(value), scale, causal);
    ctx.checkLastError();
}

float* delta_workspace(size_t rows) {
    struct Buffer {
        float* data = nullptr;
        size_t rows = 0;
    };
    static Buffer buffer;
    if (buffer.rows < rows) {
        if (buffer.data) cudaFree(buffer.data);
        buffer.data = nullptr;
        buffer.rows = 0;
        cudaError_t error = cudaMalloc(&buffer.data, rows * sizeof(float));
        if (error != cudaSuccess)
            throw std::runtime_error("CUDA attention_backward could not allocate the row delta");
        buffer.rows = rows;
    }
    return buffer.data;
}

template <int32_t HeadDim>
void launch_backward(
    const View& grad_query, const View& grad_key, const View& grad_value,
    const View& grad_out, const View& query, const View& key, const View& value,
    const View& out, const View& lse, float scale, bool causal
) {
    constexpr int32_t BR = Tile<HeadDim>::BR;
    constexpr int32_t BC = Tile<HeadDim>::BC;
    const int32_t batches = query.shape[0];
    const int32_t heads = query.shape[1];
    const int32_t time = query.shape[2];
    const int32_t query_blocks = batches * heads * ((time + BR - 1) / BR);
    const int32_t key_value_blocks = batches * heads * ((time + BC - 1) / BC);
    const int32_t rows = batches * heads * time;
    if (rows <= 0) return;

    Context& ctx = Context::get();
    cudaStream_t stream = static_cast<cudaStream_t>(ctx.getStream());
    float* delta = delta_workspace(static_cast<size_t>(rows));
    const int32_t delta_blocks = (rows + (THREADS / 32) - 1) / (THREADS / 32);
    attention_delta_kernel<HeadDim><<<delta_blocks, THREADS, 0, stream>>>(
        delta, out.data, layout_from(out), grad_out.data, layout_from(grad_out));
    ctx.checkLastError();

    if (query_blocks > 0) {
        attention_backward_query_kernel<HeadDim><<<query_blocks, THREADS, 0, stream>>>(
            grad_query.data, layout_from(grad_query),
            grad_out.data, layout_from(grad_out),
            query.data, layout_from(query), key.data, layout_from(key),
            value.data, layout_from(value),
            lse.data, layout_from(lse), delta, scale, causal);
        ctx.checkLastError();
    }
    if (key_value_blocks > 0) {
        attention_backward_key_value_kernel<HeadDim><<<key_value_blocks, THREADS, 0, stream>>>(
            grad_key.data, layout_from(grad_key),
            grad_value.data, layout_from(grad_value),
            grad_out.data, layout_from(grad_out),
            query.data, layout_from(query), key.data, layout_from(key),
            value.data, layout_from(value),
            lse.data, layout_from(lse), delta, scale, causal);
        ctx.checkLastError();
    }
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
