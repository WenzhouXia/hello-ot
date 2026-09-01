#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace {

// CN: 方案D 两套 kernel：
//   A. d <= 32：block 对应一个 query 行，thread-per-column，query 行放 shared
//      memory（采用固定容量的局部 top-k 归并结构）。
//   B. d > 32：block 处理一组 R 个 query 行（每 warp 一行），db 按列分块进
//      shared memory 复用，每 warp 在列分片上按维度分片计算距离并维护该行 top-k。
// 两者都在 kernel 内就地计算 l1/linf/l2 距离，并减去对偶。
// EN: Plan D uses two kernels:
//   A. d <= 32: one block per query row, thread-per-column, query row in shared
//      memory (using a fixed-capacity local top-k merge structure).
//   B. d > 32: one block per group of R query rows (one warp per row); db columns
//      are tiled into shared memory for reuse, each warp computes distances over
//      lane-sliced dimensions and maintains the row's top-k.
// Both compute l1/linf/l2 distance on the fly and subtract the database dual
// inside the kernel.
constexpr int kSmallThreads = 128;
constexpr int kSmallThreadsK32 = 64;
constexpr int kLargeThreads = 256;
constexpr int kLargeRows = 8;
constexpr int64_t kThreadPerColMaxD = 32;

enum CostKind { kL1 = 0, kLinf = 1, kL2 = 2 };

__device__ __forceinline__ void atomic_max_nonnegative_double(double* address, double value) {
    auto* raw = reinterpret_cast<unsigned long long*>(address);
    unsigned long long old = *raw;
    while (__longlong_as_double(old) < value) {
        const unsigned long long assumed = old;
        old = atomicCAS(raw, assumed, __double_as_longlong(value));
        if (old == assumed) {
            break;
        }
    }
}

template <typename ValueT, typename IndexT>
__device__ __forceinline__ bool is_better_pair(ValueT value, IndexT idx, ValueT best, IndexT best_idx) {
    return value > best || (value == best && idx >= 0 && (best_idx < 0 || idx < best_idx));
}

template <int K, typename ValueT, typename IndexT>
__device__ __forceinline__ void init_topk(ValueT (&vals)[K], IndexT (&idxs)[K]) {
#pragma unroll
    for (int rank = 0; rank < K; ++rank) {
        vals[rank] = -std::numeric_limits<ValueT>::infinity();
        idxs[rank] = static_cast<IndexT>(-1);
    }
}

template <int K, typename ValueT, typename IndexT>
__device__ __forceinline__ void insert_topk(ValueT (&vals)[K], IndexT (&idxs)[K], ValueT value, IndexT idx) {
    if (!is_better_pair(value, idx, vals[K - 1], idxs[K - 1])) {
        return;
    }
    int pos = K - 1;
#pragma unroll
    for (int rank = 0; rank < K; ++rank) {
        if (is_better_pair(value, idx, vals[rank], idxs[rank])) {
            pos = rank;
            break;
        }
    }
#pragma unroll
    for (int rank = K - 1; rank > 0; --rank) {
        if (rank > pos) {
            vals[rank] = vals[rank - 1];
            idxs[rank] = idxs[rank - 1];
        }
    }
    vals[pos] = value;
    idxs[pos] = idx;
}

// CN: 单个维度对距离的贡献；l2 只累加平方和，开方在归约之后做。
// EN: Per-dimension distance contribution; l2 accumulates squared sums and takes
//     the square root after the reduction.
template <int COST>
__device__ __forceinline__ float accumulate_dim(float acc, float a, float b) {
    if (COST == kL1) {
        float v = a - b;
        return acc + fabsf(v);
    } else if (COST == kLinf) {
        float v = a - b;
        return fmaxf(acc, fabsf(v));
    } else {  // kL2
        float v = a - b;
        return acc + v * v;
    }
}

template <int COST>
__device__ __forceinline__ float warp_reduce(float acc) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float other = __shfl_down_sync(0xffffffffu, acc, off);
        acc = (COST == kLinf) ? fmaxf(acc, other) : (acc + other);
    }
    return acc;
}

// CN: 把 4 路部分累加合并成一个（l1/l2 求和，linf 取最大）。
// EN: Merge four partial accumulators (sum for l1/l2, max for linf).
template <int COST>
__device__ __forceinline__ float merge_partials(float a, float b, float c, float d) {
    if (COST == kLinf) {
        return fmaxf(fmaxf(a, b), fmaxf(c, d));
    }
    return (a + b) + (c + d);
}

// CN: 大 d 路径的单个 (row, col) 距离：lane 按维度分片，warp shuffle 归约；
//     l2 在归约后开方。
// EN: Large-d per-(row, col) distance: lanes slice dimensions, warp-shuffle
//     reduction, l2 takes sqrt after the reduction.
template <int COST>
__device__ __forceinline__ float warp_column_dist(const float* drow, const float* qrow, int64_t d, int lane) {
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    int64_t k = lane;
    for (; k + 96 < d; k += 128) {
        acc0 = accumulate_dim<COST>(acc0, drow[k], qrow[k]);
        acc1 = accumulate_dim<COST>(acc1, drow[k + 32], qrow[k + 32]);
        acc2 = accumulate_dim<COST>(acc2, drow[k + 64], qrow[k + 64]);
        acc3 = accumulate_dim<COST>(acc3, drow[k + 96], qrow[k + 96]);
    }
    for (; k < d; k += 32) {
        acc0 = accumulate_dim<COST>(acc0, drow[k], qrow[k]);
    }
    float acc = merge_partials<COST>(acc0, acc1, acc2, acc3);
    acc = warp_reduce<COST>(acc);
    return (COST == kL2) ? sqrtf(acc) : acc;
}

// CN: 小 d 路径（d <= kThreadPerColMaxD）。query 行进 shared memory，线程按列
//     步进，最后 shared 树合并。无 shuffle、无 tile 同步。
//     rc = dual_db[col] - dist，取最大；输出取负得到 (cost - dual) 升序。
// EN: Small-d path (d <= kThreadPerColMaxD). Query row lives in shared memory;
//     threads stride over columns; a shared-memory tree merges per-thread top-k.
//     rc = dual_db[col] - dist is maximized; outputs are negated to obtain the
//     ascending (cost - dual) convention.
template <int K, int COST, int THREADS>
__global__ void fused_gcost_topk_thread_per_col_kernel(
        const float* __restrict__ query,
        const float* __restrict__ db,
        const double* __restrict__ dual_db,
        int64_t n_q,
        int64_t n_db,
        int64_t d,
        double* __restrict__ out_val,
        int64_t* __restrict__ out_idx) {
    const int64_t row = blockIdx.x;
    if (row >= n_q) {
        return;
    }

    extern __shared__ float s_query[];
    for (int64_t k = threadIdx.x; k < d; k += blockDim.x) {
        s_query[k] = query[row * d + k];
    }
    __syncthreads();

    double local_vals[K];
    int local_idxs[K];
    init_topk<K>(local_vals, local_idxs);

    for (int64_t col = threadIdx.x; col < n_db; col += blockDim.x) {
        const float* db_row = db + col * d;
        float acc = 0.0f;
        for (int64_t k = 0; k < d; ++k) {
            acc = accumulate_dim<COST>(acc, db_row[k], s_query[k]);
        }
        double rc = dual_db[col] - static_cast<double>((COST == kL2) ? sqrtf(acc) : acc);
        insert_topk<K>(local_vals, local_idxs, rc, static_cast<int>(col));
    }

    __shared__ double s_vals[THREADS][K];
    __shared__ int s_idxs[THREADS][K];
#pragma unroll
    for (int rank = 0; rank < K; ++rank) {
        s_vals[threadIdx.x][rank] = local_vals[rank];
        s_idxs[threadIdx.x][rank] = local_idxs[rank];
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
#pragma unroll
            for (int rank = 0; rank < K; ++rank) {
                insert_topk<K>(
                        local_vals,
                        local_idxs,
                        s_vals[threadIdx.x + stride][rank],
                        s_idxs[threadIdx.x + stride][rank]);
            }
#pragma unroll
            for (int rank = 0; rank < K; ++rank) {
                s_vals[threadIdx.x][rank] = local_vals[rank];
                s_idxs[threadIdx.x][rank] = local_idxs[rank];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const int64_t base = row * K;
#pragma unroll
        for (int rank = 0; rank < K; ++rank) {
            out_val[base + rank] = -local_vals[rank];
            out_idx[base + rank] = static_cast<int64_t>(local_idxs[rank]);
        }
    }
}

// CN: 大 d 路径（d > 32）。block = R 个 query 行（warp w 负责第 w 行），db 按
//     db_tile 列分块进 shared 复用；每 warp 对列分片内每列按维度分片求距离。
// EN: Large-d path (d > 32). A block owns R query rows (warp w handles row w);
//     db columns are tiled into shared memory for reuse across rows; each warp
//     computes per-column distances with lane-sliced dimension loops.
template <int K, int COST>
__global__ void fused_gcost_topk_large_d_kernel(
        const float* __restrict__ query,
        const float* __restrict__ db,
        const double* __restrict__ dual_db,
        int64_t n_q,
        int64_t n_db,
        int64_t d,
        int64_t rows_per_block,
        int64_t db_tile,
        double* __restrict__ out_val,
        int64_t* __restrict__ out_idx) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int64_t row = static_cast<int64_t>(blockIdx.x) * rows_per_block + warp;
    const bool row_active = warp < rows_per_block && row < n_q;

    extern __shared__ float smem[];
    float* s_q = smem;
    float* s_db = smem + rows_per_block * d;

    // CN: 协作载入本 block 的 R 行 query；所有 warp 都参与，行越界时跳过写入。
    // EN: Cooperative load of the block's R query rows; all warps participate even
    //     when trailing rows are out of range.
    {
        const int64_t rr0 = static_cast<int64_t>(blockIdx.x) * rows_per_block;
        for (int64_t rr = threadIdx.x >> 5; rr < rows_per_block; rr += blockDim.x >> 5) {
            const int64_t global_row = rr0 + rr;
            if (global_row >= n_q) {
                continue;
            }
            const float* q_src = query + global_row * d;
            for (int64_t kk = threadIdx.x & 31; kk < d; kk += 32) {
                s_q[rr * d + kk] = q_src[kk];
            }
        }
    }

    double local_vals[K];
    int local_idxs[K];
    init_topk<K>(local_vals, local_idxs);

    const float* qrow = s_q + warp * d;
    const int64_t n_tiles = (n_db + db_tile - 1) / db_tile;
    for (int64_t tile = 0; tile < n_tiles; ++tile) {
        const int64_t c0 = tile * db_tile;
        const int64_t c_count = min(db_tile, n_db - c0);
        for (int64_t cc = threadIdx.x >> 5; cc < c_count; cc += blockDim.x >> 5) {
            const float* db_src = db + (c0 + cc) * d;
            for (int64_t kk = threadIdx.x & 31; kk < d; kk += 32) {
                s_db[cc * d + kk] = db_src[kk];
            }
        }
        __syncthreads();
        if (row_active) {
            for (int64_t c = 0; c < c_count; ++c) {
                const float dist = warp_column_dist<COST>(s_db + c * d, qrow, d, lane);
                if (lane == 0) {
                    double rc = dual_db[c0 + c] - static_cast<double>(dist);
                    insert_topk<K>(local_vals, local_idxs, rc, static_cast<int>(c0 + c));
                }
            }
        }
        __syncthreads();
    }

    if (row_active && lane == 0) {
        const int64_t base = row * K;
#pragma unroll
        for (int rank = 0; rank < K; ++rank) {
            out_val[base + rank] = -local_vals[rank];
            out_idx[base + rank] = static_cast<int64_t>(local_idxs[rank]);
        }
    }
}

// CN: 一次 pairwise traversal 同时产生双向 dual-violation top-k 与 L2/Linf certificate。
// EN: One pairwise traversal producing bidirectional dual-violation top-k and L2/Linf certificates.
template <int K, int COST>
__global__ void fused_gcost_bidir_certificate_kernel(
        const float* __restrict__ query,
        const float* __restrict__ db,
        const double* __restrict__ query_dual,
        const double* __restrict__ db_dual,
        int64_t n_q,
        int64_t n_db,
        int64_t d,
        int64_t rows_per_block,
        int64_t db_tile,
        double* __restrict__ row_values,
        int64_t* __restrict__ row_indices,
        double* __restrict__ col_partial_values,
        int64_t* __restrict__ col_partial_indices,
        double* __restrict__ out_num,
        double* __restrict__ out_den,
        double* __restrict__ out_max_violation,
        double* __restrict__ out_cost_linf,
        int64_t* __restrict__ out_positive_count) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int64_t row_start = static_cast<int64_t>(blockIdx.x) * rows_per_block;
    const int64_t row = row_start + warp;
    const bool row_active = warp < rows_per_block && row < n_q;

    extern __shared__ float smem[];
    float* s_q = smem;
    float* s_db = smem + rows_per_block * d;
    __shared__ double s_scores[kLargeRows][64];

    for (int64_t rr = threadIdx.x >> 5; rr < rows_per_block; rr += blockDim.x >> 5) {
        const int64_t global_row = row_start + rr;
        if (global_row < n_q) {
            const float* q_src = query + global_row * d;
            for (int64_t kk = threadIdx.x & 31; kk < d; kk += 32) {
                s_q[rr * d + kk] = q_src[kk];
            }
        }
    }

    double local_values[K];
    int local_indices[K];
    init_topk<K>(local_values, local_indices);
    double acc_num = 0.0;
    double acc_den = 0.0;
    double acc_max_violation = 0.0;
    double acc_cost_linf = 0.0;
    unsigned long long acc_positive_count = 0;
    const float* qrow = s_q + warp * d;
    const double u_row = row_active ? query_dual[row] : 0.0;
    const int64_t n_tiles = (n_db + db_tile - 1) / db_tile;
    for (int64_t tile = 0; tile < n_tiles; ++tile) {
        const int64_t c0 = tile * db_tile;
        const int64_t c_count = min(db_tile, n_db - c0);
        for (int64_t cc = threadIdx.x >> 5; cc < c_count; cc += blockDim.x >> 5) {
            const float* db_src = db + (c0 + cc) * d;
            for (int64_t kk = threadIdx.x & 31; kk < d; kk += 32) {
                s_db[cc * d + kk] = db_src[kk];
            }
        }
        __syncthreads();
        if (row_active) {
            for (int64_t c = 0; c < c_count; ++c) {
                const float dist = warp_column_dist<COST>(s_db + c * d, qrow, d, lane);
                if (lane == 0) {
                    double cij = static_cast<double>(dist);
                    const double score = u_row + db_dual[c0 + c] - cij;
                    s_scores[warp][c] = score;
                    insert_topk<K>(local_values, local_indices, score, static_cast<int>(c0 + c));
                    acc_den += cij * cij;
                    acc_cost_linf = fmax(acc_cost_linf, fabs(cij));
                    if (score > 0.0) {
                        acc_num += score * score;
                        acc_max_violation = fmax(acc_max_violation, score);
                        ++acc_positive_count;
                    }
                }
            }
        } else if (lane == 0) {
            for (int64_t c = 0; c < c_count; ++c) {
                s_scores[warp][c] = -std::numeric_limits<double>::infinity();
            }
        }
        __syncthreads();

        for (int64_t c = threadIdx.x; c < c_count; c += blockDim.x) {
            double col_values[K];
            int col_indices[K];
            init_topk<K>(col_values, col_indices);
            for (int rr = 0; rr < rows_per_block; ++rr) {
                const int64_t global_row = row_start + rr;
                if (global_row < n_q) {
                    insert_topk<K>(col_values, col_indices, s_scores[rr][c], static_cast<int>(global_row));
                }
            }
            const int64_t base = (static_cast<int64_t>(blockIdx.x) * n_db + c0 + c) * K;
#pragma unroll
            for (int rank = 0; rank < K; ++rank) {
                col_partial_values[base + rank] = col_values[rank];
                col_partial_indices[base + rank] = static_cast<int64_t>(col_indices[rank]);
            }
        }
        __syncthreads();
    }

    if (row_active && lane == 0) {
        const int64_t base = row * K;
#pragma unroll
        for (int rank = 0; rank < K; ++rank) {
            row_values[base + rank] = local_values[rank];
            row_indices[base + rank] = static_cast<int64_t>(local_indices[rank]);
        }
        atomicAdd(out_num, acc_num);
        atomicAdd(out_den, acc_den);
        atomic_max_nonnegative_double(out_max_violation, acc_max_violation);
        atomic_max_nonnegative_double(out_cost_linf, acc_cost_linf);
        atomicAdd(reinterpret_cast<unsigned long long*>(out_positive_count), acc_positive_count);
    }
}

template <int K>
__global__ void reduce_gcost_col_partials_kernel(
        const double* __restrict__ partial_values,
        const int64_t* __restrict__ partial_indices,
        int64_t n_blocks,
        int64_t n_db,
        double* __restrict__ out_values,
        int64_t* __restrict__ out_indices) {
    const int64_t col = blockIdx.x;
    double local_values[K];
    int64_t local_indices[K];
    init_topk<K>(local_values, local_indices);
    for (int64_t block = threadIdx.x; block < n_blocks; block += blockDim.x) {
        const int64_t base = (block * n_db + col) * K;
#pragma unroll
        for (int rank = 0; rank < K; ++rank) {
            insert_topk<K>(local_values, local_indices, partial_values[base + rank], partial_indices[base + rank]);
        }
    }
    __shared__ double shared_values[64][K];
    __shared__ int64_t shared_indices[64][K];
#pragma unroll
    for (int rank = 0; rank < K; ++rank) {
        shared_values[threadIdx.x][rank] = local_values[rank];
        shared_indices[threadIdx.x][rank] = local_indices[rank];
    }
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
#pragma unroll
            for (int rank = 0; rank < K; ++rank) {
                insert_topk<K>(
                        local_values,
                        local_indices,
                        shared_values[threadIdx.x + stride][rank],
                        shared_indices[threadIdx.x + stride][rank]);
            }
#pragma unroll
            for (int rank = 0; rank < K; ++rank) {
                shared_values[threadIdx.x][rank] = local_values[rank];
                shared_indices[threadIdx.x][rank] = local_indices[rank];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const int64_t base = col * K;
#pragma unroll
        for (int rank = 0; rank < K; ++rank) {
            out_values[base + rank] = local_values[rank];
            out_indices[base + rank] = local_indices[rank];
        }
    }
}

// CN: 证书归约 kernel：一次全矩阵遍历同时计算
//     numerator = sum relu(u_i + v_j - c_ij)^2 与
//     denominator = sum c_ij^2。
//     结构与大 d top-k 相同（warp 按 query 行、db 列分块进 shared），每个 warp
//     用 double 累加，block 归约后 atomicAdd 到全局 fp64。
// EN: Certificate reduction kernel: one full-matrix pass computes
//     numerator = sum relu(u_i + v_j - c_ij)^2 and
//     denominator = sum c_ij^2.
//     Same structure as the large-d top-k kernel; each warp accumulates in double
//     and block-reduced partials are atomicAdded into global fp64.
template <int COST>
__global__ void fused_gcost_certificate_kernel(
        const float* __restrict__ query,
        const float* __restrict__ db,
        const double* __restrict__ query_dual,
        const double* __restrict__ db_dual,
        int64_t n_q,
        int64_t n_db,
        int64_t d,
        int64_t rows_per_block,
        int64_t db_tile,
        bool compute_denominator,
        double* __restrict__ out_num,
        double* __restrict__ out_den,
        double* __restrict__ out_max_violation,
        double* __restrict__ out_cost_linf,
        int64_t* __restrict__ out_positive_count) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int64_t row = static_cast<int64_t>(blockIdx.x) * rows_per_block + warp;
    const bool row_active = warp < rows_per_block && row < n_q;

    extern __shared__ float smem[];
    float* s_q = smem;
    float* s_db = smem + rows_per_block * d;

    {
        const int64_t rr0 = static_cast<int64_t>(blockIdx.x) * rows_per_block;
        for (int64_t rr = threadIdx.x >> 5; rr < rows_per_block; rr += blockDim.x >> 5) {
            const int64_t global_row = rr0 + rr;
            if (global_row >= n_q) {
                continue;
            }
            const float* q_src = query + global_row * d;
            for (int64_t kk = threadIdx.x & 31; kk < d; kk += 32) {
                s_q[rr * d + kk] = q_src[kk];
            }
        }
    }

    const float* qrow = s_q + warp * d;
    const double u_row = row_active ? query_dual[row] : 0.0;
    double acc_num = 0.0;
    double acc_den = 0.0;
    double acc_max_violation = 0.0;
    double acc_cost_linf = 0.0;
    unsigned long long acc_positive_count = 0;
    const int64_t n_tiles = (n_db + db_tile - 1) / db_tile;
    for (int64_t tile = 0; tile < n_tiles; ++tile) {
        const int64_t c0 = tile * db_tile;
        const int64_t c_count = min(db_tile, n_db - c0);
        for (int64_t cc = threadIdx.x >> 5; cc < c_count; cc += blockDim.x >> 5) {
            const float* db_src = db + (c0 + cc) * d;
            for (int64_t kk = threadIdx.x & 31; kk < d; kk += 32) {
                s_db[cc * d + kk] = db_src[kk];
            }
        }
        __syncthreads();
        if (row_active) {
            for (int64_t c = 0; c < c_count; ++c) {
                const float dist = warp_column_dist<COST>(s_db + c * d, qrow, d, lane);
                if (lane == 0) {
                    double cij = static_cast<double>(dist);
                    const double viol = u_row + db_dual[c0 + c] - cij;
                    if (viol > 0.0) {
                        acc_num += viol * viol;
                        acc_max_violation = fmax(acc_max_violation, viol);
                        ++acc_positive_count;
                    }
                    acc_cost_linf = fmax(acc_cost_linf, fabs(cij));
                    if (compute_denominator) {
                        acc_den += cij * cij;
                    }
                }
            }
        }
        __syncthreads();
    }

    __shared__ double s_part[kLargeRows][2];
    __shared__ double s_max_part[kLargeRows][2];
    __shared__ unsigned long long s_count_part[kLargeRows];
    if (lane == 0) {
        s_part[warp][0] = acc_num;
        s_part[warp][1] = acc_den;
        s_max_part[warp][0] = acc_max_violation;
        s_max_part[warp][1] = acc_cost_linf;
        s_count_part[warp] = acc_positive_count;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        double num = 0.0;
        double den = 0.0;
        double max_violation = 0.0;
        double cost_linf = 0.0;
        unsigned long long positive_count = 0;
        for (int w = 0; w < kLargeRows; ++w) {
            num += s_part[w][0];
            den += s_part[w][1];
            max_violation = fmax(max_violation, s_max_part[w][0]);
            cost_linf = fmax(cost_linf, s_max_part[w][1]);
            positive_count += s_count_part[w];
        }
        atomicAdd(out_num, num);
        if (compute_denominator) {
            atomicAdd(out_den, den);
        }
        atomic_max_nonnegative_double(out_max_violation, max_violation);
        atomic_max_nonnegative_double(out_cost_linf, cost_linf);
        atomicAdd(reinterpret_cast<unsigned long long*>(out_positive_count), positive_count);
    }
}

void check_cuda_inputs(
        torch::Tensor query,
        torch::Tensor db,
        torch::Tensor dual_db,
        int64_t k,
        int64_t cost_type) {
    TORCH_CHECK(query.is_cuda(), "query must be a CUDA tensor");
    TORCH_CHECK(db.is_cuda(), "db must be a CUDA tensor");
    TORCH_CHECK(dual_db.is_cuda(), "dual_db must be a CUDA tensor");
    TORCH_CHECK(query.scalar_type() == torch::kFloat32, "query must be float32");
    TORCH_CHECK(db.scalar_type() == torch::kFloat32, "db must be float32");
    TORCH_CHECK(dual_db.scalar_type() == torch::kFloat64, "dual_db must be float64");
    TORCH_CHECK(query.is_contiguous(), "query must be contiguous");
    TORCH_CHECK(db.is_contiguous(), "db must be contiguous");
    TORCH_CHECK(dual_db.is_contiguous(), "dual_db must be contiguous");
    TORCH_CHECK(query.dim() == 2, "query must be 2D");
    TORCH_CHECK(db.dim() == 2, "db must be 2D");
    TORCH_CHECK(query.size(1) == db.size(1), "query/db dims must match");
    TORCH_CHECK(dual_db.numel() == db.size(0), "dual_db length must equal db rows");
    TORCH_CHECK(k == 1 || k == 2 || k == 4 || k == 8 || k == 16 || k == 32, "k must be one of {1, 2, 4, 8, 16, 32}");
    TORCH_CHECK(cost_type == kL1 || cost_type == kLinf || cost_type == kL2, "cost_type must be 0 (l1), 1 (linf), 2 (l2)");
    TORCH_CHECK(query.numel() > 0 && db.numel() > 0, "inputs must be non-empty");
}

void check_certificate_inputs(
        torch::Tensor query,
        torch::Tensor db,
        torch::Tensor query_dual,
        torch::Tensor db_dual,
        int64_t cost_type) {
    TORCH_CHECK(query.is_cuda() && db.is_cuda() && query_dual.is_cuda() && db_dual.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(query.scalar_type() == torch::kFloat32 && db.scalar_type() == torch::kFloat32, "query/db must be float32");
    TORCH_CHECK(query_dual.scalar_type() == torch::kFloat64 && db_dual.scalar_type() == torch::kFloat64, "duals must be float64");
    TORCH_CHECK(query.is_contiguous() && db.is_contiguous() && query_dual.is_contiguous() && db_dual.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(query.dim() == 2 && db.dim() == 2, "query/db must be 2D");
    TORCH_CHECK(query.size(1) == db.size(1), "query/db dims must match");
    TORCH_CHECK(query_dual.numel() == query.size(0), "query_dual length must equal query rows");
    TORCH_CHECK(db_dual.numel() == db.size(0), "db_dual length must equal db rows");
    TORCH_CHECK(cost_type == kL1 || cost_type == kLinf || cost_type == kL2, "cost_type must be 0 (l1), 1 (linf), 2 (l2)");
    TORCH_CHECK(query.numel() > 0 && db.numel() > 0, "inputs must be non-empty");
}

template <int K, int COST>
void launch_thread_per_col(
        const float* query,
        const float* db,
        const double* dual_db,
        int64_t n_q,
        int64_t n_db,
        int64_t d,
        double* out_val,
        int64_t* out_idx,
        cudaStream_t stream) {
    const size_t shmem = static_cast<size_t>(d) * sizeof(float);
    constexpr int threads = K == 32 ? kSmallThreadsK32 : kSmallThreads;
    fused_gcost_topk_thread_per_col_kernel<K, COST, threads><<<static_cast<unsigned int>(n_q), threads, shmem, stream>>>(
            query, db, dual_db, n_q, n_db, d, out_val, out_idx);
}

template <int K, int COST>
void launch_large_d(
        const float* query,
        const float* db,
        const double* dual_db,
        int64_t n_q,
        int64_t n_db,
        int64_t d,
        int64_t rows_per_block,
        int64_t db_tile,
        double* out_val,
        int64_t* out_idx,
        cudaStream_t stream) {
    const int64_t grid = (n_q + rows_per_block - 1) / rows_per_block;
    const size_t shmem = static_cast<size_t>((rows_per_block + db_tile) * d) * sizeof(float);
    static int max_dyn_shared = 0;
    if (max_dyn_shared == 0) {
        int value = 0;
        int device = 0;
        const cudaError_t dev_err = cudaGetDevice(&device);
        TORCH_CHECK(dev_err == cudaSuccess, "cudaGetDevice failed: ", cudaGetErrorString(dev_err));
        const cudaError_t err = cudaDeviceGetAttribute(&value, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
        TORCH_CHECK(err == cudaSuccess, "cudaDeviceGetAttribute failed: ", cudaGetErrorString(err));
        max_dyn_shared = value;
    }
    TORCH_CHECK(shmem <= static_cast<size_t>(max_dyn_shared), "large-d kernel shared memory too large: ", shmem);
    if (shmem > 48 * 1024) {
        const cudaError_t attr_err = cudaFuncSetAttribute(
                fused_gcost_topk_large_d_kernel<K, COST>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(shmem));
        TORCH_CHECK(attr_err == cudaSuccess, "cudaFuncSetAttribute failed: ", cudaGetErrorString(attr_err));
    }
    fused_gcost_topk_large_d_kernel<K, COST><<<static_cast<unsigned int>(grid), kLargeThreads, shmem, stream>>>(
            query, db, dual_db, n_q, n_db, d, rows_per_block, db_tile, out_val, out_idx);
}

template <int K>
void dispatch_by_cost(
        int64_t cost_type,
        const float* query,
        const float* db,
        const double* dual_db,
        int64_t n_q,
        int64_t n_db,
        int64_t d,
        int64_t rows_per_block,
        int64_t db_tile,
        double* out_val,
        int64_t* out_idx,
        cudaStream_t stream) {
    const bool small_d = d <= kThreadPerColMaxD;
    if (cost_type == kL1) {
        if (small_d) {
            launch_thread_per_col<K, kL1>(query, db, dual_db, n_q, n_db, d, out_val, out_idx, stream);
        } else {
            launch_large_d<K, kL1>(query, db, dual_db, n_q, n_db, d, rows_per_block, db_tile, out_val, out_idx, stream);
        }
    } else if (cost_type == kLinf) {
        if (small_d) {
            launch_thread_per_col<K, kLinf>(query, db, dual_db, n_q, n_db, d, out_val, out_idx, stream);
        } else {
            launch_large_d<K, kLinf>(query, db, dual_db, n_q, n_db, d, rows_per_block, db_tile, out_val, out_idx, stream);
        }
    } else {
        if (small_d) {
            launch_thread_per_col<K, kL2>(query, db, dual_db, n_q, n_db, d, out_val, out_idx, stream);
        } else {
            launch_large_d<K, kL2>(query, db, dual_db, n_q, n_db, d, rows_per_block, db_tile, out_val, out_idx, stream);
        }
    }
}

template <int K>
void dispatch_topk(
        int64_t cost_type,
        const float* query,
        const float* db,
        const double* dual_db,
        int64_t n_q,
        int64_t n_db,
        int64_t d,
        int64_t rows_per_block,
        int64_t db_tile,
        double* out_val,
        int64_t* out_idx,
        cudaStream_t stream) {
    dispatch_by_cost<K>(cost_type, query, db, dual_db, n_q, n_db, d, rows_per_block, db_tile, out_val, out_idx, stream);
}

void launch_certificate(
        int64_t cost_type,
        const float* query,
        const float* db,
        const double* query_dual,
        const double* db_dual,
        int64_t n_q,
        int64_t n_db,
        int64_t d,
        int64_t rows_per_block,
        int64_t db_tile,
        bool compute_denominator,
        double* out_num,
        double* out_den,
        double* out_max_violation,
        double* out_cost_linf,
        int64_t* out_positive_count,
        cudaStream_t stream) {
    const int64_t grid = (n_q + rows_per_block - 1) / rows_per_block;
    const size_t shmem = static_cast<size_t>((rows_per_block + db_tile) * d) * sizeof(float);
    static int max_dyn_shared = 0;
    if (max_dyn_shared == 0) {
        int value = 0;
        int device = 0;
        const cudaError_t dev_err = cudaGetDevice(&device);
        TORCH_CHECK(dev_err == cudaSuccess, "cudaGetDevice failed: ", cudaGetErrorString(dev_err));
        const cudaError_t err = cudaDeviceGetAttribute(&value, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
        TORCH_CHECK(err == cudaSuccess, "cudaDeviceGetAttribute failed: ", cudaGetErrorString(err));
        max_dyn_shared = value;
    }
    TORCH_CHECK(shmem <= static_cast<size_t>(max_dyn_shared), "certificate kernel shared memory too large: ", shmem);
    if (shmem > 48 * 1024) {
        const cudaError_t attr_err = cudaFuncSetAttribute(
                fused_gcost_certificate_kernel<kL1>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(shmem));
        TORCH_CHECK(attr_err == cudaSuccess, "cudaFuncSetAttribute failed: ", cudaGetErrorString(attr_err));
        const cudaError_t attr_err2 = cudaFuncSetAttribute(
                fused_gcost_certificate_kernel<kLinf>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(shmem));
        TORCH_CHECK(attr_err2 == cudaSuccess, "cudaFuncSetAttribute failed: ", cudaGetErrorString(attr_err2));
        const cudaError_t attr_err3 = cudaFuncSetAttribute(
                fused_gcost_certificate_kernel<kL2>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(shmem));
        TORCH_CHECK(attr_err3 == cudaSuccess, "cudaFuncSetAttribute failed: ", cudaGetErrorString(attr_err3));
    }
    if (cost_type == kL1) {
        fused_gcost_certificate_kernel<kL1><<<static_cast<unsigned int>(grid), kLargeThreads, shmem, stream>>>(
                query, db, query_dual, db_dual, n_q, n_db, d, rows_per_block, db_tile, compute_denominator, out_num, out_den, out_max_violation, out_cost_linf, out_positive_count);
    } else if (cost_type == kLinf) {
        fused_gcost_certificate_kernel<kLinf><<<static_cast<unsigned int>(grid), kLargeThreads, shmem, stream>>>(
                query, db, query_dual, db_dual, n_q, n_db, d, rows_per_block, db_tile, compute_denominator, out_num, out_den, out_max_violation, out_cost_linf, out_positive_count);
    } else {
        fused_gcost_certificate_kernel<kL2><<<static_cast<unsigned int>(grid), kLargeThreads, shmem, stream>>>(
                query, db, query_dual, db_dual, n_q, n_db, d, rows_per_block, db_tile, compute_denominator, out_num, out_den, out_max_violation, out_cost_linf, out_positive_count);
    }
}

template <int K, int COST>
void launch_bidir_certificate(
        const float* query,
        const float* db,
        const double* query_dual,
        const double* db_dual,
        int64_t n_q,
        int64_t n_db,
        int64_t d,
        int64_t rows_per_block,
        int64_t db_tile,
        double* row_values,
        int64_t* row_indices,
        double* partial_values,
        int64_t* partial_indices,
        double* out_num,
        double* out_den,
        double* out_max_violation,
        double* out_cost_linf,
        int64_t* out_positive_count,
        cudaStream_t stream) {
    const int64_t n_blocks = (n_q + rows_per_block - 1) / rows_per_block;
    const size_t shmem = static_cast<size_t>((rows_per_block + db_tile) * d) * sizeof(float);
    static int max_dyn_shared = 0;
    if (max_dyn_shared == 0) {
        int value = 0;
        int device = 0;
        TORCH_CHECK(cudaGetDevice(&device) == cudaSuccess, "cudaGetDevice failed");
        TORCH_CHECK(
                cudaDeviceGetAttribute(&value, cudaDevAttrMaxSharedMemoryPerBlockOptin, device) == cudaSuccess,
                "cudaDeviceGetAttribute failed");
        max_dyn_shared = value;
    }
    TORCH_CHECK(shmem <= static_cast<size_t>(max_dyn_shared), "bidir norm-cost kernel shared memory too large: ", shmem);
    if (shmem > 48 * 1024) {
        const cudaError_t attr_err = cudaFuncSetAttribute(
                fused_gcost_bidir_certificate_kernel<K, COST>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(shmem));
        TORCH_CHECK(attr_err == cudaSuccess, "cudaFuncSetAttribute failed: ", cudaGetErrorString(attr_err));
    }
    fused_gcost_bidir_certificate_kernel<K, COST><<<static_cast<unsigned int>(n_blocks), kLargeThreads, shmem, stream>>>(
            query, db, query_dual, db_dual,
            n_q, n_db, d, rows_per_block, db_tile, row_values, row_indices,
            partial_values, partial_indices, out_num, out_den, out_max_violation,
            out_cost_linf, out_positive_count);
}

template <int K>
void dispatch_bidir_certificate(
        int64_t cost_type,
        const float* query,
        const float* db,
        const double* query_dual,
        const double* db_dual,
        int64_t n_q,
        int64_t n_db,
        int64_t d,
        int64_t rows_per_block,
        int64_t db_tile,
        double* row_values,
        int64_t* row_indices,
        double* partial_values,
        int64_t* partial_indices,
        double* out_num,
        double* out_den,
        double* out_max_violation,
        double* out_cost_linf,
        int64_t* out_positive_count,
        cudaStream_t stream) {
    if (cost_type == kL1) {
        launch_bidir_certificate<K, kL1>(query, db, query_dual, db_dual, n_q, n_db, d, rows_per_block, db_tile, row_values, row_indices, partial_values, partial_indices, out_num, out_den, out_max_violation, out_cost_linf, out_positive_count, stream);
    } else if (cost_type == kLinf) {
        launch_bidir_certificate<K, kLinf>(query, db, query_dual, db_dual, n_q, n_db, d, rows_per_block, db_tile, row_values, row_indices, partial_values, partial_indices, out_num, out_den, out_max_violation, out_cost_linf, out_positive_count, stream);
    } else {
        launch_bidir_certificate<K, kL2>(query, db, query_dual, db_dual, n_q, n_db, d, rows_per_block, db_tile, row_values, row_indices, partial_values, partial_indices, out_num, out_den, out_max_violation, out_cost_linf, out_positive_count, stream);
    }
}

template <int K>
void launch_reduce_gcost_col_partials(
        const double* partial_values,
        const int64_t* partial_indices,
        int64_t n_blocks,
        int64_t n_db,
        double* out_values,
        int64_t* out_indices,
        cudaStream_t stream) {
    reduce_gcost_col_partials_kernel<K><<<static_cast<unsigned int>(n_db), 64, 0, stream>>>(
            partial_values, partial_indices, n_blocks, n_db, out_values, out_indices);
}

// ============================================================================
// CN: MIPS top-k 融合 kernel（用于 dual-assignment augment）。
//     score(i, j) = dot(query[i], db[j])，即 augmented 向量内积（query 最后一维为
//     1、db 最后一维为 bias 时，等价于 feature 内积加 bias）。每行输出最大的 K 个
//     score 及其列号，降序、同值按列号升序，语义与 torch.topk(largest=True) 一致
//     （仅允许浮点相等时的并列列号交换）。
//     两个结构：
//       A. d <= 32：block 对应一个 query 行，thread-per-column，query 行进 shared，
//          每线程维护一个 int16 最小堆（存最大的 K 个），shared 树合并。
//       B. d > 32：block 对应 8 个 query 行（每 warp 一行），db 按列分块进 shared
//          复用，每 lane 维护局部 int32 最小堆，warp shuffle 合并。
//     k 仅支持 {1, 2, 4, 8, 16, 32}。
// EN: Fused MIPS top-k kernel for the dual-assignment augment.
//     score(i, j) = dot(query[i], db[j]) over the
//     augmented vectors (query's last dim is 1 and db's last dim is the bias, so
//     this equals feature dot plus bias). Outputs the K largest scores per row
//     with column ids, descending, ties by ascending column id, matching
//     torch.topk(largest=True) semantics (tolerating column swaps on equal values).
//     Two layouts:
//       A. d <= 32: one block per query row, thread-per-column, query row in
//          shared memory, per-thread int16 min-heap of the K largest, shared
//          tree merge.
//       B. d > 32: one block per 8 query rows (one warp per row), db tiled into
//          shared, per-lane int32 min-heap, warp-shuffle merge.
//     k is restricted to {1, 2, 4, 8, 16, 32}.
constexpr int kMipsThreads = 64;
constexpr int kMipsWarps = 8;
constexpr int kMipsThreadPerColMaxD = 32;

__device__ __forceinline__ bool mips_is_worse_f32(float a, int32_t ia, float b, int32_t ib) {
    return a < b || (a == b && ib >= 0 && (ia < 0 || ia > ib));
}

// CN: 最小堆存“已见最大的 K 个”；堆顶是第 K 大（最差者）。比较用 is_worse：
//     数值小或（数值相等时）索引大者更差。
// EN: Min-heap of the K largest seen; root is the K-th best (worst kept).
//     is_worse: smaller value, or equal value with larger index, is worse.
template <typename IndexT, int K>
struct MipsHeapTopK {
    float v[K];
    IndexT i[K];
    __device__ __forceinline__ void init() {
#pragma unroll
        for (int r = 0; r < K; ++r) {
            v[r] = -3.4028234663852886e38F;
            i[r] = static_cast<IndexT>(-1);
        }
    }
    __device__ __forceinline__ void push(float value, IndexT idx) {
        if (mips_is_worse_f32(value, static_cast<int32_t>(idx), v[0], static_cast<int32_t>(i[0]))) {
            return;
        }
        int p = 0;
        while (true) {
            const int l = 2 * p + 1;
            const int r = 2 * p + 2;
            if (l >= K) {
                break;
            }
            int worse_child = l;
            if (r < K && mips_is_worse_f32(v[r], static_cast<int32_t>(i[r]), v[l], static_cast<int32_t>(i[l]))) {
                worse_child = r;
            }
            if (mips_is_worse_f32(value, static_cast<int32_t>(idx), v[worse_child], static_cast<int32_t>(i[worse_child]))) {
                break;
            }
            v[p] = v[worse_child];
            i[p] = i[worse_child];
            p = worse_child;
        }
        v[p] = value;
        i[p] = idx;
    }
    __device__ __forceinline__ void sort_desc(float* ov, int64_t* oi) {
        // CN: 堆数组无序；用插入排序按 is_better 降序输出。
        // EN: Heap array is unsorted; insertion-sort by is_better descending.
        float tv[K];
        IndexT ti[K];
#pragma unroll
        for (int r = 0; r < K; ++r) {
            tv[r] = v[r];
            ti[r] = i[r];
        }
#pragma unroll
        for (int a = 1; a < K; ++a) {
            float x = tv[a];
            IndexT xi = ti[a];
            int b = a - 1;
            while (b >= 0 && mips_is_worse_f32(tv[b], static_cast<int32_t>(ti[b]), x, static_cast<int32_t>(xi))) {
                tv[b + 1] = tv[b];
                ti[b + 1] = ti[b];
                --b;
            }
            tv[b + 1] = x;
            ti[b + 1] = xi;
        }
#pragma unroll
        for (int r = 0; r < K; ++r) {
            ov[r] = tv[r];
            oi[r] = static_cast<int64_t>(ti[r]);
        }
    }
    __device__ __forceinline__ float val_at(int r) const { return v[r]; }
    __device__ __forceinline__ IndexT idx_at(int r) const { return i[r]; }
};

// CN: 小 d 路径（d <= kMipsThreadPerColMaxD）。query 行进 shared memory，
//     线程按列步进，每线程维护 int16 最小堆（列号 < 65536 时安全）。
// EN: Small-d path (d <= kMipsThreadPerColMaxD). Query row in shared memory;
//     threads stride over columns with a per-thread int16 min-heap (safe while
//     the column count is below 65536).
template <int K>
__global__ void fused_mips_topk_thread_per_col_kernel(
        const float* __restrict__ query,   // (n_q, d)
        const float* __restrict__ db,      // (n_db, d)
        int64_t n_q,
        int64_t n_db,
        int64_t d,
        float* __restrict__ out_val,       // (n_q, K)
        int64_t* __restrict__ out_idx) {   // (n_q, K)
    const int64_t row = blockIdx.x;
    if (row >= n_q) {
        return;
    }

    extern __shared__ float s_query[];
    for (int64_t k = threadIdx.x; k < d; k += blockDim.x) {
        s_query[k] = query[row * d + k];
    }
    __syncthreads();

    MipsHeapTopK<unsigned short, K> local;
    local.init();

    for (int64_t col = threadIdx.x; col < n_db; col += blockDim.x) {
        const float* db_row = db + col * d;
        float acc = 0.0f;
        for (int64_t k = 0; k < d; ++k) {
            acc += db_row[k] * s_query[k];
        }
        local.push(acc, static_cast<unsigned short>(col));
    }

    __shared__ float s_vals[kMipsThreads][K];
    __shared__ unsigned short s_idxs[kMipsThreads][K];
#pragma unroll
    for (int r = 0; r < K; ++r) {
        s_vals[threadIdx.x][r] = local.val_at(r);
        s_idxs[threadIdx.x][r] = local.idx_at(r);
    }
    __syncthreads();
    for (int stride = kMipsThreads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
#pragma unroll
            for (int r = 0; r < K; ++r) {
                local.push(s_vals[threadIdx.x + stride][r], s_idxs[threadIdx.x + stride][r]);
            }
#pragma unroll
            for (int r = 0; r < K; ++r) {
                s_vals[threadIdx.x][r] = local.val_at(r);
                s_idxs[threadIdx.x][r] = local.idx_at(r);
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float ov[K];
        int64_t oi[K];
        local.sort_desc(ov, oi);
        const int64_t base = row * K;
#pragma unroll
        for (int r = 0; r < K; ++r) {
            out_val[base + r] = ov[r];
            out_idx[base + r] = oi[r];
        }
    }
}

// CN: 大 d 路径（d > kMipsThreadPerColMaxD）。block = kMipsWarps 个 warp（每 warp
//     一行），db 按 db_tile 列分块进 shared 复用；每 lane 维护局部 int32 最小堆，
//     warp shuffle 蝶形合并。
// EN: Large-d path (d > kMipsThreadPerColMaxD). A block owns kMipsWarps warps
//     (one warp per row); db columns are tiled into shared for reuse; each lane
//     keeps a local int32 min-heap; a warp-shuffle butterfly merges the lanes.
template <int K>
__global__ void fused_mips_topk_warp_kernel(
        const float* __restrict__ query,   // (n_q, d)
        const float* __restrict__ db,      // (n_db, d)
        int64_t n_q,
        int64_t n_db,
        int64_t d,
        int64_t db_tile,
        float* __restrict__ out_val,       // (n_q, K)
        int64_t* __restrict__ out_idx) {   // (n_q, K)
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int64_t row = static_cast<int64_t>(blockIdx.x) * kMipsWarps + warp;
    // CN: 不能提前 return：未满的 block 里越界 warp 仍需参与 __syncthreads。
    // EN: Must not return early: out-of-range warps in a partial block still need
    //     to participate in __syncthreads.
    const bool row_active = row < n_q;

    extern __shared__ float smem[];
    float* s_q = smem;                          // (kMipsWarps, d)
    float* s_db = smem + kMipsWarps * d;        // (db_tile, d)

    // CN: 协作载入本 block 的 query 行。
    // EN: Cooperative load of this block's query rows.
    {
        const int64_t rr0 = static_cast<int64_t>(blockIdx.x) * kMipsWarps;
        for (int64_t rr = warp; rr < kMipsWarps; rr += kMipsWarps) {
            const int64_t g = rr0 + rr;
            if (g >= n_q) {
                continue;
            }
            const float* qs = query + g * d;
            for (int64_t kk = lane; kk < d; kk += 32) {
                s_q[rr * d + kk] = qs[kk];
            }
        }
    }

    MipsHeapTopK<int32_t, K> local;
    local.init();
    const float* qrow = s_q + warp * d;

    const int64_t n_tiles = (n_db + db_tile - 1) / db_tile;
    for (int64_t tile = 0; tile < n_tiles; ++tile) {
        const int64_t c0 = tile * db_tile;
        const int64_t c_count = min(db_tile, n_db - c0);
        // CN: 协作载入 db tile；所有 warp 参与载入与同步，仅 active 行做 top-k。
        // EN: All warps participate in the cooperative db-tile load and sync; only
        //     active rows run the top-k work.
        for (int64_t cc = warp; cc < c_count; cc += kMipsWarps) {
            const float* dsrc = db + (c0 + cc) * d;
            for (int64_t kk = lane; kk < d; kk += 32) {
                s_db[cc * d + kk] = dsrc[kk];
            }
        }
        __syncthreads();
        if (row_active) {
            for (int64_t c = lane; c < c_count; c += 32) {
                const float* drow = s_db + c * d;
                float acc = 0.0f;
                for (int64_t k = 0; k < d; ++k) {
                    acc += drow[k] * qrow[k];
                }
                local.push(acc, static_cast<int32_t>(c0 + c));
            }
        }
        __syncthreads();
    }

    if (row_active) {
        // CN: warp shuffle 蝶形合并 32 个 lane 的局部 top-K。先快照 partner 的全部 K 对，
        //     再统一 push，避免 push 过程中读取到被修改的堆导致重复/丢失。
        // EN: Warp-shuffle butterfly merge of the 32 per-lane local top-Ks. Snapshot
        //     all K pairs of the partner lane first, then push them all, so pushes do
        //     not read a heap mutated mid-iteration (which would duplicate/lose entries).
        for (int off = 16; off > 0; off >>= 1) {
            float pv[K];
            int32_t pi[K];
#pragma unroll
            for (int r = 0; r < K; ++r) {
                pv[r] = __shfl_down_sync(0xffffffffu, local.val_at(r), off);
                pi[r] = __shfl_down_sync(0xffffffffu, static_cast<int32_t>(local.idx_at(r)), off);
            }
            if (lane < 32 - off) {
#pragma unroll
                for (int r = 0; r < K; ++r) {
                    local.push(pv[r], pi[r]);
                }
            }
        }

        if (lane == 0) {
            float ov[K];
            int64_t oi[K];
            local.sort_desc(ov, oi);
            const int64_t base = row * K;
#pragma unroll
            for (int r = 0; r < K; ++r) {
                out_val[base + r] = ov[r];
                out_idx[base + r] = oi[r];
            }
        }
    }
}

template <int K>
void launch_mips_thread_per_col(
        const float* query,
        const float* db,
        int64_t n_q,
        int64_t n_db,
        int64_t d,
        float* out_val,
        int64_t* out_idx,
        cudaStream_t stream) {
    const size_t shmem = static_cast<size_t>(d) * sizeof(float);
    fused_mips_topk_thread_per_col_kernel<K><<<static_cast<unsigned int>(n_q), kMipsThreads, shmem, stream>>>(
            query, db, n_q, n_db, d, out_val, out_idx);
}

static int64_t get_device_shared_budget() {
    int max_dyn = 0;
    int device = 0;
    if (cudaGetDevice(&device) == cudaSuccess) {
        if (cudaDeviceGetAttribute(&max_dyn, cudaDevAttrMaxSharedMemoryPerBlockOptin, device) == cudaSuccess && max_dyn > 0) {
            return static_cast<int64_t>(max_dyn);
        }
    }
    return static_cast<int64_t>(48 * 1024);
}

// CN: 为大维度 norm-cost kernel 联合选择 query 行数与 database tile。
//     shared memory 至少需要同时容纳一行 query 和一行 database；维度增大时先减少 query 行数。
// EN: Jointly select the query-row count and database tile for large-d norm-cost kernels.
//     Shared memory must hold at least one query and one database row; reduce query rows first as d grows.
static std::pair<int64_t, int64_t> plan_large_d_shared_tiles(int64_t d) {
    // CN: bidirectional/certificate kernels 另有 FP64 静态归约状态；为它预留 8 KiB。
    // EN: Bidirectional/certificate kernels also own FP64 static reduction state; reserve 8 KiB for it.
    const int64_t shared_budget = std::max<int64_t>(1, get_device_shared_budget() - 8 * 1024);
    const int64_t row_bytes = d * static_cast<int64_t>(sizeof(float));
    TORCH_CHECK(
            row_bytes > 0 && shared_budget / row_bytes >= 2,
            "feature dimension exceeds norm-cost scan shared-memory capacity: d=",
            d,
            ", device opt-in limit=",
            shared_budget);
    const int64_t resident_rows = shared_budget / row_bytes;
    const int64_t rows_per_block = std::max<int64_t>(1, std::min<int64_t>(kLargeRows, resident_rows - 1));
    const int64_t db_tile = std::max<int64_t>(1, std::min<int64_t>(64, resident_rows - rows_per_block));
    return {rows_per_block, db_tile};
}

template <int K>
void launch_mips_warp(
        const float* query,
        const float* db,
        int64_t n_q,
        int64_t n_db,
        int64_t d,
        float* out_val,
        int64_t* out_idx,
        cudaStream_t stream) {
    const int64_t grid = (n_q + kMipsWarps - 1) / kMipsWarps;
    const int64_t kSharedBudgetBytes = get_device_shared_budget();
    const int64_t query_bytes = kMipsWarps * d * static_cast<int64_t>(sizeof(float));
    int64_t db_tile = (kSharedBudgetBytes > query_bytes)
            ? ((kSharedBudgetBytes - query_bytes) / (d * static_cast<int64_t>(sizeof(float))))
            : 1;
    db_tile = std::max<int64_t>(1, std::min<int64_t>(256, db_tile));
    const size_t shmem = static_cast<size_t>((kMipsWarps + db_tile) * d) * sizeof(float);
    TORCH_CHECK(
            shmem <= static_cast<size_t>(kSharedBudgetBytes),
            "MIPS warp kernel shared memory too large: requested ",
            shmem,
            " bytes, device opt-in limit ",
            kSharedBudgetBytes,
            " bytes");
    if (shmem > 48 * 1024) {
        // CN: 超过 CUDA 默认动态 shared-memory 上限时，必须为这个模板实例显式申请 opt-in 容量。
        // EN: Explicitly opt this template instance into the larger dynamic shared-memory limit.
        const cudaError_t attr_err = cudaFuncSetAttribute(
                fused_mips_topk_warp_kernel<K>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(shmem));
        TORCH_CHECK(attr_err == cudaSuccess, "cudaFuncSetAttribute failed: ", cudaGetErrorString(attr_err));
    }
    fused_mips_topk_warp_kernel<K><<<static_cast<unsigned int>(grid), kMipsWarps * 32, shmem, stream>>>(
            query, db, n_q, n_db, d, db_tile, out_val, out_idx);
    // CN: 在真实 launch 位置报告配置错误，避免异步延迟到后续无关的 tensor 操作。
    // EN: Report launch-configuration errors here instead of surfacing asynchronously at an unrelated tensor operation.
    const cudaError_t launch_err = cudaGetLastError();
    TORCH_CHECK(launch_err == cudaSuccess, "MIPS warp kernel launch failed: ", cudaGetErrorString(launch_err));
}

template <int K>
void dispatch_mips_topk(
        int64_t d,
        int64_t n_db,
        const float* query,
        const float* db,
        int64_t n_q,
        float* out_val,
        int64_t* out_idx,
        cudaStream_t stream) {
    // CN: thread-per-col 路径用 uint16 列号，覆盖 0..65535（n_db <= 65536）。
    // EN: The thread-per-col path uses uint16 column ids covering 0..65535
    //     (n_db <= 65536).
    if (d <= kMipsThreadPerColMaxD && n_db <= 65536) {
        launch_mips_thread_per_col<K>(query, db, n_q, n_db, d, out_val, out_idx, stream);
    } else {
        launch_mips_warp<K>(query, db, n_q, n_db, d, out_val, out_idx, stream);
    }
}

}  // namespace

std::vector<torch::Tensor> fused_mips_topk_cuda(
        torch::Tensor query,
        torch::Tensor db,
        int64_t k) {
    TORCH_CHECK(query.is_cuda(), "query must be a CUDA tensor");
    TORCH_CHECK(db.is_cuda(), "db must be a CUDA tensor");
    TORCH_CHECK(query.scalar_type() == torch::kFloat32, "query must be float32");
    TORCH_CHECK(db.scalar_type() == torch::kFloat32, "db must be float32");
    TORCH_CHECK(query.is_contiguous(), "query must be contiguous");
    TORCH_CHECK(db.is_contiguous(), "db must be contiguous");
    TORCH_CHECK(query.dim() == 2 && db.dim() == 2, "query/db must be 2D");
    TORCH_CHECK(query.size(1) == db.size(1), "query/db dims must match");
    TORCH_CHECK(k == 1 || k == 2 || k == 4 || k == 8 || k == 16 || k == 32, "k must be one of {1, 2, 4, 8, 16, 32}");
    TORCH_CHECK(query.numel() > 0 && db.numel() > 0, "inputs must be non-empty");

    const c10::cuda::CUDAGuard guard(query.device());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto query_c = query.contiguous();
    auto db_c = db.contiguous();
    const int64_t n_q = query_c.size(0);
    const int64_t n_db = db_c.size(0);
    const int64_t d = query_c.size(1);

    auto out_val = torch::empty({n_q, k}, query_c.options());
    auto out_idx = torch::empty({n_q, k}, query_c.options().dtype(torch::kInt64));

    const float* q_ptr = query_c.data_ptr<float>();
    const float* db_ptr = db_c.data_ptr<float>();
    float* val_ptr = out_val.data_ptr<float>();
    int64_t* idx_ptr = out_idx.data_ptr<int64_t>();

    if (k == 1) {
        dispatch_mips_topk<1>(d, n_db, q_ptr, db_ptr, n_q, val_ptr, idx_ptr, stream);
    } else if (k == 2) {
        dispatch_mips_topk<2>(d, n_db, q_ptr, db_ptr, n_q, val_ptr, idx_ptr, stream);
    } else if (k == 4) {
        dispatch_mips_topk<4>(d, n_db, q_ptr, db_ptr, n_q, val_ptr, idx_ptr, stream);
    } else if (k == 8) {
        dispatch_mips_topk<8>(d, n_db, q_ptr, db_ptr, n_q, val_ptr, idx_ptr, stream);
    } else if (k == 16) {
        dispatch_mips_topk<16>(d, n_db, q_ptr, db_ptr, n_q, val_ptr, idx_ptr, stream);
    } else {
        dispatch_mips_topk<32>(d, n_db, q_ptr, db_ptr, n_q, val_ptr, idx_ptr, stream);
    }
    return {out_val, out_idx};
}

std::vector<torch::Tensor> fused_gcost_topk_cuda(
        torch::Tensor query,
        torch::Tensor db,
        torch::Tensor dual_db,
        int64_t k,
        int64_t cost_type) {
    check_cuda_inputs(query, db, dual_db, k, cost_type);
    const c10::cuda::CUDAGuard guard(query.device());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto query_c = query.contiguous();
    auto db_c = db.contiguous();
    auto dual_c = dual_db.contiguous();
    const int64_t n_q = query_c.size(0);
    const int64_t n_db = db_c.size(0);
    const int64_t d = query_c.size(1);

    auto scalar_options = query_c.options().dtype(torch::kFloat64);
    auto out_val = torch::empty({n_q, k}, scalar_options);
    auto out_idx = torch::empty({n_q, k}, query_c.options().dtype(torch::kInt64));

    const float* q_ptr = query_c.data_ptr<float>();
    const float* db_ptr = db_c.data_ptr<float>();
    const double* dual_ptr = dual_c.data_ptr<double>();
    double* val_ptr = out_val.data_ptr<double>();
    int64_t* idx_ptr = out_idx.data_ptr<int64_t>();

    // CN: 大 d 路径的 shared memory 预算：R 行 query + db_tile 列，动态适应设备上限。
    // EN: Shared-memory budget for large-d path: R query rows plus db tile, adapting to device limits.
    const auto large_d_tiles = plan_large_d_shared_tiles(d);
    const int64_t rows_per_block = large_d_tiles.first;
    const int64_t db_tile = large_d_tiles.second;

    if (k == 1) {
        dispatch_topk<1>(cost_type, q_ptr, db_ptr, dual_ptr, n_q, n_db, d, rows_per_block, db_tile, val_ptr, idx_ptr, stream);
    } else if (k == 2) {
        dispatch_topk<2>(cost_type, q_ptr, db_ptr, dual_ptr, n_q, n_db, d, rows_per_block, db_tile, val_ptr, idx_ptr, stream);
    } else if (k == 4) {
        dispatch_topk<4>(cost_type, q_ptr, db_ptr, dual_ptr, n_q, n_db, d, rows_per_block, db_tile, val_ptr, idx_ptr, stream);
    } else if (k == 8) {
        dispatch_topk<8>(cost_type, q_ptr, db_ptr, dual_ptr, n_q, n_db, d, rows_per_block, db_tile, val_ptr, idx_ptr, stream);
    } else if (k == 16) {
        dispatch_topk<16>(cost_type, q_ptr, db_ptr, dual_ptr, n_q, n_db, d, rows_per_block, db_tile, val_ptr, idx_ptr, stream);
    } else {
        dispatch_topk<32>(cost_type, q_ptr, db_ptr, dual_ptr, n_q, n_db, d, rows_per_block, db_tile, val_ptr, idx_ptr, stream);
    }
    return {out_val, out_idx};
}

std::vector<torch::Tensor> fused_gcost_certificate_cuda(
        torch::Tensor query,
        torch::Tensor db,
        torch::Tensor query_dual,
        torch::Tensor db_dual,
        int64_t cost_type,
        int64_t compute_denominator) {
    check_certificate_inputs(query, db, query_dual, db_dual, cost_type);
    const c10::cuda::CUDAGuard guard(query.device());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto query_c = query.contiguous();
    auto db_c = db.contiguous();
    auto query_dual_c = query_dual.contiguous();
    auto db_dual_c = db_dual.contiguous();
    const int64_t n_q = query_c.size(0);
    const int64_t n_db = db_c.size(0);
    const int64_t d = query_c.size(1);

    auto scalar_options = query_c.options().dtype(torch::kFloat64);
    auto out_num = torch::zeros({}, scalar_options);
    auto out_den = torch::zeros({}, scalar_options);
    auto out_max_violation = torch::zeros({}, scalar_options);
    auto out_cost_linf = torch::zeros({}, scalar_options);
    auto out_positive_count = torch::zeros({}, query_c.options().dtype(torch::kInt64));

    const auto large_d_tiles = plan_large_d_shared_tiles(d);
    const int64_t rows_per_block = large_d_tiles.first;
    const int64_t db_tile = large_d_tiles.second;

    launch_certificate(
            cost_type,
            query_c.data_ptr<float>(),
            db_c.data_ptr<float>(),
            query_dual_c.data_ptr<double>(),
            db_dual_c.data_ptr<double>(),
            n_q,
            n_db,
            d,
            rows_per_block,
            db_tile,
            bool(compute_denominator),
            out_num.data_ptr<double>(),
            out_den.data_ptr<double>(),
            out_max_violation.data_ptr<double>(),
            out_cost_linf.data_ptr<double>(),
            out_positive_count.data_ptr<int64_t>(),
            stream);
    return {out_num, out_den, out_max_violation, out_cost_linf, out_positive_count};
}

std::vector<torch::Tensor> fused_gcost_bidir_certificate_cuda(
        torch::Tensor query,
        torch::Tensor db,
        torch::Tensor query_dual,
        torch::Tensor db_dual,
        int64_t k,
        int64_t cost_type) {
    check_cuda_inputs(query, db, db_dual, k, cost_type);
    TORCH_CHECK(query_dual.is_cuda() && query_dual.scalar_type() == torch::kFloat64, "query_dual must be CUDA float64");
    TORCH_CHECK(query_dual.is_contiguous() && query_dual.numel() == query.size(0), "query_dual length mismatch");
    const c10::cuda::CUDAGuard guard(query.device());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int64_t n_q = query.size(0);
    const int64_t n_db = db.size(0);
    const int64_t d = query.size(1);
    const auto large_d_tiles = plan_large_d_shared_tiles(d);
    const int64_t rows_per_block = large_d_tiles.first;
    const int64_t db_tile = large_d_tiles.second;
    const int64_t n_blocks = (n_q + rows_per_block - 1) / rows_per_block;

    auto scalar_options = query.options().dtype(torch::kFloat64);
    auto row_values = torch::empty({n_q, k}, scalar_options);
    auto row_indices = torch::empty({n_q, k}, query.options().dtype(torch::kInt64));
    auto col_values = torch::empty({n_db, k}, scalar_options);
    auto col_indices = torch::empty({n_db, k}, query.options().dtype(torch::kInt64));
    auto partial_values = torch::empty({n_blocks, n_db, k}, scalar_options);
    auto partial_indices = torch::empty({n_blocks, n_db, k}, query.options().dtype(torch::kInt64));
    auto out_num = torch::zeros({}, scalar_options);
    auto out_den = torch::zeros({}, scalar_options);
    auto out_max_violation = torch::zeros({}, scalar_options);
    auto out_cost_linf = torch::zeros({}, scalar_options);
    auto out_positive_count = torch::zeros({}, query.options().dtype(torch::kInt64));
#define DISPATCH_BIDIR(K) \
    dispatch_bidir_certificate<K>(cost_type, query.data_ptr<float>(), db.data_ptr<float>(), query_dual.data_ptr<double>(), db_dual.data_ptr<double>(), n_q, n_db, d, rows_per_block, db_tile, row_values.data_ptr<double>(), row_indices.data_ptr<int64_t>(), partial_values.data_ptr<double>(), partial_indices.data_ptr<int64_t>(), out_num.data_ptr<double>(), out_den.data_ptr<double>(), out_max_violation.data_ptr<double>(), out_cost_linf.data_ptr<double>(), out_positive_count.data_ptr<int64_t>(), stream); \
    launch_reduce_gcost_col_partials<K>(partial_values.data_ptr<double>(), partial_indices.data_ptr<int64_t>(), n_blocks, n_db, col_values.data_ptr<double>(), col_indices.data_ptr<int64_t>(), stream)
    if (k == 1) {
        DISPATCH_BIDIR(1);
    } else if (k == 2) {
        DISPATCH_BIDIR(2);
    } else if (k == 4) {
        DISPATCH_BIDIR(4);
    } else if (k == 8) {
        DISPATCH_BIDIR(8);
    } else if (k == 16) {
        DISPATCH_BIDIR(16);
    } else {
        DISPATCH_BIDIR(32);
    }
#undef DISPATCH_BIDIR
    return {row_values, row_indices, col_values, col_indices, out_num, out_den, out_max_violation, out_cost_linf, out_positive_count};
}
