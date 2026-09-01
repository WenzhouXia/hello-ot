#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cublas_v2.h>
#include <cub/warp/warp_merge_sort.cuh>
#include <torch/extension.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

constexpr int kThreads = 256;
constexpr int kTopkThreads = 128;
// CN: 仅 K=32 使用 64 threads，使 FP64 top-k 静态共享内存保持在默认上限内。
// EN: Only K=32 uses 64 threads, keeping FP64 top-k static shared memory below the default limit.
constexpr int kTopkThreadsK32 = 64;
constexpr int kColGroupCols = 128;
constexpr int kColGroupRows = 2;
constexpr int kInitializationTopk16DbTile = 8192;
constexpr int kInitializationTopk16Threads = 128;
constexpr int kInitializationTopk16Warps = kInitializationTopk16Threads / 32;

__global__ void init_stats_kernel(double* stats) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        stats[0] = 0.0;
        stats[1] = 0.0;
        stats[2] = 0.0;
        stats[3] = 0.0;
    }
}

__device__ double atomic_max_nonnegative_double(double* address, double value) {
    auto* address_as_ull = reinterpret_cast<unsigned long long int*>(address);
    unsigned long long int old = *address_as_ull;
    unsigned long long int assumed;
    do {
        assumed = old;
        if (__longlong_as_double(assumed) >= value) {
            break;
        }
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(value));
    } while (assumed != old);
    return __longlong_as_double(old);
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

template <int K, int THREADS>
__global__ void update_topk_stats_raw_kernel(
        const float* __restrict__ dots,
        int64_t dot_ld,
        int64_t q_offset,
        int64_t db_offset,
        int64_t q_count,
        int64_t db_count,
        const double* __restrict__ query_cost,
        const double* __restrict__ db_cost,
        const double* __restrict__ query_bias,
        const double* __restrict__ db_bias,
        double* __restrict__ out_val,
        int32_t* __restrict__ out_idx,
        double* __restrict__ stats,
        bool compute_stats,
        bool collect_linf_stats) {
    int row = blockIdx.x;
    if (row >= q_count) {
        return;
    }

    __shared__ double best_vals[THREADS][K];
    __shared__ int best_idxs[THREADS][K];
    __shared__ double num_vals[THREADS];
    __shared__ double den_vals[THREADS];
    extern __shared__ double cost_linf_vals[];
    __shared__ unsigned long long pos_vals[THREADS];

    double local_vals[K];
    int local_idxs[K];
    init_topk<K>(local_vals, local_idxs);
    double num_sum = 0.0;
    double den_sum = 0.0;
    double cost_linf = 0.0;
    unsigned long long pos_count = 0;
    const double qc = query_cost[q_offset + row];
    const double qb = query_bias[q_offset + row];

    for (int64_t col = threadIdx.x; col < db_count; col += blockDim.x) {
        float dot = dots[row * dot_ld + col];
        double rc = static_cast<double>(dot) + qb + db_bias[db_offset + col];
        insert_topk<K>(local_vals, local_idxs, rc, static_cast<int>(db_offset + col));
        if (compute_stats) {
            double cost_d = qc + db_cost[db_offset + col] - static_cast<double>(dot);
            den_sum += cost_d * cost_d;
            if (collect_linf_stats) {
                cost_linf = fmax(cost_linf, fabs(cost_d));
            }
            if (rc > 0.0) {
                num_sum += rc * rc;
                ++pos_count;
            }
        }
    }

#pragma unroll
    for (int rank = 0; rank < K; ++rank) {
        best_vals[threadIdx.x][rank] = local_vals[rank];
        best_idxs[threadIdx.x][rank] = local_idxs[rank];
    }
    num_vals[threadIdx.x] = num_sum;
    den_vals[threadIdx.x] = den_sum;
    if (collect_linf_stats) {
        cost_linf_vals[threadIdx.x] = cost_linf;
    }
    pos_vals[threadIdx.x] = pos_count;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
#pragma unroll
            for (int rank = 0; rank < K; ++rank) {
                insert_topk<K>(
                        local_vals,
                        local_idxs,
                        best_vals[threadIdx.x + stride][rank],
                        best_idxs[threadIdx.x + stride][rank]);
            }
            num_vals[threadIdx.x] += num_vals[threadIdx.x + stride];
            den_vals[threadIdx.x] += den_vals[threadIdx.x + stride];
            if (collect_linf_stats) {
                cost_linf_vals[threadIdx.x] = fmax(
                        cost_linf_vals[threadIdx.x], cost_linf_vals[threadIdx.x + stride]);
            }
            pos_vals[threadIdx.x] += pos_vals[threadIdx.x + stride];
#pragma unroll
            for (int rank = 0; rank < K; ++rank) {
                best_vals[threadIdx.x][rank] = local_vals[rank];
                best_idxs[threadIdx.x][rank] = local_idxs[rank];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        int64_t global_row = q_offset + row;
        double final_vals[K];
        int final_idxs[K];
        init_topk<K>(final_vals, final_idxs);
#pragma unroll
        for (int rank = 0; rank < K; ++rank) {
            int64_t out_offset = global_row * K + rank;
            insert_topk<K>(final_vals, final_idxs, out_val[out_offset], static_cast<int>(out_idx[out_offset]));
        }
#pragma unroll
        for (int rank = 0; rank < K; ++rank) {
            insert_topk<K>(final_vals, final_idxs, local_vals[rank], local_idxs[rank]);
        }
#pragma unroll
        for (int rank = 0; rank < K; ++rank) {
            int64_t out_offset = global_row * K + rank;
            out_val[out_offset] = final_vals[rank];
            out_idx[out_offset] = static_cast<int32_t>(final_idxs[rank]);
        }
        if (compute_stats) {
            atomicAdd(stats + 0, num_vals[0]);
            atomicAdd(stats + 1, den_vals[0]);
            atomicAdd(stats + 2, static_cast<double>(pos_vals[0]));
            if (collect_linf_stats) {
                atomic_max_nonnegative_double(stats + 3, cost_linf_vals[0]);
            }
        }
    }
}

struct TopkCandidate {
    double value;
    int32_t index;
};

struct SortBetterCandidate {
    __device__ __forceinline__ bool operator()(const TopkCandidate& lhs, const TopkCandidate& rhs) const {
        return is_better_pair(lhs.value, lhs.index, rhs.value, rhs.index);
    }
};

template <typename WarpSort>
__device__ __forceinline__ void merge_initialization_thread_queue(
        TopkCandidate (&thread_queue)[2],
        int& thread_queue_size,
        TopkCandidate* warp_queue,
        TopkCandidate* merge_items,
        typename WarpSort::TempStorage& sort_storage) {
    // CN: 只在小型 per-thread queue 满时才合并到 persistent warp queue。
    // EN: Merge into the persistent warp queue only when the small per-thread queue fills.
    constexpr int items_per_thread = 3;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    TopkCandidate items[items_per_thread] = {
            warp_queue[lane], thread_queue[0], thread_queue[1]};
    WarpSort(sort_storage).Sort(items, SortBetterCandidate{});
#pragma unroll
    for (int item = 0; item < items_per_thread; ++item) {
        merge_items[lane * items_per_thread + item] = items[item];
    }
    __syncwarp();
    // CN: WarpMergeSort 使用 blocked layout；前 32 项即新的 warp candidate queue。
    // EN: WarpMergeSort uses blocked layout; its first 32 items form the new warp candidate queue.
    warp_queue[lane] = merge_items[lane];
    thread_queue[0] = TopkCandidate{-std::numeric_limits<double>::infinity(), -1};
    thread_queue[1] = TopkCandidate{-std::numeric_limits<double>::infinity(), -1};
    thread_queue_size = 0;
    __syncwarp();
}

__global__ void update_initialization_topk16_warp_queue_kernel(
        const float* __restrict__ dots,
        int64_t dot_ld,
        int64_t q_offset,
        int64_t db_offset,
        int64_t q_count,
        int64_t db_count,
        const double* __restrict__ query_bias,
        const double* __restrict__ db_bias,
        double* __restrict__ out_val,
        int32_t* __restrict__ out_idx) {
    // CN: initialization K=16 专用 selection。每个 warp 处理一行，以 persistent warp queue
    //     和两项 thread queue 单遍读取 score，只在候选 queue 满时排序合并。
    // EN: Initialization-specific K=16 selection. One warp handles one row and scans scores once,
    //     using a persistent warp queue plus two-entry thread queues that merge only when full.
    constexpr int k = 16;
    constexpr int warp_size = 32;
    constexpr int items_per_thread = 3;
    using WarpSort = cub::WarpMergeSort<TopkCandidate, items_per_thread, warp_size>;

    const int warp_id = static_cast<int>(threadIdx.x) / warp_size;
    const int lane = static_cast<int>(threadIdx.x) & (warp_size - 1);
    const int row = static_cast<int>(blockIdx.x) * kInitializationTopk16Warps + warp_id;
    if (row >= q_count) {
        return;
    }

    __shared__ typename WarpSort::TempStorage sort_storage[kInitializationTopk16Warps];
    __shared__ TopkCandidate warp_queues[kInitializationTopk16Warps][warp_size];
    __shared__ TopkCandidate merge_items[kInitializationTopk16Warps][warp_size * items_per_thread];
    TopkCandidate* warp_queue = warp_queues[warp_id];
    TopkCandidate* warp_merge_items = merge_items[warp_id];

    const int64_t global_row = q_offset + row;
    if (lane < k) {
        const int64_t output_offset = global_row * k + lane;
        warp_queue[lane] = TopkCandidate{out_val[output_offset], out_idx[output_offset]};
    } else {
        warp_queue[lane] = TopkCandidate{-std::numeric_limits<double>::infinity(), -1};
    }
    __syncwarp();

    TopkCandidate thread_queue[2] = {
            TopkCandidate{-std::numeric_limits<double>::infinity(), -1},
            TopkCandidate{-std::numeric_limits<double>::infinity(), -1}};
    int thread_queue_size = 0;
    const double query_offset = query_bias[global_row];
    for (int64_t col_base = 0; col_base < db_count; col_base += warp_size) {
        const int64_t col = col_base + lane;
        if (col < db_count) {
            const TopkCandidate candidate{
                    static_cast<double>(dots[row * dot_ld + col]) + query_offset + db_bias[db_offset + col],
                    static_cast<int32_t>(db_offset + col)};
            const TopkCandidate threshold = warp_queue[k - 1];
            if (is_better_pair(candidate.value, candidate.index, threshold.value, threshold.index)) {
                thread_queue[1] = thread_queue[0];
                thread_queue[0] = candidate;
                ++thread_queue_size;
            }
        }
        if (__any_sync(0xffffffffU, thread_queue_size == 2)) {
            merge_initialization_thread_queue<WarpSort>(
                    thread_queue,
                    thread_queue_size,
                    warp_queue,
                    warp_merge_items,
                    sort_storage[warp_id]);
        }
    }
    // CN: 合并不足两项的尾部候选；空 queue 由 sentinel 填充，不改变结果。
    // EN: Merge the final partially filled queues; sentinels make empty queues a no-op.
    merge_initialization_thread_queue<WarpSort>(
            thread_queue,
            thread_queue_size,
            warp_queue,
            warp_merge_items,
            sort_storage[warp_id]);
    if (lane < k) {
        const int64_t output_offset = global_row * k + lane;
        out_val[output_offset] = warp_queue[lane].value;
        out_idx[output_offset] = warp_queue[lane].index;
    }
}

template <int K, int COLS>
__global__ void update_col_partial_topk_raw_grouped_kernel(
        const float* __restrict__ dots,
        int64_t dot_ld,
        int64_t q_offset,
        int64_t db_offset,
        int64_t q_count,
        int64_t db_count,
        int64_t n_db,
        int64_t q_tile_id,
        const double* __restrict__ query_bias,
        const double* __restrict__ db_bias,
        double* __restrict__ partial_val,
        int32_t* __restrict__ partial_idx) {
    int local_col = threadIdx.x;
    int row_lane = threadIdx.y;
    int col = blockIdx.x * COLS + local_col;
    if (local_col >= COLS) {
        return;
    }

    __shared__ double best_vals[kColGroupRows][COLS][K];
    __shared__ int best_idxs[kColGroupRows][COLS][K];

    double local_vals[K];
    int local_idxs[K];
    init_topk<K>(local_vals, local_idxs);
    if (col < db_count) {
        const double dbb = db_bias[db_offset + col];
        for (int64_t row = row_lane; row < q_count; row += kColGroupRows) {
            float dot = dots[row * dot_ld + col];
            double rc = static_cast<double>(dot) + query_bias[q_offset + row] + dbb;
            insert_topk<K>(local_vals, local_idxs, rc, static_cast<int>(q_offset + row));
        }
    }

#pragma unroll
    for (int rank = 0; rank < K; ++rank) {
        best_vals[row_lane][local_col][rank] = local_vals[rank];
        best_idxs[row_lane][local_col][rank] = local_idxs[rank];
    }
    __syncthreads();

    if (row_lane == 0 && col < db_count) {
        double final_vals[K];
        int final_idxs[K];
        init_topk<K>(final_vals, final_idxs);
        for (int lane = 0; lane < kColGroupRows; ++lane) {
#pragma unroll
            for (int rank = 0; rank < K; ++rank) {
                insert_topk<K>(final_vals, final_idxs, best_vals[lane][local_col][rank], best_idxs[lane][local_col][rank]);
            }
        }
        int64_t global_col = db_offset + static_cast<int64_t>(col);
        int64_t out_offset = (q_tile_id * n_db + global_col) * K;
#pragma unroll
        for (int rank = 0; rank < K; ++rank) {
            partial_val[out_offset + rank] = final_vals[rank];
            partial_idx[out_offset + rank] = static_cast<int32_t>(final_idxs[rank]);
        }
    }
}


__global__ void col_partial_load_only_raw_grouped_kernel(
        const float* __restrict__ dots,
        int64_t dot_ld,
        int64_t q_offset,
        int64_t db_offset,
        int64_t q_count,
        int64_t db_count,
        int64_t n_db,
        int64_t q_tile_id,
        const double* __restrict__ query_bias,
        const double* __restrict__ db_bias,
        double* __restrict__ sink) {
    int local_col = threadIdx.x;
    int row_lane = threadIdx.y;
    int col = blockIdx.x * kColGroupCols + local_col;
    if (local_col >= kColGroupCols) {
        return;
    }

    __shared__ double lane_sums[kColGroupRows][kColGroupCols];
    double local_sum = 0.0;
    if (col < db_count) {
        const double dbb = db_bias[db_offset + col];
        for (int64_t row = row_lane; row < q_count; row += kColGroupRows) {
            float dot = dots[row * dot_ld + col];
            local_sum += static_cast<double>(dot) + query_bias[q_offset + row] + dbb;
        }
    }
    lane_sums[row_lane][local_col] = local_sum;
    __syncthreads();

    if (row_lane == 0 && col < db_count) {
        int64_t global_col = db_offset + static_cast<int64_t>(col);
        sink[q_tile_id * n_db + global_col] = lane_sums[0][local_col] + lane_sums[1][local_col];
    }
}

template <int K, int COLS>
__global__ void col_partial_topk_no_partial_write_raw_grouped_kernel(
        const float* __restrict__ dots,
        int64_t dot_ld,
        int64_t q_offset,
        int64_t db_offset,
        int64_t q_count,
        int64_t db_count,
        int64_t n_db,
        int64_t q_tile_id,
        const double* __restrict__ query_bias,
        const double* __restrict__ db_bias,
        double* __restrict__ sink_val,
        int32_t* __restrict__ sink_idx) {
    int local_col = threadIdx.x;
    int row_lane = threadIdx.y;
    int col = blockIdx.x * COLS + local_col;
    if (local_col >= COLS) {
        return;
    }

    __shared__ double best_vals[kColGroupRows][COLS][K];
    __shared__ int best_idxs[kColGroupRows][COLS][K];

    double local_vals[K];
    int local_idxs[K];
    init_topk<K>(local_vals, local_idxs);
    if (col < db_count) {
        const double dbb = db_bias[db_offset + col];
        for (int64_t row = row_lane; row < q_count; row += kColGroupRows) {
            float dot = dots[row * dot_ld + col];
            double rc = static_cast<double>(dot) + query_bias[q_offset + row] + dbb;
            insert_topk<K>(local_vals, local_idxs, rc, static_cast<int>(q_offset + row));
        }
    }

#pragma unroll
    for (int rank = 0; rank < K; ++rank) {
        best_vals[row_lane][local_col][rank] = local_vals[rank];
        best_idxs[row_lane][local_col][rank] = local_idxs[rank];
    }
    __syncthreads();

    if (row_lane == 0 && col < db_count) {
        double final_vals[K];
        int final_idxs[K];
        init_topk<K>(final_vals, final_idxs);
        for (int lane = 0; lane < kColGroupRows; ++lane) {
#pragma unroll
            for (int rank = 0; rank < K; ++rank) {
                insert_topk<K>(final_vals, final_idxs, best_vals[lane][local_col][rank], best_idxs[lane][local_col][rank]);
            }
        }
        int64_t global_col = db_offset + static_cast<int64_t>(col);
        int64_t out_offset = q_tile_id * n_db + global_col;
        sink_val[out_offset] = final_vals[0];
        sink_idx[out_offset] = static_cast<int32_t>(final_idxs[0]);
    }
}

template <int K>
__global__ void col_partial_write_only_raw_grouped_kernel(
        int64_t db_offset,
        int64_t db_count,
        int64_t n_db,
        int64_t q_tile_id,
        double* __restrict__ partial_val,
        int32_t* __restrict__ partial_idx) {
    int local_col = threadIdx.x;
    int row_lane = threadIdx.y;
    int col = blockIdx.x * kColGroupCols + local_col;
    if (row_lane != 0 || col >= db_count) {
        return;
    }
    int64_t global_col = db_offset + static_cast<int64_t>(col);
    int64_t out_offset = (q_tile_id * n_db + global_col) * K;
#pragma unroll
    for (int rank = 0; rank < K; ++rank) {
        partial_val[out_offset + rank] = -std::numeric_limits<double>::infinity();
        partial_idx[out_offset + rank] = -1;
    }
}



__device__ __forceinline__ void merge_top2_arrays(
        double (&vals)[2],
        int (&idxs)[2],
        const double (&other_vals)[2],
        const int (&other_idxs)[2]) {
#pragma unroll
    for (int rank = 0; rank < 2; ++rank) {
        insert_topk<2>(vals, idxs, other_vals[rank], other_idxs[rank]);
    }
}

__global__ void update_col_partial_topk2_raw_warp_kernel(
        const float* __restrict__ dots,
        int64_t dot_ld,
        int64_t q_offset,
        int64_t db_offset,
        int64_t q_count,
        int64_t db_count,
        int64_t n_db,
        int64_t q_tile_id,
        const double* __restrict__ query_bias,
        const double* __restrict__ db_bias,
        double* __restrict__ partial_val,
        int32_t* __restrict__ partial_idx) {
    constexpr int kColsPerWarp = 4;
    constexpr int kLanesPerCol = 8;
    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane = tid & 31;
    int col_in_warp = lane / kLanesPerCol;
    int lane_in_col = lane & (kLanesPerCol - 1);
    int col = (blockIdx.x * (blockDim.x / 32) + warp_id) * kColsPerWarp + col_in_warp;

    double local_vals[2];
    int local_idxs[2];
    init_topk<2>(local_vals, local_idxs);
    if (col < db_count) {
        const double dbb = db_bias[db_offset + col];
        for (int64_t row = lane_in_col; row < q_count; row += kLanesPerCol) {
            float dot = dots[row * dot_ld + col];
            double rc = static_cast<double>(dot) + query_bias[q_offset + row] + dbb;
            insert_topk<2>(local_vals, local_idxs, rc, static_cast<int>(q_offset + row));
        }
    }

    unsigned mask = __activemask();
#pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
        double other_vals[2];
        int other_idxs[2];
#pragma unroll
        for (int rank = 0; rank < 2; ++rank) {
            other_vals[rank] = __shfl_down_sync(mask, local_vals[rank], offset, kLanesPerCol);
            other_idxs[rank] = __shfl_down_sync(mask, local_idxs[rank], offset, kLanesPerCol);
        }
        if (lane_in_col < offset) {
            merge_top2_arrays(local_vals, local_idxs, other_vals, other_idxs);
        }
    }

    if (lane_in_col == 0 && col < db_count) {
        int64_t global_col = db_offset + static_cast<int64_t>(col);
        int64_t out_offset = (q_tile_id * n_db + global_col) * 2;
#pragma unroll
        for (int rank = 0; rank < 2; ++rank) {
            partial_val[out_offset + rank] = local_vals[rank];
            partial_idx[out_offset + rank] = static_cast<int32_t>(local_idxs[rank]);
        }
    }
}

__device__ __forceinline__ void merge_top4_arrays(
        double (&vals)[4],
        int (&idxs)[4],
        const double (&other_vals)[4],
        const int (&other_idxs)[4]) {
#pragma unroll
    for (int rank = 0; rank < 4; ++rank) {
        insert_topk<4>(vals, idxs, other_vals[rank], other_idxs[rank]);
    }
}

__global__ void update_col_partial_topk4_raw_warp_kernel(
        const float* __restrict__ dots,
        int64_t dot_ld,
        int64_t q_offset,
        int64_t db_offset,
        int64_t q_count,
        int64_t db_count,
        int64_t n_db,
        int64_t q_tile_id,
        const double* __restrict__ query_bias,
        const double* __restrict__ db_bias,
        double* __restrict__ partial_val,
        int32_t* __restrict__ partial_idx) {
    constexpr int kColsPerWarp = 4;
    constexpr int kLanesPerCol = 8;
    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane = tid & 31;
    int col_in_warp = lane / kLanesPerCol;
    int lane_in_col = lane & (kLanesPerCol - 1);
    int col = (blockIdx.x * (blockDim.x / 32) + warp_id) * kColsPerWarp + col_in_warp;

    double local_vals[4];
    int local_idxs[4];
    init_topk<4>(local_vals, local_idxs);
    if (col < db_count) {
        const double dbb = db_bias[db_offset + col];
        for (int64_t row = lane_in_col; row < q_count; row += kLanesPerCol) {
            float dot = dots[row * dot_ld + col];
            double rc = static_cast<double>(dot) + query_bias[q_offset + row] + dbb;
            insert_topk<4>(local_vals, local_idxs, rc, static_cast<int>(q_offset + row));
        }
    }

    unsigned mask = __activemask();
#pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
        double other_vals[4];
        int other_idxs[4];
#pragma unroll
        for (int rank = 0; rank < 4; ++rank) {
            other_vals[rank] = __shfl_down_sync(mask, local_vals[rank], offset, kLanesPerCol);
            other_idxs[rank] = __shfl_down_sync(mask, local_idxs[rank], offset, kLanesPerCol);
        }
        if (lane_in_col < offset) {
            merge_top4_arrays(local_vals, local_idxs, other_vals, other_idxs);
        }
    }

    if (lane_in_col == 0 && col < db_count) {
        int64_t global_col = db_offset + static_cast<int64_t>(col);
        int64_t out_offset = (q_tile_id * n_db + global_col) * 4;
#pragma unroll
        for (int rank = 0; rank < 4; ++rank) {
            partial_val[out_offset + rank] = local_vals[rank];
            partial_idx[out_offset + rank] = static_cast<int32_t>(local_idxs[rank]);
        }
    }
}

template <int K, int THREADS>
__global__ void reduce_col_partials_topk_i32_kernel(
        const double* __restrict__ partial_val,
        const int32_t* __restrict__ partial_idx,
        int64_t n_tiles,
        int64_t n_db,
        double* __restrict__ out_val,
        int32_t* __restrict__ out_idx) {
    int col = blockIdx.x;
    if (col >= n_db) {
        return;
    }

    __shared__ double best_vals[THREADS][K];
    __shared__ int best_idxs[THREADS][K];

    double local_vals[K];
    int local_idxs[K];
    init_topk<K>(local_vals, local_idxs);
    for (int64_t tile = threadIdx.x; tile < n_tiles; tile += blockDim.x) {
        int64_t offset = (tile * n_db + static_cast<int64_t>(col)) * K;
#pragma unroll
        for (int rank = 0; rank < K; ++rank) {
            insert_topk<K>(local_vals, local_idxs, partial_val[offset + rank], static_cast<int>(partial_idx[offset + rank]));
        }
    }

#pragma unroll
    for (int rank = 0; rank < K; ++rank) {
        best_vals[threadIdx.x][rank] = local_vals[rank];
        best_idxs[threadIdx.x][rank] = local_idxs[rank];
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
#pragma unroll
            for (int rank = 0; rank < K; ++rank) {
                insert_topk<K>(
                        local_vals,
                        local_idxs,
                        best_vals[threadIdx.x + stride][rank],
                        best_idxs[threadIdx.x + stride][rank]);
            }
#pragma unroll
            for (int rank = 0; rank < K; ++rank) {
                best_vals[threadIdx.x][rank] = local_vals[rank];
                best_idxs[threadIdx.x][rank] = local_idxs[rank];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        int64_t out_offset = static_cast<int64_t>(col) * K;
#pragma unroll
        for (int rank = 0; rank < K; ++rank) {
            out_val[out_offset + rank] = local_vals[rank];
            out_idx[out_offset + rank] = static_cast<int32_t>(local_idxs[rank]);
        }
    }
}

void check_cublas(cublasStatus_t status, const char* what) {
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, what, " failed with cuBLAS status ", static_cast<int>(status));
}

int reduce_threads_from_tile(int64_t col_reduce_tile) {
    int threads = 1;
    while (threads < kThreads && static_cast<int64_t>(threads * 2) <= col_reduce_tile) {
        threads *= 2;
    }
    return std::min<int>(threads, kTopkThreads);
}

template <int K>
void launch_update_topk_stats_raw(
        const float* dot_ptr,
        int64_t dot_ld,
        int64_t q_offset,
        int64_t db_offset,
        int64_t q_count,
        int64_t db_count,
        const double* query_cost,
        const double* db_cost,
        const double* query_bias,
        const double* db_bias,
        double* out_val,
        int32_t* out_idx,
        double* stats,
        bool compute_stats,
        bool collect_linf_stats,
        cudaStream_t stream) {
    constexpr int threads = K == 32 ? kTopkThreadsK32 : kTopkThreads;
    const size_t dynamic_shared_bytes =
            collect_linf_stats ? static_cast<size_t>(threads) * sizeof(double) : 0;
    update_topk_stats_raw_kernel<K, threads><<<
            static_cast<unsigned int>(q_count), threads, dynamic_shared_bytes, stream>>>(
            dot_ptr,
            dot_ld,
            q_offset,
            db_offset,
            q_count,
            db_count,
            query_cost,
            db_cost,
            query_bias,
            db_bias,
            out_val,
            out_idx,
            stats,
            compute_stats,
            collect_linf_stats);
}

void launch_update_topk_stats_raw_for_k(
        int64_t k,
        const float* dot_ptr,
        int64_t dot_ld,
        int64_t q_offset,
        int64_t db_offset,
        int64_t q_count,
        int64_t db_count,
        const double* query_cost,
        const double* db_cost,
        const double* query_bias,
        const double* db_bias,
        double* out_val,
        int32_t* out_idx,
        double* stats,
        bool compute_stats,
        bool collect_linf_stats,
        cudaStream_t stream) {
    if (k == 1) {
        launch_update_topk_stats_raw<1>(
                dot_ptr, dot_ld, q_offset, db_offset, q_count, db_count, query_cost, db_cost, query_bias, db_bias,
                out_val, out_idx, stats, compute_stats, collect_linf_stats, stream);
    } else if (k == 2) {
        launch_update_topk_stats_raw<2>(
                dot_ptr, dot_ld, q_offset, db_offset, q_count, db_count, query_cost, db_cost, query_bias, db_bias,
                out_val, out_idx, stats, compute_stats, collect_linf_stats, stream);
    } else if (k == 4) {
        launch_update_topk_stats_raw<4>(
                dot_ptr, dot_ld, q_offset, db_offset, q_count, db_count, query_cost, db_cost, query_bias, db_bias,
                out_val, out_idx, stats, compute_stats, collect_linf_stats, stream);
    } else if (k == 8) {
        launch_update_topk_stats_raw<8>(
                dot_ptr, dot_ld, q_offset, db_offset, q_count, db_count, query_cost, db_cost, query_bias, db_bias,
                out_val, out_idx, stats, compute_stats, collect_linf_stats, stream);
    } else if (k == 16) {
        launch_update_topk_stats_raw<16>(
                dot_ptr, dot_ld, q_offset, db_offset, q_count, db_count, query_cost, db_cost, query_bias, db_bias,
                out_val, out_idx, stats, compute_stats, collect_linf_stats, stream);
    } else if (k == 32) {
        launch_update_topk_stats_raw<32>(
                dot_ptr, dot_ld, q_offset, db_offset, q_count, db_count, query_cost, db_cost, query_bias, db_bias,
                out_val, out_idx, stats, compute_stats, collect_linf_stats, stream);
    } else {
        TORCH_CHECK(false, "unsupported k");
    }
}

void launch_initialization_topk_raw_for_k(
        int64_t k,
        const float* dot_ptr,
        int64_t dot_ld,
        int64_t q_offset,
        int64_t db_offset,
        int64_t q_count,
        int64_t db_count,
        const double* query_cost,
        const double* db_cost,
        const double* query_bias,
        const double* db_bias,
        double* out_val,
        int32_t* out_idx,
        double* stats,
        cudaStream_t stream) {
    if (k == 16 && db_count <= kInitializationTopk16DbTile) {
        const unsigned int blocks = static_cast<unsigned int>(
                (q_count + kInitializationTopk16Warps - 1) / kInitializationTopk16Warps);
        update_initialization_topk16_warp_queue_kernel<<<blocks, kInitializationTopk16Threads, 0, stream>>>(
                dot_ptr,
                dot_ld,
                q_offset,
                db_offset,
                q_count,
                db_count,
                query_bias,
                db_bias,
                out_val,
                out_idx);
        return;
    }
    // CN: 非 K=16 initialization 配置继续使用既有 directional reduction，不影响 refinement dispatcher。
    // EN: Non-K=16 initialization configurations retain the existing directional reduction without changing the refinement dispatcher.
    launch_update_topk_stats_raw_for_k(
            k,
            dot_ptr,
            dot_ld,
            q_offset,
            db_offset,
            q_count,
            db_count,
            query_cost,
            db_cost,
            query_bias,
            db_bias,
            out_val,
            out_idx,
            stats,
            false,
            false,
            stream);
}

template <int K>
void launch_update_col_partial_topk_raw(
        const float* dot_ptr,
        int64_t dot_ld,
        int64_t q_offset,
        int64_t db_offset,
        int64_t q_count,
        int64_t db_count,
        int64_t n_db,
        int64_t q_tile_id,
        const double* query_bias,
        const double* db_bias,
        double* partial_val,
        int32_t* partial_idx,
        cudaStream_t stream) {
    constexpr int cols = K == 32 ? 64 : kColGroupCols;
    dim3 col_grid(static_cast<unsigned int>((db_count + cols - 1) / cols));
    dim3 col_block(cols, kColGroupRows);
    update_col_partial_topk_raw_grouped_kernel<K, cols><<<col_grid, col_block, 0, stream>>>(
            dot_ptr,
            dot_ld,
            q_offset,
            db_offset,
            q_count,
            db_count,
            n_db,
            q_tile_id,
            query_bias,
            db_bias,
            partial_val,
            partial_idx);
}



void launch_update_col_partial_topk2_raw_warp(
        const float* dot_ptr,
        int64_t dot_ld,
        int64_t q_offset,
        int64_t db_offset,
        int64_t q_count,
        int64_t db_count,
        int64_t n_db,
        int64_t q_tile_id,
        const double* query_bias,
        const double* db_bias,
        double* partial_val,
        int32_t* partial_idx,
        cudaStream_t stream) {
    constexpr int kThreadsPerBlock = 128;
    constexpr int kColsPerWarp = 4;
    constexpr int kWarpsPerBlock = kThreadsPerBlock / 32;
    constexpr int kColsPerBlock = kColsPerWarp * kWarpsPerBlock;
    dim3 grid(static_cast<unsigned int>((db_count + kColsPerBlock - 1) / kColsPerBlock));
    update_col_partial_topk2_raw_warp_kernel<<<grid, kThreadsPerBlock, 0, stream>>>(
            dot_ptr,
            dot_ld,
            q_offset,
            db_offset,
            q_count,
            db_count,
            n_db,
            q_tile_id,
            query_bias,
            db_bias,
            partial_val,
            partial_idx);
}

void launch_update_col_partial_topk4_raw_warp(
        const float* dot_ptr,
        int64_t dot_ld,
        int64_t q_offset,
        int64_t db_offset,
        int64_t q_count,
        int64_t db_count,
        int64_t n_db,
        int64_t q_tile_id,
        const double* query_bias,
        const double* db_bias,
        double* partial_val,
        int32_t* partial_idx,
        cudaStream_t stream) {
    constexpr int kThreadsPerBlock = 128;
    constexpr int kColsPerWarp = 4;
    constexpr int kWarpsPerBlock = kThreadsPerBlock / 32;
    constexpr int kColsPerBlock = kColsPerWarp * kWarpsPerBlock;
    dim3 grid(static_cast<unsigned int>((db_count + kColsPerBlock - 1) / kColsPerBlock));
    update_col_partial_topk4_raw_warp_kernel<<<grid, kThreadsPerBlock, 0, stream>>>(
            dot_ptr,
            dot_ld,
            q_offset,
            db_offset,
            q_count,
            db_count,
            n_db,
            q_tile_id,
            query_bias,
            db_bias,
            partial_val,
            partial_idx);
}

void launch_update_col_partial_topk_raw_for_k(
        int64_t k,
        const float* dot_ptr,
        int64_t dot_ld,
        int64_t q_offset,
        int64_t db_offset,
        int64_t q_count,
        int64_t db_count,
        int64_t n_db,
        int64_t q_tile_id,
        const double* query_bias,
        const double* db_bias,
        double* partial_val,
        int32_t* partial_idx,
        cudaStream_t stream) {
    if (k == 1) {
        launch_update_col_partial_topk_raw<1>(
                dot_ptr, dot_ld, q_offset, db_offset, q_count, db_count, n_db, q_tile_id, query_bias, db_bias,
                partial_val, partial_idx, stream);
    } else if (k == 2) {
        launch_update_col_partial_topk_raw<2>(
                dot_ptr, dot_ld, q_offset, db_offset, q_count, db_count, n_db, q_tile_id, query_bias, db_bias,
                partial_val, partial_idx, stream);
    } else if (k == 4) {
        launch_update_col_partial_topk_raw<4>(
                dot_ptr, dot_ld, q_offset, db_offset, q_count, db_count, n_db, q_tile_id, query_bias, db_bias,
                partial_val, partial_idx, stream);
    } else if (k == 8) {
        launch_update_col_partial_topk_raw<8>(
                dot_ptr, dot_ld, q_offset, db_offset, q_count, db_count, n_db, q_tile_id, query_bias, db_bias,
                partial_val, partial_idx, stream);
    } else if (k == 16) {
        launch_update_col_partial_topk_raw<16>(
                dot_ptr, dot_ld, q_offset, db_offset, q_count, db_count, n_db, q_tile_id, query_bias, db_bias,
                partial_val, partial_idx, stream);
    } else if (k == 32) {
        launch_update_col_partial_topk_raw<32>(
                dot_ptr, dot_ld, q_offset, db_offset, q_count, db_count, n_db, q_tile_id, query_bias, db_bias,
                partial_val, partial_idx, stream);
    } else {
        TORCH_CHECK(false, "unsupported k");
    }
}


void launch_col_partial_load_only_raw(
        const float* dot_ptr,
        int64_t dot_ld,
        int64_t q_offset,
        int64_t db_offset,
        int64_t q_count,
        int64_t db_count,
        int64_t n_db,
        int64_t q_tile_id,
        const double* query_bias,
        const double* db_bias,
        double* sink,
        cudaStream_t stream) {
    dim3 col_grid(static_cast<unsigned int>((db_count + kColGroupCols - 1) / kColGroupCols));
    dim3 col_block(kColGroupCols, kColGroupRows);
    col_partial_load_only_raw_grouped_kernel<<<col_grid, col_block, 0, stream>>>(
            dot_ptr, dot_ld, q_offset, db_offset, q_count, db_count, n_db, q_tile_id, query_bias, db_bias, sink);
}

template <int K>
void launch_col_partial_topk_no_partial_write_raw(
        const float* dot_ptr,
        int64_t dot_ld,
        int64_t q_offset,
        int64_t db_offset,
        int64_t q_count,
        int64_t db_count,
        int64_t n_db,
        int64_t q_tile_id,
        const double* query_bias,
        const double* db_bias,
        double* sink_val,
        int32_t* sink_idx,
        cudaStream_t stream) {
    constexpr int cols = K == 32 ? 64 : kColGroupCols;
    dim3 col_grid(static_cast<unsigned int>((db_count + cols - 1) / cols));
    dim3 col_block(cols, kColGroupRows);
    col_partial_topk_no_partial_write_raw_grouped_kernel<K, cols><<<col_grid, col_block, 0, stream>>>(
            dot_ptr,
            dot_ld,
            q_offset,
            db_offset,
            q_count,
            db_count,
            n_db,
            q_tile_id,
            query_bias,
            db_bias,
            sink_val,
            sink_idx);
}

void launch_col_partial_topk_no_partial_write_raw_for_k(
        int64_t k,
        const float* dot_ptr,
        int64_t dot_ld,
        int64_t q_offset,
        int64_t db_offset,
        int64_t q_count,
        int64_t db_count,
        int64_t n_db,
        int64_t q_tile_id,
        const double* query_bias,
        const double* db_bias,
        double* sink_val,
        int32_t* sink_idx,
        cudaStream_t stream) {
    if (k == 1) {
        launch_col_partial_topk_no_partial_write_raw<1>(
                dot_ptr, dot_ld, q_offset, db_offset, q_count, db_count, n_db, q_tile_id, query_bias, db_bias,
                sink_val, sink_idx, stream);
    } else if (k == 2) {
        launch_col_partial_topk_no_partial_write_raw<2>(
                dot_ptr, dot_ld, q_offset, db_offset, q_count, db_count, n_db, q_tile_id, query_bias, db_bias,
                sink_val, sink_idx, stream);
    } else if (k == 4) {
        launch_col_partial_topk_no_partial_write_raw<4>(
                dot_ptr, dot_ld, q_offset, db_offset, q_count, db_count, n_db, q_tile_id, query_bias, db_bias,
                sink_val, sink_idx, stream);
    } else if (k == 8) {
        launch_col_partial_topk_no_partial_write_raw<8>(
                dot_ptr, dot_ld, q_offset, db_offset, q_count, db_count, n_db, q_tile_id, query_bias, db_bias,
                sink_val, sink_idx, stream);
    } else if (k == 16) {
        launch_col_partial_topk_no_partial_write_raw<16>(
                dot_ptr, dot_ld, q_offset, db_offset, q_count, db_count, n_db, q_tile_id, query_bias, db_bias,
                sink_val, sink_idx, stream);
    } else if (k == 32) {
        launch_col_partial_topk_no_partial_write_raw<32>(
                dot_ptr, dot_ld, q_offset, db_offset, q_count, db_count, n_db, q_tile_id, query_bias, db_bias,
                sink_val, sink_idx, stream);
    } else {
        TORCH_CHECK(false, "unsupported k");
    }
}

template <int K>
void launch_col_partial_write_only_raw(
        int64_t db_offset,
        int64_t db_count,
        int64_t n_db,
        int64_t q_tile_id,
        double* partial_val,
        int32_t* partial_idx,
        cudaStream_t stream) {
    dim3 col_grid(static_cast<unsigned int>((db_count + kColGroupCols - 1) / kColGroupCols));
    dim3 col_block(kColGroupCols, kColGroupRows);
    col_partial_write_only_raw_grouped_kernel<K><<<col_grid, col_block, 0, stream>>>(
            db_offset, db_count, n_db, q_tile_id, partial_val, partial_idx);
}

void launch_col_partial_write_only_raw_for_k(
        int64_t k,
        int64_t db_offset,
        int64_t db_count,
        int64_t n_db,
        int64_t q_tile_id,
        double* partial_val,
        int32_t* partial_idx,
        cudaStream_t stream) {
    if (k == 1) {
        launch_col_partial_write_only_raw<1>(db_offset, db_count, n_db, q_tile_id, partial_val, partial_idx, stream);
    } else if (k == 2) {
        launch_col_partial_write_only_raw<2>(db_offset, db_count, n_db, q_tile_id, partial_val, partial_idx, stream);
    } else if (k == 4) {
        launch_col_partial_write_only_raw<4>(db_offset, db_count, n_db, q_tile_id, partial_val, partial_idx, stream);
    } else if (k == 8) {
        launch_col_partial_write_only_raw<8>(db_offset, db_count, n_db, q_tile_id, partial_val, partial_idx, stream);
    } else if (k == 16) {
        launch_col_partial_write_only_raw<16>(db_offset, db_count, n_db, q_tile_id, partial_val, partial_idx, stream);
    } else if (k == 32) {
        launch_col_partial_write_only_raw<32>(db_offset, db_count, n_db, q_tile_id, partial_val, partial_idx, stream);
    } else {
        TORCH_CHECK(false, "unsupported k");
    }
}

template <int K>
void launch_reduce_col_partials_topk_i32(
        const double* partial_val,
        const int32_t* partial_idx,
        int64_t n_tiles,
        int64_t n_db,
        int reduce_threads,
        double* out_val,
        int32_t* out_idx,
        cudaStream_t stream) {
    constexpr int max_threads = K == 32 ? kTopkThreadsK32 : kTopkThreads;
    const int actual_threads = std::min<int>(reduce_threads, max_threads);
    reduce_col_partials_topk_i32_kernel<K, max_threads><<<static_cast<unsigned int>(n_db), actual_threads, 0, stream>>>(
            partial_val,
            partial_idx,
            n_tiles,
            n_db,
            out_val,
            out_idx);
}

void launch_reduce_col_partials_topk_i32_for_k(
        int64_t k,
        const double* partial_val,
        const int32_t* partial_idx,
        int64_t n_tiles,
        int64_t n_db,
        int reduce_threads,
        double* out_val,
        int32_t* out_idx,
        cudaStream_t stream) {
    if (k == 1) {
        launch_reduce_col_partials_topk_i32<1>(partial_val, partial_idx, n_tiles, n_db, reduce_threads, out_val, out_idx, stream);
    } else if (k == 2) {
        launch_reduce_col_partials_topk_i32<2>(partial_val, partial_idx, n_tiles, n_db, reduce_threads, out_val, out_idx, stream);
    } else if (k == 4) {
        launch_reduce_col_partials_topk_i32<4>(partial_val, partial_idx, n_tiles, n_db, reduce_threads, out_val, out_idx, stream);
    } else if (k == 8) {
        launch_reduce_col_partials_topk_i32<8>(partial_val, partial_idx, n_tiles, n_db, reduce_threads, out_val, out_idx, stream);
    } else if (k == 16) {
        launch_reduce_col_partials_topk_i32<16>(partial_val, partial_idx, n_tiles, n_db, reduce_threads, out_val, out_idx, stream);
    } else if (k == 32) {
        launch_reduce_col_partials_topk_i32<32>(partial_val, partial_idx, n_tiles, n_db, reduce_threads, out_val, out_idx, stream);
    } else {
        TORCH_CHECK(false, "unsupported k");
    }
}

void record_start(cudaEvent_t event, cudaStream_t stream, bool collect_timing) {
    if (collect_timing) {
        C10_CUDA_CHECK(cudaEventRecord(event, stream));
    }
}

void record_stop_accumulate(
        cudaEvent_t start_event,
        cudaEvent_t stop_event,
        cudaStream_t stream,
        bool collect_timing,
        double& out_ms) {
    if (collect_timing) {
        C10_CUDA_CHECK(cudaEventRecord(stop_event, stream));
        C10_CUDA_CHECK(cudaEventSynchronize(stop_event));
        float elapsed_ms = 0.0f;
        C10_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event));
        out_ms += static_cast<double>(elapsed_ms);
    }
}

} // namespace

std::vector<torch::Tensor> directional_topk_raw_cuda_impl(
        torch::Tensor query_feat,
        torch::Tensor db_feat,
        torch::Tensor query_bias,
        torch::Tensor db_bias,
        int64_t query_tile,
        int64_t db_tile,
        int64_t k,
        double dot_scale,
        bool initialization_select) {
    const c10::cuda::CUDAGuard device_guard(query_feat.device());
    auto stream = at::cuda::getCurrentCUDAStream(query_feat.device().index());
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    check_cublas(cublasSetStream(handle, stream.stream()), "cublasSetStream");
    cublasMath_t previous_math_mode;
    check_cublas(cublasGetMathMode(handle, &previous_math_mode), "cublasGetMathMode");
    check_cublas(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH), "cublasSetMathMode");

    const int64_t n_query = query_feat.size(0);
    const int64_t n_db = db_feat.size(0);
    const int64_t dim = query_feat.size(1);
    const int64_t q_tile = std::min<int64_t>(query_tile, n_query);
    // CN: 只有 initialization K=16 使用专用 selection tile；refinement 保留原 dispatcher 与 tile。
    // EN: Only initialization K=16 uses the specialized selection tile; refinement retains its original dispatcher and tile.
    const int64_t requested_d_tile = initialization_select && k == 16
            ? std::min<int64_t>(db_tile, kInitializationTopk16DbTile)
            : db_tile;
    const int64_t d_tile = std::min<int64_t>(requested_d_tile, n_db);
    auto float_opts = query_feat.options().dtype(torch::kFloat32);
    auto double_opts = query_feat.options().dtype(torch::kFloat64);
    auto int_opts = query_feat.options().dtype(torch::kInt32);

    torch::Tensor values = torch::full(
            {n_query, k}, -std::numeric_limits<double>::infinity(), double_opts);
    torch::Tensor indices = torch::full({n_query, k}, -1, int_opts);
    torch::Tensor dots = torch::empty({q_tile, d_tile}, float_opts);
    torch::Tensor query_cost;
    torch::Tensor db_cost;
    torch::Tensor stats;
    // CN: refinement directional 路径保留原 generic kernel 的零 cost/stat 输入；
    //     initialization K=16 专用 kernel 不为该旧契约分配无用 tensor。
    // EN: The refinement directional path retains the generic kernel's original zero cost/stat inputs;
    //     the initialization-only K=16 kernel avoids allocating tensors required only by that legacy contract.
    if (!initialization_select || k != 16) {
        query_cost = torch::zeros({n_query}, double_opts);
        db_cost = torch::zeros({n_db}, double_opts);
        stats = torch::zeros({4}, double_opts);
    }
    const float alpha = static_cast<float>(dot_scale);
    const float beta = 0.0f;
    for (int64_t q0 = 0; q0 < n_query; q0 += q_tile) {
        const int64_t cur_q = std::min<int64_t>(q_tile, n_query - q0);
        const float* q_ptr = query_feat.data_ptr<float>() + q0 * dim;
        for (int64_t d0 = 0; d0 < n_db; d0 += d_tile) {
            const int64_t cur_db = std::min<int64_t>(d_tile, n_db - d0);
            const float* db_ptr = db_feat.data_ptr<float>() + d0 * dim;
            float* dot_ptr = dots.data_ptr<float>();
            check_cublas(
                    cublasSgemm(
                            handle,
                            CUBLAS_OP_T,
                            CUBLAS_OP_N,
                            static_cast<int>(cur_db),
                            static_cast<int>(cur_q),
                            static_cast<int>(dim),
                            &alpha,
                            db_ptr,
                            static_cast<int>(dim),
                            q_ptr,
                            static_cast<int>(dim),
                            &beta,
                            dot_ptr,
                            static_cast<int>(cur_db)),
                    "cublasSgemm");
            if (initialization_select) {
                launch_initialization_topk_raw_for_k(
                        k,
                        dot_ptr,
                        cur_db,
                        q0,
                        d0,
                        cur_q,
                        cur_db,
                        query_cost.defined() ? query_cost.data_ptr<double>() : nullptr,
                        db_cost.defined() ? db_cost.data_ptr<double>() : nullptr,
                        query_bias.data_ptr<double>(),
                        db_bias.data_ptr<double>(),
                        values.data_ptr<double>(),
                        indices.data_ptr<int32_t>(),
                        stats.defined() ? stats.data_ptr<double>() : nullptr,
                        stream.stream());
            } else {
                launch_update_topk_stats_raw_for_k(
                        k,
                        dot_ptr,
                        cur_db,
                        q0,
                        d0,
                        cur_q,
                        cur_db,
                        query_cost.data_ptr<double>(),
                        db_cost.data_ptr<double>(),
                        query_bias.data_ptr<double>(),
                        db_bias.data_ptr<double>(),
                        values.data_ptr<double>(),
                        indices.data_ptr<int32_t>(),
                        stats.data_ptr<double>(),
                        false,
                        false,
                        stream.stream());
            }
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    }
    check_cublas(cublasSetMathMode(handle, previous_math_mode), "cublasSetMathMode restore");
    return {values, indices};
}

std::vector<torch::Tensor> fused_directional_topk_raw_cuda(
        torch::Tensor query_feat,
        torch::Tensor db_feat,
        torch::Tensor query_bias,
        torch::Tensor db_bias,
        int64_t query_tile,
        int64_t db_tile,
        int64_t k,
        double dot_scale) {
    return directional_topk_raw_cuda_impl(
            query_feat, db_feat, query_bias, db_bias, query_tile, db_tile, k, dot_scale, false);
}

std::vector<torch::Tensor> initialization_directional_topk_raw_cuda(
        torch::Tensor query_feat,
        torch::Tensor db_feat,
        torch::Tensor query_bias,
        torch::Tensor db_bias,
        int64_t query_tile,
        int64_t db_tile,
        int64_t k,
        double dot_scale) {
    return directional_topk_raw_cuda_impl(
            query_feat, db_feat, query_bias, db_bias, query_tile, db_tile, k, dot_scale, true);
}

std::vector<torch::Tensor> fused_bidir_topk_stats_raw_cuda(
        torch::Tensor query_feat,
        torch::Tensor db_feat,
        torch::Tensor query_cost,
        torch::Tensor db_cost,
        torch::Tensor query_bias,
        torch::Tensor db_bias,
        int64_t query_tile,
        int64_t db_tile,
        int64_t col_reduce_tile,
        int64_t k,
        bool compute_stats,
        bool collect_linf_stats,
        bool collect_timing,
        int64_t col_kernel,
        double dot_scale) {
    const c10::cuda::CUDAGuard device_guard(query_feat.device());
    auto stream = at::cuda::getCurrentCUDAStream(query_feat.device().index());
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    check_cublas(cublasSetStream(handle, stream.stream()), "cublasSetStream");
    cublasMath_t previous_math_mode;
    check_cublas(cublasGetMathMode(handle, &previous_math_mode), "cublasGetMathMode");
    check_cublas(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH), "cublasSetMathMode");

    const int64_t n_query = query_feat.size(0);
    const int64_t n_db = db_feat.size(0);
    const int64_t dim = query_feat.size(1);
    const int64_t q_tile = std::min<int64_t>(query_tile, n_query);
    const int64_t d_tile = std::min<int64_t>(db_tile, n_db);
    const int64_t n_q_tiles = (n_query + q_tile - 1) / q_tile;
    const int reduce_threads = reduce_threads_from_tile(col_reduce_tile);

    auto float_opts = query_feat.options().dtype(torch::kFloat32);
    auto int_opts = query_feat.options().dtype(torch::kInt32);
    auto double_opts = query_feat.options().dtype(torch::kFloat64);

    torch::Tensor s2t_val = torch::empty({n_query, k}, double_opts);
    torch::Tensor s2t_idx = torch::empty({n_query, k}, int_opts);
    torch::Tensor t2s_val = torch::empty({n_db, k}, double_opts);
    torch::Tensor t2s_idx = torch::empty({n_db, k}, int_opts);
    s2t_val.fill_(-std::numeric_limits<double>::infinity());
    s2t_idx.fill_(-1);

    torch::Tensor dots = torch::empty({q_tile, d_tile}, float_opts);
    torch::Tensor partial_t2s_val = torch::empty({n_q_tiles, n_db, k}, double_opts);
    torch::Tensor partial_t2s_idx = torch::empty({n_q_tiles, n_db, k}, int_opts);
    torch::Tensor stats = torch::empty({4}, double_opts);
    init_stats_kernel<<<1, 1, 0, stream.stream()>>>(stats.data_ptr<double>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    std::array<double, 4> phase_ms = {0.0, 0.0, 0.0, 0.0};
    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;
    if (collect_timing) {
        C10_CUDA_CHECK(cudaEventCreate(&start_event));
        C10_CUDA_CHECK(cudaEventCreate(&stop_event));
    }

    const float alpha = static_cast<float>(dot_scale);
    const float beta = 0.0f;
    int64_t q_tile_id = 0;
    for (int64_t q0 = 0; q0 < n_query; q0 += q_tile, ++q_tile_id) {
        int64_t cur_q = std::min<int64_t>(q_tile, n_query - q0);
        const float* q_ptr = query_feat.data_ptr<float>() + q0 * dim;
        for (int64_t d0 = 0; d0 < n_db; d0 += d_tile) {
            int64_t cur_db = std::min<int64_t>(d_tile, n_db - d0);
            const float* db_ptr = db_feat.data_ptr<float>() + d0 * dim;
            float* dot_ptr = dots.data_ptr<float>();

            // CN: row-major dot(q, db) 被当成 column-major dot^T(db, q) 写入。
            // EN: Row-major dot(q, db) is written as column-major dot^T(db, q).
            record_start(start_event, stream.stream(), collect_timing);
            check_cublas(
                    cublasSgemm(
                            handle,
                            CUBLAS_OP_T,
                            CUBLAS_OP_N,
                            static_cast<int>(cur_db),
                            static_cast<int>(cur_q),
                            static_cast<int>(dim),
                            &alpha,
                            db_ptr,
                            static_cast<int>(dim),
                            q_ptr,
                            static_cast<int>(dim),
                            &beta,
                            dot_ptr,
                            static_cast<int>(cur_db)),
                    "cublasSgemm");
            record_stop_accumulate(start_event, stop_event, stream.stream(), collect_timing, phase_ms[0]);

            record_start(start_event, stream.stream(), collect_timing);
            launch_update_topk_stats_raw_for_k(
                    k,
                    dot_ptr,
                    cur_db,
                    q0,
                    d0,
                    cur_q,
                    cur_db,
                    query_cost.data_ptr<double>(),
                    db_cost.data_ptr<double>(),
                    query_bias.data_ptr<double>(),
                    db_bias.data_ptr<double>(),
                    s2t_val.data_ptr<double>(),
                    s2t_idx.data_ptr<int32_t>(),
                    stats.data_ptr<double>(),
                    compute_stats,
                    collect_linf_stats,
                    stream.stream());
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            record_stop_accumulate(start_event, stop_event, stream.stream(), collect_timing, phase_ms[1]);

            record_start(start_event, stream.stream(), collect_timing);
            if (col_kernel == 2 && k == 2) {
                launch_update_col_partial_topk2_raw_warp(
                        dot_ptr,
                        cur_db,
                        q0,
                        d0,
                        cur_q,
                        cur_db,
                        n_db,
                        q_tile_id,
                        query_bias.data_ptr<double>(),
                        db_bias.data_ptr<double>(),
                        partial_t2s_val.data_ptr<double>(),
                        partial_t2s_idx.data_ptr<int32_t>(),
                        stream.stream());
            } else if (col_kernel == 1 && k == 4) {
                launch_update_col_partial_topk4_raw_warp(
                        dot_ptr,
                        cur_db,
                        q0,
                        d0,
                        cur_q,
                        cur_db,
                        n_db,
                        q_tile_id,
                        query_bias.data_ptr<double>(),
                        db_bias.data_ptr<double>(),
                        partial_t2s_val.data_ptr<double>(),
                        partial_t2s_idx.data_ptr<int32_t>(),
                        stream.stream());
            } else {
                launch_update_col_partial_topk_raw_for_k(
                        k,
                        dot_ptr,
                        cur_db,
                        q0,
                        d0,
                        cur_q,
                        cur_db,
                        n_db,
                        q_tile_id,
                        query_bias.data_ptr<double>(),
                        db_bias.data_ptr<double>(),
                        partial_t2s_val.data_ptr<double>(),
                        partial_t2s_idx.data_ptr<int32_t>(),
                        stream.stream());
            }
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            record_stop_accumulate(start_event, stop_event, stream.stream(), collect_timing, phase_ms[2]);
        }
    }

    record_start(start_event, stream.stream(), collect_timing);
    launch_reduce_col_partials_topk_i32_for_k(
            k,
            partial_t2s_val.data_ptr<double>(),
            partial_t2s_idx.data_ptr<int32_t>(),
            n_q_tiles,
            n_db,
            reduce_threads,
            t2s_val.data_ptr<double>(),
            t2s_idx.data_ptr<int32_t>(),
            stream.stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    record_stop_accumulate(start_event, stop_event, stream.stream(), collect_timing, phase_ms[3]);

    if (collect_timing) {
        C10_CUDA_CHECK(cudaEventDestroy(start_event));
        C10_CUDA_CHECK(cudaEventDestroy(stop_event));
    }

    torch::Tensor phase_tensor = torch::empty({4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto phase_acc = phase_tensor.accessor<double, 1>();
    for (int idx = 0; idx < 4; ++idx) {
        phase_acc[idx] = phase_ms[idx];
    }

    check_cublas(cublasSetMathMode(handle, previous_math_mode), "cublasSetMathMode restore");
    return {s2t_val, s2t_idx, t2s_val, t2s_idx, stats, phase_tensor};
}


torch::Tensor col_partial_diagnostics_raw_cuda(
        torch::Tensor query_feat,
        torch::Tensor db_feat,
        torch::Tensor query_bias,
        torch::Tensor db_bias,
        int64_t query_tile,
        int64_t db_tile,
        int64_t k) {
    const c10::cuda::CUDAGuard device_guard(query_feat.device());
    auto stream = at::cuda::getCurrentCUDAStream(query_feat.device().index());
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    check_cublas(cublasSetStream(handle, stream.stream()), "cublasSetStream");

    const int64_t n_query = query_feat.size(0);
    const int64_t n_db = db_feat.size(0);
    const int64_t dim = query_feat.size(1);
    const int64_t q_tile = std::min<int64_t>(query_tile, n_query);
    const int64_t d_tile = std::min<int64_t>(db_tile, n_db);
    const int64_t n_q_tiles = (n_query + q_tile - 1) / q_tile;

    auto float_opts = query_feat.options().dtype(torch::kFloat32);
    auto double_opts = query_feat.options().dtype(torch::kFloat64);
    auto int_opts = query_feat.options().dtype(torch::kInt32);
    torch::Tensor dots = torch::empty({q_tile, d_tile}, float_opts);
    torch::Tensor sink_val = torch::empty({n_q_tiles, n_db}, double_opts);
    torch::Tensor sink_idx = torch::empty({n_q_tiles, n_db}, int_opts);
    torch::Tensor partial_val = torch::empty({n_q_tiles, n_db, k}, double_opts);
    torch::Tensor partial_idx = torch::empty({n_q_tiles, n_db, k}, int_opts);

    std::array<double, 5> phase_ms = {0.0, 0.0, 0.0, 0.0, 0.0};
    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;
    C10_CUDA_CHECK(cudaEventCreate(&start_event));
    C10_CUDA_CHECK(cudaEventCreate(&stop_event));

    const float alpha = 1.0f;
    const float beta = 0.0f;
    int64_t q_tile_id = 0;
    for (int64_t q0 = 0; q0 < n_query; q0 += q_tile, ++q_tile_id) {
        int64_t cur_q = std::min<int64_t>(q_tile, n_query - q0);
        const float* q_ptr = query_feat.data_ptr<float>() + q0 * dim;
        for (int64_t d0 = 0; d0 < n_db; d0 += d_tile) {
            int64_t cur_db = std::min<int64_t>(d_tile, n_db - d0);
            const float* db_ptr = db_feat.data_ptr<float>() + d0 * dim;
            float* dot_ptr = dots.data_ptr<float>();

            record_start(start_event, stream.stream(), true);
            check_cublas(
                    cublasSgemm(
                            handle,
                            CUBLAS_OP_T,
                            CUBLAS_OP_N,
                            static_cast<int>(cur_db),
                            static_cast<int>(cur_q),
                            static_cast<int>(dim),
                            &alpha,
                            db_ptr,
                            static_cast<int>(dim),
                            q_ptr,
                            static_cast<int>(dim),
                            &beta,
                            dot_ptr,
                            static_cast<int>(cur_db)),
                    "cublasSgemm");
            record_stop_accumulate(start_event, stop_event, stream.stream(), true, phase_ms[0]);

            record_start(start_event, stream.stream(), true);
            launch_col_partial_load_only_raw(
                    dot_ptr,
                    cur_db,
                    q0,
                    d0,
                    cur_q,
                    cur_db,
                    n_db,
                    q_tile_id,
                    query_bias.data_ptr<double>(),
                    db_bias.data_ptr<double>(),
                    sink_val.data_ptr<double>(),
                    stream.stream());
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            record_stop_accumulate(start_event, stop_event, stream.stream(), true, phase_ms[1]);

            record_start(start_event, stream.stream(), true);
            launch_col_partial_topk_no_partial_write_raw_for_k(
                    k,
                    dot_ptr,
                    cur_db,
                    q0,
                    d0,
                    cur_q,
                    cur_db,
                    n_db,
                    q_tile_id,
                    query_bias.data_ptr<double>(),
                    db_bias.data_ptr<double>(),
                    sink_val.data_ptr<double>(),
                    sink_idx.data_ptr<int32_t>(),
                    stream.stream());
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            record_stop_accumulate(start_event, stop_event, stream.stream(), true, phase_ms[2]);

            record_start(start_event, stream.stream(), true);
            launch_col_partial_write_only_raw_for_k(
                    k,
                    d0,
                    cur_db,
                    n_db,
                    q_tile_id,
                    partial_val.data_ptr<double>(),
                    partial_idx.data_ptr<int32_t>(),
                    stream.stream());
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            record_stop_accumulate(start_event, stop_event, stream.stream(), true, phase_ms[3]);

            record_start(start_event, stream.stream(), true);
            launch_update_col_partial_topk_raw_for_k(
                    k,
                    dot_ptr,
                    cur_db,
                    q0,
                    d0,
                    cur_q,
                    cur_db,
                    n_db,
                    q_tile_id,
                    query_bias.data_ptr<double>(),
                    db_bias.data_ptr<double>(),
                    partial_val.data_ptr<double>(),
                    partial_idx.data_ptr<int32_t>(),
                    stream.stream());
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            record_stop_accumulate(start_event, stop_event, stream.stream(), true, phase_ms[4]);
        }
    }

    C10_CUDA_CHECK(cudaEventDestroy(start_event));
    C10_CUDA_CHECK(cudaEventDestroy(stop_event));

    torch::Tensor phase_tensor = torch::empty({5}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto phase_acc = phase_tensor.accessor<double, 1>();
    for (int idx = 0; idx < 5; ++idx) {
        phase_acc[idx] = phase_ms[idx];
    }
    return phase_tensor;
}
