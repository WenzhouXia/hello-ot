#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>
#include <torch/extension.h>

#include <cstdint>
#include <limits>
#include <vector>

namespace {

constexpr int kThreads = 256;

__global__ void init_stats_kernel(double* stats) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int idx = 0; idx < 5; ++idx) {
            stats[idx] = 0.0;
        }
    }
}

__device__ double atomic_max_nonnegative_double(double* address, double value) {
    auto* bits = reinterpret_cast<unsigned long long*>(address);
    unsigned long long old = *bits;
    while (__longlong_as_double(old) < value) {
        const unsigned long long assumed = old;
        old = atomicCAS(bits, assumed, __double_as_longlong(value));
        if (old == assumed) {
            break;
        }
    }
    return __longlong_as_double(old);
}

__global__ void update_top1_stats_kernel(
        const float* __restrict__ scores,
        int64_t score_ld,
        int64_t q_offset,
        int64_t db_offset,
        int64_t q_count,
        int64_t db_count,
        const float* __restrict__ query_dual,
        const float* __restrict__ db_dual,
        float* __restrict__ out_val,
        int64_t* __restrict__ out_idx,
        double* __restrict__ stats,
        bool compute_stats) {
    int row = blockIdx.x;
    if (row >= q_count) {
        return;
    }

    __shared__ float best_vals[kThreads];
    __shared__ int best_idxs[kThreads];
    __shared__ double num_vals[kThreads];
    __shared__ double den_vals[kThreads];
    __shared__ unsigned long long pos_vals[kThreads];

    float best = -3.4028234663852886e38F;
    int best_idx = -1;
    double num_sum = 0.0;
    double den_sum = 0.0;
    unsigned long long pos_count = 0;
    const float qd = query_dual[q_offset + row];

    for (int64_t col = threadIdx.x; col < db_count; col += blockDim.x) {
        float rc = scores[row * score_ld + col];
        if (rc > best) {
            best = rc;
            best_idx = static_cast<int>(col);
        }
        if (compute_stats) {
            float cost = qd + db_dual[db_offset + col] - rc;
            double cost_d = static_cast<double>(cost);
            den_sum += cost_d * cost_d;
            if (rc > 0.0f) {
                double rc_d = static_cast<double>(rc);
                num_sum += rc_d * rc_d;
                ++pos_count;
            }
        }
    }

    best_vals[threadIdx.x] = best;
    best_idxs[threadIdx.x] = best_idx;
    num_vals[threadIdx.x] = num_sum;
    den_vals[threadIdx.x] = den_sum;
    pos_vals[threadIdx.x] = pos_count;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            float other = best_vals[threadIdx.x + stride];
            int other_idx = best_idxs[threadIdx.x + stride];
            if (other > best_vals[threadIdx.x] ||
                (other == best_vals[threadIdx.x] && other_idx >= 0 &&
                 (best_idxs[threadIdx.x] < 0 || other_idx < best_idxs[threadIdx.x]))) {
                best_vals[threadIdx.x] = other;
                best_idxs[threadIdx.x] = other_idx;
            }
            num_vals[threadIdx.x] += num_vals[threadIdx.x + stride];
            den_vals[threadIdx.x] += den_vals[threadIdx.x + stride];
            pos_vals[threadIdx.x] += pos_vals[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        int64_t global_row = q_offset + row;
        float prev = out_val[global_row];
        int64_t candidate_idx = db_offset + static_cast<int64_t>(best_idxs[0]);
        if (best_vals[0] > prev ||
            (best_vals[0] == prev && best_idxs[0] >= 0 &&
             (out_idx[global_row] < 0 || candidate_idx < out_idx[global_row]))) {
            out_val[global_row] = best_vals[0];
            out_idx[global_row] = candidate_idx;
        }
        if (compute_stats) {
            atomicAdd(stats + 0, num_vals[0]);
            atomicAdd(stats + 1, den_vals[0]);
            atomicAdd(stats + 2, static_cast<double>(pos_vals[0]));
        }
    }
}

__global__ void update_stats_only_kernel(
        const float* __restrict__ scores,
        int64_t score_ld,
        int64_t q_offset,
        int64_t db_offset,
        int64_t q_count,
        int64_t db_count,
        const float* __restrict__ query_dual,
        const float* __restrict__ db_dual,
        double* __restrict__ stats,
        int64_t total_count) {
    __shared__ double num_vals[kThreads];
    __shared__ double den_vals[kThreads];
    __shared__ unsigned long long pos_vals[kThreads];

    double num_sum = 0.0;
    double den_sum = 0.0;
    unsigned long long pos_count = 0;
    int64_t thread = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    for (int64_t idx = thread; idx < total_count; idx += stride) {
        int64_t row = idx / db_count;
        int64_t col = idx - row * db_count;
        float rc = scores[row * score_ld + col];
        float cost = query_dual[q_offset + row] + db_dual[db_offset + col] - rc;
        double cost_d = static_cast<double>(cost);
        den_sum += cost_d * cost_d;
        if (rc > 0.0f) {
            double rc_d = static_cast<double>(rc);
            num_sum += rc_d * rc_d;
            ++pos_count;
        }
    }

    num_vals[threadIdx.x] = num_sum;
    den_vals[threadIdx.x] = den_sum;
    pos_vals[threadIdx.x] = pos_count;
    __syncthreads();

    for (int stride_reduce = blockDim.x / 2; stride_reduce > 0; stride_reduce >>= 1) {
        if (threadIdx.x < stride_reduce) {
            num_vals[threadIdx.x] += num_vals[threadIdx.x + stride_reduce];
            den_vals[threadIdx.x] += den_vals[threadIdx.x + stride_reduce];
            pos_vals[threadIdx.x] += pos_vals[threadIdx.x + stride_reduce];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(stats + 0, num_vals[0]);
        atomicAdd(stats + 1, den_vals[0]);
        atomicAdd(stats + 2, static_cast<double>(pos_vals[0]));
    }
}


__global__ void update_raw_lowrank_stats_only_kernel(
        const float* __restrict__ scores,
        int64_t score_ld,
        int64_t q_offset,
        int64_t db_offset,
        int64_t db_count,
        const double* __restrict__ query_cost,
        const double* __restrict__ db_cost,
        const double* __restrict__ query_dual,
        const double* __restrict__ db_dual,
        double* __restrict__ stats,
        int64_t total_count) {
    __shared__ double num_vals[kThreads];
    __shared__ double den_vals[kThreads];
    __shared__ unsigned long long pos_vals[kThreads];
    __shared__ double max_violation_vals[kThreads];
    __shared__ double cost_linf_vals[kThreads];

    double num_sum = 0.0;
    double den_sum = 0.0;
    unsigned long long pos_count = 0;
    double max_violation = 0.0;
    double cost_linf = 0.0;
    int64_t thread = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    for (int64_t idx = thread; idx < total_count; idx += stride) {
        int64_t row = idx / db_count;
        int64_t col = idx - row * db_count;
        float dot = scores[row * score_ld + col];
        double cost_d = query_cost[q_offset + row] + db_cost[db_offset + col] - static_cast<double>(dot);
        double reduced_cost = query_dual[q_offset + row] + db_dual[db_offset + col] - cost_d;
        den_sum += cost_d * cost_d;
        cost_linf = fmax(cost_linf, fabs(cost_d));
        if (reduced_cost > 0.0) {
            num_sum += reduced_cost * reduced_cost;
            ++pos_count;
            max_violation = fmax(max_violation, reduced_cost);
        }
    }

    num_vals[threadIdx.x] = num_sum;
    den_vals[threadIdx.x] = den_sum;
    pos_vals[threadIdx.x] = pos_count;
    max_violation_vals[threadIdx.x] = max_violation;
    cost_linf_vals[threadIdx.x] = cost_linf;
    __syncthreads();

    for (int stride_reduce = blockDim.x / 2; stride_reduce > 0; stride_reduce >>= 1) {
        if (threadIdx.x < stride_reduce) {
            num_vals[threadIdx.x] += num_vals[threadIdx.x + stride_reduce];
            den_vals[threadIdx.x] += den_vals[threadIdx.x + stride_reduce];
            pos_vals[threadIdx.x] += pos_vals[threadIdx.x + stride_reduce];
            max_violation_vals[threadIdx.x] = fmax(
                    max_violation_vals[threadIdx.x], max_violation_vals[threadIdx.x + stride_reduce]);
            cost_linf_vals[threadIdx.x] = fmax(
                    cost_linf_vals[threadIdx.x], cost_linf_vals[threadIdx.x + stride_reduce]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(stats + 0, num_vals[0]);
        atomicAdd(stats + 1, den_vals[0]);
        atomicAdd(stats + 2, static_cast<double>(pos_vals[0]));
        atomic_max_nonnegative_double(stats + 3, max_violation_vals[0]);
        atomic_max_nonnegative_double(stats + 4, cost_linf_vals[0]);
    }
}

void check_cublas(cublasStatus_t status, const char* what) {
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, what, " failed with cuBLAS status ", static_cast<int>(status));
}

} // namespace

torch::Tensor fused_stats_only_cuda(
        torch::Tensor query_aug,
        torch::Tensor db_aug,
        torch::Tensor query_dual,
        torch::Tensor db_dual,
        int64_t query_tile,
        int64_t db_tile) {
    const c10::cuda::CUDAGuard device_guard(query_aug.device());
    auto stream = at::cuda::getCurrentCUDAStream(query_aug.device().index());
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    check_cublas(cublasSetStream(handle, stream.stream()), "cublasSetStream");

    const int64_t n_query = query_aug.size(0);
    const int64_t n_db = db_aug.size(0);
    const int64_t dim = query_aug.size(1);
    const int64_t q_tile = std::min<int64_t>(query_tile, n_query);
    const int64_t d_tile = std::min<int64_t>(db_tile, n_db);

    auto float_opts = query_aug.options().dtype(torch::kFloat32);
    auto double_opts = query_aug.options().dtype(torch::kFloat64);

    torch::Tensor scores = torch::empty({q_tile, d_tile}, float_opts);
    torch::Tensor stats = torch::empty({5}, double_opts);
    init_stats_kernel<<<1, 1, 0, stream.stream()>>>(stats.data_ptr<double>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const float alpha = 1.0f;
    const float beta = 0.0f;
    for (int64_t q0 = 0; q0 < n_query; q0 += q_tile) {
        int64_t cur_q = std::min<int64_t>(q_tile, n_query - q0);
        const float* q_ptr = query_aug.data_ptr<float>() + q0 * dim;
        for (int64_t d0 = 0; d0 < n_db; d0 += d_tile) {
            int64_t cur_db = std::min<int64_t>(d_tile, n_db - d0);
            const float* db_ptr = db_aug.data_ptr<float>() + d0 * dim;
            float* score_ptr = scores.data_ptr<float>();
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
                            score_ptr,
                            static_cast<int>(cur_db)),
                    "cublasSgemm");

            int64_t total_count = cur_q * cur_db;
            int64_t blocks = std::min<int64_t>(1024, (total_count + kThreads - 1) / kThreads);
            update_stats_only_kernel<<<static_cast<unsigned int>(blocks), kThreads, 0, stream.stream()>>>(
                    score_ptr,
                    cur_db,
                    q0,
                    d0,
                    cur_q,
                    cur_db,
                    query_dual.data_ptr<float>(),
                    db_dual.data_ptr<float>(),
                    stats.data_ptr<double>(),
                    total_count);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    }
    return stats;
}

torch::Tensor raw_lowrank_stats_only_cuda(
        torch::Tensor query_feat,
        torch::Tensor db_feat,
        torch::Tensor query_cost,
        torch::Tensor db_cost,
        torch::Tensor query_dual,
        torch::Tensor db_dual,
        int64_t query_tile,
        int64_t db_tile,
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

    auto float_opts = query_feat.options().dtype(torch::kFloat32);
    auto double_opts = query_feat.options().dtype(torch::kFloat64);

    torch::Tensor scores = torch::empty({q_tile, d_tile}, float_opts);
    torch::Tensor stats = torch::empty({5}, double_opts);
    init_stats_kernel<<<1, 1, 0, stream.stream()>>>(stats.data_ptr<double>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const float alpha = static_cast<float>(dot_scale);
    const float beta = 0.0f;
    for (int64_t q0 = 0; q0 < n_query; q0 += q_tile) {
        int64_t cur_q = std::min<int64_t>(q_tile, n_query - q0);
        const float* q_ptr = query_feat.data_ptr<float>() + q0 * dim;
        for (int64_t d0 = 0; d0 < n_db; d0 += d_tile) {
            int64_t cur_db = std::min<int64_t>(d_tile, n_db - d0);
            const float* db_ptr = db_feat.data_ptr<float>() + d0 * dim;
            float* score_ptr = scores.data_ptr<float>();
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
                            score_ptr,
                            static_cast<int>(cur_db)),
                    "cublasSgemm");

            int64_t total_count = cur_q * cur_db;
            int64_t blocks = std::min<int64_t>(1024, (total_count + kThreads - 1) / kThreads);
            update_raw_lowrank_stats_only_kernel<<<static_cast<unsigned int>(blocks), kThreads, 0, stream.stream()>>>(
                    score_ptr,
                    cur_db,
                    q0,
                    d0,
                    cur_db,
                    query_cost.data_ptr<double>(),
                    db_cost.data_ptr<double>(),
                    query_dual.data_ptr<double>(),
                    db_dual.data_ptr<double>(),
                    stats.data_ptr<double>(),
                    total_count);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    }
    check_cublas(cublasSetMathMode(handle, previous_math_mode), "cublasSetMathMode restore");
    return stats;
}
