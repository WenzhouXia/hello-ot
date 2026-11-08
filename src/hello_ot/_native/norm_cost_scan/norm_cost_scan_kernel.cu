#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cstdint>
#include <limits>
#include <utility>
#include <type_traits>
#include <vector>

namespace {

enum CostKind { kL1 = 0, kLinf = 1, kL2 = 2 };
enum PerturbMode { kRank2 = 0, kIndexHash = 1 };

// CN: index_hash 噪声的常数与系数；系数在 host 端用 numpy 的
//     _index_hash_parameters(seed) 生成（与 KeOps 路径完全一致）。
// EN: Constants and per-seed coefficients for index_hash noise; coefficients are
//     generated host-side with _index_hash_parameters(seed), identical to KeOps.
constexpr int64_t kHashPrime0 = 2003;
constexpr int64_t kHashPrime1 = 2011;
constexpr float kHashModulusF = 4028033.0f;
// CN: KeOps 把“除以 IntCst”改写成“乘以 RatCst((float)(1/prime))”的 fp32 常数，
//     复刻噪声时必须用同一常数才能逐位一致。
// EN: KeOps rewrites division by an IntCst as a multiply by the fp32 constant
//     (float)(1/prime); bitwise replication needs the same constants.
constexpr float kRecipHash0 = (float)(1.0 / 2003.0);
constexpr float kRecipHash1 = (float)(1.0 / 2011.0);

struct NoiseCoeffs {
    float sc0, tc0, off0;
    float sc1, tc1, off1;
};

// CN: 扰动的运行参数；sigma == 0 时关闭扰动，seed_term 为
//     (float)((double)seed * 0.001)，与 KeOps 的 fp32 标量嵌入一致。
// EN: Runtime perturbation parameters; sigma == 0 disables it. seed_term is
//     (float)((double)seed * 0.001), matching KeOps fp32 scalar embedding.
struct NoiseParams {
    int perturb_mode;
    double sigma;
    float seed_term;
    bool row_is_source;
    NoiseCoeffs coeffs;
};

__device__ __forceinline__ double sigma_noise(
        const NoiseParams& params,
        int64_t q_idx,
        int64_t d_idx) {
    if (params.perturb_mode == kRank2) {
        float src = static_cast<float>(q_idx);
        float tgt = static_cast<float>(d_idx);
        if (!params.row_is_source) {
            const float tmp = src;
            src = tgt;
            tgt = tmp;
        }
        // CN: 复刻 KeOps（pykeops 默认 --use_fast_math 编译）：
        //     z 的乘加被 NVRTC 收缩为 fmaf(tgt, 78.233, src*12.9898)，
        //     sin 解析为 __sinf，两者都是确定性硬件指令，可逐位一致。
        // EN: Replicate KeOps compiled with pykeops' default --use_fast_math:
        //     NVRTC contracts z's mul-add into fmaf(tgt, 78.233, src*12.9898)
        //     and sin resolves to __sinf; both are deterministic HW ops.
        const float z = __fadd_rn(fmaf(tgt, 78.233f, __fmul_rn(src, 12.9898f)), params.seed_term);
        return __fmul_rn(params.sigma, __fadd_rn(__fmul_rn(__sinf(z), 0.5f), 0.5f));
    }
    int64_t s0 = q_idx % kHashPrime0;
    int64_t t0 = d_idx % kHashPrime0;
    int64_t s1 = q_idx % kHashPrime1;
    int64_t t1 = d_idx % kHashPrime1;
    if (!params.row_is_source) {
        const int64_t tmp0 = s0;
        s0 = t0;
        t0 = tmp0;
        const int64_t tmp1 = s1;
        s1 = t1;
            t1 = tmp1;
        }
    // CN: 与 NumPy/Torch 共用 FP32 近似取模；显式舍入禁止 FMA 改变顺序。
    // EN: Share FP32 approximate remainders with NumPy/Torch; explicit rounding prevents FMA reordering.
    const float s0f = static_cast<float>(s0), t0f = static_cast<float>(t0);
    const float s1f = static_cast<float>(s1), t1f = static_cast<float>(t1);
    const float x0 = __fadd_rn(__fadd_rn(__fadd_rn(__fmul_rn(params.coeffs.sc0, s0f), __fmul_rn(params.coeffs.tc0, t0f)), __fmul_rn(s0f, t0f)), params.coeffs.off0);
    const float x1 = __fadd_rn(__fadd_rn(__fadd_rn(__fmul_rn(params.coeffs.sc1, s1f), __fmul_rn(params.coeffs.tc1, t1f)), __fmul_rn(s1f, t1f)), params.coeffs.off1);
    const float h0 = __fsub_rn(x0, __fmul_rn(2003.0f, floorf(__fmul_rn(kRecipHash0, x0))));
    const float h1 = __fsub_rn(x1, __fmul_rn(2011.0f, floorf(__fmul_rn(kRecipHash1, x1))));
    const float value = __fmul_rn(__fadd_rn(__fadd_rn(h0, __fmul_rn(2003.0f, h1)), 0.5f), (float)(1.0 / 4028033.0));
    return __dmul_rn(params.sigma, static_cast<double>(value));
}

// CN: 单个维度对距离的贡献；l2 只累加平方和，开方在归约之后做。
// EN: Per-dimension distance contribution; l2 accumulates squared sums and takes
//     the square root after the reduction.
template <typename AccT>
__device__ __forceinline__ AccT metric_abs(AccT value);

template <>
__device__ __forceinline__ float metric_abs<float>(float value) {
    return fabsf(value);
}

template <>
__device__ __forceinline__ double metric_abs<double>(double value) {
    return fabs(value);
}

template <typename AccT>
__device__ __forceinline__ AccT metric_max(AccT a, AccT b);

template <>
__device__ __forceinline__ float metric_max<float>(float a, float b) {
    return fmaxf(a, b);
}

template <>
__device__ __forceinline__ double metric_max<double>(double a, double b) {
    return fmax(a, b);
}

template <typename AccT>
__device__ __forceinline__ AccT metric_sqrt(AccT value);

template <>
__device__ __forceinline__ float metric_sqrt<float>(float value) {
    return sqrtf(value);
}

template <>
__device__ __forceinline__ double metric_sqrt<double>(double value) {
    return sqrt(value);
}

template <int COST, typename AccT>
__device__ __forceinline__ AccT accumulate_dim(AccT acc, float a, float b) {
    const AccT a_acc = static_cast<AccT>(a);
    const AccT b_acc = static_cast<AccT>(b);
    if (COST == kL1) {
        const AccT v = a_acc - b_acc;
        return acc + metric_abs(v);
    } else if (COST == kLinf) {
        const AccT v = a_acc - b_acc;
        return metric_max(acc, metric_abs(v));
    } else {  // kL2
        const AccT v = a_acc - b_acc;
        return acc + v * v;
    }
}

template <int COST, typename AccT>
__device__ __forceinline__ AccT warp_reduce(AccT acc) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const AccT other = __shfl_down_sync(0xffffffffu, acc, off);
        acc = (COST == kLinf) ? metric_max(acc, other) : (acc + other);
    }
    return acc;
}

// CN: 把 4 路部分累加合并成一个（l1/l2 求和，linf 取最大）。
// EN: Merge four partial accumulators (sum for l1/l2, max for linf).
template <int COST, typename AccT>
__device__ __forceinline__ AccT merge_partials(AccT a, AccT b, AccT c, AccT d) {
    if (COST == kLinf) {
        return metric_max(metric_max(a, b), metric_max(c, d));
    }
    return (a + b) + (c + d);
}

// CN: 大 d 路径的单个 (row, col) 距离：lane 按维度分片，warp shuffle 归约；
//     l2 在归约后开方。
// EN: Large-d per-(row, col) distance: lanes slice dimensions, warp-shuffle
//     reduction, l2 takes sqrt after the reduction.
template <int COST, typename AccT>
__device__ __forceinline__ AccT warp_column_dist(const float* drow, const float* qrow, int64_t d, int lane) {
    AccT acc0 = static_cast<AccT>(0);
    AccT acc1 = static_cast<AccT>(0);
    AccT acc2 = static_cast<AccT>(0);
    AccT acc3 = static_cast<AccT>(0);
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
    AccT acc = merge_partials<COST>(acc0, acc1, acc2, acc3);
    acc = warp_reduce<COST>(acc);
    return (COST == kL2) ? metric_sqrt(acc) : acc;
}

NoiseParams make_noise_params(
        const torch::Tensor& query_global_index,
        const torch::Tensor& database_global_index,
        double sigma,
        int64_t seed,
        int64_t perturb_mode,
        bool row_is_source,
        const torch::Tensor& index_hash_coeffs) {
    NoiseParams params;
    params.perturb_mode = static_cast<int>(perturb_mode);
    params.sigma = sigma;
    params.seed_term = static_cast<float>(static_cast<double>(seed) * 0.001);
    params.row_is_source = row_is_source;
    params.coeffs = NoiseCoeffs{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    if (sigma != 0.0 && perturb_mode == kIndexHash) {
        TORCH_CHECK(index_hash_coeffs.defined() && index_hash_coeffs.numel() == 6, "index_hash_coeffs must have 6 elements");
        TORCH_CHECK(index_hash_coeffs.scalar_type() == torch::kInt64, "index_hash_coeffs must be int64");
        auto c = index_hash_coeffs.to(torch::kCPU).contiguous();
        const int64_t* p = c.data_ptr<int64_t>();
        params.coeffs = NoiseCoeffs{
                static_cast<float>(p[0]), static_cast<float>(p[1]), static_cast<float>(p[2]),
                static_cast<float>(p[3]), static_cast<float>(p[4]), static_cast<float>(p[5])};
    }
    (void)query_global_index;
    (void)database_global_index;
    return params;
}

const int64_t* global_index_ptr(const torch::Tensor& tensor, double sigma) {
    if (sigma == 0.0) {
        return nullptr;
    }
    return tensor.data_ptr<int64_t>();
}

// CN: 每个 warp 计算一条稀疏边，不物化 E×d 中间数组。
// EN: Compute one sparse edge per warp without materializing E-by-d intermediates.
template <int COST>
__global__ void gcost_sparse_pairs_kernel(const float* source, const float* target,
        const int64_t* rows, const int64_t* cols, int64_t count, int64_t d,
        const int64_t* source_index, const int64_t* target_index, NoiseParams noise, double* out) {
    const int lane = threadIdx.x & 31;
    const int64_t edge = static_cast<int64_t>(blockIdx.x)*8+(threadIdx.x >> 5);
    if (edge >= count) return;
    const int64_t row = rows[edge], col = cols[edge];
    const double dist = warp_column_dist<COST, double>(target+col*d, source+row*d, d, lane);
    if (lane == 0) {
        out[edge] = dist+(noise.sigma > 0 ? sigma_noise(noise, source_index[row], target_index[col]) : 0.0);
    }
}

}  // namespace

torch::Tensor fused_gcost_pairs_cuda(torch::Tensor query, torch::Tensor db,
        torch::Tensor rows, torch::Tensor cols, int64_t cost_type,
        torch::Tensor source_index, torch::Tensor target_index, double sigma,
        int64_t seed, int64_t mode, torch::Tensor coefficients) {
    TORCH_CHECK(query.is_cuda() && db.is_cuda() && rows.is_cuda() && cols.is_cuda(), "pair inputs must be CUDA");
    TORCH_CHECK(query.device() == db.device() && query.device() == rows.device() && query.device() == cols.device(), "pair inputs must share a device");
    TORCH_CHECK(query.scalar_type() == torch::kFloat32 && db.scalar_type() == torch::kFloat32, "pair features must be float32");
    TORCH_CHECK(query.dim() == 2 && db.dim() == 2 && query.size(1) == db.size(1), "pair feature shape mismatch");
    TORCH_CHECK(query.is_contiguous() && db.is_contiguous() && rows.is_contiguous() && cols.is_contiguous(), "pair inputs must be contiguous");
    TORCH_CHECK(rows.scalar_type() == torch::kInt64 && cols.scalar_type() == torch::kInt64 && rows.numel() == cols.numel(), "pair indices must be matching int64 arrays");
    TORCH_CHECK(cost_type >= 0 && cost_type <= 2, "unsupported norm cost");
    const c10::cuda::CUDAGuard guard(query.device());
    const auto stream = at::cuda::getCurrentCUDAStream();
    auto out = torch::empty({rows.numel()}, query.options().dtype(torch::kFloat64));
    if (rows.numel() == 0) return out;
    const NoiseParams noise = make_noise_params(source_index, target_index, sigma, seed, mode, true, coefficients);
    const int64_t* si = global_index_ptr(source_index, sigma);
    const int64_t* ti = global_index_ptr(target_index, sigma);
    const int64_t blocks = (rows.numel()+7)/8;
#define PAIR_LAUNCH(COST) gcost_sparse_pairs_kernel<COST><<<blocks, 256, 0, stream>>>(query.data_ptr<float>(), db.data_ptr<float>(), rows.data_ptr<int64_t>(), cols.data_ptr<int64_t>(), rows.numel(), query.size(1), si, ti, noise, out.data_ptr<double>())
    if (cost_type == kL1) { PAIR_LAUNCH(kL1); }
    else if (cost_type == kLinf) { PAIR_LAUNCH(kLinf); }
    else { PAIR_LAUNCH(kL2); }
#undef PAIR_LAUNCH
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
