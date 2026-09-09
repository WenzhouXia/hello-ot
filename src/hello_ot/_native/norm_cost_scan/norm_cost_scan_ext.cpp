#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> fused_gcost_topk_cuda(
        torch::Tensor query,
        torch::Tensor db,
        torch::Tensor dual_db,
        int64_t k,
        int64_t cost_type,
        torch::Tensor query_global_index,
        torch::Tensor database_global_index,
        double sigma,
        int64_t seed,
        int64_t perturb_mode,
        bool row_is_source,
        torch::Tensor index_hash_coeffs);

std::vector<torch::Tensor> fused_gcost_certificate_cuda(
        torch::Tensor query,
        torch::Tensor db,
        torch::Tensor query_dual,
        torch::Tensor db_dual,
        int64_t cost_type,
        torch::Tensor query_global_index,
        torch::Tensor database_global_index,
        double sigma,
        int64_t seed,
        int64_t perturb_mode,
        bool row_is_source,
        torch::Tensor index_hash_coeffs,
        int64_t compute_denominator);

std::vector<torch::Tensor> fused_gcost_bidir_certificate_cuda(
        torch::Tensor query,
        torch::Tensor db,
        torch::Tensor query_dual,
        torch::Tensor db_dual,
        int64_t k,
        int64_t cost_type,
        torch::Tensor query_global_index,
        torch::Tensor database_global_index,
        double sigma,
        int64_t seed,
        int64_t perturb_mode,
        bool row_is_source,
        torch::Tensor index_hash_coeffs);

std::vector<torch::Tensor> fused_mips_topk_cuda(
        torch::Tensor query,
        torch::Tensor db,
        int64_t k);

// CN: 方案D fused 扩展入口。语义与 hierarchical_ot 里 metric 链的
//     _keops_metric_kmin 完全一致：
//       values  = (cost + perturb - dual_db) 每行升序（K 个最小），
//       indices = 对应列号（int64）。
//     cost_type: 0 = l1, 1 = linf, 2 = l2。
//     perturb_mode: 0 = rank2, 1 = index_hash；sigma == 0 时关闭扰动。
//     row_is_source 指明 query 侧是否为 source（决定噪声公式里 src/tgt 的取法）。
// EN: Plan-D fused extension entrypoint. Semantics match _keops_metric_kmin in the
//     metric chain: values are the K smallest (cost + perturb - dual_db) per row
//     (ascending), indices are the matching column ids (int64).
//     cost_type: 0 = l1, 1 = linf, 2 = l2.
//     perturb_mode: 0 = rank2, 1 = index_hash; sigma == 0 disables perturbation.
//     row_is_source selects which side plays src in the noise formula.
std::vector<torch::Tensor> fused_gcost_topk(
        torch::Tensor query,
        torch::Tensor db,
        torch::Tensor dual_db,
        int64_t k,
        int64_t cost_type,
        torch::Tensor query_global_index,
        torch::Tensor database_global_index,
        double sigma,
        int64_t seed,
        int64_t perturb_mode,
        bool row_is_source,
        torch::Tensor index_hash_coeffs) {
    return fused_gcost_topk_cuda(
            query,
            db,
            dual_db,
            k,
            cost_type,
            query_global_index,
            database_global_index,
            sigma,
            seed,
            perturb_mode,
            row_is_source,
            index_hash_coeffs);
}

// CN: 证书归约入口：一次全矩阵遍历返回 (numerator, denominator)，
//     numerator = sum relu(u + v - (cost + perturb))^2，
//     denominator = sum (cost + perturb)^2。语义与 KeOps 的
//     run_metric_dual_feasibility_scan 中两个 lazy reduction 一致。
// EN: Certificate reduction entrypoint: one full-matrix pass returns
//     (numerator, denominator) where
//     numerator = sum relu(u + v - (cost + perturb))^2 and
//     denominator = sum (cost + perturb)^2, matching the two lazy reductions in
//     the KeOps run_metric_dual_feasibility_scan.
std::vector<torch::Tensor> fused_gcost_certificate(
        torch::Tensor query,
        torch::Tensor db,
        torch::Tensor query_dual,
        torch::Tensor db_dual,
        int64_t cost_type,
        torch::Tensor query_global_index,
        torch::Tensor database_global_index,
        double sigma,
        int64_t seed,
        int64_t perturb_mode,
        bool row_is_source,
        torch::Tensor index_hash_coeffs,
        int64_t compute_denominator) {
    return fused_gcost_certificate_cuda(
            query,
            db,
            query_dual,
            db_dual,
            cost_type,
            query_global_index,
            database_global_index,
            sigma,
            seed,
            perturb_mode,
            row_is_source,
            index_hash_coeffs,
            compute_denominator);
}

std::vector<torch::Tensor> fused_gcost_bidir_certificate(
        torch::Tensor query,
        torch::Tensor db,
        torch::Tensor query_dual,
        torch::Tensor db_dual,
        int64_t k,
        int64_t cost_type,
        torch::Tensor query_global_index,
        torch::Tensor database_global_index,
        double sigma,
        int64_t seed,
        int64_t perturb_mode,
        bool row_is_source,
        torch::Tensor index_hash_coeffs) {
    return fused_gcost_bidir_certificate_cuda(
            query, db, query_dual, db_dual, k, cost_type,
            query_global_index, database_global_index, sigma, seed,
            perturb_mode, row_is_source, index_hash_coeffs);
}

// CN: MIPS top-k 入口：score(i, j) = dot(query[i], db[j])（augmented 向量内积），
//     每行返回最大的 K 个 score 与列号，降序、同值按列号升序，语义与
//     torch.topk(largest=True) 一致（允许浮点相等时的并列列号交换）。
//     k 仅支持 {1, 2, 4, 8, 16, 32}。
// EN: MIPS top-k entrypoint: score(i, j) = dot(query[i], db[j]) over the
//     augmented vectors. Returns the K largest scores per row with column ids,
//     descending with ties by ascending column id, matching torch.topk semantics
//     (tolerating column swaps on equal values). k is restricted to {1, 2, 4, 8, 16, 32}.
std::vector<torch::Tensor> fused_mips_topk(
        torch::Tensor query,
        torch::Tensor db,
        int64_t k) {
    return fused_mips_topk_cuda(query, db, k);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.attr("supports_cost_perturbation") = true;
    m.attr("index_hash_arithmetic") = "fp32_remainder_v1";
    m.def("fused_gcost_topk", [](torch::Tensor q, torch::Tensor d, torch::Tensor v, int64_t k, int64_t c) {
        auto idx = torch::empty({0}, q.options().dtype(torch::kInt64));
        return fused_gcost_topk(q, d, v, k, c, idx, idx, 0.0, 0, 1, true, torch::zeros({6}, idx.options()));
    });

    m.def("fused_gcost_certificate", [](torch::Tensor q, torch::Tensor d, torch::Tensor u, torch::Tensor v, int64_t c, int64_t den) {
        auto idx = torch::empty({0}, q.options().dtype(torch::kInt64));
        return fused_gcost_certificate(q, d, u, v, c, idx, idx, 0.0, 0, 1, true, torch::zeros({6}, idx.options()), den);
    });

    m.def("fused_gcost_bidir_certificate", [](torch::Tensor q, torch::Tensor d, torch::Tensor u, torch::Tensor v, int64_t k, int64_t c) {
        auto idx = torch::empty({0}, q.options().dtype(torch::kInt64));
        return fused_gcost_bidir_certificate(q, d, u, v, k, c, idx, idx, 0.0, 0, 1, true, torch::zeros({6}, idx.options()));
    });

    m.def(
            "fused_gcost_topk",
            &fused_gcost_topk,
            "Fused CUDA top-k for general metric costs (l1/linf/l2) with on-the-fly distance, optional cost perturbation, and dual subtraction");
    m.def(
            "fused_gcost_certificate",
            &fused_gcost_certificate,
            "Fused CUDA dual-feasibility certificate (numerator/denominator reductions) for general metric costs with optional perturbation; set compute_denominator=0 to skip the denominator reduction");
    m.def(
            "fused_gcost_bidir_certificate",
            &fused_gcost_bidir_certificate,
            "One-pass bidirectional dual-violation top-k and L2/Linf certificate for norm costs");
    m.def(
            "fused_mips_topk",
            &fused_mips_topk,
            "Fused CUDA MIPS top-k (dot product over augmented vectors) with in-kernel top-K selection; returns (values, indices) per row, k in {1,2,4,8,16,32}");
}
