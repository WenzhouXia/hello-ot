#include <torch/extension.h>

#include <vector>

torch::Tensor fused_stats_only_cuda(
        torch::Tensor query_aug,
        torch::Tensor db_aug,
        torch::Tensor query_dual,
        torch::Tensor db_dual,
        int64_t query_tile,
        int64_t db_tile);

torch::Tensor raw_lowrank_stats_only_cuda(
        torch::Tensor query_feat,
        torch::Tensor db_feat,
        torch::Tensor query_cost,
        torch::Tensor db_cost,
        torch::Tensor query_dual,
        torch::Tensor db_dual,
        int64_t query_tile,
        int64_t db_tile,
        double dot_scale);

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
        double dot_scale);

std::vector<torch::Tensor> fused_directional_topk_raw_cuda(
        torch::Tensor query_feat,
        torch::Tensor db_feat,
        torch::Tensor query_bias,
        torch::Tensor db_bias,
        int64_t query_tile,
        int64_t db_tile,
        int64_t k,
        double dot_scale);

std::vector<torch::Tensor> initialization_directional_topk_raw_cuda(
        torch::Tensor query_feat,
        torch::Tensor db_feat,
        torch::Tensor query_bias,
        torch::Tensor db_bias,
        int64_t query_tile,
        int64_t db_tile,
        int64_t k,
        double dot_scale);

void check_common_inputs(
        torch::Tensor query_aug,
        torch::Tensor db_aug,
        torch::Tensor query_dual,
        torch::Tensor db_dual,
        int64_t query_tile,
        int64_t db_tile) {
    // CN: 两个 fused 接口共享同一组输入约束。
    // EN: Both fused entrypoints share the same input constraints.
    TORCH_CHECK(query_aug.is_cuda(), "query_aug must be a CUDA tensor");
    TORCH_CHECK(db_aug.is_cuda(), "db_aug must be a CUDA tensor");
    TORCH_CHECK(query_dual.is_cuda(), "query_dual must be a CUDA tensor");
    TORCH_CHECK(db_dual.is_cuda(), "db_dual must be a CUDA tensor");
    TORCH_CHECK(query_aug.scalar_type() == torch::kFloat32, "query_aug must be float32");
    TORCH_CHECK(db_aug.scalar_type() == torch::kFloat32, "db_aug must be float32");
    TORCH_CHECK(query_dual.scalar_type() == torch::kFloat32, "query_dual must be float32");
    TORCH_CHECK(db_dual.scalar_type() == torch::kFloat32, "db_dual must be float32");
    TORCH_CHECK(query_aug.is_contiguous(), "query_aug must be contiguous");
    TORCH_CHECK(db_aug.is_contiguous(), "db_aug must be contiguous");
    TORCH_CHECK(query_dual.is_contiguous(), "query_dual must be contiguous");
    TORCH_CHECK(db_dual.is_contiguous(), "db_dual must be contiguous");
    TORCH_CHECK(query_aug.dim() == 2, "query_aug must be 2D");
    TORCH_CHECK(db_aug.dim() == 2, "db_aug must be 2D");
    TORCH_CHECK(query_aug.size(1) == db_aug.size(1), "query/db dimensions must match");
    TORCH_CHECK(query_dual.numel() == query_aug.size(0), "query_dual length mismatch");
    TORCH_CHECK(db_dual.numel() == db_aug.size(0), "db_dual length mismatch");
    TORCH_CHECK(query_tile > 0 && db_tile > 0, "tile sizes must be positive");
}

torch::Tensor fused_stats_only(
        torch::Tensor query_aug,
        torch::Tensor db_aug,
        torch::Tensor query_dual,
        torch::Tensor db_dual,
        int64_t query_tile,
        int64_t db_tile) {
    check_common_inputs(query_aug, db_aug, query_dual, db_dual, query_tile, db_tile);
    return fused_stats_only_cuda(
            query_aug,
            db_aug,
            query_dual,
            db_dual,
            query_tile,
            db_tile);
}

void check_raw_lowrank_inputs(
        torch::Tensor query_feat,
        torch::Tensor db_feat,
        torch::Tensor query_cost,
        torch::Tensor db_cost,
        torch::Tensor query_dual,
        torch::Tensor db_dual,
        int64_t query_tile,
        int64_t db_tile) {
    // CN: raw lowrank 接口直接接收 feature/cost/dual，避免构造 augmented feature。
    // EN: The raw lowrank entrypoint consumes feature/cost/dual directly and avoids augmented features.
    TORCH_CHECK(query_feat.is_cuda(), "query_feat must be a CUDA tensor");
    TORCH_CHECK(db_feat.is_cuda(), "db_feat must be a CUDA tensor");
    TORCH_CHECK(query_cost.is_cuda(), "query_cost must be a CUDA tensor");
    TORCH_CHECK(db_cost.is_cuda(), "db_cost must be a CUDA tensor");
    TORCH_CHECK(query_dual.is_cuda(), "query_dual must be a CUDA tensor");
    TORCH_CHECK(db_dual.is_cuda(), "db_dual must be a CUDA tensor");
    TORCH_CHECK(query_feat.scalar_type() == torch::kFloat32, "query_feat must be float32");
    TORCH_CHECK(db_feat.scalar_type() == torch::kFloat32, "db_feat must be float32");
    TORCH_CHECK(query_cost.scalar_type() == torch::kFloat64, "query_cost must be float64");
    TORCH_CHECK(db_cost.scalar_type() == torch::kFloat64, "db_cost must be float64");
    TORCH_CHECK(query_dual.scalar_type() == torch::kFloat64, "query_dual must be float64");
    TORCH_CHECK(db_dual.scalar_type() == torch::kFloat64, "db_dual must be float64");
    TORCH_CHECK(query_feat.is_contiguous(), "query_feat must be contiguous");
    TORCH_CHECK(db_feat.is_contiguous(), "db_feat must be contiguous");
    TORCH_CHECK(query_cost.is_contiguous(), "query_cost must be contiguous");
    TORCH_CHECK(db_cost.is_contiguous(), "db_cost must be contiguous");
    TORCH_CHECK(query_dual.is_contiguous(), "query_dual must be contiguous");
    TORCH_CHECK(db_dual.is_contiguous(), "db_dual must be contiguous");
    TORCH_CHECK(query_feat.dim() == 2, "query_feat must be 2D");
    TORCH_CHECK(db_feat.dim() == 2, "db_feat must be 2D");
    TORCH_CHECK(query_feat.size(1) == db_feat.size(1), "query/db feature dimensions must match");
    TORCH_CHECK(query_cost.numel() == query_feat.size(0), "query_cost length mismatch");
    TORCH_CHECK(db_cost.numel() == db_feat.size(0), "db_cost length mismatch");
    TORCH_CHECK(query_dual.numel() == query_feat.size(0), "query_dual length mismatch");
    TORCH_CHECK(db_dual.numel() == db_feat.size(0), "db_dual length mismatch");
    TORCH_CHECK(query_tile > 0 && db_tile > 0, "tile sizes must be positive");
}

torch::Tensor raw_lowrank_stats_only(
        torch::Tensor query_feat,
        torch::Tensor db_feat,
        torch::Tensor query_cost,
        torch::Tensor db_cost,
        torch::Tensor query_dual,
        torch::Tensor db_dual,
        int64_t query_tile,
        int64_t db_tile,
        double dot_scale) {
    check_raw_lowrank_inputs(query_feat, db_feat, query_cost, db_cost, query_dual, db_dual, query_tile, db_tile);
    TORCH_CHECK(dot_scale == 1.0 || dot_scale == 2.0, "dot_scale must be exactly 1 or 2");
    return raw_lowrank_stats_only_cuda(
            query_feat,
            db_feat,
            query_cost.view({-1}),
            db_cost.view({-1}),
            query_dual.view({-1}),
            db_dual.view({-1}),
            query_tile,
            db_tile,
            dot_scale);
}

void check_raw_bidir_inputs(
        torch::Tensor query_feat,
        torch::Tensor db_feat,
        torch::Tensor query_cost,
        torch::Tensor db_cost,
        torch::Tensor query_bias,
        torch::Tensor db_bias,
        int64_t query_tile,
        int64_t db_tile) {
    // CN: raw bidirectional scan 直接接收 feature/cost/bias，不构造 augmented feature。
    // EN: The raw bidirectional scan receives feature/cost/bias directly without building augmented features.
    TORCH_CHECK(query_feat.is_cuda(), "query_feat must be a CUDA tensor");
    TORCH_CHECK(db_feat.is_cuda(), "db_feat must be a CUDA tensor");
    TORCH_CHECK(query_cost.is_cuda(), "query_cost must be a CUDA tensor");
    TORCH_CHECK(db_cost.is_cuda(), "db_cost must be a CUDA tensor");
    TORCH_CHECK(query_bias.is_cuda(), "query_bias must be a CUDA tensor");
    TORCH_CHECK(db_bias.is_cuda(), "db_bias must be a CUDA tensor");
    TORCH_CHECK(query_feat.scalar_type() == torch::kFloat32, "query_feat must be float32");
    TORCH_CHECK(db_feat.scalar_type() == torch::kFloat32, "db_feat must be float32");
    TORCH_CHECK(query_cost.scalar_type() == torch::kFloat64, "query_cost must be float64");
    TORCH_CHECK(db_cost.scalar_type() == torch::kFloat64, "db_cost must be float64");
    TORCH_CHECK(query_bias.scalar_type() == torch::kFloat64, "query_bias must be float64");
    TORCH_CHECK(db_bias.scalar_type() == torch::kFloat64, "db_bias must be float64");
    TORCH_CHECK(query_feat.is_contiguous(), "query_feat must be contiguous");
    TORCH_CHECK(db_feat.is_contiguous(), "db_feat must be contiguous");
    TORCH_CHECK(query_cost.is_contiguous(), "query_cost must be contiguous");
    TORCH_CHECK(db_cost.is_contiguous(), "db_cost must be contiguous");
    TORCH_CHECK(query_bias.is_contiguous(), "query_bias must be contiguous");
    TORCH_CHECK(db_bias.is_contiguous(), "db_bias must be contiguous");
    TORCH_CHECK(query_feat.dim() == 2, "query_feat must be 2D");
    TORCH_CHECK(db_feat.dim() == 2, "db_feat must be 2D");
    TORCH_CHECK(query_feat.size(1) == db_feat.size(1), "query/db dimensions must match");
    TORCH_CHECK(query_cost.numel() == query_feat.size(0), "query_cost length mismatch");
    TORCH_CHECK(query_bias.numel() == query_feat.size(0), "query_bias length mismatch");
    TORCH_CHECK(db_cost.numel() == db_feat.size(0), "db_cost length mismatch");
    TORCH_CHECK(db_bias.numel() == db_feat.size(0), "db_bias length mismatch");
    TORCH_CHECK(query_tile > 0 && db_tile > 0, "tile sizes must be positive");
}

void check_supported_raw_topk(int64_t k) {
    TORCH_CHECK(
            k == 1 || k == 2 || k == 4 || k == 8 || k == 16 || k == 32,
            "k must be one of {1, 2, 4, 8, 16, 32}");
}

void check_raw_directional_inputs(
        torch::Tensor query_feat,
        torch::Tensor db_feat,
        torch::Tensor query_bias,
        torch::Tensor db_bias,
        int64_t query_tile,
        int64_t db_tile) {
    // CN: directional top-k 不接收 cost/certificate 输入，也不应为校验临时分配 CUDA tensor。
    // EN: Directional top-k has no cost/certificate inputs and must not allocate temporary CUDA tensors for validation.
    TORCH_CHECK(query_feat.is_cuda(), "query_feat must be a CUDA tensor");
    TORCH_CHECK(db_feat.is_cuda(), "db_feat must be a CUDA tensor");
    TORCH_CHECK(query_bias.is_cuda(), "query_bias must be a CUDA tensor");
    TORCH_CHECK(db_bias.is_cuda(), "db_bias must be a CUDA tensor");
    TORCH_CHECK(query_feat.scalar_type() == torch::kFloat32, "query_feat must be float32");
    TORCH_CHECK(db_feat.scalar_type() == torch::kFloat32, "db_feat must be float32");
    TORCH_CHECK(query_bias.scalar_type() == torch::kFloat64, "query_bias must be float64");
    TORCH_CHECK(db_bias.scalar_type() == torch::kFloat64, "db_bias must be float64");
    TORCH_CHECK(query_feat.is_contiguous(), "query_feat must be contiguous");
    TORCH_CHECK(db_feat.is_contiguous(), "db_feat must be contiguous");
    TORCH_CHECK(query_bias.is_contiguous(), "query_bias must be contiguous");
    TORCH_CHECK(db_bias.is_contiguous(), "db_bias must be contiguous");
    TORCH_CHECK(query_feat.dim() == 2, "query_feat must be 2D");
    TORCH_CHECK(db_feat.dim() == 2, "db_feat must be 2D");
    TORCH_CHECK(query_feat.size(1) == db_feat.size(1), "query/db dimensions must match");
    TORCH_CHECK(query_bias.numel() == query_feat.size(0), "query_bias length mismatch");
    TORCH_CHECK(db_bias.numel() == db_feat.size(0), "db_bias length mismatch");
    TORCH_CHECK(query_tile > 0 && db_tile > 0, "tile sizes must be positive");
}

std::vector<torch::Tensor> fused_directional_topk_raw(
        torch::Tensor query_feat,
        torch::Tensor db_feat,
        torch::Tensor query_bias,
        torch::Tensor db_bias,
        int64_t query_tile,
        int64_t db_tile,
        int64_t k,
        double dot_scale) {
    check_raw_directional_inputs(
            query_feat, db_feat, query_bias, db_bias, query_tile, db_tile);
    check_supported_raw_topk(k);
    TORCH_CHECK(dot_scale == 1.0 || dot_scale == 2.0, "dot_scale must be exactly 1 or 2");
    return fused_directional_topk_raw_cuda(
            query_feat,
            db_feat,
            query_bias.view({-1}),
            db_bias.view({-1}),
            query_tile,
            db_tile,
            k,
            dot_scale);
}

std::vector<torch::Tensor> initialization_directional_topk_raw(
        torch::Tensor query_feat,
        torch::Tensor db_feat,
        torch::Tensor query_bias,
        torch::Tensor db_bias,
        int64_t query_tile,
        int64_t db_tile,
        int64_t k,
        double dot_scale) {
    check_raw_directional_inputs(
            query_feat, db_feat, query_bias, db_bias, query_tile, db_tile);
    check_supported_raw_topk(k);
    TORCH_CHECK(dot_scale == 1.0 || dot_scale == 2.0, "dot_scale must be exactly 1 or 2");
    return initialization_directional_topk_raw_cuda(
            query_feat,
            db_feat,
            query_bias.view({-1}),
            db_bias.view({-1}),
            query_tile,
            db_tile,
            k,
            dot_scale);
}

std::vector<torch::Tensor> fused_bidir_topk_stats_raw(
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
    check_raw_bidir_inputs(query_feat, db_feat, query_cost, db_cost, query_bias, db_bias, query_tile, db_tile);
    TORCH_CHECK(col_reduce_tile > 0, "col_reduce_tile must be positive");
    TORCH_CHECK(col_reduce_tile <= 256, "col_reduce_tile must be <= 256");
    check_supported_raw_topk(k);
    TORCH_CHECK(col_kernel == 0 || col_kernel == 1 || col_kernel == 2, "col_kernel must be 0=current, 1=warp4, or 2=warp2");
    TORCH_CHECK(col_kernel != 1 || k == 4, "warp4 col_kernel only supports k=4");
    TORCH_CHECK(col_kernel != 2 || k == 2, "warp2 col_kernel only supports k=2");
    TORCH_CHECK(dot_scale == 1.0 || dot_scale == 2.0, "dot_scale must be exactly 1 or 2");
    return fused_bidir_topk_stats_raw_cuda(
            query_feat,
            db_feat,
            query_cost.view({-1}),
            db_cost.view({-1}),
            query_bias.view({-1}),
            db_bias.view({-1}),
            query_tile,
            db_tile,
            col_reduce_tile,
            k,
            compute_stats,
            collect_linf_stats,
            collect_timing,
            col_kernel,
            dot_scale);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
            "fused_stats_only",
            &fused_stats_only,
            "Fused CUDA reduced-cost stats without top-k selection");
    m.def(
            "raw_lowrank_stats_only",
            &raw_lowrank_stats_only,
            "Raw lowrank CUDA reduced-cost stats without augmented features");
    m.def(
            "fused_bidir_topk_stats_raw",
            &fused_bidir_topk_stats_raw,
            "Raw CUDA bidirectional top-k inner-product search with reduced-cost stats");
    m.def(
            "fused_directional_topk_raw",
            &fused_directional_topk_raw,
            "Raw CUDA directional top-k inner-product scan with one resident feature side");
    m.def(
            "initialization_directional_topk_raw",
            &initialization_directional_topk_raw,
            "Initialization-only CUDA directional top-k inner-product scan");
}
