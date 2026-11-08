#include <torch/extension.h>

torch::Tensor fused_gcost_pairs_cuda(torch::Tensor query, torch::Tensor db,
        torch::Tensor rows, torch::Tensor cols, int64_t cost_type,
        torch::Tensor source_index, torch::Tensor target_index, double sigma,
        int64_t seed, int64_t mode, torch::Tensor coefficients);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_gcost_pairs", &fused_gcost_pairs_cuda);
}
