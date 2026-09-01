#include <cstdint>
#include <stdexcept>
#include <string>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

extern "C" int sparsify_transport_support_lct(
    std::int64_t n_source,
    std::int64_t n_target,
    std::int64_t edge_count,
    const std::int64_t* rows,
    const std::int64_t* cols,
    double* values,
    const double* costs,
    std::int64_t target_nnz,
    double zero_tolerance,
    std::int64_t* output_nnz,
    std::int64_t* output_cycle_updates,
    std::int64_t* output_tree_operations);

namespace {

py::dict sparsify_transport_support(
    std::int64_t n_source,
    std::int64_t n_target,
    const py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>& rows,
    const py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>& cols,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& values,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& costs,
    std::int64_t target_nnz,
    double zero_tolerance) {
    if (rows.ndim() != 1 || cols.ndim() != 1 || values.ndim() != 1 || costs.ndim() != 1) {
        throw std::invalid_argument("rows, cols, values, and costs must be one-dimensional");
    }
    const std::int64_t edge_count = static_cast<std::int64_t>(rows.size());
    if (cols.size() != edge_count || values.size() != edge_count || costs.size() != edge_count) {
        throw std::invalid_argument("rows, cols, values, and costs must have equal lengths");
    }

    py::array_t<double> output_values(edge_count);
    auto output_buffer = output_values.mutable_unchecked<1>();
    auto input_buffer = values.unchecked<1>();
    for (std::int64_t edge = 0; edge < edge_count; ++edge) {
        output_buffer(edge) = input_buffer(edge);
    }

    std::int64_t output_nnz = 0;
    std::int64_t cycle_updates = 0;
    std::int64_t tree_operations = 0;
    int status = 0;
    {
        py::gil_scoped_release release;
        status = sparsify_transport_support_lct(
            n_source,
            n_target,
            edge_count,
            rows.data(),
            cols.data(),
            output_values.mutable_data(),
            costs.data(),
            target_nnz,
            zero_tolerance,
            &output_nnz,
            &cycle_updates,
            &tree_operations);
    }
    if (status != 0) {
        throw std::runtime_error("link-cut-tree support sparsification failed with status=" + std::to_string(status));
    }

    py::dict result;
    result["values"] = std::move(output_values);
    result["output_nnz"] = output_nnz;
    result["cycle_updates"] = cycle_updates;
    result["tree_operations"] = tree_operations;
    return result;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.doc() = "Link-cut-tree support sparsification for restricted optimal transport";
    module.def(
        "sparsify_transport_support",
        &sparsify_transport_support,
        py::arg("n_source"),
        py::arg("n_target"),
        py::arg("rows"),
        py::arg("cols"),
        py::arg("values"),
        py::arg("costs"),
        py::arg("target_nnz"),
        py::arg("zero_tolerance") = 1e-15);
}
