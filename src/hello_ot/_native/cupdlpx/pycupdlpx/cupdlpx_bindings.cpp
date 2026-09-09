// pycupdlpx / cupdlpx_bindings.cpp  (CSC direct, safe device->host copy, unscale, atexit cleanup)
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <vector>
#include <string>
#include <stdexcept>
#include <limits>
#include <cstdint>
#include <climits>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <chrono>

#include <cuda_runtime.h>

#include "../cupdlpx/struct.h"
#include "../cupdlpx/cupdlpx.h"
#include "../cupdlpx/utils.h"

namespace py = pybind11;
// ====== C 接口：求解与释放（实现于 utils.cu/solver.cu）======
extern "C"
{
  pdhg_solver_state_t *optimize(pdhg_parameters_t *params, lp_problem_t *problem);
  pdhg_solver_state_t *optimize_device(const pdhg_parameters_t *params, const device_lp_problem_t *device_problem);
  void pdhg_solver_state_free(pdhg_solver_state_t *state);
  void print_solver_summary(const pdhg_solver_state_t *solver_state);
  // [新增] 声明获取内存的函数
  size_t get_last_solver_peak_mem();
}

static const char *support_limit_mode_to_string(support_limit_mode_t mode)
{
  switch (mode)
  {
  case SUPPORT_LIMIT_MODE_OPTIMAL_ONLY:
    return "optimal_only";
  case SUPPORT_LIMIT_MODE_SUPPORT_ONLY:
    return "support_only";
  case SUPPORT_LIMIT_MODE_BOTH:
    return "both";
  case SUPPORT_LIMIT_MODE_EITHER:
    return "either";
  default:
    return "unknown";
  }
}

static termination_norm_t parse_termination_norm(const py::handle &value)
{
  std::string norm = value.cast<std::string>();
  std::transform(norm.begin(), norm.end(), norm.begin(), [](unsigned char ch) { return (char)std::tolower(ch); });
  if (norm == "l2")
    return TERMINATION_NORM_L2;
  if (norm == "linf")
    return TERMINATION_NORM_L_INF;
  throw std::invalid_argument("termination_norm must be one of: l2, linf.");
}

static const char *termination_norm_to_string(termination_norm_t norm)
{
  return norm == TERMINATION_NORM_L_INF ? "linf" : "l2";
}

static support_limit_mode_t parse_support_limit_mode(const py::handle &value)
{
  const std::string mode = value.cast<std::string>();
  if (mode == "optimal_only")
    return SUPPORT_LIMIT_MODE_OPTIMAL_ONLY;
  if (mode == "support_only")
    return SUPPORT_LIMIT_MODE_SUPPORT_ONLY;
  if (mode == "both")
    return SUPPORT_LIMIT_MODE_BOTH;
  if (mode == "either")
    return SUPPORT_LIMIT_MODE_EITHER;
  throw std::invalid_argument("support_limit_mode must be one of: optimal_only, support_only, both, either.");
}

static variable_bound_mode_t parse_variable_bound_mode(const py::handle &value)
{
  const std::string mode = value.cast<std::string>();
  if (mode == "explicit")
    return VARIABLE_BOUNDS_EXPLICIT;
  if (mode == "constant")
    return VARIABLE_BOUNDS_CONSTANT;
  throw std::invalid_argument("variable_bound_mode must be one of: explicit, constant.");
}

static matrix_value_mode_t parse_matrix_value_mode(const py::handle &value)
{
  const std::string mode = value.cast<std::string>();
  if (mode == "explicit")
    return MATRIX_VALUES_EXPLICIT;
  if (mode == "implicit_aty")
    return MATRIX_VALUES_IMPLICIT_ATY;
  if (mode == "implicit_ax")
    return MATRIX_VALUES_IMPLICIT_AX;
  if (mode == "implicit_both")
    return MATRIX_VALUES_IMPLICIT_BOTH;
  throw std::invalid_argument("matrix_value_mode must be one of: explicit, implicit_aty, implicit_ax, implicit_both.");
}

static const char *matrix_value_mode_to_string(matrix_value_mode_t mode)
{
  switch (mode)
  {
  case MATRIX_VALUES_EXPLICIT:
    return "explicit";
  case MATRIX_VALUES_IMPLICIT_ATY:
    return "implicit_aty";
  case MATRIX_VALUES_IMPLICIT_AX:
    return "implicit_ax";
  case MATRIX_VALUES_IMPLICIT_BOTH:
    return "implicit_both";
  default:
    return "unknown";
  }
}

static const char *implicit_ax_agg_mode_to_string(implicit_ax_agg_mode_t mode)
{
  switch (mode)
  {
  case IMPLICIT_AX_AGG_ROW0:
    return "row0";
  case IMPLICIT_AX_AGG_ROW1:
    return "row1";
  case IMPLICIT_AX_AGG_BOTH:
    return "both";
  case IMPLICIT_AX_AGG_NONE:
  default:
    return "none";
  }
}

static vector_sum_mode_t parse_vector_sum_mode(const py::handle &value)
{
  const std::string mode = value.cast<std::string>();
  if (mode == "resident_ones")
    return VECTOR_SUM_RESIDENT_ONES;
  if (mode == "direct_reduce")
    return VECTOR_SUM_DIRECT_REDUCE;
  throw std::invalid_argument("vector_sum_mode must be one of: resident_ones, direct_reduce.");
}

static const char *vector_sum_mode_to_string(vector_sum_mode_t mode)
{
  switch (mode)
  {
  case VECTOR_SUM_RESIDENT_ONES:
    return "resident_ones";
  case VECTOR_SUM_DIRECT_REDUCE:
    return "direct_reduce";
  default:
    return "unknown";
  }
}

static void validate_support_limit_params(const pdhg_parameters_t &params)
{
  if (params.support_limit_mode == SUPPORT_LIMIT_MODE_OPTIMAL_ONLY && params.support_stop_nnz > 0)
    throw std::invalid_argument("support_limit_mode='optimal_only' conflicts with support_stop_nnz > 0.");
  if (params.support_limit_mode != SUPPORT_LIMIT_MODE_OPTIMAL_ONLY && params.support_stop_nnz <= 0)
    throw std::invalid_argument("support_limit_mode requires support_stop_nnz > 0 unless mode is 'optimal_only'.");
}

static bool is_torch_tensor(const py::handle &obj)
{
  if (obj.is_none())
    return false;
  py::object torch_tensor_type = py::module_::import("torch").attr("Tensor");
  return py::isinstance(obj, torch_tensor_type);
}

static int require_cuda_tensor(
    const py::handle &obj,
    const char *name,
    py::object &owner_out,
    int expected_ndim,
    const char *expected_dtype,
    std::vector<py::ssize_t> expected_shape = {})
{
  if (!is_torch_tensor(obj))
    throw std::runtime_error(std::string(name) + " must be a torch.Tensor.");
  py::object tensor = py::reinterpret_borrow<py::object>(obj);
  if (!tensor.attr("is_cuda").cast<bool>())
    throw std::runtime_error(std::string(name) + " must be a CUDA tensor.");
  if (!tensor.attr("is_contiguous")().cast<bool>())
    throw std::runtime_error(std::string(name) + " must be contiguous.");
  const std::string dtype_str = py::str(tensor.attr("dtype"));
  if (dtype_str != expected_dtype)
    throw std::runtime_error(std::string(name) + " must have dtype " + expected_dtype + ", got " + dtype_str + ".");
  py::tuple shape = tensor.attr("shape").cast<py::tuple>();
  const int ndim = (int)shape.size();
  if (ndim != expected_ndim)
    throw std::runtime_error(std::string(name) + " must be " + std::to_string(expected_ndim) + "D.");
  if (!expected_shape.empty())
  {
    if ((int)expected_shape.size() != ndim)
      throw std::runtime_error("internal error: expected_shape rank mismatch.");
    for (int i = 0; i < ndim; ++i)
    {
      const py::ssize_t got = shape[i].cast<py::ssize_t>();
      if (expected_shape[(size_t)i] >= 0 && got != expected_shape[(size_t)i])
      {
        throw std::runtime_error(
            std::string(name) + " shape mismatch at dim " + std::to_string(i) +
            ": expected " + std::to_string((long long)expected_shape[(size_t)i]) +
            ", got " + std::to_string((long long)got) + ".");
      }
    }
  }
  owner_out = tensor;
  return tensor.attr("device").attr("index").cast<int>();
}

static py::object slice_1d(const py::object &tensor, py::ssize_t start, py::ssize_t stop)
{
  return tensor.attr("__getitem__")(py::slice(start, stop, 1));
}

static bool tensor_all_true(const py::object &tensor_bool)
{
  py::object torch = py::module_::import("torch");
  return torch.attr("all")(tensor_bool).attr("item")().cast<bool>();
}

static long long tensor_scalar_to_long_long(const py::object &tensor_scalar)
{
  return tensor_scalar.attr("item")().cast<long long>();
}

// ====== 安全的设备/主机拷贝工具 ======
static void copy_to_host(double *h_dst, const double *src, size_t n)
{
  if (!h_dst || !src || n == 0)
    return;

  cudaPointerAttributes attr;
  memset(&attr, 0, sizeof(attr));
  cudaError_t e = cudaPointerGetAttributes(&attr, (const void *)src);

  // CUDA 版本间结构字段有差异：保守判断
#if CUDART_VERSION >= 10000
  auto is_dev = (e == cudaSuccess) && (attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged);
#else
  auto is_dev = (e == cudaSuccess) && (attr.memoryType == cudaMemoryTypeDevice); // 旧字段
#endif

  if (is_dev)
  {
    // 设备或统一内存，直接 D2H
    cudaError_t ce = cudaMemcpy(h_dst, src, n * sizeof(double), cudaMemcpyDeviceToHost);
    if (ce != cudaSuccess)
    {
      throw std::runtime_error(std::string("cudaMemcpy D2H failed: ") + cudaGetErrorString(ce));
    }
  }
  else
  {
    // 当作 Host 指针
    std::memcpy(h_dst, src, n * sizeof(double));
  }
}

static py::array_t<double> export_unscaled_primal_buffer(
    const pdhg_solver_state_t *state,
    const double *src)
{
  const py::ssize_t n = static_cast<py::ssize_t>((state != nullptr) ? state->num_variables : 0);
  py::array_t<double> out(n);
  if (state == nullptr || src == nullptr || n <= 0)
    return out;

  std::vector<double> buf(static_cast<size_t>(n), 0.0);
  copy_to_host(buf.data(), src, static_cast<size_t>(n));
  std::vector<double> svar(static_cast<size_t>(n), 1.0);
  if (state->variable_rescaling)
    copy_to_host(svar.data(), state->variable_rescaling, static_cast<size_t>(n));
  const double alpha_x = (state->constraint_bound_rescaling != 0.0) ? state->constraint_bound_rescaling : 1.0;

  auto out_m = out.mutable_unchecked<1>();
  for (py::ssize_t i = 0; i < n; ++i)
  {
    const double sv = (svar[static_cast<size_t>(i)] != 0.0) ? svar[static_cast<size_t>(i)] : 1.0;
    out_m(i) = buf[static_cast<size_t>(i)] / (sv * alpha_x);
  }
  return out;
}

static py::array_t<double> export_unscaled_dual_buffer(
    const pdhg_solver_state_t *state,
    const double *src)
{
  const py::ssize_t n = static_cast<py::ssize_t>((state != nullptr) ? state->num_constraints : 0);
  py::array_t<double> out(n);
  if (state == nullptr || src == nullptr || n <= 0)
    return out;

  std::vector<double> buf(static_cast<size_t>(n), 0.0);
  copy_to_host(buf.data(), src, static_cast<size_t>(n));
  std::vector<double> scon(static_cast<size_t>(n), 1.0);
  if (state->constraint_rescaling)
    copy_to_host(scon.data(), state->constraint_rescaling, static_cast<size_t>(n));
  const double alpha_y = (state->objective_vector_rescaling != 0.0) ? state->objective_vector_rescaling : 1.0;

  auto out_m = out.mutable_unchecked<1>();
  for (py::ssize_t i = 0; i < n; ++i)
  {
    const double sc = (scon[static_cast<size_t>(i)] != 0.0) ? scon[static_cast<size_t>(i)] : 1.0;
    out_m(i) = buf[static_cast<size_t>(i)] / (sc * alpha_y);
  }
  return out;
}

// ====== CSC -> CSR ======
static void csc_to_csr(
    int m, int n,
    const int *csc_col_ptr, // n+1
    const int *csc_row_idx, // nnz
    const double *csc_val,  // nnz
    std::vector<int> &csr_row_ptr,
    std::vector<int> &csr_col_idx,
    std::vector<double> &csr_val)
{
  const int nnz = csc_col_ptr[n];
  csr_row_ptr.assign(m + 1, 0);
  csr_col_idx.resize(nnz);
  csr_val.resize(nnz);

  for (int k = 0; k < nnz; ++k)
  {
    int r = csc_row_idx[k];
    if (r < 0 || r >= m)
      throw std::runtime_error("CSC row index out of range");
    csr_row_ptr[r + 1]++;
  }
  for (int i = 0; i < m; ++i)
    csr_row_ptr[i + 1] += csr_row_ptr[i];

  std::vector<int> next = csr_row_ptr;
  for (int j = 0; j < n; ++j)
  {
    for (int p = csc_col_ptr[j]; p < csc_col_ptr[j + 1]; ++p)
    {
      int r = csc_row_idx[p];
      int dst = next[r]++;
      csr_col_idx[dst] = j;
      csr_val[dst] = csc_val[p];
    }
  }
}

// 释放策略：0=全释放；1=只 free state（默认）；2=只 free problem
static int g_free_mode = 1;

// ====== 绑定类 ======
class CupdlpxHolder
{
public:
  CupdlpxHolder()
      : problem_(nullptr),
        device_problem_loaded_(false),
        m_(0),
        n_(0),
        nnz_(0),
        last_load_validate_time_sec_(0.0),
        last_load_copy_vectors_time_sec_(0.0),
        last_load_constraint_bounds_time_sec_(0.0),
        last_load_copy_csc_triplets_time_sec_(0.0),
        last_load_csc_to_csr_time_sec_(0.0),
        last_load_problem_bind_time_sec_(0.0),
        last_load_total_time_sec_(0.0) {}
  ~CupdlpxHolder()
  {
    if (problem_)
    {
      delete problem_;
      problem_ = nullptr;
    } // 只 delete 壳体
    clear_device_problem_refs();
  }

  void clear_device_problem_refs()
  {
    device_problem_loaded_ = false;
    device_problem_ = device_lp_problem_t{};
    d_row_ptr_owner_ = py::none();
    d_col_idx_owner_ = py::none();
    d_val_owner_ = py::none();
    d_c_owner_ = py::none();
    d_rhs_owner_ = py::none();
    d_lb_owner_ = py::none();
    d_ub_owner_ = py::none();
    d_constraint_rescaling_owner_ = py::none();
    d_variable_rescaling_owner_ = py::none();
    device_ordinal_ = -1;
  }

  // ====== 直接接收 scipy.sparse.{csc,csr}_matrix ======
  void loadData(py::object A,
                py::array_t<double, py::array::c_style | py::array::forcecast> c,
                py::array_t<double, py::array::c_style | py::array::forcecast> rhs,
                py::array_t<double, py::array::c_style | py::array::forcecast> lb,
                py::array_t<double, py::array::c_style | py::array::forcecast> ub,
                int nEqs)
  {
    py::module_ sp = py::module_::import("scipy.sparse");
    bool is_csc = sp.attr("isspmatrix_csc")(A).cast<bool>();
    bool is_csr = sp.attr("isspmatrix_csr")(A).cast<bool>();
    if (!(is_csc || is_csr))
    {
      throw std::runtime_error("A must be a scipy.sparse.csc_matrix or scipy.sparse.csr_matrix");
    }

    py::tuple shape = A.attr("shape").cast<py::tuple>();
    if (shape.size() != 2)
    {
      throw std::runtime_error("A.shape must be a 2-tuple");
    }
    int m = shape[0].cast<int>();
    int n = shape[1].cast<int>();

    auto indptr = A.attr("indptr").cast<py::array_t<int, py::array::c_style | py::array::forcecast>>();
    auto indices = A.attr("indices").cast<py::array_t<int, py::array::c_style | py::array::forcecast>>();
    auto data = A.attr("data").cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();

    if (is_csc)
    {
      this->loadData_csc(indptr, indices, data, m, n, c, rhs, lb, ub, nEqs);
    }
    else
    {
      this->loadData_csr(indptr, indices, data, m, n, c, rhs, lb, ub, nEqs);
    }
  }

  void loadData_csr(py::array_t<int, py::array::c_style | py::array::forcecast> indptr,
                    py::array_t<int, py::array::c_style | py::array::forcecast> indices,
                    py::array_t<double, py::array::c_style | py::array::forcecast> data,
                    int m, int n,
                    py::array_t<double, py::array::c_style | py::array::forcecast> c,
                    py::array_t<double, py::array::c_style | py::array::forcecast> rhs,
                    py::array_t<double, py::array::c_style | py::array::forcecast> lb,
                    py::array_t<double, py::array::c_style | py::array::forcecast> ub,
                    int nEqs)
  {
    using clk = std::chrono::high_resolution_clock;
    auto sec = [](auto a, auto b)
    { return std::chrono::duration<double>(b - a).count(); };
    auto t_all0 = clk::now();
    auto t0 = t_all0;
    clear_device_problem_refs();

    if (m <= 0 || n <= 0)
      throw std::runtime_error("m,n must be positive.");
    if (indptr.ndim() != 1 || (int)indptr.size() != m + 1)
      throw std::runtime_error("CSR indptr must have length m+1.");
    if (c.ndim() != 1 || (int)c.size() != n)
      throw std::runtime_error("c must have length n.");
    if (rhs.ndim() != 1 || (int)rhs.size() != m)
      throw std::runtime_error("rhs must have length m.");
    if (lb.ndim() != 1 || (int)lb.size() != n)
      throw std::runtime_error("lb must have length n.");
    if (ub.ndim() != 1 || (int)ub.size() != n)
      throw std::runtime_error("ub must have length n.");
    const int nnz = indptr.at(m);
    if (indices.ndim() != 1 || (int)indices.size() != nnz)
      throw std::runtime_error("indices length mismatch.");
    if (data.ndim() != 1 || (int)data.size() != nnz)
      throw std::runtime_error("data length mismatch.");
    if (nEqs < 0 || nEqs > m)
      throw std::runtime_error("nEqs out of range.");

    {
      auto ip = indptr.unchecked<1>();
      for (int i = 0; i < m; ++i)
        if (ip(i) > ip(i + 1))
          throw std::runtime_error("CSR indptr must be non-decreasing.");
      auto cidx = indices.unchecked<1>();
      for (int k = 0; k < nnz; ++k)
        if (cidx(k) < 0 || cidx(k) >= n)
          throw std::runtime_error("CSR column index out of range.");
    }
    auto t1 = clk::now();
    last_load_validate_time_sec_ = sec(t0, t1);

    m_ = m;
    n_ = n;
    nnz_ = nnz;

    t0 = clk::now();
    var_lb_.assign((double *)lb.data(), (double *)lb.data() + n_);
    var_ub_.assign((double *)ub.data(), (double *)ub.data() + n_);
    obj_.assign((double *)c.data(), (double *)c.data() + n_);
    rhs_.assign((double *)rhs.data(), (double *)rhs.data() + m_);
    t1 = clk::now();
    last_load_copy_vectors_time_sec_ = sec(t0, t1);

    t0 = clk::now();
    const double INF = std::numeric_limits<double>::infinity();
    con_lb_.assign(m_, -INF);
    con_ub_.assign(m_, INF);
    for (int i = 0; i < m_; ++i)
    {
      if (i < nEqs)
      {
        con_lb_[i] = rhs_[i];
        con_ub_[i] = rhs_[i];
      }
      else
      {
        con_ub_[i] = rhs_[i];
      }
    }
    t1 = clk::now();
    last_load_constraint_bounds_time_sec_ = sec(t0, t1);

    t0 = clk::now();
    csr_row_ptr_.assign((int *)indptr.data(), (int *)indptr.data() + (m_ + 1));
    csr_col_idx_.assign((int *)indices.data(), (int *)indices.data() + nnz_);
    csr_val_.assign((double *)data.data(), (double *)data.data() + nnz_);
    t1 = clk::now();
    last_load_copy_csc_triplets_time_sec_ = sec(t0, t1);
    last_load_csc_to_csr_time_sec_ = 0.0;

    t0 = clk::now();
    if (!problem_)
      problem_ = new lp_problem_t();
    problem_->num_variables = n_;
    problem_->num_constraints = m_;
    problem_->variable_lower_bound = const_cast<double *>(var_lb_.data());
    problem_->variable_upper_bound = const_cast<double *>(var_ub_.data());
    problem_->objective_vector = const_cast<double *>(obj_.data());
    problem_->objective_constant = 0.0;
    problem_->constraint_matrix_row_pointers = const_cast<int *>(csr_row_ptr_.data());
    problem_->constraint_matrix_col_indices = const_cast<int *>(csr_col_idx_.data());
    problem_->constraint_matrix_values = const_cast<double *>(csr_val_.data());
    problem_->constraint_matrix_num_nonzeros = nnz_;
    problem_->matrix_value_mode = MATRIX_VALUES_EXPLICIT;
    problem_->constraint_lower_bound = const_cast<double *>(con_lb_.data());
    problem_->constraint_upper_bound = const_cast<double *>(con_ub_.data());
    t1 = clk::now();
    last_load_problem_bind_time_sec_ = sec(t0, t1);
    last_load_total_time_sec_ = sec(t_all0, t1);
  }

  void loadData_csc(py::array_t<int, py::array::c_style | py::array::forcecast> indptr,
                    py::array_t<int, py::array::c_style | py::array::forcecast> indices,
                    py::array_t<double, py::array::c_style | py::array::forcecast> data,
                    int m, int n,
                    py::array_t<double, py::array::c_style | py::array::forcecast> c,
                    py::array_t<double, py::array::c_style | py::array::forcecast> rhs,
                    py::array_t<double, py::array::c_style | py::array::forcecast> lb,
                    py::array_t<double, py::array::c_style | py::array::forcecast> ub,
                    int nEqs)
  {
    using clk = std::chrono::high_resolution_clock;
    auto sec = [](auto a, auto b)
    { return std::chrono::duration<double>(b - a).count(); };
    auto t_all0 = clk::now();
    auto t0 = t_all0;
    clear_device_problem_refs();

    if (m <= 0 || n <= 0)
      throw std::runtime_error("m,n must be positive.");
    if (indptr.ndim() != 1 || (int)indptr.size() != n + 1)
      throw std::runtime_error("indptr must have length n+1.");
    if (c.ndim() != 1 || (int)c.size() != n)
      throw std::runtime_error("c must have length n.");
    if (rhs.ndim() != 1 || (int)rhs.size() != m)
      throw std::runtime_error("rhs must have length m.");
    if (lb.ndim() != 1 || (int)lb.size() != n)
      throw std::runtime_error("lb must have length n.");
    if (ub.ndim() != 1 || (int)ub.size() != n)
      throw std::runtime_error("ub must have length n.");
    const int nnz = indptr.at(n);
    if (indices.ndim() != 1 || (int)indices.size() != nnz)
      throw std::runtime_error("indices length mismatch.");
    if (data.ndim() != 1 || (int)data.size() != nnz)
      throw std::runtime_error("data length mismatch.");
    if (nEqs < 0 || nEqs > m)
      throw std::runtime_error("nEqs out of range.");

    // 基本检查
    {
      auto ip = indptr.unchecked<1>();
      for (int j = 0; j < n; ++j)
        if (ip(j) > ip(j + 1))
          throw std::runtime_error("indptr must be non-decreasing.");
      auto ridx = indices.unchecked<1>();
      for (int k = 0; k < nnz; ++k)
        if (ridx(k) < 0 || ridx(k) >= m)
          throw std::runtime_error("row index out of range.");
    }
    auto t1 = clk::now();
    last_load_validate_time_sec_ = sec(t0, t1);

    m_ = m;
    n_ = n;
    nnz_ = nnz;

    t0 = clk::now();
    var_lb_.assign((double *)lb.data(), (double *)lb.data() + n_);
    var_ub_.assign((double *)ub.data(), (double *)ub.data() + n_);
    obj_.assign((double *)c.data(), (double *)c.data() + n_);
    rhs_.assign((double *)rhs.data(), (double *)rhs.data() + m_);
    t1 = clk::now();
    last_load_copy_vectors_time_sec_ = sec(t0, t1);

    t0 = clk::now();
    const double INF = std::numeric_limits<double>::infinity();
    con_lb_.assign(m_, -INF);
    con_ub_.assign(m_, INF);
    for (int i = 0; i < m_; ++i)
    {
      if (i < nEqs)
      {
        con_lb_[i] = rhs_[i];
        con_ub_[i] = rhs_[i];
      }
      else
      {
        con_ub_[i] = rhs_[i];
      }
    }
    t1 = clk::now();
    last_load_constraint_bounds_time_sec_ = sec(t0, t1);

    // CSC -> CSR
    t0 = clk::now();
    {
      auto ip = indptr.unchecked<1>();
      tmp_col_ptr_.resize(n_ + 1);
      for (int j = 0; j <= n_; ++j)
        tmp_col_ptr_[j] = ip(j);

      tmp_row_idx_.assign((int *)indices.data(), (int *)indices.data() + nnz_);
      tmp_val_.assign((double *)data.data(), (double *)data.data() + nnz_);
    }
    t1 = clk::now();
    last_load_copy_csc_triplets_time_sec_ = sec(t0, t1);

    t0 = clk::now();
    csc_to_csr(m_, n_, tmp_col_ptr_.data(), tmp_row_idx_.data(), tmp_val_.data(),
               csr_row_ptr_, csr_col_idx_, csr_val_);
    t1 = clk::now();
    last_load_csc_to_csr_time_sec_ = sec(t0, t1);

    t0 = clk::now();
    if (!problem_)
      problem_ = new lp_problem_t();
    problem_->num_variables = n_;
    problem_->num_constraints = m_;
    problem_->variable_lower_bound = const_cast<double *>(var_lb_.data());
    problem_->variable_upper_bound = const_cast<double *>(var_ub_.data());
    problem_->objective_vector = const_cast<double *>(obj_.data());
    problem_->objective_constant = 0.0;

    problem_->constraint_matrix_row_pointers = const_cast<int *>(csr_row_ptr_.data());
    problem_->constraint_matrix_col_indices = const_cast<int *>(csr_col_idx_.data());
    problem_->constraint_matrix_values = const_cast<double *>(csr_val_.data());
    problem_->constraint_matrix_num_nonzeros = nnz_;

    problem_->constraint_lower_bound = const_cast<double *>(con_lb_.data());
    problem_->constraint_upper_bound = const_cast<double *>(con_ub_.data());
    t1 = clk::now();
    last_load_problem_bind_time_sec_ = sec(t0, t1);
    last_load_total_time_sec_ = sec(t_all0, t1);
  }

  void loadData_device_csr(py::object row_ptr,
                           py::object col_ind,
                           py::object values,
                           int m, int n,
                           py::object c,
                           py::object rhs,
                           py::object lb,
                           py::object ub,
                           int nEqs,
                           py::object variable_bound_mode = py::str("explicit"),
                           py::object matrix_value_mode = py::str("explicit"),
                           py::object constraint_rescaling = py::none(),
                           py::object variable_rescaling = py::none(),
                           double constraint_bound_rescaling = 1.0,
                           double objective_vector_rescaling = 1.0,
                           py::object original_objective_vector_norm = py::none(),
                           py::object original_constraint_bound_norm = py::none(),
                           bool has_precomputed_rescaling = false,
                           py::object original_objective_vector_linf_norm = py::none(),
                           py::object original_constraint_bound_linf_norm = py::none())
  {
    using clk = std::chrono::high_resolution_clock;
    auto sec = [](auto a, auto b)
    { return std::chrono::duration<double>(b - a).count(); };
    auto t_all0 = clk::now();
    auto t0 = t_all0;

    clear_device_problem_refs();
    if (m <= 0 || n <= 0)
      throw std::runtime_error("m,n must be positive.");
    if (nEqs < 0 || nEqs > m)
      throw std::runtime_error("nEqs out of range.");

    int device_idx = -1;
    auto expect_same_device = [&](int got, const char *name) {
      if (device_idx < 0)
      {
        device_idx = got;
      }
      else if (got != device_idx)
      {
        throw std::runtime_error(std::string(name) + " is on a different CUDA device than previous tensors.");
      }
    };

    py::object torch = py::module_::import("torch");
    const matrix_value_mode_t value_mode = parse_matrix_value_mode(matrix_value_mode);
    int got_dev = require_cuda_tensor(row_ptr, "row_ptr", d_row_ptr_owner_, 1, "torch.int32", {(py::ssize_t)m + 1});
    expect_same_device(got_dev, "row_ptr");
    got_dev = require_cuda_tensor(col_ind, "col_ind", d_col_idx_owner_, 1, "torch.int32");
    expect_same_device(got_dev, "col_ind");
    if (matrix_value_mode_has_implicit_ax(value_mode))
    {
      if (!values.is_none())
        throw std::runtime_error("implicit_ax and implicit_both matrix values require values=None.");
      d_val_owner_ = py::none();
    }
    else
    {
      got_dev = require_cuda_tensor(values, "values", d_val_owner_, 1, "torch.float64");
      expect_same_device(got_dev, "values");
    }
    got_dev = require_cuda_tensor(c, "c", d_c_owner_, 1, "torch.float64", {(py::ssize_t)n});
    expect_same_device(got_dev, "c");
    got_dev = require_cuda_tensor(rhs, "rhs", d_rhs_owner_, 1, "torch.float64", {(py::ssize_t)m});
    expect_same_device(got_dev, "rhs");
    if (has_precomputed_rescaling)
    {
      got_dev = require_cuda_tensor(constraint_rescaling, "constraint_rescaling", d_constraint_rescaling_owner_, 1, "torch.float64", {(py::ssize_t)m});
      expect_same_device(got_dev, "constraint_rescaling");
      got_dev = require_cuda_tensor(variable_rescaling, "variable_rescaling", d_variable_rescaling_owner_, 1, "torch.float64", {(py::ssize_t)n});
      expect_same_device(got_dev, "variable_rescaling");
      if (original_objective_vector_norm.is_none() || original_constraint_bound_norm.is_none())
        throw std::runtime_error("precomputed rescaling requires original objective and constraint-bound norms.");
    }
    const variable_bound_mode_t bound_mode = parse_variable_bound_mode(variable_bound_mode);
    double lb_constant = 0.0;
    double ub_constant = 0.0;
    if (bound_mode == VARIABLE_BOUNDS_EXPLICIT)
    {
      got_dev = require_cuda_tensor(lb, "lb", d_lb_owner_, 1, "torch.float64", {(py::ssize_t)n});
      expect_same_device(got_dev, "lb");
      got_dev = require_cuda_tensor(ub, "ub", d_ub_owner_, 1, "torch.float64", {(py::ssize_t)n});
      expect_same_device(got_dev, "ub");
    }
    else
    {
      if (is_torch_tensor(lb) || is_torch_tensor(ub))
        throw std::runtime_error("constant variable bounds require scalar lb and ub, not tensors.");
      lb_constant = lb.cast<double>();
      ub_constant = ub.cast<double>();
      d_lb_owner_ = py::none();
      d_ub_owner_ = py::none();
    }

    const long long row_ptr_last = tensor_scalar_to_long_long(d_row_ptr_owner_.attr("__getitem__")(m));
    if (row_ptr_last < 0)
      throw std::runtime_error("CSR row_ptr[m] must be non-negative.");
    const py::ssize_t nnz = (py::ssize_t)row_ptr_last;
    if (col_ind.attr("numel")().cast<py::ssize_t>() != nnz)
      throw std::runtime_error("CSR col_ind length must equal row_ptr[m].");
    if (matrix_value_mode_has_explicit_a(value_mode) && values.attr("numel")().cast<py::ssize_t>() != nnz)
      throw std::runtime_error("CSR values length must equal row_ptr[m].");
    if (m > 0)
    {
      py::object row_ptr_head = slice_1d(d_row_ptr_owner_, 0, m);
      py::object row_ptr_tail = slice_1d(d_row_ptr_owner_, 1, m + 1);
      if (!tensor_all_true(row_ptr_tail.attr("ge")(row_ptr_head)))
        throw std::runtime_error("CSR row_ptr must be non-decreasing.");
    }
    if (nnz > 0)
    {
      py::object nonnegative = d_col_idx_owner_.attr("ge")(py::int_(0));
      py::object below_n = d_col_idx_owner_.attr("lt")(py::int_(n));
      if (!tensor_all_true(torch.attr("logical_and")(nonnegative, below_n)))
        throw std::runtime_error("CSR column index out of range.");
    }
    if (nnz > (py::ssize_t)INT_MAX)
      throw std::runtime_error("nnz exceeds int32 solver capacity.");

    device_problem_.num_variables = n;
    device_problem_.num_constraints = m;
    device_problem_.constraint_matrix_num_nonzeros = (int)nnz;
    device_problem_.constraint_matrix_row_pointers = reinterpret_cast<const int *>(d_row_ptr_owner_.attr("data_ptr")().cast<std::uintptr_t>());
    device_problem_.constraint_matrix_col_indices = reinterpret_cast<const int *>(d_col_idx_owner_.attr("data_ptr")().cast<std::uintptr_t>());
    device_problem_.constraint_matrix_values = matrix_value_mode_has_implicit_ax(value_mode)
        ? nullptr
        : reinterpret_cast<const double *>(d_val_owner_.attr("data_ptr")().cast<std::uintptr_t>());
    device_problem_.matrix_value_mode = value_mode;
    device_problem_.variable_bound_mode = bound_mode;
    device_problem_.variable_lower_bound_constant = lb_constant;
    device_problem_.variable_upper_bound_constant = ub_constant;
    if (bound_mode == VARIABLE_BOUNDS_EXPLICIT)
    {
      device_problem_.variable_lower_bound = reinterpret_cast<const double *>(d_lb_owner_.attr("data_ptr")().cast<std::uintptr_t>());
      device_problem_.variable_upper_bound = reinterpret_cast<const double *>(d_ub_owner_.attr("data_ptr")().cast<std::uintptr_t>());
    }
    else
    {
      device_problem_.variable_lower_bound = nullptr;
      device_problem_.variable_upper_bound = nullptr;
    }
    device_problem_.objective_vector = reinterpret_cast<const double *>(d_c_owner_.attr("data_ptr")().cast<std::uintptr_t>());
    device_problem_.right_hand_side = reinterpret_cast<const double *>(d_rhs_owner_.attr("data_ptr")().cast<std::uintptr_t>());
    device_problem_.num_equalities = nEqs;
    device_problem_.objective_constant = 0.0;
    device_problem_.has_precomputed_rescaling = has_precomputed_rescaling;
    if (has_precomputed_rescaling)
    {
      device_problem_.constraint_rescaling = reinterpret_cast<const double *>(d_constraint_rescaling_owner_.attr("data_ptr")().cast<std::uintptr_t>());
      device_problem_.variable_rescaling = reinterpret_cast<const double *>(d_variable_rescaling_owner_.attr("data_ptr")().cast<std::uintptr_t>());
      device_problem_.constraint_bound_rescaling = constraint_bound_rescaling;
      device_problem_.objective_vector_rescaling = objective_vector_rescaling;
      device_problem_.original_objective_vector_norm = original_objective_vector_norm.cast<double>();
      device_problem_.original_constraint_bound_norm = original_constraint_bound_norm.cast<double>();
      device_problem_.original_objective_vector_linf_norm = original_objective_vector_linf_norm.is_none()
          ? std::numeric_limits<double>::quiet_NaN()
          : original_objective_vector_linf_norm.cast<double>();
      device_problem_.original_constraint_bound_linf_norm = original_constraint_bound_linf_norm.is_none()
          ? std::numeric_limits<double>::quiet_NaN()
          : original_constraint_bound_linf_norm.cast<double>();
    }
    else
    {
      device_problem_.constraint_rescaling = nullptr;
      device_problem_.variable_rescaling = nullptr;
      device_problem_.constraint_bound_rescaling = 1.0;
      device_problem_.objective_vector_rescaling = 1.0;
      device_problem_.original_objective_vector_norm = 0.0;
      device_problem_.original_constraint_bound_norm = 0.0;
      device_problem_.original_objective_vector_linf_norm = 0.0;
      device_problem_.original_constraint_bound_linf_norm = 0.0;
    }

    device_problem_loaded_ = true;
    device_ordinal_ = device_idx;
    m_ = m;
    n_ = n;
    nnz_ = (int)nnz;

    auto t1 = clk::now();
    last_load_validate_time_sec_ = sec(t0, t1);
    last_load_copy_vectors_time_sec_ = 0.0;
    last_load_constraint_bounds_time_sec_ = 0.0;
    last_load_copy_csc_triplets_time_sec_ = 0.0;
    last_load_csc_to_csr_time_sec_ = 0.0;
    last_load_problem_bind_time_sec_ = 0.0;
    last_load_total_time_sec_ = sec(t_all0, t1);
  }

  py::dict getLoadStats() const
  {
    py::dict d;
    if (device_problem_loaded_)
      d["sparse_format"] = py::str("device_csr");
    else
      d["sparse_format"] = (last_load_csc_to_csr_time_sec_ > 0.0) ? py::str("csc") : py::str("csr");
    d["validate_time_sec"] = last_load_validate_time_sec_;
    d["copy_vectors_time_sec"] = last_load_copy_vectors_time_sec_;
    d["constraint_bounds_time_sec"] = last_load_constraint_bounds_time_sec_;
    d["copy_sparse_triplets_time_sec"] = last_load_copy_csc_triplets_time_sec_;
    d["copy_csc_triplets_time_sec"] = last_load_copy_csc_triplets_time_sec_;
    d["csc_to_csr_time_sec"] = last_load_csc_to_csr_time_sec_;
    d["problem_bind_time_sec"] = last_load_problem_bind_time_sec_;
    d["total_time_sec"] = last_load_total_time_sec_;
    return d;
  }

  // py::dict solve(py::dict user_params)
  // {
  //   if (!problem_)
  //     throw std::runtime_error("No problem loaded. Call loadData_csc() first.");

  //   pdhg_parameters_t params;
  //   set_default_parameters(&params);
  //   auto pull = [&](const char *k, auto &dst)
  //   {
  //     if (user_params.contains(k))
  //       dst = user_params[k].cast<std::remove_reference_t<decltype(dst)>>();
  //   };
  //   pull("verbose", params.verbose);
  //   pull("time_sec_limit", params.termination_criteria.time_sec_limit);
  //   pull("iteration_limit", params.termination_criteria.iteration_limit);
  //   pull("eps_optimal_relative", params.termination_criteria.eps_optimal_relative);
  //   pull("eps_feasible_relative", params.termination_criteria.eps_feasible_relative);
  //   pull("eps_feasible_relative_primal", params.termination_criteria.eps_feasible_relative_primal);
  //   pull("eps_feasible_relative_dual", params.termination_criteria.eps_feasible_relative_dual);
  //   pull("eps_infeasible", params.termination_criteria.eps_infeasible);
  //   pull("l_inf_ruiz_iterations", params.l_inf_ruiz_iterations);
  //   pull("bound_objective_rescaling", params.bound_objective_rescaling);
  //   pull("has_pock_chambolle_alpha", params.has_pock_chambolle_alpha);
  //   pull("pock_chambolle_alpha", params.pock_chambolle_alpha);
  //   pull("termination_evaluation_frequency", params.termination_evaluation_frequency);
  //   pull("reflection_coefficient", params.reflection_coefficient);
  //   pull("use_dual_nnz_gate", params.termination_criteria.use_dual_nnz_gate);
  //   pull("dual_nnz_factor", params.termination_criteria.dual_nnz_factor);
  //   pull("dual_nnz_abs_tol", params.termination_criteria.dual_nnz_abs_tol);

  //   if (params.termination_criteria.use_dual_nnz_gate)
  //   {
  //     printf("Use checkSparsity, tol = %f x nCols, eps = %f\n", params.termination_criteria.dual_nnz_factor, params.termination_criteria.dual_nnz_abs_tol);
  //   }
  //   else
  //   {
  //     printf("Do Not Use checkSparsity.\n");
  //   }

  //   if (has_init_iterate_)
  //   {
  //     params.has_initial_iterate = true;
  //     if (x0_buf_.ndim() > 0 && x0_buf_.size() == n_)
  //     {
  //       params.initial_primal_unscaled = static_cast<const double *>(x0_buf_.data());
  //     }
  //     if (y0_buf_.ndim() > 0 && y0_buf_.size() == m_)
  //     {
  //       params.initial_dual_unscaled = static_cast<const double *>(y0_buf_.data());
  //     }
  //   }

  //   // 求解
  //   pdhg_solver_state_t *state = optimize(&params, problem_);
  //   if (!state)
  //     throw std::runtime_error("cuPDLPx optimize() failed.");
  //   print_solver_summary(state);
  //   // === 从可能的 device/managed 指针拷到 host ===
  //   std::vector<double> hx(n_), hy(m_);
  //   copy_to_host(hx.data(), state->pdhg_primal_solution, (size_t)n_);
  //   copy_to_host(hy.data(), state->pdhg_dual_solution, (size_t)m_);

  //   std::vector<double> svar(n_), scon(m_);
  //   svar.assign(n_, 1.0);
  //   scon.assign(m_, 1.0);
  //   if (state->variable_rescaling)
  //     copy_to_host(svar.data(), state->variable_rescaling, (size_t)n_);
  //   if (state->constraint_rescaling)
  //     copy_to_host(scon.data(), state->constraint_rescaling, (size_t)m_);

  //   // py::array_t<double> x(n_), y(m_);
  //   // {
  //   //   auto xb = x.mutable_unchecked<1>();
  //   //   auto yb = y.mutable_unchecked<1>();
  //   //   for (int i = 0; i < n_; ++i)
  //   //     xb(i) = hx[i] / (svar[i] == 0.0 ? 1.0 : svar[i]);
  //   //   for (int j = 0; j < m_; ++j)
  //   //     yb(j) = hy[j] / (scon[j] == 0.0 ? 1.0 : scon[j]);
  //   // }
  //   const double alpha_x = (state->constraint_bound_rescaling != 0.0)
  //                              ? state->constraint_bound_rescaling
  //                              : 1.0;
  //   const double alpha_y = (state->objective_vector_rescaling != 0.0)
  //                              ? state->objective_vector_rescaling
  //                              : 1.0;

  //   py::array_t<double> x(n_), y(m_);
  //   {
  //     auto xb = x.mutable_unchecked<1>();
  //     auto yb = y.mutable_unchecked<1>();
  //     for (int i = 0; i < n_; ++i)
  //     {
  //       const double sv = (i < (int)svar.size() && svar[i] != 0.0) ? svar[i] : 1.0;
  //       xb(i) = hx[i] / (sv * alpha_x);
  //     }
  //     for (int j = 0; j < m_; ++j)
  //     {
  //       const double sc = (j < (int)scon.size() && scon[j] != 0.0) ? scon[j] : 1.0;
  //       yb(j) = hy[j] / (sc * alpha_y);
  //     }
  //     // 假设你已将未缩放解放到 x_unscaled / y_unscaled（或 x_cache_/y_cache_ 已经填好）
  //     x_cache_.assign(n_, 0.0);
  //     y_cache_.assign(m_, 0.0);
  //     for (int i = 0; i < n_; ++i)
  //       x_cache_[i] = hx[i] / ((svar[i] != 0.0 ? svar[i] : 1.0) * (alpha_x != 0.0 ? alpha_x : 1.0));
  //     for (int j = 0; j < m_; ++j)
  //       y_cache_[j] = hy[j] / ((scon[j] != 0.0 ? scon[j] : 1.0) * (alpha_y != 0.0 ? alpha_y : 1.0));
  //   }

  //   // ====== 输出字典（与原有保持一致） ======
  //   py::dict out;
  //   out["termination_reason"] = py::str(termination_reason_tToString(state->termination_reason));
  //   out["runtime_sec"] = state->cumulative_time_sec;
  //   out["iterations"] = state->total_count;
  //   out["primal_objective"] = state->primal_objective_value;
  //   out["dual_objective"] = state->dual_objective_value;
  //   out["abs_primal_res"] = state->absolute_primal_residual;
  //   out["rel_primal_res"] = state->relative_primal_residual;
  //   out["abs_dual_res"] = state->absolute_dual_residual;
  //   out["rel_dual_res"] = state->relative_dual_residual;
  //   out["abs_obj_gap"] = state->objective_gap;
  //   out["rel_obj_gap"] = state->relative_objective_gap;
  //   out["x"] = x;
  //   out["y"] = y;

  //   // ====== 缓存可直接获取的指标（供 getSolution 返回） ======
  //   iters_cache_ = state->total_count;
  //   solve_time_cache_ = state->cumulative_time_sec;
  //   primal_obj_cache_ = state->primal_objective_value;
  //   dual_obj_cache_ = state->dual_objective_value;
  //   primal_feas_abs_cache_ = state->absolute_primal_residual;
  //   dual_feas_abs_cache_ = state->absolute_dual_residual;
  //   gap_abs_cache_ = fabs(state->primal_objective_value - state->dual_objective_value);
  //   primal_feas_rel_cache_ = state->relative_primal_residual;
  //   dual_feas_rel_cache_ = state->relative_dual_residual;
  //   gap_rel_cache_ = state->relative_objective_gap;
  //   // beta_cache_ = params.reflection_coefficient;

  //   // SaveInfo：仅做直观映射（不引入“平均最优”等推断）
  //   switch (state->termination_reason)
  //   {
  //   case TERMINATION_REASON_OPTIMAL:
  //     saveinfo_cache_ = 1;
  //     break;
  //   case TERMINATION_REASON_TIME_LIMIT:
  //     saveinfo_cache_ = 3;
  //     break;
  //   case TERMINATION_REASON_ITERATION_LIMIT:
  //     saveinfo_cache_ = 3;
  //     break;
  //   case TERMINATION_REASON_PRIMAL_INFEASIBLE:
  //     saveinfo_cache_ = 4;
  //     break;
  //   case TERMINATION_REASON_DUAL_INFEASIBLE:
  //     saveinfo_cache_ = 4;
  //     break;
  //   default:
  //     saveinfo_cache_ = 0;
  //     break;
  //   }

  //   // 步长/重启迭代等内部量：当前不做推断，给出占位
  //   primal_step_cache_ = 0.0;
  //   dual_step_cache_ = 0.0;
  //   last_restart_iter_cache_ = -1;

  //   if (g_free_mode == 0 || g_free_mode == 1)
  //   {
  //     if (state)
  //       pdhg_solver_state_free(state);
  //     state = nullptr;
  //   }
  //   return out;
  // }
  py::dict solve(py::dict user_params)
  {
    if (!problem_ && !device_problem_loaded_)
      throw std::runtime_error("No problem loaded. Call loadData/loadData_csr/loadData_csc/loadData_device_csr first.");

    using clk = std::chrono::high_resolution_clock;
    auto t_all0 = clk::now();

    // ---- 读取用户参数（新增可选项） ----
    bool print_summary = false; // NEW: 默认不打印大表
    bool return_x = true;       // NEW
    bool return_y = true;       // NEW
    bool do_unscale = true;     // NEW

    if (user_params.contains("print_summary"))
      print_summary = user_params["print_summary"].cast<bool>();
    if (user_params.contains("return_x"))
      return_x = user_params["return_x"].cast<bool>();
    if (user_params.contains("return_y"))
      return_y = user_params["return_y"].cast<bool>();
    if (user_params.contains("unscale"))
      do_unscale = user_params["unscale"].cast<bool>();

    pdhg_parameters_t params;
    set_default_parameters(&params);
    std::vector<double> continuation_anchor_primal;
    std::vector<double> continuation_anchor_dual;
    std::vector<double> continuation_current_primal;
    std::vector<double> continuation_current_dual;

    auto pull = [&](const char *k, auto &dst)
    {
      if (user_params.contains(k))
        dst = user_params[k].cast<std::remove_reference_t<decltype(dst)>>();
    };
    pull("verbose", params.verbose);
    pull("verbose_time", params.verbose_time);
    pull("time_sec_limit", params.termination_criteria.time_sec_limit);
    pull("iteration_limit", params.termination_criteria.iteration_limit);
    pull("eps_optimal_relative", params.termination_criteria.eps_optimal_relative);
    if (user_params.contains("eps_optimal_absolute"))
      pull("eps_optimal_absolute", params.termination_criteria.eps_optimal_absolute);
    else if (user_params.contains("eps_optimal_relative"))
      params.termination_criteria.eps_optimal_absolute = params.termination_criteria.eps_optimal_relative;
    pull("eps_feasible_relative", params.termination_criteria.eps_feasible_relative);
    pull("eps_feasible_relative_primal", params.termination_criteria.eps_feasible_relative_primal);
    if (user_params.contains("eps_feasible_absolute_primal"))
      pull("eps_feasible_absolute_primal", params.termination_criteria.eps_feasible_absolute_primal);
    else if (user_params.contains("eps_feasible_relative_primal"))
      params.termination_criteria.eps_feasible_absolute_primal = params.termination_criteria.eps_feasible_relative_primal;
    pull("eps_feasible_relative_dual", params.termination_criteria.eps_feasible_relative_dual);
    if (user_params.contains("eps_feasible_absolute_dual"))
      pull("eps_feasible_absolute_dual", params.termination_criteria.eps_feasible_absolute_dual);
    else if (user_params.contains("eps_feasible_relative_dual"))
      params.termination_criteria.eps_feasible_absolute_dual = params.termination_criteria.eps_feasible_relative_dual;
    auto validate_tolerance_pair = [](const char *name, double absolute_tolerance, double relative_tolerance)
    {
      if (!std::isfinite(absolute_tolerance) || !std::isfinite(relative_tolerance) ||
          absolute_tolerance < 0.0 || relative_tolerance < 0.0 ||
          (absolute_tolerance == 0.0 && relative_tolerance == 0.0))
        throw std::invalid_argument(std::string(name) + " tolerances must be finite, nonnegative, and not both zero.");
    };
    validate_tolerance_pair("objective gap", params.termination_criteria.eps_optimal_absolute, params.termination_criteria.eps_optimal_relative);
    validate_tolerance_pair("primal feasibility", params.termination_criteria.eps_feasible_absolute_primal, params.termination_criteria.eps_feasible_relative_primal);
    validate_tolerance_pair("dual feasibility", params.termination_criteria.eps_feasible_absolute_dual, params.termination_criteria.eps_feasible_relative_dual);
    pull("eps_infeasible", params.termination_criteria.eps_infeasible);
    pull("l_inf_ruiz_iterations", params.l_inf_ruiz_iterations);
    pull("bound_objective_rescaling", params.bound_objective_rescaling);
    pull("has_pock_chambolle_alpha", params.has_pock_chambolle_alpha);
    pull("pock_chambolle_alpha", params.pock_chambolle_alpha);
    pull("termination_evaluation_frequency", params.termination_evaluation_frequency);
    if (user_params.contains("termination_norm"))
      params.termination_norm = parse_termination_norm(user_params["termination_norm"]);
    pull("reflection_coefficient", params.reflection_coefficient);
    pull("use_dual_nnz_gate", params.termination_criteria.use_dual_nnz_gate);
    pull("dual_nnz_factor", params.termination_criteria.dual_nnz_factor);
    pull("dual_nnz_abs_tol", params.termination_criteria.dual_nnz_abs_tol);
    // //
    pull("step_size_method", params.step_size_method);
    pull("step_size_safety", params.step_size_safety);
    pull("power_max_iterations", params.power_max_iterations);
    pull("power_tolerance", params.power_tolerance);
    pull("hybrid_refine_iterations", params.hybrid_refine_iterations);
    pull("stepsize_power_reference", params.stepsize_power_reference);
    pull("stepsize_reference_max_iterations", params.stepsize_reference_max_iterations);
    pull("stepsize_reference_tolerance", params.stepsize_reference_tolerance);
    pull("trace_enabled", params.trace_enabled);
    pull("trace_max_snapshots", params.trace_max_snapshots);
    pull("support_stop_nnz", params.support_stop_nnz);
    pull("support_stop_threshold", params.support_stop_threshold);
    pull("support_stop_obj_rel_change_tol", params.support_stop_obj_rel_change_tol);
    if (user_params.contains("support_limit_mode"))
      params.support_limit_mode = parse_support_limit_mode(user_params["support_limit_mode"]);
    validate_support_limit_params(params);
    pull("enable_objective_gap_divergence_check", params.enable_objective_gap_divergence_check);
    if (user_params.contains("vector_sum_mode"))
      params.vector_sum_mode = parse_vector_sum_mode(user_params["vector_sum_mode"]);
    if (user_params.contains("continuation_state") && !user_params["continuation_state"].is_none())
    {
      py::dict cont = user_params["continuation_state"].cast<py::dict>();
      auto cont_pull_bool = [&](const char *k, bool &dst)
      {
        if (cont.contains(k))
          dst = cont[k].cast<bool>();
      };
      auto cont_pull_double = [&](const char *k, double &dst)
      {
        if (cont.contains(k))
          dst = cont[k].cast<double>();
      };
      auto cont_pull_int = [&](const char *k, int &dst)
      {
        if (cont.contains(k))
          dst = cont[k].cast<int>();
      };
      auto assign_optional_vec = [&](const char *k, std::vector<double> &dst, const int expected_size, const double *&ptr_out, bool &has_out)
      {
        if (!cont.contains(k) || cont[k].is_none())
          return;
        py::array_t<double, py::array::c_style | py::array::forcecast> arr = cont[k].cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
        if ((int)arr.size() != expected_size)
        {
          throw std::runtime_error(std::string("continuation_state[") + k + "] length mismatch.");
        }
        dst.assign(arr.data(), arr.data() + arr.size());
        ptr_out = dst.data();
        has_out = true;
      };

      cont_pull_bool("enabled", params.continuation.enabled);
      cont_pull_bool("reuse_step_size", params.continuation.reuse_step_size);
      cont_pull_bool("apply_restart_on_entry", params.continuation.apply_restart_on_entry);
      cont_pull_double("step_size", params.continuation.step_size);
      cont_pull_double("primal_weight", params.continuation.primal_weight);
      cont_pull_double("primal_weight_error_sum", params.continuation.primal_weight_error_sum);
      cont_pull_double("primal_weight_last_error", params.continuation.primal_weight_last_error);
      cont_pull_double("best_primal_weight", params.continuation.best_primal_weight);
      cont_pull_double("best_primal_dual_residual_gap", params.continuation.best_primal_dual_residual_gap);
      cont_pull_double("previous_restart_dual_residual", params.continuation.previous_restart_dual_residual);
      cont_pull_double("previous_restart_gap", params.continuation.previous_restart_gap);
      cont_pull_int("total_count", params.continuation.total_count);
      cont_pull_int("inner_count", params.continuation.inner_count);
      assign_optional_vec("anchor_primal_unscaled", continuation_anchor_primal, n_, params.continuation.anchor_primal_unscaled, params.continuation.has_anchor_primal);
      assign_optional_vec("anchor_dual_unscaled", continuation_anchor_dual, m_, params.continuation.anchor_dual_unscaled, params.continuation.has_anchor_dual);
      assign_optional_vec("current_primal_unscaled", continuation_current_primal, n_, params.continuation.current_primal_unscaled, params.continuation.has_current_primal);
      assign_optional_vec("current_dual_unscaled", continuation_current_dual, m_, params.continuation.current_dual_unscaled, params.continuation.has_current_dual);
      params.has_continuation = params.continuation.enabled;
    }
    if (params.verbose)
    {
      if (params.termination_criteria.use_dual_nnz_gate)
      {
        printf("Use checkSparsity, tol = %f x nCols, eps = %f\n",
               params.termination_criteria.dual_nnz_factor,
               params.termination_criteria.dual_nnz_abs_tol);
      }
      else
      {
        printf("Do Not Use checkSparsity.\n");
      }
    }

    if (has_init_iterate_)
    {
      params.has_initial_iterate = true;
      if (x0_buf_.ndim() > 0 && (int)x0_buf_.size() == n_)
      {
        params.initial_primal_unscaled = static_cast<const double *>(x0_buf_.data());
      }
      if (y0_buf_.ndim() > 0 && (int)y0_buf_.size() == m_)
      {
        params.initial_dual_unscaled = static_cast<const double *>(y0_buf_.data());
      }
    }

    // ---- 求解 ----
    auto t_opt0 = clk::now();
    pdhg_solver_state_t *state = nullptr;
    if (device_problem_loaded_)
    {
      int current_device = -1;
      cudaError_t device_err = cudaGetDevice(&current_device);
      if (device_err != cudaSuccess)
        throw std::runtime_error(std::string("cudaGetDevice failed: ") + cudaGetErrorString(device_err));
      if (device_ordinal_ >= 0 && current_device != device_ordinal_)
      {
        device_err = cudaSetDevice(device_ordinal_);
        if (device_err != cudaSuccess)
          throw std::runtime_error(std::string("cudaSetDevice failed: ") + cudaGetErrorString(device_err));
      }
      state = optimize_device(&params, &device_problem_);
      if (device_ordinal_ >= 0 && current_device != device_ordinal_)
      {
        device_err = cudaSetDevice(current_device);
        if (device_err != cudaSuccess)
          throw std::runtime_error(std::string("cudaSetDevice restore failed: ") + cudaGetErrorString(device_err));
      }
    }
    else
    {
      state = optimize(&params, problem_);
    }
    auto t_opt1 = clk::now();
    if (!state)
      throw std::runtime_error("cuPDLPx optimize() failed.");

    // ---- 可选打印（避免把大打印算进 Python 的 solve 计时）----
    auto t_prn0 = clk::now();
    if (print_summary)
    {
      print_solver_summary(state);
    }
    auto t_prn1 = clk::now();

    // ---- D→H：拷到缓存（只拷贝一次）----
    auto t_d2h0 = clk::now();
    x_cache_.assign((size_t)n_, 0.0);
    y_cache_.assign((size_t)m_, 0.0);

    if (return_x)
      copy_to_host(x_cache_.data(), state->pdhg_primal_solution, (size_t)n_);
    if (return_y)
      copy_to_host(y_cache_.data(), state->pdhg_dual_solution, (size_t)m_);

    std::vector<double> svar, scon;
    double alpha_x = 1.0, alpha_y = 1.0;
    if (do_unscale)
    {
      if (return_x)
      {
        svar.assign((size_t)n_, 1.0);
        if (state->variable_rescaling)
          copy_to_host(svar.data(), state->variable_rescaling, (size_t)n_);
        if (state->constraint_bound_rescaling != 0.0)
          alpha_x = state->constraint_bound_rescaling;
      }
      if (return_y)
      {
        scon.assign((size_t)m_, 1.0);
        if (state->constraint_rescaling)
          copy_to_host(scon.data(), state->constraint_rescaling, (size_t)m_);
        if (state->objective_vector_rescaling != 0.0)
          alpha_y = state->objective_vector_rescaling;
      }
    }
    auto t_d2h1 = clk::now();

    // ---- 原地 unscale（避免再复制一遍）----
    auto t_uns0 = clk::now();
    if (do_unscale)
    {
      if (return_x)
      {
        const double ax = (alpha_x != 0.0 ? alpha_x : 1.0);
        for (int i = 0; i < n_; ++i)
        {
          const double sv = (i < (int)svar.size() && svar[i] != 0.0) ? svar[i] : 1.0;
          x_cache_[(size_t)i] /= (sv * ax);
        }
      }
      if (return_y)
      {
        const double ay = (alpha_y != 0.0 ? alpha_y : 1.0);
        for (int j = 0; j < m_; ++j)
        {
          const double sc = (j < (int)scon.size() && scon[j] != 0.0) ? scon[j] : 1.0;
          y_cache_[(size_t)j] /= (sc * ay);
        }
      }
    }
    auto t_uns1 = clk::now();

    const double cumulative_time_sec = state->cumulative_time_sec;

    // ---- 组织输出 ----
    py::dict out;
    out["termination_reason"] = py::str(termination_reason_tToString(state->termination_reason));
    out["iterations"] = state->total_count;
    out["primal_objective"] = state->primal_objective_value;
    out["dual_objective"] = state->dual_objective_value;
    out["abs_primal_res"] = state->absolute_primal_residual;
    out["rel_primal_res"] = state->relative_primal_residual;
    out["abs_dual_res"] = state->absolute_dual_residual;
    out["rel_dual_res"] = state->relative_dual_residual;
    out["termination_norm"] = py::str(termination_norm_to_string(state->termination_norm));
    out["termination_abs_primal_res"] = state->termination_absolute_primal_residual;
    out["termination_rel_primal_res"] = state->termination_relative_primal_residual;
    out["termination_abs_dual_res"] = state->termination_absolute_dual_residual;
    out["termination_rel_dual_res"] = state->termination_relative_dual_residual;
    out["termination_objective_vector_norm"] = state->termination_objective_vector_norm;
    out["termination_constraint_bound_norm"] = state->termination_constraint_bound_norm;
    out["abs_obj_gap"] = state->objective_gap;
    out["rel_obj_gap"] = state->relative_objective_gap;
    out["support_stop_obj_rel_change"] = state->support_stop_obj_rel_change;
    out["support_stop_has_last_eval_obj"] = state->support_stop_has_last_eval_obj;
    out["support_stop_last_eval_nnz"] = state->support_stop_last_eval_nnz;
    out["support_limit_mode"] = py::str(support_limit_mode_to_string(params.support_limit_mode));
    out["vector_sum_mode"] = py::str(vector_sum_mode_to_string(params.vector_sum_mode));
    out["matrix_value_mode"] = py::str(matrix_value_mode_to_string(state->matrix_value_mode));
    out["implicit_ax_enabled"] = matrix_value_mode_has_implicit_ax(state->matrix_value_mode);
    out["implicit_aty_enabled"] = matrix_value_mode_has_implicit_aty(state->matrix_value_mode);
    out["implicit_ax_agg"] = py::str(implicit_ax_agg_mode_to_string(state->implicit_ax_agg_mode));
    out["implicit_ax_row0_unique_p50"] = state->implicit_ax_row0_unique_p50;
    out["implicit_ax_row1_unique_p50"] = state->implicit_ax_row1_unique_p50;
    py::dict continuation_state;
    continuation_state["enabled"] = true;
    continuation_state["reuse_step_size"] = true;
    continuation_state["apply_restart_on_entry"] = true;
    continuation_state["anchor_primal_unscaled"] = export_unscaled_primal_buffer(state, state->initial_primal_solution);
    continuation_state["anchor_dual_unscaled"] = export_unscaled_dual_buffer(state, state->initial_dual_solution);
    continuation_state["current_primal_unscaled"] = export_unscaled_primal_buffer(state, state->pdhg_primal_solution);
    continuation_state["current_dual_unscaled"] = export_unscaled_dual_buffer(state, state->pdhg_dual_solution);
    continuation_state["step_size"] = state->step_size;
    continuation_state["primal_weight"] = state->primal_weight;
    continuation_state["primal_weight_error_sum"] = state->primal_weight_error_sum;
    continuation_state["primal_weight_last_error"] = state->primal_weight_last_error;
    continuation_state["best_primal_weight"] = state->best_primal_weight;
    continuation_state["best_primal_dual_residual_gap"] = state->best_primal_dual_residual_gap;
    continuation_state["previous_restart_dual_residual"] = state->previous_restart_dual_residual;
    continuation_state["previous_restart_gap"] = state->previous_restart_gap;
    continuation_state["total_count"] = state->total_count;
    continuation_state["inner_count"] = state->inner_count;
    out["continuation_state"] = std::move(continuation_state);
    // =================== [新增] 注入内存统计 ===================
    // 获取刚才 C++ 记录的峰值
    size_t peak_bytes = get_last_solver_peak_mem();
    double peak_mib = (double)peak_bytes / (1024.0 * 1024.0);
    out["peak_gpu_mem_mib"] = peak_mib; // 传回 Python
    // =========================================================

    // ---- 零拷贝包装 numpy（直接引用 x_cache_/y_cache_）----
    auto t_wrap0 = clk::now();
    if (return_x)
    {
      // shape=(n_,), strides=(sizeof(double),)
      py::array_t<double> x({(py::ssize_t)n_}, {(py::ssize_t)sizeof(double)}, x_cache_.data(), py::none());
      out["x"] = std::move(x);
    }
    else
    {
      out["x"] = py::none();
      x_cache_.clear();
      x_cache_.shrink_to_fit();
    }
    if (return_y)
    {
      py::array_t<double> y({(py::ssize_t)m_}, {(py::ssize_t)sizeof(double)}, y_cache_.data(), py::none());
      out["y"] = std::move(y);
    }
    else
    {
      out["y"] = py::none();
      y_cache_.clear();
      y_cache_.shrink_to_fit();
    }

    if (state->trace_enabled && state->trace_num_snapshots > 0)
    {
      py::list trace_snapshots;
      for (int snap = 0; snap < state->trace_num_snapshots; ++snap)
      {
        py::dict item;
        item["iter"] = state->trace_iters_host[snap];
        item["primal_objective"] = state->trace_primal_objectives_host[snap];
        item["dual_objective"] = state->trace_dual_objectives_host[snap];

        const double *x_ptr = state->trace_primal_solutions_host + (size_t)snap * (size_t)n_;
        py::array_t<double> x_arr({(py::ssize_t)n_});
        std::memcpy(x_arr.mutable_data(), x_ptr, (size_t)n_ * sizeof(double));
        item["x"] = std::move(x_arr);

        const double *y_ptr = state->trace_dual_solutions_host + (size_t)snap * (size_t)m_;
        py::array_t<double> y_arr({(py::ssize_t)m_});
        std::memcpy(y_arr.mutable_data(), y_ptr, (size_t)m_ * sizeof(double));
        item["y"] = std::move(y_arr);

        trace_snapshots.append(std::move(item));
      }
      out["trace_snapshots"] = std::move(trace_snapshots);
    }
    auto t_wrap1 = clk::now();

    // ---- 缓存 summary（供 getSolution() 用）----
    iters_cache_ = state->total_count;
    solve_time_cache_ = cumulative_time_sec;
    primal_obj_cache_ = state->primal_objective_value;
    dual_obj_cache_ = state->dual_objective_value;
    primal_feas_abs_cache_ = state->absolute_primal_residual;
    dual_feas_abs_cache_ = state->absolute_dual_residual;
    gap_abs_cache_ = std::fabs(state->primal_objective_value - state->dual_objective_value);
    primal_feas_rel_cache_ = state->relative_primal_residual;
    dual_feas_rel_cache_ = state->relative_dual_residual;
    gap_rel_cache_ = state->relative_objective_gap;
    // SaveInfo 简单映射
    switch (state->termination_reason)
    {
    case TERMINATION_REASON_OPTIMAL:
    case TERMINATION_REASON_OPTIMAL_WITH_SUPPORT_LIMIT:
      saveinfo_cache_ = 1;
      break;
    case TERMINATION_REASON_TIME_LIMIT:
    case TERMINATION_REASON_ITERATION_LIMIT:
    case TERMINATION_REASON_NUMERICAL_DIVERGENCE:
      saveinfo_cache_ = 3;
      break;
    case TERMINATION_REASON_PRIMAL_INFEASIBLE:
    case TERMINATION_REASON_DUAL_INFEASIBLE:
      saveinfo_cache_ = 4;
      break;
    default:
      saveinfo_cache_ = 0;
      break;
    }
    primal_step_cache_ = 0.0;
    dual_step_cache_ = 0.0;
    last_restart_iter_cache_ = -1;

    if (g_free_mode == 0 || g_free_mode == 1)
    {
      if (state)
        pdhg_solver_state_free(state);
      state = nullptr;
    }

    auto t_all1 = clk::now();

    auto ms = [](auto a, auto b)
    { return std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count(); };
    const double runtime_sec_wall = std::chrono::duration<double>(t_all1 - t_all0).count();
    py::dict timing;
    timing["optimize_ms"] = ms(t_opt0, t_opt1);
    timing["print_ms"] = ms(t_prn0, t_prn1);
    timing["d2h_ms"] = ms(t_d2h0, t_d2h1);
    timing["unscale_ms"] = ms(t_uns0, t_uns1);
    timing["wrap_numpy_ms"] = ms(t_wrap0, t_wrap1);
    timing["total_ms"] = ms(t_all0, t_all1);
    out["runtime_sec"] = runtime_sec_wall;
    out["cumulative_time_sec"] = cumulative_time_sec;
    out["timing_ms"] = std::move(timing);
    solve_time_cache_ = runtime_sec_wall;
    // if (params.verbose_time)
    // {
    //   printf("=== timing (ms) ===\n");
    //   printf("total_ms: %lld\n", timing["total_ms"].cast<long long>());
    //   printf("  optimize_ms: %lld\n", timing["optimize_ms"].cast<long long>());
    //   printf("  print_ms: %lld\n", timing["print_ms"].cast<long long>());
    //   printf("  d2h_ms: %lld\n", timing["d2h_ms"].cast<long long>());
    //   printf("  unscale_ms: %lld\n", timing["unscale_ms"].cast<long long>());
    //   printf("  wrap_numpy_ms: %lld\n", timing["wrap_numpy_ms"].cast<long long>());
    // }
    return out;
  }

  // ====== 新增：以 Python dict 返回一次 solve() 的统计量（仅含可直接获得者） ======
  py::dict getSolution() const
  {
    py::dict d;
    d["iters"] = iters_cache_;
    d["solve_time"] = solve_time_cache_;

    d["PrimalObj"] = primal_obj_cache_;
    d["DualObj"] = dual_obj_cache_;
    // 绝对/相对误差与间隙
    d["PrimalFeas"] = primal_feas_abs_cache_;
    d["DualFeas"] = dual_feas_abs_cache_;
    d["DualityGap"] = gap_abs_cache_;
    d["PrimalFeasRel"] = primal_feas_rel_cache_;
    d["DualFeasRel"] = dual_feas_rel_cache_;
    d["RelObjGap"] = gap_rel_cache_;

    // 平均量：当前 solver 未提供 → 先返回 None 占位（避免 KeyError）
    d["PrimalFeasAvg"] = py::none();
    d["DualFeasAvg"] = py::none();
    d["DualityGapAvg"] = py::none();
    d["PrimalFeasAvgRel"] = py::none();
    d["DualFeasAvgRel"] = py::none();
    d["RelObjGapAverage"] = py::none();

    // 其它控制量
    d["SaveInfo"] = saveinfo_cache_;
    d["LastRestartIter"] = last_restart_iter_cache_; // -1 表示未知/未提供
    d["PrimalStep"] = primal_step_cache_;            // 0.0 仅占位，不做推断
    d["DualStep"] = dual_step_cache_;                // 0.0 仅占位，不做推断
    d["Beta"] = beta_cache_;                         // 直接来自 params.reflection_coefficient

    // 返回最近一次解向量（如果没有求解过则返回 None）
    // ---------- x: always return a 1D numpy array ----------
    {
      const py::ssize_t nx = static_cast<py::ssize_t>(x_cache_.size());
      py::array_t<double> x_arr(nx);
      if (nx > 0)
      {
        auto xb = x_arr.mutable_unchecked<1>();
        for (py::ssize_t i = 0; i < nx; ++i)
          xb(i) = x_cache_[static_cast<size_t>(i)];
      }
      d["x"] = std::move(x_arr); // shape = (nx,)
    }

    // ---------- y: always return a 1D numpy array ----------
    {
      const py::ssize_t ny = static_cast<py::ssize_t>(y_cache_.size());
      py::array_t<double> y_arr(ny);
      if (ny > 0)
      {
        auto yb = y_arr.mutable_unchecked<1>();
        for (py::ssize_t j = 0; j < ny; ++j)
          yb(j) = y_cache_[static_cast<size_t>(j)];
      }
      d["y"] = std::move(y_arr); // shape = (ny,)
    }

    return d;
  }

  // Set initial iterate in UN-SCALED space.
  // 允许传 None：只设 x0 或只设 y0
  void setInitSol(py::object x0_obj, py::object y0_obj)
  {
    // 若你有 loadData 才知道 n_/m_，可以做个防御
    if (n_ <= 0 || m_ <= 0)
    {
      throw std::runtime_error("setInitSol must be called after loadData()");
    }

    x0_buf_ = py::array_t<double>();
    y0_buf_ = py::array_t<double>();
    has_init_iterate_ = false;

    if (!x0_obj.is_none())
    {
      auto x0 = x0_obj.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
      if ((int)x0.size() != n_)
      {
        throw std::runtime_error("x0 length mismatch with num_variables");
      }
      x0_buf_ = x0; // 保存以保证生命周期贯穿 solve()
      has_init_iterate_ = true;
    }
    if (!y0_obj.is_none())
    {
      auto y0 = y0_obj.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
      if ((int)y0.size() != m_)
      {
        throw std::runtime_error("y0 length mismatch with num_constraints");
      }
      y0_buf_ = y0;
      has_init_iterate_ = true;
    }
  }

  int num_variables() const { return n_; }
  int num_constraints() const { return m_; }
  int nnz() const { return nnz_; }

private:
  lp_problem_t *problem_;
  device_lp_problem_t device_problem_{};
  bool device_problem_loaded_;
  int device_ordinal_ = -1;
  int m_, n_, nnz_;
  double last_load_validate_time_sec_;
  double last_load_copy_vectors_time_sec_;
  double last_load_constraint_bounds_time_sec_;
  double last_load_copy_csc_triplets_time_sec_;
  double last_load_csc_to_csr_time_sec_;
  double last_load_problem_bind_time_sec_;
  double last_load_total_time_sec_;
  std::vector<double> var_lb_, var_ub_, obj_, rhs_;
  std::vector<double> con_lb_, con_ub_;
  std::vector<int> tmp_col_ptr_, tmp_row_idx_;
  std::vector<double> tmp_val_;
  std::vector<int> csr_row_ptr_, csr_col_idx_;
  std::vector<double> csr_val_;
  // ==== 缓存最近一次的解向量 ====
  std::vector<double> x_cache_;
  std::vector<double> y_cache_;

  // ==== 缓存最近一次 solve() 的统计量 ====
  int iters_cache_ = 0;
  double solve_time_cache_ = 0.0;
  double primal_obj_cache_ = 0.0, dual_obj_cache_ = 0.0;
  double primal_feas_abs_cache_ = 0.0, dual_feas_abs_cache_ = 0.0, gap_abs_cache_ = 0.0;
  double primal_feas_rel_cache_ = 0.0, dual_feas_rel_cache_ = 0.0, gap_rel_cache_ = 0.0;
  int saveinfo_cache_ = 0;
  double beta_cache_ = 0.0;
  double primal_step_cache_ = 0.0, dual_step_cache_ = 0.0; // 占位
  int last_restart_iter_cache_ = -1;                       // 占位

  py::array_t<double> x0_buf_;
  py::array_t<double> y0_buf_;
  bool has_init_iterate_ = false;
  py::object d_row_ptr_owner_ = py::none();
  py::object d_col_idx_owner_ = py::none();
  py::object d_val_owner_ = py::none();
  py::object d_c_owner_ = py::none();
  py::object d_rhs_owner_ = py::none();
  py::object d_lb_owner_ = py::none();
  py::object d_ub_owner_ = py::none();
  py::object d_constraint_rescaling_owner_ = py::none();
  py::object d_variable_rescaling_owner_ = py::none();
};

// ====== Python 侧辅助 ======
static py::dict make_default_params()
{
  pdhg_parameters_t p;
  set_default_parameters(&p);
  py::dict d;
  d["l_inf_ruiz_iterations"] = p.l_inf_ruiz_iterations;
  d["has_pock_chambolle_alpha"] = p.has_pock_chambolle_alpha;
  d["pock_chambolle_alpha"] = p.pock_chambolle_alpha;
  d["bound_objective_rescaling"] = p.bound_objective_rescaling;
  d["verbose"] = p.verbose;
  d["verbose_time"] = p.verbose_time;
  d["termination_evaluation_frequency"] = p.termination_evaluation_frequency;
  d["termination_norm"] = py::str(termination_norm_to_string(p.termination_norm));
  d["reflection_coefficient"] = p.reflection_coefficient;
  d["time_sec_limit"] = p.termination_criteria.time_sec_limit;
  d["iteration_limit"] = p.termination_criteria.iteration_limit;
  d["eps_optimal_absolute"] = p.termination_criteria.eps_optimal_absolute;
  d["eps_optimal_relative"] = p.termination_criteria.eps_optimal_relative;
  d["eps_feasible_relative"] = p.termination_criteria.eps_feasible_relative;
  d["eps_feasible_absolute_primal"] = p.termination_criteria.eps_feasible_absolute_primal;
  d["eps_feasible_relative_primal"] = p.termination_criteria.eps_feasible_relative_primal;
  d["eps_feasible_absolute_dual"] = p.termination_criteria.eps_feasible_absolute_dual;
  d["eps_feasible_relative_dual"] = p.termination_criteria.eps_feasible_relative_dual;
  d["eps_infeasible"] = p.termination_criteria.eps_infeasible;
  d["use_dual_nnz_gate"] = p.termination_criteria.use_dual_nnz_gate; // 0
  d["dual_nnz_factor"] = p.termination_criteria.dual_nnz_factor;     // 2.0
  d["dual_nnz_abs_tol"] = p.termination_criteria.dual_nnz_abs_tol;   // 0.0
  d["vector_sum_mode"] = vector_sum_mode_to_string(p.vector_sum_mode);

  // CN: 这一组默认值必须和 native cupdlpx 以及 Python wrapper 中的通用默认完全一致，
  // CN: 否则 Python 侧看到的“默认值”会和真实求解行为脱节，后续又会引入 host/device 路径漂移。
  // EN: This default bundle must exactly match the native cupdlpx defaults and the common defaults in the Python wrapper,
  // EN: otherwise the Python-visible “defaults” drift away from the actual solver behavior and reintroduce host/device path divergence.
  d["step_size_method"] = 3;     // 0=power,1=one_inf,2=hybrid,3=constant
  d["step_size_safety"] = 0.998;

  d["power_max_iterations"] = 5000;
  d["power_tolerance"] = 1e-4;
  d["hybrid_refine_iterations"] = 100;

  d["stepsize_power_reference"] = false;
  d["stepsize_reference_max_iterations"] = 5000;
  d["stepsize_reference_tolerance"] = 1e-4;
  d["support_stop_nnz"] = 0;
  d["support_stop_threshold"] = 1e-8;
  // CN: 暂时复用旧键名；当前它控制的是 0 < (primal-dual)/primal <= tol 这一 support-stop 判据。
  // EN: Temporarily reuse the legacy key name; it now controls the support-stop criterion 0 < (primal-dual)/primal <= tol.
  d["support_stop_obj_rel_change_tol"] = 1e-3;
  d["support_limit_mode"] = py::str(support_limit_mode_to_string(p.support_limit_mode));
  d["enable_objective_gap_divergence_check"] = p.enable_objective_gap_divergence_check;
  return d;
}
static void reset_cuda_device()
{
  cudaDeviceSynchronize();
  cudaDeviceReset();
}
static py::dict benchmark_spmv_device_csr(py::object row_ptr,
                                          py::object col_ind,
                                          py::object values,
                                          int m,
                                          int n,
                                          py::object row_ptr_t,
                                          py::object col_ind_t,
                                          py::object values_t,
                                          py::object x,
                                          py::object y,
                                          py::object out_A,
                                          py::object out_AT,
                                          int warmup = 10,
                                          int repeat = 50,
                                          py::object matrix_value_mode = py::str("explicit"),
                                          py::object constraint_rescaling = py::none(),
                                          py::object variable_rescaling = py::none())
{
  if (m <= 0 || n <= 0)
    throw std::runtime_error("m,n must be positive.");
  if (warmup < 0 || repeat <= 0)
    throw std::runtime_error("warmup must be non-negative and repeat must be positive.");

  const matrix_value_mode_t value_mode = parse_matrix_value_mode(matrix_value_mode);
  int device_idx = -1;
  auto expect_same_device = [&](int got, const char *name) {
    if (device_idx < 0)
      device_idx = got;
    else if (got != device_idx)
      throw std::runtime_error(std::string(name) + " is on a different CUDA device than previous tensors.");
  };

  py::object row_ptr_owner, col_ind_owner, values_owner = py::none();
  py::object row_ptr_t_owner, col_ind_t_owner, values_t_owner = py::none();
  py::object x_owner, y_owner, out_A_owner, out_AT_owner;
  py::object constraint_rescaling_owner = py::none();
  py::object variable_rescaling_owner = py::none();
  int got = require_cuda_tensor(row_ptr, "row_ptr", row_ptr_owner, 1, "torch.int32", {(py::ssize_t)m + 1});
  expect_same_device(got, "row_ptr");
  got = require_cuda_tensor(col_ind, "col_ind", col_ind_owner, 1, "torch.int32");
  expect_same_device(got, "col_ind");
  got = require_cuda_tensor(row_ptr_t, "row_ptr_t", row_ptr_t_owner, 1, "torch.int32", {(py::ssize_t)n + 1});
  expect_same_device(got, "row_ptr_t");
  got = require_cuda_tensor(col_ind_t, "col_ind_t", col_ind_t_owner, 1, "torch.int32");
  expect_same_device(got, "col_ind_t");
  got = require_cuda_tensor(x, "x", x_owner, 1, "torch.float64", {(py::ssize_t)n});
  expect_same_device(got, "x");
  got = require_cuda_tensor(y, "y", y_owner, 1, "torch.float64", {(py::ssize_t)m});
  expect_same_device(got, "y");
  got = require_cuda_tensor(out_A, "out_A", out_A_owner, 1, "torch.float64", {(py::ssize_t)m});
  expect_same_device(got, "out_A");
  got = require_cuda_tensor(out_AT, "out_AT", out_AT_owner, 1, "torch.float64", {(py::ssize_t)n});
  expect_same_device(got, "out_AT");
  if (!constraint_rescaling.is_none())
  {
    got = require_cuda_tensor(constraint_rescaling, "constraint_rescaling", constraint_rescaling_owner, 1, "torch.float64", {(py::ssize_t)m});
    expect_same_device(got, "constraint_rescaling");
  }
  if (!variable_rescaling.is_none())
  {
    got = require_cuda_tensor(variable_rescaling, "variable_rescaling", variable_rescaling_owner, 1, "torch.float64", {(py::ssize_t)n});
    expect_same_device(got, "variable_rescaling");
  }

  const py::ssize_t nnz = col_ind.attr("numel")().cast<py::ssize_t>();
  if (col_ind_t.attr("numel")().cast<py::ssize_t>() != nnz)
    throw std::runtime_error("col_ind_t length must match col_ind length.");
  const long long row_ptr_last = tensor_scalar_to_long_long(row_ptr_owner.attr("__getitem__")(m));
  const long long row_ptr_t_last = tensor_scalar_to_long_long(row_ptr_t_owner.attr("__getitem__")(n));
  if (row_ptr_last != nnz || row_ptr_t_last != nnz)
    throw std::runtime_error("CSR row_ptr endpoints must match nnz.");

  double *values_ptr = nullptr;
  double *values_t_ptr = nullptr;
  if (value_mode == MATRIX_VALUES_EXPLICIT)
  {
    got = require_cuda_tensor(values, "values", values_owner, 1, "torch.float64", {nnz});
    expect_same_device(got, "values");
    got = require_cuda_tensor(values_t, "values_t", values_t_owner, 1, "torch.float64", {nnz});
    expect_same_device(got, "values_t");
    values_ptr = reinterpret_cast<double *>(values_owner.attr("data_ptr")().cast<std::uintptr_t>());
    values_t_ptr = reinterpret_cast<double *>(values_t_owner.attr("data_ptr")().cast<std::uintptr_t>());
  }
  else if (value_mode == MATRIX_VALUES_IMPLICIT_ATY)
  {
    got = require_cuda_tensor(values, "values", values_owner, 1, "torch.float64", {nnz});
    expect_same_device(got, "values");
    if (!values_t.is_none())
      throw std::runtime_error("implicit_aty benchmark requires values_t=None.");
    values_ptr = reinterpret_cast<double *>(values_owner.attr("data_ptr")().cast<std::uintptr_t>());
    values_t_ptr = nullptr;
  }
  else if (value_mode == MATRIX_VALUES_IMPLICIT_AX)
  {
    if (!values.is_none())
      throw std::runtime_error("implicit_ax benchmark requires values=None.");
    got = require_cuda_tensor(values_t, "values_t", values_t_owner, 1, "torch.float64", {nnz});
    expect_same_device(got, "values_t");
    values_ptr = nullptr;
    values_t_ptr = reinterpret_cast<double *>(values_t_owner.attr("data_ptr")().cast<std::uintptr_t>());
  }
  else if (!values.is_none() || !values_t.is_none())
  {
    throw std::runtime_error("implicit_both benchmark requires values=None and values_t=None.");
  }

  cu_sparse_matrix_csr_t A;
  cu_sparse_matrix_csr_t AT;
  std::memset(&A, 0, sizeof(A));
  std::memset(&AT, 0, sizeof(AT));
  A.num_rows = m;
  A.num_cols = n;
  A.num_nonzeros = (int)nnz;
  A.row_ptr = reinterpret_cast<int *>(row_ptr_owner.attr("data_ptr")().cast<std::uintptr_t>());
  A.col_ind = reinterpret_cast<int *>(col_ind_owner.attr("data_ptr")().cast<std::uintptr_t>());
  A.val = values_ptr;
  AT.num_rows = n;
  AT.num_cols = m;
  AT.num_nonzeros = (int)nnz;
  AT.row_ptr = reinterpret_cast<int *>(row_ptr_t_owner.attr("data_ptr")().cast<std::uintptr_t>());
  AT.col_ind = reinterpret_cast<int *>(col_ind_t_owner.attr("data_ptr")().cast<std::uintptr_t>());
  AT.val = values_t_ptr;

  pdhg_solver_state_t state;
  std::memset(&state, 0, sizeof(state));
  state.num_constraints = m;
  state.num_variables = n;
  state.num_blocks_dual = (m + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  state.num_blocks_primal = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  state.constraint_matrix = &A;
  state.constraint_matrix_t = &AT;
  state.matrix_value_mode = value_mode;
  state.pdhg_primal_solution = reinterpret_cast<double *>(x_owner.attr("data_ptr")().cast<std::uintptr_t>());
  state.pdhg_dual_solution = reinterpret_cast<double *>(y_owner.attr("data_ptr")().cast<std::uintptr_t>());
  state.primal_product = reinterpret_cast<double *>(out_A_owner.attr("data_ptr")().cast<std::uintptr_t>());
  state.dual_product = reinterpret_cast<double *>(out_AT_owner.attr("data_ptr")().cast<std::uintptr_t>());
  state.constraint_rescaling = constraint_rescaling_owner.is_none()
      ? nullptr
      : reinterpret_cast<double *>(constraint_rescaling_owner.attr("data_ptr")().cast<std::uintptr_t>());
  state.variable_rescaling = variable_rescaling_owner.is_none()
      ? nullptr
      : reinterpret_cast<double *>(variable_rescaling_owner.attr("data_ptr")().cast<std::uintptr_t>());

  if (matrix_value_mode_has_explicit_a(value_mode))
  {
    CUSPARSE_CHECK(cusparseCreate(&state.sparse_handle));
    CUSPARSE_CHECK(cusparseCreateCsr(&state.matA, m, n, (int)nnz, A.row_ptr, A.col_ind, A.val,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&state.vec_primal_sol, n, state.pdhg_primal_solution, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&state.vec_primal_prod, m, state.primal_product, CUDA_R_64F));
    size_t primal_buffer_size = 0;
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(state.sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &HOST_ONE, state.matA, state.vec_primal_sol, &HOST_ZERO,
                                           state.vec_primal_prod, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2,
                                           &primal_buffer_size));
    CUDA_CHECK(cudaMalloc(&state.primal_spmv_buffer, primal_buffer_size));
  }
  if (matrix_value_mode_has_explicit_at(value_mode))
  {
    if (!state.sparse_handle)
      CUSPARSE_CHECK(cusparseCreate(&state.sparse_handle));
    CUSPARSE_CHECK(cusparseCreateCsr(&state.matAt, n, m, (int)nnz, AT.row_ptr, AT.col_ind, AT.val,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&state.vec_dual_sol, m, state.pdhg_dual_solution, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&state.vec_dual_prod, n, state.dual_product, CUDA_R_64F));
    size_t dual_buffer_size = 0;
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(state.sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &HOST_ONE, state.matAt, state.vec_dual_sol, &HOST_ZERO,
                                           state.vec_dual_prod, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2,
                                           &dual_buffer_size));
    CUDA_CHECK(cudaMalloc(&state.dual_spmv_buffer, dual_buffer_size));
  }

  auto measure_ms = [&](bool transpose) -> float {
    for (int i = 0; i < warmup; ++i)
    {
      if (transpose)
        spmv_AT(&state, state.pdhg_dual_solution, state.dual_product);
      else
        spmv_A(&state, state.pdhg_primal_solution, state.primal_product);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventRecord(e0));
    for (int i = 0; i < repeat; ++i)
    {
      if (transpose)
        spmv_AT(&state, state.pdhg_dual_solution, state.dual_product);
      else
        spmv_A(&state, state.pdhg_primal_solution, state.primal_product);
    }
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, e0, e1));
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
    return elapsed_ms / (float)repeat;
  };

  const float ax_ms = measure_ms(false);
  const float aty_ms = measure_ms(true);
  CUDA_CHECK(cudaDeviceSynchronize());

  if (state.implicit_spmv_heavy_rows)
    CUDA_CHECK(cudaFree(state.implicit_spmv_heavy_rows));
  if (state.primal_spmv_buffer)
    CUDA_CHECK(cudaFree(state.primal_spmv_buffer));
  if (state.dual_spmv_buffer)
    CUDA_CHECK(cudaFree(state.dual_spmv_buffer));
  if (state.vec_primal_sol)
    CUSPARSE_CHECK(cusparseDestroyDnVec(state.vec_primal_sol));
  if (state.vec_dual_sol)
    CUSPARSE_CHECK(cusparseDestroyDnVec(state.vec_dual_sol));
  if (state.vec_primal_prod)
    CUSPARSE_CHECK(cusparseDestroyDnVec(state.vec_primal_prod));
  if (state.vec_dual_prod)
    CUSPARSE_CHECK(cusparseDestroyDnVec(state.vec_dual_prod));
  if (state.matA)
    CUSPARSE_CHECK(cusparseDestroySpMat(state.matA));
  if (state.matAt)
    CUSPARSE_CHECK(cusparseDestroySpMat(state.matAt));
  if (state.sparse_handle)
    CUSPARSE_CHECK(cusparseDestroy(state.sparse_handle));

  py::dict out;
  out["ax_ms"] = ax_ms;
  out["aty_ms"] = aty_ms;
  out["total_ms"] = ax_ms + aty_ms;
  out["matrix_value_mode"] = py::str(matrix_value_mode_to_string(value_mode));
  out["implicit_ax_agg"] = py::str(implicit_ax_agg_mode_to_string(state.implicit_ax_agg_mode));
  out["implicit_ax_row0_unique_p50"] = state.implicit_ax_row0_unique_p50;
  out["implicit_ax_row1_unique_p50"] = state.implicit_ax_row1_unique_p50;
  return out;
}

static void set_free_mode(int mode)
{
  if (mode < 0 || mode > 2)
    throw std::runtime_error("free_mode must be 0,1,2.");
  g_free_mode = mode;
}

// ====== PYBIND11 ======
PYBIND11_MODULE(pycupdlpx, m)
{
  m.doc() = "Python bindings for cuPDLPx (direct CSC input, safe device->host copy, unscale)";

  py::class_<CupdlpxHolder>(m, "cupdlpx")
      .def(py::init<>())
      .def("loadData_csc", &CupdlpxHolder::loadData_csc,
           py::arg("indptr"), py::arg("indices"), py::arg("data"),
           py::arg("m"), py::arg("n"),
           py::arg("c"), py::arg("rhs"),
           py::arg("lb"), py::arg("ub"),
           py::arg("nEqs"),
           "Load problem in CSC (indptr, indices, data) + vectors (c, rhs, lb, ub).")
      .def("loadData_csr", &CupdlpxHolder::loadData_csr,
           py::arg("indptr"), py::arg("indices"), py::arg("data"),
           py::arg("m"), py::arg("n"),
           py::arg("c"), py::arg("rhs"),
           py::arg("lb"), py::arg("ub"),
           py::arg("nEqs"),
           "Load problem in CSR (indptr, indices, data) + vectors (c, rhs, lb, ub).")
      .def("loadData", &CupdlpxHolder::loadData,
           py::arg("A"), py::arg("c"), py::arg("rhs"), py::arg("lb"), py::arg("ub"), py::arg("nEqs"),
           "Load problem from a scipy.sparse.csc_matrix or csr_matrix A plus vectors (c, rhs, lb, ub).")
      .def("loadData_device_csr", &CupdlpxHolder::loadData_device_csr,
           py::arg("row_ptr"), py::arg("col_ind"), py::arg("values"),
           py::arg("m"), py::arg("n"),
           py::arg("c"), py::arg("rhs"), py::arg("lb"), py::arg("ub"), py::arg("nEqs"),
           py::arg("variable_bound_mode") = py::str("explicit"),
           py::arg("matrix_value_mode") = py::str("explicit"),
           py::arg("constraint_rescaling") = py::none(),
           py::arg("variable_rescaling") = py::none(),
           py::arg("constraint_bound_rescaling") = 1.0,
           py::arg("objective_vector_rescaling") = 1.0,
           py::arg("original_objective_vector_norm") = py::none(),
           py::arg("original_constraint_bound_norm") = py::none(),
           py::arg("has_precomputed_rescaling") = false,
           py::arg("original_objective_vector_linf_norm") = py::none(),
           py::arg("original_constraint_bound_linf_norm") = py::none(),
           "Load problem from CUDA torch.Tensor CSR inputs without host-side sparse copies.")
      .def("getLoadStats", &CupdlpxHolder::getLoadStats,
           "Return timing breakdown for the most recent loadData/loadData_csc call.")
      .def("solve", &CupdlpxHolder::solve, py::arg("params") = py::dict(),
           "Solve and return a dict with summary and unscaled (x,y).")
      .def("getSolution", &CupdlpxHolder::getSolution,
           "Return solver stats (iters/time/feasibility/gap and placeholders).")
      .def("setInitSol", &CupdlpxHolder::setInitSol,
           py::arg("x0"), py::arg("y0"),
           R"doc(Set initial iterate (unscaled) for PDHG: x0 length = n, y0 length = m.
Pass None for either to skip.)doc")
      .def_property_readonly("num_variables", &CupdlpxHolder::num_variables)
      .def_property_readonly("num_constraints", &CupdlpxHolder::num_constraints)
      .def_property_readonly("nnz", &CupdlpxHolder::nnz);

  m.def("make_default_params", &make_default_params);
  m.def("reset_cuda_device", &reset_cuda_device);
  m.def("set_free_mode", &set_free_mode, "0:free all, 1:free state only (default), 2:free problem only");
  m.def("benchmark_spmv_device_csr", &benchmark_spmv_device_csr,
        py::arg("row_ptr"), py::arg("col_ind"), py::arg("values"),
        py::arg("m"), py::arg("n"),
        py::arg("row_ptr_t"), py::arg("col_ind_t"), py::arg("values_t"),
        py::arg("x"), py::arg("y"), py::arg("out_A"), py::arg("out_AT"),
        py::arg("warmup") = 10, py::arg("repeat") = 50,
        py::arg("matrix_value_mode") = py::str("explicit"),
        py::arg("constraint_rescaling") = py::none(),
        py::arg("variable_rescaling") = py::none(),
        "Benchmark explicit cuSPARSE or implicit-ones CSR SpMV on CUDA tensors.");
  m.def("print_solver_summary", [](py::capsule state_capsule)
        {
    auto* state = state_capsule.get_pointer<pdhg_solver_state_t>();
    if (!state) throw std::runtime_error("Invalid solver state capsule.");
    print_solver_summary(state); });

  // 不在解释器退出阶段强制 cudaDeviceReset():
  // 若与 CuPy 共存，提前 reset 会导致 CuPy 模块析构时出现
  // CUDA_ERROR_CONTEXT_IS_DESTROYED / CUDA_ERROR_INVALID_HANDLE。
  // 需要时可显式调用 reset_cuda_device()。
}
