/*
Copyright 2025 Haihao Lu

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "solver.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>

#include <vector>
#include <algorithm>
#include <cstring>
#include <stdexcept>

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/count.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/execution_policy.h>
#include <cmath>    // std::fabs


#include <cfloat>   // DBL_MAX

#include <chrono>
#include <cuda_runtime.h>
#include <cstdio>
// =================== SMART PATCH START ===================
// 仅在 CUDA 版本低于 11.7 (11070) 时启用此补丁
#include <cuda_runtime_api.h>

#if defined(CUDART_VERSION) && CUDART_VERSION < 11070

#include <cublas_v2.h>

// 1. 补丁：cublasGetStatusName
#ifndef CUBLAS_GET_STATUS_NAME_H_
#define CUBLAS_GET_STATUS_NAME_H_
static const char* cublasGetStatusName(cublasStatus_t status) {
    switch(status) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
        default: return "CUBLAS_STATUS_UNKNOWN";
    }
}
#endif

// 2. 补丁：cublasDnrm2_v2_64
// 旧版本没有 64 位接口，回退到标准接口
#ifndef cublasDnrm2_v2_64
#define cublasDnrm2_v2_64 cublasDnrm2
#endif

#endif
// =================== SMART PATCH END ===================

using clk = std::chrono::high_resolution_clock;
__global__ void compute_next_pdhg_primal_solution_kernel(
    const double *current_primal, double *reflected_primal, const double *dual_product,
    const double *objective, const double *var_lb, const double *var_ub,
    int n, double step_size);
__global__ void compute_next_pdhg_primal_solution_major_kernel(
    const double *current_primal, double *pdhg_primal, double *reflected_primal,
    const double *dual_product, const double *objective, const double *var_lb,
    const double *var_ub, int n, double step_size, double *dual_slack);
__global__ void compute_next_pdhg_dual_solution_kernel(
    const double *current_dual, double *reflected_dual, const double *primal_product,
    const double *const_lb, const double *const_ub, int n, double step_size);
__global__ void compute_next_pdhg_dual_solution_major_kernel(
    const double *current_dual, double *pdhg_dual, double *reflected_dual,
    const double *primal_product, const double *const_lb, const double *const_ub,
    int n, double step_size);
__global__ void halpern_update_kernel(
    const double *initial_primal, double *current_primal, const double *reflected_primal,
    const double *initial_dual, double *current_dual, const double *reflected_dual,
    int n_vars, int n_cons, double weight, double reflection_coeff);
__global__ void compute_delta_solution_kernel(
    const double *initial_primal, const double *pdhg_primal, double *delta_primal,
    const double *initial_dual, const double *pdhg_dual, double *delta_dual,
    int n_vars, int n_cons);
static void compute_next_pdhg_primal_solution(pdhg_solver_state_t *state);
static void compute_next_pdhg_dual_solution(pdhg_solver_state_t *state);
static void halpern_update(pdhg_solver_state_t *state, double reflection_coefficient);
static void perform_restart(pdhg_solver_state_t *state, const pdhg_parameters_t *cudaMemsetParams);
static void initialize_step_size_and_primal_weight(pdhg_solver_state_t *state, const pdhg_parameters_t *params);

static pdhg_solver_state_t *initialize_solver_state(
    const lp_problem_t *original_problem,
    const rescale_info_t *rescale_info,
    const pdhg_parameters_t *params);
typedef struct
{
    device_lp_problem_t scaled_problem;
    double *constraint_lower_bound;
    double *constraint_upper_bound;
    int *constraint_matrix_t_row_pointers;
    int *constraint_matrix_t_col_indices;
    double *constraint_matrix_t_values;
    double *con_rescale;
    double *var_rescale;
    double con_bound_rescale;
    double obj_vec_rescale;
    double rescaling_time_sec;
} device_rescale_info_t;

static pdhg_solver_state_t *initialize_solver_state_device_no_rescale(
    const device_lp_problem_t *device_problem,
    const pdhg_parameters_t *params);
static pdhg_solver_state_t *initialize_solver_state_device_rescaled(
    const device_lp_problem_t *original_problem,
    device_rescale_info_t *rescale_info,
    const pdhg_parameters_t *params);
static device_rescale_info_t *rescale_problem_device(
    const pdhg_parameters_t *params,
    const device_lp_problem_t *original_problem);
static void device_rescale_info_free(device_rescale_info_t *info);
static pdhg_solver_state_t *run_solver_loop(
    pdhg_solver_state_t *state,
    const pdhg_parameters_t *params,
    double t_setup);
static void compute_fixed_point_error(pdhg_solver_state_t *state);
static void maybe_record_trace_snapshot(pdhg_solver_state_t *state);
static void maybe_stop_on_support_limit(pdhg_solver_state_t *state, const pdhg_parameters_t *params);
static void maybe_stop_on_objective_gap_divergence(pdhg_solver_state_t *state, const pdhg_parameters_t *params);
static void apply_continuation_state(pdhg_solver_state_t *state, const pdhg_parameters_t *params);
void rescale_info_free(rescale_info_t *info);


static inline double toMiB(std::size_t b){ return b / (1024.0 * 1024.0); }

// =================== [新增] 全局内存统计变量 ===================
static size_t g_solver_peak_mem_bytes = 0;

extern "C" {
    // 供 binding 调用的接口
    size_t get_last_solver_peak_mem() {
        return g_solver_peak_mem_bytes;
    }
}
// =============================================================

// 在任意位置调用以打印当前 GPU 显存用量；建议在主循环前调用一次
static inline void print_gpu_mem_once(const char* tag = "before loop", bool sync_before_query = true) {
    if (sync_before_query) cudaDeviceSynchronize();   // 确保前面 kernel/异步分配都完成
    std::size_t freeB = 0, totalB = 0;
    cudaError_t st = cudaMemGetInfo(&freeB, &totalB);
    if (st != cudaSuccess) {
        std::fprintf(stderr, "[GPU MEM] %s: cudaMemGetInfo failed: %s\n",
                     tag, cudaGetErrorString(st));
        return;
    }
    std::size_t usedB = totalB - freeB;
    int dev = 0; cudaGetDevice(&dev);
    std::printf("[GPU MEM][dev=%d] %s: used=%.2f MiB  free=%.2f MiB  total=%.2f MiB\n",
                dev, tag, toMiB(usedB), toMiB(freeB), toMiB(totalB));
    std::fflush(stdout);
}

static void compact_trace_snapshots_inplace(pdhg_solver_state_t *state) {
    if (!state->trace_enabled || state->trace_num_snapshots <= 1) {
        return;
    }
    const int old_count = state->trace_num_snapshots;
    int dst = 0;
    for (int src = 0; src < old_count; src += 2, ++dst) {
        if (dst != src) {
            state->trace_iters_host[dst] = state->trace_iters_host[src];
            state->trace_primal_objectives_host[dst] = state->trace_primal_objectives_host[src];
            state->trace_dual_objectives_host[dst] = state->trace_dual_objectives_host[src];
            std::memmove(
                state->trace_primal_solutions_host + (size_t)dst * (size_t)state->num_variables,
                state->trace_primal_solutions_host + (size_t)src * (size_t)state->num_variables,
                (size_t)state->num_variables * sizeof(double)
            );
            std::memmove(
                state->trace_dual_solutions_host + (size_t)dst * (size_t)state->num_constraints,
                state->trace_dual_solutions_host + (size_t)src * (size_t)state->num_constraints,
                (size_t)state->num_constraints * sizeof(double)
            );
        }
    }
    state->trace_num_snapshots = dst;
}

static void maybe_record_trace_snapshot(pdhg_solver_state_t *state) {
    if (!state->trace_enabled || state->trace_max_snapshots <= 0) {
        return;
    }
    if (state->trace_num_snapshots >= state->trace_max_snapshots) {
        compact_trace_snapshots_inplace(state);
    }
    if (state->trace_num_snapshots >= state->trace_max_snapshots) {
        return;
    }

    const size_t x_offset = (size_t)state->trace_num_snapshots * (size_t)state->num_variables;
    const size_t y_offset = (size_t)state->trace_num_snapshots * (size_t)state->num_constraints;
    double* x_dst = state->trace_primal_solutions_host + x_offset;
    double* y_dst = state->trace_dual_solutions_host + y_offset;
    CUDA_CHECK(cudaMemcpy(x_dst, state->pdhg_primal_solution, (size_t)state->num_variables * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(y_dst, state->pdhg_dual_solution, (size_t)state->num_constraints * sizeof(double), cudaMemcpyDeviceToHost));

    const double alpha_x = (state->constraint_bound_rescaling != 0.0) ? state->constraint_bound_rescaling : 1.0;
    const double alpha_y = (state->objective_vector_rescaling != 0.0) ? state->objective_vector_rescaling : 1.0;
    for (int i = 0; i < state->num_variables; ++i) {
        const double sv = (state->trace_variable_rescaling_host != nullptr && state->trace_variable_rescaling_host[i] != 0.0)
            ? state->trace_variable_rescaling_host[i]
            : 1.0;
        x_dst[(size_t)i] /= (sv * alpha_x);
    }
    for (int j = 0; j < state->num_constraints; ++j) {
        const double sc = (state->trace_constraint_rescaling_host != nullptr && state->trace_constraint_rescaling_host[j] != 0.0)
            ? state->trace_constraint_rescaling_host[j]
            : 1.0;
        y_dst[(size_t)j] /= (sc * alpha_y);
    }

    state->trace_iters_host[state->trace_num_snapshots] = state->total_count;
    state->trace_primal_objectives_host[state->trace_num_snapshots] = state->primal_objective_value;
    state->trace_dual_objectives_host[state->trace_num_snapshots] = state->dual_objective_value;
    state->trace_num_snapshots += 1;
}

static void* checked_host_malloc(size_t bytes) {
    if (bytes == 0) {
        return nullptr;
    }
    void* ptr = std::malloc(bytes);
    if (ptr == nullptr) {
        std::fprintf(stderr, "host malloc failed for %zu bytes\n", bytes);
        std::exit(EXIT_FAILURE);
    }
    return ptr;
}

static void* checked_host_calloc(size_t count, size_t elem_size) {
    if (count == 0 || elem_size == 0) {
        return nullptr;
    }
    void* ptr = std::calloc(count, elem_size);
    if (ptr == nullptr) {
        std::fprintf(stderr, "host calloc failed for %zu elements of size %zu\n", count, elem_size);
        std::exit(EXIT_FAILURE);
    }
    return ptr;
}

struct SupportThresholdPredicate {
    double threshold;
    double alpha_x;

    __host__ __device__ bool operator()(const thrust::tuple<double, double>& value_and_scale) const {
        const double value = thrust::get<0>(value_and_scale);
        const double scale = thrust::get<1>(value_and_scale);
        const double safe_scale = (scale != 0.0) ? scale : 1.0;
        return value > threshold * safe_scale * alpha_x;
    }
};

struct SupportLimitEvaluation {
    bool has_eval;
    int support_nnz;
    double rel_obj_change;
    bool support_limit_met;
    bool support_stop_met;
};

static SupportLimitEvaluation evaluate_support_limit(pdhg_solver_state_t *state, const pdhg_parameters_t *params) {
    SupportLimitEvaluation eval{false, -1, INFINITY, false, false};
    if (state == nullptr || params == nullptr) {
        return eval;
    }
    if (params->support_stop_nnz <= 0) {
        return eval;
    }
    const double alpha_x = (state->constraint_bound_rescaling != 0.0) ? state->constraint_bound_rescaling : 1.0;
    auto begin = thrust::make_zip_iterator(thrust::make_tuple(state->pdhg_primal_solution, state->variable_rescaling));
    auto end = thrust::make_zip_iterator(thrust::make_tuple(state->pdhg_primal_solution + state->num_variables, state->variable_rescaling + state->num_variables));
    const int support_nnz = static_cast<int>(
        thrust::count_if(
            thrust::device,
            begin,
            end,
            SupportThresholdPredicate{params->support_stop_threshold, alpha_x}
        )
    );
    eval.has_eval = true;
    eval.support_nnz = support_nnz;
    const double primal_obj = state->primal_objective_value;
    const double dual_obj = state->dual_objective_value;
    // CN: 暂时复用 `support_stop_obj_rel_change_tol` 这个旧参数名，
    // CN: 但这里实际比较的是当前 primal-dual 相对差 `(primal-dual)/primal`，而不是相邻两次 primal objective 的变化。
    // EN: Temporarily reuse the legacy parameter name `support_stop_obj_rel_change_tol`,
    // EN: but compare it against the current primal-dual relative difference `(primal-dual)/primal` rather than the change between two consecutive primal objectives.
    double rel_obj_change = INFINITY;
    if (isfinite(primal_obj) && isfinite(dual_obj) && primal_obj > 1e-12) {
        rel_obj_change = (primal_obj - dual_obj) / primal_obj;
    }
    eval.rel_obj_change = rel_obj_change;
    eval.support_limit_met = support_nnz <= params->support_stop_nnz;
    eval.support_stop_met = eval.support_limit_met &&
        rel_obj_change > 0.0 &&
        rel_obj_change <= params->support_stop_obj_rel_change_tol;
    state->support_stop_obj_rel_change = rel_obj_change;
    state->support_stop_last_eval_obj = primal_obj;
    state->support_stop_last_eval_nnz = support_nnz;
    state->support_stop_has_last_eval_obj = true;
    return eval;
}

static void maybe_apply_resource_limits_after_deferred_optimal(pdhg_solver_state_t *state, const pdhg_parameters_t *params) {
    if (state == nullptr || params == nullptr) {
        return;
    }
    if (state->termination_reason != TERMINATION_REASON_UNSPECIFIED) {
        return;
    }
    if (state->total_count >= params->termination_criteria.iteration_limit) {
        state->termination_reason = TERMINATION_REASON_ITERATION_LIMIT;
        return;
    }
    if (state->cumulative_time_sec >= params->termination_criteria.time_sec_limit) {
        state->termination_reason = TERMINATION_REASON_TIME_LIMIT;
        return;
    }
}

static void maybe_stop_on_support_limit(pdhg_solver_state_t *state, const pdhg_parameters_t *params) {
    if (state == nullptr || params == nullptr) {
        return;
    }
    if (params->support_limit_mode == SUPPORT_LIMIT_MODE_OPTIMAL_ONLY) {
        return;
    }
    const bool optimal_reached = state->termination_reason == TERMINATION_REASON_OPTIMAL;
    if (state->termination_reason != TERMINATION_REASON_UNSPECIFIED && !optimal_reached) {
        return;
    }
    const SupportLimitEvaluation eval = evaluate_support_limit(state, params);
    if (!eval.has_eval) {
        return;
    }

    if (params->support_limit_mode == SUPPORT_LIMIT_MODE_SUPPORT_ONLY) {
        if (eval.support_stop_met) {
            state->termination_reason = TERMINATION_REASON_SUPPORT_LIMIT_REACHED;
            return;
        }
        if (optimal_reached) {
            state->termination_reason = TERMINATION_REASON_UNSPECIFIED;
            maybe_apply_resource_limits_after_deferred_optimal(state, params);
            maybe_stop_on_objective_gap_divergence(state, params);
        }
        return;
    }

    if (params->support_limit_mode == SUPPORT_LIMIT_MODE_BOTH) {
        if (!optimal_reached) {
            return;
        }
        if (eval.support_limit_met) {
            state->termination_reason = TERMINATION_REASON_OPTIMAL_WITH_SUPPORT_LIMIT;
            return;
        }
        if (params->verbose) {
            printf(
                "OPTIMAL reached but support_nnz=%d > support_limit=%d; continuing due to support_limit_mode=both.\n",
                eval.support_nnz,
                params->support_stop_nnz
            );
        }
        state->termination_reason = TERMINATION_REASON_UNSPECIFIED;
        maybe_apply_resource_limits_after_deferred_optimal(state, params);
        maybe_stop_on_objective_gap_divergence(state, params);
        return;
    }

    if (params->support_limit_mode == SUPPORT_LIMIT_MODE_EITHER) {
        if (optimal_reached) {
            return;
        }
        if (eval.support_stop_met) {
            state->termination_reason = TERMINATION_REASON_SUPPORT_LIMIT_REACHED;
        }
    }
}

static void maybe_stop_on_objective_gap_divergence(pdhg_solver_state_t *state, const pdhg_parameters_t *params) {
    if (state == nullptr || params == nullptr) {
        return;
    }
    if (state->termination_reason != TERMINATION_REASON_UNSPECIFIED) {
        return;
    }
    const double current_gap = state->relative_objective_gap;
    if (!std::isfinite(current_gap) || current_gap <= 0.0) {
        state->check_gap_divergence_after_restart = false;
        return;
    }

    const double previous_best = state->best_relative_objective_gap;
    const double eps_primal = params->termination_criteria.eps_feasible_relative_primal > 0.0
        ? params->termination_criteria.eps_feasible_relative_primal
        : 1e-4;
    const double eps_dual = params->termination_criteria.eps_feasible_relative_dual > 0.0
        ? params->termination_criteria.eps_feasible_relative_dual
        : 1e-4;
    // CN: objective gap 只有在 primal/dual residual 已经足够小时才有可比意义；否则 iter 0 这类不可行状态可能产生虚假的极小 gap。
    // EN: The objective gap is comparable only when primal/dual residuals are already small enough; otherwise infeasible states such as iter 0 may produce a spurious tiny gap.
    const double feasibility_gate = std::max(1e-3, 1000.0 * std::max(eps_primal, eps_dual));
    const bool objective_gap_is_comparable =
        std::isfinite(state->relative_primal_residual) &&
        std::isfinite(state->relative_dual_residual) &&
        state->relative_primal_residual <= feasibility_gate &&
        state->relative_dual_residual <= feasibility_gate;
    if (params->enable_objective_gap_divergence_check &&
        state->check_gap_divergence_after_restart &&
        objective_gap_is_comparable) {
        const double eps_opt = params->termination_criteria.eps_optimal_relative > 0.0
            ? params->termination_criteria.eps_optimal_relative
            : 1e-4;
        const double divergence_min_best_gap = 100.0 * eps_opt;
        const double divergence_current_gap_floor = 1e-2;
        const double divergence_ratio = 100.0;
        if (std::isfinite(previous_best) &&
            previous_best > 0.0 &&
            previous_best <= divergence_min_best_gap &&
            current_gap >= divergence_current_gap_floor &&
            current_gap > divergence_ratio * previous_best) {
            state->termination_reason = TERMINATION_REASON_NUMERICAL_DIVERGENCE;
            if (params->verbose) {
                printf(
                    "\n[NUMERICAL_DIVERGENCE] Iter %d: rel_gap %.6e > %.1f * best_rel_gap %.6e "
                    "(threshold best <= %.6e, current >= %.6e).\n",
                    state->total_count,
                    current_gap,
                    divergence_ratio,
                    previous_best,
                    divergence_min_best_gap,
                    divergence_current_gap_floor);
                fflush(stdout);
            }
        }
    }
    state->check_gap_divergence_after_restart = false;
    if (state->termination_reason == TERMINATION_REASON_UNSPECIFIED &&
        objective_gap_is_comparable &&
        (!std::isfinite(state->best_relative_objective_gap) || current_gap < state->best_relative_objective_gap)) {
        state->best_relative_objective_gap = current_gap;
    }
}

pdhg_solver_state_t *optimize(const pdhg_parameters_t *params, const lp_problem_t *original_problem)
{
    cudaFree(0);

    auto sec = [](auto a, auto b) {
        return std::chrono::duration<double>(b - a).count();
    };

    // -------- setup 分解计时 --------
    double t_setup = 0.0, t_print_init = 0.0, t_rescale = 0.0, t_init_state = 0.0, t_step_init = 0.0;

    auto t_a = clk::now();
    print_initial_info(params, original_problem);
    cudaDeviceSynchronize();
    auto t_b = clk::now();
    t_print_init = sec(t_a, t_b);

    t_a = clk::now();
    rescale_info_t *rescale_info = rescale_problem(params, original_problem);
    cudaDeviceSynchronize();
    t_b = clk::now();
    t_rescale = sec(t_a, t_b);

    t_a = clk::now();
    pdhg_solver_state_t *state = initialize_solver_state(original_problem, rescale_info, params);
    cudaDeviceSynchronize();
    t_b = clk::now();
    t_init_state = sec(t_a, t_b);

    rescale_info_free(rescale_info);

    t_a = clk::now();
    initialize_step_size_and_primal_weight(state, params);
    apply_continuation_state(state, params);
    cudaDeviceSynchronize();
    t_b = clk::now();
    t_step_init = sec(t_a, t_b);

    t_setup = t_print_init + t_rescale + t_init_state + t_step_init;

    return run_solver_loop(state, params, t_setup);
}

pdhg_solver_state_t *optimize_device(
    const pdhg_parameters_t *params,
    const device_lp_problem_t *device_problem)
{
    cudaFree(0);

    auto sec = [](auto a, auto b) {
        return std::chrono::duration<double>(b - a).count();
    };

    double t_setup = 0.0, t_print_init = 0.0, t_init_state = 0.0, t_step_init = 0.0;

    auto t_a = clk::now();
    if (params->verbose) {
        std::printf("device-native solve (no internal rescaling)\n");
        std::printf("num_variables = %d, num_constraints = %d, nnz = %d\n",
                    device_problem->num_variables,
                    device_problem->num_constraints,
                    device_problem->constraint_matrix_num_nonzeros);
        std::fflush(stdout);
    }
    cudaDeviceSynchronize();
    auto t_b = clk::now();
    t_print_init = sec(t_a, t_b);

    pdhg_solver_state_t *state = nullptr;
    device_rescale_info_t *rescale_info = nullptr;
    t_a = clk::now();
    if (params->l_inf_ruiz_iterations > 0 ||
        params->has_pock_chambolle_alpha ||
        params->bound_objective_rescaling)
    {
        rescale_info = rescale_problem_device(params, device_problem);
        state = initialize_solver_state_device_rescaled(device_problem, rescale_info, params);
        device_rescale_info_free(rescale_info);
    }
    else
    {
        state = initialize_solver_state_device_no_rescale(device_problem, params);
    }
    cudaDeviceSynchronize();
    t_b = clk::now();
    t_init_state = sec(t_a, t_b);

    t_a = clk::now();
    initialize_step_size_and_primal_weight(state, params);
    apply_continuation_state(state, params);
    cudaDeviceSynchronize();
    t_b = clk::now();
    t_step_init = sec(t_a, t_b);

    t_setup = t_print_init + t_init_state + t_step_init;
    return run_solver_loop(state, params, t_setup);
}

static pdhg_solver_state_t *run_solver_loop(
    pdhg_solver_state_t *state,
    const pdhg_parameters_t *params,
    double t_setup)
{
    auto sec = [](auto a, auto b) {
        return std::chrono::duration<double>(b - a).count();
    };

    double t_eval = 0.0;
    double t_restart = 0.0;
    double t_kernel = 0.0;

    auto t0 = clk::now();
    clock_t start_time_clock = clock();
    auto t_loop_start = clk::now();

    bool do_restart = false;
    {
        cudaDeviceSynchronize();
        size_t free_byte, total_byte;
        if (cudaMemGetInfo(&free_byte, &total_byte) == cudaSuccess) {
            g_solver_peak_mem_bytes = total_byte - free_byte;
            if (params->verbose) {
                printf("[GPU MEM STATS] Peak Memory at Loop Start: %.2f MiB (Used: %.2f / Total: %.2f)\n",
                       g_solver_peak_mem_bytes / (1024.0 * 1024.0),
                       (double)(total_byte - free_byte) / (1024.0 * 1024.0),
                       (double)total_byte / (1024.0 * 1024.0));
            }
        }
    }
    if (params->verbose) {
        print_gpu_mem_once("before main loop (after allocs)", /*sync_before_query=*/true);
    }
    while (state->termination_reason == TERMINATION_REASON_UNSPECIFIED)
    {
        if ((state->is_this_major_iteration || state->total_count == 0) ||
            (state->total_count % get_print_frequency(state->total_count) == 0))
        {
            auto t1 = clk::now();
            compute_residual(state);
            cudaDeviceSynchronize();

            if (state->is_this_major_iteration &&
                state->total_count < 3 * params->termination_evaluation_frequency)
            {
                compute_infeasibility_information(state);
                cudaDeviceSynchronize();
            }

            state->cumulative_time_sec = (double)(clock() - start_time_clock) / CLOCKS_PER_SEC;

            check_termination_criteria(state, &params->termination_criteria);
            maybe_stop_on_objective_gap_divergence(state, params);
            maybe_stop_on_support_limit(state, params);
            maybe_record_trace_snapshot(state);
            display_iteration_stats(state, params->verbose);
            cudaDeviceSynchronize();

            auto t2 = clk::now();
            t_eval += sec(t1, t2);

            if (state->termination_reason != TERMINATION_REASON_UNSPECIFIED)
            {
                break;
            }
        }

        if ((state->is_this_major_iteration || state->total_count == 0))
        {
            auto t1 = clk::now();
            do_restart = should_do_adaptive_restart(
                state, &params->restart_params,
                params->termination_evaluation_frequency,
                params->verbose);
            if (do_restart)
            {
                perform_restart(state, params);
                state->check_gap_divergence_after_restart = true;
                cudaDeviceSynchronize();
            }
            auto t2 = clk::now();
            t_restart += sec(t1, t2);
        }

        state->is_this_major_iteration =
            ((state->total_count + 1) % params->termination_evaluation_frequency) == 0;

        auto t1 = clk::now();
        compute_next_pdhg_primal_solution(state);
        compute_next_pdhg_dual_solution(state);
        cudaDeviceSynchronize();

        if (state->is_this_major_iteration || do_restart)
        {
            compute_fixed_point_error(state);
            cudaDeviceSynchronize();

            if (do_restart)
            {
                state->initial_fixed_point_error = state->fixed_point_error;
                do_restart = false;
            }
        }

        halpern_update(state, params->reflection_coefficient);
        cudaDeviceSynchronize();

        auto t2 = clk::now();
        t_kernel += sec(t1, t2);

        state->inner_count++;
        state->total_count++;
    }

    auto t_loop_end = clk::now();
    auto t1 = clk::now();
    pdhg_final_log(state, params->verbose, state->termination_reason);
    cudaDeviceSynchronize();
    auto t2 = clk::now();
    double t_finalize = sec(t1, t2);

    auto t_end = clk::now();
    double t_total = sec(t0, t_end);
    double t_loop  = sec(t_loop_start, t_loop_end);

    (void)t_setup;
    (void)t_eval;
    (void)t_restart;
    (void)t_kernel;
    (void)t_finalize;
    (void)t_total;
    (void)t_loop;
    return state;
}



static pdhg_solver_state_t *initialize_solver_state(
    const lp_problem_t *original_problem,
    const rescale_info_t *rescale_info,
    const pdhg_parameters_t *params)
{
    pdhg_solver_state_t *state = (pdhg_solver_state_t *)safe_calloc(1, sizeof(pdhg_solver_state_t));

    int n_vars = original_problem->num_variables;
    int n_cons = original_problem->num_constraints;
    size_t var_bytes = n_vars * sizeof(double);
    size_t con_bytes = n_cons * sizeof(double);

    state->num_variables = n_vars;
    state->num_constraints = n_cons;
    state->objective_constant = original_problem->objective_constant;
    state->termination_norm = (params != nullptr) ? params->termination_norm : TERMINATION_NORM_L2;
    state->trace_enabled = (params != nullptr) ? bool(params->trace_enabled) : false;
    state->trace_max_snapshots = (params != nullptr && params->trace_max_snapshots > 0) ? int(params->trace_max_snapshots) : 0;
    state->trace_num_snapshots = 0;

    state->constraint_matrix = (cu_sparse_matrix_csr_t *)safe_malloc(sizeof(cu_sparse_matrix_csr_t));
    state->constraint_matrix_t = (cu_sparse_matrix_csr_t *)safe_malloc(sizeof(cu_sparse_matrix_csr_t));
    state->matrix_value_mode = MATRIX_VALUES_EXPLICIT;
    state->vector_sum_mode = (params != nullptr) ? params->vector_sum_mode : VECTOR_SUM_RESIDENT_ONES;

    state->constraint_matrix->num_rows = n_cons;
    state->constraint_matrix->num_cols = n_vars;
    state->constraint_matrix->num_nonzeros = original_problem->constraint_matrix_num_nonzeros;

    state->constraint_matrix_t->num_rows = n_vars;
    state->constraint_matrix_t->num_cols = n_cons;
    state->constraint_matrix_t->num_nonzeros = original_problem->constraint_matrix_num_nonzeros;

    state->termination_reason = TERMINATION_REASON_UNSPECIFIED;
    state->support_stop_has_last_eval_obj = false;
    state->support_stop_last_eval_obj = 0.0;
    state->support_stop_obj_rel_change = INFINITY;
    state->support_stop_last_eval_nnz = -1;

#define ALLOC_AND_COPY(dest, src, bytes)  \
    CUDA_CHECK(cudaMalloc(&dest, bytes)); \
    CUDA_CHECK(cudaMemcpy(dest, src, bytes, cudaMemcpyHostToDevice));

    ALLOC_AND_COPY(state->constraint_matrix->row_ptr, rescale_info->scaled_problem->constraint_matrix_row_pointers, (n_cons + 1) * sizeof(int));
    ALLOC_AND_COPY(state->constraint_matrix->col_ind, rescale_info->scaled_problem->constraint_matrix_col_indices, rescale_info->scaled_problem->constraint_matrix_num_nonzeros * sizeof(int));
    ALLOC_AND_COPY(state->constraint_matrix->val, rescale_info->scaled_problem->constraint_matrix_values, rescale_info->scaled_problem->constraint_matrix_num_nonzeros * sizeof(double));

    CUDA_CHECK(cudaMalloc(&state->constraint_matrix_t->row_ptr, (n_vars + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state->constraint_matrix_t->col_ind, rescale_info->scaled_problem->constraint_matrix_num_nonzeros * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state->constraint_matrix_t->val, rescale_info->scaled_problem->constraint_matrix_num_nonzeros * sizeof(double)));

    CUSPARSE_CHECK(cusparseCreate(&state->sparse_handle));
    CUBLAS_CHECK(cublasCreate(&state->blas_handle));
    CUBLAS_CHECK(cublasSetPointerMode(state->blas_handle, CUBLAS_POINTER_MODE_HOST));

    size_t buffer_size = 0;
    void *buffer = nullptr;
    // CUSPARSE_CHECK(cusparseCsr2cscEx2_bufferSize(
    //     state->sparse_handle, state->constraint_matrix->num_rows, state->constraint_matrix->num_cols, state->constraint_matrix->num_nonzeros,
    //     state->constraint_matrix->val, state->constraint_matrix->row_ptr, state->constraint_matrix->col_ind,
    //     state->constraint_matrix_t->val, state->constraint_matrix_t->row_ptr, state->constraint_matrix_t->col_ind,
    //     CUDA_R_64F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
    //     CUSPARSE_CSR2CSC_ALG_DEFAULT, &buffer_size));
    CUSPARSE_CHECK(cusparseCsr2cscEx2_bufferSize(
        state->sparse_handle, state->constraint_matrix->num_rows, state->constraint_matrix->num_cols, state->constraint_matrix->num_nonzeros,
        state->constraint_matrix->val, state->constraint_matrix->row_ptr, state->constraint_matrix->col_ind,
        state->constraint_matrix_t->val, state->constraint_matrix_t->row_ptr, state->constraint_matrix_t->col_ind,
        CUDA_R_64F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
        CUSPARSE_CSR2CSC_ALG1, &buffer_size));
    CUDA_CHECK(cudaMalloc(&buffer, buffer_size));

    // CUSPARSE_CHECK(cusparseCsr2cscEx2(
    //     state->sparse_handle, state->constraint_matrix->num_rows, state->constraint_matrix->num_cols, state->constraint_matrix->num_nonzeros,
    //     state->constraint_matrix->val, state->constraint_matrix->row_ptr, state->constraint_matrix->col_ind,
    //     state->constraint_matrix_t->val, state->constraint_matrix_t->row_ptr, state->constraint_matrix_t->col_ind,
    //     CUDA_R_64F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
    //     CUSPARSE_CSR2CSC_ALG_DEFAULT, buffer));
    CUSPARSE_CHECK(cusparseCsr2cscEx2(
        state->sparse_handle, state->constraint_matrix->num_rows, state->constraint_matrix->num_cols, state->constraint_matrix->num_nonzeros,
        state->constraint_matrix->val, state->constraint_matrix->row_ptr, state->constraint_matrix->col_ind,
        state->constraint_matrix_t->val, state->constraint_matrix_t->row_ptr, state->constraint_matrix_t->col_ind,
        CUDA_R_64F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
        CUSPARSE_CSR2CSC_ALG1, buffer));

    CUDA_CHECK(cudaFree(buffer));

    ALLOC_AND_COPY(state->variable_lower_bound, rescale_info->scaled_problem->variable_lower_bound, var_bytes);
    ALLOC_AND_COPY(state->variable_upper_bound, rescale_info->scaled_problem->variable_upper_bound, var_bytes);
    ALLOC_AND_COPY(state->objective_vector, rescale_info->scaled_problem->objective_vector, var_bytes);
    ALLOC_AND_COPY(state->constraint_lower_bound, rescale_info->scaled_problem->constraint_lower_bound, con_bytes);
    ALLOC_AND_COPY(state->constraint_upper_bound, rescale_info->scaled_problem->constraint_upper_bound, con_bytes);
    ALLOC_AND_COPY(state->constraint_rescaling, rescale_info->con_rescale, con_bytes);
    ALLOC_AND_COPY(state->variable_rescaling, rescale_info->var_rescale, var_bytes);

    state->constraint_bound_rescaling = rescale_info->con_bound_rescale;
    state->objective_vector_rescaling = rescale_info->obj_vec_rescale;

    if (state->trace_enabled && state->trace_max_snapshots > 0) {
        state->trace_iters_host = (int*)checked_host_calloc((size_t)state->trace_max_snapshots, sizeof(int));
        state->trace_primal_objectives_host = (double*)checked_host_calloc((size_t)state->trace_max_snapshots, sizeof(double));
        state->trace_dual_objectives_host = (double*)checked_host_calloc((size_t)state->trace_max_snapshots, sizeof(double));
        state->trace_primal_solutions_host = (double*)checked_host_calloc((size_t)state->trace_max_snapshots, var_bytes);
        state->trace_dual_solutions_host = (double*)checked_host_calloc((size_t)state->trace_max_snapshots, con_bytes);
        state->trace_variable_rescaling_host = (double*)checked_host_malloc(var_bytes);
        state->trace_constraint_rescaling_host = (double*)checked_host_malloc(con_bytes);
        std::memcpy(state->trace_variable_rescaling_host, rescale_info->var_rescale, var_bytes);
        std::memcpy(state->trace_constraint_rescaling_host, rescale_info->con_rescale, con_bytes);
    }

#define ALLOC_ZERO(dest, bytes)           \
    CUDA_CHECK(cudaMalloc(&dest, bytes)); \
    CUDA_CHECK(cudaMemset(dest, 0, bytes));

    ALLOC_ZERO(state->initial_primal_solution, var_bytes);
    ALLOC_ZERO(state->current_primal_solution, var_bytes);
    ALLOC_ZERO(state->pdhg_primal_solution, var_bytes);
    ALLOC_ZERO(state->reflected_primal_solution, var_bytes);
    ALLOC_ZERO(state->dual_product, var_bytes);
    ALLOC_ZERO(state->dual_slack, var_bytes);
    ALLOC_ZERO(state->dual_residual, var_bytes);
    ALLOC_ZERO(state->delta_primal_solution, var_bytes);

    ALLOC_ZERO(state->initial_dual_solution, con_bytes);
    ALLOC_ZERO(state->current_dual_solution, con_bytes);
    ALLOC_ZERO(state->pdhg_dual_solution, con_bytes);
    ALLOC_ZERO(state->reflected_dual_solution, con_bytes);
    ALLOC_ZERO(state->primal_product, con_bytes);
    ALLOC_ZERO(state->primal_slack, con_bytes);
    ALLOC_ZERO(state->primal_residual, con_bytes);
    ALLOC_ZERO(state->delta_dual_solution, con_bytes);

    // ---- Warm start: map UN-SCALED x0/y0 -> INTERNAL SCALED and write to all start buffers ----
    if (params && params->has_initial_iterate &&
        (params->initial_primal_unscaled || params->initial_dual_unscaled)) {

        const double *s_var = rescale_info->var_rescale; // len = n_vars (host)
        const double *s_con = rescale_info->con_rescale; // len = n_cons (host)

        // 这两个“标量缩放”也要乘上，否则会导致 Ax-b 在内部空间上来就巨大
        const double alpha_x = (state->constraint_bound_rescaling  != 0.0)
                                ? state->constraint_bound_rescaling  : 1.0;
        const double alpha_y = (state->objective_vector_rescaling != 0.0)
                                ? state->objective_vector_rescaling : 1.0;

        // 用“未缩放空间”的边界先夹紧 x0（避免跨尺度延拓后出界造成大残差）
        const double *lb_u = original_problem->variable_lower_bound; // 未缩放 lb
        const double *ub_u = original_problem->variable_upper_bound; // 未缩放 ub

        // 1) 处理 x0：x_internal = clip_unscaled(x0, lb, ub) * (var_rescale[i] * alpha_x)
        if (params->initial_primal_unscaled) {
            double *x_scaled_h = (double *)safe_malloc(var_bytes);
            for (int i = 0; i < n_vars; ++i) {
                double xu = params->initial_primal_unscaled[i];   // 未缩放
                if (lb_u) xu = (xu < lb_u[i]) ? lb_u[i] : xu;
                if (ub_u) xu = (xu > ub_u[i]) ? ub_u[i] : xu;
                const double sv = s_var ? s_var[i] : 1.0;
                x_scaled_h[i] = xu * (sv * alpha_x);              // ✅ 正确映射
            }

            CUDA_CHECK(cudaMemcpy(state->initial_primal_solution,   x_scaled_h, var_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(state->current_primal_solution,   x_scaled_h, var_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(state->pdhg_primal_solution,      x_scaled_h, var_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(state->reflected_primal_solution, x_scaled_h, var_bytes, cudaMemcpyHostToDevice));
            free(x_scaled_h);
        }

        // 2) 处理 y0：y_internal = y0_unscaled * (con_rescale[j] * alpha_y)
        if (params->initial_dual_unscaled) {
            double *y_scaled_h = (double *)safe_malloc(con_bytes);
            for (int j = 0; j < n_cons; ++j) {
                const double yu = params->initial_dual_unscaled[j]; // 未缩放
                const double sc = s_con ? s_con[j] : 1.0;
                y_scaled_h[j] = yu * (sc * alpha_y);                // ✅ 正确映射
            }

            CUDA_CHECK(cudaMemcpy(state->initial_dual_solution,   y_scaled_h, con_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(state->current_dual_solution,   y_scaled_h, con_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(state->pdhg_dual_solution,      y_scaled_h, con_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(state->reflected_dual_solution, y_scaled_h, con_bytes, cudaMemcpyHostToDevice));
            free(y_scaled_h);
        }
    }



    double *temp_host = (double *)safe_malloc(fmax(var_bytes, con_bytes));
    for (int i = 0; i < n_cons; ++i)
        temp_host[i] = isfinite(rescale_info->scaled_problem->constraint_lower_bound[i]) ? rescale_info->scaled_problem->constraint_lower_bound[i] : 0.0;
    ALLOC_AND_COPY(state->constraint_lower_bound_finite_val, temp_host, con_bytes);
    for (int i = 0; i < n_cons; ++i)
        temp_host[i] = isfinite(rescale_info->scaled_problem->constraint_upper_bound[i]) ? rescale_info->scaled_problem->constraint_upper_bound[i] : 0.0;
    ALLOC_AND_COPY(state->constraint_upper_bound_finite_val, temp_host, con_bytes);
    for (int i = 0; i < n_vars; ++i)
        temp_host[i] = isfinite(rescale_info->scaled_problem->variable_lower_bound[i]) ? rescale_info->scaled_problem->variable_lower_bound[i] : 0.0;
    ALLOC_AND_COPY(state->variable_lower_bound_finite_val, temp_host, var_bytes);
    for (int i = 0; i < n_vars; ++i)
        temp_host[i] = isfinite(rescale_info->scaled_problem->variable_upper_bound[i]) ? rescale_info->scaled_problem->variable_upper_bound[i] : 0.0;
    ALLOC_AND_COPY(state->variable_upper_bound_finite_val, temp_host, var_bytes);
    free(temp_host);

    double sum_of_squares = 0.0;
    double objective_linf_norm = 0.0;

    for (int i = 0; i < n_vars; ++i)
    {
        sum_of_squares += original_problem->objective_vector[i] * original_problem->objective_vector[i];
        objective_linf_norm = fmax(objective_linf_norm, fabs(original_problem->objective_vector[i]));
    }
    state->objective_vector_norm = sqrt(sum_of_squares);

    sum_of_squares = 0.0;
    double constraint_bound_linf_norm = 0.0;

    for (int i = 0; i < n_cons; ++i)
    {
        double lower = original_problem->constraint_lower_bound[i];
        double upper = original_problem->constraint_upper_bound[i];

        if (isfinite(lower) && (lower != upper))
        {
            sum_of_squares += lower * lower;
            constraint_bound_linf_norm = fmax(constraint_bound_linf_norm, fabs(lower));
        }

        if (isfinite(upper))
        {
            sum_of_squares += upper * upper;
            constraint_bound_linf_norm = fmax(constraint_bound_linf_norm, fabs(upper));
        }
    }

    state->constraint_bound_norm = sqrt(sum_of_squares);
    state->termination_objective_vector_norm =
        (state->termination_norm == TERMINATION_NORM_L_INF) ? objective_linf_norm : state->objective_vector_norm;
    state->termination_constraint_bound_norm =
        (state->termination_norm == TERMINATION_NORM_L_INF) ? constraint_bound_linf_norm : state->constraint_bound_norm;
    state->num_blocks_primal = (state->num_variables + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    state->num_blocks_dual = (state->num_constraints + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    state->num_blocks_primal_dual = (state->num_variables + state->num_constraints + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    state->best_primal_dual_residual_gap = INFINITY;
    state->best_relative_objective_gap = INFINITY;
    state->last_trial_fixed_point_error = INFINITY;
    state->step_size = 0.0;
    state->is_this_major_iteration = false;

    size_t primal_spmv_buffer_size;
    size_t dual_spmv_buffer_size;

    CUSPARSE_CHECK(cusparseCreateCsr(&state->matA, state->num_constraints, state->num_variables, state->constraint_matrix->num_nonzeros, state->constraint_matrix->row_ptr, state->constraint_matrix->col_ind, state->constraint_matrix->val, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

    CUDA_CHECK(cudaGetLastError());

    CUSPARSE_CHECK(cusparseCreateCsr(&state->matAt, state->num_variables, state->num_constraints, state->constraint_matrix_t->num_nonzeros, state->constraint_matrix_t->row_ptr, state->constraint_matrix_t->col_ind, state->constraint_matrix_t->val, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    CUDA_CHECK(cudaGetLastError());

    CUSPARSE_CHECK(cusparseCreateDnVec(&state->vec_primal_sol, state->num_variables, state->pdhg_primal_solution, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&state->vec_dual_sol, state->num_constraints, state->pdhg_dual_solution, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&state->vec_primal_prod, state->num_constraints, state->primal_product, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&state->vec_dual_prod, state->num_variables, state->dual_product, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &HOST_ONE, state->matA, state->vec_primal_sol, &HOST_ZERO, state->vec_primal_prod, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, &primal_spmv_buffer_size));

    CUSPARSE_CHECK(cusparseSpMV_bufferSize(state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &HOST_ONE, state->matAt, state->vec_dual_sol, &HOST_ZERO, state->vec_dual_prod, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, &dual_spmv_buffer_size));
    CUDA_CHECK(cudaMalloc(&state->primal_spmv_buffer, primal_spmv_buffer_size));
    // CUSPARSE_CHECK(cusparseSpMV_preprocess(state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
    //                                        &HOST_ONE, state->matA, state->vec_primal_sol, &HOST_ZERO, state->vec_primal_prod,
    //                                        CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, state->primal_spmv_buffer));

    CUDA_CHECK(cudaMalloc(&state->dual_spmv_buffer, dual_spmv_buffer_size));
    // CUSPARSE_CHECK(cusparseSpMV_preprocess(state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
    //                                        &HOST_ONE, state->matAt, state->vec_dual_sol, &HOST_ZERO, state->vec_dual_prod,
    //                                        CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, state->dual_spmv_buffer));

    if (state->vector_sum_mode == VECTOR_SUM_RESIDENT_ONES) {
        CUDA_CHECK(cudaMalloc(&state->ones_primal_d, state->num_variables * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&state->ones_dual_d, state->num_constraints * sizeof(double)));

        double *ones_primal_h = (double *)safe_malloc(state->num_variables * sizeof(double));
        for (int i = 0; i < state->num_variables; ++i)
            ones_primal_h[i] = 1.0;
        CUDA_CHECK(cudaMemcpy(state->ones_primal_d, ones_primal_h, state->num_variables * sizeof(double), cudaMemcpyHostToDevice));
        free(ones_primal_h);

        double *ones_dual_h = (double *)safe_malloc(state->num_constraints * sizeof(double));
        for (int i = 0; i < state->num_constraints; ++i)
            ones_dual_h[i] = 1.0;
        CUDA_CHECK(cudaMemcpy(state->ones_dual_d, ones_dual_h, state->num_constraints * sizeof(double), cudaMemcpyHostToDevice));
        free(ones_dual_h);
    }

    state->k_p = params->restart_params.k_p;
    state->k_i = params->restart_params.k_i;
    state->k_d = params->restart_params.k_d;
    state->previous_restart_dual_residual = DBL_MAX;
    state->previous_restart_gap = DBL_MAX;
    return state;
}

__global__ static void fill_constraint_bounds_from_rhs_kernel(
    const double *rhs,
    double *lower,
    double *upper,
    int n_cons,
    int n_eqs)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_cons) {
        return;
    }
    const double rhs_val = rhs[i];
    if (i < n_eqs) {
        lower[i] = rhs_val;
        upper[i] = rhs_val;
    } else {
        lower[i] = -INFINITY;
        upper[i] = rhs_val;
    }
}

__global__ static void fill_double_kernel(double *dst, int n, double value)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = value;
    }
}

__global__ static void finite_or_zero_kernel(const double *src, double *dst, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const double v = src[i];
        dst[i] = isfinite(v) ? v : 0.0;
    }
}

static double device_inf_norm(cublasHandle_t handle, const double *values, int n)
{
    if (n <= 0) {
        return 0.0;
    }
    int one_based_index = 0;
    CUBLAS_CHECK(cublasIdamax(handle, n, values, 1, &one_based_index));
    double value = 0.0;
    CUDA_CHECK(cudaMemcpy(&value, values + (one_based_index - 1), sizeof(double), cudaMemcpyDeviceToHost));
    return std::fabs(value);
}

__global__ static void constraint_bound_square_contrib_kernel(
    const double *lower,
    const double *upper,
    double *out,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const double lo = lower[i];
        const double up = upper[i];
        double acc = 0.0;
        if (isfinite(lo) && lo != up) {
            acc += lo * lo;
        }
        if (isfinite(up)) {
            acc += up * up;
        }
        out[i] = acc;
    }
}

__global__ static void scale_primal_unscaled_to_scaled_kernel(
    const double *x_unscaled,
    const double *var_rescale,
    double alpha_x,
    const double *scaled_lb,
    const double *scaled_ub,
    int variable_bound_mode,
    double scaled_lb_constant,
    double scaled_ub_constant,
    bool clip_to_scaled_bounds,
    double *x_scaled,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const double sv = (var_rescale != nullptr && var_rescale[i] != 0.0) ? var_rescale[i] : 1.0;
        double xs = x_unscaled[i] * (sv * alpha_x);
        if (clip_to_scaled_bounds) {
            const double lo = (variable_bound_mode == VARIABLE_BOUNDS_CONSTANT) ? scaled_lb_constant : scaled_lb[i];
            const double hi = (variable_bound_mode == VARIABLE_BOUNDS_CONSTANT) ? scaled_ub_constant : scaled_ub[i];
            if (isfinite(lo) && xs < lo) xs = lo;
            if (isfinite(hi) && xs > hi) xs = hi;
        }
        x_scaled[i] = xs;
    }
}

__global__ static void scale_dual_unscaled_to_scaled_kernel(
    const double *y_unscaled,
    const double *con_rescale,
    double alpha_y,
    double *y_scaled,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const double sc = (con_rescale != nullptr && con_rescale[i] != 0.0) ? con_rescale[i] : 1.0;
        y_scaled[i] = y_unscaled[i] * (sc * alpha_y);
    }
}

static void copy_scaled_primal_from_unscaled_to_buffer(
    pdhg_solver_state_t *state,
    const double *x_unscaled,
    double *dst_buffer,
    bool clip_to_scaled_bounds)
{
    if (state == nullptr || x_unscaled == nullptr || dst_buffer == nullptr) {
        return;
    }
    const int n_vars = state->num_variables;
    const int blocks_primal = (n_vars + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    const double alpha_x = (state->constraint_bound_rescaling != 0.0) ? state->constraint_bound_rescaling : 1.0;
    double *x_unscaled_d = nullptr;
    CUDA_CHECK(cudaMalloc(&x_unscaled_d, (size_t)n_vars * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(x_unscaled_d, x_unscaled, (size_t)n_vars * sizeof(double), cudaMemcpyHostToDevice));
    scale_primal_unscaled_to_scaled_kernel<<<blocks_primal, THREADS_PER_BLOCK>>>(
        x_unscaled_d,
        state->variable_rescaling,
        alpha_x,
        state->variable_lower_bound,
        state->variable_upper_bound,
        (int)state->variable_bound_mode,
        state->variable_lower_bound_constant,
        state->variable_upper_bound_constant,
        clip_to_scaled_bounds,
        dst_buffer,
        n_vars);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFree(x_unscaled_d));
}

static void copy_scaled_dual_from_unscaled_to_buffer(
    pdhg_solver_state_t *state,
    const double *y_unscaled,
    double *dst_buffer)
{
    if (state == nullptr || y_unscaled == nullptr || dst_buffer == nullptr) {
        return;
    }
    const int n_cons = state->num_constraints;
    const int blocks_dual = (n_cons + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    const double alpha_y = (state->objective_vector_rescaling != 0.0) ? state->objective_vector_rescaling : 1.0;
    double *y_unscaled_d = nullptr;
    CUDA_CHECK(cudaMalloc(&y_unscaled_d, (size_t)n_cons * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(y_unscaled_d, y_unscaled, (size_t)n_cons * sizeof(double), cudaMemcpyHostToDevice));
    scale_dual_unscaled_to_scaled_kernel<<<blocks_dual, THREADS_PER_BLOCK>>>(
        y_unscaled_d,
        state->constraint_rescaling,
        alpha_y,
        dst_buffer,
        n_cons);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFree(y_unscaled_d));
}

static void copy_scaled_initial_buffers(
    pdhg_solver_state_t *state,
    const pdhg_parameters_t *params)
{
    if (!(params && params->has_initial_iterate &&
          (params->initial_primal_unscaled || params->initial_dual_unscaled))) {
        return;
    }

    if (params->initial_primal_unscaled) {
        copy_scaled_primal_from_unscaled_to_buffer(
            state,
            params->initial_primal_unscaled,
            state->initial_primal_solution,
            true);
        CUDA_CHECK(cudaMemcpy(state->current_primal_solution, state->initial_primal_solution, (size_t)state->num_variables * sizeof(double), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(state->pdhg_primal_solution, state->initial_primal_solution, (size_t)state->num_variables * sizeof(double), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(state->reflected_primal_solution, state->initial_primal_solution, (size_t)state->num_variables * sizeof(double), cudaMemcpyDeviceToDevice));
    }

    if (params->initial_dual_unscaled) {
        copy_scaled_dual_from_unscaled_to_buffer(
            state,
            params->initial_dual_unscaled,
            state->initial_dual_solution);
        CUDA_CHECK(cudaMemcpy(state->current_dual_solution, state->initial_dual_solution, (size_t)state->num_constraints * sizeof(double), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(state->pdhg_dual_solution, state->initial_dual_solution, (size_t)state->num_constraints * sizeof(double), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(state->reflected_dual_solution, state->initial_dual_solution, (size_t)state->num_constraints * sizeof(double), cudaMemcpyDeviceToDevice));
    }
}

static void apply_continuation_state(
    pdhg_solver_state_t *state,
    const pdhg_parameters_t *params)
{
    if (state == nullptr || params == nullptr || !params->has_continuation || !params->continuation.enabled) {
        return;
    }
    const continuation_parameters_t *cont = &params->continuation;
    const bool has_anchor_primal = bool(cont->has_anchor_primal && cont->anchor_primal_unscaled != nullptr);
    const bool has_anchor_dual = bool(cont->has_anchor_dual && cont->anchor_dual_unscaled != nullptr);
    const bool has_current_primal = bool(cont->has_current_primal && cont->current_primal_unscaled != nullptr);
    const bool has_current_dual = bool(cont->has_current_dual && cont->current_dual_unscaled != nullptr);

    if (has_anchor_primal) {
        copy_scaled_primal_from_unscaled_to_buffer(
            state,
            cont->anchor_primal_unscaled,
            state->initial_primal_solution,
            true);
    } else if (has_current_primal) {
        copy_scaled_primal_from_unscaled_to_buffer(
            state,
            cont->current_primal_unscaled,
            state->initial_primal_solution,
            true);
    }
    if (has_anchor_dual) {
        copy_scaled_dual_from_unscaled_to_buffer(
            state,
            cont->anchor_dual_unscaled,
            state->initial_dual_solution);
    } else if (has_current_dual) {
        copy_scaled_dual_from_unscaled_to_buffer(
            state,
            cont->current_dual_unscaled,
            state->initial_dual_solution);
    }
    if (has_current_primal) {
        copy_scaled_primal_from_unscaled_to_buffer(
            state,
            cont->current_primal_unscaled,
            state->pdhg_primal_solution,
            true);
        CUDA_CHECK(cudaMemcpy(state->current_primal_solution, state->pdhg_primal_solution, (size_t)state->num_variables * sizeof(double), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(state->reflected_primal_solution, state->pdhg_primal_solution, (size_t)state->num_variables * sizeof(double), cudaMemcpyDeviceToDevice));
    }
    if (has_current_dual) {
        copy_scaled_dual_from_unscaled_to_buffer(
            state,
            cont->current_dual_unscaled,
            state->pdhg_dual_solution);
        CUDA_CHECK(cudaMemcpy(state->current_dual_solution, state->pdhg_dual_solution, (size_t)state->num_constraints * sizeof(double), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(state->reflected_dual_solution, state->pdhg_dual_solution, (size_t)state->num_constraints * sizeof(double), cudaMemcpyDeviceToDevice));
    }

    if (cont->reuse_step_size && std::isfinite(cont->step_size) && cont->step_size > 0.0) {
        state->step_size = cont->step_size;
    }
    if (std::isfinite(cont->primal_weight) && cont->primal_weight > 0.0) {
        state->primal_weight = cont->primal_weight;
    }
    if (std::isfinite(cont->primal_weight_error_sum)) {
        state->primal_weight_error_sum = cont->primal_weight_error_sum;
    }
    if (std::isfinite(cont->primal_weight_last_error)) {
        state->primal_weight_last_error = cont->primal_weight_last_error;
    }
    if (std::isfinite(cont->best_primal_weight) && cont->best_primal_weight > 0.0) {
        state->best_primal_weight = cont->best_primal_weight;
    }
    if (std::isfinite(cont->best_primal_dual_residual_gap) && cont->best_primal_dual_residual_gap > 0.0) {
        state->best_primal_dual_residual_gap = cont->best_primal_dual_residual_gap;
    }
    if (std::isfinite(cont->previous_restart_dual_residual)) {
        state->previous_restart_dual_residual = cont->previous_restart_dual_residual;
    }
    if (std::isfinite(cont->previous_restart_gap)) {
        state->previous_restart_gap = cont->previous_restart_gap;
    }
    state->total_count = cont->total_count >= 0 ? cont->total_count : 0;
    state->inner_count = cont->inner_count >= 0 ? cont->inner_count : 0;
    state->is_this_major_iteration = false;

    if (cont->apply_restart_on_entry) {
        compute_residual(state);
        CUDA_CHECK(cudaDeviceSynchronize());
        perform_restart(state, params);
        state->check_gap_divergence_after_restart = true;
        CUDA_CHECK(cudaDeviceSynchronize());
        compute_fixed_point_error(state);
        CUDA_CHECK(cudaDeviceSynchronize());
        state->initial_fixed_point_error = state->fixed_point_error;
        state->last_trial_fixed_point_error = INFINITY;
        state->inner_count = 0;
        state->is_this_major_iteration = false;
    }
}

__global__ static void row_max_abs_csr_kernel(
    const int *row_ptr,
    const double *val,
    double *out,
    int n_rows)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) {
        return;
    }
    double mx = 0.0;
    for (int p = row_ptr[row]; p < row_ptr[row + 1]; ++p) {
        const double a = fabs(val[p]);
        if (a > mx) {
            mx = a;
        }
    }
    out[row] = mx;
}

__global__ static void row_power_sum_csr_kernel(
    const int *row_ptr,
    const double *val,
    double *out,
    int n_rows,
    double power)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) {
        return;
    }
    double acc = 0.0;
    for (int p = row_ptr[row]; p < row_ptr[row + 1]; ++p) {
        acc += pow(fabs(val[p]), power);
    }
    out[row] = acc;
}

__global__ static void sqrt_clip_inplace_kernel(double *x, int n, double eps)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const double v = x[i];
        x[i] = (v < eps) ? 1.0 : sqrt(v);
    }
}

__global__ static void multiply_inplace_kernel(double *dst, const double *src, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] *= src[i];
    }
}

__global__ static void scale_problem_variables_kernel(
    double *obj,
    double *lb,
    double *ub,
    const double *var_rescale,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const double s = var_rescale[i];
        obj[i] /= s;
        lb[i] *= s;
        ub[i] *= s;
    }
}

__global__ static void scale_objective_by_variable_kernel(
    double *obj,
    const double *var_rescale,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        obj[i] /= var_rescale[i];
    }
}

__global__ static void scale_objective_by_scalar_kernel(
    double *objective,
    int n,
    double objective_scale)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        objective[i] *= objective_scale;
    }
}

__global__ static void scale_problem_constraints_kernel(
    double *lb,
    double *ub,
    const double *con_rescale,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const double s = con_rescale[i];
        lb[i] /= s;
        ub[i] /= s;
    }
}

__global__ static void scale_matrix_csr_kernel(
    const int *row_ptr,
    const int *col_ind,
    double *val,
    const double *row_scale,
    const double *col_scale,
    int n_rows)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) {
        return;
    }
    const double rs = row_scale[row];
    for (int p = row_ptr[row]; p < row_ptr[row + 1]; ++p) {
        const int col = col_ind[p];
        val[p] /= (rs * col_scale[col]);
    }
}

__global__ static void row_max_abs_implicit_unit_csr_kernel(
    const int *row_ptr,
    const int *col_ind,
    const double *row_scale,
    const double *col_scale,
    double *out,
    int n_rows)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) {
        return;
    }
    const double rs = (row_scale != nullptr && row_scale[row] != 0.0) ? row_scale[row] : 1.0;
    double max_val = 0.0;
    for (int p = row_ptr[row]; p < row_ptr[row + 1]; ++p) {
        const int col = col_ind[p];
        const double cs = (col_scale != nullptr && col_scale[col] != 0.0) ? col_scale[col] : 1.0;
        max_val = fmax(max_val, fabs(1.0 / (rs * cs)));
    }
    out[row] = max_val;
}

__global__ static void row_power_sum_implicit_unit_csr_kernel(
    const int *row_ptr,
    const int *col_ind,
    const double *row_scale,
    const double *col_scale,
    double *out,
    int n_rows,
    double power)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) {
        return;
    }
    const double rs = (row_scale != nullptr && row_scale[row] != 0.0) ? row_scale[row] : 1.0;
    double sum = 0.0;
    for (int p = row_ptr[row]; p < row_ptr[row + 1]; ++p) {
        const int col = col_ind[p];
        const double cs = (col_scale != nullptr && col_scale[col] != 0.0) ? col_scale[col] : 1.0;
        sum += pow(fabs(1.0 / (rs * cs)), power);
    }
    out[row] = sum;
}


__global__ static void scale_constraint_bounds_by_scalar_kernel(
    double *lower,
    double *upper,
    int n,
    double scale)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        lower[i] *= scale;
        upper[i] *= scale;
    }
}

__global__ static void scale_variable_bounds_and_objective_kernel(
    double *lower,
    double *upper,
    double *objective,
    int n,
    double bound_scale,
    double objective_scale)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        lower[i] *= bound_scale;
        upper[i] *= bound_scale;
        objective[i] *= objective_scale;
    }
}

__global__ static void rhs_bound_square_contrib_kernel(
    const double *rhs,
    double *out,
    int n,
    int n_eqs)
{
    (void)n_eqs;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const double up = rhs[i];
        out[i] = up * up;
    }
}


__global__ static void count_transpose_rows_kernel(
    const int *col_ind,
    int *row_ptr_t,
    int nnz)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < nnz) {
        atomicAdd(row_ptr_t + col_ind[p] + 1, 1);
    }
}

__global__ static void fill_transpose_col_indices_kernel(
    const int *row_ptr,
    const int *col_ind,
    int *next_t,
    int *col_ind_t,
    int n_rows)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) {
        return;
    }
    for (int p = row_ptr[row]; p < row_ptr[row + 1]; ++p) {
        const int col = col_ind[p];
        const int dst = atomicAdd(next_t + col, 1);
        col_ind_t[dst] = row;
    }
}

static void build_transpose_structure_from_csr_device(
    const int *row_ptr,
    const int *col_ind,
    int n_rows,
    int n_cols,
    int nnz,
    int **row_ptr_t,
    int **col_ind_t)
{
    CUDA_CHECK(cudaMalloc(row_ptr_t, (size_t)(n_cols + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(col_ind_t, (size_t)nnz * sizeof(int)));
    CUDA_CHECK(cudaMemset(*row_ptr_t, 0, (size_t)(n_cols + 1) * sizeof(int)));
    const int blocks_nnz = (nnz + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    const int blocks_rows = (n_rows + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    count_transpose_rows_kernel<<<blocks_nnz, THREADS_PER_BLOCK>>>(col_ind, *row_ptr_t, nnz);
    CUDA_CHECK(cudaGetLastError());
    thrust::device_ptr<int> row_ptr_t_begin(*row_ptr_t);
    thrust::inclusive_scan(thrust::device, row_ptr_t_begin, row_ptr_t_begin + n_cols + 1, row_ptr_t_begin);
    int *next_t = nullptr;
    CUDA_CHECK(cudaMalloc(&next_t, (size_t)n_cols * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(next_t, *row_ptr_t, (size_t)n_cols * sizeof(int), cudaMemcpyDeviceToDevice));
    fill_transpose_col_indices_kernel<<<blocks_rows, THREADS_PER_BLOCK>>>(row_ptr, col_ind, next_t, *col_ind_t, n_rows);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFree(next_t));
}

static void build_transpose_from_csr_device(
    const int *row_ptr,
    const int *col_ind,
    const double *val,
    int n_rows,
    int n_cols,
    int nnz,
    int **row_ptr_t,
    int **col_ind_t,
    double **val_t)
{
    CUDA_CHECK(cudaMalloc(row_ptr_t, (size_t)(n_cols + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(col_ind_t, (size_t)nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(val_t, (size_t)nnz * sizeof(double)));

    cusparseHandle_t handle = nullptr;
    CUSPARSE_CHECK(cusparseCreate(&handle));
    size_t buffer_size = 0;
    void *buffer = nullptr;
    CUSPARSE_CHECK(cusparseCsr2cscEx2_bufferSize(
        handle, n_rows, n_cols, nnz,
        val, row_ptr, col_ind,
        *val_t, *row_ptr_t, *col_ind_t,
        CUDA_R_64F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
        CUSPARSE_CSR2CSC_ALG1, &buffer_size));
    CUDA_CHECK(cudaMalloc(&buffer, buffer_size));
    CUSPARSE_CHECK(cusparseCsr2cscEx2(
        handle, n_rows, n_cols, nnz,
        val, row_ptr, col_ind,
        *val_t, *row_ptr_t, *col_ind_t,
        CUDA_R_64F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
        CUSPARSE_CSR2CSC_ALG1, buffer));
    CUDA_CHECK(cudaFree(buffer));
    CUSPARSE_CHECK(cusparseDestroy(handle));
}

__global__ static void materialize_implicit_transpose_values_kernel(
    const int *row_ptr_t,
    const int *col_ind_t,
    const double *variable_rescaling,
    const double *constraint_rescaling,
    double *val_t,
    int n_vars)
{
    int var = blockIdx.x * blockDim.x + threadIdx.x;
    if (var >= n_vars) {
        return;
    }
    const double vs = (variable_rescaling != nullptr && variable_rescaling[var] != 0.0) ? variable_rescaling[var] : 1.0;
    for (int p = row_ptr_t[var]; p < row_ptr_t[var + 1]; ++p) {
        const int con = col_ind_t[p];
        const double cs = (constraint_rescaling != nullptr && constraint_rescaling[con] != 0.0) ? constraint_rescaling[con] : 1.0;
        val_t[p] = 1.0 / (vs * cs);
    }
}

static device_rescale_info_t *device_rescale_info_create_zeroed()
{
    device_rescale_info_t *info = (device_rescale_info_t *)safe_calloc(1, sizeof(device_rescale_info_t));
    info->con_bound_rescale = 1.0;
    info->obj_vec_rescale = 1.0;
    return info;
}

static void device_rescale_info_release_to_state(device_rescale_info_t *info)
{
    if (info == nullptr) {
        return;
    }
    info->scaled_problem = device_lp_problem_t{};
    info->constraint_lower_bound = nullptr;
    info->constraint_upper_bound = nullptr;
    info->constraint_matrix_t_row_pointers = nullptr;
    info->constraint_matrix_t_col_indices = nullptr;
    info->constraint_matrix_t_values = nullptr;
    info->con_rescale = nullptr;
    info->var_rescale = nullptr;
}

static void device_rescale_info_free(device_rescale_info_t *info)
{
    if (info == nullptr) {
        return;
    }
    if (info->scaled_problem.constraint_matrix_row_pointers)
        CUDA_CHECK(cudaFree((void *)info->scaled_problem.constraint_matrix_row_pointers));
    if (info->scaled_problem.constraint_matrix_col_indices)
        CUDA_CHECK(cudaFree((void *)info->scaled_problem.constraint_matrix_col_indices));
    if (info->scaled_problem.constraint_matrix_values)
        CUDA_CHECK(cudaFree((void *)info->scaled_problem.constraint_matrix_values));
    if (info->scaled_problem.variable_lower_bound)
        CUDA_CHECK(cudaFree((void *)info->scaled_problem.variable_lower_bound));
    if (info->scaled_problem.variable_upper_bound)
        CUDA_CHECK(cudaFree((void *)info->scaled_problem.variable_upper_bound));
    if (info->scaled_problem.objective_vector)
        CUDA_CHECK(cudaFree((void *)info->scaled_problem.objective_vector));
    if (info->constraint_lower_bound)
        CUDA_CHECK(cudaFree(info->constraint_lower_bound));
    if (info->constraint_upper_bound)
        CUDA_CHECK(cudaFree(info->constraint_upper_bound));
    if (info->constraint_matrix_t_row_pointers)
        CUDA_CHECK(cudaFree(info->constraint_matrix_t_row_pointers));
    if (info->constraint_matrix_t_col_indices)
        CUDA_CHECK(cudaFree(info->constraint_matrix_t_col_indices));
    if (info->constraint_matrix_t_values)
        CUDA_CHECK(cudaFree(info->constraint_matrix_t_values));
    if (info->con_rescale)
        CUDA_CHECK(cudaFree(info->con_rescale));
    if (info->var_rescale)
        CUDA_CHECK(cudaFree(info->var_rescale));
    free(info);
}

static device_rescale_info_t *rescale_problem_device(
    const pdhg_parameters_t *params,
    const device_lp_problem_t *original_problem)
{
    auto t0 = clk::now();
    const int n_vars = original_problem->num_variables;
    const int n_cons = original_problem->num_constraints;
    const int nnz = original_problem->constraint_matrix_num_nonzeros;
    const size_t var_bytes = (size_t)n_vars * sizeof(double);
    const size_t con_bytes = (size_t)n_cons * sizeof(double);
    const size_t nnz_val_bytes = (size_t)nnz * sizeof(double);
    const size_t nnz_idx_bytes = (size_t)nnz * sizeof(int);
    const size_t row_ptr_bytes = (size_t)(n_cons + 1) * sizeof(int);
    const bool implicit_a_values = matrix_value_mode_has_implicit_ax(original_problem->matrix_value_mode);
    const bool implicit_at_values = matrix_value_mode_has_implicit_aty(original_problem->matrix_value_mode);
    const int blocks_primal = (n_vars + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    const int blocks_dual = (n_cons + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    device_rescale_info_t *info = device_rescale_info_create_zeroed();
    info->scaled_problem.num_variables = n_vars;
    info->scaled_problem.num_constraints = n_cons;
    info->scaled_problem.constraint_matrix_num_nonzeros = nnz;
    info->scaled_problem.matrix_value_mode = original_problem->matrix_value_mode;
    info->scaled_problem.num_equalities = original_problem->num_equalities;
    info->scaled_problem.objective_constant = original_problem->objective_constant;
    info->scaled_problem.right_hand_side = nullptr;
    const bool input_constant_bounds = original_problem->variable_bound_mode == VARIABLE_BOUNDS_CONSTANT;
    const bool keep_constant_bounds = input_constant_bounds &&
        original_problem->variable_lower_bound_constant == 0.0 &&
        isinf(original_problem->variable_upper_bound_constant) &&
        original_problem->variable_upper_bound_constant > 0.0;
    info->scaled_problem.variable_bound_mode = keep_constant_bounds ? VARIABLE_BOUNDS_CONSTANT : VARIABLE_BOUNDS_EXPLICIT;
    info->scaled_problem.variable_lower_bound_constant = original_problem->variable_lower_bound_constant;
    info->scaled_problem.variable_upper_bound_constant = original_problem->variable_upper_bound_constant;

    CUDA_CHECK(cudaMalloc((void **)&info->scaled_problem.constraint_matrix_row_pointers, row_ptr_bytes));
    CUDA_CHECK(cudaMalloc((void **)&info->scaled_problem.constraint_matrix_col_indices, nnz_idx_bytes));
    if (!implicit_a_values) {
        CUDA_CHECK(cudaMalloc((void **)&info->scaled_problem.constraint_matrix_values, nnz_val_bytes));
    }
    if (info->scaled_problem.variable_bound_mode == VARIABLE_BOUNDS_EXPLICIT) {
        CUDA_CHECK(cudaMalloc((void **)&info->scaled_problem.variable_lower_bound, var_bytes));
        CUDA_CHECK(cudaMalloc((void **)&info->scaled_problem.variable_upper_bound, var_bytes));
    }
    CUDA_CHECK(cudaMalloc((void **)&info->scaled_problem.objective_vector, var_bytes));
    CUDA_CHECK(cudaMalloc((void **)&info->con_rescale, con_bytes));
    CUDA_CHECK(cudaMalloc((void **)&info->var_rescale, var_bytes));

    CUDA_CHECK(cudaMemcpy((void *)info->scaled_problem.constraint_matrix_row_pointers, original_problem->constraint_matrix_row_pointers, row_ptr_bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy((void *)info->scaled_problem.constraint_matrix_col_indices, original_problem->constraint_matrix_col_indices, nnz_idx_bytes, cudaMemcpyDeviceToDevice));
    if (!implicit_a_values) {
        CUDA_CHECK(cudaMemcpy((void *)info->scaled_problem.constraint_matrix_values, original_problem->constraint_matrix_values, nnz_val_bytes, cudaMemcpyDeviceToDevice));
    }
    if (info->scaled_problem.variable_bound_mode == VARIABLE_BOUNDS_EXPLICIT) {
        if (input_constant_bounds) {
            fill_double_kernel<<<blocks_primal, THREADS_PER_BLOCK>>>(
                (double *)info->scaled_problem.variable_lower_bound, n_vars, original_problem->variable_lower_bound_constant);
            fill_double_kernel<<<blocks_primal, THREADS_PER_BLOCK>>>(
                (double *)info->scaled_problem.variable_upper_bound, n_vars, original_problem->variable_upper_bound_constant);
        } else {
            CUDA_CHECK(cudaMemcpy((void *)info->scaled_problem.variable_lower_bound, original_problem->variable_lower_bound, var_bytes, cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy((void *)info->scaled_problem.variable_upper_bound, original_problem->variable_upper_bound, var_bytes, cudaMemcpyDeviceToDevice));
        }
    }
    CUDA_CHECK(cudaMemcpy((void *)info->scaled_problem.objective_vector, original_problem->objective_vector, var_bytes, cudaMemcpyDeviceToDevice));
    fill_double_kernel<<<blocks_dual, THREADS_PER_BLOCK>>>(info->con_rescale, n_cons, 1.0);
    fill_double_kernel<<<blocks_primal, THREADS_PER_BLOCK>>>(info->var_rescale, n_vars, 1.0);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMalloc(&info->constraint_lower_bound, con_bytes));
    CUDA_CHECK(cudaMalloc(&info->constraint_upper_bound, con_bytes));
    fill_constraint_bounds_from_rhs_kernel<<<blocks_dual, THREADS_PER_BLOCK>>>(
        original_problem->right_hand_side,
        info->constraint_lower_bound,
        info->constraint_upper_bound,
        n_cons,
        original_problem->num_equalities);
    CUDA_CHECK(cudaGetLastError());

    if (implicit_a_values || implicit_at_values) {
        build_transpose_structure_from_csr_device(
            original_problem->constraint_matrix_row_pointers,
            original_problem->constraint_matrix_col_indices,
            n_cons,
            n_vars,
            nnz,
            &info->constraint_matrix_t_row_pointers,
            &info->constraint_matrix_t_col_indices);
        info->constraint_matrix_t_values = nullptr;
    } else {
        build_transpose_from_csr_device(
            original_problem->constraint_matrix_row_pointers,
            original_problem->constraint_matrix_col_indices,
            original_problem->constraint_matrix_values,
            n_cons,
            n_vars,
            nnz,
            &info->constraint_matrix_t_row_pointers,
            &info->constraint_matrix_t_col_indices,
            &info->constraint_matrix_t_values);
    }

    double *tmp_con = nullptr;
    double *tmp_var = nullptr;
    CUDA_CHECK(cudaMalloc(&tmp_con, con_bytes));
    CUDA_CHECK(cudaMalloc(&tmp_var, var_bytes));

    for (int iter = 0; iter < params->l_inf_ruiz_iterations; ++iter) {
        if (implicit_a_values) {
            row_max_abs_implicit_unit_csr_kernel<<<blocks_dual, THREADS_PER_BLOCK>>>(
                (const int *)info->scaled_problem.constraint_matrix_row_pointers,
                (const int *)info->scaled_problem.constraint_matrix_col_indices,
                info->con_rescale,
                info->var_rescale,
                tmp_con,
                n_cons);
        } else {
            row_max_abs_csr_kernel<<<blocks_dual, THREADS_PER_BLOCK>>>(
                (const int *)info->scaled_problem.constraint_matrix_row_pointers,
                (const double *)info->scaled_problem.constraint_matrix_values,
                tmp_con,
                n_cons);
        }
        if (implicit_at_values || implicit_a_values) {
            row_max_abs_implicit_unit_csr_kernel<<<blocks_primal, THREADS_PER_BLOCK>>>(
                info->constraint_matrix_t_row_pointers,
                info->constraint_matrix_t_col_indices,
                info->var_rescale,
                info->con_rescale,
                tmp_var,
                n_vars);
        } else {
            row_max_abs_csr_kernel<<<blocks_primal, THREADS_PER_BLOCK>>>(
                info->constraint_matrix_t_row_pointers,
                info->constraint_matrix_t_values,
                tmp_var,
                n_vars);
        }
        sqrt_clip_inplace_kernel<<<blocks_dual, THREADS_PER_BLOCK>>>(tmp_con, n_cons, 1e-12);
        sqrt_clip_inplace_kernel<<<blocks_primal, THREADS_PER_BLOCK>>>(tmp_var, n_vars, 1e-12);
        scale_problem_constraints_kernel<<<blocks_dual, THREADS_PER_BLOCK>>>(info->constraint_lower_bound, info->constraint_upper_bound, tmp_con, n_cons);
        if (info->scaled_problem.variable_bound_mode == VARIABLE_BOUNDS_EXPLICIT) {
            scale_problem_variables_kernel<<<blocks_primal, THREADS_PER_BLOCK>>>(
                (double *)info->scaled_problem.objective_vector,
                (double *)info->scaled_problem.variable_lower_bound,
                (double *)info->scaled_problem.variable_upper_bound,
                tmp_var,
                n_vars);
        } else {
            scale_objective_by_variable_kernel<<<blocks_primal, THREADS_PER_BLOCK>>>(
                (double *)info->scaled_problem.objective_vector,
                tmp_var,
                n_vars);
        }
        if (!implicit_a_values) {
            scale_matrix_csr_kernel<<<blocks_dual, THREADS_PER_BLOCK>>>(
                (const int *)info->scaled_problem.constraint_matrix_row_pointers,
                (const int *)info->scaled_problem.constraint_matrix_col_indices,
                (double *)info->scaled_problem.constraint_matrix_values,
                tmp_con,
                tmp_var,
                n_cons);
            if (!implicit_at_values) {
                scale_matrix_csr_kernel<<<blocks_primal, THREADS_PER_BLOCK>>>(
                    info->constraint_matrix_t_row_pointers,
                    info->constraint_matrix_t_col_indices,
                    info->constraint_matrix_t_values,
                    tmp_var,
                    tmp_con,
                    n_vars);
            }
        }
        multiply_inplace_kernel<<<blocks_dual, THREADS_PER_BLOCK>>>(info->con_rescale, tmp_con, n_cons);
        multiply_inplace_kernel<<<blocks_primal, THREADS_PER_BLOCK>>>(info->var_rescale, tmp_var, n_vars);
        CUDA_CHECK(cudaGetLastError());
    }

    if (params->has_pock_chambolle_alpha) {
        if (implicit_a_values) {
            row_power_sum_implicit_unit_csr_kernel<<<blocks_dual, THREADS_PER_BLOCK>>>(
                (const int *)info->scaled_problem.constraint_matrix_row_pointers,
                (const int *)info->scaled_problem.constraint_matrix_col_indices,
                info->con_rescale,
                info->var_rescale,
                tmp_con,
                n_cons,
                params->pock_chambolle_alpha);
        } else {
            row_power_sum_csr_kernel<<<blocks_dual, THREADS_PER_BLOCK>>>(
                (const int *)info->scaled_problem.constraint_matrix_row_pointers,
                (const double *)info->scaled_problem.constraint_matrix_values,
                tmp_con,
                n_cons,
                params->pock_chambolle_alpha);
        }
        if (implicit_at_values || implicit_a_values) {
            row_power_sum_implicit_unit_csr_kernel<<<blocks_primal, THREADS_PER_BLOCK>>>(
                info->constraint_matrix_t_row_pointers,
                info->constraint_matrix_t_col_indices,
                info->var_rescale,
                info->con_rescale,
                tmp_var,
                n_vars,
                2.0 - params->pock_chambolle_alpha);
        } else {
            row_power_sum_csr_kernel<<<blocks_primal, THREADS_PER_BLOCK>>>(
                info->constraint_matrix_t_row_pointers,
                info->constraint_matrix_t_values,
                tmp_var,
                n_vars,
                2.0 - params->pock_chambolle_alpha);
        }
        sqrt_clip_inplace_kernel<<<blocks_dual, THREADS_PER_BLOCK>>>(tmp_con, n_cons, 1e-12);
        sqrt_clip_inplace_kernel<<<blocks_primal, THREADS_PER_BLOCK>>>(tmp_var, n_vars, 1e-12);
        scale_problem_constraints_kernel<<<blocks_dual, THREADS_PER_BLOCK>>>(info->constraint_lower_bound, info->constraint_upper_bound, tmp_con, n_cons);
        if (info->scaled_problem.variable_bound_mode == VARIABLE_BOUNDS_EXPLICIT) {
            scale_problem_variables_kernel<<<blocks_primal, THREADS_PER_BLOCK>>>(
                (double *)info->scaled_problem.objective_vector,
                (double *)info->scaled_problem.variable_lower_bound,
                (double *)info->scaled_problem.variable_upper_bound,
                tmp_var,
                n_vars);
        } else {
            scale_objective_by_variable_kernel<<<blocks_primal, THREADS_PER_BLOCK>>>(
                (double *)info->scaled_problem.objective_vector,
                tmp_var,
                n_vars);
        }
        if (!implicit_a_values) {
            scale_matrix_csr_kernel<<<blocks_dual, THREADS_PER_BLOCK>>>(
                (const int *)info->scaled_problem.constraint_matrix_row_pointers,
                (const int *)info->scaled_problem.constraint_matrix_col_indices,
                (double *)info->scaled_problem.constraint_matrix_values,
                tmp_con,
                tmp_var,
                n_cons);
            if (!implicit_at_values) {
                scale_matrix_csr_kernel<<<blocks_primal, THREADS_PER_BLOCK>>>(
                    info->constraint_matrix_t_row_pointers,
                    info->constraint_matrix_t_col_indices,
                    info->constraint_matrix_t_values,
                    tmp_var,
                    tmp_con,
                    n_vars);
            }
        }
        multiply_inplace_kernel<<<blocks_dual, THREADS_PER_BLOCK>>>(info->con_rescale, tmp_con, n_cons);
        multiply_inplace_kernel<<<blocks_primal, THREADS_PER_BLOCK>>>(info->var_rescale, tmp_var, n_vars);
        CUDA_CHECK(cudaGetLastError());
    }

    if (params->bound_objective_rescaling) {
        cublasHandle_t blas = nullptr;
        CUBLAS_CHECK(cublasCreate(&blas));
        CUBLAS_CHECK(cublasSetPointerMode(blas, CUBLAS_POINTER_MODE_HOST));
        double obj_norm = 0.0;
        CUBLAS_CHECK(cublasDnrm2(blas, n_vars, (const double *)info->scaled_problem.objective_vector, 1, &obj_norm));
        double *bound_contrib = nullptr;
        CUDA_CHECK(cudaMalloc(&bound_contrib, con_bytes));
        constraint_bound_square_contrib_kernel<<<blocks_dual, THREADS_PER_BLOCK>>>(
            info->constraint_lower_bound, info->constraint_upper_bound, bound_contrib, n_cons);
        CUDA_CHECK(cudaGetLastError());
        const double bound_norm_sq = thrust::reduce(thrust::device, bound_contrib, bound_contrib + n_cons, 0.0, thrust::plus<double>());
        CUDA_CHECK(cudaFree(bound_contrib));
        info->con_bound_rescale = 1.0 / (sqrt(bound_norm_sq) + 1.0);
        info->obj_vec_rescale = 1.0 / (obj_norm + 1.0);
        scale_constraint_bounds_by_scalar_kernel<<<blocks_dual, THREADS_PER_BLOCK>>>(
            info->constraint_lower_bound, info->constraint_upper_bound, n_cons, info->con_bound_rescale);
        if (info->scaled_problem.variable_bound_mode == VARIABLE_BOUNDS_EXPLICIT) {
            scale_variable_bounds_and_objective_kernel<<<blocks_primal, THREADS_PER_BLOCK>>>(
                (double *)info->scaled_problem.variable_lower_bound,
                (double *)info->scaled_problem.variable_upper_bound,
                (double *)info->scaled_problem.objective_vector,
                n_vars,
                info->con_bound_rescale,
                info->obj_vec_rescale);
        } else {
            scale_objective_by_scalar_kernel<<<blocks_primal, THREADS_PER_BLOCK>>>(
                (double *)info->scaled_problem.objective_vector,
                n_vars,
                info->obj_vec_rescale);
        }
        CUDA_CHECK(cudaGetLastError());
        CUBLAS_CHECK(cublasDestroy(blas));
    }

    if (implicit_a_values && !implicit_at_values) {
        CUDA_CHECK(cudaMalloc(&info->constraint_matrix_t_values, nnz_val_bytes));
        materialize_implicit_transpose_values_kernel<<<blocks_primal, THREADS_PER_BLOCK>>>(
            info->constraint_matrix_t_row_pointers,
            info->constraint_matrix_t_col_indices,
            info->var_rescale,
            info->con_rescale,
            info->constraint_matrix_t_values,
            n_vars);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaFree(tmp_con));
    CUDA_CHECK(cudaFree(tmp_var));

    auto t1 = clk::now();
    info->rescaling_time_sec = std::chrono::duration<double>(t1 - t0).count();
    return info;
}

static pdhg_solver_state_t *initialize_solver_state_device_rescaled(
    const device_lp_problem_t *original_problem,
    device_rescale_info_t *rescale_info,
    const pdhg_parameters_t *params)
{
    pdhg_solver_state_t *state = (pdhg_solver_state_t *)safe_calloc(1, sizeof(pdhg_solver_state_t));

    const int n_vars = original_problem->num_variables;
    const int n_cons = original_problem->num_constraints;
    const int nnz = original_problem->constraint_matrix_num_nonzeros;
    const size_t var_bytes = (size_t)n_vars * sizeof(double);
    const size_t con_bytes = (size_t)n_cons * sizeof(double);

    state->num_variables = n_vars;
    state->num_constraints = n_cons;
    state->objective_constant = original_problem->objective_constant;
    state->trace_enabled = (params != nullptr) ? bool(params->trace_enabled) : false;
    state->trace_max_snapshots = (params != nullptr && params->trace_max_snapshots > 0) ? int(params->trace_max_snapshots) : 0;
    state->trace_num_snapshots = 0;
    state->borrows_input_buffers = false;
    state->termination_norm = (params != nullptr) ? params->termination_norm : TERMINATION_NORM_L2;

    state->constraint_matrix = (cu_sparse_matrix_csr_t *)safe_malloc(sizeof(cu_sparse_matrix_csr_t));
    state->constraint_matrix_t = (cu_sparse_matrix_csr_t *)safe_malloc(sizeof(cu_sparse_matrix_csr_t));
    state->constraint_matrix->num_rows = n_cons;
    state->constraint_matrix->num_cols = n_vars;
    state->constraint_matrix->num_nonzeros = nnz;
    state->constraint_matrix_t->num_rows = n_vars;
    state->constraint_matrix_t->num_cols = n_cons;
    state->constraint_matrix_t->num_nonzeros = nnz;
    state->termination_reason = TERMINATION_REASON_UNSPECIFIED;
    state->support_stop_has_last_eval_obj = false;
    state->support_stop_last_eval_obj = 0.0;
    state->support_stop_obj_rel_change = INFINITY;
    state->support_stop_last_eval_nnz = -1;

    state->matrix_value_mode = rescale_info->scaled_problem.matrix_value_mode;
    state->vector_sum_mode = (params != nullptr) ? params->vector_sum_mode : VECTOR_SUM_RESIDENT_ONES;
    state->constraint_matrix->row_ptr = (int *)rescale_info->scaled_problem.constraint_matrix_row_pointers;
    state->constraint_matrix->col_ind = (int *)rescale_info->scaled_problem.constraint_matrix_col_indices;
    state->constraint_matrix->val = (double *)rescale_info->scaled_problem.constraint_matrix_values;
    state->constraint_matrix_t->row_ptr = rescale_info->constraint_matrix_t_row_pointers;
    state->constraint_matrix_t->col_ind = rescale_info->constraint_matrix_t_col_indices;
    state->constraint_matrix_t->val = rescale_info->constraint_matrix_t_values;
    state->variable_bound_mode = rescale_info->scaled_problem.variable_bound_mode;
    state->variable_lower_bound_constant = rescale_info->scaled_problem.variable_lower_bound_constant;
    state->variable_upper_bound_constant = rescale_info->scaled_problem.variable_upper_bound_constant;
    state->variable_lower_bound = (double *)rescale_info->scaled_problem.variable_lower_bound;
    state->variable_upper_bound = (double *)rescale_info->scaled_problem.variable_upper_bound;
    state->objective_vector = (double *)rescale_info->scaled_problem.objective_vector;
    state->constraint_lower_bound = rescale_info->constraint_lower_bound;
    state->constraint_upper_bound = rescale_info->constraint_upper_bound;
    state->constraint_rescaling = rescale_info->con_rescale;
    state->variable_rescaling = rescale_info->var_rescale;
    state->constraint_bound_rescaling = rescale_info->con_bound_rescale;
    state->objective_vector_rescaling = rescale_info->obj_vec_rescale;

    CUSPARSE_CHECK(cusparseCreate(&state->sparse_handle));
    CUBLAS_CHECK(cublasCreate(&state->blas_handle));
    CUBLAS_CHECK(cublasSetPointerMode(state->blas_handle, CUBLAS_POINTER_MODE_HOST));

    if (state->trace_enabled && state->trace_max_snapshots > 0) {
        state->trace_iters_host = (int*)checked_host_calloc((size_t)state->trace_max_snapshots, sizeof(int));
        state->trace_primal_objectives_host = (double*)checked_host_calloc((size_t)state->trace_max_snapshots, sizeof(double));
        state->trace_dual_objectives_host = (double*)checked_host_calloc((size_t)state->trace_max_snapshots, sizeof(double));
        state->trace_primal_solutions_host = (double*)checked_host_calloc((size_t)state->trace_max_snapshots, var_bytes);
        state->trace_dual_solutions_host = (double*)checked_host_calloc((size_t)state->trace_max_snapshots, con_bytes);
        state->trace_variable_rescaling_host = (double*)checked_host_malloc(var_bytes);
        state->trace_constraint_rescaling_host = (double*)checked_host_malloc(con_bytes);
        CUDA_CHECK(cudaMemcpy(state->trace_variable_rescaling_host, state->variable_rescaling, var_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(state->trace_constraint_rescaling_host, state->constraint_rescaling, con_bytes, cudaMemcpyDeviceToHost));
    }

#define ALLOC_ZERO_DEVICE2(dest, bytes)   \
    CUDA_CHECK(cudaMalloc(&dest, bytes)); \
    CUDA_CHECK(cudaMemset(dest, 0, bytes));

    ALLOC_ZERO_DEVICE2(state->initial_primal_solution, var_bytes);
    ALLOC_ZERO_DEVICE2(state->current_primal_solution, var_bytes);
    ALLOC_ZERO_DEVICE2(state->pdhg_primal_solution, var_bytes);
    ALLOC_ZERO_DEVICE2(state->reflected_primal_solution, var_bytes);
    ALLOC_ZERO_DEVICE2(state->dual_product, var_bytes);
    ALLOC_ZERO_DEVICE2(state->dual_slack, var_bytes);
    ALLOC_ZERO_DEVICE2(state->dual_residual, var_bytes);
    ALLOC_ZERO_DEVICE2(state->delta_primal_solution, var_bytes);

    ALLOC_ZERO_DEVICE2(state->initial_dual_solution, con_bytes);
    ALLOC_ZERO_DEVICE2(state->current_dual_solution, con_bytes);
    ALLOC_ZERO_DEVICE2(state->pdhg_dual_solution, con_bytes);
    ALLOC_ZERO_DEVICE2(state->reflected_dual_solution, con_bytes);
    ALLOC_ZERO_DEVICE2(state->primal_product, con_bytes);
    ALLOC_ZERO_DEVICE2(state->primal_slack, con_bytes);
    ALLOC_ZERO_DEVICE2(state->primal_residual, con_bytes);
    ALLOC_ZERO_DEVICE2(state->delta_dual_solution, con_bytes);

    copy_scaled_initial_buffers(state, params);

    CUDA_CHECK(cudaMalloc(&state->constraint_lower_bound_finite_val, con_bytes));
    CUDA_CHECK(cudaMalloc(&state->constraint_upper_bound_finite_val, con_bytes));
    const int blocks_dual = (n_cons + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    const int blocks_primal = (n_vars + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (state->variable_bound_mode == VARIABLE_BOUNDS_EXPLICIT) {
        CUDA_CHECK(cudaMalloc(&state->variable_lower_bound_finite_val, var_bytes));
        CUDA_CHECK(cudaMalloc(&state->variable_upper_bound_finite_val, var_bytes));
    }
    finite_or_zero_kernel<<<blocks_dual, THREADS_PER_BLOCK>>>(state->constraint_lower_bound, state->constraint_lower_bound_finite_val, n_cons);
    finite_or_zero_kernel<<<blocks_dual, THREADS_PER_BLOCK>>>(state->constraint_upper_bound, state->constraint_upper_bound_finite_val, n_cons);
    if (state->variable_bound_mode == VARIABLE_BOUNDS_EXPLICIT) {
        finite_or_zero_kernel<<<blocks_primal, THREADS_PER_BLOCK>>>(state->variable_lower_bound, state->variable_lower_bound_finite_val, n_vars);
        finite_or_zero_kernel<<<blocks_primal, THREADS_PER_BLOCK>>>(state->variable_upper_bound, state->variable_upper_bound_finite_val, n_vars);
    }
    CUDA_CHECK(cudaGetLastError());

    CUBLAS_CHECK(cublasDnrm2(state->blas_handle, n_vars, original_problem->objective_vector, 1, &state->objective_vector_norm));
    double *bound_contrib = nullptr;
    CUDA_CHECK(cudaMalloc(&bound_contrib, con_bytes));
    rhs_bound_square_contrib_kernel<<<blocks_dual, THREADS_PER_BLOCK>>>(
        original_problem->right_hand_side, bound_contrib, n_cons, original_problem->num_equalities);
    CUDA_CHECK(cudaGetLastError());
    const double bound_norm_sq = thrust::reduce(thrust::device, bound_contrib, bound_contrib + n_cons, 0.0, thrust::plus<double>());
    state->constraint_bound_norm = sqrt(bound_norm_sq);
    CUDA_CHECK(cudaFree(bound_contrib));
    state->termination_objective_vector_norm = state->objective_vector_norm;
    state->termination_constraint_bound_norm = state->constraint_bound_norm;
    if (state->termination_norm == TERMINATION_NORM_L_INF) {
        state->termination_objective_vector_norm =
            device_inf_norm(state->blas_handle, original_problem->objective_vector, n_vars);
        state->termination_constraint_bound_norm =
            device_inf_norm(state->blas_handle, original_problem->right_hand_side, n_cons);
    }

    state->num_blocks_primal = blocks_primal;
    state->num_blocks_dual = blocks_dual;
    state->num_blocks_primal_dual = (state->num_variables + state->num_constraints + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    state->best_primal_dual_residual_gap = INFINITY;
    state->best_relative_objective_gap = INFINITY;
    state->last_trial_fixed_point_error = INFINITY;
    state->step_size = 0.0;
    state->is_this_major_iteration = false;

    size_t primal_spmv_buffer_size = 0;
    size_t dual_spmv_buffer_size = 0;
    const bool has_explicit_A = matrix_value_mode_has_explicit_a(state->matrix_value_mode);
    const bool has_explicit_AT = matrix_value_mode_has_explicit_at(state->matrix_value_mode);
    if (has_explicit_A) {
        CUSPARSE_CHECK(cusparseCreateCsr(&state->matA, state->num_constraints, state->num_variables, state->constraint_matrix->num_nonzeros, state->constraint_matrix->row_ptr, state->constraint_matrix->col_ind, state->constraint_matrix->val, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
        CUSPARSE_CHECK(cusparseCreateDnVec(&state->vec_primal_sol, state->num_variables, state->pdhg_primal_solution, CUDA_R_64F));
        CUSPARSE_CHECK(cusparseCreateDnVec(&state->vec_primal_prod, state->num_constraints, state->primal_product, CUDA_R_64F));
        CUSPARSE_CHECK(cusparseSpMV_bufferSize(state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &HOST_ONE, state->matA, state->vec_primal_sol, &HOST_ZERO, state->vec_primal_prod, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, &primal_spmv_buffer_size));
        CUDA_CHECK(cudaMalloc(&state->primal_spmv_buffer, primal_spmv_buffer_size));
    }
    if (has_explicit_AT) {
        CUSPARSE_CHECK(cusparseCreateCsr(&state->matAt, state->num_variables, state->num_constraints, state->constraint_matrix_t->num_nonzeros, state->constraint_matrix_t->row_ptr, state->constraint_matrix_t->col_ind, state->constraint_matrix_t->val, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
        CUSPARSE_CHECK(cusparseCreateDnVec(&state->vec_dual_sol, state->num_constraints, state->pdhg_dual_solution, CUDA_R_64F));
        CUSPARSE_CHECK(cusparseCreateDnVec(&state->vec_dual_prod, state->num_variables, state->dual_product, CUDA_R_64F));
        CUSPARSE_CHECK(cusparseSpMV_bufferSize(state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &HOST_ONE, state->matAt, state->vec_dual_sol, &HOST_ZERO, state->vec_dual_prod, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, &dual_spmv_buffer_size));
        CUDA_CHECK(cudaMalloc(&state->dual_spmv_buffer, dual_spmv_buffer_size));
    }
    if (state->vector_sum_mode == VECTOR_SUM_RESIDENT_ONES) {
        CUDA_CHECK(cudaMalloc(&state->ones_primal_d, var_bytes));
        CUDA_CHECK(cudaMalloc(&state->ones_dual_d, con_bytes));
        fill_double_kernel<<<blocks_primal, THREADS_PER_BLOCK>>>(state->ones_primal_d, n_vars, 1.0);
        fill_double_kernel<<<blocks_dual, THREADS_PER_BLOCK>>>(state->ones_dual_d, n_cons, 1.0);
        CUDA_CHECK(cudaGetLastError());
    }

    state->k_p = params->restart_params.k_p;
    state->k_i = params->restart_params.k_i;
    state->k_d = params->restart_params.k_d;
    state->previous_restart_dual_residual = DBL_MAX;
    state->previous_restart_gap = DBL_MAX;

    device_rescale_info_release_to_state(rescale_info);
    return state;
}

static pdhg_solver_state_t *initialize_solver_state_device_no_rescale(
    const device_lp_problem_t *device_problem,
    const pdhg_parameters_t *params)
{
    pdhg_solver_state_t *state = (pdhg_solver_state_t *)safe_calloc(1, sizeof(pdhg_solver_state_t));

    const int n_vars = device_problem->num_variables;
    const int n_cons = device_problem->num_constraints;
    const int nnz = device_problem->constraint_matrix_num_nonzeros;
    const size_t var_bytes = (size_t)n_vars * sizeof(double);
    const size_t con_bytes = (size_t)n_cons * sizeof(double);

    state->num_variables = n_vars;
    state->num_constraints = n_cons;
    state->objective_constant = device_problem->objective_constant;
    state->trace_enabled = (params != nullptr) ? bool(params->trace_enabled) : false;
    state->trace_max_snapshots = (params != nullptr && params->trace_max_snapshots > 0) ? int(params->trace_max_snapshots) : 0;
    state->trace_num_snapshots = 0;
    state->borrows_input_buffers = true;
    state->borrows_rescaling_buffers = false;
    state->termination_norm = (params != nullptr) ? params->termination_norm : TERMINATION_NORM_L2;

    state->constraint_matrix = (cu_sparse_matrix_csr_t *)safe_malloc(sizeof(cu_sparse_matrix_csr_t));
    state->constraint_matrix_t = (cu_sparse_matrix_csr_t *)safe_malloc(sizeof(cu_sparse_matrix_csr_t));
    state->constraint_matrix->num_rows = n_cons;
    state->constraint_matrix->num_cols = n_vars;
    state->constraint_matrix->num_nonzeros = nnz;
    state->matrix_value_mode = device_problem->matrix_value_mode;
    state->vector_sum_mode = (params != nullptr) ? params->vector_sum_mode : VECTOR_SUM_RESIDENT_ONES;
    state->constraint_matrix->row_ptr = const_cast<int *>(device_problem->constraint_matrix_row_pointers);
    state->constraint_matrix->col_ind = const_cast<int *>(device_problem->constraint_matrix_col_indices);
    state->constraint_matrix->val = const_cast<double *>(device_problem->constraint_matrix_values);

    state->constraint_matrix_t->num_rows = n_vars;
    state->constraint_matrix_t->num_cols = n_cons;
    state->constraint_matrix_t->num_nonzeros = nnz;
    state->termination_reason = TERMINATION_REASON_UNSPECIFIED;
    state->support_stop_has_last_eval_obj = false;
    state->support_stop_last_eval_obj = 0.0;
    state->support_stop_obj_rel_change = INFINITY;
    state->support_stop_last_eval_nnz = -1;

    state->variable_bound_mode = device_problem->variable_bound_mode;
    state->variable_lower_bound_constant = device_problem->variable_lower_bound_constant;
    state->variable_upper_bound_constant = device_problem->variable_upper_bound_constant;
    state->variable_lower_bound = const_cast<double *>(device_problem->variable_lower_bound);
    state->variable_upper_bound = const_cast<double *>(device_problem->variable_upper_bound);
    state->objective_vector = const_cast<double *>(device_problem->objective_vector);

    CUDA_CHECK(cudaMalloc(&state->constraint_lower_bound, con_bytes));
    CUDA_CHECK(cudaMalloc(&state->constraint_upper_bound, con_bytes));
    const int blocks_dual = (n_cons + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    fill_constraint_bounds_from_rhs_kernel<<<blocks_dual, THREADS_PER_BLOCK>>>(
        device_problem->right_hand_side,
        state->constraint_lower_bound,
        state->constraint_upper_bound,
        n_cons,
        device_problem->num_equalities);
    CUDA_CHECK(cudaGetLastError());

    CUSPARSE_CHECK(cusparseCreate(&state->sparse_handle));
    CUBLAS_CHECK(cublasCreate(&state->blas_handle));
    CUBLAS_CHECK(cublasSetPointerMode(state->blas_handle, CUBLAS_POINTER_MODE_HOST));

    if (matrix_value_mode_has_implicit_ax(state->matrix_value_mode) || matrix_value_mode_has_implicit_aty(state->matrix_value_mode)) {
        build_transpose_structure_from_csr_device(
            state->constraint_matrix->row_ptr,
            state->constraint_matrix->col_ind,
            n_cons,
            n_vars,
            nnz,
            &state->constraint_matrix_t->row_ptr,
            &state->constraint_matrix_t->col_ind);
        state->constraint_matrix_t->val = nullptr;
    } else {
        CUDA_CHECK(cudaMalloc(&state->constraint_matrix_t->row_ptr, (size_t)(n_vars + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&state->constraint_matrix_t->col_ind, (size_t)nnz * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&state->constraint_matrix_t->val, (size_t)nnz * sizeof(double)));

        size_t buffer_size = 0;
        void *buffer = nullptr;
        CUSPARSE_CHECK(cusparseCsr2cscEx2_bufferSize(
            state->sparse_handle, state->constraint_matrix->num_rows, state->constraint_matrix->num_cols, state->constraint_matrix->num_nonzeros,
            state->constraint_matrix->val, state->constraint_matrix->row_ptr, state->constraint_matrix->col_ind,
            state->constraint_matrix_t->val, state->constraint_matrix_t->row_ptr, state->constraint_matrix_t->col_ind,
            CUDA_R_64F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
            CUSPARSE_CSR2CSC_ALG1, &buffer_size));
        CUDA_CHECK(cudaMalloc(&buffer, buffer_size));
        CUSPARSE_CHECK(cusparseCsr2cscEx2(
            state->sparse_handle, state->constraint_matrix->num_rows, state->constraint_matrix->num_cols, state->constraint_matrix->num_nonzeros,
            state->constraint_matrix->val, state->constraint_matrix->row_ptr, state->constraint_matrix->col_ind,
            state->constraint_matrix_t->val, state->constraint_matrix_t->row_ptr, state->constraint_matrix_t->col_ind,
            CUDA_R_64F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
            CUSPARSE_CSR2CSC_ALG1, buffer));
        CUDA_CHECK(cudaFree(buffer));
    }

    const int blocks_primal = (n_vars + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    const bool has_precomputed_rescaling = device_problem->has_precomputed_rescaling;
    if (has_precomputed_rescaling) {
        state->constraint_rescaling = const_cast<double *>(device_problem->constraint_rescaling);
        state->variable_rescaling = const_cast<double *>(device_problem->variable_rescaling);
        state->constraint_bound_rescaling = device_problem->constraint_bound_rescaling;
        state->objective_vector_rescaling = device_problem->objective_vector_rescaling;
        state->objective_vector_norm = device_problem->original_objective_vector_norm;
        state->constraint_bound_norm = device_problem->original_constraint_bound_norm;
        state->termination_objective_vector_norm = state->objective_vector_norm;
        state->termination_constraint_bound_norm = state->constraint_bound_norm;
        if (state->termination_norm == TERMINATION_NORM_L_INF) {
            if (!(isfinite(device_problem->original_objective_vector_linf_norm) &&
                  device_problem->original_objective_vector_linf_norm >= 0.0 &&
                  isfinite(device_problem->original_constraint_bound_linf_norm) &&
                  device_problem->original_constraint_bound_linf_norm >= 0.0)) {
                throw std::invalid_argument(
                    "precomputed rescaling with termination_norm='linf' requires original Linf normalizers");
            }
            state->termination_objective_vector_norm = device_problem->original_objective_vector_linf_norm;
            state->termination_constraint_bound_norm = device_problem->original_constraint_bound_linf_norm;
        }
        state->borrows_rescaling_buffers = true;
    } else {
        CUDA_CHECK(cudaMalloc(&state->constraint_rescaling, con_bytes));
        CUDA_CHECK(cudaMalloc(&state->variable_rescaling, var_bytes));
        fill_double_kernel<<<blocks_dual, THREADS_PER_BLOCK>>>(state->constraint_rescaling, n_cons, 1.0);
        fill_double_kernel<<<blocks_primal, THREADS_PER_BLOCK>>>(state->variable_rescaling, n_vars, 1.0);
        CUDA_CHECK(cudaGetLastError());
        state->constraint_bound_rescaling = 1.0;
        state->objective_vector_rescaling = 1.0;
    }

    if (state->trace_enabled && state->trace_max_snapshots > 0) {
        state->trace_iters_host = (int*)checked_host_calloc((size_t)state->trace_max_snapshots, sizeof(int));
        state->trace_primal_objectives_host = (double*)checked_host_calloc((size_t)state->trace_max_snapshots, sizeof(double));
        state->trace_dual_objectives_host = (double*)checked_host_calloc((size_t)state->trace_max_snapshots, sizeof(double));
        state->trace_primal_solutions_host = (double*)checked_host_calloc((size_t)state->trace_max_snapshots, var_bytes);
        state->trace_dual_solutions_host = (double*)checked_host_calloc((size_t)state->trace_max_snapshots, con_bytes);
        state->trace_variable_rescaling_host = (double*)checked_host_malloc(var_bytes);
        state->trace_constraint_rescaling_host = (double*)checked_host_malloc(con_bytes);
        if (has_precomputed_rescaling) {
            CUDA_CHECK(cudaMemcpy(state->trace_variable_rescaling_host, state->variable_rescaling, var_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(state->trace_constraint_rescaling_host, state->constraint_rescaling, con_bytes, cudaMemcpyDeviceToHost));
        } else {
            std::fill_n(state->trace_variable_rescaling_host, n_vars, 1.0);
            std::fill_n(state->trace_constraint_rescaling_host, n_cons, 1.0);
        }
    }

    if (matrix_value_mode_has_implicit_ax(state->matrix_value_mode) && matrix_value_mode_has_explicit_at(state->matrix_value_mode)) {
        CUDA_CHECK(cudaMalloc(&state->constraint_matrix_t->val, (size_t)nnz * sizeof(double)));
        materialize_implicit_transpose_values_kernel<<<blocks_primal, THREADS_PER_BLOCK>>>(
            state->constraint_matrix_t->row_ptr,
            state->constraint_matrix_t->col_ind,
            state->variable_rescaling,
            state->constraint_rescaling,
            state->constraint_matrix_t->val,
            n_vars);
        CUDA_CHECK(cudaGetLastError());
    }

#define ALLOC_ZERO_DEVICE(dest, bytes)    \
    CUDA_CHECK(cudaMalloc(&dest, bytes)); \
    CUDA_CHECK(cudaMemset(dest, 0, bytes));

    ALLOC_ZERO_DEVICE(state->initial_primal_solution, var_bytes);
    ALLOC_ZERO_DEVICE(state->current_primal_solution, var_bytes);
    ALLOC_ZERO_DEVICE(state->pdhg_primal_solution, var_bytes);
    ALLOC_ZERO_DEVICE(state->reflected_primal_solution, var_bytes);
    ALLOC_ZERO_DEVICE(state->dual_product, var_bytes);
    ALLOC_ZERO_DEVICE(state->dual_slack, var_bytes);
    ALLOC_ZERO_DEVICE(state->dual_residual, var_bytes);
    ALLOC_ZERO_DEVICE(state->delta_primal_solution, var_bytes);

    ALLOC_ZERO_DEVICE(state->initial_dual_solution, con_bytes);
    ALLOC_ZERO_DEVICE(state->current_dual_solution, con_bytes);
    ALLOC_ZERO_DEVICE(state->pdhg_dual_solution, con_bytes);
    ALLOC_ZERO_DEVICE(state->reflected_dual_solution, con_bytes);
    ALLOC_ZERO_DEVICE(state->primal_product, con_bytes);
    ALLOC_ZERO_DEVICE(state->primal_slack, con_bytes);
    ALLOC_ZERO_DEVICE(state->primal_residual, con_bytes);
    ALLOC_ZERO_DEVICE(state->delta_dual_solution, con_bytes);

    copy_scaled_initial_buffers(state, params);

    CUDA_CHECK(cudaMalloc(&state->constraint_lower_bound_finite_val, con_bytes));
    CUDA_CHECK(cudaMalloc(&state->constraint_upper_bound_finite_val, con_bytes));
    if (state->variable_bound_mode == VARIABLE_BOUNDS_EXPLICIT) {
        CUDA_CHECK(cudaMalloc(&state->variable_lower_bound_finite_val, var_bytes));
        CUDA_CHECK(cudaMalloc(&state->variable_upper_bound_finite_val, var_bytes));
    }
    finite_or_zero_kernel<<<blocks_dual, THREADS_PER_BLOCK>>>(state->constraint_lower_bound, state->constraint_lower_bound_finite_val, n_cons);
    finite_or_zero_kernel<<<blocks_dual, THREADS_PER_BLOCK>>>(state->constraint_upper_bound, state->constraint_upper_bound_finite_val, n_cons);
    if (state->variable_bound_mode == VARIABLE_BOUNDS_EXPLICIT) {
        finite_or_zero_kernel<<<blocks_primal, THREADS_PER_BLOCK>>>(state->variable_lower_bound, state->variable_lower_bound_finite_val, n_vars);
        finite_or_zero_kernel<<<blocks_primal, THREADS_PER_BLOCK>>>(state->variable_upper_bound, state->variable_upper_bound_finite_val, n_vars);
    }
    CUDA_CHECK(cudaGetLastError());

    if (!has_precomputed_rescaling) {
        CUBLAS_CHECK(cublasDnrm2(state->blas_handle, n_vars, state->objective_vector, 1, &state->objective_vector_norm));

        double *bound_contrib = nullptr;
        CUDA_CHECK(cudaMalloc(&bound_contrib, con_bytes));
        constraint_bound_square_contrib_kernel<<<blocks_dual, THREADS_PER_BLOCK>>>(
            state->constraint_lower_bound,
            state->constraint_upper_bound,
            bound_contrib,
            n_cons);
        CUDA_CHECK(cudaGetLastError());
        const double bound_norm_sq = thrust::reduce(thrust::device, bound_contrib, bound_contrib + n_cons, 0.0, thrust::plus<double>());
        state->constraint_bound_norm = sqrt(bound_norm_sq);
        CUDA_CHECK(cudaFree(bound_contrib));
        state->termination_objective_vector_norm = state->objective_vector_norm;
        state->termination_constraint_bound_norm = state->constraint_bound_norm;
        if (state->termination_norm == TERMINATION_NORM_L_INF) {
            state->termination_objective_vector_norm =
                device_inf_norm(state->blas_handle, state->objective_vector, n_vars);
            state->termination_constraint_bound_norm = fmax(
                device_inf_norm(state->blas_handle, state->constraint_lower_bound_finite_val, n_cons),
                device_inf_norm(state->blas_handle, state->constraint_upper_bound_finite_val, n_cons));
        }
    }

    state->num_blocks_primal = blocks_primal;
    state->num_blocks_dual = blocks_dual;
    state->num_blocks_primal_dual = (state->num_variables + state->num_constraints + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    state->best_primal_dual_residual_gap = INFINITY;
    state->best_relative_objective_gap = INFINITY;
    state->last_trial_fixed_point_error = INFINITY;
    state->step_size = 0.0;
    state->is_this_major_iteration = false;

    size_t primal_spmv_buffer_size = 0;
    size_t dual_spmv_buffer_size = 0;
    const bool has_explicit_A = matrix_value_mode_has_explicit_a(state->matrix_value_mode);
    const bool has_explicit_AT = matrix_value_mode_has_explicit_at(state->matrix_value_mode);
    if (has_explicit_A) {
        CUSPARSE_CHECK(cusparseCreateCsr(&state->matA, state->num_constraints, state->num_variables, state->constraint_matrix->num_nonzeros, state->constraint_matrix->row_ptr, state->constraint_matrix->col_ind, state->constraint_matrix->val, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
        CUSPARSE_CHECK(cusparseCreateDnVec(&state->vec_primal_sol, state->num_variables, state->pdhg_primal_solution, CUDA_R_64F));
        CUSPARSE_CHECK(cusparseCreateDnVec(&state->vec_primal_prod, state->num_constraints, state->primal_product, CUDA_R_64F));
        CUSPARSE_CHECK(cusparseSpMV_bufferSize(state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &HOST_ONE, state->matA, state->vec_primal_sol, &HOST_ZERO, state->vec_primal_prod, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, &primal_spmv_buffer_size));
        CUDA_CHECK(cudaMalloc(&state->primal_spmv_buffer, primal_spmv_buffer_size));
    }
    if (has_explicit_AT) {
        CUSPARSE_CHECK(cusparseCreateCsr(&state->matAt, state->num_variables, state->num_constraints, state->constraint_matrix_t->num_nonzeros, state->constraint_matrix_t->row_ptr, state->constraint_matrix_t->col_ind, state->constraint_matrix_t->val, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
        CUSPARSE_CHECK(cusparseCreateDnVec(&state->vec_dual_sol, state->num_constraints, state->pdhg_dual_solution, CUDA_R_64F));
        CUSPARSE_CHECK(cusparseCreateDnVec(&state->vec_dual_prod, state->num_variables, state->dual_product, CUDA_R_64F));
        CUSPARSE_CHECK(cusparseSpMV_bufferSize(state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &HOST_ONE, state->matAt, state->vec_dual_sol, &HOST_ZERO, state->vec_dual_prod, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, &dual_spmv_buffer_size));
        CUDA_CHECK(cudaMalloc(&state->dual_spmv_buffer, dual_spmv_buffer_size));
    }

    if (state->vector_sum_mode == VECTOR_SUM_RESIDENT_ONES) {
        CUDA_CHECK(cudaMalloc(&state->ones_primal_d, var_bytes));
        CUDA_CHECK(cudaMalloc(&state->ones_dual_d, con_bytes));
        fill_double_kernel<<<blocks_primal, THREADS_PER_BLOCK>>>(state->ones_primal_d, n_vars, 1.0);
        fill_double_kernel<<<blocks_dual, THREADS_PER_BLOCK>>>(state->ones_dual_d, n_cons, 1.0);
        CUDA_CHECK(cudaGetLastError());
    }

    state->k_p = params->restart_params.k_p;
    state->k_i = params->restart_params.k_i;
    state->k_d = params->restart_params.k_d;
    state->previous_restart_dual_residual = DBL_MAX;
    state->previous_restart_gap = DBL_MAX;
    return state;
}

__global__ void compute_next_pdhg_primal_solution_kernel(
    const double *current_primal, double *reflected_primal, const double *dual_product,
    const double *objective, const double *var_lb, const double *var_ub,
    int n, double step_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        double temp = current_primal[i] - step_size * (objective[i] - dual_product[i]);
        double temp_proj = fmax(var_lb[i], fmin(temp, var_ub[i]));
        reflected_primal[i] = 2.0 * temp_proj - current_primal[i];
    }
}

__global__ void compute_next_pdhg_primal_solution_major_kernel(
    const double *current_primal, double *pdhg_primal, double *reflected_primal,
    const double *dual_product, const double *objective, const double *var_lb,
    const double *var_ub, int n, double step_size, double *dual_slack)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        double temp = current_primal[i] - step_size * (objective[i] - dual_product[i]);
        pdhg_primal[i] = fmax(var_lb[i], fmin(temp, var_ub[i]));
        dual_slack[i] = (pdhg_primal[i] - temp) / step_size;
        reflected_primal[i] = 2.0 * pdhg_primal[i] - current_primal[i];
    }
}

__global__ void compute_next_pdhg_primal_solution_constant_bounds_kernel(
    const double *current_primal, double *reflected_primal, const double *dual_product,
    const double *objective, double var_lb, double var_ub,
    int n, double step_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        double temp = current_primal[i] - step_size * (objective[i] - dual_product[i]);
        double temp_proj = fmax(var_lb, fmin(temp, var_ub));
        reflected_primal[i] = 2.0 * temp_proj - current_primal[i];
    }
}

__global__ void compute_next_pdhg_primal_solution_major_constant_bounds_kernel(
    const double *current_primal, double *pdhg_primal, double *reflected_primal,
    const double *dual_product, const double *objective, double var_lb,
    double var_ub, int n, double step_size, double *dual_slack)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        double temp = current_primal[i] - step_size * (objective[i] - dual_product[i]);
        pdhg_primal[i] = fmax(var_lb, fmin(temp, var_ub));
        dual_slack[i] = (pdhg_primal[i] - temp) / step_size;
        reflected_primal[i] = 2.0 * pdhg_primal[i] - current_primal[i];
    }
}

__global__ void compute_next_pdhg_dual_solution_kernel(
    const double *current_dual, double *reflected_dual, const double *primal_product,
    const double *const_lb, const double *const_ub, int n, double step_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        double temp = current_dual[i] / step_size - primal_product[i];
        double temp_proj = fmax(-const_ub[i], fmin(temp, -const_lb[i]));
        reflected_dual[i] = 2.0 * (temp - temp_proj) * step_size - current_dual[i];
    }
}

__global__ void compute_next_pdhg_dual_solution_major_kernel(
    const double *current_dual, double *pdhg_dual, double *reflected_dual,
    const double *primal_product, const double *const_lb, const double *const_ub,
    int n, double step_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        double temp = current_dual[i] / step_size - primal_product[i];
        double temp_proj = fmax(-const_ub[i], fmin(temp, -const_lb[i]));
        pdhg_dual[i] = (temp - temp_proj) * step_size;
        reflected_dual[i] = 2.0 * pdhg_dual[i] - current_dual[i];
    }
}

__global__ void halpern_update_kernel(
    const double *initial_primal, double *current_primal, const double *reflected_primal,
    const double *initial_dual, double *current_dual, const double *reflected_dual,
    int n_vars, int n_cons, double weight, double reflection_coeff)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_vars)
    {
        double reflected = reflection_coeff * reflected_primal[i] + (1.0 - reflection_coeff) * current_primal[i];
        current_primal[i] = weight * reflected + (1.0 - weight) * initial_primal[i];
    }
    else if (i < n_vars + n_cons)
    {
        int idx = i - n_vars;
        double reflected = reflection_coeff * reflected_dual[idx] + (1.0 - reflection_coeff) * current_dual[idx];
        current_dual[idx] = weight * reflected + (1.0 - weight) * initial_dual[idx];
    }
}

__global__ void compute_delta_solution_kernel(
    const double *initial_primal, const double *pdhg_primal, double *delta_primal,
    const double *initial_dual, const double *pdhg_dual, double *delta_dual,
    int n_vars, int n_cons)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_vars)
    {
        delta_primal[i] = pdhg_primal[i] - initial_primal[i];
    }
    else if (i < n_vars + n_cons)
    {
        int idx = i - n_vars;
        delta_dual[idx] = pdhg_dual[idx] - initial_dual[idx];
    }
}

static void compute_next_pdhg_primal_solution(pdhg_solver_state_t *state)
{
    spmv_AT(state, state->current_dual_solution, state->dual_product);

    double step = state->step_size / state->primal_weight;

    const bool use_constant_bounds = state->variable_bound_mode == VARIABLE_BOUNDS_CONSTANT;
    if (state->is_this_major_iteration || ((state->total_count + 2) % get_print_frequency(state->total_count + 2)) == 0)
    {
        if (use_constant_bounds)
        {
            compute_next_pdhg_primal_solution_major_constant_bounds_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
                state->current_primal_solution, state->pdhg_primal_solution, state->reflected_primal_solution,
                state->dual_product, state->objective_vector, state->variable_lower_bound_constant,
                state->variable_upper_bound_constant, state->num_variables, step, state->dual_slack);
        }
        else
        {
            compute_next_pdhg_primal_solution_major_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
                state->current_primal_solution, state->pdhg_primal_solution, state->reflected_primal_solution,
                state->dual_product, state->objective_vector, state->variable_lower_bound,
                state->variable_upper_bound, state->num_variables, step, state->dual_slack);
        }
    }
    else
    {
        if (use_constant_bounds)
        {
            compute_next_pdhg_primal_solution_constant_bounds_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
                state->current_primal_solution, state->reflected_primal_solution, state->dual_product,
                state->objective_vector, state->variable_lower_bound_constant, state->variable_upper_bound_constant,
                state->num_variables, step);
        }
        else
        {
            compute_next_pdhg_primal_solution_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
                state->current_primal_solution, state->reflected_primal_solution, state->dual_product,
                state->objective_vector, state->variable_lower_bound, state->variable_upper_bound,
                state->num_variables, step);
        }
    }
}

static void compute_next_pdhg_dual_solution(pdhg_solver_state_t *state)
{
    spmv_A(state, state->reflected_primal_solution, state->primal_product);

    double step = state->step_size * state->primal_weight;

    if (state->is_this_major_iteration || ((state->total_count + 2) % get_print_frequency(state->total_count + 2)) == 0)
    {
        compute_next_pdhg_dual_solution_major_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK>>>(
            state->current_dual_solution, state->pdhg_dual_solution, state->reflected_dual_solution,
            state->primal_product, state->constraint_lower_bound, state->constraint_upper_bound,
            state->num_constraints, step);
    }
    else
    {
        compute_next_pdhg_dual_solution_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK>>>(
            state->current_dual_solution, state->reflected_dual_solution, state->primal_product,
            state->constraint_lower_bound, state->constraint_upper_bound, state->num_constraints, step);
    }
}

static void halpern_update(pdhg_solver_state_t *state, double reflection_coefficient)
{
    double weight = (double)(state->inner_count + 1) / (state->inner_count + 2);
    halpern_update_kernel<<<state->num_blocks_primal_dual, THREADS_PER_BLOCK>>>(
        state->initial_primal_solution, state->current_primal_solution, state->reflected_primal_solution,
        state->initial_dual_solution, state->current_dual_solution, state->reflected_dual_solution,
        state->num_variables, state->num_constraints, weight, reflection_coefficient);
}

// static void perform_restart(pdhg_solver_state_t *state, const pdhg_parameters_t *params)
// {
//     compute_delta_solution_kernel<<<state->num_blocks_primal_dual, THREADS_PER_BLOCK>>>(
//         state->initial_primal_solution, state->pdhg_primal_solution, state->delta_primal_solution,
//         state->initial_dual_solution, state->pdhg_dual_solution, state->delta_dual_solution,
//         state->num_variables, state->num_constraints);

//     double primal_dist, dual_dist;
//     CUBLAS_CHECK(cublasDnrm2_v2_64(state->blas_handle, state->num_variables, state->delta_primal_solution, 1, &primal_dist));
//     CUBLAS_CHECK(cublasDnrm2_v2_64(state->blas_handle, state->num_constraints, state->delta_dual_solution, 1, &dual_dist));

//     double ratio_infeas = state->relative_dual_residual / state->relative_primal_residual;

//     if (primal_dist > 1e-16 && dual_dist > 1e-16 && primal_dist < 1e12 && dual_dist < 1e12 && ratio_infeas > 1e-8 && ratio_infeas < 1e8)
//     {
//         double error = log(dual_dist) - log(primal_dist) - log(state->primal_weight);
//         state->primal_weight_error_sum *= params->restart_params.i_smooth;
//         state->primal_weight_error_sum += error;
//         double delta_error = error - state->primal_weight_last_error;
//         state->primal_weight *= exp(params->restart_params.k_p * error +
//                                     params->restart_params.k_i * state->primal_weight_error_sum +
//                                     params->restart_params.k_d * delta_error);
//         state->primal_weight_last_error = error;
//     }
//     else
//     {
//         state->primal_weight = state->best_primal_weight;
//         state->primal_weight_error_sum = 0.0;
//         state->primal_weight_last_error = 0.0;
//     }

//     double primal_dual_residual_gap = abs(log10(state->relative_dual_residual / state->relative_primal_residual));
//     if (primal_dual_residual_gap < state->best_primal_dual_residual_gap)
//     {
//         state->best_primal_dual_residual_gap = primal_dual_residual_gap;
//         state->best_primal_weight = state->primal_weight;
//     }

//     CUDA_CHECK(cudaMemcpy(state->initial_primal_solution, state->pdhg_primal_solution, state->num_variables * sizeof(double), cudaMemcpyDeviceToDevice));
//     CUDA_CHECK(cudaMemcpy(state->current_primal_solution, state->pdhg_primal_solution, state->num_variables * sizeof(double), cudaMemcpyDeviceToDevice));
//     CUDA_CHECK(cudaMemcpy(state->initial_dual_solution, state->pdhg_dual_solution, state->num_constraints * sizeof(double), cudaMemcpyDeviceToDevice));
//     CUDA_CHECK(cudaMemcpy(state->current_dual_solution, state->pdhg_dual_solution, state->num_constraints * sizeof(double), cudaMemcpyDeviceToDevice));

//     state->inner_count = 0;
//     state->last_trial_fixed_point_error = INFINITY;
// }
static void perform_restart(pdhg_solver_state_t *state, const pdhg_parameters_t *params)
{
    compute_delta_solution_kernel<<<state->num_blocks_primal_dual, THREADS_PER_BLOCK>>>(
        state->initial_primal_solution, state->pdhg_primal_solution, state->delta_primal_solution,
        state->initial_dual_solution, state->pdhg_dual_solution, state->delta_dual_solution,
        state->num_variables, state->num_constraints);

    double primal_dist, dual_dist;
    // CUBLAS_CHECK(cublasDnrm2_v2_64(state->blas_handle, state->num_variables, state->delta_primal_solution, 1, &primal_dist));
    // CUBLAS_CHECK(cublasDnrm2_v2_64(state->blas_handle, state->num_constraints, state->delta_dual_solution, 1, &dual_dist));
    CUBLAS_CHECK(cublasDnrm2(state->blas_handle, state->num_variables, state->delta_primal_solution, 1, &primal_dist));
    CUBLAS_CHECK(cublasDnrm2(state->blas_handle, state->num_constraints, state->delta_dual_solution, 1, &dual_dist));

    double ratio_infeas = state->relative_dual_residual / state->relative_primal_residual;

    // --- PID DEBUG: 保存旧权重以便打印 ---
    const double old_primal_weight = state->primal_weight;
    // 确保在 C++ 中包含 stdio.h (solver.cu 顶部已有)
    // #include <stdio.h>

    if (primal_dist > 1e-16 && dual_dist > 1e-16 && primal_dist < 1e12 && dual_dist < 1e12 && ratio_infeas > 1e-8 && ratio_infeas < 1e8)
    {
        // if (params->verbose) {
        //     printf("\n[PID DEBUG] Iter %d: PID controller ACTIVE.\n", state->total_count);
        //     printf("  [PID INPUT] primal_dist (||x_new-x_old||) = %.6e\n", primal_dist);
        //     printf("  [PID INPUT] dual_dist   (||y_new-y_old||) = %.6e\n", dual_dist);
        //     printf("  [PID INPUT] old_primal_weight             = %.6e\n", old_primal_weight);
        //     printf("  [PID INPUT] old_primal_weight_last_error  = %.6e\n", state->primal_weight_last_error);
        //     printf("  [PID INPUT] old_primal_weight_error_sum   = %.6e\n", state->primal_weight_error_sum);
        // }
        // --- PID DEBUG: 打印 PID 输入 ---

        double error = log(dual_dist) - log(primal_dist) - log(state->primal_weight);

        // --- PID DEBUG: 打印 P, I, D 系数和中间值 ---
        const double Kp = params->restart_params.k_p;
        const double Ki = params->restart_params.k_i;
        const double Kd = params->restart_params.k_d;
        const double i_smooth = params->restart_params.i_smooth;

        state->primal_weight_error_sum *= i_smooth;
        state->primal_weight_error_sum += error;
        double delta_error = error - state->primal_weight_last_error;

        const double term_P = Kp * error;
        const double term_I = Ki * state->primal_weight_error_sum;
        const double term_D = Kd * delta_error;
        const double exp_term = term_P + term_I + term_D;

        // if (params->verbose) {
        //         printf("  [PID CALC]  Kp=%.2e, Ki=%.2e, Kd=%.2e, i_smooth=%.2e\n", Kp, Ki, Kd, i_smooth);
        //     printf("  [PID CALC]  error (P term base)           = %.6e\n", error);
        //     printf("  [PID CALC]  delta_error (D term base)     = %.6e\n", delta_error);
        //     printf("  [PID CALC]  new_error_sum (I term base)   = %.6e\n", state->primal_weight_error_sum);
        //     printf("  [PID CALC]  Term P (Kp * error)           = %.6e\n", term_P);
        //     printf("  [PID CALC]  Term I (Ki * sum)             = %.6e\n", term_I);
        //     printf("  [PID CALC]  Term D (Kd * delta)           = %.6e\n", term_D);
        //     printf("  [PID CALC]  exp_term (P+I+D)              = %.6e\n", exp_term);

        // }

        state->primal_weight *= exp(exp_term);
        state->primal_weight_last_error = error;

        // --- PID DEBUG: 打印最终结果 ---
        // if (params->verbose) {
        //     printf("  [PID OUTPUT] new_primal_weight            = %.6e\n", state->primal_weight);
        //     fflush(stdout); // 确保立即刷新缓冲区
        // }
    }
    else
    {
        // // --- PID DEBUG: 打印重置情况 ---
        // if (params->verbose) {
        //     printf("\n[PID DEBUG] Iter %d: PID controller SKIPPED (safety check fail).\n", state->total_count);
        //     printf("  [PID INFO]  primal_dist = %.6e, dual_dist = %.6e, ratio_infeas = %.6e\n",
        //             primal_dist, dual_dist, ratio_infeas);
        //     printf("  [PID RESET] Resetting primal_weight from %.6e to best_primal_weight %.6e\n",
        //             old_primal_weight, state->best_primal_weight);
        //     fflush(stdout); // 确保立即刷新缓冲区
        // }
        state->primal_weight = state->best_primal_weight;
        state->primal_weight_error_sum = 0.0;
        state->primal_weight_last_error = 0.0;
    }

    double primal_dual_residual_gap = abs(log10(state->relative_dual_residual / state->relative_primal_residual));
    if (primal_dual_residual_gap < state->best_primal_dual_residual_gap)
    {
        state->best_primal_dual_residual_gap = primal_dual_residual_gap;
        state->best_primal_weight = state->primal_weight;
    }

    CUDA_CHECK(cudaMemcpy(state->initial_primal_solution, state->pdhg_primal_solution, state->num_variables * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(state->current_primal_solution, state->pdhg_primal_solution, state->num_variables * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(state->initial_dual_solution, state->pdhg_dual_solution, state->num_constraints * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(state->current_dual_solution, state->pdhg_dual_solution, state->num_constraints * sizeof(double), cudaMemcpyDeviceToDevice));

    state->inner_count = 0;
    state->last_trial_fixed_point_error = INFINITY;
}

// === 计算 CSR 每行绝对值和的 kernel ===
__global__ void row_abs_sum_kernel(const int* __restrict__ row_ptr,
                                   const double* __restrict__ val,
                                   int m, double* __restrict__ out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) {
        double s = 0.0;
        int beg = row_ptr[i];
        int end = row_ptr[i + 1];
        #pragma unroll 4
        for (int p = beg; p < end; ++p) s += fabs(val[p]);
        out[i] = s;
    }
}

// === 用 Thrust 取“最大行绝对值和”（即 ||M||_inf）===
// 要求：M->row_ptr / M->val 是设备指针（本项目就是）
static double csr_max_row_abs_sum_thrust(const cu_sparse_matrix_csr_t* M, cudaStream_t stream) {
    const int m = M->num_rows;
    if (m <= 0) return 0.0;

    double* d_row_sums = nullptr;
    CUDA_CHECK(cudaMalloc(&d_row_sums, sizeof(double) * (size_t)m));

    dim3 block(256);
    dim3 grid((m + block.x - 1) / block.x);
    row_abs_sum_kernel<<<grid, block, 0, stream>>>(M->row_ptr, M->val, m, d_row_sums);
    CUDA_CHECK(cudaGetLastError());

    // 用与 cuSPARSE 同一条 stream 的执行策略
    thrust::device_ptr<double> beg(d_row_sums);
    thrust::device_ptr<double> end = beg + m;
    double mx = thrust::reduce(thrust::cuda::par.on(stream), beg, end, 0.0, thrust::maximum<double>());

    CUDA_CHECK(cudaFree(d_row_sums));
    return mx;
}

// === \|A\|_2 的上界：sqrt(\|A\|_1 * \|A\|_inf)
// 其中 \|A\|_1 = max col sum(A) = \|A^T\|_inf（AT 为 A 的转置，已是 CSR）
static double estimate_sigma_max_upper_1inf_thrust(const cu_sparse_matrix_csr_t* A,
                                                   const cu_sparse_matrix_csr_t* AT,
                                                   cudaStream_t stream) {
    double infA = csr_max_row_abs_sum_thrust(A,  stream); // \|A\|_inf
    double oneA = csr_max_row_abs_sum_thrust(AT, stream); // \|A\|_1 = \|A^T\|_inf

    if (!std::isfinite(infA) || infA < 0.0) infA = 0.0;
    if (!std::isfinite(oneA) || oneA < 0.0) oneA = 0.0;

    double prod = oneA * infA;
    if (!(std::isfinite(prod)) || prod <= 0.0) return 0.0;
    return sqrt(prod);
}
// ||A||_2 的一阶上界：sqrt(||A||_1 * ||A||_inf)
// 其中 ||A||_1 用 AT 的“最大行和”（= A 的最大列和）来等价计算
static void initialize_step_size_and_primal_weight(pdhg_solver_state_t *state,
                                                   const pdhg_parameters_t *params)
{
    const cu_sparse_matrix_csr_t* A  = state->constraint_matrix;
    const cu_sparse_matrix_csr_t* AT = state->constraint_matrix_t;

    // —— 与 cuSPARSE 同流，并让 cuBLAS 使用同一条流 —— //
    cudaStream_t stream = 0;
    CUSPARSE_CHECK(cusparseGetStream(state->sparse_handle, &stream));
    CUBLAS_CHECK(cublasSetStream(state->blas_handle, stream));

    auto valid_pos = [](double x)->bool { return std::isfinite(x) && (x > 0.0); };
    auto name_of = [](int m)->const char* {
        switch ((step_size_method_t)m) {
            case STEP_SIZE_POWER_ITERATION: return "power";
            case STEP_SIZE_ONE_INF_UPPER  : return "one_inf";
            case STEP_SIZE_CONSTANT       : return "constant";
            case STEP_SIZE_HYBRID         : return "hybrid";
            default: return "unknown";
        }
    };

    // —— 安全系数：兜底为 0.998（或按需改成 0.95） —— //
    double safety = params->step_size_safety;
    if (!std::isfinite(safety) || safety <= 1e-6 || safety >= 1.0) safety = 0.998;

    // —— 计时事件 —— //
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));

    float  t_oneinf_ms = 0.0f;
    float  t_power_ms  = 0.0f;
    float  t_ref_ms    = 0.0f;
    float t_constant_ms = 0.0f;

    // —— 结果容器 —— //
    double L_final = 0.0;
    double L_oneinf = 0.0;
    double L_power  = 0.0;
    // double L_constant = 0.0;

    step_size_method_t step_size_method = (step_size_method_t)params->step_size_method;
    if (state->matrix_value_mode != MATRIX_VALUES_EXPLICIT && step_size_method != STEP_SIZE_CONSTANT) {
        fprintf(stderr, "implicit matrix_value_mode currently uses constant step-size initialization; requested %s was ignored.\n", name_of((int)step_size_method));
        step_size_method = STEP_SIZE_CONSTANT;
    }

    // ===== 分支：计算 L =====
    switch (step_size_method) {
    case STEP_SIZE_POWER_ITERATION: {
        // 纯幂迭代
        const int    it  = (params->power_max_iterations > 0) ? params->power_max_iterations : 5000;
        const double tol = (params->power_tolerance > 0.0)     ? params->power_tolerance     : 1e-4;
        CUDA_CHECK(cudaEventRecord(e0, stream));
        L_power = estimate_maximum_singular_value(
                    state->sparse_handle, state->blas_handle, A, AT, it, tol);
        CUDA_CHECK(cudaEventRecord(e1, stream));
        CUDA_CHECK(cudaEventSynchronize(e1));
        CUDA_CHECK(cudaEventElapsedTime(&t_power_ms, e0, e1));
        L_final = valid_pos(L_power) ? L_power : 1.0;
        break;
    }

    case STEP_SIZE_ONE_INF_UPPER: {
        // 上界 sqrt(||A||_1 * ||A||_inf)
        CUDA_CHECK(cudaEventRecord(e0, stream));
        L_oneinf = estimate_sigma_max_upper_1inf_thrust(A, AT, stream);
        CUDA_CHECK(cudaEventRecord(e1, stream));
        CUDA_CHECK(cudaEventSynchronize(e1));
        CUDA_CHECK(cudaEventElapsedTime(&t_oneinf_ms, e0, e1));
        L_final = valid_pos(L_oneinf) ? L_oneinf : 1.0;
        break;
    }

    case STEP_SIZE_CONSTANT: {
        // L_constant = 1.0;
        // const int    it  = (params->power_max_iterations > 0) ? params->power_max_iterations : 5000;
        // const double tol = (params->power_tolerance > 0.0)     ? params->power_tolerance     : 1e-4;
        // CUDA_CHECK(cudaEventRecord(e0, stream));
        // L_power = estimate_maximum_singular_value(
        //             state->sparse_handle, state->blas_handle, A, AT, it, tol);
        // CUDA_CHECK(cudaEventRecord(e1, stream));
        // CUDA_CHECK(cudaEventSynchronize(e1));
        // CUDA_CHECK(cudaEventElapsedTime(&t_constant_ms, e0, e1));
        // L_final = min(L_constant, L_power);
        CUDA_CHECK(cudaEventRecord(e0, stream));
        L_final = 1.0;
        CUDA_CHECK(cudaEventRecord(e1, stream));
        CUDA_CHECK(cudaEventSynchronize(e1));
        CUDA_CHECK(cudaEventElapsedTime(&t_constant_ms, e0, e1));
        break;
    }
    case STEP_SIZE_HYBRID:
    default: {
        // 1) 先算 one-inf 上界（保证不低估）
        CUDA_CHECK(cudaEventRecord(e0, stream));
        L_oneinf = estimate_sigma_max_upper_1inf_thrust(A, AT, stream);
        CUDA_CHECK(cudaEventRecord(e1, stream));
        CUDA_CHECK(cudaEventSynchronize(e1));
        CUDA_CHECK(cudaEventElapsedTime(&t_oneinf_ms, e0, e1));
        if (!valid_pos(L_oneinf)) L_oneinf = 1.0;

        // 2) 少量幂迭代 refine
        const int    it_ref  = (params->hybrid_refine_iterations > 0) ? params->hybrid_refine_iterations : 10;
        const double tol_ref = (params->power_tolerance > 0.0)        ? params->power_tolerance        : 1e-3;

        if (it_ref > 0) {
            CUDA_CHECK(cudaEventRecord(e0, stream));
            L_power = estimate_maximum_singular_value(
                        state->sparse_handle, state->blas_handle, A, AT, it_ref, tol_ref);
            CUDA_CHECK(cudaEventRecord(e1, stream));
            CUDA_CHECK(cudaEventSynchronize(e1));
            CUDA_CHECK(cudaEventElapsedTime(&t_power_ms, e0, e1));
            if (!valid_pos(L_power)) L_power = L_oneinf;
        } else {
            L_power = L_oneinf;
        }

        // 3) “min-guarded”策略（原 hybrid 精神：尽量用幂迭代更准的值，但**带安全栅**）
        //    只有当 L_power 与上界足够接近时才允许把 L 降低，否则仍用上界。
        //    guard_rel 默认 2%，也可用 power_tolerance 作为替代来源（限制在 [1%, 10%]）
        double guard_rel = 0.02;
        if (params->power_tolerance > 0.0 && std::isfinite(params->power_tolerance)) {
            guard_rel = fmax(0.01, fmin(0.10, params->power_tolerance));
        }
        const double L_guard = (1.0 - guard_rel) * L_oneinf;

        if (L_power >= L_guard) {
            // 幂迭代结果“足够接近”上界，采用较小的 L（步长更大，但仍受 guard 约束）
            L_final = L_power;
        } else {
            // 幂迭代远低于上界，可能低估谱范数 → 回退到上界，保证完全稳定
            L_final = L_oneinf;
        }
        break;
    }
    }

    if (!valid_pos(L_final)) L_final = 1.0;
    state->step_size = safety / L_final;

    // ===== 可选：参考幂迭代（只打印，不参与步长） =====
    double L_ref = 0.0;
    bool have_ref = false;
    if (params->stepsize_power_reference && state->matrix_value_mode == MATRIX_VALUES_EXPLICIT) {
        const int    itR  = (params->stepsize_reference_max_iterations > 0)
                            ? params->stepsize_reference_max_iterations : 5000;
        const double tolR = (params->stepsize_reference_tolerance  > 0.0)
                            ? params->stepsize_reference_tolerance      : 1e-4;

        CUDA_CHECK(cudaEventRecord(e0, stream));
        L_ref = estimate_maximum_singular_value(
                    state->sparse_handle, state->blas_handle, A, AT, itR, tolR);
        CUDA_CHECK(cudaEventRecord(e1, stream));
        CUDA_CHECK(cudaEventSynchronize(e1));
        CUDA_CHECK(cudaEventElapsedTime(&t_ref_ms, e0, e1));

        have_ref = valid_pos(L_ref);
    }

    // ===== 打印 =====
    if (params->verbose) {
        const char* mname = name_of(params->step_size_method);
        if ((step_size_method_t)params->step_size_method == STEP_SIZE_ONE_INF_UPPER) {
            fprintf(stdout,
                "[StepSize] method=%s  safety=%.6e  L=%.6e  step=%.6e  (t_oneinf=%.3f ms)\n",
                mname, safety, L_final, state->step_size, t_oneinf_ms);
        } else if ((step_size_method_t)params->step_size_method == STEP_SIZE_POWER_ITERATION) {
            fprintf(stdout,
                "[StepSize] method=%s  safety=%.6e  L=%.6e  step=%.6e  (t_power=%.3f ms)\n",
                mname, safety, L_final, state->step_size, t_power_ms);
        } else if ((step_size_method_t)params->step_size_method == STEP_SIZE_CONSTANT) {
            fprintf(stdout,
                "[StepSize] method=%s  safety=%.6e  L=%.6e  step=%.6e  (t_constant=%.3f ms)\n",
                mname, safety, L_final, state->step_size, t_constant_ms);
        } else {
            // HYBRID：把两条都打出来，便于对比
            fprintf(stdout,
                "[StepSize] method=%s  safety=%.6e  L_oneinf=%.6e  L_power=%.6e  -> L=%.6e  step=%.6e  "
                "(t_oneinf=%.3f ms, t_power=%.3f ms)\n",
                mname, safety, L_oneinf, L_power, L_final, state->step_size, t_oneinf_ms, t_power_ms);
        }

        if (have_ref) {
            const double step_ref = safety / L_ref;
            const double rel_err_step = fabs(state->step_size - step_ref) / step_ref;
            const double rel_err_L    = fabs(L_final - L_ref) / L_ref;
            fprintf(stdout,
                "[StepSize] ref(power) L_ref=%.6e  step_ref=%.6e  rel_err_step=%.3e  rel_err_L=%.3e  (t_ref=%.3f ms)\n",
                L_ref, step_ref, rel_err_step, rel_err_L, t_ref_ms);
        }
        fflush(stdout);
    }

    // ===== 原有 primal_weight 逻辑 =====
    // CN: Python 可能已经在 degree-scaled LP 上追加了标量缩放；此时参数为 false 以避免 native 重复缩放，
    // CN: 但初始 primal weight 仍应与 native bound/objective rescaling 路径一致。
    // EN: Python may have appended scalar scaling to a degree-scaled LP; the parameter is false to avoid native rescaling,
    // EN: but the initial primal weight must still match the native bound/objective rescaling path.
    const bool has_bound_objective_rescaling =
        params->bound_objective_rescaling ||
        state->constraint_bound_rescaling != 1.0 ||
        state->objective_vector_rescaling != 1.0;
    if (has_bound_objective_rescaling) {
        state->primal_weight = 1.0;
    } else {
        state->primal_weight = state->objective_vector_norm / state->constraint_bound_norm;
    }
    state->best_primal_weight = state->primal_weight;

    // 清理事件
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
}


static void compute_fixed_point_error(pdhg_solver_state_t *state)
{
    compute_delta_solution_kernel<<<state->num_blocks_primal_dual, THREADS_PER_BLOCK>>>(
        state->current_primal_solution,
        state->reflected_primal_solution,
        state->delta_primal_solution,
        state->current_dual_solution,
        state->reflected_dual_solution,
        state->delta_dual_solution,
        state->num_variables,
        state->num_constraints);

    spmv_AT(state, state->delta_dual_solution, state->dual_product);

    double interaction, movement;

    double primal_norm = 0.0;
    double dual_norm = 0.0;
    double cross_term = 0.0;

    CUBLAS_CHECK(cublasDnrm2(state->blas_handle,
                                   state->num_constraints,
                                   state->delta_dual_solution,
                                   1,
                                   &dual_norm));
    CUBLAS_CHECK(cublasDnrm2(state->blas_handle,
                                   state->num_variables,
                                   state->delta_primal_solution,
                                   1,
                                   &primal_norm));
    movement = primal_norm * primal_norm * state->primal_weight + dual_norm * dual_norm / state->primal_weight;

    CUBLAS_CHECK(cublasDdot(state->blas_handle, state->num_variables, state->dual_product, 1, state->delta_primal_solution, 1, &cross_term));
    interaction = 2 * state->step_size * cross_term;

    state->fixed_point_error = sqrt(movement + interaction);
}

void pdhg_solver_state_free(pdhg_solver_state_t *state)
{
    if (state == NULL)
    {
        return;
    }

    if (!state->borrows_input_buffers && state->variable_lower_bound)
        CUDA_CHECK(cudaFree(state->variable_lower_bound));
    if (!state->borrows_input_buffers && state->variable_upper_bound)
        CUDA_CHECK(cudaFree(state->variable_upper_bound));
    if (!state->borrows_input_buffers && state->objective_vector)
        CUDA_CHECK(cudaFree(state->objective_vector));
    if (!state->borrows_input_buffers && state->constraint_matrix->row_ptr)
        CUDA_CHECK(cudaFree(state->constraint_matrix->row_ptr));
    if (!state->borrows_input_buffers && state->constraint_matrix->col_ind)
        CUDA_CHECK(cudaFree(state->constraint_matrix->col_ind));
    if (!state->borrows_input_buffers && state->constraint_matrix->val)
        CUDA_CHECK(cudaFree(state->constraint_matrix->val));
    if (state->constraint_matrix_t->row_ptr)
        CUDA_CHECK(cudaFree(state->constraint_matrix_t->row_ptr));
    if (state->constraint_matrix_t->col_ind)
        CUDA_CHECK(cudaFree(state->constraint_matrix_t->col_ind));
    if (state->constraint_matrix_t->val)
        CUDA_CHECK(cudaFree(state->constraint_matrix_t->val));
    if (state->constraint_lower_bound)
        CUDA_CHECK(cudaFree(state->constraint_lower_bound));
    if (state->constraint_upper_bound)
        CUDA_CHECK(cudaFree(state->constraint_upper_bound));
    if (state->constraint_lower_bound_finite_val)
        CUDA_CHECK(cudaFree(state->constraint_lower_bound_finite_val));
    if (state->constraint_upper_bound_finite_val)
        CUDA_CHECK(cudaFree(state->constraint_upper_bound_finite_val));
    if (state->variable_lower_bound_finite_val)
        CUDA_CHECK(cudaFree(state->variable_lower_bound_finite_val));
    if (state->variable_upper_bound_finite_val)
        CUDA_CHECK(cudaFree(state->variable_upper_bound_finite_val));
    if (state->implicit_spmv_heavy_rows)
        CUDA_CHECK(cudaFree(state->implicit_spmv_heavy_rows));
    if (state->initial_primal_solution)
        CUDA_CHECK(cudaFree(state->initial_primal_solution));
    if (state->current_primal_solution)
        CUDA_CHECK(cudaFree(state->current_primal_solution));
    if (state->pdhg_primal_solution)
        CUDA_CHECK(cudaFree(state->pdhg_primal_solution));
    if (state->reflected_primal_solution)
        CUDA_CHECK(cudaFree(state->reflected_primal_solution));
    if (state->dual_product)
        CUDA_CHECK(cudaFree(state->dual_product));
    if (state->initial_dual_solution)
        CUDA_CHECK(cudaFree(state->initial_dual_solution));
    if (state->current_dual_solution)
        CUDA_CHECK(cudaFree(state->current_dual_solution));
    if (state->pdhg_dual_solution)
        CUDA_CHECK(cudaFree(state->pdhg_dual_solution));
    if (state->reflected_dual_solution)
        CUDA_CHECK(cudaFree(state->reflected_dual_solution));
    if (state->primal_product)
        CUDA_CHECK(cudaFree(state->primal_product));
    if (!state->borrows_rescaling_buffers && state->constraint_rescaling)
        CUDA_CHECK(cudaFree(state->constraint_rescaling));
    if (!state->borrows_rescaling_buffers && state->variable_rescaling)
        CUDA_CHECK(cudaFree(state->variable_rescaling));
    if (state->primal_slack)
        CUDA_CHECK(cudaFree(state->primal_slack));
    if (state->dual_slack)
        CUDA_CHECK(cudaFree(state->dual_slack));
    if (state->primal_residual)
        CUDA_CHECK(cudaFree(state->primal_residual));
    if (state->dual_residual)
        CUDA_CHECK(cudaFree(state->dual_residual));
    if (state->delta_primal_solution)
        CUDA_CHECK(cudaFree(state->delta_primal_solution));
    if (state->delta_dual_solution)
        CUDA_CHECK(cudaFree(state->delta_dual_solution));
    if (state->ones_primal_d)
        CUDA_CHECK(cudaFree(state->ones_primal_d));
    if (state->ones_dual_d)
        CUDA_CHECK(cudaFree(state->ones_dual_d));
    if (state->trace_iters_host)
        free(state->trace_iters_host);
    if (state->trace_primal_objectives_host)
        free(state->trace_primal_objectives_host);
    if (state->trace_dual_objectives_host)
        free(state->trace_dual_objectives_host);
    if (state->trace_primal_solutions_host)
        free(state->trace_primal_solutions_host);
    if (state->trace_dual_solutions_host)
        free(state->trace_dual_solutions_host);
    if (state->trace_variable_rescaling_host)
        free(state->trace_variable_rescaling_host);
    if (state->trace_constraint_rescaling_host)
        free(state->trace_constraint_rescaling_host);

    if (state->primal_spmv_buffer)
        CUDA_CHECK(cudaFree(state->primal_spmv_buffer));
    if (state->dual_spmv_buffer)
        CUDA_CHECK(cudaFree(state->dual_spmv_buffer));

    // 2) 销毁 cuSPARSE 稠密向量/稀疏矩阵句柄（先 vec，后 mat）
    if (state->vec_primal_sol)  CUSPARSE_CHECK(cusparseDestroyDnVec(state->vec_primal_sol));
    if (state->vec_dual_sol)    CUSPARSE_CHECK(cusparseDestroyDnVec(state->vec_dual_sol));
    if (state->vec_primal_prod) CUSPARSE_CHECK(cusparseDestroyDnVec(state->vec_primal_prod));
    if (state->vec_dual_prod)   CUSPARSE_CHECK(cusparseDestroyDnVec(state->vec_dual_prod));

    if (state->matA)   CUSPARSE_CHECK(cusparseDestroySpMat(state->matA));
    if (state->matAt)  CUSPARSE_CHECK(cusparseDestroySpMat(state->matAt));

    // 3) 销毁 cuSPARSE/cuBLAS handle（最后销毁 handle）
    if (state->sparse_handle) CUSPARSE_CHECK(cusparseDestroy(state->sparse_handle));
    if (state->blas_handle)   CUBLAS_CHECK(cublasDestroy(state->blas_handle));

    // 4) 释放 host 侧壳体结构体
    if (state->constraint_matrix)   free(state->constraint_matrix);
    if (state->constraint_matrix_t) free(state->constraint_matrix_t);



    free(state);
}

void rescale_info_free(rescale_info_t *info)
{
    if (info == NULL)
    {
        return;
    }

    lp_problem_free(info->scaled_problem);
    free(info->con_rescale);
    free(info->var_rescale);

    free(info);
}
