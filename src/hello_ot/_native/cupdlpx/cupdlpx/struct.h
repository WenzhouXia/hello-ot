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

#pragma once

#include <stdlib.h>
#include <stdbool.h>
#include <cusparse.h>
#include <cublas_v2.h>

typedef struct
{
	int num_rows;
	int num_cols;
	int num_nonzeros;
	int *row_ptr;
	int *col_ind;
	double *val;
} cu_sparse_matrix_csr_t;

typedef enum
{
	VARIABLE_BOUNDS_EXPLICIT = 0,
	VARIABLE_BOUNDS_CONSTANT = 1
} variable_bound_mode_t;

typedef enum
{
	MATRIX_VALUES_EXPLICIT = 0,
	MATRIX_VALUES_IMPLICIT_ATY = 1,
	MATRIX_VALUES_IMPLICIT_AX = 2,
	MATRIX_VALUES_IMPLICIT_BOTH = 3
} matrix_value_mode_t;

typedef enum
{
	IMPLICIT_AX_AGG_NONE = 0,
	IMPLICIT_AX_AGG_ROW0 = 1,
	IMPLICIT_AX_AGG_ROW1 = 2,
	IMPLICIT_AX_AGG_BOTH = 3
} implicit_ax_agg_mode_t;

static inline bool matrix_value_mode_has_implicit_ax(matrix_value_mode_t mode)
{
	return mode == MATRIX_VALUES_IMPLICIT_AX || mode == MATRIX_VALUES_IMPLICIT_BOTH;
}

static inline bool matrix_value_mode_has_implicit_aty(matrix_value_mode_t mode)
{
	return mode == MATRIX_VALUES_IMPLICIT_ATY || mode == MATRIX_VALUES_IMPLICIT_BOTH;
}

static inline bool matrix_value_mode_has_explicit_a(matrix_value_mode_t mode)
{
	return !matrix_value_mode_has_implicit_ax(mode);
}

static inline bool matrix_value_mode_has_explicit_at(matrix_value_mode_t mode)
{
	return !matrix_value_mode_has_implicit_aty(mode);
}

typedef enum
{
	VECTOR_SUM_RESIDENT_ONES = 0,
	VECTOR_SUM_DIRECT_REDUCE = 1
} vector_sum_mode_t;

typedef struct
{
	int num_variables;
	int num_constraints;
	double *variable_lower_bound;
	double *variable_upper_bound;
	variable_bound_mode_t variable_bound_mode;
	double variable_lower_bound_constant;
	double variable_upper_bound_constant;
	double *objective_vector;
	double objective_constant;

	int *constraint_matrix_row_pointers;
	int *constraint_matrix_col_indices;
	double *constraint_matrix_values;
	matrix_value_mode_t matrix_value_mode;
	int constraint_matrix_num_nonzeros;

	double *constraint_lower_bound;
	double *constraint_upper_bound;
} lp_problem_t;

typedef struct
{
	int num_variables;
	int num_constraints;
	int constraint_matrix_num_nonzeros;
	const int *constraint_matrix_row_pointers;
	const int *constraint_matrix_col_indices;
	const double *constraint_matrix_values;
	matrix_value_mode_t matrix_value_mode;
	const double *variable_lower_bound;
	const double *variable_upper_bound;
	variable_bound_mode_t variable_bound_mode;
	double variable_lower_bound_constant;
	double variable_upper_bound_constant;
	const double *objective_vector;
	const double *right_hand_side;
	int num_equalities;
	double objective_constant;
	const double *constraint_rescaling;
	const double *variable_rescaling;
	double constraint_bound_rescaling;
	double objective_vector_rescaling;
	double original_objective_vector_norm;
	double original_constraint_bound_norm;
	double original_objective_vector_linf_norm;
	double original_constraint_bound_linf_norm;
	bool has_precomputed_rescaling;
} device_lp_problem_t;

typedef enum
{
	TERMINATION_NORM_L2 = 0,
	TERMINATION_NORM_L_INF = 1
} termination_norm_t;

typedef enum
{
	TERMINATION_REASON_UNSPECIFIED,
	TERMINATION_REASON_OPTIMAL,
	TERMINATION_REASON_PRIMAL_INFEASIBLE,
	TERMINATION_REASON_DUAL_INFEASIBLE,
	TERMINATION_REASON_TIME_LIMIT,
	TERMINATION_REASON_ITERATION_LIMIT,
	TERMINATION_REASON_SUPPORT_LIMIT_REACHED,
	TERMINATION_REASON_OPTIMAL_WITH_SUPPORT_LIMIT,
	TERMINATION_REASON_NUMERICAL_DIVERGENCE
} termination_reason_t;

typedef enum
{
	SUPPORT_LIMIT_MODE_OPTIMAL_ONLY,
	SUPPORT_LIMIT_MODE_SUPPORT_ONLY,
	SUPPORT_LIMIT_MODE_BOTH,
	SUPPORT_LIMIT_MODE_EITHER
} support_limit_mode_t;

typedef struct
{
	lp_problem_t *scaled_problem;
	double *con_rescale;
	double *var_rescale;
	double con_bound_rescale;
	double obj_vec_rescale;
	double rescaling_time_sec;
} rescale_info_t;

typedef struct
{
	double artificial_restart_threshold;
	double sufficient_reduction_for_restart;
	double necessary_reduction_for_restart;
	double k_p;
	double k_i;
	double k_d;
	double i_smooth;
} restart_parameters_t;

typedef struct
{
	double eps_optimal_relative;
	double eps_feasible_relative;
	double eps_infeasible;
	double time_sec_limit;
	int iteration_limit;
	double eps_feasible_relative_primal;
	double eps_feasible_relative_dual;
	int use_dual_nnz_gate;	 // 0=关闭(默认)，1=开启
	double dual_nnz_factor;	 // 阈值系数，默认 2.0 (阈值 = factor * num_variables)
	double dual_nnz_abs_tol; // 非零判定的 |y_i| > tol，默认 0.0
} termination_criteria_t;

typedef enum
{
	STEP_SIZE_POWER_ITERATION = 0, // 幂迭代
	STEP_SIZE_ONE_INF_UPPER = 1,   // sqrt(||A||_1 * ||A||_inf)
	STEP_SIZE_HYBRID = 2,		   // 上界 + 少量幂迭代 refine
	STEP_SIZE_CONSTANT = 3,
} step_size_method_t;

typedef struct
{
	const double *anchor_primal_unscaled;
	const double *anchor_dual_unscaled;
	const double *current_primal_unscaled;
	const double *current_dual_unscaled;
	bool has_anchor_primal;
	bool has_anchor_dual;
	bool has_current_primal;
	bool has_current_dual;
	bool enabled;
	bool reuse_step_size;
	bool apply_restart_on_entry;
	double step_size;
	double primal_weight;
	double primal_weight_error_sum;
	double primal_weight_last_error;
	double best_primal_weight;
	double best_primal_dual_residual_gap;
	double previous_restart_dual_residual;
	double previous_restart_gap;
	int total_count;
	int inner_count;
} continuation_parameters_t;

typedef struct
{
	int l_inf_ruiz_iterations;
	bool has_pock_chambolle_alpha;
	double pock_chambolle_alpha;
	bool bound_objective_rescaling;
	bool verbose;
	bool verbose_time;
	int termination_evaluation_frequency;
	termination_norm_t termination_norm;
	termination_criteria_t termination_criteria;
	restart_parameters_t restart_params;
	double reflection_coefficient;
	const double *initial_primal_unscaled;
	const double *initial_dual_unscaled;
	bool has_initial_iterate;
	continuation_parameters_t continuation;
	bool has_continuation;

	///
	int step_size_method;		  // 0/1/2，见上 enum
	double step_size_safety;	  // 安全系数，默认 0.95（或保留你原来的 0.998）
	int power_max_iterations;	  // 幂迭代最大步数（power/hybrid）
	double power_tolerance;		  // 幂迭代相对/残差容忍
	int hybrid_refine_iterations; // hybrid 时的幂迭代步数（更小，例如 5~10）
	int stepsize_power_reference; // bool
	int stepsize_reference_max_iterations;
	double stepsize_reference_tolerance;
	bool trace_enabled;
	int trace_max_snapshots;
	int support_stop_nnz;
	double support_stop_threshold;
	double support_stop_obj_rel_change_tol;
	support_limit_mode_t support_limit_mode;
	bool enable_objective_gap_divergence_check;
	vector_sum_mode_t vector_sum_mode;
	///
} pdhg_parameters_t;

typedef struct
{
	int num_variables;
	int num_constraints;
	double *variable_lower_bound;
	double *variable_upper_bound;
	variable_bound_mode_t variable_bound_mode;
	double variable_lower_bound_constant;
	double variable_upper_bound_constant;
	double *objective_vector;
	double objective_constant;
	cu_sparse_matrix_csr_t *constraint_matrix;
	cu_sparse_matrix_csr_t *constraint_matrix_t;
	matrix_value_mode_t matrix_value_mode;
	double *constraint_lower_bound;
	double *constraint_upper_bound;
	int num_blocks_primal;
	int num_blocks_dual;
	int num_blocks_primal_dual;
	double objective_vector_norm;
	double constraint_bound_norm;
	double termination_objective_vector_norm;
	double termination_constraint_bound_norm;
	termination_norm_t termination_norm;
	double *constraint_lower_bound_finite_val;
	double *constraint_upper_bound_finite_val;
	double *variable_lower_bound_finite_val;
	double *variable_upper_bound_finite_val;
	int *implicit_spmv_heavy_rows;
	int num_implicit_spmv_heavy_rows;
	int implicit_spmv_heavy_threshold;
	implicit_ax_agg_mode_t implicit_ax_agg_mode;
	bool implicit_ax_agg_initialized;
	double implicit_ax_row0_unique_p50;
	double implicit_ax_row1_unique_p50;

	double *initial_primal_solution;
	double *current_primal_solution;
	double *pdhg_primal_solution;
	double *reflected_primal_solution;
	double *dual_product;
	double *initial_dual_solution;
	double *current_dual_solution;
	double *pdhg_dual_solution;
	double *reflected_dual_solution;
	double *primal_product;
	double step_size;
	double primal_weight;
	int total_count;
	bool is_this_major_iteration;
	double primal_weight_error_sum;
	double primal_weight_last_error;
	double best_primal_weight;
	double best_primal_dual_residual_gap;

	double *constraint_rescaling;
	double *variable_rescaling;
	double constraint_bound_rescaling;
	double objective_vector_rescaling;
	double *primal_slack;
	double *dual_slack;
	double rescaling_time_sec;
	double gpu_to_cpu_time_sec;
	double basic_time_sec;
	double cumulative_time_sec;

	double *primal_residual;
	double absolute_primal_residual;
	double relative_primal_residual;
	double termination_absolute_primal_residual;
	double termination_relative_primal_residual;
	double *dual_residual;
	double absolute_dual_residual;
	double relative_dual_residual;
	double termination_absolute_dual_residual;
	double termination_relative_dual_residual;
	double primal_objective_value;
	double dual_objective_value;
	double objective_gap;
	double relative_objective_gap;
	double best_relative_objective_gap;
	bool check_gap_divergence_after_restart;
	double max_primal_ray_infeasibility;
	double max_dual_ray_infeasibility;
	double primal_ray_linear_objective;
	double dual_ray_objective;
	termination_reason_t termination_reason;

	double *delta_primal_solution;
	double *delta_dual_solution;
	double fixed_point_error;
	double initial_fixed_point_error;
	double last_trial_fixed_point_error;
	int inner_count;

	cusparseHandle_t sparse_handle;
	cublasHandle_t blas_handle;
	size_t spmv_buffer_size;
	size_t primal_spmv_buffer_size;
	size_t dual_spmv_buffer_size;
	void *primal_spmv_buffer;
	void *dual_spmv_buffer;
	void *spmv_buffer;

	cusparseSpMatDescr_t matA;
	cusparseSpMatDescr_t matAt;
	cusparseDnVecDescr_t vec_primal_sol;
	cusparseDnVecDescr_t vec_dual_sol;
	cusparseDnVecDescr_t vec_primal_prod;
	cusparseDnVecDescr_t vec_dual_prod;

	double *ones_primal_d;
	double *ones_dual_d;
	vector_sum_mode_t vector_sum_mode;

	// 新增，为了避免PID导致的爆炸
	double k_p;
	double k_i;
	double k_d;
	double previous_restart_dual_residual;
	double previous_restart_gap;

	bool trace_enabled;
	int trace_max_snapshots;
	int trace_num_snapshots;
	int *trace_iters_host;
	double *trace_primal_objectives_host;
	double *trace_dual_objectives_host;
	double *trace_primal_solutions_host;
	double *trace_dual_solutions_host;
	double *trace_variable_rescaling_host;
	double *trace_constraint_rescaling_host;
	bool support_stop_has_last_eval_obj;
	double support_stop_last_eval_obj;
	double support_stop_obj_rel_change;
	int support_stop_last_eval_nnz;
	bool borrows_input_buffers;
	bool borrows_rescaling_buffers;
} pdhg_solver_state_t;
