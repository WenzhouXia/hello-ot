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

#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <random>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
// =================== SMART PATCH START ===================
// 仅在 CUDA 版本低于 11.7 (11070) 时启用此补丁
// 如果是新版本 CUDA，这段代码会被编译器自动忽略，零副作用。
#include <cuda_runtime_api.h>

#if defined(CUDART_VERSION) && CUDART_VERSION < 11070

#include <cublas_v2.h>
#ifndef CUBLAS_GET_STATUS_NAME_H_
#define CUBLAS_GET_STATUS_NAME_H_
// 手动补充旧版本缺失的函数
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

#endif
// =================== SMART PATCH END ===================
std::mt19937 gen(1);
std::normal_distribution<double> dist(0.0, 1.0);

const double HOST_ONE = 1.0;
const double HOST_ZERO = 0.0;

void *safe_malloc(int size)
{
    void *ptr = malloc(size);
    if (ptr == NULL)
    {
        perror("Fatal error: malloc failed");
        exit(EXIT_FAILURE);
    }
    return ptr;
}

void *safe_calloc(int num, int size)
{
    void *ptr = calloc(num, size);
    if (ptr == NULL)
    {
        perror("Fatal error: calloc failed");
        exit(EXIT_FAILURE);
    }
    return ptr;
}


__global__ static void csr_implicit_unit_spmv_kernel(
    const int *row_ptr,
    const int *col_ind,
    const double *row_scale,
    const double *col_scale,
    const double *x,
    double *out,
    int n_rows)
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
        sum += x[col] / (rs * cs);
    }
    out[row] = sum;
}

__global__ static void validate_coldeg2_transpose_kernel(
    const int *row_ptr_t,
    int n_cols,
    int *bad)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n_cols) {
        return;
    }
    if (row_ptr_t[col + 1] - row_ptr_t[col] != 2) {
        atomicExch(bad, 1);
    }
}

__global__ static void implicit_ax_unique_counts_kernel(
    const int *row_ptr_t,
    const int *row_ind,
    int *unique0,
    int *unique1,
    int full_warps)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_idx = col >> 5;
    if (warp_idx >= full_warps) {
        return;
    }
    const unsigned mask = 0xffffffffu;
    const int lane = threadIdx.x & 31;
    const int p = row_ptr_t[col];
    const int row0 = row_ind[p];
    const int row1 = row_ind[p + 1];
    bool first0 = true;
    bool first1 = true;
    for (int i = 0; i < 32; ++i) {
        const int other0 = __shfl_sync(mask, row0, i);
        const int other1 = __shfl_sync(mask, row1, i);
        if (i < lane) {
            first0 = first0 && (other0 != row0);
            first1 = first1 && (other1 != row1);
        }
    }
    const unsigned first0_mask = __ballot_sync(mask, first0);
    const unsigned first1_mask = __ballot_sync(mask, first1);
    if (lane == 0) {
        unique0[warp_idx] = __popc(first0_mask);
        unique1[warp_idx] = __popc(first1_mask);
    }
}

__device__ static void implicit_ax_atomic_add_maybe_agg(
    double *out,
    int row,
    double value,
    bool aggregate)
{
    if (!aggregate) {
        atomicAdd(out + row, value);
        return;
    }
    const unsigned mask = 0xffffffffu;
    const unsigned group = __match_any_sync(mask, row);
    double sum = 0.0;
    for (int src_lane = 0; src_lane < 32; ++src_lane) {
        if (group & (1u << src_lane)) {
            sum += __shfl_sync(group, value, src_lane);
        }
    }
    const int leader = __ffs(group) - 1;
    if ((threadIdx.x & 31) == leader) {
        atomicAdd(out + row, sum);
    }
}

__global__ static void csr_implicit_coldeg2_ax_kernel(
    const int *row_ptr_t,
    const int *row_ind,
    const double *row_scale,
    const double *col_scale,
    const double *x,
    double *out,
    int start_col,
    int n_cols)
{
    int col = start_col + blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n_cols) {
        return;
    }
    const int p = row_ptr_t[col];
    const int row0 = row_ind[p];
    const int row1 = row_ind[p + 1];
    const double cs = (col_scale != nullptr && col_scale[col] != 0.0) ? col_scale[col] : 1.0;
    const double x_scaled = x[col] / cs;
    const double rs0 = (row_scale != nullptr && row_scale[row0] != 0.0) ? row_scale[row0] : 1.0;
    const double rs1 = (row_scale != nullptr && row_scale[row1] != 0.0) ? row_scale[row1] : 1.0;
    atomicAdd(out + row0, x_scaled / rs0);
    atomicAdd(out + row1, x_scaled / rs1);
}

__global__ static void csr_implicit_coldeg2_ax_agg_kernel(
    const int *row_ptr_t,
    const int *row_ind,
    const double *row_scale,
    const double *col_scale,
    const double *x,
    double *out,
    int full_cols,
    int agg_row0,
    int agg_row1)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= full_cols) {
        return;
    }
    const int p = row_ptr_t[col];
    const int row0 = row_ind[p];
    const int row1 = row_ind[p + 1];
    const double cs = (col_scale != nullptr && col_scale[col] != 0.0) ? col_scale[col] : 1.0;
    const double x_scaled = x[col] / cs;
    const double rs0 = (row_scale != nullptr && row_scale[row0] != 0.0) ? row_scale[row0] : 1.0;
    const double rs1 = (row_scale != nullptr && row_scale[row1] != 0.0) ? row_scale[row1] : 1.0;
    implicit_ax_atomic_add_maybe_agg(out, row0, x_scaled / rs0, agg_row0 != 0);
    implicit_ax_atomic_add_maybe_agg(out, row1, x_scaled / rs1, agg_row1 != 0);
}

static void ensure_implicit_ax_auto_agg(pdhg_solver_state_t *state)
{
    if (state->implicit_ax_agg_initialized) {
        return;
    }
    state->implicit_ax_agg_initialized = true;
    state->implicit_ax_agg_mode = IMPLICIT_AX_AGG_NONE;
    state->implicit_ax_row0_unique_p50 = 32.0;
    state->implicit_ax_row1_unique_p50 = 32.0;

    int *bad_d = nullptr;
    CUDA_CHECK(cudaMalloc(&bad_d, sizeof(int)));
    CUDA_CHECK(cudaMemset(bad_d, 0, sizeof(int)));
    validate_coldeg2_transpose_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
        state->constraint_matrix_t->row_ptr,
        state->num_variables,
        bad_d);
    CUDA_CHECK(cudaGetLastError());
    int bad_h = 0;
    CUDA_CHECK(cudaMemcpy(&bad_h, bad_d, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(bad_d));
    if (bad_h != 0) {
        fprintf(stderr, "implicit_ax and implicit_both require GPU device-CSR with col_deg == 2.\n");
        exit(EXIT_FAILURE);
    }

    const int full_warps = state->num_variables / 32;
    if (full_warps <= 0) {
        return;
    }
    int *unique0 = nullptr;
    int *unique1 = nullptr;
    CUDA_CHECK(cudaMalloc(&unique0, (size_t)full_warps * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&unique1, (size_t)full_warps * sizeof(int)));
    const int full_cols = full_warps * 32;
    const int blocks = (full_cols + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    implicit_ax_unique_counts_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        state->constraint_matrix_t->row_ptr,
        state->constraint_matrix_t->col_ind,
        unique0,
        unique1,
        full_warps);
    CUDA_CHECK(cudaGetLastError());
    thrust::sort(thrust::device, unique0, unique0 + full_warps);
    thrust::sort(thrust::device, unique1, unique1 + full_warps);
    int p50_0 = 32;
    int p50_1 = 32;
    const int mid = full_warps / 2;
    CUDA_CHECK(cudaMemcpy(&p50_0, unique0 + mid, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&p50_1, unique1 + mid, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(unique0));
    CUDA_CHECK(cudaFree(unique1));
    state->implicit_ax_row0_unique_p50 = (double)p50_0;
    state->implicit_ax_row1_unique_p50 = (double)p50_1;
    const bool agg0 = p50_0 <= 16;
    const bool agg1 = p50_1 <= 16;
    if (agg0 && agg1) {
        state->implicit_ax_agg_mode = IMPLICIT_AX_AGG_BOTH;
    } else if (agg0) {
        state->implicit_ax_agg_mode = IMPLICIT_AX_AGG_ROW0;
    } else if (agg1) {
        state->implicit_ax_agg_mode = IMPLICIT_AX_AGG_ROW1;
    } else {
        state->implicit_ax_agg_mode = IMPLICIT_AX_AGG_NONE;
    }
}

void spmv_A(pdhg_solver_state_t *state, const double *x, double *out)
{
    if (matrix_value_mode_has_implicit_ax(state->matrix_value_mode)) {
        ensure_implicit_ax_auto_agg(state);
        CUDA_CHECK(cudaMemset(out, 0, (size_t)state->num_constraints * sizeof(double)));
        const int full_cols = (state->num_variables / 32) * 32;
        const bool agg0 = state->implicit_ax_agg_mode == IMPLICIT_AX_AGG_ROW0 ||
            state->implicit_ax_agg_mode == IMPLICIT_AX_AGG_BOTH;
        const bool agg1 = state->implicit_ax_agg_mode == IMPLICIT_AX_AGG_ROW1 ||
            state->implicit_ax_agg_mode == IMPLICIT_AX_AGG_BOTH;
        if ((agg0 || agg1) && full_cols > 0) {
            const int blocks_full = (full_cols + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            csr_implicit_coldeg2_ax_agg_kernel<<<blocks_full, THREADS_PER_BLOCK>>>(
                state->constraint_matrix_t->row_ptr,
                state->constraint_matrix_t->col_ind,
                state->constraint_rescaling,
                state->variable_rescaling,
                x,
                out,
                full_cols,
                agg0 ? 1 : 0,
                agg1 ? 1 : 0);
            CUDA_CHECK(cudaGetLastError());
            if (full_cols < state->num_variables) {
                const int tail = state->num_variables - full_cols;
                const int blocks_tail = (tail + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
                csr_implicit_coldeg2_ax_kernel<<<blocks_tail, THREADS_PER_BLOCK>>>(
                    state->constraint_matrix_t->row_ptr,
                    state->constraint_matrix_t->col_ind,
                    state->constraint_rescaling,
                    state->variable_rescaling,
                    x,
                    out,
                    full_cols,
                    state->num_variables);
                CUDA_CHECK(cudaGetLastError());
            }
        } else {
            csr_implicit_coldeg2_ax_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
                state->constraint_matrix_t->row_ptr,
                state->constraint_matrix_t->col_ind,
                state->constraint_rescaling,
                state->variable_rescaling,
                x,
                out,
                0,
                state->num_variables);
            CUDA_CHECK(cudaGetLastError());
        }
        return;
    }
    CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_primal_sol, (void *)x));
    CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_primal_prod, out));
    CUSPARSE_CHECK(cusparseSpMV(
        state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &HOST_ONE, state->matA, state->vec_primal_sol, &HOST_ZERO, state->vec_primal_prod,
        CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, state->primal_spmv_buffer));
}

void spmv_AT(pdhg_solver_state_t *state, const double *y, double *out)
{
    if (matrix_value_mode_has_implicit_aty(state->matrix_value_mode)) {
        csr_implicit_unit_spmv_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
            state->constraint_matrix_t->row_ptr,
            state->constraint_matrix_t->col_ind,
            state->variable_rescaling,
            state->constraint_rescaling,
            y,
            out,
            state->num_variables);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_dual_sol, (void *)y));
    CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_dual_prod, out));
    CUSPARSE_CHECK(cusparseSpMV(
        state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &HOST_ONE, state->matAt, state->vec_dual_sol, &HOST_ZERO, state->vec_dual_prod,
        CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, state->dual_spmv_buffer));
}

double estimate_maximum_singular_value(
    cusparseHandle_t sparse_handle,
    cublasHandle_t blas_handle,
    const cu_sparse_matrix_csr_t *A,
    const cu_sparse_matrix_csr_t *AT,
    int max_iterations,
    double tolerance)
{
    int m = A->num_rows;
    int n = A->num_cols;
    double *eigenvector_d, *next_eigenvector_d, *dual_product_d;

    CUDA_CHECK(cudaMalloc(&eigenvector_d, m * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&next_eigenvector_d, m * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dual_product_d, n * sizeof(double)));

    double *eigenvector_h = (double *)safe_malloc(m * sizeof(double));
    for (int i = 0; i < m; ++i)
    {
        eigenvector_h[i] = dist(gen);
    }

    CUDA_CHECK(cudaMemcpy(eigenvector_d, eigenvector_h, m * sizeof(double), cudaMemcpyHostToDevice));
    free(eigenvector_h);

    double sigma_max_sq = 1.0;
    const double one = 1.0;
    const double zero = 0.0;

    cusparseSpMatDescr_t matA, matAT;
    CUSPARSE_CHECK(cusparseCreateCsr(&matA, A->num_rows, A->num_cols, A->num_nonzeros, A->row_ptr, A->col_ind, A->val, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateCsr(&matAT, AT->num_rows, AT->num_cols, AT->num_nonzeros, AT->row_ptr, AT->col_ind, AT->val, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

    cusparseDnVecDescr_t vecEigen, vecNextEigen, vecDual;
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecEigen, m, eigenvector_d, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecNextEigen, m, next_eigenvector_d, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecDual, n, dual_product_d, CUDA_R_64F));

    void *dBufferAT = NULL;
    void *dBufferA = NULL;
    size_t bufferSizeAT = 0, bufferSizeA = 0;
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matAT, vecNextEigen, &zero, vecDual, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, &bufferSizeAT));
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matA, vecDual, &zero, vecEigen, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, &bufferSizeA));

    CUDA_CHECK(cudaMalloc(&dBufferAT, bufferSizeAT));
    CUDA_CHECK(cudaMalloc(&dBufferA, bufferSizeA));

    for (int i = 0; i < max_iterations; ++i)
    {

        CUDA_CHECK(cudaMemcpy(next_eigenvector_d, eigenvector_d, m * sizeof(double), cudaMemcpyDeviceToDevice));
        double eigenvector_norm;
        CUBLAS_CHECK(cublasDnrm2(blas_handle, m, next_eigenvector_d, 1, &eigenvector_norm));

        double inv_eigenvector_norm = 1.0 / eigenvector_norm;
        CUBLAS_CHECK(cublasDscal(blas_handle, m, &inv_eigenvector_norm, next_eigenvector_d, 1));

        CUSPARSE_CHECK(cusparseSpMV(sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matAT, vecNextEigen, &zero, vecDual, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, dBufferAT));

        CUSPARSE_CHECK(cusparseSpMV(sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matA, vecDual, &zero, vecEigen, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, dBufferA));

        CUBLAS_CHECK(cublasDdot(blas_handle, m, next_eigenvector_d, 1, eigenvector_d, 1, &sigma_max_sq));

        double neg_sigma_sq = -sigma_max_sq;
        CUBLAS_CHECK(cublasDscal(blas_handle, m, &neg_sigma_sq, next_eigenvector_d, 1));
        CUBLAS_CHECK(cublasDaxpy(blas_handle, m, &one, eigenvector_d, 1, next_eigenvector_d, 1));

        double residual_norm;
        CUBLAS_CHECK(cublasDnrm2(blas_handle, m, next_eigenvector_d, 1, &residual_norm));

        if (residual_norm < tolerance)
            break;
    }

    CUDA_CHECK(cudaFree(dBufferAT));
    CUDA_CHECK(cudaFree(dBufferA));
    CUSPARSE_CHECK(cusparseDestroySpMat(matA));
    CUSPARSE_CHECK(cusparseDestroySpMat(matAT));
    CUSPARSE_CHECK(cusparseDestroyDnVec(vecEigen));
    CUSPARSE_CHECK(cusparseDestroyDnVec(vecNextEigen));
    CUSPARSE_CHECK(cusparseDestroyDnVec(vecDual));
    CUDA_CHECK(cudaFree(eigenvector_d));
    CUDA_CHECK(cudaFree(next_eigenvector_d));
    CUDA_CHECK(cudaFree(dual_product_d));

    return sqrt(sigma_max_sq);
}

void compute_interaction_and_movement(pdhg_solver_state_t *state, double *interaction, double *movement)
{
    double dual_norm, primal_norm, cross_term;

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
    *movement = 0.5 * (primal_norm * primal_norm * state->primal_weight + dual_norm * dual_norm / state->primal_weight);

    CUBLAS_CHECK(cublasDdot(state->blas_handle, state->num_variables, state->dual_product, 1, state->delta_primal_solution, 1, &cross_term));
    *interaction = fabs(cross_term);
}

const char *termination_reason_to_string(termination_reason_t reason)
{
    switch (reason)
    {
    case TERMINATION_REASON_OPTIMAL:
        return "OPTIMAL";
    case TERMINATION_REASON_PRIMAL_INFEASIBLE:
        return "PRIMAL_INFEASIBLE";
    case TERMINATION_REASON_DUAL_INFEASIBLE:
        return "DUAL_INFEASIBLE";
    case TERMINATION_REASON_TIME_LIMIT:
        return "TIME_LIMIT";
    case TERMINATION_REASON_ITERATION_LIMIT:
        return "ITERATION_LIMIT";
    case TERMINATION_REASON_SUPPORT_LIMIT_REACHED:
        return "SUPPORT_LIMIT_REACHED";
    case TERMINATION_REASON_OPTIMAL_WITH_SUPPORT_LIMIT:
        return "OPTIMAL_WITH_SUPPORT_LIMIT";
    case TERMINATION_REASON_NUMERICAL_DIVERGENCE:
        return "NUMERICAL_DIVERGENCE";
    case TERMINATION_REASON_UNSPECIFIED:
    default:
        return "UNSPECIFIED";
    }
}

bool optimality_criteria_met(const pdhg_solver_state_t *state, double rel_opt_tol, double rel_feas_tol)
{
    return state->termination_relative_dual_residual < rel_feas_tol &&
           state->termination_relative_primal_residual < rel_feas_tol &&
           state->relative_objective_gap < rel_opt_tol;
}

bool optimality_criteria_met_separate(const pdhg_solver_state_t *state, double rel_opt_tol, double rel_feas_tol_primal, double rel_feas_tol_dual)
{
    return state->termination_relative_dual_residual < rel_feas_tol_dual &&
           state->termination_relative_primal_residual < rel_feas_tol_primal &&
           state->relative_objective_gap < rel_opt_tol;
}

bool primal_infeasibility_criteria_met(const pdhg_solver_state_t *state, double eps)
{
    if (state->dual_ray_objective <= 0.0)
    {
        return false;
    }
    return state->max_dual_ray_infeasibility / state->dual_ray_objective <= eps;
}

bool dual_infeasibility_criteria_met(const pdhg_solver_state_t *state, double eps)
{
    if (state->primal_ray_linear_objective >= 0.0)
    {
        return false;
    }
    return state->max_primal_ray_infeasibility / (-state->primal_ray_linear_objective) <= eps;
}

void check_termination_criteria(
    pdhg_solver_state_t *solver_state,
    const termination_criteria_t *criteria)
{
    // if (optimality_criteria_met(solver_state, criteria->eps_optimal_relative, criteria->eps_feasible_relative))
    if (optimality_criteria_met_separate(solver_state, criteria->eps_optimal_relative, criteria->eps_feasible_relative_primal, criteria->eps_feasible_relative_dual))
    {
        // solver_state->termination_reason = TERMINATION_REASON_OPTIMAL;
        // return;
        // === 新增：dual nnz 约束 ===
        if (criteria->use_dual_nnz_gate) {
            const int nnz_thr = (int)(criteria->dual_nnz_factor * solver_state->num_variables);
            const double tol_abs = criteria->dual_nnz_abs_tol;
            const int nnz_dual = count_dual_nnz_cuda(solver_state, tol_abs); // 你已实现

            if (nnz_dual < nnz_thr) {
                solver_state->termination_reason = TERMINATION_REASON_OPTIMAL;
                printf(
                    "Dual NNZ = %d (%.3f x nVars) < Threshold = %.3f x nVars (= %d), Tolerance = %.2e.\n",
                    nnz_dual,
                    (double)nnz_dual / (double)solver_state->num_variables,
                    criteria->dual_nnz_factor,
                    nnz_thr,
                    tol_abs
                );
                return;
            }
            else{
                printf(
                    "Dual NNZ = %d (%.3f x nVars) > Threshold = %.3f x nVars (= %d), Tolerance = %.2e.\n",
                    nnz_dual,
                    (double)nnz_dual / (double)solver_state->num_variables,
                    criteria->dual_nnz_factor,
                    nnz_thr,
                    tol_abs);
            }
            // 否则不设置 OPTIMAL，继续往下判定
        } else {
            solver_state->termination_reason = TERMINATION_REASON_OPTIMAL;
            return;
        }
    }
    if (primal_infeasibility_criteria_met(solver_state, criteria->eps_infeasible))
    {
        solver_state->termination_reason = TERMINATION_REASON_PRIMAL_INFEASIBLE;
        return;
    }
    if (dual_infeasibility_criteria_met(solver_state, criteria->eps_infeasible))
    {
        solver_state->termination_reason = TERMINATION_REASON_DUAL_INFEASIBLE;
        return;
    }
    if (solver_state->total_count >= criteria->iteration_limit)
    {
        solver_state->termination_reason = TERMINATION_REASON_ITERATION_LIMIT;
        return;
    }
    if (solver_state->cumulative_time_sec >= criteria->time_sec_limit)
    {
        solver_state->termination_reason = TERMINATION_REASON_TIME_LIMIT;
        return;
    }
}


bool should_do_adaptive_restart(
    pdhg_solver_state_t *solver_state,
    const restart_parameters_t *restart_params,
    int termination_evaluation_frequency,
    bool verbose)
{
    // 检查是否是第一次重启 (这通常是一个预定的重启)
    if (solver_state->total_count == termination_evaluation_frequency)
    {
        if (verbose){
            printf("\n[RESTART] Iter %d: Triggered by initial major iteration.\n", solver_state->total_count);
            fflush(stdout);
        }
        solver_state->last_trial_fixed_point_error = solver_state->fixed_point_error;
        return true;
    }
    else if (solver_state->total_count > termination_evaluation_frequency)
    {
        // 准则 (iii): 强制重启 (我们最怀疑这个)
        // 我们把它放在最前面检查
        if (solver_state->inner_count >= restart_params->artificial_restart_threshold * solver_state->total_count)
        {
            if (verbose){
                printf("\n[RESTART] Iter %d: Triggered by ARTIFICIAL RESTART (inner_count=%d >= %.1f * total_count=%d)\n",
                   solver_state->total_count, solver_state->inner_count,
                   restart_params->artificial_restart_threshold, solver_state->total_count);
                fflush(stdout);
            }
            solver_state->last_trial_fixed_point_error = solver_state->fixed_point_error;
            return true;
        }

        // 准则 (i): 充分下降
        if (solver_state->fixed_point_error <= restart_params->sufficient_reduction_for_restart * solver_state->initial_fixed_point_error)
        {
            if (verbose){
                printf("\n[RESTART] Iter %d: Triggered by SUFFICIENT DECAY (error=%.2e <= %.1f * initial_error=%.2e)\n",
                   solver_state->total_count, solver_state->fixed_point_error,
                   restart_params->sufficient_reduction_for_restart, solver_state->initial_fixed_point_error);
                fflush(stdout);
            }
            solver_state->last_trial_fixed_point_error = solver_state->fixed_point_error;
            return true;
        }

        // 准则 (ii): 必要下降 + 停止前进
        if (solver_state->fixed_point_error <= restart_params->necessary_reduction_for_restart * solver_state->initial_fixed_point_error)
        {
            if (solver_state->fixed_point_error > solver_state->last_trial_fixed_point_error)
            {
                if (verbose){
                    printf("\n[RESTART] Iter %d: Triggered by NO LOCAL PROGRESS (error=%.2e > last_error=%.2e)\n",
                       solver_state->total_count, solver_state->fixed_point_error,
                       solver_state->last_trial_fixed_point_error);
                    fflush(stdout);
                }
                solver_state->last_trial_fixed_point_error = solver_state->fixed_point_error;
                return true;
            }
        }
    }

    // --- 没有触发重启 ---
    solver_state->last_trial_fixed_point_error = solver_state->fixed_point_error;
    return false;
}
void print_initial_info(const pdhg_parameters_t *params, const lp_problem_t *problem)
{
    if (!params->verbose)
    {
        return;
    }
    printf("---------------------------------------------------------------------------------------\n");
    printf("                                    cuPDLPx v0.1.0                                     \n");
    printf("                        A GPU-Accelerated First-Order LP Solver                        \n");
    printf("               (c) Haihao Lu, Massachusetts Institute of Technology, 2025              \n");
    printf("---------------------------------------------------------------------------------------\n");

    printf("problem:\n");
    printf("  variables     : %d\n", problem->num_variables);
    printf("  constraints   : %d\n", problem->num_constraints);
    printf("  nnz(A)        : %d\n", problem->constraint_matrix_num_nonzeros);

    printf("settings:\n");
    printf("  iter_limit         : %d\n", params->termination_criteria.iteration_limit);
    printf("  time_limit         : %.2f sec\n", params->termination_criteria.time_sec_limit);
    printf("  eps_opt            : %.1e\n", params->termination_criteria.eps_optimal_relative);
    // printf("  eps_feas           : %.1e\n", params->termination_criteria.eps_feasible_relative);
    printf("  eps_feas_primal    : %.1e\n", params->termination_criteria.eps_feasible_relative_primal);
    printf("  eps_feas_dual      : %.1e\n", params->termination_criteria.eps_feasible_relative_dual);
    printf("  eps_infeas_detect  : %.1e\n", params->termination_criteria.eps_infeasible);
    printf("  termination_norm   : %s\n", params->termination_norm == TERMINATION_NORM_L_INF ? "linf" : "l2");

    printf("---------------------------------------------------------------------------------------\n");
    printf("%s | %s | %s | %s \n",
           "   runtime    ", "    objective     ", "  absolute residuals   ", "  relative residuals   ");
    printf("%s %s | %s %s | %s %s %s | %s %s %s \n",
           "  iter", "  time ", " pr obj ", "  du obj ", " pr res", " du res", "  gap  ", " pr res", " du res", "  gap  ");
    printf("---------------------------------------------------------------------------------------\n");
}

void pdhg_final_log(const pdhg_solver_state_t *state, bool verbose, termination_reason_t reason)
{
    if (!verbose)
    {
        return;
    }
    // printf("Solution Summary\n");
    // printf("  Status        : %s\n", termination_reason_to_string(reason));
    // printf("  Iterations    : %d\n", state->total_count - 1);
    // printf("  Solve time    : %.3g sec\n", state->cumulative_time_sec);
    // printf("  primal infeas : %.3e / %.3e\n", state->absolute_primal_residual, state->relative_primal_residual);
    // printf("  dual infeas   : %.3e / %.3e\n", state->absolute_dual_residual, state->relative_dual_residual);
    // printf("  primal obj    : %.10g\n", state->primal_objective_value);
    // printf("  dual obj      : %.10g\n", state->dual_objective_value);
    printf("Summary [%s]: Iterations: %d, Time: %.3f sec\n",
        termination_reason_to_string(reason),
        state->total_count - 1,
        state->cumulative_time_sec);
    printf("RelRes (P/D): %.3e / %.3e | Obj (P/D): %.10g / %.10g | Gap: %.4e\n",
            state->relative_primal_residual,
            state->relative_dual_residual,
            state->primal_objective_value,
            state->dual_objective_value,
            fabs(state->primal_objective_value - state->dual_objective_value) /
            (1.0 + fabs(state->primal_objective_value) + fabs(state->dual_objective_value)));
    printf("TerminationRelRes[%s] (P/D): %.3e / %.3e\n",
            state->termination_norm == TERMINATION_NORM_L_INF ? "linf" : "l2",
            state->termination_relative_primal_residual,
            state->termination_relative_dual_residual);
}

void display_iteration_stats(const pdhg_solver_state_t *state, bool verbose)
{
    if (!verbose)
    {
        return;
    }
    if (state->total_count % get_print_frequency(state->total_count) == 0)
    {
        printf("%6d %.1e | %8.1e  %8.1e | %.1e %.1e %.1e | %.1e %.1e %.1e \n",
               state->total_count,
               state->cumulative_time_sec,
               state->primal_objective_value,
               state->dual_objective_value,
               state->absolute_primal_residual,
               state->absolute_dual_residual,
               state->objective_gap,
               state->relative_primal_residual,
               state->relative_dual_residual,
               state->relative_objective_gap);
    }
}

int get_print_frequency(int iter)
{
    int step = 10;
    long long threshold = 1000;

    while (iter >= threshold)
    {
        step *= 10;
        threshold *= 10;
    }
    return step;
}

__global__ void compute_residual_kernel(
    double *primal_residual,
    const double *primal_product,
    const double *constraint_lower_bound,
    const double *constraint_upper_bound,
    const double *dual_solution,
    double *dual_residual,
    const double *dual_product,
    const double *dual_slack,
    const double *objective_vector,
    const double *constraint_rescaling,
    const double *variable_rescaling,
    double *dual_obj_contribution,
    const double *const_lb_finite,
    const double *const_ub_finite,
    int num_constraints,
    int num_variables)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num_constraints)
    {

        double clamped_val = fmax(constraint_lower_bound[i], fmin(primal_product[i], constraint_upper_bound[i]));
        primal_residual[i] = (primal_product[i] - clamped_val) * constraint_rescaling[i];

        dual_obj_contribution[i] = fmax(dual_solution[i], 0.0) * const_lb_finite[i] + fmin(dual_solution[i], 0.0) * const_ub_finite[i];
    }
    else if (i < num_constraints + num_variables)
    {
        int idx = i - num_constraints;
        dual_residual[idx] = (objective_vector[idx] - dual_product[idx] - dual_slack[idx]) * variable_rescaling[idx];
    }
}

__global__ void primal_infeasibility_project_kernel(
    double *primal_ray_estimate,
    const double *variable_lower_bound,
    const double *variable_upper_bound,
    int variable_bound_mode,
    double variable_lower_bound_constant,
    double variable_upper_bound_constant,
    int num_variables)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_variables)
    {
        const double lb = (variable_bound_mode == VARIABLE_BOUNDS_CONSTANT) ? variable_lower_bound_constant : variable_lower_bound[i];
        const double ub = (variable_bound_mode == VARIABLE_BOUNDS_CONSTANT) ? variable_upper_bound_constant : variable_upper_bound[i];
        if (isfinite(lb))
        {
            primal_ray_estimate[i] = fmax(primal_ray_estimate[i], 0.0);
        }
        if (isfinite(ub))
        {
            primal_ray_estimate[i] = fmin(primal_ray_estimate[i], 0.0);
        }
    }
}

__global__ void dual_infeasibility_project_kernel(
    double *dual_ray_estimate,
    const double *constraint_lower_bound,
    const double *constraint_upper_bound,
    int num_constraints)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_constraints)
    {
        if (!isfinite(constraint_lower_bound[i]))
        {
            dual_ray_estimate[i] = fmin(dual_ray_estimate[i], 0.0);
        }
        if (!isfinite(constraint_upper_bound[i]))
        {
            dual_ray_estimate[i] = fmax(dual_ray_estimate[i], 0.0);
        }
    }
}

__global__ void compute_primal_infeasibility_kernel(
    const double *primal_product,
    const double *const_lb,
    const double *const_ub,
    int num_constraints,
    double *primal_infeasibility,
    const double *constraint_rescaling)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_constraints)
    {
        double pp_val = primal_product[i];
        primal_infeasibility[i] = (fmax(0.0, -pp_val) * isfinite(const_lb[i]) + fmax(0.0, pp_val) * isfinite(const_ub[i])) * constraint_rescaling[i];
    }
}

__global__ void compute_dual_infeasibility_kernel(
    const double *dual_product,
    const double *var_lb,
    const double *var_ub,
    int variable_bound_mode,
    double var_lb_constant,
    double var_ub_constant,
    int num_variables,
    double *dual_infeasibility,
    const double *variable_rescaling)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_variables)
    {
        const double lb = (variable_bound_mode == VARIABLE_BOUNDS_CONSTANT) ? var_lb_constant : var_lb[i];
        const double ub = (variable_bound_mode == VARIABLE_BOUNDS_CONSTANT) ? var_ub_constant : var_ub[i];
        double dp_val = -dual_product[i];
        dual_infeasibility[i] = (fmax(0.0, dp_val) * !isfinite(lb) - fmin(0.0, dp_val) * !isfinite(ub)) * variable_rescaling[i];
    }
}

__global__ void dual_solution_dual_objective_contribution_kernel(
    const double *constraint_lower_bound_finite_val,
    const double *constraint_upper_bound_finite_val,
    const double *dual_solution,
    int num_constraints,
    double *dual_objective_dual_solution_contribution_array)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num_constraints)
    {
        dual_objective_dual_solution_contribution_array[i] =
            fmax(dual_solution[i], 0.0) * constraint_lower_bound_finite_val[i] +
            fmin(dual_solution[i], 0.0) * constraint_upper_bound_finite_val[i];
    }
}

__global__ void dual_objective_dual_slack_contribution_array_kernel(
    const double *dual_slack,
    double *dual_objective_dual_slack_contribution_array,
    const double *variable_lower_bound_finite_val,
    const double *variable_upper_bound_finite_val,
    int variable_bound_mode,
    double variable_lower_bound_constant,
    double variable_upper_bound_constant,
    int num_variables)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num_variables)
    {
        const double lb_finite = (variable_bound_mode == VARIABLE_BOUNDS_CONSTANT)
            ? (isfinite(variable_lower_bound_constant) ? variable_lower_bound_constant : 0.0)
            : variable_lower_bound_finite_val[i];
        const double ub_finite = (variable_bound_mode == VARIABLE_BOUNDS_CONSTANT)
            ? (isfinite(variable_upper_bound_constant) ? variable_upper_bound_constant : 0.0)
            : variable_upper_bound_finite_val[i];
        dual_objective_dual_slack_contribution_array[i] =
            fmax(-dual_slack[i], 0.0) * lb_finite +
            fmin(-dual_slack[i], 0.0) * ub_finite;
    }
}

static double get_vector_inf_norm(cublasHandle_t handle, int n, const double *x_d)
{
    if (n <= 0)
        return 0.0;
    int index;

    CUBLAS_CHECK(cublasIdamax(handle, n, x_d, 1, &index));
    double max_val;

    CUDA_CHECK(cudaMemcpy(&max_val, x_d + (index - 1), sizeof(double), cudaMemcpyDeviceToHost));
    return fabs(max_val);
}

static double get_vector_sum(pdhg_solver_state_t *state, int n, double *ones_d, const double *x_d)
{
    if (n <= 0)
        return 0.0;
    if (state->vector_sum_mode == VECTOR_SUM_DIRECT_REDUCE)
        return thrust::reduce(thrust::device, x_d, x_d + n, 0.0, thrust::plus<double>());

    double sum;
    CUBLAS_CHECK(cublasDdot(state->blas_handle, n, x_d, 1, ones_d, 1, &sum));
    return sum;
}

void compute_residual(pdhg_solver_state_t *state)
{
    spmv_A(state, state->pdhg_primal_solution, state->primal_product);
    // --- 在每个异步调用后立刻加这些（SpMV 之后一次，kernel 之后一次） ---
    CUDA_CHECK(cudaGetLastError());        // 捕获前一个 kernel/异步错误
    CUDA_CHECK(cudaDeviceSynchronize());   // 强制完成，避免拿到未写入/旧数据

    spmv_AT(state, state->pdhg_dual_solution, state->dual_product);
    // --- 在每个异步调用后立刻加这些（SpMV 之后一次，kernel 之后一次） ---
    CUDA_CHECK(cudaGetLastError());        // 捕获前一个 kernel/异步错误
    CUDA_CHECK(cudaDeviceSynchronize());   // 强制完成，避免拿到未写入/旧数据

    compute_residual_kernel<<<state->num_blocks_primal_dual, THREADS_PER_BLOCK>>>(
        state->primal_residual, state->primal_product, state->constraint_lower_bound,
        state->constraint_upper_bound, state->pdhg_dual_solution, state->dual_residual,
        state->dual_product, state->dual_slack, state->objective_vector,
        state->constraint_rescaling, state->variable_rescaling, state->primal_slack,
        state->constraint_lower_bound_finite_val, state->constraint_upper_bound_finite_val,
        state->num_constraints, state->num_variables);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUBLAS_CHECK(cublasDnrm2(state->blas_handle, state->num_constraints, state->primal_residual, 1, &state->absolute_primal_residual));
    state->absolute_primal_residual /= state->constraint_bound_rescaling;
    CUBLAS_CHECK(cublasDnrm2(state->blas_handle, state->num_variables, state->dual_residual, 1, &state->absolute_dual_residual));
    state->absolute_dual_residual /= state->objective_vector_rescaling;

    CUBLAS_CHECK(cublasDdot(state->blas_handle, state->num_variables, state->objective_vector, 1, state->pdhg_primal_solution, 1, &state->primal_objective_value));
    state->primal_objective_value = state->primal_objective_value / (state->constraint_bound_rescaling * state->objective_vector_rescaling) + state->objective_constant;

    double base_dual_objective;
    CUBLAS_CHECK(cublasDdot(state->blas_handle, state->num_variables, state->dual_slack, 1, state->pdhg_primal_solution, 1, &base_dual_objective));
    double dual_slack_sum = get_vector_sum(state, state->num_constraints, state->ones_dual_d, state->primal_slack);
    state->dual_objective_value = (base_dual_objective + dual_slack_sum) / (state->constraint_bound_rescaling * state->objective_vector_rescaling) + state->objective_constant;

    state->relative_primal_residual = state->absolute_primal_residual / (1.0 + state->constraint_bound_norm);
    state->relative_dual_residual = state->absolute_dual_residual / (1.0 + state->objective_vector_norm);
    if (state->termination_norm == TERMINATION_NORM_L_INF)
    {
        // CN: termination-only Linf 不覆盖 legacy L2 residual；restart、PID 与 support gate 继续读取 L2 字段。
        // EN: Termination-only Linf leaves legacy L2 residuals intact so restart, PID, and support gates remain L2-based.
        state->termination_absolute_primal_residual =
            get_vector_inf_norm(state->blas_handle, state->num_constraints, state->primal_residual) /
            state->constraint_bound_rescaling;
        state->termination_absolute_dual_residual =
            get_vector_inf_norm(state->blas_handle, state->num_variables, state->dual_residual) /
            state->objective_vector_rescaling;
        state->termination_relative_primal_residual =
            state->termination_absolute_primal_residual / (1.0 + state->termination_constraint_bound_norm);
        state->termination_relative_dual_residual =
            state->termination_absolute_dual_residual / (1.0 + state->termination_objective_vector_norm);
    }
    else
    {
        state->termination_absolute_primal_residual = state->absolute_primal_residual;
        state->termination_relative_primal_residual = state->relative_primal_residual;
        state->termination_absolute_dual_residual = state->absolute_dual_residual;
        state->termination_relative_dual_residual = state->relative_dual_residual;
    }
    state->relative_objective_gap = fabs(state->primal_objective_value - state->dual_objective_value) /
                                    (1.0 + fabs(state->primal_objective_value) + fabs(state->dual_objective_value));
}

void compute_infeasibility_information(pdhg_solver_state_t *state)
{
    primal_infeasibility_project_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
        state->delta_primal_solution,
        state->variable_lower_bound,
        state->variable_upper_bound,
        (int)state->variable_bound_mode,
        state->variable_lower_bound_constant,
        state->variable_upper_bound_constant,
        state->num_variables);
    dual_infeasibility_project_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK>>>(state->delta_dual_solution, state->constraint_lower_bound, state->constraint_upper_bound, state->num_constraints);

    double primal_ray_inf_norm = get_vector_inf_norm(state->blas_handle, state->num_variables, state->delta_primal_solution);
    if (primal_ray_inf_norm > 0.0)
    {
        double scale = 1.0 / primal_ray_inf_norm;
        cublasDscal(state->blas_handle, state->num_variables, &scale, state->delta_primal_solution, 1);
    }
    double dual_ray_inf_norm = get_vector_inf_norm(state->blas_handle, state->num_constraints, state->delta_dual_solution);

    spmv_A(state, state->delta_primal_solution, state->primal_product);
    spmv_AT(state, state->delta_dual_solution, state->dual_product);

    CUBLAS_CHECK(cublasDdot(state->blas_handle, state->num_variables, state->objective_vector, 1, state->delta_primal_solution, 1, &state->primal_ray_linear_objective));
    state->primal_ray_linear_objective /= (state->constraint_bound_rescaling * state->objective_vector_rescaling);

    dual_solution_dual_objective_contribution_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK>>>(
        state->constraint_lower_bound_finite_val,
        state->constraint_upper_bound_finite_val,
        state->delta_dual_solution,
        state->num_constraints,
        state->primal_slack);

    dual_objective_dual_slack_contribution_array_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
        state->dual_product,
        state->dual_slack,
        state->variable_lower_bound_finite_val,
        state->variable_upper_bound_finite_val,
        (int)state->variable_bound_mode,
        state->variable_lower_bound_constant,
        state->variable_upper_bound_constant,
        state->num_variables);

    double sum_primal_slack = get_vector_sum(state, state->num_constraints, state->ones_dual_d, state->primal_slack);
    double sum_dual_slack = get_vector_sum(state, state->num_variables, state->ones_primal_d, state->dual_slack);
    state->dual_ray_objective = (sum_primal_slack + sum_dual_slack) / (state->constraint_bound_rescaling * state->objective_vector_rescaling);

    compute_primal_infeasibility_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK>>>(state->primal_product, state->constraint_lower_bound, state->constraint_upper_bound, state->num_constraints, state->primal_slack, state->constraint_rescaling);
    compute_dual_infeasibility_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
        state->dual_product,
        state->variable_lower_bound,
        state->variable_upper_bound,
        (int)state->variable_bound_mode,
        state->variable_lower_bound_constant,
        state->variable_upper_bound_constant,
        state->num_variables,
        state->dual_slack,
        state->variable_rescaling);

    state->max_primal_ray_infeasibility = get_vector_inf_norm(state->blas_handle, state->num_constraints, state->primal_slack);
    double dual_slack_norm = get_vector_inf_norm(state->blas_handle, state->num_variables, state->dual_slack);
    state->max_dual_ray_infeasibility = dual_slack_norm;

    double scaling_factor = fmax(dual_ray_inf_norm, dual_slack_norm);
    if (scaling_factor > 0.0)
    {
        state->max_dual_ray_infeasibility /= scaling_factor;
        state->dual_ray_objective /= scaling_factor;
    }
    else
    {
        state->max_dual_ray_infeasibility = 0.0;
        state->dual_ray_objective = 0.0;
    }
}

// ===== nnz counter: dual solution =====
#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 256
#endif

template <typename T>
__global__ void count_nnz_block_reduce_kernel(const T* __restrict__ x,
                                              int n,
                                              int* __restrict__ out_count,
                                              T tol_abs) {
    extern __shared__ int s_count[];
    int tid   = threadIdx.x;
    int gtid  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    int local = 0;
    for (int i = gtid; i < n; i += stride) {
        T v = x[i];
        // 依据浮点数判断的稳健性，这里用 |v| > tol_abs 作为非零判据
        if (v > tol_abs || v < -tol_abs) local += 1;
    }
    s_count[tid] = local;
    __syncthreads();

    // 块内归约
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) s_count[tid] += s_count[tid + offset];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out_count, s_count[0]);
}

// 计数 dual 解的 nnz：对接 state 里的设备指针与维度
static inline int count_dual_nnz_cuda(const pdhg_solver_state_t* state, double tol_abs) {
    const int n = state->num_constraints;            // dual 维度
    const double* d_y = state->pdhg_dual_solution;   // dual 向量（设备端）

    int *d_counter = nullptr;
    CUDA_CHECK(cudaMalloc(&d_counter, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_counter, 0, sizeof(int)));

    const int tpb   = THREADS_PER_BLOCK;
    const int blocks = (n + tpb - 1) / tpb;
    const size_t shmem = tpb * sizeof(int);

    count_nnz_block_reduce_kernel<double><<<blocks, tpb, shmem>>>(d_y, n, d_counter, (double)tol_abs);
    CUDA_CHECK(cudaGetLastError());

    int h_counter = 0;
    CUDA_CHECK(cudaMemcpy(&h_counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_counter));
    return h_counter;
}
