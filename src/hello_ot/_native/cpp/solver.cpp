#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <omp.h>

namespace py = pybind11;
using Vec = std::vector<double>;
using Clock = std::chrono::steady_clock;

// CN: 保留 Torch 的 L2 范数和未缩放 KKT 定义。
// EN: Preserve the Torch L2 norms and unscaled KKT definitions.
double norm(const Vec& x) {
    double sum = 0.0;
    for (double v : x) sum += v * v;
    return std::sqrt(sum);
}

struct Residual {
    double pobj = 0, dobj = 0, ap = 0, rp = 0, ad = 0, rd = 0, gap = 0;
    bool finite() const {
        return std::isfinite(pobj) && std::isfinite(dobj) && std::isfinite(ap)
            && std::isfinite(rp) && std::isfinite(ad) && std::isfinite(rd) && std::isfinite(gap);
    }
};
struct Checkpoint {
    int64_t iteration;
    Residual residual;
    double weight;
};

py::dict solve_once(
    py::array_t<int64_t, py::array::c_style> row_array,
    py::array_t<int64_t, py::array::c_style> col_array,
    py::array_t<double, py::array::c_style> cost_array,
    py::array_t<double, py::array::c_style> mass_array,
    py::array_t<double, py::array::c_style> primal_array,
    py::array_t<double, py::array::c_style> dual_array,
    int64_t n, int threads, int frequency, double time_limit, int iteration_limit,
    double tol_objective, double tol_primal, double tol_dual, bool rescale, bool trace_enabled) {
    const auto started = Clock::now();
    const int64_t e = cost_array.size(), nodes = mass_array.size(), m = nodes - n;
    if (n < 1 || m < 1 || e < 1 || threads < 1 || frequency < 1 || time_limit <= 0 || iteration_limit < 1)
        throw py::value_error("invalid restricted OT dimensions or runtime limits");
    if (row_array.size() != e || col_array.size() != e || primal_array.size() != e || dual_array.size() != nodes)
        throw py::value_error("restricted OT array dimensions do not match");
    const int64_t* rows = row_array.data();
    const int64_t* cols = col_array.data();
    const double* c = cost_array.data();
    const double* b = mass_array.data();
    const double* initial_x = primal_array.data();
    const double* initial_y = dual_array.data();
    py::array_t<double> x_out(e), y_out(nodes);
    double* output_x = x_out.mutable_data();
    double* output_y = y_out.mutable_data();
    Residual residual;
    std::vector<Checkpoint> trace;
    std::string reason = "UNSPECIFIED";
    int64_t count = 0, inner = 0, restarts = 0;
    int actual_threads = 1;
    double weight = 1, cost_norm = 0, mass_norm = 0;
    double setup_seconds = 0;
    double iteration_seconds = 0;
    double check_seconds = 0, restart_seconds = 0, fixed_seconds = 0;
    {
        py::gil_scoped_release release;
        Vec masses(b, b + nodes), costs(c, c + e), scale(nodes, 0.0);
        std::vector<int64_t> offsets(nodes + 1, 0), edge_ids(2 * e);
        for (int64_t j = 0; j < e; ++j) {
            if (rows[j] < 0 || rows[j] >= n || cols[j] < 0 || cols[j] >= m)
                throw std::invalid_argument("active-support endpoint index is out of bounds");
            ++offsets[rows[j] + 1];
            ++offsets[n + cols[j] + 1];
        }
        for (int64_t i = 0; i < nodes; ++i) {
            scale[i] = std::sqrt(double(std::max<int64_t>(offsets[i + 1], 1)));
        }
        std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());
        auto cursor = offsets;
        for (int64_t j = 0; j < e; ++j) {
            edge_ids[cursor[rows[j]]++] = j;
            edge_ids[cursor[n + cols[j]]++] = j;
        }
        mass_norm = norm(masses);
        cost_norm = norm(costs);
        const double bound = rescale ? 1.0 / std::max(1.0, mass_norm) : 1.0;
        const double objective = rescale ? 1.0 / std::max(1.0, cost_norm) : 1.0;
        const double variable = std::sqrt(2.0);
        Vec cs(e), bs(nodes), node_scale(nodes), anchor_x(e), anchor_y(nodes);
        Vec x(e), y(nodes), px(e), py_dual(nodes), rx(e), dx(e), dy(nodes), target_dual(m);
        for (int64_t j = 0; j < e; ++j) {
            cs[j] = c[j] * objective / variable;
            anchor_x[j] = std::max(initial_x[j], 0.0) * variable * bound;
        }
        for (int64_t i = 0; i < nodes; ++i) {
            bs[i] = b[i] * bound / scale[i];
            node_scale[i] = variable * scale[i];
            anchor_y[i] = initial_y[i] * scale[i] * objective;
        }
        x = px = anchor_x;
        y = py_dual = anchor_y;
        for (int64_t j = 0; j < m; ++j) target_dual[j] = y[n + j] / node_scale[n + j];
        weight = rescale ? 1.0 : std::max(norm(cs) / std::max(norm(bs), 1e-16), 1e-16);
        double best_weight = weight, error_sum = 0.0;
        const double inf = std::numeric_limits<double>::infinity();
        double best_balance = inf, initial_error = inf, last_trial_error = inf, fixed_error = inf;
        bool restart = false, done = false;
        const auto evaluate = [&](int64_t node_begin, int64_t node_end, int64_t edge_begin, int64_t edge_end) {
            Residual r;
            double primal_sq = 0, dual_sq = 0;
            for (int64_t i = node_begin; i < node_end; ++i) {
                double sum = 0;
                for (int64_t k = offsets[i]; k < offsets[i + 1]; ++k)
                    sum += px[edge_ids[k]] / (variable * bound);
                double delta = sum - b[i];
                primal_sq += delta * delta;
                r.dobj += b[i] * (py_dual[i] / (scale[i] * objective));
            }
            for (int64_t j = edge_begin; j < edge_end; ++j) {
                double score = py_dual[rows[j]] / (scale[rows[j]] * objective)
                    + py_dual[n + cols[j]] / (scale[n + cols[j]] * objective) - c[j];
                double violation = std::max(score, 0.0);
                dual_sq += violation * violation;
                r.pobj += c[j] * (px[j] / (variable * bound));
            }
            r.ap = primal_sq;
            r.ad = dual_sq;
            return r;
        };
        std::vector<Residual> partial_residual(threads);
        Vec partial_primal(threads), partial_dual(threads), partial_interaction(threads);
        setup_seconds = std::chrono::duration<double>(Clock::now() - started).count();
        // CN: source 行融合 primal、边际和 Halpern 更新；target 归约后只需两次常规阶段同步。
        // EN: Fuse primal, marginal, and Halpern updates by source row; ordinary iterations need two phase barriers.
        const auto iteration_started = Clock::now();
        #pragma omp parallel num_threads(threads)
        {
            const int tid = omp_get_thread_num(), team = omp_get_num_threads();
            const int64_t edge_begin = e * tid / team, edge_end = e * (tid + 1) / team;
            const int64_t node_begin = nodes * tid / team, node_end = nodes * (tid + 1) / team;
            #pragma omp single
            actual_threads = team;
            // CN: 用累计边数划分节点区间；空度节点和任意输入边序也被完整覆盖。
            // EN: Partition node ranges by cumulative edge counts, covering zero-degree nodes and arbitrary edge orders.
            const int64_t source_begin = tid == 0 ? 0 : std::lower_bound(offsets.begin(), offsets.begin() + n + 1, e * tid / team) - offsets.begin();
            const int64_t source_end = tid + 1 == team ? n : std::lower_bound(offsets.begin(), offsets.begin() + n + 1, e * (tid + 1) / team) - offsets.begin();
            const int64_t target_begin = tid == 0 ? n : std::lower_bound(offsets.begin() + n, offsets.end(), e + e * tid / team) - offsets.begin();
            const int64_t target_end = tid + 1 == team ? nodes : std::lower_bound(offsets.begin() + n, offsets.end(), e + e * (tid + 1) / team) - offsets.begin();
            int64_t local_count = 0, local_inner = 0;
            while (true) {
                if (local_count % frequency == 0) {
                    const auto check_started = Clock::now();
                    partial_residual[tid] = evaluate(node_begin, node_end, edge_begin, edge_end);
                    #pragma omp barrier
                    #pragma omp single
                    {
                        count = local_count;
                        inner = local_inner;
                        {
                            residual = Residual{};
                            for (int t = 0; t < team; ++t) {
                                residual.ap += partial_residual[t].ap;
                                residual.ad += partial_residual[t].ad;
                                residual.pobj += partial_residual[t].pobj;
                                residual.dobj += partial_residual[t].dobj;
                            }
                            residual.ap = std::sqrt(residual.ap);
                            residual.ad = std::sqrt(residual.ad);
                            residual.rp = residual.ap / (1.0 + mass_norm);
                            residual.rd = residual.ad / (1.0 + cost_norm);
                            residual.gap = std::abs(residual.pobj - residual.dobj) / (1.0 + std::abs(residual.pobj) + std::abs(residual.dobj));
                            check_seconds += std::chrono::duration<double>(Clock::now() - check_started).count();
                            if (trace_enabled) trace.push_back({count, residual, weight});
                            if (!residual.finite() || !std::isfinite(weight)) reason = "NUMERICAL_DIVERGENCE";
                            else if (residual.rp < tol_primal && residual.rd < tol_dual && residual.gap < tol_objective)
                                reason = "OPTIMAL";
                            else if (count >= iteration_limit) reason = "ITERATION_LIMIT";
                            else if (std::chrono::duration<double>(Clock::now() - started).count() >= time_limit)
                                reason = "TIME_LIMIT";
                        }
                        done = reason != "UNSPECIFIED";
                        if (!done) {
                            restart = false;
                            {
                                if (count == frequency) restart = true;
                                else if (count > frequency)
                                    restart = inner >= 0.36 * count || fixed_error <= 0.2 * initial_error
                                        || (fixed_error <= 0.5 * initial_error && fixed_error > last_trial_error);
                                last_trial_error = fixed_error;
                            }
                        }
                    }
                    if (done) break;
                    if (restart) {
                        const auto restart_started = Clock::now();
                        double pd_sq = 0, dd_sq = 0;
                        for (int64_t j = edge_begin; j < edge_end; ++j) {
                            const double delta = px[j] - anchor_x[j];
                            pd_sq += delta * delta;
                        }
                        for (int64_t i = node_begin; i < node_end; ++i) {
                            const double delta = py_dual[i] - anchor_y[i];
                            dd_sq += delta * delta;
                        }
                        partial_primal[tid] = pd_sq;
                        partial_dual[tid] = dd_sq;
                        #pragma omp barrier
                        #pragma omp single
                        {
                            double pd = 0, dd = 0;
                            for (int t = 0; t < team; ++t) { pd += partial_primal[t]; dd += partial_dual[t]; }
                            pd = std::sqrt(pd);
                            dd = std::sqrt(dd);
                            double ratio = residual.rd / std::max(residual.rp, 1e-300);
                            if (pd > 1e-16 && pd < 1e12 && dd > 1e-16 && dd < 1e12 && ratio > 1e-8 && ratio < 1e8) {
                                double error = std::log(dd) - std::log(pd) - std::log(weight);
                                error_sum = 0.3 * error_sum + error;
                                weight *= std::exp(0.99 * error + 0.01 * error_sum);
                            } else {
                                weight = best_weight;
                                error_sum = 0;
                            }
                            if (residual.rp > 0 && residual.rd > 0) {
                                double balance = std::abs(std::log10(residual.rd / residual.rp));
                                if (balance < best_balance) { best_balance = balance; best_weight = weight; }
                            }
                            inner = 0;
                            last_trial_error = inf;
                            ++restarts;
                        }
                        for (int64_t j = edge_begin; j < edge_end; ++j) anchor_x[j] = x[j] = px[j];
                        for (int64_t i = node_begin; i < node_end; ++i) {
                            anchor_y[i] = y[i] = py_dual[i];
                            if (i >= n) target_dual[i - n] = y[i] / node_scale[i];
                        }
                        #pragma omp barrier
                        #pragma omp single
                        restart_seconds += std::chrono::duration<double>(Clock::now() - restart_started).count();
                    }
                    if (restart) local_inner = 0;
                }
                const double primal_step = 0.998 / weight;
                const double dual_step = 0.998 * weight;
                const double halpern = double(local_inner + 1) / double(local_inner + 2);
                const bool major = ((local_count + 1) % frequency) == 0;
                const bool is_restart = local_count % frequency == 0 && restart;
                const bool checkpoint = major || is_restart;
                for (int64_t i = source_begin; i < source_end; ++i) {
                    const double source_dual = y[i] / node_scale[i];
                    double sum = 0.0;
                    for (int64_t k = offsets[i]; k < offsets[i + 1]; ++k) {
                        int64_t j = edge_ids[k];
                        double at = source_dual + target_dual[cols[j]];
                        px[j] = std::max(x[j] - primal_step * (cs[j] - at), 0.0);
                        rx[j] = 2.0 * px[j] - x[j];
                        if (checkpoint) dx[j] = rx[j] - x[j];
                        x[j] = halpern * rx[j] + (1.0 - halpern) * anchor_x[j];
                        sum += rx[j];
                    }
                    py_dual[i] = y[i] - dual_step * (sum / node_scale[i] - bs[i]);
                    const double reflected = 2.0 * py_dual[i] - y[i];
                    if (checkpoint) dy[i] = reflected - y[i];
                    y[i] = halpern * reflected + (1.0 - halpern) * anchor_y[i];
                }
                #pragma omp barrier
                for (int64_t i = target_begin; i < target_end; ++i) {
                    double sum = 0.0;
                    for (int64_t k = offsets[i]; k < offsets[i + 1]; ++k) sum += rx[edge_ids[k]];
                    py_dual[i] = y[i] - dual_step * (sum / node_scale[i] - bs[i]);
                    const double reflected = 2.0 * py_dual[i] - y[i];
                    if (checkpoint) dy[i] = reflected - y[i];
                    y[i] = halpern * reflected + (1.0 - halpern) * anchor_y[i];
                    target_dual[i - n] = y[i] / node_scale[i];
                }
                #pragma omp barrier
                if (checkpoint)
                {
                    const auto fixed_started = Clock::now();
                    double movement = 0, dual_movement = 0, interaction = 0;
                    for (int64_t j = edge_begin; j < edge_end; ++j) {
                        movement += dx[j] * dx[j];
                        interaction += (dy[rows[j]] / node_scale[rows[j]] + dy[n + cols[j]] / node_scale[n + cols[j]]) * dx[j];
                    }
                    for (int64_t i = node_begin; i < node_end; ++i) dual_movement += dy[i] * dy[i];
                    partial_primal[tid] = movement;
                    partial_dual[tid] = dual_movement;
                    partial_interaction[tid] = interaction;
                    #pragma omp barrier
                    #pragma omp single
                    {
                        double movement = 0, dual_movement = 0, interaction = 0;
                        for (int t = 0; t < team; ++t) {
                            movement += partial_primal[t];
                            dual_movement += partial_dual[t];
                            interaction += partial_interaction[t];
                        }
                        double squared = movement * weight + dual_movement / weight + 2.0 * 0.998 * interaction;
                        fixed_error = std::sqrt(std::max(0.0, squared));
                        if (is_restart) initial_error = fixed_error;
                        fixed_seconds += std::chrono::duration<double>(Clock::now() - fixed_started).count();
                    }
                }
                ++local_count;
                ++local_inner;
            }
        }
        iteration_seconds = std::chrono::duration<double>(Clock::now() - iteration_started).count();
        for (int64_t j = 0; j < e; ++j) output_x[j] = px[j] / (variable * bound);
        for (int64_t i = 0; i < nodes; ++i) output_y[i] = py_dual[i] / (scale[i] * objective);
    }
    const double duration = std::chrono::duration<double>(Clock::now() - started).count();
    py::dict diagnostics;
    diagnostics["backend"] = "cpp_lp";
    diagnostics["kkt_check_time_sec"] = check_seconds;
    diagnostics["restart_time_sec"] = restart_seconds;
    diagnostics["fixed_error_time_sec"] = fixed_seconds;
    diagnostics["device"] = "cpu";
    diagnostics["dtype"] = "float64";
    diagnostics["scaling_mode"] = rescale ? "degree_sqrt_with_scalar_retry" : "degree_sqrt";
    diagnostics["step_size"] = 0.998;
    diagnostics["termination_norm"] = "l2";
    diagnostics["termination_abs_primal_res"] = residual.ap;
    diagnostics["algorithm_abs_primal_res_l2"] = residual.ap;
    diagnostics["termination_rel_primal_res"] = residual.rp;
    diagnostics["algorithm_rel_primal_res_l2"] = residual.rp;
    diagnostics["termination_abs_dual_res"] = residual.ad;
    diagnostics["algorithm_abs_dual_res_l2"] = residual.ad;
    diagnostics["termination_rel_dual_res"] = residual.rd;
    diagnostics["algorithm_rel_dual_res_l2"] = residual.rd;
    diagnostics["termination_objective_vector_norm"] = cost_norm;
    diagnostics["termination_constraint_bound_norm"] = mass_norm;
    diagnostics["relative_primal_dual_gap"] = residual.gap;
    diagnostics["primal_weight"] = weight;
    diagnostics["restart_count"] = restarts;
    diagnostics["runtime_sec"] = duration;
    diagnostics["setup_time_sec"] = setup_seconds;
    diagnostics["iteration_time_sec"] = iteration_seconds;
    diagnostics["termination_reason"] = reason;
    diagnostics["numerical_rescaling_retry"] = rescale;
    diagnostics["requested_num_threads"] = threads;
    diagnostics["actual_num_threads"] = actual_threads;
    if (trace_enabled) {
        py::list checkpoints;
        for (const auto& item : trace) {
            py::dict state;
            state["iteration"] = item.iteration;
            state["primal_objective"] = item.residual.pobj;
            state["dual_objective"] = item.residual.dobj;
            state["relative_primal_residual"] = item.residual.rp;
            state["relative_dual_residual"] = item.residual.rd;
            state["relative_gap"] = item.residual.gap;
            state["primal_weight"] = item.weight;
            checkpoints.append(state);
        }
        diagnostics["state_trace"] = checkpoints;
    }
    py::dict result;
    result["success"] = reason == "OPTIMAL";
    result["x"] = x_out;
    result["y"] = y_out;
    result["obj_val"] = residual.pobj;
    result["dual_obj_val"] = residual.dobj;
    result["duration"] = duration;
    result["iterations"] = count;
    result["termination_reason"] = reason;
    result["primal_feas"] = residual.rp;
    result["dual_feas"] = residual.rd;
    result["gap"] = residual.gap;
    result["solver_diag"] = diagnostics;
    return result;
}

PYBIND11_MODULE(_lp, module) {
    module.def("solve_once", &solve_once);
}
