#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <omp.h>
#if defined(__x86_64__)
#include <immintrin.h>
#endif

namespace py = pybind11;

inline void insert_candidate(double* values, int64_t* indices, int64_t k,
                             double value, int64_t index) {
    if (!(value > values[k - 1])) return;
    int64_t p = k - 1;
    while (p > 0 && value > values[p - 1]) {
        values[p] = values[p - 1];
        indices[p] = indices[p - 1];
        --p;
    }
    values[p] = value;
    indices[p] = index;
}

struct TileStats {
    double numerator = 0, denominator = 0, maximum = 0, cost_maximum = 0;
    int64_t count = 0;
};

#if defined(__x86_64__)
template<class T>
__attribute__((target("avx2"))) TileStats vector_row(
        const T* dot, const double* b, const double* g, double a, double f,
        int64_t width, double dot_scale, int mode, double* values, int64_t* indices,
        int64_t k, int64_t target_start) {
    __m256d num = _mm256_setzero_pd(), den = num, mx = num, cm = num;
    const __m256d av = _mm256_set1_pd(a), fv = _mm256_set1_pd(f);
    const __m256d zero = _mm256_setzero_pd(), sign = _mm256_set1_pd(-0.0);
    TileStats stats;
    int64_t c = 0;
    for (; c + 4 <= width; c += 4) {
        __m256d scaled;
        if constexpr (std::is_same_v<T, float>) {
            scaled = _mm256_cvtps_pd(_mm_mul_ps(_mm_loadu_ps(dot + c), _mm_set1_ps(float(dot_scale))));
        } else {
            scaled = _mm256_mul_pd(_mm256_loadu_pd(dot + c), _mm256_set1_pd(dot_scale));
        }
        const __m256d cost = _mm256_sub_pd(_mm256_add_pd(av, _mm256_loadu_pd(b + c)), scaled);
        const __m256d dual = mode == 1 ? _mm256_loadu_pd(g + c) : _mm256_add_pd(fv, _mm256_loadu_pd(g + c));
        const __m256d score = _mm256_sub_pd(dual, cost);
        if (mode == 0) {
            const __m256d positive = _mm256_max_pd(score, zero);
            num = _mm256_add_pd(num, _mm256_mul_pd(positive, positive));
            den = _mm256_add_pd(den, _mm256_mul_pd(cost, cost));
            mx = _mm256_max_pd(mx, positive);
            cm = _mm256_max_pd(cm, _mm256_andnot_pd(sign, cost));
            stats.count += __builtin_popcount(unsigned(_mm256_movemask_pd(_mm256_cmp_pd(score, zero, _CMP_GT_OQ))));
        }
        // CN: 向量过滤无竞争力的候选，只有过阈值的 lane 执行小 top-k 插入。
        // EN: Filter uncompetitive candidates in vectors and insert only lanes exceeding the top-k threshold.
        unsigned mask = unsigned(_mm256_movemask_pd(_mm256_cmp_pd(score, _mm256_set1_pd(values[k - 1]), _CMP_GT_OQ)));
        if (mask) {
            double lanes[4];
            _mm256_storeu_pd(lanes, score);
            for (int lane = 0; lane < 4; ++lane)
                if (mask & (1u << lane)) insert_candidate(values, indices, k, lanes[lane], target_start + c + lane);
        }
    }
    double ns[4], ds[4], ms[4], cs[4];
    _mm256_storeu_pd(ns, num); _mm256_storeu_pd(ds, den);
    _mm256_storeu_pd(ms, mx); _mm256_storeu_pd(cs, cm);
    for (int lane = 0; lane < 4; ++lane) {
        stats.numerator += ns[lane]; stats.denominator += ds[lane];
        stats.maximum = std::max(stats.maximum, ms[lane]);
        stats.cost_maximum = std::max(stats.cost_maximum, cs[lane]);
    }
    for (; c < width; ++c) {
        const T scaled = T(dot_scale) * dot[c];
        const double cost = (a + b[c]) - double(scaled);
        const double score = (mode == 1 ? g[c] : f + g[c]) - cost;
        insert_candidate(values, indices, k, score, target_start + c);
        if (mode == 0) {
            const double positive = std::max(score, 0.0);
            stats.numerator += positive * positive; stats.denominator += cost * cost;
            stats.maximum = std::max(stats.maximum, positive);
            stats.cost_maximum = std::max(stats.cost_maximum, std::abs(cost));
            stats.count += score > 0;
        }
    }
    return stats;
}

template<class T>
__attribute__((target("avx2"))) void vector_columns(
        const T* dot, int64_t stride, const double* a, const double* f,
        const double* b, const double* g, int64_t rows, int64_t width,
        double dot_scale, int mode, double* values, int64_t* indices,
        int64_t value_stride, int64_t index_stride, int64_t k, int64_t source_start) {
    for (int64_t r = 0; r < rows; ++r) {
        int64_t c = 0;
        for (; c + 4 <= width; c += 4) {
            __m256d scaled;
            if constexpr (std::is_same_v<T, float>) {
                scaled = _mm256_cvtps_pd(_mm_mul_ps(_mm_loadu_ps(dot + r*stride + c), _mm_set1_ps(float(dot_scale))));
            } else {
                scaled = _mm256_mul_pd(_mm256_loadu_pd(dot + r*stride + c), _mm256_set1_pd(dot_scale));
            }
            const __m256d cost = _mm256_sub_pd(_mm256_add_pd(_mm256_set1_pd(a[r]), _mm256_loadu_pd(b+c)), scaled);
            const __m256d dual = mode == 2 ? _mm256_set1_pd(f[r]) : _mm256_add_pd(_mm256_set1_pd(f[r]), _mm256_loadu_pd(g+c));
            const __m256d score = _mm256_sub_pd(dual, cost);
            const __m256d threshold = _mm256_set_pd(values[(c+3)*value_stride+k-1], values[(c+2)*value_stride+k-1],
                                                    values[(c+1)*value_stride+k-1], values[c*value_stride+k-1]);
            unsigned mask = unsigned(_mm256_movemask_pd(_mm256_cmp_pd(score, threshold, _CMP_GT_OQ)));
            if (mask) {
                double lanes[4];
                _mm256_storeu_pd(lanes, score);
                for (int lane = 0; lane < 4; ++lane) {
                    if (mask & (1u << lane)) insert_candidate(values+(c+lane)*value_stride,
                        indices+(c+lane)*index_stride, k, lanes[lane], source_start+r);
                }
            }
        }
        for (; c < width; ++c) {
            const T scaled = T(dot_scale) * dot[r*stride+c];
            const double cost = (a[r]+b[c])-double(scaled);
            const double score = (mode == 2 ? f[r] : f[r]+g[c])-cost;
            insert_candidate(values+c*value_stride, indices+c*index_stride, k, score, source_start+r);
        }
    }
}
#endif

template<class T>
py::tuple dot_tile(py::array_t<T> products, py::array_t<double> source_bias,
                   py::array_t<double> target_bias, py::array_t<double> source_dual,
                   py::array_t<double> target_dual, py::array_t<double> row_values,
                   py::array_t<int64_t> row_indices, py::array_t<double> col_values,
                   py::array_t<int64_t> col_indices, double dot_scale,
                   int64_t source_start, int64_t target_start, int mode, int threads) {
    auto dot = products.template unchecked<2>();
    auto a = source_bias.unchecked<1>(), b = target_bias.unchecked<1>();
    auto f = source_dual.unchecked<1>(), g = target_dual.unchecked<1>();
    auto rv = row_values.mutable_unchecked<2>();
    auto ri = row_indices.mutable_unchecked<2>();
    auto cv = col_values.mutable_unchecked<2>();
    auto ci = col_indices.mutable_unchecked<2>();
    int64_t rows = dot.shape(0), cols = dot.shape(1), rk = rv.shape(1), ck = cv.shape(1);
    if (threads < 1 || mode < 0 || mode > 2 || a.shape(0) != rows || b.shape(0) != cols ||
        f.shape(0) != rows || g.shape(0) != cols || rv.shape(0) != rows || cv.shape(0) != cols ||
        ri.shape(0) != rows || ri.shape(1) != rk || ci.shape(0) != cols || ci.shape(1) != ck ||
        rk < 1 || ck < 1 || row_values.strides(1) != sizeof(double) || col_values.strides(1) != sizeof(double) ||
        row_indices.strides(1) != sizeof(int64_t) || col_indices.strides(1) != sizeof(int64_t))
        throw std::invalid_argument("Invalid fused dot tile shapes");
    double numerator = 0, denominator = 0, maximum = 0, cost_maximum = 0;
    int64_t count = 0;
    bool vectorized = false;
    #if defined(__x86_64__)
    vectorized = __builtin_cpu_supports("avx2") && products.strides(1) == sizeof(T) &&
                 target_bias.strides(0) == sizeof(double) && target_dual.strides(0) == sizeof(double);
    #endif
    {
        py::gil_scoped_release release;
        // CN: 保留 Torch 的乘法精度及运算顺序，直接筛选候选，不写出 FP64 cost/score tile。
        // EN: Preserve Torch multiplication precision and operation order, selecting candidates without FP64 cost/score tiles.
        #pragma omp parallel num_threads(threads) reduction(+:numerator,denominator,count) reduction(max:maximum,cost_maximum)
        {
            if (mode != 2) {
                #pragma omp for schedule(static) nowait
                for (int64_t r = 0; r < rows; ++r) {
                    #if defined(__x86_64__)
                    if (vectorized) {
                        const auto stats = vector_row(&dot(r, 0), &b(0), &g(0), a(r), f(r), cols,
                                                       dot_scale, mode, &rv(r, 0), &ri(r, 0), rk, target_start);
                        numerator += stats.numerator; denominator += stats.denominator;
                        maximum = std::max(maximum, stats.maximum);
                        cost_maximum = std::max(cost_maximum, stats.cost_maximum);
                        count += stats.count;
                        continue;
                    }
                    #endif
                    for (int64_t c = 0; c < cols; ++c) {
                        const T scaled = T(dot_scale) * dot(r, c);
                        const double cost = (a(r) + b(c)) - double(scaled);
                        const double score = mode == 1 ? g(c) - cost : (f(r) + g(c)) - cost;
                        insert_candidate(&rv(r, 0), &ri(r, 0), rk, score, target_start + c);
                        if (mode == 0) {
                            const double positive = std::max(score, 0.0);
                            numerator += positive * positive;
                            denominator += cost * cost;
                            count += score > 0;
                            maximum = std::max(maximum, positive);
                            cost_maximum = std::max(cost_maximum, std::abs(cost));
                        }
                    }
                }
            }
            if (mode != 1) {
                // CN: 相邻列成组遍历，共享 cache line，避免逐列跨大步长读取。
                // EN: Traverse adjacent columns together to reuse cache lines instead of reading one strided column at a time.
                #pragma omp for schedule(static) nowait
                for (int64_t block = 0; block < cols; block += 16) {
                    #if defined(__x86_64__)
                    if (vectorized && source_bias.strides(0) == sizeof(double) && source_dual.strides(0) == sizeof(double)) {
                        vector_columns(&dot(0, block), products.strides(0)/sizeof(T), &a(0), &f(0), &b(block), &g(block),
                                       rows, std::min<int64_t>(16, cols-block), dot_scale, mode,
                                       &cv(block, 0), &ci(block, 0), col_values.strides(0)/sizeof(double),
                                       col_indices.strides(0)/sizeof(int64_t), ck, source_start);
                        continue;
                    }
                    #endif
                    for (int64_t r = 0; r < rows; ++r) {
                        for (int64_t c = block; c < std::min(cols, block + 16); ++c) {
                            const T scaled = T(dot_scale) * dot(r, c);
                            const double cost = (a(r) + b(c)) - double(scaled);
                            const double score = mode == 2 ? f(r) - cost : (f(r) + g(c)) - cost;
                            insert_candidate(&cv(c, 0), &ci(c, 0), ck, score, source_start + r);
                        }
                    }
                }
            }
        }
    }
    return py::make_tuple(numerator, denominator, count, maximum, cost_maximum);
}

py::tuple merge_topk(py::array_t<double> old_values, py::array_t<int64_t> old_indices,
                     py::array_t<double> new_values, py::array_t<int64_t> new_indices,
                     int64_t k, int threads) {
    auto old = old_values.unchecked<2>();
    auto old_id = old_indices.unchecked<2>();
    auto input = new_values.unchecked<2>();
    auto input_id = new_indices.unchecked<2>();
    int64_t rows = input.shape(0), width = input.shape(1);
    if (k < 1 || old.shape(0) != rows || old.shape(1) != k ||
        old_id.shape(0) != rows || old_id.shape(1) != k ||
        input_id.shape(0) != rows || input_id.shape(1) != width || threads < 1)
        throw std::invalid_argument("Invalid top-k shapes or thread count");
    py::array_t<double> values({rows, k});
    py::array_t<int64_t> indices({rows, k});
    auto out = values.mutable_unchecked<2>();
    auto out_id = indices.mutable_unchecked<2>();
    {
        py::gil_scoped_release release;
        // CN: 每行维护已排序的小候选集，直接读取 strided tile，不物化完整索引。
        // EN: Maintain a small sorted candidate set per row and read strided tiles without materializing indices.
        #pragma omp parallel for num_threads(threads) schedule(static)
        for (int64_t r = 0; r < rows; ++r) {
            for (int64_t j = 0; j < k; ++j) {
                out(r, j) = old(r, j);
                out_id(r, j) = old_id(r, j);
            }
            // CN: 排序初始候选，使接口也支持未排序的输入。
            // EN: Sort initial candidates so the interface also accepts unsorted inputs.
            for (int64_t j = 1; j < k; ++j) {
                double value = out(r, j);
                int64_t index = out_id(r, j), p = j;
                while (p > 0 && value > out(r, p - 1)) {
                    out(r, p) = out(r, p - 1);
                    out_id(r, p) = out_id(r, p - 1);
                    --p;
                }
                out(r, p) = value;
                out_id(r, p) = index;
            }
            for (int64_t c = 0; c < width; ++c) {
                double value = input(r, c);
                if (!(value > out(r, k - 1))) continue;
                int64_t p = k - 1;
                while (p > 0 && value > out(r, p - 1)) {
                    out(r, p) = out(r, p - 1);
                    out_id(r, p) = out_id(r, p - 1);
                    --p;
                }
                out(r, p) = value;
                out_id(r, p) = input_id(r, c);
            }
        }
    }
    return py::make_tuple(values, indices);
}

py::tuple score_and_stats(py::array_t<double> costs, py::array_t<double> source_dual,
                         py::array_t<double> target_dual, int threads) {
    auto cost = costs.unchecked<2>();
    auto f = source_dual.unchecked<1>();
    auto g = target_dual.unchecked<1>();
    int64_t rows = cost.shape(0), cols = cost.shape(1);
    if (f.shape(0) != rows || g.shape(0) != cols || threads < 1)
        throw std::invalid_argument("Invalid score tile shapes or thread count");
    py::array_t<double> scores({rows, cols});
    auto out = scores.mutable_unchecked<2>();
    double numerator = 0, denominator = 0, maximum = 0, cost_maximum = 0;
    int64_t count = 0;
    {
        py::gil_scoped_release release;
        // CN: 一次遍历生成 score 并归约证书，避免多份 tile 大小的中间数组。
        // EN: Generate scores and reduce the certificate in one pass without tile-sized intermediate arrays.
        #pragma omp parallel for num_threads(threads) schedule(static) reduction(+:numerator,denominator,count) reduction(max:maximum,cost_maximum)
        for (int64_t r = 0; r < rows; ++r) {
            for (int64_t c = 0; c < cols; ++c) {
                double value = cost(r, c), score = (f(r) + g(c)) - value;
                out(r, c) = score;
                double positive = std::max(score, 0.0);
                numerator += positive * positive;
                denominator += value * value;
                count += score > 0;
                maximum = std::max(maximum, positive);
                cost_maximum = std::max(cost_maximum, std::abs(value));
            }
        }
    }
    return py::make_tuple(scores, numerator, denominator, count, maximum, cost_maximum);
}

PYBIND11_MODULE(_scan, module) {
    module.def("merge_topk", &merge_topk);
    module.def("score_and_stats", &score_and_stats);
    module.def("dot_tile", &dot_tile<float>);
    module.def("dot_tile", &dot_tile<double>);
}
