#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

namespace {

constexpr double kInfinity = std::numeric_limits<double>::infinity();
using CompactIndex = std::int32_t;

class DisjointSetLct {
public:
    explicit DisjointSetLct(std::int64_t size)
        : parent_(static_cast<std::size_t>(size)), rank_(static_cast<std::size_t>(size), 0) {
        std::iota(parent_.begin(), parent_.end(), 0);
    }

    std::int64_t find(std::int64_t value) {
        if (parent_[static_cast<std::size_t>(value)] != value) {
            parent_[static_cast<std::size_t>(value)] = find(parent_[static_cast<std::size_t>(value)]);
        }
        return parent_[static_cast<std::size_t>(value)];
    }

    bool unite(std::int64_t first, std::int64_t second) {
        first = find(first);
        second = find(second);
        if (first == second) {
            return false;
        }
        if (rank_[static_cast<std::size_t>(first)] < rank_[static_cast<std::size_t>(second)]) {
            std::swap(first, second);
        }
        parent_[static_cast<std::size_t>(second)] = first;
        if (rank_[static_cast<std::size_t>(first)] == rank_[static_cast<std::size_t>(second)]) {
            ++rank_[static_cast<std::size_t>(first)];
        }
        return true;
    }

private:
    std::vector<std::int64_t> parent_;
    std::vector<std::uint8_t> rank_;
};

struct DynamicTreeNode {
    CompactIndex child[2] = {0, 0};
    CompactIndex parent = 0;
    CompactIndex edge_count = 0;
    bool reverse = false;
    bool is_edge = false;
    bool has_lazy_add = false;
    double value = 0.0;
    double cost = 0.0;
    std::array<double, 2> minimum = {kInfinity, kInfinity};
    std::array<CompactIndex, 2> minimum_node = {0, 0};
    std::array<double, 2> cost_sum = {0.0, 0.0};
    std::array<double, 2> lazy_add = {0.0, 0.0};
};

class LinkCutForest {
public:
    explicit LinkCutForest(std::int64_t size) : nodes_(static_cast<std::size_t>(size + 1)) {
        ancestors_.reserve(128);
        traversal_.reserve(128);
    }

    void initialize_vertex(std::int64_t node) {
        nodes_[static_cast<std::size_t>(node)] = DynamicTreeNode{};
        pull(node);
    }

    void initialize_edge(std::int64_t node, double value, double cost) {
        DynamicTreeNode initialized;
        initialized.is_edge = true;
        initialized.value = value;
        initialized.cost = cost;
        nodes_[static_cast<std::size_t>(node)] = initialized;
        pull(node);
    }

    void make_root(std::int64_t node) {
        access(node);
        apply_reverse(node);
    }

    void link(std::int64_t first, std::int64_t second) {
        make_root(first);
        nodes_[static_cast<std::size_t>(first)].parent = second;
    }

    void link_edge_between(
        std::int64_t first,
        std::int64_t edge,
        std::int64_t second) {
        // CN: 新边是孤立节点；一次 evert 后可直接设置两条 represented-tree parent 关系。
        // EN: A new edge is isolated, so one evert suffices before setting both represented-tree parents.
        make_root(first);
        nodes_[static_cast<std::size_t>(first)].parent = edge;
        nodes_[static_cast<std::size_t>(edge)].parent = second;
    }

    bool detach_exposed_edge(std::int64_t edge) {
        // CN: fundamental path 已暴露为一棵 Splay 树；将离树边旋至根即可一次切开两侧。
        // EN: The fundamental path is already exposed; splaying the leaving edge splits both sides at once.
        splay(edge);
        DynamicTreeNode& edge_node = nodes_[static_cast<std::size_t>(edge)];
        const std::int64_t first_side = edge_node.child[0];
        const std::int64_t second_side = edge_node.child[1];
        if (first_side == 0 || second_side == 0) {
            return false;
        }
        nodes_[static_cast<std::size_t>(first_side)].parent = 0;
        nodes_[static_cast<std::size_t>(second_side)].parent = 0;
        edge_node.child[0] = 0;
        edge_node.child[1] = 0;
        pull(edge);
        return true;
    }

    std::int64_t expose_path(std::int64_t first, std::int64_t second) {
        make_root(first);
        access(second);
        return second;
    }

    double minimum(std::int64_t root, int residue) const {
        return nodes_[static_cast<std::size_t>(root)].minimum[static_cast<std::size_t>(residue)];
    }

    std::int64_t minimum_node(std::int64_t root, int residue) const {
        return nodes_[static_cast<std::size_t>(root)].minimum_node[static_cast<std::size_t>(residue)];
    }

    double cost_sum(std::int64_t root, int residue) const {
        return nodes_[static_cast<std::size_t>(root)].cost_sum[static_cast<std::size_t>(residue)];
    }

    void add_by_residue(std::int64_t root, const std::array<double, 2>& additions) {
        apply_add(root, additions);
    }

    void flush_lazy_tags() {
        // CN: 对每棵辅助 Splay 树自顶向下传播标记，使全部节点值在线性时间内可读。
        // EN: Push tags top-down in every auxiliary splay tree so all node values become readable in linear time.
        for (std::int64_t node = 1; node < static_cast<std::int64_t>(nodes_.size()); ++node) {
            if (!is_auxiliary_root(node)) {
                continue;
            }
            traversal_.clear();
            traversal_.push_back(node);
            while (!traversal_.empty()) {
                const std::int64_t current = traversal_.back();
                traversal_.pop_back();
                push(current);
                const DynamicTreeNode& current_node = nodes_[static_cast<std::size_t>(current)];
                if (current_node.child[0] != 0) {
                    traversal_.push_back(current_node.child[0]);
                }
                if (current_node.child[1] != 0) {
                    traversal_.push_back(current_node.child[1]);
                }
            }
        }
    }

    double value(std::int64_t node) const {
        return nodes_[static_cast<std::size_t>(node)].value;
    }

private:
    bool is_auxiliary_root(std::int64_t node) const {
        const std::int64_t parent = nodes_[static_cast<std::size_t>(node)].parent;
        return parent == 0 ||
            (nodes_[static_cast<std::size_t>(parent)].child[0] != node &&
             nodes_[static_cast<std::size_t>(parent)].child[1] != node);
    }

    std::int64_t subtree_edge_count(std::int64_t node) const {
        return node == 0 ? 0 : nodes_[static_cast<std::size_t>(node)].edge_count;
    }

    void append_aggregate(
        DynamicTreeNode& destination,
        const DynamicTreeNode& segment,
        std::int64_t offset) {
        for (int residue = 0; residue < 2; ++residue) {
            const int shifted = static_cast<int>((offset + residue) & 1);
            destination.cost_sum[static_cast<std::size_t>(shifted)] +=
                segment.cost_sum[static_cast<std::size_t>(residue)];
            if (segment.minimum[static_cast<std::size_t>(residue)] <
                destination.minimum[static_cast<std::size_t>(shifted)]) {
                destination.minimum[static_cast<std::size_t>(shifted)] =
                    segment.minimum[static_cast<std::size_t>(residue)];
                destination.minimum_node[static_cast<std::size_t>(shifted)] =
                    segment.minimum_node[static_cast<std::size_t>(residue)];
            }
        }
    }

    void pull(std::int64_t node) {
        if (node == 0) {
            return;
        }
        DynamicTreeNode& current = nodes_[static_cast<std::size_t>(node)];
        const std::int64_t left = current.child[0];
        const std::int64_t right = current.child[1];
        current.edge_count =
            subtree_edge_count(left) + (current.is_edge ? 1 : 0) + subtree_edge_count(right);
        current.minimum = {kInfinity, kInfinity};
        current.minimum_node = {0, 0};
        current.cost_sum = {0.0, 0.0};
        if (left != 0) {
            append_aggregate(current, nodes_[static_cast<std::size_t>(left)], 0);
        }
        const int self_residue = static_cast<int>(subtree_edge_count(left) & 1);
        if (current.is_edge) {
            if (current.value < current.minimum[static_cast<std::size_t>(self_residue)]) {
                current.minimum[static_cast<std::size_t>(self_residue)] = current.value;
                current.minimum_node[static_cast<std::size_t>(self_residue)] = node;
            }
            current.cost_sum[static_cast<std::size_t>(self_residue)] += current.cost;
        }
        if (right != 0) {
            append_aggregate(
                current,
                nodes_[static_cast<std::size_t>(right)],
                subtree_edge_count(left) + (current.is_edge ? 1 : 0));
        }
    }

    void apply_add(std::int64_t node, const std::array<double, 2>& additions) {
        if (node == 0) {
            return;
        }
        DynamicTreeNode& current = nodes_[static_cast<std::size_t>(node)];
        const bool has_addition = additions[0] != 0.0 || additions[1] != 0.0;
        if (!has_addition) {
            return;
        }
        current.has_lazy_add = true;
        for (int residue = 0; residue < 2; ++residue) {
            const double addition = additions[static_cast<std::size_t>(residue)];
            if (current.minimum[static_cast<std::size_t>(residue)] < kInfinity) {
                current.minimum[static_cast<std::size_t>(residue)] += addition;
            }
            current.lazy_add[static_cast<std::size_t>(residue)] += addition;
        }
        if (current.is_edge) {
            const int self_residue = static_cast<int>(subtree_edge_count(current.child[0]) & 1);
            current.value += additions[static_cast<std::size_t>(self_residue)];
        }
    }

    void apply_reverse(std::int64_t node) {
        if (node == 0) {
            return;
        }
        DynamicTreeNode& current = nodes_[static_cast<std::size_t>(node)];
        std::swap(current.child[0], current.child[1]);
        const auto old_minimum = current.minimum;
        const auto old_minimum_node = current.minimum_node;
        const auto old_cost_sum = current.cost_sum;
        const auto old_lazy_add = current.lazy_add;
        for (int residue = 0; residue < 2; ++residue) {
            const int old_residue = static_cast<int>((current.edge_count - 1 - residue) & 1);
            current.minimum[static_cast<std::size_t>(residue)] = old_minimum[static_cast<std::size_t>(old_residue)];
            current.minimum_node[static_cast<std::size_t>(residue)] =
                old_minimum_node[static_cast<std::size_t>(old_residue)];
            current.cost_sum[static_cast<std::size_t>(residue)] = old_cost_sum[static_cast<std::size_t>(old_residue)];
            current.lazy_add[static_cast<std::size_t>(residue)] = old_lazy_add[static_cast<std::size_t>(old_residue)];
        }
        current.reverse = !current.reverse;
    }

    void push(std::int64_t node) {
        if (node == 0) {
            return;
        }
        DynamicTreeNode& current = nodes_[static_cast<std::size_t>(node)];
        if (current.reverse) {
            apply_reverse(current.child[0]);
            apply_reverse(current.child[1]);
            current.reverse = false;
        }
        if (!current.has_lazy_add) {
            return;
        }
        if (current.child[0] != 0) {
            apply_add(current.child[0], current.lazy_add);
        }
        if (current.child[1] != 0) {
            std::array<double, 2> shifted{};
            const std::int64_t offset =
                subtree_edge_count(current.child[0]) + (current.is_edge ? 1 : 0);
            for (int residue = 0; residue < 2; ++residue) {
                shifted[static_cast<std::size_t>(residue)] =
                    current.lazy_add[static_cast<std::size_t>((offset + residue) & 1)];
            }
            apply_add(current.child[1], shifted);
        }
        current.lazy_add = {0.0, 0.0};
        current.has_lazy_add = false;
    }

    void rotate(std::int64_t node) {
        const std::int64_t parent = nodes_[static_cast<std::size_t>(node)].parent;
        const std::int64_t grandparent = nodes_[static_cast<std::size_t>(parent)].parent;
        const int direction = nodes_[static_cast<std::size_t>(parent)].child[1] == node ? 1 : 0;
        const std::int64_t middle = nodes_[static_cast<std::size_t>(node)].child[direction ^ 1];
        if (!is_auxiliary_root(parent)) {
            const int parent_direction = nodes_[static_cast<std::size_t>(grandparent)].child[1] == parent ? 1 : 0;
            nodes_[static_cast<std::size_t>(grandparent)].child[parent_direction] = node;
        }
        nodes_[static_cast<std::size_t>(node)].parent = grandparent;
        nodes_[static_cast<std::size_t>(node)].child[direction ^ 1] = parent;
        nodes_[static_cast<std::size_t>(parent)].parent = node;
        nodes_[static_cast<std::size_t>(parent)].child[direction] = middle;
        if (middle != 0) {
            nodes_[static_cast<std::size_t>(middle)].parent = parent;
        }
        pull(parent);
        pull(node);
    }

    void splay(std::int64_t node) {
        ancestors_.clear();
        ancestors_.push_back(node);
        for (std::int64_t current = node; !is_auxiliary_root(current);) {
            current = nodes_[static_cast<std::size_t>(current)].parent;
            ancestors_.push_back(current);
        }
        for (auto iterator = ancestors_.rbegin(); iterator != ancestors_.rend(); ++iterator) {
            push(*iterator);
        }
        while (!is_auxiliary_root(node)) {
            const std::int64_t parent = nodes_[static_cast<std::size_t>(node)].parent;
            const std::int64_t grandparent = nodes_[static_cast<std::size_t>(parent)].parent;
            if (!is_auxiliary_root(parent)) {
                const bool zig_zig =
                    (nodes_[static_cast<std::size_t>(parent)].child[0] == node) ==
                    (nodes_[static_cast<std::size_t>(grandparent)].child[0] == parent);
                rotate(zig_zig ? parent : node);
            }
            rotate(node);
        }
    }

    void access(std::int64_t node) {
        std::int64_t previous = 0;
        for (std::int64_t current = node; current != 0;
             current = nodes_[static_cast<std::size_t>(current)].parent) {
            splay(current);
            nodes_[static_cast<std::size_t>(current)].child[1] = previous;
            if (previous != 0) {
                nodes_[static_cast<std::size_t>(previous)].parent = current;
            }
            pull(current);
            previous = current;
        }
        splay(node);
    }

    std::vector<DynamicTreeNode> nodes_;
    std::vector<std::int64_t> ancestors_;
    std::vector<std::int64_t> traversal_;
};

}  // namespace

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
    std::int64_t* output_tree_operations) {
    if (n_source <= 0 || n_target <= 0 || edge_count < 0 || rows == nullptr || cols == nullptr ||
        values == nullptr || costs == nullptr || target_nnz < 0 || output_nnz == nullptr || output_cycle_updates == nullptr ||
        output_tree_operations == nullptr || !std::isfinite(zero_tolerance) || zero_tolerance < 0.0) {
        return 1;
    }
    const std::int64_t vertex_count = n_source + n_target;
    const std::int64_t maximum_node_count = 2 * vertex_count - 1;
    // CN: 32-bit 内部索引显著降低动态树的内存流量；可表示的规模远超当前内存上限。
    // EN: Compact 32-bit indices reduce dynamic-tree memory traffic; their range exceeds practical RAM limits.
    if (maximum_node_count >= static_cast<std::int64_t>(std::numeric_limits<CompactIndex>::max())) {
        return 8;
    }
    std::vector<CompactIndex> first_vertex(static_cast<std::size_t>(edge_count));
    std::vector<CompactIndex> second_vertex(static_cast<std::size_t>(edge_count));
    std::vector<CompactIndex> tree_edges;
    tree_edges.reserve(static_cast<std::size_t>(std::min(edge_count, vertex_count - 1)));
    std::vector<CompactIndex> non_tree_edges;
    non_tree_edges.reserve(static_cast<std::size_t>(edge_count));
    DisjointSetLct components(vertex_count);
    std::int64_t initial_nnz = 0;

    for (std::int64_t edge = 0; edge < edge_count; ++edge) {
        if (rows[edge] < 0 || rows[edge] >= n_source || cols[edge] < 0 || cols[edge] >= n_target ||
            !std::isfinite(values[edge]) || values[edge] < -zero_tolerance || !std::isfinite(costs[edge])) {
            return 2;
        }
        if (values[edge] <= zero_tolerance) {
            values[edge] = 0.0;
        } else {
            ++initial_nnz;
        }
        const std::int64_t source_vertex = rows[edge] + 1;
        const std::int64_t target_vertex = n_source + cols[edge] + 1;
        first_vertex[static_cast<std::size_t>(edge)] = source_vertex;
        second_vertex[static_cast<std::size_t>(edge)] = target_vertex;
        if (values[edge] <= zero_tolerance) {
            continue;
        }
        if (components.unite(source_vertex - 1, target_vertex - 1)) {
            tree_edges.push_back(static_cast<CompactIndex>(edge));
        } else {
            non_tree_edges.push_back(static_cast<CompactIndex>(edge));
        }
    }
    // CN: LCT 只保存当前生成森林；pivot 时复用离树节点，而不是为所有输入边分配节点。
    // EN: Store only the current spanning forest and recycle leaving nodes instead of allocating one node per input edge.
    LinkCutForest forest(vertex_count + static_cast<std::int64_t>(tree_edges.size()));
    for (std::int64_t vertex = 0; vertex < vertex_count; ++vertex) {
        forest.initialize_vertex(vertex + 1);
    }
    std::vector<CompactIndex> edge_for_tree_slot(tree_edges.size());
    for (std::size_t slot = 0; slot < tree_edges.size(); ++slot) {
        const std::int64_t edge = tree_edges[slot];
        const std::int64_t edge_node = vertex_count + static_cast<std::int64_t>(slot) + 1;
        edge_for_tree_slot[slot] = static_cast<CompactIndex>(edge);
        forest.initialize_edge(edge_node, values[edge], costs[edge]);
        forest.link(first_vertex[static_cast<std::size_t>(edge)], edge_node);
        forest.link(edge_node, second_vertex[static_cast<std::size_t>(edge)]);
    }

    std::int64_t cycle_updates = 0;
    std::int64_t tree_operations = 0;
    std::int64_t next_support_check =
        target_nnz > 0 ? std::max<std::int64_t>(0, initial_nnz - target_nnz) : edge_count;
    for (std::size_t entering_position = 0; entering_position < non_tree_edges.size(); ++entering_position) {
        if (target_nnz > 0 && cycle_updates >= next_support_check) {
            // CN: 数值退化时一次 pivot 未必减少正 support；在线性 flush 后按真实 nnz 决定是否停止。
            // EN: A degenerate pivot need not reduce positive support; use an exact nnz check after a linear flush.
            forest.flush_lazy_tags();
            std::int64_t tree_nnz = 0;
            for (std::size_t slot = 0; slot < edge_for_tree_slot.size(); ++slot) {
                const std::int64_t edge_node = vertex_count + static_cast<std::int64_t>(slot) + 1;
                tree_nnz += forest.value(edge_node) > zero_tolerance ? 1 : 0;
            }
            const std::int64_t exact_nnz = tree_nnz +
                static_cast<std::int64_t>(non_tree_edges.size() - entering_position);
            if (exact_nnz <= target_nnz) {
                break;
            }
            next_support_check = cycle_updates + (exact_nnz - target_nnz);
        }
        const std::int64_t entering_edge = non_tree_edges[entering_position];
        // CN: 每条初始非树边只访问一次，进入树前从未收到 path update。
        // EN: Each initial non-tree edge is visited once and receives no path update before entering the tree.
        double entering_value = values[entering_edge];
        if (entering_value <= zero_tolerance) {
            continue;
        }
        const std::int64_t path_root = forest.expose_path(
            first_vertex[static_cast<std::size_t>(entering_edge)],
            second_vertex[static_cast<std::size_t>(entering_edge)]);
        tree_operations += 2;
        const double directional_cost =
            costs[entering_edge] - forest.cost_sum(path_root, 0) + forest.cost_sum(path_root, 1);
        const double orientation = directional_cost <= 0.0 ? 1.0 : -1.0;
        const int decreasing_residue = orientation > 0.0 ? 0 : 1;
        double step = forest.minimum(path_root, decreasing_residue);
        const std::int64_t leaving_node = forest.minimum_node(path_root, decreasing_residue);
        if (orientation < 0.0) {
            step = std::min(step, entering_value);
        }
        if (!std::isfinite(step)) {
            return 3;
        }
        if (step < -zero_tolerance) {
            return 6;
        }
        step = std::max(0.0, step);
        std::array<double, 2> additions = {0.0, 0.0};
        additions[0] = -orientation * step;
        additions[1] = orientation * step;
        forest.add_by_residue(path_root, additions);
        entering_value += orientation * step;
        if (entering_value <= zero_tolerance) {
            entering_value = 0.0;
        }

        if (entering_value > zero_tolerance) {
            if (leaving_node <= vertex_count) {
                return 4;
            }
            const std::size_t leaving_slot =
                static_cast<std::size_t>(leaving_node - vertex_count - 1);
            const std::int64_t leaving_edge = edge_for_tree_slot[leaving_slot];
            if (!forest.detach_exposed_edge(leaving_node)) {
                return 7;
            }
            values[leaving_edge] = 0.0;
            forest.initialize_edge(leaving_node, entering_value, costs[entering_edge]);
            edge_for_tree_slot[leaving_slot] = static_cast<CompactIndex>(entering_edge);
            forest.link_edge_between(
                first_vertex[static_cast<std::size_t>(entering_edge)],
                leaving_node,
                second_vertex[static_cast<std::size_t>(entering_edge)]);
            tree_operations += 4;
        } else {
            values[entering_edge] = 0.0;
        }
        ++cycle_updates;
    }

    // CN: 逐边 access 会令写回阶段退化为 O(E log V)；一次线性 flush 即可物化所有 lazy update。
    // EN: Per-edge access makes write-back O(E log V); one linear flush materializes every lazy update.
    forest.flush_lazy_tags();
    for (std::size_t slot = 0; slot < edge_for_tree_slot.size(); ++slot) {
        const std::int64_t edge_node = vertex_count + static_cast<std::int64_t>(slot) + 1;
        const std::int64_t edge = edge_for_tree_slot[slot];
        const double materialized = forest.value(edge_node);
        if (!std::isfinite(materialized) || materialized < -zero_tolerance) {
            return 5;
        }
        values[edge] = materialized > zero_tolerance ? materialized : 0.0;
    }
    std::int64_t final_nnz = 0;
    for (std::int64_t edge = 0; edge < edge_count; ++edge) {
        if (!std::isfinite(values[edge]) || values[edge] < -zero_tolerance) {
            return 5;
        }
        values[edge] = values[edge] > zero_tolerance ? values[edge] : 0.0;
        final_nnz += values[edge] > zero_tolerance ? 1 : 0;
    }
    if (target_nnz > 0 && final_nnz > target_nnz) {
        return 9;
    }
    *output_nnz = final_nnz;
    *output_cycle_updates = cycle_updates;
    *output_tree_operations = tree_operations;
    return 0;
}
