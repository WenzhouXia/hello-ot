from __future__ import annotations

import numpy as np
from numba import boolean, float64, int64, njit
from numba.experimental import jitclass


_FOREST_SPEC = [
    ("child", int64[:, :]),
    ("parent", int64[:]),
    ("edge_count", int64[:]),
    ("reverse", boolean[:]),
    ("is_edge", boolean[:]),
    ("has_lazy_add", boolean[:]),
    ("value_array", float64[:]),
    ("cost", float64[:]),
    ("minimum", float64[:, :]),
    ("minimum_node", int64[:, :]),
    ("cost_sum", float64[:, :]),
    ("lazy_add", float64[:, :]),
    ("ancestors", int64[:]),
    ("traversal", int64[:]),
]


@jitclass(_FOREST_SPEC)
class _LinkCutForest:
    """
    CN: 用数组保存的 Numba link-cut forest，与正式 C++ 实现共享同一 pivot 规则。
    EN: Array-backed Numba link-cut forest using the same pivot rule as the formal C++ implementation.
    """

    def __init__(self, size: int) -> None:
        array_size = int(size) + 1
        self.child = np.zeros((array_size, 2), dtype=np.int64)
        self.parent = np.zeros(array_size, dtype=np.int64)
        self.edge_count = np.zeros(array_size, dtype=np.int64)
        self.reverse = np.zeros(array_size, dtype=np.bool_)
        self.is_edge = np.zeros(array_size, dtype=np.bool_)
        self.has_lazy_add = np.zeros(array_size, dtype=np.bool_)
        self.value_array = np.zeros(array_size, dtype=np.float64)
        self.cost = np.zeros(array_size, dtype=np.float64)
        self.minimum = np.full((array_size, 2), np.inf, dtype=np.float64)
        self.minimum_node = np.zeros((array_size, 2), dtype=np.int64)
        self.cost_sum = np.zeros((array_size, 2), dtype=np.float64)
        self.lazy_add = np.zeros((array_size, 2), dtype=np.float64)
        self.ancestors = np.empty(array_size, dtype=np.int64)
        self.traversal = np.empty(array_size, dtype=np.int64)

    def _is_auxiliary_root(self, node: int) -> bool:
        parent = self.parent[node]
        return parent == 0 or (
            self.child[parent, 0] != node and self.child[parent, 1] != node
        )

    def _append_aggregate(self, destination: int, segment: int, offset: int) -> None:
        for residue in range(2):
            shifted = (offset + residue) & 1
            self.cost_sum[destination, shifted] += self.cost_sum[segment, residue]
            if self.minimum[segment, residue] < self.minimum[destination, shifted]:
                self.minimum[destination, shifted] = self.minimum[segment, residue]
                self.minimum_node[destination, shifted] = self.minimum_node[segment, residue]

    def _pull(self, node: int) -> None:
        if node == 0:
            return
        left = self.child[node, 0]
        right = self.child[node, 1]
        left_count = 0 if left == 0 else self.edge_count[left]
        right_count = 0 if right == 0 else self.edge_count[right]
        self.edge_count[node] = left_count + (1 if self.is_edge[node] else 0) + right_count
        self.minimum[node, 0] = np.inf
        self.minimum[node, 1] = np.inf
        self.minimum_node[node, 0] = 0
        self.minimum_node[node, 1] = 0
        self.cost_sum[node, 0] = 0.0
        self.cost_sum[node, 1] = 0.0
        if left != 0:
            self._append_aggregate(node, left, 0)
        self_residue = left_count & 1
        if self.is_edge[node]:
            if self.value_array[node] < self.minimum[node, self_residue]:
                self.minimum[node, self_residue] = self.value_array[node]
                self.minimum_node[node, self_residue] = node
            self.cost_sum[node, self_residue] += self.cost[node]
        if right != 0:
            self._append_aggregate(node, right, left_count + (1 if self.is_edge[node] else 0))

    def _apply_add(self, node: int, first: float, second: float) -> None:
        if node == 0 or (first == 0.0 and second == 0.0):
            return
        self.has_lazy_add[node] = True
        additions = (first, second)
        for residue in range(2):
            addition = additions[residue]
            if self.minimum[node, residue] < np.inf:
                self.minimum[node, residue] += addition
            self.lazy_add[node, residue] += addition
        if self.is_edge[node]:
            left = self.child[node, 0]
            left_count = 0 if left == 0 else self.edge_count[left]
            self.value_array[node] += additions[left_count & 1]

    def _apply_reverse(self, node: int) -> None:
        if node == 0:
            return
        old_left = self.child[node, 0]
        self.child[node, 0] = self.child[node, 1]
        self.child[node, 1] = old_left
        old_minimum_0 = self.minimum[node, 0]
        old_minimum_1 = self.minimum[node, 1]
        old_minimum_node_0 = self.minimum_node[node, 0]
        old_minimum_node_1 = self.minimum_node[node, 1]
        old_cost_sum_0 = self.cost_sum[node, 0]
        old_cost_sum_1 = self.cost_sum[node, 1]
        old_lazy_add_0 = self.lazy_add[node, 0]
        old_lazy_add_1 = self.lazy_add[node, 1]
        parity = (self.edge_count[node] - 1) & 1
        if parity == 0:
            self.minimum[node, 0] = old_minimum_0
            self.minimum[node, 1] = old_minimum_1
            self.minimum_node[node, 0] = old_minimum_node_0
            self.minimum_node[node, 1] = old_minimum_node_1
            self.cost_sum[node, 0] = old_cost_sum_0
            self.cost_sum[node, 1] = old_cost_sum_1
            self.lazy_add[node, 0] = old_lazy_add_0
            self.lazy_add[node, 1] = old_lazy_add_1
        else:
            self.minimum[node, 0] = old_minimum_1
            self.minimum[node, 1] = old_minimum_0
            self.minimum_node[node, 0] = old_minimum_node_1
            self.minimum_node[node, 1] = old_minimum_node_0
            self.cost_sum[node, 0] = old_cost_sum_1
            self.cost_sum[node, 1] = old_cost_sum_0
            self.lazy_add[node, 0] = old_lazy_add_1
            self.lazy_add[node, 1] = old_lazy_add_0
        self.reverse[node] = not self.reverse[node]

    def _push(self, node: int) -> None:
        if node == 0:
            return
        if self.reverse[node]:
            self._apply_reverse(self.child[node, 0])
            self._apply_reverse(self.child[node, 1])
            self.reverse[node] = False
        if not self.has_lazy_add[node]:
            return
        first = self.lazy_add[node, 0]
        second = self.lazy_add[node, 1]
        left = self.child[node, 0]
        right = self.child[node, 1]
        if left != 0:
            self._apply_add(left, first, second)
        if right != 0:
            left_count = 0 if left == 0 else self.edge_count[left]
            offset = left_count + (1 if self.is_edge[node] else 0)
            if (offset & 1) == 0:
                self._apply_add(right, first, second)
            else:
                self._apply_add(right, second, first)
        self.lazy_add[node, 0] = 0.0
        self.lazy_add[node, 1] = 0.0
        self.has_lazy_add[node] = False

    def _rotate(self, node: int) -> None:
        parent = self.parent[node]
        grandparent = self.parent[parent]
        direction = 1 if self.child[parent, 1] == node else 0
        middle = self.child[node, direction ^ 1]
        if not self._is_auxiliary_root(parent):
            parent_direction = 1 if self.child[grandparent, 1] == parent else 0
            self.child[grandparent, parent_direction] = node
        self.parent[node] = grandparent
        self.child[node, direction ^ 1] = parent
        self.parent[parent] = node
        self.child[parent, direction] = middle
        if middle != 0:
            self.parent[middle] = parent
        self._pull(parent)
        self._pull(node)

    def _splay(self, node: int) -> None:
        count = 1
        self.ancestors[0] = node
        current = node
        while not self._is_auxiliary_root(current):
            current = self.parent[current]
            self.ancestors[count] = current
            count += 1
        for position in range(count - 1, -1, -1):
            self._push(self.ancestors[position])
        while not self._is_auxiliary_root(node):
            parent = self.parent[node]
            grandparent = self.parent[parent]
            if not self._is_auxiliary_root(parent):
                zig_zig = (self.child[parent, 0] == node) == (
                    self.child[grandparent, 0] == parent
                )
                self._rotate(parent if zig_zig else node)
            self._rotate(node)

    def _access(self, node: int) -> None:
        previous = 0
        current = node
        while current != 0:
            self._splay(current)
            self.child[current, 1] = previous
            if previous != 0:
                self.parent[previous] = current
            self._pull(current)
            previous = current
            current = self.parent[current]
        self._splay(node)

    def initialize_vertex(self, node: int) -> None:
        self.is_edge[node] = False
        self._pull(node)

    def initialize_edge(self, node: int, value: float, cost: float) -> None:
        self.child[node, 0] = 0
        self.child[node, 1] = 0
        self.parent[node] = 0
        self.edge_count[node] = 0
        self.reverse[node] = False
        self.is_edge[node] = True
        self.has_lazy_add[node] = False
        self.value_array[node] = value
        self.cost[node] = cost
        self.minimum[node, 0] = np.inf
        self.minimum[node, 1] = np.inf
        self.minimum_node[node, 0] = 0
        self.minimum_node[node, 1] = 0
        self.cost_sum[node, 0] = 0.0
        self.cost_sum[node, 1] = 0.0
        self.lazy_add[node, 0] = 0.0
        self.lazy_add[node, 1] = 0.0
        self._pull(node)

    def make_root(self, node: int) -> None:
        self._access(node)
        self._apply_reverse(node)

    def link(self, first: int, second: int) -> None:
        self.make_root(first)
        self.parent[first] = second

    def link_edge_between(self, first: int, edge: int, second: int) -> None:
        self.make_root(first)
        self.parent[first] = edge
        self.parent[edge] = second

    def detach_exposed_edge(self, edge: int) -> bool:
        self._splay(edge)
        first_side = self.child[edge, 0]
        second_side = self.child[edge, 1]
        if first_side == 0 or second_side == 0:
            return False
        self.parent[first_side] = 0
        self.parent[second_side] = 0
        self.child[edge, 0] = 0
        self.child[edge, 1] = 0
        self._pull(edge)
        return True

    def expose_path(self, first: int, second: int) -> int:
        self.make_root(first)
        self._access(second)
        return second

    def add_by_residue(self, root: int, first: float, second: float) -> None:
        self._apply_add(root, first, second)

    def flush_lazy_tags(self) -> None:
        node_count = self.parent.size
        for node in range(1, node_count):
            if not self._is_auxiliary_root(node):
                continue
            count = 1
            self.traversal[0] = node
            while count > 0:
                count -= 1
                current = self.traversal[count]
                self._push(current)
                left = self.child[current, 0]
                right = self.child[current, 1]
                if left != 0:
                    self.traversal[count] = left
                    count += 1
                if right != 0:
                    self.traversal[count] = right
                    count += 1


@njit(cache=True)
def _find(parent: np.ndarray, value: int) -> int:
    """
    CN: 在 Numba DSU 中执行路径压缩查找。
    EN: Find a Numba DSU root with path compression.
    """
    root = value
    while parent[root] != root:
        root = parent[root]
    while parent[value] != value:
        next_value = parent[value]
        parent[value] = root
        value = next_value
    return root


@njit(cache=True)
def _unite(parent: np.ndarray, rank: np.ndarray, first: int, second: int) -> bool:
    """
    CN: 合并两个 Numba DSU 分量。
    EN: Unite two Numba DSU components.
    """
    first = _find(parent, first)
    second = _find(parent, second)
    if first == second:
        return False
    if rank[first] < rank[second]:
        first, second = second, first
    parent[second] = first
    if rank[first] == rank[second]:
        rank[first] += 1
    return True


@njit(cache=True)
def _sparsify_transport_support_numba_impl(
    n_source: int,
    n_target: int,
    rows: np.ndarray,
    cols: np.ndarray,
    input_values: np.ndarray,
    costs: np.ndarray,
    target_nnz: int,
    zero_tolerance: float,
) -> tuple[np.ndarray, int, int, int, int]:
    """
    CN: 用 Numba link-cut tree 执行与 C++ 路径等价的 cycle cancellation。
    EN: Run cycle cancellation with a Numba link-cut tree equivalent to the C++ path.
    """
    values = input_values.copy()
    edge_count = rows.size
    vertex_count = n_source + n_target
    first_vertex = np.empty(edge_count, dtype=np.int64)
    second_vertex = np.empty(edge_count, dtype=np.int64)
    tree_edges = np.empty(min(edge_count, vertex_count - 1), dtype=np.int64)
    non_tree_edges = np.empty(edge_count, dtype=np.int64)
    tree_count = 0
    non_tree_count = 0
    parent = np.arange(vertex_count, dtype=np.int64)
    rank = np.zeros(vertex_count, dtype=np.int64)
    initial_nnz = 0

    for edge in range(edge_count):
        if values[edge] <= zero_tolerance:
            values[edge] = 0.0
        else:
            initial_nnz += 1
        source_vertex = rows[edge] + 1
        target_vertex = n_source + cols[edge] + 1
        first_vertex[edge] = source_vertex
        second_vertex[edge] = target_vertex
        if values[edge] <= zero_tolerance:
            continue
        if _unite(parent, rank, source_vertex - 1, target_vertex - 1):
            tree_edges[tree_count] = edge
            tree_count += 1
        else:
            non_tree_edges[non_tree_count] = edge
            non_tree_count += 1

    forest = _LinkCutForest(vertex_count + tree_count)
    for vertex in range(vertex_count):
        forest.initialize_vertex(vertex + 1)
    edge_for_tree_slot = np.empty(tree_count, dtype=np.int64)
    for slot in range(tree_count):
        edge = tree_edges[slot]
        edge_node = vertex_count + slot + 1
        edge_for_tree_slot[slot] = edge
        forest.initialize_edge(edge_node, values[edge], costs[edge])
        forest.link(first_vertex[edge], edge_node)
        forest.link(edge_node, second_vertex[edge])

    cycle_updates = 0
    tree_operations = 0
    next_support_check = max(0, initial_nnz - target_nnz)
    entering_position = 0
    while entering_position < non_tree_count:
        if cycle_updates >= next_support_check:
            forest.flush_lazy_tags()
            tree_nnz = 0
            for slot in range(tree_count):
                edge_node = vertex_count + slot + 1
                tree_nnz += 1 if forest.value_array[edge_node] > zero_tolerance else 0
            exact_nnz = tree_nnz + (non_tree_count - entering_position)
            if exact_nnz <= target_nnz:
                break
            next_support_check = cycle_updates + (exact_nnz - target_nnz)

        entering_edge = non_tree_edges[entering_position]
        entering_position += 1
        entering_value = values[entering_edge]
        if entering_value <= zero_tolerance:
            continue
        path_root = forest.expose_path(
            first_vertex[entering_edge],
            second_vertex[entering_edge],
        )
        tree_operations += 2
        directional_cost = (
            costs[entering_edge]
            - forest.cost_sum[path_root, 0]
            + forest.cost_sum[path_root, 1]
        )
        orientation = 1.0 if directional_cost <= 0.0 else -1.0
        decreasing_residue = 0 if orientation > 0.0 else 1
        step = forest.minimum[path_root, decreasing_residue]
        leaving_node = forest.minimum_node[path_root, decreasing_residue]
        if orientation < 0.0:
            step = min(step, entering_value)
        if not np.isfinite(step):
            return values, 0, cycle_updates, tree_operations, 3
        if step < -zero_tolerance:
            return values, 0, cycle_updates, tree_operations, 6
        step = max(0.0, step)
        forest.add_by_residue(path_root, -orientation * step, orientation * step)
        entering_value += orientation * step
        if entering_value <= zero_tolerance:
            entering_value = 0.0

        if entering_value > zero_tolerance:
            if leaving_node <= vertex_count:
                return values, 0, cycle_updates, tree_operations, 4
            leaving_slot = leaving_node - vertex_count - 1
            leaving_edge = edge_for_tree_slot[leaving_slot]
            if not forest.detach_exposed_edge(leaving_node):
                return values, 0, cycle_updates, tree_operations, 7
            values[leaving_edge] = 0.0
            forest.initialize_edge(leaving_node, entering_value, costs[entering_edge])
            edge_for_tree_slot[leaving_slot] = entering_edge
            forest.link_edge_between(
                first_vertex[entering_edge],
                leaving_node,
                second_vertex[entering_edge],
            )
            tree_operations += 4
        else:
            values[entering_edge] = 0.0
        cycle_updates += 1

    forest.flush_lazy_tags()
    for slot in range(tree_count):
        edge_node = vertex_count + slot + 1
        edge = edge_for_tree_slot[slot]
        materialized = forest.value_array[edge_node]
        if not np.isfinite(materialized) or materialized < -zero_tolerance:
            return values, 0, cycle_updates, tree_operations, 5
        values[edge] = materialized if materialized > zero_tolerance else 0.0

    output_nnz = 0
    for edge in range(edge_count):
        if not np.isfinite(values[edge]) or values[edge] < -zero_tolerance:
            return values, 0, cycle_updates, tree_operations, 5
        values[edge] = values[edge] if values[edge] > zero_tolerance else 0.0
        output_nnz += 1 if values[edge] > zero_tolerance else 0
    if output_nnz > target_nnz:
        return values, output_nnz, cycle_updates, tree_operations, 9
    return values, output_nnz, cycle_updates, tree_operations, 0


def sparsify_transport_support_numba(
    n_source: int,
    n_target: int,
    rows: np.ndarray,
    cols: np.ndarray,
    values: np.ndarray,
    costs: np.ndarray,
    target_nnz: int,
    zero_tolerance: float,
) -> dict[str, object]:
    """
    CN: 调用 Numba LCT 并将状态码转换为正式 sparsifier 的返回结构。
    EN: Invoke the Numba LCT and convert its status into the formal sparsifier result shape.
    """
    output, output_nnz, cycle_updates, tree_operations, status = (
        _sparsify_transport_support_numba_impl(
            int(n_source),
            int(n_target),
            rows,
            cols,
            values,
            costs,
            int(target_nnz),
            float(zero_tolerance),
        )
    )
    if int(status) != 0:
        raise RuntimeError(f"Numba support sparsifier failed with status={int(status)}")
    return {
        "values": output,
        "output_nnz": int(output_nnz),
        "cycle_updates": int(cycle_updates),
        "tree_operations": int(tree_operations),
    }


__all__ = ["sparsify_transport_support_numba"]
