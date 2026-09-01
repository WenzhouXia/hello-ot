from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional, Tuple

import numpy as np
import torch


STRUCTURAL_EDGE = -1
INITIAL_EDGE = -2


class ActiveSupport:
    """
    CN: HELLO refinement 使用的 Torch-device active support。
    EN: Torch-device active support used by HELLO refinement.

    CN: 该类型只维护数据、不决定 dual violation、判敛或 budgeted pruning。
    EN: This type only maintains data; it does not decide dual violations, stopping, or budgeted pruning.
    """

    def __init__(
        self,
        *,
        n_source: int,
        n_target: int,
        level_cache: Optional[Dict[str, Any]] = None,
        device: Any = "cuda",
        track_creation: bool = True,
    ) -> None:
        self.n_source = int(n_source)
        self.n_target = int(n_target)
        # CN: 低秩 cost SDDMM 仍通过该只读视图访问本层问题数据；后续迁移 kernel 时再移除。
        # EN: Low-rank cost SDDMM still reads level data through this read-only view; remove it with the later kernel migration.
        self.level_cache = {} if level_cache is None else level_cache
        if self.n_source < 1 or self.n_target < 1:
            raise ValueError("n_source and n_target must be positive")
        self.device = str(torch.device(device))
        if torch.device(self.device).type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("HELLO ActiveSupport requested CUDA, but torch CUDA is unavailable.")
        self.backend = "torch"
        self.track_creation = bool(track_creation)
        self.rows = torch.empty(0, dtype=torch.int32, device=self.device)
        self.cols = torch.empty(0, dtype=torch.int32, device=self.device)
        self.c_vec = torch.empty(0, dtype=torch.float64, device=self.device)
        self.x_prev = torch.empty(0, dtype=torch.float64, device=self.device)
        self.keys = torch.empty(0, dtype=torch.int64, device=self.device)
        self.creation_iteration = (
            torch.empty(0, dtype=torch.int32, device=self.device)
            if self.track_creation
            else None
        )
        self._order_cache: Optional[Dict[str, Any]] = None
        self._order_cache_dirty = True

    @property
    def size(self) -> int:
        return int(self.rows.numel())

    @property
    def is_torch_backend(self) -> bool:
        return True

    @property
    def is_cuda(self) -> bool:
        return torch.device(self.device).type == "cuda"

    @staticmethod
    def _array_size(value: Any) -> int:
        if torch.is_tensor(value):
            return int(value.numel())
        return int(np.asarray(value).size)

    @staticmethod
    def _torch_dtype(dtype: Any) -> torch.dtype:
        mapping = {
            np.dtype(np.int32): torch.int32,
            np.dtype(np.int64): torch.int64,
            np.dtype(np.float32): torch.float32,
            np.dtype(np.float64): torch.float64,
            np.dtype(np.bool_): torch.bool,
        }
        try:
            return mapping[np.dtype(dtype)]
        except KeyError as exc:
            raise TypeError(f"unsupported ActiveSupport dtype={dtype!r}") from exc

    def _coerce_array(self, value: Any, dtype: Any) -> torch.Tensor:
        target_dtype = self._torch_dtype(dtype)
        if torch.is_tensor(value):
            return value.detach().to(device=self.device, dtype=target_dtype).contiguous().view(-1)
        return torch.as_tensor(value, dtype=target_dtype, device=self.device).contiguous().view(-1)

    def _zeros(self, size: int, dtype: Any) -> torch.Tensor:
        return torch.zeros(int(size), dtype=self._torch_dtype(dtype), device=self.device)

    def initialize_from_arrays(
        self,
        *,
        rows: Any,
        cols: Any,
        costs: Any,
        x_prev: Any,
        keys: Optional[Any] = None,
        creation_iteration: Optional[Any] = None,
    ) -> None:
        """
        CN: 一次性初始化 support arrays；同设备、同 dtype 的连续 tensor 直接接管 storage。
        EN: Initialize all support arrays at once, adopting contiguous same-device/same-dtype tensor storage.
        """
        rows_t = self._coerce_array(rows, np.int32)
        cols_t = self._coerce_array(cols, np.int32)
        costs_t = self._coerce_array(costs, np.float64)
        x_prev_t = self._coerce_array(x_prev, np.float64)
        size = int(rows_t.numel())
        if not (int(cols_t.numel()) == int(costs_t.numel()) == int(x_prev_t.numel()) == size):
            raise ValueError("rows, cols, costs, and x_prev must have equal lengths")
        if bool(torch.any(rows_t < 0)) or bool(torch.any(rows_t >= self.n_source)):
            raise ValueError("source index outside ActiveSupport shape")
        if bool(torch.any(cols_t < 0)) or bool(torch.any(cols_t >= self.n_target)):
            raise ValueError("target index outside ActiveSupport shape")
        if keys is None:
            keys_t = rows_t.to(torch.int64) * self.n_target + cols_t.to(torch.int64)
        else:
            keys_t = self._coerce_array(keys, np.int64)
            if int(keys_t.numel()) != size:
                raise ValueError("keys length must match active support size")
        if self.track_creation:
            if creation_iteration is None:
                creation_t = torch.full_like(rows_t, INITIAL_EDGE, dtype=torch.int32)
            else:
                creation_t = self._coerce_array(creation_iteration, np.int32)
                if int(creation_t.numel()) != size:
                    raise ValueError("creation_iteration length must match active support size")
        else:
            creation_t = None
        self.rows = rows_t
        self.cols = cols_t
        self.c_vec = costs_t
        self.x_prev = x_prev_t
        self.keys = keys_t
        self.creation_iteration = creation_t
        self.invalidate_order_cache()

    @staticmethod
    def _to_numpy(value: Any, dtype: Any) -> np.ndarray:
        if torch.is_tensor(value):
            return value.detach().cpu().numpy().astype(dtype, copy=False)
        return np.asarray(value, dtype=dtype)

    def invalidate_order_cache(self) -> None:
        self._order_cache = None
        self._order_cache_dirty = True

    def clear_order_cache(self) -> None:
        self.invalidate_order_cache()

    @contextmanager
    def stage_storage_on_cpu(self) -> Iterator[Dict[str, Any]]:
        """
        CN: 在不需要 active support 的 GPU 阶段临时把持久字段迁到 CPU，并在退出时恢复。
        EN: Temporarily stage persistent fields on CPU while a GPU phase does not need active support, then restore them on exit.
        """
        field_names = ("rows", "cols", "c_vec", "x_prev", "keys")
        if self.creation_iteration is not None:
            field_names = (*field_names, "creation_iteration")
        fields = {name: getattr(self, name) for name in field_names}
        if any(not torch.is_tensor(value) or value.device.type != "cuda" for value in fields.values()):
            raise RuntimeError("ActiveSupport CPU staging requires all persistent fields to be CUDA tensors")
        staged_bytes = int(sum(value.numel() * value.element_size() for value in fields.values()))
        # CN: LP order cache 在 scan 中不会使用；直接释放，恢复后需要时再构造。
        # EN: The LP order cache is unused by the scan; release it and rebuild lazily after restoration.
        self.clear_order_cache()
        stage_started = time.perf_counter()
        cpu_fields = {
            name: value.detach().to(device="cpu", non_blocking=False).contiguous()
            for name, value in fields.items()
        }
        for name, value in cpu_fields.items():
            setattr(self, name, value)
        del fields
        profile: Dict[str, Any] = {
            "active_support_staged_bytes": int(staged_bytes),
            "active_support_stage_to_cpu_time_sec": float(time.perf_counter() - stage_started),
            "active_support_restore_to_cuda_time_sec": 0.0,
        }
        try:
            yield profile
        finally:
            restore_started = time.perf_counter()
            restored = {
                name: value.to(device=self.device, non_blocking=False).contiguous()
                for name, value in cpu_fields.items()
            }
            for name, value in restored.items():
                setattr(self, name, value)
            profile["active_support_restore_to_cuda_time_sec"] = float(
                time.perf_counter() - restore_started
            )

    def get_order_cache(
        self,
        n_source: Optional[int] = None,
        n_target: Optional[int] = None,
        device: Optional[Any] = None,
    ) -> Dict[str, Any]:
        n_source_i = self.n_source if n_source is None else int(n_source)
        n_target_i = self.n_target if n_target is None else int(n_target)
        target_device = self.device if device is None else str(torch.device(device))
        cache = self._order_cache
        if (
            cache is not None
            and not self._order_cache_dirty
            and int(cache["n_vars"]) == self.size
            and int(cache["n_source"]) == n_source_i
            and int(cache["n_target"]) == n_target_i
            and str(cache["device"]) == target_device
        ):
            return cache
        rows = self.rows.to(device=target_device, dtype=torch.int64)
        cols = self.cols.to(device=target_device, dtype=torch.int64)
        dtype = torch.int32 if max(self.size, n_source_i, n_target_i) <= np.iinfo(np.int32).max else torch.int64
        source_order = torch.argsort(rows, stable=True).to(dtype=dtype)
        target_order = torch.argsort(cols, stable=True).to(dtype=dtype)
        source_counts = torch.bincount(rows, minlength=n_source_i).to(dtype=dtype)
        target_counts = torch.bincount(cols, minlength=n_target_i).to(dtype=dtype)
        source_rowptr = torch.empty(n_source_i + 1, dtype=dtype, device=target_device)
        target_rowptr = torch.empty(n_target_i + 1, dtype=dtype, device=target_device)
        source_rowptr[0] = 0
        target_rowptr[0] = 0
        source_rowptr[1:] = torch.cumsum(source_counts, dim=0, dtype=torch.int64).to(dtype=dtype)
        target_rowptr[1:] = torch.cumsum(target_counts, dim=0, dtype=torch.int64).to(dtype=dtype)
        cache = {
            "n_vars": self.size,
            "n_source": n_source_i,
            "n_target": n_target_i,
            "device": target_device,
            "source_order": source_order,
            "target_order": target_order,
            "source_counts": source_counts,
            "target_counts": target_counts,
            "source_rowptr": source_rowptr,
            "target_rowptr": target_rowptr,
            "source_nonempty": torch.nonzero(source_counts > 0, as_tuple=False).flatten().to(dtype=dtype),
            "target_nonempty": torch.nonzero(target_counts > 0, as_tuple=False).flatten().to(dtype=dtype),
        }
        # CN: 当前低秩流式 SDDMM 仍在 CPU 上组织 source blocks；保留与旧实现相同的只读排序视图。
        # EN: The current streamed low-rank SDDMM still organizes source blocks on CPU; retain the same read-only ordering views as before.
        np_dtype = np.int32 if dtype == torch.int32 else np.int64
        cache.update(
            {
                "source_order_cpu": source_order.detach().cpu().numpy().astype(np_dtype, copy=False),
                "target_order_cpu": target_order.detach().cpu().numpy().astype(np_dtype, copy=False),
                "source_rowptr_cpu": source_rowptr.detach().cpu().numpy().astype(np_dtype, copy=False),
                "target_rowptr_cpu": target_rowptr.detach().cpu().numpy().astype(np_dtype, copy=False),
                "source_nonempty_cpu": cache["source_nonempty"].detach().cpu().numpy().astype(np_dtype, copy=False),
                "target_nonempty_cpu": cache["target_nonempty"].detach().cpu().numpy().astype(np_dtype, copy=False),
            }
        )
        self._order_cache = cache
        self._order_cache_dirty = False
        return cache

    def add_pairs(self, rows: Any, cols: Any, costs: Any, iter_idx: int = INITIAL_EDGE) -> None:
        rows_add = self._coerce_array(rows, np.int32)
        if int(rows_add.numel()) == 0:
            return
        cols_add = self._coerce_array(cols, np.int32)
        costs_add = self._coerce_array(costs, np.float64)
        if not (rows_add.numel() == cols_add.numel() == costs_add.numel()):
            raise ValueError("rows, cols, and costs must have equal lengths")
        if bool(torch.any(rows_add < 0)) or bool(torch.any(rows_add >= self.n_source)):
            raise ValueError("source index outside ActiveSupport shape")
        if bool(torch.any(cols_add < 0)) or bool(torch.any(cols_add >= self.n_target)):
            raise ValueError("target index outside ActiveSupport shape")
        self.rows = torch.cat((self.rows, rows_add))
        self.cols = torch.cat((self.cols, cols_add))
        self.c_vec = torch.cat((self.c_vec, costs_add))
        self.x_prev = torch.cat((self.x_prev, torch.zeros_like(costs_add)))
        keys_add = rows_add.to(torch.int64) * self.n_target + cols_add.to(torch.int64)
        self.keys = torch.cat((self.keys, keys_add))
        if self.creation_iteration is not None:
            created = torch.full_like(rows_add, int(iter_idx), dtype=torch.int32)
            self.creation_iteration = torch.cat((self.creation_iteration, created))
        self.invalidate_order_cache()

    def add_pairs_placeholder(self, rows: Any, cols: Any, iter_idx: int = INITIAL_EDGE) -> None:
        size = self._array_size(rows)
        self.add_pairs(rows, cols, self._zeros(size, np.float64), iter_idx=iter_idx)

    def replace_costs(self, costs: Any) -> None:
        values = self._coerce_array(costs, np.float64)
        if int(values.numel()) != self.size:
            raise ValueError("replacement cost vector length must match active support size")
        self.c_vec = values

    def set_x_prev(self, values: Any) -> None:
        primal = self._coerce_array(values, np.float64)
        if int(primal.numel()) != self.size:
            raise ValueError("primal vector length must match active support size")
        self.x_prev = primal

    def prune(self, mask: Any) -> None:
        keep = self._coerce_array(mask, np.bool_)
        if int(keep.numel()) != self.size:
            raise ValueError("pruning mask length must match active support size")
        self.rows = self.rows[keep]
        self.cols = self.cols[keep]
        self.c_vec = self.c_vec[keep]
        self.x_prev = self.x_prev[keep]
        self.keys = self.keys[keep]
        if self.creation_iteration is not None:
            self.creation_iteration = self.creation_iteration[keep]
        self.invalidate_order_cache()

    def to(self, device: Any) -> "ActiveSupport":
        target = str(torch.device(device))
        if not target.startswith("cuda"):
            raise ValueError("HELLO ActiveSupport cannot migrate to CPU")
        if target == self.device:
            return self
        migrated = ActiveSupport(
            n_source=self.n_source,
            n_target=self.n_target,
            level_cache=self.level_cache,
            device=target,
            track_creation=self.track_creation,
        )
        migrated.rows = self.rows.to(target)
        migrated.cols = self.cols.to(target)
        migrated.c_vec = self.c_vec.to(target)
        migrated.x_prev = self.x_prev.to(target)
        migrated.keys = self.keys.to(target)
        if self.creation_iteration is not None:
            migrated.creation_iteration = self.creation_iteration.to(target)
        return migrated

    def export_numpy(self) -> Dict[str, np.ndarray]:
        payload = {
            "rows": self._to_numpy(self.rows, np.int32),
            "cols": self._to_numpy(self.cols, np.int32),
            "c_vec": self._to_numpy(self.c_vec, np.float64),
            "x_prev": self._to_numpy(self.x_prev, np.float64),
            "keys": self._to_numpy(self.keys, np.int64),
        }
        if self.creation_iteration is not None:
            payload["creation_iteration"] = self._to_numpy(self.creation_iteration, np.int32)
        return payload

    def export_sparse_coupling_numpy(self, threshold: float = 1e-12) -> Dict[str, np.ndarray]:
        mask = self.x_prev > float(threshold)
        return {
            "rows": self._to_numpy(self.rows[mask], np.int32),
            "cols": self._to_numpy(self.cols[mask], np.int32),
            "values": self._to_numpy(self.x_prev[mask], np.float64),
        }

    def sorted_keys_with_positions_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        keys = self._to_numpy(self.keys, np.int64)
        order = np.argsort(keys)
        return keys[order], order.astype(np.int64, copy=False)

    def filter_new_candidate_pairs(self, rows: Any, cols: Any) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        candidate_rows = self._coerce_array(rows, np.int32)
        candidate_cols = self._coerce_array(cols, np.int32)
        found = int(candidate_rows.numel())
        if found == 0:
            return candidate_rows, candidate_cols, 0, 0
        candidate_keys = torch.unique(
            candidate_rows.to(torch.int64) * self.n_target + candidate_cols.to(torch.int64),
            sorted=True,
        )
        if self.size == 0:
            new_keys = candidate_keys
        else:
            active_keys = torch.sort(self.keys).values
            positions = torch.searchsorted(active_keys, candidate_keys)
            membership = positions < active_keys.numel()
            if bool(membership.any()):
                bounded = positions.clamp_max(active_keys.numel() - 1)
                membership &= active_keys[bounded] == candidate_keys
            new_keys = candidate_keys[~membership]
        new_rows = torch.div(new_keys, self.n_target, rounding_mode="floor").to(torch.int32)
        new_cols = torch.remainder(new_keys, self.n_target).to(torch.int32)
        return new_rows, new_cols, found, int(new_keys.numel())

    def __getitem__(self, item: str) -> Any:
        if hasattr(self, item):
            return getattr(self, item)
        raise KeyError(f"ActiveSupport has no attribute {item}")


__all__ = ["ActiveSupport", "INITIAL_EDGE", "STRUCTURAL_EDGE"]
