from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import fields
from typing import Any


class RecordMapping(Mapping[str, Any]):
    """
    CN: 为 frozen variant record 保留旧的只读 Mapping 接口。
    EN: Preserve the legacy read-only Mapping interface for frozen variant records.
    """

    def __getitem__(self, key: str) -> Any:
        names = tuple(item.name for item in fields(self))
        if key not in names:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return (item.name for item in fields(self))

    def __len__(self) -> int:
        return len(fields(self))


__all__ = ["RecordMapping"]
