def normalize_cost_type_name(cost_type: str) -> str:
    """
    CN: 规范化 HELLO 支持的代价名称。
    EN: Normalize a HELLO cost name.
    """
    normalized = str(cost_type).strip().lower()
    aliases = {
        "euclidean": "l2",
        "l_inf": "linf",
        "l2sq": "l2^2",
        "sq_euclidean": "l2^2",
        "sqeuclidean": "l2^2",
        "squared_euclidean": "l2^2",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"l2^2", "l2", "l1", "linf", "lowrank"}:
        raise ValueError(f"unsupported cost_type={cost_type!r}")
    return normalized
