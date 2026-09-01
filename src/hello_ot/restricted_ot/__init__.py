"""CN: HELLO restricted OT 数据、SolveLP 与 support 稀疏化。EN: HELLO restricted-OT data, SolveLP, and support sparsification."""

from hello_ot.restricted_ot.support_sparsification import (
    SupportSparsificationResult,
    sparsify_transport_support,
)

__all__ = ["SupportSparsificationResult", "sparsify_transport_support"]
