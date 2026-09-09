"""CN: 汇总公开 Pareto 的最小结果表。EN: Collect a minimal public Pareto result table."""

import argparse
import csv
import json
from pathlib import Path

from .protocol import FIELDS, METHODS


def collect(output_dir):
    """CN: 保留成功与失败项，排除请求和参考值文件。EN: Retain successes and failures, excluding requests and reference files."""
    rows = []
    for path in sorted(Path(output_dir).glob("n*_d*/*.json")):
        if path.name.endswith(".request.json") or path.name.startswith("reference_"):
            continue
        record = json.loads(path.read_text())
        if record.get("method") in METHODS:
            rows.append({key: record.get(key) for key in FIELDS})
    with (Path(output_dir) / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    failures = [r for r in rows if r["status"] != "success"]
    text = f"# Public Pareto\n\n{len(rows) - len(failures)}/{len(rows)} configurations succeeded. Seed=42.\n"
    if failures:
        text += "\nFailed configurations:\n\n" + "\n".join(
            f"- N={r['n']} D={r['d']} {r['method']} ({r['parameter']}): {r['error']}" for r in failures
        ) + "\n"
    (Path(output_dir) / "summary.md").write_text(text)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("results/accuracy_runtime_pareto"))
    collect(parser.parse_args().output_dir)
