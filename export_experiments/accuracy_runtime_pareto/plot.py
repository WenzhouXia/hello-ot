"""CN: 绘制公开 Pareto 时间与误差。EN: Plot public Pareto runtime and error."""

import argparse
from pathlib import Path

from .collect import collect
from .protocol import METHODS


def plot(output_dir):
    """CN: 每个 N/D 输出一张图，零误差使用 symlog 显示。EN: Plot each N/D separately, displaying zero error with symlog."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = collect(output_dir)
    for n, d in sorted({(row["n"], row["d"]) for row in rows}):
        fig, ax = plt.subplots()
        count = 0
        for method in METHODS:
            points = sorted(
                (r["relative_error"], r["runtime_sec"]) for r in rows
                if r["n"] == n and r["d"] == d and r["method"] == method and r["status"] == "success"
            )
            if points:
                ax.plot(*zip(*points), marker="o", label={"hello": "HELLO", "sinkhorn": "Sinkhorn",
                        "ipot": "IPOT", "mdot": "MDOT-TNT"}[method])
                count += 1
        ax.set_xscale("symlog", linthresh=1e-8)
        ax.set_yscale("log")
        ax.set(xlabel="Relative objective error", ylabel="Solver time (s)",
               title=f"Gaussian-to-ImageNet, N={n}, D={d}, seed=42")
        if count:
            ax.legend()
        else:
            ax.text(0.5, 0.5, "No valid results", transform=ax.transAxes, ha="center")
        failed = sum(r["status"] != "success" for r in rows if r["n"] == n and r["d"] == d)
        fig.text(0.02, 0.01, f"Failed configurations: {failed}")
        fig.tight_layout(rect=(0, .04, 1, 1))
        for suffix in ("pdf", "png"):
            fig.savefig(Path(output_dir) / f"pareto_n{n}_d{d}.{suffix}")
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("results/accuracy_runtime_pareto"))
    plot(parser.parse_args().output_dir)
