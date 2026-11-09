"""Report figures from results/eval_summary.json and the agent benchmarks.

  python eval/make_figures.py      -> results/figures/*.png

Style: dataviz reference palette (fixed categorical order B0=slot 1 blue,
B1=slot 2 orange, P1=slot 3 aqua - the first three slots validate all-pairs for
colour-vision deficiency), one y-axis per chart, solid hairline grid, thin bars
with surface gaps, legend for every multi-series chart.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.common import DATASETS, RESULTS  # noqa: E402

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e9e8e4"
SLOTS = {"B0": "#2a78d6", "B1": "#eb6834", "P1": "#1baf7a"}
NAMES = {"B0": "B0 stakeholder baseline", "B1": "B1 greedy baseline", "P1": "P1 agentic system"}
OUT = RESULTS / "figures"


def _style(ax, title: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=24)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=10)
    ax.grid(axis="y", color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)


def grouped(summary: dict, key: str, title: str, ylabel: str, fname: str, target: float | None = None,
            target_label: str = "", ymax: float | None = None) -> Path | None:
    datasets = [d for d in DATASETS if any(k.startswith(d + "|") for k in summary)]
    systems = [s for s in ("B0", "B1", "P1") if any(f"{d}|{s}" in summary for d in datasets)]
    if not datasets or not systems:
        return None
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=200)
    width = 0.8 / len(systems)
    for i, s in enumerate(systems):
        xs, ys = [], []
        for j, d in enumerate(datasets):
            v = (summary.get(f"{d}|{s}") or {}).get(key)
            if v is not None:
                xs.append(j - 0.4 + width * (i + 0.5))
                ys.append(float(v))
        ax.bar(xs, ys, width=width, color=SLOTS[s], edgecolor=SURFACE, linewidth=1.5, label=NAMES[s])
    if target is not None:
        ax.axhline(target, color=INK_2, linewidth=1)
        ax.text(len(datasets) - 0.45, target, f" {target_label}", va="bottom", ha="right", fontsize=8, color=INK_2)
    ax.set_xticks(range(len(datasets)), datasets)
    if ymax:
        ax.set_ylim(0, ymax)
    _style(ax, title, ylabel)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(systems), frameon=False, fontsize=8,
              labelcolor=INK_2, handlelength=1, borderaxespad=0.2)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / fname
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def main_panels(summary: dict) -> Path | None:
    """Report figure: three small panels (one measure and one y-axis each) with a shared legend."""
    datasets = [d for d in DATASETS if any(k.startswith(d + "|") for k in summary)]
    systems = [s for s in ("B0", "B1", "P1") if any(f"{d}|{s}" in summary for d in datasets)]
    if not datasets or not systems:
        return None
    panels = [("cross_wave_edge_pct", "Dependencies split across waves (%)\nlower is better", None),
              ("early_satisfied_pct", "Early requests placed on time (%)\nhigher is better", None),
              ("utilization_pct", "Wave utilization (%)\nhigher is better", 60)]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), dpi=250)
    width = 0.8 / len(systems)
    for ax, (key, title, target) in zip(axes, panels):
        for i, s in enumerate(systems):
            xs, ys = [], []
            for j, d in enumerate(datasets):
                v = (summary.get(f"{d}|{s}") or {}).get(key)
                if v is not None:
                    xs.append(j - 0.4 + width * (i + 0.5))
                    ys.append(float(v))
            ax.bar(xs, ys, width=width, color=SLOTS[s], edgecolor=SURFACE, linewidth=0.8, label=NAMES[s])
            for x, y in zip(xs, ys):  # a zero-height bar would read as missing data
                if y == 0:
                    ax.text(x, 1.5, "0", ha="center", va="bottom", fontsize=5.5, color=SLOTS[s], fontweight="bold")
        if target:
            ax.axhline(target, color=INK_2, linewidth=0.8)
            ax.text(len(datasets) - 0.5, target + 1.5, "M1 target", ha="right", va="bottom", fontsize=6, color=INK_2)
        ax.set_xticks(range(len(datasets)), [d.replace("_", " ") for d in datasets], fontsize=6.5)
        ax.set_ylim(0, 105)
        ax.set_facecolor(SURFACE)
        ax.set_title(title, fontsize=7.5, color=INK, loc="left")
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=6.5, length=0)
    fig.set_facecolor(SURFACE)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(systems), frameon=False, fontsize=7,
               labelcolor=INK_2, handlelength=1, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "main_results.png"
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def qa_chart() -> Path | None:
    files = sorted(RESULTS.glob("qa_eval_*.json"))
    if not files:
        return None
    summ = json.loads(files[0].read_text())["summary"]
    metrics = [("executed_pct", "Query executed"), ("rows_correct_pct", "Rows correct"),
               ("answer_correct_pct", "Answer correct")]
    systems = [s for s in ("B0-style", "P1") if s in summ]
    colors = {"B0-style": SLOTS["B0"], "P1": SLOTS["P1"]}
    names = {"B0-style": "B0-style GraphCypherQAChain", "P1": "P1 Q&A agent"}
    fig, ax = plt.subplots(figsize=(6.5, 3.4), dpi=200)
    width = 0.8 / len(systems)
    for i, s in enumerate(systems):
        xs = [j - 0.4 + width * (i + 0.5) for j in range(len(metrics))]
        ax.bar(xs, [summ[s][k] for k, _ in metrics], width=width, color=colors[s], edgecolor=SURFACE,
               linewidth=1.5, label=names[s])
    ax.set_xticks(range(len(metrics)), [m for _, m in metrics])
    ax.set_ylim(0, 100)
    _style(ax, f"Ask-the-plan accuracy ({summ[systems[0]]['questions']} questions, D1)", "% of questions")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=8, labelcolor=INK_2,
              handlelength=1, borderaxespad=0.2)
    path = OUT / "qa_accuracy.png"
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def main() -> None:
    summary = json.loads((RESULTS / "eval_summary.json").read_text())
    made = [
        grouped(summary, "cross_wave_edge_pct", "Dependencies split across waves (lower is better)",
                "% of dependencies", "split_dependencies.png", ymax=100),
        grouped(summary, "early_satisfied_pct", "Early-migration requests placed on time",
                "% of feasible requests", "early_on_time.png", ymax=105),
        grouped(summary, "utilization_pct", "Wave capacity utilization (excl. last, pilot, over-limit waves)",
                "% of 3,000 GB", "utilization.png", target=60, target_label="Milestone 1 target 60%", ymax=105),
        grouped(summary, "wave_count", "Number of migration waves", "waves", "wave_count.png"),
        main_panels(summary),
        qa_chart(),
    ]
    for p in made:
        if p:
            print(p)


if __name__ == "__main__":
    main()
