"""Rebuild the cross-task reuse figure from selected numerical outputs."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mgpa-forest-mpl-"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments/task_reuse/artifacts/tables/published.json"
METHODS = (
    ("identity", "Identity", "#777777", 5.5),
    ("leace", "LEACE", "#8A5EA6", 4.5),
    ("mgpa_cf", "MGPA-CF", "#D55E00", 3.5),
    ("mgpa_iter", "MGPA-Iter", "#0072B2", 2.5),
    ("coral", "CORAL", "#60968C", .7),
    ("featmap", "FEATMAP", "#9E8559", -.3),
)


def render(output):
    """Plot the absolute worst-association AUROC and participant intervals."""
    OUT = Path(output)
    data = json.loads(SOURCE.read_text())
    assert data["status"] == "COMPLETE"
    assert len(data["participants"]) == 8
    assert data["bootstrap"]["resamples"] == 20000
    assert data["bootstrap"]["common_draws_across_methods_tasks_and_fits"]
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.2,
        "axes.titlesize": 11, "axes.labelsize": 9.4,
        "xtick.labelsize": 8.8, "ytick.labelsize": 9.2,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.35), sharey=True)
    fig.subplots_adjust(left=.16, right=.915, bottom=.215, top=.88, wspace=.32)
    records = {}
    for ax, task, limits, ticks in zip(
        axes, ("n170", "mmn"), ((.08, .64), (.39, .705)),
        ([.1, .2, .3, .4, .5, .6], [.4, .5, .6, .7]),
    ):
        leace = data["results"][task]["leace"]["metrics"]["worst"]["estimate"]
        ax.set(xlim=limits, ylim=(-.9, 6.3))
        ax.set_title(r"P300 $\rightarrow$ " + task.upper(), loc="left", pad=6)
        ax.set_xticks(ticks)
        ax.set_xlabel("Worst-association AUROC", labelpad=5)
        ax.axvline(leace, color="#8A5EA6", linestyle=(0, (4, 3)), linewidth=1.25, zorder=1)
        ax.axvline(.5, color="#BBC1C9", linestyle=(0, (1, 3)), linewidth=.85, zorder=0)
        ax.axhline(1.75, color="#D4D8DC", linewidth=.7, zorder=0)
        ax.grid(axis="x", color="#EEF0F2", linewidth=.55)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#8B929A")
        ax.tick_params(axis="y", length=0, pad=8)
        ax.tick_params(axis="x", color="#8B929A")
        records[task] = {}
        for key, label, color, y in METHODS:
            item = data["results"][task][key]
            assert item["n_fits"] == (3 if key == "mgpa_iter" else 1)
            stat = item["metrics"]["worst"]
            contrast = data["contrasts"][task][key]["leace"]["worst"]
            mean, low, high = (stat[k] for k in ("estimate", "lower", "upper"))
            assert low <= mean <= high
            assert np.isclose(mean - leace, contrast["estimate"], rtol=0, atol=1e-12)
            supported = contrast["lower"] > 0
            mgpa = key in ("mgpa_cf", "mgpa_iter")
            source_aware = key in ("coral", "featmap")
            if mgpa:
                ax.axhspan(y-.38, y+.38, color=color, alpha=.045, zorder=0)
            ax.errorbar(
                mean, y, xerr=np.array([[mean-low], [high-mean]]),
                fmt="D" if source_aware else "o", markersize=6.8 if mgpa else 5.5,
                color=color, markerfacecolor="white" if source_aware else color,
                markeredgewidth=1.25, elinewidth=1.5 if mgpa else 1.1,
                capsize=3, capthick=1, zorder=3,
            )
            value = f"{mean:.3f}" + (r"$^{\dagger}$" if supported else "")
            ax.text(1.015, y, value, transform=ax.get_yaxis_transform(),
                    va="center", ha="left", fontsize=8.7, color=color,
                    fontweight="bold" if mgpa else "normal", clip_on=False)
            records[task][key] = {
                "absolute_worst": stat, "paired_difference_from_leace": contrast,
                "dagger": supported, "n_fits": item["n_fits"],
            }
    axes[0].set_yticks([row[3] for row in METHODS], [row[1] for row in METHODS])
    for label, (key, _, color, _) in zip(axes[0].get_yticklabels(), METHODS):
        label.set_color(color)
        label.set_fontweight("bold" if key in ("mgpa_cf", "mgpa_iter") else "normal")
    axes[0].text(-.035, 6.12, "Source-blind", transform=axes[0].get_yaxis_transform(),
                 ha="right", va="center", fontsize=8, color="#555D66", fontstyle="italic")
    axes[0].text(-.035, 1.35, "Source-aware", transform=axes[0].get_yaxis_transform(),
                 ha="right", va="center", fontsize=8, color="#555D66", fontstyle="italic")
    for suffix in ("pdf", "png"):
        metadata = {"CreationDate": None, "ModDate": None} if suffix == "pdf" else {}
        fig.savefig(OUT / f"reuse_worst_preview.{suffix}", dpi=260, metadata=metadata)
    plt.close(fig)
