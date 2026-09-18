"""Render the controlled mechanism figure from published numerical values."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
TARGET_METHODS = ("mgpa_iter","fit_mean")
METHODS = (("measurement","Measurement","#0072B2"),("pca","PCA","#8A5EA6"),("random","Random","#609768"),("ungated","Ungated","#D55E00"))

def load_data():
    """Read only the selected gate and target outputs."""
    data=json.loads((ROOT/"paper_assets/published_values.json").read_text())["controlled_mechanisms"]
    return data["gate"],data["targets"]

def draw_interval(ax, stat, y, color, scale=1., highlighted=False):
    """Draw an estimate and its original participant-bootstrap interval."""
    mean = stat["mean"] * scale
    low, high = np.asarray(stat["ci95"]) * scale
    assert low <= mean <= high
    ax.errorbar(mean, y, xerr=[[mean-low], [high-mean]], fmt="o",
                color=color, markersize=5.8 if highlighted else 4.8,
                capsize=2.5, elinewidth=1.3, markeredgewidth=1., zorder=3)


def render(output):
    """Draw source, paired task change, and conditional-target cost panels."""
    OUT = Path(output)
    rows, targets = load_data()
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.2,
        "axes.labelsize": 9., "xtick.labelsize": 8.5,
        "ytick.labelsize": 9.2, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(7.2, 2.15))
    source_ax = fig.add_axes([.15, .25, .19, .57])
    task_ax = fig.add_axes([.385, .25, .205, .57], sharey=source_ax)
    table_ax = fig.add_axes([.64, .23, .35, .595])
    fig.text(.15, .935, r"Gate: validation movement budget $0.05$", fontsize=9.3, fontweight="bold")
    fig.text(.64, .935, "(c) Target: native outputs", fontsize=9.3, fontweight="bold")

    for i, (name, label, color) in enumerate(METHODS):
        y = 3-i
        for ax in (source_ax, task_ax):
            if name == "measurement":
                ax.axhspan(y-.34, y+.34, color=color, alpha=.05, zorder=0)
        draw_interval(source_ax, rows[name]["metrics"]["S"], y, color,
                      highlighted=name == "measurement")
        draw_interval(task_ax, rows[name]["delta_Tf"], y, color,
                      scale=100., highlighted=name == "measurement")
    source_ax.set(xlim=(.49, .725), ylim=(-.45, 3.55), xlabel=r"(a) Source AUROC $\downarrow$")
    source_ax.set_xticks([.5, .6, .7])
    source_ax.set_yticks([3, 2, 1, 0], [row[1] for row in METHODS])
    for label, (name, _, color) in zip(source_ax.get_yticklabels(), METHODS):
        label.set_color(color)
        label.set_fontweight("bold" if name == "measurement" else "normal")
    identity_source = rows["identity"]["metrics"]["S"]["mean"]
    source_ax.axvline(identity_source, color="#777777", linestyle=(0, (4, 3)), linewidth=.9, zorder=1)
    source_ax.axvline(.5, color="#BCC2C9", linestyle=(0, (1, 3)), linewidth=.8, zorder=1)
    task_ax.set(xlim=(-1.75, .16), xlabel=r"(b) Frozen-task $\Delta$ (pp)")
    task_ax.set_xticks([-1.5, -1., -.5, 0.])
    task_ax.axvline(0, color="#777777", linestyle=(0, (4, 3)), linewidth=.9, zorder=1)
    task_ax.tick_params(labelleft=False)
    for ax in (source_ax, task_ax):
        ax.grid(axis="x", color="#EEF0F2", linewidth=.55)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0, pad=6)
        ax.tick_params(axis="x", color="#8B929A", pad=3)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#8B929A")

    table_ax.set(xlim=(0, 1), ylim=(0, 1))
    table_ax.axis("off")
    x_values = (.57, .88)
    for x, title, color in zip(x_values, ("Cond.\nmean", "Uncond.\nmean"), ("#0072B2", "#555D66")):
        table_ax.text(x, .94, title, ha="center", va="center", fontsize=8.5,
                      color=color, fontweight="bold", linespacing=1.05)
    table_ax.plot([0, 1], [.81, .81], color="#9EA6AE", linewidth=.65)
    table_ax.axhspan(.49, .80, color="#0072B2", alpha=.045, zorder=0)
    table_ax.text(0, .69, r"Movement $\downarrow$", fontsize=8.6, fontweight="bold", va="center")
    for x, method, color in zip(x_values, TARGET_METHODS, ("#0072B2", "#555D66")):
        stat = targets[method]["M"]
        table_ax.text(x, .70, f"{stat['mean']:.3f}", ha="center", va="center",
                      fontsize=10.8, fontweight="bold", color=color)
        lo, hi = stat["ci95"]
        table_ax.text(x, .555, f"[{lo:.3f}, {hi:.3f}]", ha="center", va="center",
                      fontsize=7.0, color="#58616A")
    table_ax.plot([0, 1], [.47, .47], color="#D4D8DC", linewidth=.6)
    for y, key, title in ((.35, "S", r"Source $\downarrow$"),
                           (.185, "Tf", r"Frozen task $\uparrow$"),
                           (.02, "Tr", r"Refitted task $\uparrow$")):
        table_ax.text(0, y, title, fontsize=8.4, va="center")
        for x, method in zip(x_values, TARGET_METHODS):
            table_ax.text(x, y, f"{targets[method][key]['mean']:.3f}",
                          ha="center", va="center", fontsize=8.7)

    for suffix in ("pdf", "png"):
        metadata = {"CreationDate": None, "ModDate": None} if suffix == "pdf" else {}
        fig.savefig(OUT / f"controlled_mechanisms_compact.{suffix}", dpi=260, metadata=metadata)
    plt.close(fig)
