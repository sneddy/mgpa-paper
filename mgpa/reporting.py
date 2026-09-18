"""Readable published or fresh result views, with no scientific fitting."""
import csv
import hashlib
import json
import math
from pathlib import Path
import re


def report_experiment(experiment_dir, output_dir=None):
    """Write readable estimates from one experiment's selected result bundle."""
    root = Path(experiment_dir)
    source = root/'artifacts/tables/published.json'
    values = json.loads(source.read_text())
    output = Path(output_dir) if output_dir else root/'artifacts/tables/rebuilt'
    output.mkdir(parents=True,exist_ok=True)
    rows = []
    def visit(value, path=()):
        """Flatten metric estimates without inventing values for missing outputs."""
        if isinstance(value,dict):
            estimate=value.get('mean',value.get('estimate'))
            if isinstance(estimate,(int,float)):
                low,high=value.get('ci95',(value.get('lower'),value.get('upper')))
                rows.append(('/'.join(path),float(estimate),low,high))
            else:
                for key,child in value.items():
                    visit(child,(*path,key))
        elif isinstance(value,(int,float)) and path[-1] in ('S','Tf','Tr','M'):
            rows.append(('/'.join(path),float(value),None,None))
    visit(values)
    with (output/'table.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['result','estimate','ci_lower','ci_upper']);writer.writerows(rows)
    lines=['| Result | Estimate | 95% interval |','|---|---:|---|']
    for name,value,low,high in rows:
        ci='' if low is None else f'[{low:.6f}, {high:.6f}]'
        lines.append(f'| {name} | {value:.6f} | {ci} |')
    (output/'table.md').write_text('\n'.join(lines)+'\n')
    return {'rows':len(rows),'output':str(output.resolve()),'fitting_performed':False}


METRICS = ("S", "Tf", "Tr", "M")
METRIC_TITLES = ("Source AUROC (lower)", "Frozen task AUROC (higher)",
                 "Refitted task AUROC (higher)", "Movement (lower)")
METHOD_NAMES = {"identity": "Identity", "leace": "LEACE", "mgpa_cf": "Closed-form MGPA",
    "mgpa_iter": "Iterative MGPA", "mgpa_linear": "MGPA, linear critics", "igbp": "IGBP",
    "measurement": "Measurement gate", "pca": "Same-rank PCA", "ungated": "Ungated",
    "random_mean5": "Random, mean of 5", "conditional": "Conditional-mean target",
    "fit_mean": "FIT-mean target", "zero": "Zero target"}


def _method_name(method):
    """Use readable names without conflating source-aware alignment routes."""
    for baseline in ("coral", "featmap"):
        if method.startswith(baseline + "_"):
            return f"{baseline.upper()} to source {method.rsplit('_', 1)[1]}"
    return METHOD_NAMES.get(method, method.replace("_", " "))


def _statistic(value):
    """Validate a saved estimate and interval without recomputing either."""
    if value is None:
        return None
    estimate = float(value["mean"])
    interval = value.get("ci95")
    low, high = (None, None) if interval is None else map(float, interval)
    if not math.isfinite(estimate) or (low is not None and
            (not math.isfinite(low) or not math.isfinite(high) or low > high)):
        raise ValueError("Invalid saved estimate or confidence interval")
    return estimate, low, high


def _write_csv(path, header, rows):
    """Write a transparent long-form numerical table."""
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def _render_fresh_group(rows, title, output):
    """Draw saved four-metric estimates with their participant intervals."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    height = max(2.6, 1.3 + .34 * len(rows))
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 8.5,
                         "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig, axes = plt.subplots(1, 4, figsize=(11.8, height), sharey=True)
        for ax, metric, metric_title in zip(axes, METRICS, METRIC_TITLES):
            ax.set_title(metric_title, fontsize=9)
            for index, row in enumerate(rows):
                stat = _statistic(row.get(metric))
                if stat is None:
                    continue
                mean, low, high = stat
                color = "#0072B2" if row["method"].startswith(("mgpa", "measurement", "conditional")) else "#58616A"
                if low is not None:
                    ax.hlines(index, low, high, color=color, linewidth=1.2)
                ax.scatter(mean, index, color=color, s=22, zorder=3)
                if row["method"] == "identity":
                    ax.axvline(mean, color="#A6ADB5", linestyle="--", linewidth=.8)
            if metric == "S":
                ax.axvline(.5, color="#C8CDD2", linestyle=":", linewidth=.7)
            ax.grid(axis="x", color="#EEF0F2")
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.set_axisbelow(True)
            ax.tick_params(axis="y", length=0)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.set_ylim(len(rows) - .5, -.5)
        axes[0].set_yticks(range(len(rows)), [_method_name(row["method"]) for row in rows])
        fig.suptitle(title, fontsize=11, y=.98)
        fig.text(.23, .015, "Intervals resample participants, conditional on fitted maps/readers; repetitions are not extra participants.",
                 fontsize=7.2)
        fig.tight_layout(rect=(0, .06, 1, .91))
        files = []
        for suffix in ("pdf", "png"):
            path = Path(str(output) + "." + suffix)
            metadata = {"CreationDate": None, "ModDate": None} if suffix == "pdf" else {}
            fig.savefig(path, dpi=180, metadata=metadata)
            files.append(path.name)
        plt.close(fig)
    return files


def report_run(summary_path, output_dir=None):
    """Create tables and figures solely from a fresh run's saved summary JSON."""
    source = Path(summary_path)
    saved = source.read_bytes()
    values = json.loads(saved)
    if not isinstance(values.get("summaries"), list) or "dataset" not in values:
        raise ValueError("Fresh reporting requires a run summary, not published values")
    output = Path(output_dir) if output_dir else source.parent
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    groups, numerical_rows = {}, []
    for row in values["summaries"]:
        key = row["comparison"], row["endpoint"]
        groups.setdefault(key, []).append(row)
        for metric in METRICS:
            stat = _statistic(row.get(metric))
            if stat is not None:
                numerical_rows.append((*key, row["method"], metric, *stat))
    header = ["comparison", "endpoint", "method", "metric", "estimate", "ci_lower", "ci_upper"]
    _write_csv(output / "table.csv", header, numerical_rows)
    contrast_rows = []
    for row in values.get("paired_contrasts", []):
        for metric in METRICS:
            stat = _statistic(row["metrics"].get(metric))
            if stat is not None:
                contrast_rows.append((row["comparison"], row["endpoint"], row["method"],
                                      row["reference"], metric, *stat))
    _write_csv(output / "paired_contrasts.csv", [*header[:3], "reference", *header[3:]], contrast_rows)
    lines = [f"# Fresh results: {values['dataset']}", "", f"Status: {values['status']}.", "",
             "Only this run's saved estimates are used; no fitting or published-value substitution occurs.", "",
             values.get("interpretation", ""), ""]
    rendered = []
    for (comparison, endpoint), rows in groups.items():
        # Keep each random orientation in CSV; show the correctly aggregated ensemble once.
        if any(row["method"] == "random_mean5" for row in rows):
            rows = [row for row in rows if not re.fullmatch(r"random_[0-4]", row["method"])]
        lines.extend([f"## {comparison.replace('_', ' ')} / {endpoint}", "",
            "| Method | Source AUROC (lower) | Frozen task (higher) | Refitted task (higher) | Movement (lower) |",
            "|---|---:|---:|---:|---:|"])
        for row in rows:
            entries = []
            for metric in METRICS:
                stat = _statistic(row.get(metric))
                if stat is None:
                    entries.append("not evaluated")
                else:
                    mean, low, high = stat
                    entries.append(f"{mean:.4f}" + (f" [{low:.4f}, {high:.4f}]" if low is not None else ""))
            lines.append("| " + " | ".join([_method_name(row["method"]), *entries]) + " |")
        lines.append("")
        stem = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{comparison}__{endpoint}")
        names = _render_fresh_group(rows, f"{values['dataset']}: {comparison.replace('_', ' ')} / {endpoint}", figures / stem)
        rendered.extend("figures/" + name for name in names)
        lines.extend([f"![Four-metric comparison](figures/{stem}.png)", ""])
    lines.extend(["Source-aware CORAL/FEATMAP routes are identified by their reference source. "
                  "When present, the random ensemble averages five orientation-specific reader maxima; "
                  "the individual orientations remain in `table.csv`.", "",
                  "Paired method-minus-reference estimates are in `paired_contrasts.csv`. "
                  "Source contrasts recompute both reader maxima within each bootstrap draw.", ""])
    omitted = values.get("omitted_groups", [])
    if omitted:
        lines.extend(["## Unavailable endpoint groups", "", "No partial-cohort estimate or native-output substitute is reported.", ""])
        for row in omitted:
            lines.append(f"- {row['comparison']} / {row['method']} / {row['endpoint']}: "
                         f"{row['completed']}/{row['planned']} cells; {row['reason']}.")
    (output / "table.md").write_text("\n".join(lines) + "\n")
    receipt = dict(source=source.name, source_sha256=hashlib.sha256(saved).hexdigest(),
        dataset=values["dataset"], numerical_rows=len(numerical_rows), paired_rows=len(contrast_rows),
        figures=rendered, omitted_groups=len(omitted), fitting_performed=False, published_values_used=False)
    (output / "report.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt
