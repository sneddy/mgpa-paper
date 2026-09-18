"""Frozen-result tables and figures; no fitting or endpoint reselection."""
from __future__ import annotations

import csv
from pathlib import Path
import numpy as np
from mgpa.data import reuse as data
from . import policy
from . import evaluation as ev

TASKS = ("p300", "n170", "mmn")
GROUPS = {"identity": ("identity",), "leace": ("leace",),
    "mgpa_cf": ("mgpa_cf",), "mgpa_iter": tuple(f"mgpa_iter_seed{s}" for s in (17, 29, 43)),
    "coral": ("coral",), "featmap": ("featmap",)}
NAMES = {"identity": "Identity", "leace": "LEACE", "mgpa_cf": "Closed-form MGPA",
    "mgpa_iter": "Iterative MGPA", "coral": "CORAL", "featmap": "FEATMAP"}
METRICS = ("worst", "ordinary", "aligned", "independent", "reversed")


def verify_dev_evidence(root, cfg):
    """Replay the published DEV policy from saved numerical curves, without fitting."""
    root = Path(root)
    cells, pins, count = [], {}, 0
    for path in sorted((root / "tables/dev_cells").glob("*.json")):
        cell = data.read_json(path)
        selected = ev.select_curve(cell["dev_curve"], cfg)
        if selected != cell["selected"]:
            raise ValueError(f"Saved DEV endpoint does not match exact selection policy: {path.name}")
        count += len(cell["dev_curve"])
        cells.append({**cell, "selected": selected, "model_dir": "not-used-for-policy-replay",
            "cell_path": str(path.relative_to(root)), "binding": "published-numerical-evidence"})
        pins[str(path.relative_to(root))] = data.sha256(path)
    if len(cells) != 28 or count != 1444:
        raise ValueError("Expected all 28 published candidates and 1,444 DEV endpoints")
    selected = ev.select_families([c for c in cells if c["candidate"]["family"] != "mgpa_iter"], cfg)
    actual = {family: dict(candidate=value["candidate"]["id"], endpoint=value["selected"]["endpoint"])
        for family, value in selected.items()}
    actual.update({c["candidate"]["id"]: dict(candidate=c["candidate"]["id"], endpoint=c["selected"]["endpoint"])
        for c in cells if c["candidate"]["family"] == "mgpa_iter"})
    expected = {"identity": ("identity", "strength_0"), "mgpa_cf": ("mgpa_cf", "strength_1.1"),
        "leace": ("leace_ridge0", "strength_0.98"), "coral": ("coral_ref1_ridge10", "strength_0.98"),
        "featmap": ("featmap_ref0_ridge1000", "strength_1"),
        **{f"mgpa_iter_seed{s}": (f"mgpa_iter_seed{s}", f"fixed_{k}") for s, k in ((17, 13), (29, 11), (43, 19))}}
    if any(actual[k] != dict(candidate=v[0], endpoint=v[1]) for k, v in expected.items()):
        raise ValueError("P300 DEV family selection differs from the manuscript")
    return dict(status="PASS", candidate_count=len(cells), endpoint_count=count, selections=actual,
        selected_summaries_exact=True, independent_head_guardrails=-.01,
        within_candidate_tolerance=0., between_baseline_recipes_tolerance=.002,
        source_task="p300", selection_role="dev", refitting=False, input_hashes=pins)

def minimum(coeff):
    """Vectorized continuous minimum on [-1,1], last axis [a,b,c]."""
    value = np.asarray(coeff, dtype=float)
    if value.shape[-1:] != (3,) or not np.isfinite(value).all():
        raise ValueError("Finite quadratic coefficients required")
    a, b, c = np.moveaxis(value, -1, 0)
    result = np.minimum(np.minimum(a - b + c, c), a + b + c)
    convex = a > 0
    vertex = np.zeros_like(a)
    np.divide(-b, 2 * a, out=vertex, where=convex)
    valid = convex & (vertex >= -1) & (vertex <= 1)
    return np.where(valid, np.minimum(result, (a * vertex + b) * vertex + c), result)


def coefficients(record, head, people):
    """Read saved quadratic coefficients in the exact common participant order."""
    values = record["independent_audit"][head]["coefficients"]["per_subject"]
    if sorted(values) != list(people):
        raise ValueError("All maps/tasks must use identical participants, not an intersection")
    array = np.asarray([values[p] for p in people], dtype=float)
    if array.shape != (len(people), 3) or not np.isfinite(array).all():
        raise ValueError("Invalid saved participant coefficients")
    return array


def fit_outcomes(records, people, indices=None):
    """Per-fit values: participant mean first, minimum second. Never ensemble."""
    biased = np.stack([coefficients(r, "biased", people) for r in records])
    control = np.stack([coefficients(r, "control", people) for r in records])
    if indices is None:
        biased, control = biased.mean(axis=1), control.mean(axis=1)
    else:
        indices = np.asarray(indices)
        if indices.ndim != 2 or indices.shape[1] != len(people):
            raise ValueError("Participant draw shape changed")
        biased, control = biased[:, indices].mean(axis=2), control[:, indices].mean(axis=2)
    a, b, c = np.moveaxis(biased, -1, 0)
    return dict(worst=minimum(biased), ordinary=control[..., 2],
                aligned=a+b+c, independent=c, reversed=a-b+c)


def interval(estimate, draws):
    """Pair a point estimate with its 95% percentile bootstrap interval."""
    values = np.asarray(draws, dtype=float)
    lower, upper = np.quantile(values, [.025, .975])
    return dict(estimate=float(estimate), lower=float(lower), upper=float(upper))

def collect(root):
    """Same participant draws across methods/tasks/fits; minimize each fit first."""
    root = Path(root)
    archive = not (root / "tables/eval_cells").is_dir()
    directory = root / ("tables/cells" if archive else "tables/eval_cells")
    people = [f"flex:{p}" for p in range(1026, 1034)]
    indices = np.random.default_rng(1101).integers(0, 8, size=(20000, 8))
    results, contrasts, pins, reference_selections = {}, {}, {}, {}
    for task in TASKS:
        results[task], contrasts[task] = {}, {}
        estimates, draws = {}, {}
        for family, keys in GROUPS.items():
            records = []
            for key in keys:
                path = directory / f"{task}__{key}.json"
                row = data.read_json(path)
                if archive:
                    row = {**row, "task": task, "key": key,
                        "selection": row["result"]["endpoint"],
                        "independent_audit": {h: {"coefficients": policy.coefficients(row["result"]["metrics"][h])}
                            for h in ("biased", "control")}}
                if row["task"] != task or row["key"] != key:
                    raise ValueError("Saved result identity mismatch")
                if task == "p300":
                    reference_selections[key] = row["selection"]
                elif row["selection"] != reference_selections[key]:
                    raise ValueError("Transferred correction was changed between tasks")
                if not archive:
                    pred = root / row["predictions_path"]
                    if data.sha256(pred) != row["predictions_sha256"]:
                        raise ValueError("Saved prediction hash mismatch")
                    pins[str(pred.relative_to(root))] = data.sha256(pred)
                pins[str(path.relative_to(root))] = data.sha256(path)
                records.append(row)
            e, b = fit_outcomes(records, people), fit_outcomes(records, people, indices)
            estimates[family] = {m: float(e[m].mean()) for m in METRICS}
            draws[family] = {m: b[m].mean(axis=0) for m in METRICS}
            results[task][family] = dict(n_fits=len(keys),
                metrics={m: interval(estimates[family][m], draws[family][m]) for m in METRICS},
                per_fit={key: {m: float(e[m][i]) for m in METRICS} for i, key in enumerate(keys)},
                movement=float(np.mean([r["result"]["summary"]["movement"]["mean"] for r in records])))
        for family in GROUPS:
            contrasts[task][family] = {reference: {m: interval(estimates[family][m]-estimates[reference][m],
                draws[family][m]-draws[reference][m]) for m in METRICS} for reference in ("identity", "leace")}
    return dict(schema="mgpa-task-reuse-report-v1", results=results, contrasts=contrasts,
        selections=reference_selections, participants=people, input_hashes=pins,
        bootstrap=dict(resamples=20000, seed=1101, unit="participant", confidence=.95,
            common_draws_across_methods_tasks_and_fits=True, minimizer_recomputed_per_fit_per_draw=True,
            aggregation="Participant mean, minimum per fit, then average fits; no prediction ensemble",
            limitation="Conditional on fitted maps, heads and coordinates; previously exposed cohorts; no multiplicity correction"))


def report(root, output=None):
    """Render saved outcomes as JSON, CSV, Markdown and publication figures."""
    root = Path(root)
    output = Path(output) if output else root / "report"
    summary = collect(root)
    data.write_json(output / "results.json", summary)
    lines = ["| Method | P300 worst | N170 worst | MMN worst | P300 ordinary | N170 ordinary | MMN ordinary |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    csv_rows = []
    for family in GROUPS:
        values = [summary["results"][t][family]["metrics"][m]["estimate"] for m in ("worst", "ordinary") for t in TASKS]
        lines.append("| " + NAMES[family] + " | " + " | ".join(f"{v:.4f}" for v in values) + " |")
        for task in TASKS:
            for metric, value in summary["results"][task][family]["metrics"].items():
                csv_rows.append(dict(method=NAMES[family], task=task, metric=metric, **value))
    output.mkdir(parents=True, exist_ok=True)
    (output / "table.md").write_text("\n".join(lines) + "\n")
    with (output / "table.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    for ax, task in zip(axes, TASKS):
        for i, family in enumerate(GROUPS):
            ci = summary["results"][task][family]["metrics"]["worst"]
            point, = ax.plot(i, ci["estimate"], "o")
            ax.vlines(i, ci["lower"], ci["upper"], color=point.get_color())
            ax.hlines([ci["lower"], ci["upper"]], i-.06, i+.06, color=point.get_color())
        ax.set_title(task.upper())
        ax.set_xticks(range(len(GROUPS)), [NAMES[k] for k in GROUPS], rotation=50, ha="right")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Worst association AUROC")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"worst_auroc.{suffix}", dpi=240)
    plt.close(fig)
    return summary
