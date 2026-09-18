"""Shared lifecycle for the three published component/native experiments.

Experiment directories supply a small data boundary and a readable TOML design.
This module handles portable preparation, fit-only locking, numerical replay,
independent readers, and reporting. It never imports an earlier research tree.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import time
import tomllib

import numpy as np

from .artifacts import digest, file_hash, read_json, run_guard, runtime, safe_id, write_json
from .interfaces import CONTEXTS, recorded_calibration_pairs
from .prepared import prepare_context

RANDOM_SEEDS = (104729, 130363, 155921, 196613, 228017)
ITERATIVE = {"mgpa_iter", "mgpa_linear", "conditional", "fit_mean", "zero", "measurement", "pca", "ungated",
             *(f"random_{i}" for i in range(5))}
AFFINE = {"identity", "leace", "mgpa_cf", "coral_0", "coral_1", "featmap_0", "featmap_1"}


def load_config(path, dataset):
    """Validate the published design and its source-blind fitting settings."""
    with Path(path).open("rb") as stream:
        value = tomllib.load(stream)
    if (value.get("schema") != "mgpa-published-experiment-v1" or value.get("dataset") != dataset
            or value.get("contexts") != list(CONTEXTS[dataset]) or value.get("reader_seed") != 98
            or value.get("threads") != 2 or value.get("random_orientation_seeds") != list(RANDOM_SEEDS)):
        raise ValueError("The experiment configuration does not match its data protocol")
    names = []
    for comparison in value["comparisons"]:
        name = safe_id(comparison["name"])
        names.append(name)
        if (not comparison["methods"] or len(set(comparison["methods"])) != len(comparison["methods"])
                or not set(comparison["methods"]).issubset(AFFINE | ITERATIVE | {"igbp"})
                or not set(comparison["fit_seeds"]).issubset({17, 29, 43})):
            raise ValueError("Unreported method or fit repetition in experiment configuration")
        if dataset == "recorded_ssvep" and any(m.startswith("featmap") for m in comparison["methods"]):
            raise ValueError("Recorded sessions do not provide FEATMAP's physiological pairs")
        for endpoint in comparison["endpoints"]:
            if endpoint != "native" and (not endpoint.startswith("budget_") or float(endpoint[7:]) <= 0):
                raise ValueError("Only native and prespecified movement-budget endpoints are supported")
    if len(names) != len(set(names)):
        raise ValueError("Comparison names must be unique")
    return value


def jobs(config):
    """Expand comparisons while sharing deterministic fits across repetitions."""
    result = []
    for comparison in config["comparisons"]:
        for context in config["contexts"]:
            for method in comparison["methods"]:
                for seed in comparison["fit_seeds"]:
                    # Deterministic maps are fitted once, then explicitly reused.
                    fit_seed = 17 if method in AFFINE else seed
                    fit_id = f"{context}__{method}__s{fit_seed}"
                    for endpoint in comparison["endpoints"]:
                        result.append(dict(dataset=config["dataset"], comparison=comparison["name"],
                            context=context, method=method, seed=seed, endpoint=endpoint,
                            fit_seed=fit_seed, fit_id=fit_id,
                            cell_id=f"{comparison['name']}__{context}__{method}__s{seed}__{endpoint}"))
    return result


def log(run, event, **values):
    """Append one small progress event and print it for the launcher."""
    row = dict(event=event, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **values)
    line = json.dumps(row, allow_nan=False)
    print(line, flush=True)
    path = run/"logs/progress.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(line+"\n")


def code_pins(experiment_dir):
    """Hash the release implementation without including generated artifacts."""
    package = Path(__file__).resolve().parents[1]
    return {"package/"+str(p.relative_to(package)): file_hash(p) for p in sorted(package.rglob("*.py"))} | {
        "experiment/"+str(p.relative_to(experiment_dir)): file_hash(p)
        for p in sorted(experiment_dir.rglob("*.py")) if "artifacts" not in p.parts}


def prepare(experiment_dir, data, run, config, paths_path):
    """Construct portable role bundles from explicit user-supplied inputs."""
    with Path(paths_path).open("rb") as stream:
        paths = tomllib.load(stream).get("data")
    if not isinstance(paths, dict):
        raise ValueError("Paths configuration requires a [data] table")
    paths = deepcopy(paths)
    # Explicit relative inputs are resolved against the supplied paths file.
    for key in ("source_manifest", "metadata", "feature_grid", "calibration_features", "calibration_manifest",
                "feature_directory", "raw_root", "checkpoint", "receipt"):
        if paths.get(key):
            value = Path(paths[key]).expanduser()
            paths[key] = str(value.resolve() if value.is_absolute() else (Path(paths_path).resolve().parent/value).resolve())
    if "manifests" in paths:
        paths["manifests"] = {k: str((Path(paths_path).resolve().parent/Path(v).expanduser()).resolve())
                              for k, v in paths["manifests"].items()}
    if paths.get("mode") == "raw":
        from .raw import acquire
        receipt = acquire(data.DATASET, paths["raw_root"], run/"acquisition",
            checkpoint=paths.get("checkpoint"), device=paths.get("device", "cpu"))
        paths = dict(mode="raw_receipt", receipt=str(receipt), receipt_sha256=file_hash(receipt))
    prepared = {}
    for context in config["contexts"]:
        manifest = prepare_context(paths, context, run/"prepared"/context)
        data.load(manifest, roles=("fit", "val"))
        prepared[context] = dict(manifest=str(manifest.relative_to(run)), sha256=file_hash(manifest))
    lock = dict(schema="mgpa-fresh-experiment-lock-v1", config=config, prepared=prepared,
        code_pins=code_pins(experiment_dir), runtime=read_json(run/"logs/runtime.json"),
        jobs=jobs(config), task_labels_used_for_fitting=False, source_ids_used_only_for_routed_baselines=True,
        raw_predictions_retained=True, all_adapter_fits_before_reader_fitting=True,
        preparation_may_validate_all_role_archives=True)
    write_json(run/"logs/lock.json", lock)
    log(run, "prepared", contexts=list(prepared), adapter_fitting_performed=False,
        task_labels_exposed_to_adapter=False)
    return lock


def locked(experiment_dir, run, config):
    """Authenticate a run's inputs, code, configuration, and numerical runtime."""
    lock = read_json(run/"logs/lock.json")
    if lock["config"] != config or lock["jobs"] != jobs(config) or lock["code_pins"] != code_pins(experiment_dir):
        raise ValueError("Scientific configuration or code changed; use a new run ID")
    if lock["runtime"] != read_json(run/"logs/runtime.json"):
        raise ValueError("Numerical runtime changed")
    for entry in lock["prepared"].values():
        if file_hash(run/entry["manifest"]) != entry["sha256"]:
            raise ValueError("Prepared manifest changed")
    return lock


def fit_adapter(method, records, state, data, config, seed, progress=None):
    """Construct one reported correction using only FIT and source-VAL arrays."""
    from mgpa import ClosedFormMGPA, CORAL, FEATMAP, IGBP, Identity, IterativeMGPA, LEACE
    from mgpa.core import permission_basis
    fit, val = data.fitting(records["fit"]), data.fitting(records["val"])
    if method == "mgpa_cf":
        pairs = recorded_calibration_pairs(records["fit"]) if data.DATASET == "recorded_ssvep" else fit.pairs
        return ClosedFormMGPA(**config["closed_form"]).fit(fit.x, fit.source, state["basis"], pairs,
                                                          validation_x=val.x)
    if method in ITERATIVE:
        hp = dict(config["model"])
        hp.update(target=method if method in {"fit_mean", "zero"} else "conditional",
                  critic_kind="linear" if method == "mgpa_linear" else "both")
        permission = method in {"measurement", "pca", "ungated"} or method.startswith("random_")
        # The published measurement arm uses ordinary quotient gradients.
        # Other edit permissions detach original Q0 because they may change it.
        hp["frozen_anchor"] = permission and method != "measurement"
        basis = state["basis"]
        if permission:
            random_seed = config["random_orientation_seeds"][int(method[-1])] if method.startswith("random_") else None
            basis = permission_basis("random" if random_seed is not None else method,
                fit.x, state["basis"], random_seed=random_seed)
        return IterativeMGPA(**hp).fit(fit.x, fit.source, basis, validation_x=val.x,
            validation_source=val.source, anchor_basis=state["basis"] if permission else None,
            seed=seed, on_stage=progress)
    if method == "igbp":
        return IGBP(**config["igbp"]).fit(fit.x, fit.source, validation_x=val.x,
                                        validation_source=val.source, seed=seed, on_stage=progress)
    if method == "identity":
        model = Identity()
    elif method == "leace":
        model = LEACE(ridge=0.)
    elif method.startswith("coral_"):
        model = CORAL(reference=int(method[-1]), ridge=1., moments="observed" if fit.pairs is None else "paired")
    elif method.startswith("featmap_"):
        model = FEATMAP(reference=int(method[-1]), ridge=0.)
    else:
        raise ValueError("Unknown published correction")
    return model.fit(fit.x, fit.source, pairs=fit.pairs)


def model_files(folder):
    """Hash every numeric model component excluding its separate fit receipt."""
    return {str(p.relative_to(folder)): file_hash(p) for p in sorted(folder.rglob("*"))
            if p.is_file() and p.name != "fit_receipt.json"}


def load_fit(run, lock, fit_id):
    """Authenticate and restore a portable correction from its fit receipt."""
    from mgpa import load_model
    folder = run/"models"/fit_id
    receipt = read_json(folder/"fit_receipt.json")
    if receipt["lock_hash"] != digest(lock) or receipt["files"] != model_files(folder):
        raise ValueError("Fitted model or receipt binding changed")
    return load_model(folder), receipt


def fit_all(run, lock, data):
    """Fit all declared adapters and seal their states before HEAD/EVAL access."""
    from mgpa import load_model, save_model
    from mgpa.core import Endpoint
    by_fit = defaultdict(list)
    for job in lock["jobs"]:
        by_fit[job["fit_id"]].append(job)
    receipts = {}
    for context in lock["config"]["contexts"]:
        records, state, _ = data.load(run/lock["prepared"][context]["manifest"], roles=("fit", "val"))
        for fit_id, points in by_fit.items():
            if points[0]["context"] != context:
                continue
            folder = run/"models"/fit_id
            if (folder/"fit_receipt.json").exists():
                _, receipt = load_fit(run, lock, fit_id)
                receipts[fit_id] = digest(receipt)
                continue
            if folder.exists() and any(folder.iterdir()):
                raise FileExistsError("Preserve incomplete model output; inspect before resuming")
            job = points[0]
            log(run, "fit_started", fit_id=fit_id)
            model = fit_adapter(job["method"], records, state, data, lock["config"], job["fit_seed"],
                progress=lambda row: log(run, "stage", fit_id=fit_id, details=row))
            validation = data.fitting(records["val"])
            endpoints = {}
            for endpoint in sorted({j["endpoint"] for j in points}):
                if endpoint == "native" or job["method"] == "identity":
                    point = Endpoint(stages=len(getattr(model, "stages_", [])), reason="fixed native output")
                else:
                    point = model.select_movement_budget(validation.x, float(endpoint[7:]))
                endpoints[endpoint] = asdict(point)
            save_model(model, folder)
            restored = load_model(folder)
            hashes = {}
            for endpoint, values in endpoints.items():
                point = Endpoint(**values)
                if point.attained:
                    source = validation.source if job["method"].startswith(("coral_", "featmap_")) else None
                    expected = model.transform(validation.x, endpoint=point, source_ids=source)
                    actual = restored.transform(validation.x, endpoint=point, source_ids=source)
                    np.testing.assert_array_equal(expected, actual)
                    from mgpa.audit.metrics import arrays_hash
                    hashes[endpoint] = arrays_hash(actual)
            receipt = dict(status="COMPLETE", fit_id=fit_id, lock_hash=digest(lock),
                method=job["method"], seed=job["fit_seed"], endpoints=endpoints,
                files=model_files(folder), validation_replay_hashes=hashes,
                task_labels_used=False, head_eval_loaded=False)
            write_json(folder/"fit_receipt.json", receipt)
            receipts[fit_id] = digest(receipt)
            log(run, "fit_complete", fit_id=fit_id, endpoints=endpoints)
    seal = dict(status="ALL_FITS_SEALED_BEFORE_HEAD_EVAL", lock_hash=digest(lock), fits=receipts)
    write_json(run/"logs/fit_seal.json", seal)
    log(run, "all_fits_sealed", fits=len(receipts))


def require_seal(run, lock):
    """Reject evaluation unless every planned fitted model is authenticated."""
    path = run/"logs/fit_seal.json"
    if not path.exists():
        raise ValueError("Fit every declared adapter before opening HEAD/EVAL")
    seal = read_json(path)
    expected = {job["fit_id"] for job in lock["jobs"]}
    if seal["lock_hash"] != digest(lock) or set(seal["fits"]) != expected:
        raise ValueError("The fit seal is incomplete or belongs to a different experiment")
    for fit_id in sorted(expected):
        _, receipt = load_fit(run, lock, fit_id)
        if digest(receipt) != seal["fits"][fit_id]:
            raise ValueError("Model changed after pre-evaluation seal")
    return seal


def evaluate_all(run, lock, data):
    """Replay fixed endpoints and fit the shared independent reader bank."""
    from mgpa.audit import Evaluator, EvaluationConfig
    from mgpa.audit.metrics import arrays_hash
    from mgpa.core import Endpoint
    seal = require_seal(run, lock)
    for context in lock["config"]["contexts"]:
        records, state, _ = data.load(run/lock["prepared"][context]["manifest"], roles=("head", "eval"))
        evaluator = Evaluator(records["head"], records["eval"], dataset=data.DATASET,
            basis=state["basis"], reader_seed=98, config=EvaluationConfig(include_predictions=True))
        completed_outputs = {}
        for job in [j for j in lock["jobs"] if j["context"] == context]:
            path = run/"tables/cells"/job["comparison"]/(job["cell_id"]+".json")
            if path.exists():
                cell = read_json(path)
                if cell["status"] == "COMPLETE":
                    if cell["evaluation_hash"] != digest(cell["evaluation"]):
                        raise ValueError("Saved evaluation changed")
                    completed_outputs[cell["evaluation"]["output_hash"]] = cell
                continue
            model, receipt = load_fit(run, lock, job["fit_id"])
            point = Endpoint(**receipt["endpoints"][job["endpoint"]])
            base = dict(schema="mgpa-published-cell-v1", job=job, **{k: job[k] for k in
                ("dataset", "comparison", "context", "method", "seed", "endpoint")},
                fit_receipt_hash=digest(receipt), fit_seal_hash=digest(seal), endpoint_metadata=asdict(point))
            if not point.attained:
                write_json(path, dict(base, status="UNATTAINED", reason="No native endpoint substitution"))
                continue
            source_aware = job["method"].startswith(("coral_", "featmap_"))
            hx, tx = [data.transform(model, records[r], endpoint=point, source_aware=source_aware)
                      for r in ("head", "eval")]
            output_dtype = np.float32 if data.DATASET == "recorded_ssvep" else np.float64
            output_hash = arrays_hash(np.asarray(hx, output_dtype), np.asarray(tx, output_dtype))
            alias = completed_outputs.get(output_hash)
            log(run, "evaluate_started", cell_id=job["cell_id"], source_reader_count=41 if data.DATASET == "recorded_ssvep" else 11)
            if alias is not None:
                result = deepcopy(alias["evaluation"])
                result["endpoint_id"] = job["cell_id"]
                result["endpoint_metadata"] = asdict(point)
            else:
                result = evaluator.evaluate(job["cell_id"], hx, tx, endpoint_metadata=asdict(point))
            cell = dict(base, status="COMPLETE", evaluation=result, evaluation_hash=digest(result),
                reused_identical_output_from=alias["job"]["cell_id"] if alias is not None else None)
            write_json(path, cell)
            completed_outputs[output_hash] = cell
            log(run, "evaluate_complete", cell_id=job["cell_id"], summary=result["summary"])
    write_json(run/"logs/evaluation_complete.json", dict(status="COMPLETE", lock_hash=digest(lock),
        planned_cells=len(lock["jobs"]), fit_seal_hash=digest(seal)))


def report_fresh(run, lock):
    """Aggregate saved cells with participant-wise, maximum-aware bootstrap."""
    from mgpa.audit.reporting import aggregate, ensemble_summary, paired_contrast
    groups = defaultdict(list)
    hashes, unavailable = {}, []
    for job in lock["jobs"]:
        path = run/"tables/cells"/job["comparison"]/(job["cell_id"]+".json")
        cell = read_json(path)
        hashes[str(path.relative_to(run))] = file_hash(path)
        if cell["status"] == "UNATTAINED":
            unavailable.append(job)
            continue
        if cell["evaluation_hash"] != digest(cell["evaluation"]):
            raise ValueError("Saved cell changed before reporting")
        groups[(job["comparison"], job["method"], job["endpoint"])].append(cell)
    hp = lock["config"]["report"]
    arguments = dict(resamples=hp["bootstrap_resamples"], seed=hp["bootstrap_seed"])
    expected_counts = defaultdict(int)
    for job in lock["jobs"]:
        expected_counts[job["comparison"], job["method"], job["endpoint"]] += 1
    omitted_groups = [dict(comparison=k[0], method=k[1], endpoint=k[2],
        completed=len(groups.get(k, [])), planned=count,
        reason="Incomplete declared endpoint population; no partial-cohort headline")
        for k, count in expected_counts.items() if len(groups.get(k, [])) != count]
    groups = {key: cells for key, cells in groups.items() if len(cells) == expected_counts[key]}
    summaries = [{"comparison": key[0], "method": key[1], "endpoint": key[2], **aggregate(cells, **arguments)}
                 for key, cells in groups.items()]
    # Maximize each orientation's reader bank first, then average orientations.
    for comparison in lock["config"]["comparisons"]:
        if not all(f"random_{i}" in comparison["methods"] for i in range(5)):
            continue
        for endpoint in comparison["endpoints"]:
            orientation_cells = [groups.get((comparison["name"], f"random_{i}", endpoint), []) for i in range(5)]
            expected = len(lock["config"]["contexts"])*len(comparison["fit_seeds"])
            if all(len(cells) == expected for cells in orientation_cells):
                summaries.append(dict(comparison=comparison["name"], method="random_mean5", endpoint=endpoint,
                    fit_seeds=comparison["fit_seeds"], random_orientations=5,
                    **{m: ensemble_summary(orientation_cells, m, **arguments) for m in ("S", "Tf", "Tr", "M")}))
    contrasts = []
    for (comparison, method, endpoint), cells in groups.items():
        for reference in ("identity", "leace"):
            if reference == method or (comparison, reference, endpoint) not in groups:
                continue
            controls = groups[comparison, reference, endpoint]
            if {c["seed"] for c in cells} != {c["seed"] for c in controls}:
                continue
            contrasts.append(dict(comparison=comparison, method=method, reference=reference, endpoint=endpoint,
                metrics={m: paired_contrast(cells, controls, m, **arguments) for m in ("S", "Tf", "Tr", "M")}))
    output = dict(status="COMPLETE" if not unavailable else "COMPLETE_WITH_UNATTAINED_ENDPOINTS",
        dataset=lock["config"]["dataset"], summaries=summaries, paired_contrasts=contrasts,
        unavailable=unavailable, omitted_groups=omitted_groups, cell_hashes=hashes,
        interpretation="Participant-bootstrap intervals condition on fitted maps/readers; fit repetitions and random orientations are not extra participants.")
    write_json(run/"tables/summary.json", output, replace=True)
    from mgpa.reporting import report_run
    report_run(run/"tables/summary.json")
    return output


def verify_fresh(run, lock, data):
    """Replay every fitted output and independently verify saved result hashes.

    Does not train adapters or readers. This is output/provenance verification;
    it is not a claim of an independent reimplementation of each metric.
    """
    from mgpa.audit.metrics import arrays_hash
    from mgpa.core import Endpoint
    require_seal(run, lock)
    checked = {}
    for context in lock["config"]["contexts"]:
        records, _, _ = data.load(run/lock["prepared"][context]["manifest"], roles=("head", "eval"))
        for job in [j for j in lock["jobs"] if j["context"] == context]:
            path = run/"tables/cells"/job["comparison"]/(job["cell_id"]+".json")
            cell = read_json(path)
            model, receipt = load_fit(run, lock, job["fit_id"])
            point = Endpoint(**receipt["endpoints"][job["endpoint"]])
            if not point.attained:
                if cell["status"] != "UNATTAINED":
                    raise ValueError("Unattained budget was replaced")
            else:
                hx, tx = [data.transform(model, records[r], endpoint=point,
                    source_aware=job["method"].startswith(("coral_", "featmap_"))) for r in ("head", "eval")]
                output_dtype = np.float32 if data.DATASET == "recorded_ssvep" else np.float64
                if (arrays_hash(np.asarray(hx, output_dtype), np.asarray(tx, output_dtype)) != cell["evaluation"]["output_hash"]
                        or digest(cell["evaluation"]) != cell["evaluation_hash"]):
                    raise ValueError("Portable adapter replay or saved evaluation changed")
            checked[str(path.relative_to(run))] = file_hash(path)
    result = dict(status="PASS", lock_hash=digest(lock), verified_cells=checked,
                  adapter_or_reader_fitting=False, task_labels_used_for_adapter_selection=False)
    write_json(run/"logs/verification.json", result)
    return result


def main(experiment_dir, data, argv=None):
    """Dispatch preparation, fitting, scoring, reporting, or verification from a narrow CLI."""
    experiment_dir = Path(experiment_dir).resolve()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("plan", "prepare", "fit", "evaluate", "report", "verify", "all"))
    parser.add_argument("--config", type=Path, default=experiment_dir/"configs/paper.toml")
    parser.add_argument("--paths", type=Path, help="Local [data] acquisition paths; required for prepare/all")
    parser.add_argument("--run-id", default="reproduction_v1")
    parser.add_argument("--published", action="store_true", help="Report exported published results, without new fitting")
    args = parser.parse_args(argv)
    config = load_config(args.config, data.DATASET)
    if args.phase == "plan":
        print(json.dumps(dict(config=config, planned_jobs=jobs(config)), indent=2))
        return
    if args.published:
        if args.phase != "report":
            parser.error("--published is only a saved-result reporting operation")
        from mgpa.reporting import report_experiment
        report_experiment(experiment_dir)
        return
    run = experiment_dir/"artifacts/runs"/safe_id(args.run_id)
    with run_guard(run/"logs"):
        write_json(run/"logs/runtime.json", runtime(run/"runtime", config["threads"]))
        if args.phase in ("prepare", "all"):
            if args.paths is None:
                parser.error("--paths is required for preparation")
            prepare(experiment_dir, data, run, config, args.paths)
        lock = locked(experiment_dir, run, config)
        if args.phase in ("fit", "all"):
            fit_all(run, lock, data)
        if args.phase in ("evaluate", "all"):
            evaluate_all(run, lock, data)
        if args.phase in ("report", "all"):
            report_fresh(run, lock)
        if args.phase in ("verify", "all"):
            verify_fresh(run, lock, data)
