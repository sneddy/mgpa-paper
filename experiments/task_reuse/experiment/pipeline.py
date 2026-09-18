"""Independent execution of the published P300-to-N170/MMN reuse protocol."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform

import numpy as np

from mgpa.data import reuse as data
from . import evaluation as ev, policy, protocol

TASKS = data.TASKS
SEEDS = (17, 29, 43)


def digest(value):
    """Hash canonical JSON to bind scientific configuration and candidate state."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def tree_pins(root, directory):
    """Hash every file below a directory using portable run-relative names."""
    return {str(p.relative_to(root)): data.sha256(p) for p in sorted(directory.rglob("*")) if p.is_file()}


def verify_pins(root, pins):
    """Reject changed artifacts and paths escaping the recorded run directory."""
    for name, expected in pins.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or data.sha256(path) != expected:
            raise ValueError(f"Frozen run artifact changed: {name}")


def config(path):
    """Load and validate the fixed manuscript recipe and selection grids."""
    value = data.read_json(path)
    required = dict(schema="mgpa-task-reuse-v1", seed=17, fit_seeds=list(SEEDS),
        gate_rank=8, pca_rank=256, leading_task="p300", transfer_tasks=["n170", "mmn"],
        head_seed=1000, head_C_grid=[1e-6, 1e-5, 1e-4, .001, .01, .1, 1, 10],
        head_association=1, fit_association=1, max_probability=.9,
        eval_associations=dict(aligned=1, independent=0, reversed=-1),
        utility_guardrail=-.01, recipe_tolerance=.002, bootstrap_resamples=20000, bootstrap_seed=1101,
        strength_grid=[i/50 for i in range(51)] + [1.1, 1.2, 1.3],
        leace_ridges=[0, .001, .01, .1, 1], coral_ridges=[.01, .1, 1, 10],
        featmap_ridges=[0, 1, 10, 100, 1000], references=[0, 1])
    for key, expected in required.items():
        if value.get(key) != expected:
            raise ValueError(f"Published scientific configuration differs: {key}")
    expected_iterative = dict(max_stages=48, critic_kind="both", critic_epochs=100,
        hidden_width=128, batch_size=256, early_stopping=True, patience=10,
        critic_selection="best_val", learning_rate=.0003, anchor_ridge_alpha=10.,
        max_step=.025, floor=.1, damping=.001)
    if value.get("iterative") != expected_iterative:
        raise ValueError("Expected published pooled, max_step=.025 Iterative recipe")
    return value


def check_config(root, cfg):
    """Prevent a run directory from being reused with different scientific settings."""
    snapshot = root / "config.json"
    if snapshot.is_file() and data.read_json(snapshot) != cfg:
        raise ValueError("Run configuration changed; use a new run directory")


def prepare(root, cfg, manifest, *, checkpoint=None, device="cpu"):
    """Prepare P300 development roles and pin their coordinates and configuration."""
    root = Path(root)
    check_config(root, cfg)
    if (root / "selection_lock.json").exists():
        raise FileExistsError("Selections already locked; choose a new run directory for preparation")
    data.write_json(root / "config.json", cfg)
    result = data.prepare(manifest, root, checkpoint=checkpoint, device=device)
    data.write_json(root / "logs/prepared_hashes.json", {
        **tree_pins(root, root / "prepared"), **tree_pins(root, root / "models"),
        "config.json": data.sha256(root / "config.json"),
        "logs/preparation.json": data.sha256(root / "logs/preparation.json")})
    return result


def common_bank(roles, cfg):
    """Build the shared deduplicated FIT/CAL bank and observed DEV endpoints."""
    fit, cal, dev = (roles[k] for k in ("fit", "cal", "dev"))
    uf = protocol.assign_endpoints(fit["y"], fit["subject"], cfg["fit_association"],
        cfg["seed"]+2000, cfg["max_probability"])
    uv = protocol.assign_endpoints(dev["y"], dev["subject"], cfg["fit_association"],
        cfg["seed"]+3000, cfg["max_probability"])
    xf = np.concatenate([protocol.select_endpoints(fit["x"], uf), cal["x"].reshape(-1, cal["x"].shape[-1])])
    source = np.concatenate([uf, np.tile([0, 1], len(cal["x"]))]).astype(np.int8)
    keys = [(str(p), str(e), int(u)) for p, e, u in zip(fit["subject"], fit["event_id"], uf)]
    keys += [(str(p), str(e), u) for p, e in zip(cal["subject"], cal["event_id"]) for u in (0, 1)]
    if len(keys) != len(set(keys)) or len(xf) != 3060 or len(uv) != 1332:
        raise ValueError("Common endpoint bank differs from the 3,060/1,332 published contract")
    receipt = dict(endpoint_keys=keys, rows=len(xf), observed_rows=len(uf), calibration_rows=2*len(cal["x"]),
        labels_used_only_for_experimental_device_assignment=True, all_methods_receive_identical_rows=True)
    return xf, source, protocol.select_endpoints(dev["x"], uv), uv, receipt


def heads_for(root, task, role, cfg, *, create=False):
    """Load frozen task heads or fit them using only the prescribed HEAD role."""
    path = root / "models/heads" / f"{task}.json"
    if path.exists():
        record = data.read_json(path)
    elif create:
        _, record = ev.build_heads(role, cfg)
        data.write_json(path, record)
    else:
        raise FileNotFoundError(path)
    if record["seed"] != 1017 or record["head_people"] != [f"flex:{p}" for p in data.HEAD]:
        raise ValueError("Frozen HEAD identity/seed mismatch")
    return {name: ev.NumericHead.from_state(state) for name, state in record["heads"].items()}


def candidates(cfg):
    """Yield exactly the published affine grids and three Iterative fit seeds."""
    yield dict(id="identity", family="identity", rank=0)
    yield dict(id="mgpa_cf", family="mgpa_cf", rank=8)
    for seed in SEEDS:
        yield dict(id=f"mgpa_iter_seed{seed}", family="mgpa_iter", rank=8, seed=seed)
    for ridge in cfg["leace_ridges"]:
        yield dict(id=f"leace_ridge{ridge:g}", family="leace", rank=1, ridge=ridge)
    for family in ("coral", "featmap"):
        for ridge in cfg[f"{family}_ridges"]:
            for reference in cfg["references"]:
                yield dict(id=f"{family}_ref{reference}_ridge{ridge:g}", family=family,
                    rank=0, ridge=ridge, reference=reference)


def build_model(candidate, cfg, xf, uf, pairs, basis, xv, uv):
    """Fit one public estimator without providing downstream task labels."""
    from mgpa import ClosedFormMGPA, IterativeMGPA
    from mgpa.baselines import Identity, LEACE, CORAL, FEATMAP
    family = candidate["family"]
    if family == "identity":
        return Identity().fit(xf)
    if family == "mgpa_cf":
        return ClosedFormMGPA().fit(xf, uf, basis, pairs)
    if family == "mgpa_iter":
        settings = {k: v for k, v in cfg["iterative"].items() if k != "critic_selection"}
        return IterativeMGPA(**settings).fit(xf, uf, basis,
            validation_x=xv, validation_source=uv, seed=candidate["seed"])
    if family == "leace":
        return LEACE(ridge=candidate["ridge"]).fit(xf, uf)
    if family == "coral":
        return CORAL(reference=candidate["reference"], ridge=candidate["ridge"], moments="observed").fit(xf, uf, pairs=pairs)
    return FEATMAP(reference=candidate["reference"], ridge=candidate["ridge"]).fit(xf, uf, pairs=pairs)


def endpoints(candidate, cfg):
    """Enumerate the candidate's declared affine strengths or stage prefixes."""
    if candidate["family"] == "identity":
        return [("strength_0", 0)]
    if candidate["family"] == "mgpa_iter":
        return [(f"fixed_{stage}", stage) for stage in range(49)]
    return [(f"strength_{alpha:g}", alpha) for alpha in cfg["strength_grid"]]


def transform(model, candidate, role, endpoint):
    """Replay a chosen endpoint, supplying source IDs only to routed baselines."""
    x = role["x"]
    flat = x.reshape(-1, x.shape[-1])
    if candidate["family"] == "mgpa_iter":
        value = int(endpoint.removeprefix("fixed_"))
        corrected = model.transform(flat, stages=value)
    else:
        kwargs = {}
        if candidate["family"] in ("coral", "featmap"):
            kwargs["source_ids"] = np.tile([0, 1], len(x)).astype(np.int8)
        corrected = model.transform(flat, strength=float(endpoint.removeprefix("strength_")), **kwargs)
    return corrected.reshape(x.shape)


def fit(root, cfg):
    """Fit all declared recipes; choose only on P300 DEV; freeze one final lock."""
    from mgpa import save_model, load_model
    root = Path(root)
    check_config(root, cfg)
    verify_pins(root, data.read_json(root / "logs/prepared_hashes.json"))
    if (root / "selection_lock.json").exists():
        return validate_lock(root, cfg)
    roles = {name: data.load_arrays(root / "prepared" / f"p300_{name}.npz") for name in ("fit", "cal", "head", "dev")}
    sets = {k: set(v["subject"]) for k, v in roles.items() if k != "cal"}
    if any(sets[a] & sets[b] for a in sets for b in sets if a < b):
        raise ValueError("Participant leakage across P300 roles")
    heads = heads_for(root, "p300", roles["head"], cfg, create=True)
    identity = ev.identity_metrics(heads, roles["dev"], cfg)
    xf, uf, xv, uv, bank = common_bank(roles, cfg)
    basis, geometry = data.calibration_basis(roles["cal"]["x"], cfg["gate_rank"])
    data.save_arrays(root / "models/gate.npz", {"basis": basis})
    data.write_json(root / "logs/common_fit_bank.json", bank)
    data.write_json(root / "logs/geometry.json", geometry)
    import importlib.metadata
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "torch")}
    data.write_json(root / "logs/environment.json", dict(python=platform.python_version(), packages=versions))
    cells = []
    for candidate in candidates(cfg):
        path = root / "tables/dev_cells" / f"{candidate['id']}.json"
        directory = root / "models/candidates" / candidate["id"]
        if path.exists():
            cell = data.read_json(path)
            if cell["candidate"] != candidate:
                raise ValueError("Candidate receipt changed")
            verify_pins(root, cell["model_pins"])
            cells.append(cell)
            continue
        print(f"Fitting {candidate['id']} on P300 FIT/CAL", flush=True)
        if directory.exists():
            raise FileExistsError(f"Incomplete candidate directory; preserve it and start a new run: {directory}")
        model = build_model(candidate, cfg, xf, uf, roles["cal"]["x"], basis, xv, uv)
        save_model(model, directory)
        data.write_json(root / "logs/fit" / f"{candidate['id']}.json", dict(candidate=candidate,
            input_population="P300 FIT observed endpoints plus both CAL endpoints, deduplicated",
            fit_rows=len(xf), source_validation_rows=len(xv), evaluation_opened=False,
            task_labels_passed_to_adapter=False, history=getattr(model, "history_", [])))
        model = load_model(directory)
        curve = []
        if candidate["family"] == "mgpa_iter":
            role = roles["dev"]
            states = ((f"fixed_{stage}", values.reshape(role["x"].shape))
                for stage, values in model.iter_transforms(role["x"].reshape(-1, role["x"].shape[-1])))
        else:
            states = ((endpoint, transform(model, candidate, roles["dev"], endpoint))
                for endpoint, _ in endpoints(candidate, cfg))
        for endpoint, values in states:
            row, _ = ev.endpoint_metrics(values,
                roles["dev"], heads, cfg, identity, endpoint)
            curve.append(row)
        selected = ev.select_curve(curve, cfg)
        row, scores = ev.endpoint_metrics(transform(model, candidate, roles["dev"], selected["endpoint"]),
            roles["dev"], heads, cfg, identity, selected["endpoint"])
        predictions = {**scores, **{k: roles["dev"][k] for k in ("y", "subject", "event_id")}}
        data.save_arrays(root / "predictions/dev" / f"{candidate['id']}.npz", predictions)
        cell = dict(candidate=candidate, selected=selected, dev_curve=curve,
            model_dir=str(directory.relative_to(root)), cell_path=str(path.relative_to(root)),
            model_pins=tree_pins(root, directory), binding=digest(dict(config=cfg, candidate=candidate)))
        data.write_json(path, cell)
        cells.append(cell)
    selected = ev.select_families([c for c in cells if c["candidate"]["family"] != "mgpa_iter"], cfg)
    # Each iterative seed is an independent fit, never an EVAL-selected recipe.
    for cell in cells:
        if cell["candidate"]["family"] == "mgpa_iter":
            selected[cell["candidate"]["id"]] = {k: cell[k] for k in ("candidate", "selected", "model_dir", "cell_path", "binding")}
    pins = {**tree_pins(root, root / "models"), **tree_pins(root, root / "tables/dev_cells"),
        **tree_pins(root, root / "predictions/dev"), **tree_pins(root, root / "logs/fit"),
        **data.read_json(root / "logs/prepared_hashes.json")}
    lock = dict(schema="mgpa-task-reuse-selection-v1", config_sha256=digest(cfg), selections=selected,
        pins=pins, selection_task="p300", selection_role="dev", eval_opened=False,
        target_task_outcomes_used=False, recipe="pooled conditional targets, max_step=.025, no averaging of states")
    data.write_json(root / "selection_lock.json", lock)
    return lock


def validate_lock(root, cfg):
    """Authenticate the complete P300 selection inventory and its pinned artifacts."""
    root = Path(root)
    lock = data.read_json(root / "selection_lock.json")
    if lock["schema"] != "mgpa-task-reuse-selection-v1" or lock["config_sha256"] != digest(cfg):
        raise ValueError("Selection lock configuration changed")
    if set(lock["selections"]) != {"identity", "mgpa_cf", "leace", "coral", "featmap", *(f"mgpa_iter_seed{s}" for s in SEEDS)}:
        raise ValueError("Final method inventory differs")
    verify_pins(root, lock["pins"])
    return lock


def evaluate(root, cfg, manifest, *, checkpoint=None, device="cpu"):
    """Apply locked corrections and frozen heads to all three EVAL tasks."""
    from mgpa import load_model
    root = Path(root)
    lock = validate_lock(root, cfg)
    for task in TASKS:
        data.prepare_evaluation_task(manifest, root, task, checkpoint=checkpoint, device=device)
        head_role = data.load_arrays(root / "prepared" / f"{task}_head.npz")
        role = data.load_arrays(root / "prepared" / f"{task}_eval.npz")
        if set(role["subject"]) != {f"flex:{p}" for p in data.EVAL}:
            raise ValueError("Exactly eight fixed EVAL participants required")
        heads = heads_for(root, task, head_role, cfg, create=(task != "p300"))
        identity = ev.identity_metrics(heads, role, cfg)
        for key, selection in lock["selections"].items():
            candidate = selection["candidate"]
            model = load_model(root / selection["model_dir"])
            endpoint = selection["selected"]["endpoint"]
            row, scores = ev.endpoint_metrics(transform(model, candidate, role, endpoint), role, heads, cfg, identity, endpoint)
            prediction_path = root / "predictions/eval" / f"{task}__{key}.npz"
            data.save_arrays(prediction_path, {**scores, **{k: role[k] for k in ("y", "subject", "event_id")}})
            record = dict(task=task, key=key, family=candidate["family"], fit_seed=candidate.get("seed"),
                selection=selection, result=row,
                independent_audit={h: {"coefficients": policy.coefficients(row["metrics"][h])} for h in heads},
                predictions_path=str(prediction_path.relative_to(root)), predictions_sha256=data.sha256(prediction_path),
                selection_lock_sha256=data.sha256(root / "selection_lock.json"),
                frozen_heads_sha256=data.sha256(root / "models/heads" / f"{task}.json"))
            data.write_json(root / "tables/eval_cells" / f"{task}__{key}.json", record)
        print(f"Evaluated frozen maps on {task.upper()}", flush=True)
    data.write_json(root / "logs/evaluation.json", dict(status="COMPLETE", tasks=list(TASKS),
        selection_lock_sha256=data.sha256(root / "selection_lock.json"),
        pins={**tree_pins(root, root / "tables/eval_cells"), **tree_pins(root, root / "predictions/eval"),
              **tree_pins(root, root / "models/heads")}))


def verify(root, cfg):
    """Verify the saved closure and recompute coefficients from frozen predictions."""
    root = Path(root)
    validate_lock(root, cfg)
    manifest = data.read_json(root / "logs/evaluation.json")
    verify_pins(root, manifest["pins"])
    for path in sorted((root / "tables/eval_cells").glob("*.json")):
        record = data.read_json(path)
        pred = data.load_arrays(root / record["predictions_path"])
        for head in ("biased", "control"):
            actual = policy.coefficients(protocol.evaluate_pair_predictions(pred[head], pred["y"], pred["subject"],
                cfg["eval_associations"], max_probability=cfg["max_probability"]))
            if digest(actual) != digest(record["independent_audit"][head]["coefficients"]):
                raise ValueError(f"Prediction-derived coefficients changed: {path}")
    return dict(status="PASS", predictions_verified=True, refitting=False)
