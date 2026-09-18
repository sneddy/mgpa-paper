"""Standalone, output-only evaluation for the four final experimental settings."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import re

import numpy as np

from .metrics import (arrays_hash, available_mean, binary_log_loss, binary_metrics,
                      json_compatible, movement, multiclass_metrics, summarize)
from .readers import (SOURCE_FAMILIES, fit_task_model, fit_token_task_path,
                      predict_token_task, source_bank)
from .token_readers import JOBS, RESIDUAL_JOBS, fit_reader, residual_tokens


@dataclass(frozen=True)
class EvaluationConfig:
    """Choose fixed-bank smoke diagnostics or the complete published evaluation."""
    smoke: bool = False
    include_predictions: bool = False


_DATASETS = {"controlled": "controlled", "controlled_ssvep": "controlled", "ssvep_controlled": "controlled",
             "recorded": "recorded", "recorded_ssvep": "recorded", "ssvep_recorded_wet_dry": "recorded",
             "n170": "n170", "temporal_n170": "n170", "flex_n170": "n170"}


def _role(original, *, recorded, name):
    """Validate role identities, labels and representation shapes without fitting."""
    key = "x_observed" if recorded else "x"
    if recorded and "x" in original:
        raise ValueError("Recorded SSVEP must use unpaired x_observed; invented pairs are forbidden")
    x = np.asarray(original.get(key))
    if x.ndim != 3 or not x.size or x.dtype.kind != "f" or not np.isfinite(x).all():
        raise ValueError(f"{name} needs finite {key} paired vectors or observed token grid")
    if not recorded and x.shape[1] != 2:
        raise ValueError("Paired evaluations require exactly two views")
    x = np.asarray(x, dtype=np.float32 if recorded else np.float64).copy()
    result = {key: x}
    for field, aliases in (("subject", ("subject_observed",)), ("event_id", ("event_id_observed",))):
        present = [k for k in (field, *aliases) if k in original]
        if not present:
            raise ValueError(f"Missing {name} {field}")
        value = np.asarray(original[present[0]])
        if value.shape != (len(x),) or any(not np.array_equal(value, np.asarray(original[k])) for k in present[1:]):
            raise ValueError(f"Misaligned or conflicting {field} aliases")
        if any(v is None or not str(v).strip() for v in value.tolist()):
            raise ValueError(f"Missing {name} {field} identity")
        if value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise ValueError(f"Nonfinite {name} {field} identity")
        result[field] = value.astype(str).copy()
    if len(set(result["event_id"])) != len(x):
        raise ValueError(f"Duplicate {name} event IDs")
    for field in ("y", "y_binary", "y_multiclass", "u"):
        if field in original:
            value = np.asarray(original[field])
            if value.shape != (len(x),) or value.dtype.kind not in "biu":
                raise ValueError(f"Misaligned or noninteger {field}")
            result[field] = value.copy()
    if recorded:
        for field in ("u", "y_multiclass"):
            if field not in result:
                raise ValueError(f"Recorded evaluation requires {field}")
        if not np.isin(result["u"], [0, 1]).all() or not np.isin(result["y_multiclass"], np.arange(12)).all():
            raise ValueError("Recorded source/task labels outside native ranges")
        if "y" in result and not np.array_equal(result["y"], result["y_multiclass"]):
            raise ValueError("Recorded primary task must be actual twelve-class task")
        result["y"] = result["y_multiclass"].copy()
        if name == "HEAD":
            for u in (0, 1):
                if set(result["y"][result["u"] == u]) != set(range(12)):
                    raise ValueError("Each recorded HEAD source requires all twelve task classes")
    elif "y" not in result or not np.isin(result["y"], [0, 1]).all():
        raise ValueError("Paired evaluation requires binary task labels")
    elif name == "HEAD" and set(result["y"].tolist()) != {0, 1}:
        raise ValueError("HEAD requires both binary task classes")
    for value in result.values():
        value.flags.writeable = False
    return result


class Evaluator:
    """One immutable HEAD/EVAL split and reader seed, reused across endpoints.

    The numerical response contains participant, reader and task-route rows.
    Raw predictions are optional JSON lists. The evaluator never writes files,
    fits an adapter, observes FIT/VAL, or selects an endpoint using EVAL.
    """
    def __init__(self, original_head, original_eval, *, dataset, basis=None, reader_seed=98, config=None):
        """Bind a participant-disjoint HEAD/EVAL split and independent reader bank."""
        if dataset not in _DATASETS:
            raise ValueError(f"Unsupported evaluation dataset: {dataset}")
        if isinstance(reader_seed, bool) or not isinstance(reader_seed, int):
            raise ValueError("Reader seed must be an integer")
        self.dataset, self.seed = _DATASETS[dataset], reader_seed
        self.recorded = self.dataset == "recorded"
        self.config = config if isinstance(config, EvaluationConfig) else EvaluationConfig(**(config or {}))
        self.key = "x_observed" if self.recorded else "x"
        self.head = _role(original_head, recorded=self.recorded, name="HEAD")
        self.test = _role(original_eval, recorded=self.recorded, name="EVAL")
        if self.head[self.key].shape[1:] != self.test[self.key].shape[1:]:
            raise ValueError("HEAD/EVAL feature dimensions mismatch")
        if set(self.head["subject"]) & set(self.test["subject"]) or set(self.head["event_id"]) & set(self.test["event_id"]):
            raise ValueError("HEAD/EVAL participant or event leakage")
        self.people = sorted(set(self.test["subject"]))
        self.basis = None if basis is None else np.asarray(basis, dtype=np.float64).copy()
        if self.basis is not None:
            if (self.basis.ndim != 2 or self.basis.shape[0] != self.head[self.key].shape[-1]
                    or not np.isfinite(self.basis).all()
                    or not np.allclose(self.basis.T @ self.basis, np.eye(self.basis.shape[1]), atol=2e-6, rtol=2e-6)):
                raise ValueError("Declared geometry must already be a finite orthonormal feature basis")
            self.basis.flags.writeable = False
        self.source_families = ("linear", "whitened_linear", "gelu_e12_r0") if self.config.smoke else SOURCE_FAMILIES
        self.task_epochs = (4,) if self.config.smoke else (4, 12, 50)
        self._task_models, self._token_banks, self._results, self._bindings = {}, {}, {}, {}
        self._residual_bank = None
        self.input_hash = arrays_hash(*(role[k] for role in (self.head, self.test) for k in sorted(role)))

    def _task_model(self, x, y):
        """Reuse identical fitted HEAD readouts across correction outputs."""
        key = arrays_hash(x, y)
        if key not in self._task_models:
            self._task_models[key] = fit_task_model(x, y)
        return self._task_models[key]

    def _token_bank(self, x):
        """Cache pooled and source-specific HEAD task paths for identical inputs."""
        key = arrays_hash(x, self.head["y"], self.head["u"])
        if key not in self._token_banks:
            bank = {}
            for route, seed in (("both", self.seed), ("0", self.seed + 10), ("1", self.seed + 11)):
                select = np.ones(len(x), dtype=bool) if route == "both" else self.head["u"] == int(route)
                models, details = fit_token_task_path(x[select], self.head["y"][select], seed=seed, checkpoints=self.task_epochs)
                bank[route] = {"models": models, "details": details}
            self._token_banks[key] = bank
        return self._token_banks[key]

    def _tasks(self, hx, tx, participants, predictions):
        """Score frozen and refitted task heads in both transfer directions."""
        task_history = {}
        if self.recorded:
            for mode, fitting in (("frozen_original", self.head[self.key]), ("refit", hx)):
                bank = self._token_bank(fitting)
                task_history[mode] = {route: fit["details"] for route, fit in bank.items()}
                for route, fit in bank.items():
                    for epoch, model in fit["models"].items():
                        logits = np.asarray(predict_token_task(model, tx), dtype=np.float64)
                        key = f"task__{mode}__train{route}__e{epoch}"
                        predictions[key + "__logits"] = logits
                        predictions[key + "__prediction"] = logits.argmax(axis=1).astype(np.int8)
                        for target in ("both", "0", "1"):
                            selected = np.ones(len(tx), dtype=bool) if target == "both" else self.test["u"] == int(target)
                            for person in self.people:
                                select = selected & (self.test["subject"] == person)
                                if not np.any(select):
                                    continue
                                participants.append({"kind": "task", "mode": mode, "train_view": route, "test_view": target,
                                                     "task_epoch": epoch, "participant": person, "task": "ssvep_12way",
                                                     **multiclass_metrics(self.test["y"][select], logits[select])})
        else:
            flat = tx.reshape(-1, tx.shape[-1])
            fits = [(mode, view, fitting[:, view], self.head["y"])
                    for view in (0, 1) for mode, fitting in (("refit", hx), ("frozen_original", self.head[self.key]))]
            fits.append(("pooled_refit", "pooled", hx.reshape(-1, hx.shape[-1]), np.repeat(self.head["y"], 2)))
            for mode, route, fitting, labels in fits:
                scores = self._task_model(fitting, labels).decision_function(flat).reshape(len(tx), 2)
                for target in (0, 1):
                    score = scores[:, target]
                    key = f"task__{mode}__train{route}__test{target}"
                    predictions[key + "__score"] = score.copy()
                    predictions[key + "__prediction"] = (score > 0).astype(np.int8)
                    for person in self.people:
                        select = self.test["subject"] == person
                        participants.append({"kind": "task", "mode": mode, "train_view": route, "test_view": target,
                                             "participant": person, **binary_metrics(self.test["y"][select], score[select])})
        return task_history


    def _full_token_sources(self, hx, tx, participants, predictions, metadata, histories):
        """Append direct and invariant-residual readers to the pooled source bank."""
        families = list(self.source_families)
        residual_head = residual_tokens(self.head[self.key])
        residual_eval = residual_tokens(self.test[self.key])
        # Every published adapter adds the same displacement to all tokens.
        drift = float(np.max(np.abs(residual_tokens(tx).astype(float) - residual_eval.astype(float))))
        if drift > 2e-4:
            raise ValueError("The residual-reader reuse requires a common-offset correction")
        if self._residual_bank is None:
            self._residual_bank = {
                job: fit_reader(residual_head, self.head["u"], residual_eval,
                                reader=job, smoke=self.config.smoke)
                for job in RESIDUAL_JOBS
            }
        banks = [("full", {job: fit_reader(hx, self.head["u"], tx, reader=job,
                                         smoke=self.config.smoke) for job in JOBS}),
                 ("residual", self._residual_bank)]
        for scope, bank in banks:
            for job, output in bank.items():
                metadata.append(output["metadata"])
                histories[f"{scope}__{job}"] = output["training_history"]
                for family, score in output["scores"].items():
                    name = f"{scope}__{family}"
                    families.append(name)
                    predictions[f"source__{name}__score"] = score
                    for person in self.people:
                        take = self.test["subject"] == person
                        participants.append({"kind": "source", "family": name,
                            "participant": person, "critic_epoch": None, "critic_restart": None,
                            **binary_metrics(self.test["u"][take], score[take]),
                            "pooled_log_loss": float(binary_log_loss(score[take], self.test["u"][take]).mean())})
        self.source_families = tuple(dict.fromkeys(families))

    def _evaluate(self, hx, tx):
        """Fit independent readers and aggregate participant outcomes for one output."""
        participants, predictions, metadata, histories = [], {}, [], {}
        if self.recorded:
            # Deliberately native float32 temporal mean, never token flattening.
            train, test = hx.mean(axis=1), tx.mean(axis=1)
            uh, ut, source_subject = self.head["u"], self.test["u"], self.test["subject"]
        else:
            train, test = hx.reshape(-1, hx.shape[-1]), tx.reshape(-1, tx.shape[-1])
            uh, ut = np.tile([0, 1], len(hx)), np.tile([0, 1], len(tx))
            source_subject = np.repeat(self.test["subject"], 2)
        predictions.update(test_subject=self.test["subject"], test_event_id=self.test["event_id"],
                           test_y=self.test["y"], source_u=ut if self.recorded else ut.reshape(len(tx), 2))
        for family, score, details, epoch, restart, history in source_bank(train, uh, test, seed=self.seed, smoke=self.config.smoke):
            if score.shape != (len(test),) or not np.isfinite(score).all():
                raise FloatingPointError("Misaligned or nonfinite source predictions")
            loss = binary_log_loss(score, ut)
            suffix = "pooled" if self.recorded else "full"
            predictions[f"source__{family}__{suffix}_score"] = score if self.recorded else score.reshape(len(tx), 2)
            predictions[f"source__{family}__{suffix}_loss"] = loss if self.recorded else loss.reshape(len(tx), 2)
            for person in self.people:
                select = source_subject == person
                participants.append({"kind": "source", "family": family, "participant": person,
                                     "critic_epoch": epoch, "critic_restart": restart,
                                     **binary_metrics(ut[select], score[select]), f"{suffix}_log_loss": float(loss[select].mean())})
            metadata.append(details)
            if history is not None:
                histories[f"gelu_r{restart}"] = history
        if self.recorded:
            self._full_token_sources(hx, tx, participants, predictions, metadata, histories)
        task_history = self._tasks(hx, tx, participants, predictions)
        for person in self.people:
            select = self.test["subject"] == person
            participants.append({"kind": "movement", "participant": person,
                                 **movement(self.test[self.key][select], tx[select], observed=self.recorded, basis=self.basis)})
        native = {"participants": [], "predictions": {}}
        summary, person_summary = summarize(participants, native["participants"], self.people, recorded=self.recorded,
                                            source_families=self.source_families, task_epoch=max(self.task_epochs))
        if self.recorded:
            means = summary["source_auroc_by_family"]
            summary["S_all"] = summary["S"]
            summary["S_mean"] = max(means[k] for k in SOURCE_FAMILIES if k in means)
        summary.update(dataset=self.dataset, reader_seed=self.seed, source_reader_count=len(self.source_families),
                       scientific_status="SMOKE_NOT_MAIN_RESULT" if self.config.smoke else "FIXED_PROTOCOL_EVALUATION",
                       source_input_contract="observed_float32_token_mean" if self.recorded else "full_output_event_major_paired_vectors",
                       task_primary_epoch=max(self.task_epochs) if self.recorded else None)
        grouped_tasks = {}
        for row in participants:
            if row["kind"] != "task":
                continue
            key = (row["mode"], row["train_view"], row["test_view"], row.get("task_epoch"))
            grouped_tasks.setdefault(key, []).append(row)
        task_summaries = []
        for (mode, train_view, test_view, epoch), rows in grouped_tasks.items():
            fields = ("macro_auroc", "accuracy", "balanced_accuracy", "cross_entropy") if self.recorded else ("auroc", "balanced_accuracy", "recall0", "recall1")
            task_summaries.append({"mode": mode, "train_view": train_view, "test_view": test_view, "task_epoch": epoch,
                                   "participants": len(rows), **{f"participant_mean_{key}": available_mean(r[key] for r in rows) for key in fields}})
        geometry = movement(self.test[self.key], tx, observed=self.recorded, basis=self.basis)
        source_summaries = []
        for family in self.source_families:
            rows = [r for r in participants if r["kind"] == "source" and r["family"] == family]
            fields = ("auroc", "balanced_accuracy", "recall0", "recall1", "pooled_log_loss" if self.recorded else "full_log_loss")
            source_summaries.append({"family": family, "critic_epoch": rows[0]["critic_epoch"],
                                     "critic_restart": rows[0]["critic_restart"], "participants": len(rows),
                                     **{f"participant_mean_{key}": available_mean(r[key] for r in rows) for key in fields}})
        main = {"participants": participants, "summary": {"task": task_summaries, "source": source_summaries,
                "strongest_source": {"participant_mean_auroc": summary["S"],
                                     "selection": "descriptive maximum across the complete prespecified bank, not a selected model"},
                "attacker_metadata": metadata,
                "saturation_training_histories": histories, "task_training_histories": task_history,
                "source_epochs": [12] if self.config.smoke else [12, 50, 200], "source_restarts": 1 if self.config.smoke else 3,
                "movement": {"head": movement(self.head[self.key], hx, observed=self.recorded), "test": geometry}}}
        if self.config.include_predictions:
            main["predictions"] = predictions
        else:
            native.pop("predictions", None)
        return {"summary": summary, "participant_summary": person_summary, "main": main, "native": native, "geometry": geometry}

    def evaluate(self, endpoint_id, transformed_head, transformed_eval, *, endpoint_metadata=None):
        """Evaluate a locked output, reusing only numerically identical corrections."""
        if not isinstance(endpoint_id, str) or not re.fullmatch(r"[A-Za-z0-9_.+-]+", endpoint_id):
            raise ValueError("A safe nonempty endpoint ID is required")
        metadata = json.loads(json.dumps(endpoint_metadata or {}, sort_keys=True, allow_nan=False))
        if not isinstance(metadata, dict):
            raise ValueError("Endpoint metadata must be a JSON dictionary")
        outputs = []
        for values, role in ((transformed_head, self.head), (transformed_eval, self.test)):
            if isinstance(values, dict):
                for field in ("subject", "event_id", "y", "y_binary", "y_multiclass", "u"):
                    if field in values and field in role and not np.array_equal(np.asarray(values[field]), role[field]):
                        raise ValueError("Transformed role metadata must retain original identities and labels")
                values = values[self.key]
            x = np.asarray(values)
            if x.shape != role[self.key].shape or x.dtype.kind != "f" or not np.isfinite(x).all():
                raise ValueError("Finite transformed arrays must retain exact original shape")
            outputs.append(np.asarray(x, dtype=np.float32 if self.recorded else np.float64))
        key = arrays_hash(*outputs)
        binding = {"output_hash": key, "metadata": metadata}
        if endpoint_id in self._bindings and self._bindings[endpoint_id] != binding:
            raise ValueError("Endpoint ID cannot be rebound to a different output or metadata")
        reused = key in self._results
        if not reused:
            self._results[key] = json_compatible(self._evaluate(*outputs))
        self._bindings[endpoint_id] = binding
        result = deepcopy(self._results[key])
        result.update(endpoint_id=endpoint_id, endpoint_metadata=metadata, output_hash=key, input_hash=self.input_hash,
                      reader_seed=self.seed, evaluation_config=asdict(self.config), numerical_output_reused=reused)
        # Fail before return if accidental NaN/NumPy values escaped the contract.
        json.dumps(result, allow_nan=False)
        return result
