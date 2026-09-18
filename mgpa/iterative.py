"""Published Iterative MGPA and its gate, target, and linear-critic controls."""
from copy import deepcopy

import numpy as np
import torch
from sklearn.linear_model import Ridge
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .core import (Endpoint, EndpointUnavailableError, PortableEstimator, ScoreStage,
                   crossing_fraction, quotient, relative_movement, validate_basis)
from .critics import (CriticState, ShallowBinaryCritic, binary, fit_affine_critic,
                      fit_shallow_critic, isolated_torch_seed, matrix)


def _fit_validation_critic(x, source, xv, uv, seed, hp):
    """Published reuse GELU schedule: best VAL BCE epoch, patience 10."""
    value = int(seed) % (2**31-1)
    tx, tu, vx, vu = map(torch.from_numpy, (x, source, xv, uv))
    history, best_state, best_loss, best_epoch, waiting = [], None, np.inf, 0, 0
    with isolated_torch_seed(value):
        model = ShallowBinaryCritic(x.shape[1], hp["hidden_width"])
        loader = DataLoader(TensorDataset(tx, tu), batch_size=hp["batch_size"],
            shuffle=True, generator=torch.Generator().manual_seed(value))
        optimizer = torch.optim.AdamW(model.parameters(), lr=hp["learning_rate"], weight_decay=hp["weight_decay"])
        for epoch in range(1, hp["critic_epochs"]+1):
            model.train()
            for bx, bu in loader:
                loss = F.cross_entropy(model(bx), bu)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                train_loss = float(F.cross_entropy(model(tx), tu))
                val_loss = float(F.cross_entropy(model(vx), vu))
            history.append(dict(epoch=epoch, train_bce=train_loss, val_bce=val_loss))
            if not np.isfinite(train_loss) or not np.isfinite(val_loss):
                raise FloatingPointError("Nonfinite critic loss")
            if val_loss < best_loss:
                meaningful = val_loss < best_loss-hp["min_delta"]
                best_loss, best_epoch = val_loss, epoch
                best_state = deepcopy(model.state_dict())
                waiting = 0 if meaningful else waiting+1
            else:
                waiting += 1
            if waiting >= hp["patience"]:
                break
        model.load_state_dict(best_state)
        optimizer.zero_grad(set_to_none=True)
        model.eval()
        model.requires_grad_(False)
        state = CriticState.from_model(model)
    return state, dict(selected_epoch=best_epoch, epochs_run=len(history),
                      selected_by="source_validation_bce", history=history)


class IterativeMGPA(PortableEstimator):
    """A sequence of gated trust-region corrections with pooled score anchors.

    FIT-only data determine the critics and Ridge anchors. Default native maps
    use 48 stages, 12-epoch GELU critics, and radius ``.05 * sqrt(width)``.
    Reuse changes critic epochs to 100, learning rate to .0003, max_step to
    .025, and enables source-validation early stopping. Stage seeds are
    ``seed + 20000 + 1009*k``; neural critics add one to that seed.

    ``target`` selects the published conditional mean, unconditional empirical
    FIT mean score (``fit_mean``), or distinct zero-score control (``zero``).
    ``critic_kind='linear'`` fits the linear-only control and halves damping,
    preserving regularization per mean critic loss. No task labels are used.

    For edit-permission controls pass the alternative edit ``basis`` to fit
    and the original measurement basis as ``anchor_basis``. Original Q₀ is
    then computed once from each input and held fixed throughout replay.
    """
    def __init__(self, *, max_stages=48, critic_epochs=12, max_step=.05,
                 floor=.1, damping=.001, linear_c=1., hidden_width=128,
                 batch_size=256, learning_rate=.001, weight_decay=.001,
                 anchor_ridge_alpha=10., target="conditional", critic_kind="both",
                 early_stopping=False, patience=10, min_delta=0., frozen_anchor=False):
        """Declare the stage geometry, source-critic schedule, and score-target control."""
        values = dict(max_stages=max_stages, critic_epochs=critic_epochs,
            max_step=max_step, floor=floor, damping=damping, linear_c=linear_c,
            hidden_width=hidden_width, batch_size=batch_size, learning_rate=learning_rate,
            weight_decay=weight_decay, anchor_ridge_alpha=anchor_ridge_alpha,
            target=target, critic_kind=critic_kind, early_stopping=early_stopping,
            patience=patience, min_delta=min_delta, frozen_anchor=frozen_anchor)
        for name in ("max_stages", "critic_epochs", "hidden_width", "batch_size", "patience"):
            v = values[name]
            if isinstance(v, bool) or not np.isfinite(v) or int(v) != v or v < 1:
                raise ValueError(f"{name} must be a positive integer")
            values[name] = int(v)
        for name in ("floor", "damping", "linear_c", "learning_rate"):
            if not np.isfinite(values[name]) or values[name] <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("max_step", "weight_decay", "anchor_ridge_alpha", "min_delta"):
            if not np.isfinite(values[name]) or values[name] < 0:
                raise ValueError(f"{name} must be nonnegative")
        if target not in ("conditional", "fit_mean", "zero"):
            raise ValueError("target must be conditional, fit_mean, or zero")
        if critic_kind not in ("both", "linear"):
            raise ValueError("critic_kind must be both or linear")
        if not isinstance(early_stopping, bool) or not isinstance(frozen_anchor, bool):
            raise ValueError("early_stopping and frozen_anchor must be boolean")
        self.params = values

    def fit(self, x, source, basis, *, validation_x=None, validation_source=None,
            anchor_basis=None, seed=17, on_stage=None):
        """Fit with optional source validation; return self without selecting a task endpoint.

        Validation inputs are necessary for source-validation early stopping.
        Otherwise they only record movement. Standardization is external and
        must use FIT statistics; there is no implicit refitting at transform.
        """
        hp = self.params
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        x = matrix(x, "x")
        u = binary(source, len(x))
        b = validate_basis(basis, x.shape[1])
        anchor = validate_basis(b if anchor_basis is None else anchor_basis, x.shape[1])
        frozen = hp["frozen_anchor"] or not np.array_equal(b, anchor)
        xv = None if validation_x is None else matrix(validation_x, "validation_x")
        if xv is not None and xv.shape[1] != x.shape[1]:
            raise ValueError("FIT/VAL widths differ")
        if validation_source is not None and xv is None:
            raise ValueError("validation_source requires validation_x")
        uv = None if validation_source is None else binary(validation_source, len(xv), both=False)
        if hp["early_stopping"] and (xv is None or uv is None):
            raise ValueError("early_stopping requires source validation features and labels")
        self.n_features_in_, self.basis_, self.anchor_basis_ = x.shape[1], b, anchor
        self.frozen_anchor_, self.stages_, self.history_ = frozen, [], []
        self.metadata_ = dict(source_blind=True, downstream_labels_used=False,
            seed=int(seed), target=hp["target"], gate_rank=b.shape[1],
            source_validation_used_for_critic_selection=hp["early_stopping"],
            anchor_conditioning="original_measurement_Q0", frozen_anchor=frozen,
            omitted_diagnostics="Q-only source critics do not determine the published endpoints",
            effective_damping=hp["damping"]/(2 if hp["critic_kind"] == "linear" else 1))
        qfit = quotient(x, anchor)
        qval = None if xv is None else quotient(xv, anchor)
        current, validation = x.copy(), None if xv is None else xv.copy()
        for k in range(hp["max_stages"] if b.shape[1] else 0):
            stage_seed = int(seed)+20000+1009*k
            affine = CriticState.from_model(fit_affine_critic(current, u, seed=stage_seed, c=hp["linear_c"]))
            full, training = [affine], {}
            if hp["critic_kind"] == "both":
                if hp["early_stopping"]:
                    gelu, training = _fit_validation_critic(current, u, validation, uv, stage_seed+1, hp)
                else:
                    gelu = CriticState.from_model(fit_shallow_critic(current, u,
                        seed=stage_seed+1, epochs=hp["critic_epochs"], hidden_width=hp["hidden_width"],
                        batch_size=hp["batch_size"], learning_rate=hp["learning_rate"],
                        weight_decay=hp["weight_decay"]))
                full.append(gelu)
            scores = np.column_stack([c.predict(current) for c in full])
            if hp["target"] == "conditional":
                regression = Ridge(alpha=hp["anchor_ridge_alpha"], solver="cholesky").fit(qfit, scores)
                coefficients = np.asarray(regression.coef_).reshape(len(full), x.shape[1])
                intercepts = np.asarray(regression.intercept_).reshape(len(full))
            else:
                coefficients = np.zeros((len(full), x.shape[1]))
                intercepts = scores.mean(0, dtype=np.float64) if hp["target"] == "fit_mean" else np.zeros(len(full))
            anchors = tuple(CriticState("affine", (coefficients[j].astype(np.float32),
                np.asarray(intercepts[j], dtype=np.float32))) for j in range(len(full)))
            stage = ScoreStage(tuple(full), anchors, b.copy(), anchor.copy(),
                hp["target"], hp["floor"], self.metadata_["effective_damping"], hp["max_step"], frozen)
            next_fit = stage.apply(current, original_q0=qfit if frozen else None)
            record = dict(stage=k, seed=stage_seed, critic_training=training)
            if hp["target"] == "fit_mean":
                record["fit_mean_score"] = intercepts.tolist()
            if validation is not None:
                next_val = stage.apply(validation, original_q0=qval if frozen else None)
                record.update(before_movement=relative_movement(xv, validation),
                              after_movement=relative_movement(xv, next_val))
                validation = next_val
            self.stages_.append(stage)
            self.history_.append(record)
            if on_stage is not None:
                on_stage(deepcopy(record))
            current = next_fit
        self.metadata_["rank_zero_identity"] = not bool(b.shape[1])
        return self

    def transform(self, x, stages=None, fraction=1., source_ids=None, *, endpoint=None):
        """Replay a fixed prefix; fraction scales only its last update."""
        self._require_fitted()
        original = matrix(x, "x")
        if original.shape[1] != self.n_features_in_:
            raise ValueError("Representation width changed")
        if endpoint is not None:
            if not endpoint.attained:
                raise EndpointUnavailableError("Movement endpoint was not attained")
            stages, fraction = endpoint.stages, endpoint.fraction
        count = len(self.stages_) if stages is None else stages
        if isinstance(count, bool) or int(count) != count or not 0 <= count <= len(self.stages_):
            raise ValueError("stages must be a valid fitted prefix")
        if not np.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError("fraction must be in [0,1]")
        q0 = quotient(original, self.anchor_basis_) if self.frozen_anchor_ else None
        current = original
        for k, stage in enumerate(self.stages_[:int(count)]):
            current = stage.apply(current, fraction=fraction if k == count-1 else 1., original_q0=q0)
        return current

    def iter_transforms(self, x):
        """Yield (stage_count, rows), including identity, with one replay per stage."""
        self._require_fitted()
        current = matrix(x, "x")
        if current.shape[1] != self.n_features_in_:
            raise ValueError("Representation width changed")
        q0 = quotient(current, self.anchor_basis_) if self.frozen_anchor_ else None
        yield 0, current.copy()
        for k, stage in enumerate(self.stages_, 1):
            current = stage.apply(current, original_q0=q0)
            yield k, current.copy()

    def select_movement_budget(self, validation_x, budget):
        """Select the first upward VAL crossing, or explicitly mark unattained."""
        if not np.isfinite(budget) or budget < 0:
            raise ValueError("budget must be finite and nonnegative")
        original = matrix(validation_x, "validation_x")
        relative_movement(original, original)
        if budget == 0:
            return Endpoint(0, requested_movement=0.)
        previous, before, last = original, 0., 0.
        for k, current in self.iter_transforms(original):
            if k == 0:
                continue
            last = relative_movement(original, current)
            if before < budget <= last:
                fraction = crossing_fraction(original, previous, current, budget)
                return Endpoint(k, fraction, True, float(budget),
                    relative_movement(original, previous+fraction*(current-previous)), "first upward VAL crossing")
            previous, before = current, last
        return Endpoint(len(self.stages_), attained=False, requested_movement=float(budget),
                        validation_movement=last, reason="VAL trajectory did not attain budget")
