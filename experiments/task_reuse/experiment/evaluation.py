"""Shared frozen-head evaluation; no adapter fitting or outcome-based exclusions."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit


from . import protocol, policy


def validate_role(role):
    """Check finite paired features and require both task classes per participant."""
    x, y, people = np.asarray(role['x']), np.asarray(role['y']), np.asarray(role['subject']).astype(str)
    if x.ndim != 3 or x.shape[1] != 2 or len(x) != len(y) or y.shape != people.shape:
        raise ValueError('Expected aligned paired vectors, labels and people')
    if not np.isfinite(x).all() or not np.isin(y, [0, 1]).all():
        raise ValueError('Nonfinite features or nonbinary labels')
    for person in np.unique(people):
        if set(y[people == person]) != {0, 1}:
            raise ValueError('No silent one-class participant exclusion: ' + person)
    return x, y, people


def subset(role, mask):
    """Select aligned event arrays while leaving unrelated metadata out."""
    return {k: np.asarray(v)[mask] for k, v in role.items()
            if isinstance(v, np.ndarray) and len(v) == len(mask)}


@dataclass
class NumericHead:
    """Portable frozen logistic coefficients and their original input standardizer."""
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float

    def predict_pairs(self, pairs):
        """Return paired probabilities while preserving scaler arithmetic and dtype."""
        x = np.asarray(pairs)
        if x.ndim != 3 or x.shape[1] != 2 or x.shape[2] != len(self.coef):
            raise ValueError('Frozen head shape mismatch')
        # Match StandardScaler.transform's dtype-preserving, sequential in-place
        # operations. Promoting original float32 embeddings first changes ties.
        scaled = x.reshape(-1, x.shape[-1]).copy()
        if scaled.dtype not in (np.dtype('float32'), np.dtype('float64')):
            scaled = scaled.astype(np.float64)
        scaled -= self.mean
        scaled /= self.scale
        logits = (scaled @ self.coef[:, None]).ravel() + self.intercept
        return expit(logits).reshape(len(x), 2)

    def state(self):
        """Export numeric head parameters as JSON-compatible values."""
        return dict(mean=self.mean.tolist(), scale=self.scale.tolist(),
                    coef=self.coef.tolist(), intercept=float(self.intercept))

    @classmethod
    def from_state(cls, state):
        """Restore a frozen head from its saved numeric parameters."""
        return cls(*(np.asarray(state[k], dtype=float) for k in ('mean', 'scale', 'coef')),
                   float(state['intercept']))


def fit_head(role, association, c, seed, config):
    """Fit one HEAD-only logistic model and verify its numeric export."""
    x, y, people = validate_role(role)
    u = protocol.assign_endpoints(y, people, association, seed, config['max_probability'])
    native = protocol.fit_task_head(x, y, people, u, C=c, seed=seed)
    head = NumericHead(native.scaler.mean_.copy(), native.scaler.scale_.copy(),
                       native.classifier.coef_[0].copy(), float(native.classifier.intercept_[0]))
    if not np.allclose(head.predict_pairs(x), native.predict_pairs(x), atol=2e-14, rtol=1e-12):
        raise AssertionError('Numeric head export differs')
    return head, u


def head_metrics(head, role, config, pairs=None):
    """Score one frozen head across the declared source-association regimes."""
    _, y, people = validate_role(role)
    scores = head.predict_pairs(role['x'] if pairs is None else pairs)
    metrics = protocol.evaluate_pair_predictions(scores, y, people,
        config['eval_associations'], max_probability=config['max_probability'])
    return metrics, scores


def build_heads(role, config):
    """All C selection is original-feature HEAD-only participant CV, all tasks."""
    _, _, people = validate_role(role)
    seed = int(config['head_seed']) + int(config['seed'])
    curve = []
    for c in config['head_C_grid']:
        values = []
        for person in np.unique(people):
            train, test = subset(role, people != person), subset(role, people == person)
            head, _ = fit_head(train, 0., c, seed, config)
            metric, _ = head_metrics(head, test, config)
            values.append(float(metric['association_metrics']['independent']['auroc']))
        curve.append(dict(C=float(c), per_fold=values, mean=float(np.mean(values))))
    best = min(curve, key=lambda row: (-row['mean'], row['C']))
    heads, assignments = {}, {}
    for name, association in [('biased', config['head_association']), ('control', 0.)]:
        heads[name], u = fit_head(role, association, best['C'], seed, config)
        assignments[name] = u.tolist()
    return heads, dict(C=best['C'], seed=seed, head_lopo_auroc=best['mean'],
        selected_on='HEAD-only leave-one-participant-out original independent head AUROC',
        curve=curve, assignments=assignments,
        heads={k: v.state() for k, v in heads.items()},
        head_people=sorted(np.unique(people).tolist()))


def identity_metrics(heads, role, config):
    """Evaluate original representations to establish both utility guardrails."""
    return {name: head_metrics(head, role, config)[0] for name, head in heads.items()}


def endpoint_metrics(pairs, role, heads, config, identity, endpoint):
    """Compute frozen-head outcomes, participant movement and dual guardrails."""
    metrics, predictions = {}, {}
    for name, head in heads.items():
        metrics[name], predictions[name] = head_metrics(head, role, config, pairs)
    changes = {name: metrics[name]['association_metrics']['independent']['auroc']
                    - identity[name]['association_metrics']['independent']['auroc']
               for name in heads}
    x = np.asarray(role['x'])
    change = np.linalg.norm((np.asarray(pairs)-x).reshape(len(x), -1), axis=1)
    size = np.linalg.norm(x.reshape(len(x), -1), axis=1)
    people = np.asarray(role['subject']).astype(str)
    moves = {p: float(change[people == p].mean()/max(size[people == p].mean(), 1e-12))
             for p in np.unique(people)}
    row = dict(endpoint=endpoint, metrics=metrics,
        movement=dict(mean=float(np.mean(list(moves.values()))), per_subject=moves),
        independent_change=float(changes['biased']),
        control_independent_change=float(changes['control']),
        guardrail_pass=bool(min(changes.values()) >= config['utility_guardrail']))
    row['summary'] = policy.endpoint_summary(row)
    return row, predictions


def select_curve(curve, config):
    """Select the strict best admissible DEV endpoint under the common policy."""
    selected = policy.select_endpoint(curve,
        utility_guardrail=config['utility_guardrail'], tolerance=0.)
    return selected['policy_summary']


def select_families(cells, config):
    """Choose the least-moving candidate within each family's declared tolerance."""
    result = {}
    for family in sorted({row['candidate']['family'] for row in cells}):
        group = [row for row in cells if row['candidate']['family'] == family]
        top = max(row['selected']['worst_association_auroc'] for row in group)
        eligible = [row for row in group
                    if top-row['selected']['worst_association_auroc'] <= config['recipe_tolerance']+1e-15]
        best = min(eligible, key=lambda row: (row['selected']['movement']['mean'],
            row['candidate'].get('rank', 0), row['candidate']['id']))
        result[family] = dict(candidate=best['candidate'], selected=best['selected'],
            strict_family_best=float(top), sacrifice=float(top-best['selected']['worst_association_auroc']),
            model_dir=best['model_dir'], cell_path=best['cell_path'], binding=best['binding'])
    return result
