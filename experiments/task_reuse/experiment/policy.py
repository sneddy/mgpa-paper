"""Explicit DEV operating policies for an unknown source--target association.

With fixed positive/negative weight totals, weighted AUROC is exactly quadratic
in the coupling parameter rho. We minimize the participant-mean quadratic,
not the mean of separately minimized participant curves. This module has no IO
or fitting and never selects an endpoint using EVAL data by itself.

Optional tolerance is DEV parsimony, not evidence of statistical equivalence.
Its default is zero: no implicit relaxation of the strict objective maximum.
"""
from __future__ import annotations

from copy import deepcopy
import re
from typing import Mapping

import numpy as np


OBJECTIVES = ("worst_association", "reversed")


def _triple(values):
    """Validate a finite quadratic coefficient vector in [a, b, c] order."""
    value = np.asarray(values, dtype=np.float64)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError("Expected finite quadratic coefficients [a, b, c]")
    return value


def _from_three(minus, zero, plus):
    """Recover quadratic coefficients from AUROCs at associations -1, 0 and 1."""
    value = np.asarray([minus, zero, plus], dtype=np.float64)
    if not np.isfinite(value).all() or np.any((value < 0) | (value > 1)):
        raise ValueError("Finite AUROC values in [0,1] are required for all participants")
    return np.array([(plus + minus) / 2 - zero, (plus - minus) / 2, zero], dtype=np.float64)


def coefficients(metric):
    """Extract exact quadratic coefficients from one head's three anchor regimes.

    The input is the result of ``evaluate_pair_predictions`` for ONE fixed head.
    Participants must be the same in all three regimes, with defined AUROC.
    Undefined participants are rejected, not silently dropped or reweighted.
    """
    regimes = metric["association_metrics"]
    matched = {}
    for rho in (-1, 0, 1):
        options = [row for row in regimes.values()
                   if np.isclose(float(row["association"]), rho, rtol=0, atol=1e-12)]
        if len(options) != 1:
            raise ValueError("Exactly one -1, 0, and +1 association regime is required")
        matched[rho] = options[0]
    by_person = {}
    for rho, row in matched.items():
        rows = row["per_subject"]
        people = {str(person["subject"]): person["auroc"] for person in rows}
        if not people or len(people) != len(rows):
            raise ValueError("Participant IDs must be unique and nonempty")
        if any(value is None for value in people.values()):
            raise ValueError("Defined participant AUROCs are required; no silent exclusions")
        if not np.isclose(float(row["auroc"]), np.mean(list(people.values())), rtol=1e-10, atol=1e-10):
            raise ValueError("Primary AUROC must be the equal-participant mean")
        by_person[rho] = people
    if not set(by_person[-1]) == set(by_person[0]) == set(by_person[1]):
        raise ValueError("Association regimes must have identical participant IDs")
    per_subject = {person: _from_three(by_person[-1][person], by_person[0][person], by_person[1][person]).tolist()
                   for person in sorted(by_person[0])}
    return {"mean": np.mean(list(per_subject.values()), axis=0).tolist(), "per_subject": per_subject,
            "aggregation": "equal participant mean before minimizing rho", "coefficient_order": ["a", "b", "c"]}


def quadratic_minimum(coeff):
    """Exact minimum on [-1,1], including an interior convex vertex if present."""
    a, b, c = _triple(coeff)
    candidates = [(-1., a - b + c), (0., c), (1., a + b + c)]
    # Testing |b| <= 2a before division also avoids huge, irrelevant vertices.
    if a > 0 and abs(b) <= 2 * a:
        vertex = float(-b / (2 * a))
        candidates.append((vertex, float((a * vertex + b) * vertex + c)))
    # Flat curves select rho=0 deterministically; equal extremes prefer -1.
    argmin, minimum = min(candidates, key=lambda pair: (pair[1], abs(pair[0]), pair[0]))
    return {"minimum": float(minimum), "argmin": float(argmin)}


def endpoint_summary(result, objective="worst_association"):
    """Summarize absolute fixed-head performance; preserve original recovery data."""
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}")
    coeff = coefficients(result["metrics"]["biased"])
    minimum = quadratic_minimum(coeff["mean"])
    a, b, c = coeff["mean"]
    reversed_auc, aligned_auc = a - b + c, a + b + c
    return {"endpoint": result["endpoint"], "objective": objective,
            "objective_value": minimum["minimum"] if objective == "worst_association" else float(reversed_auc),
            "worst_association_auroc": minimum["minimum"], "worst_association_rho": minimum["argmin"],
            "aligned_auroc": float(aligned_auc), "independent_auroc": float(c), "reversed_auroc": float(reversed_auc),
            "coefficients": coeff, "movement": deepcopy(result["movement"]),
            "guardrail_pass": bool(result["guardrail_pass"]),
            "independent_change": float(result["independent_change"]),
            "control_independent_change": float(result["control_independent_change"])}


def _endpoint_order(name):
    """Order named strengths or prefixes numerically for deterministic ties."""
    match = re.fullmatch(r"(fixed|strength)_([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)", str(name))
    return (float(match.group(2)), str(name)) if match else (float("inf"), str(name))


def select_endpoint(curve, objective="worst_association", utility_guardrail=-.01, tolerance=0.):
    """Select DEV endpoint using a shared explicit policy for every method.

    Both independent-regime changes (biased and independent-trained heads) must
    satisfy the guardrail, AND the upstream dual-guardrail flag must be true.
    With tolerance=0 this is the strict objective maximum, with minimum movement
    breaking ties. With an explicit positive tolerance, choose minimum movement
    among endpoints within that AUROC amount of the strict maximum. The receipt
    keeps the strict winner and the exact sacrificed score visible.
    """
    if not np.isfinite(utility_guardrail) or not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Finite utility guardrail and nonnegative tolerance required")
    eligible = []
    for original in curve:
        summary = endpoint_summary(original, objective)
        if (summary["guardrail_pass"] and min(summary["independent_change"], summary["control_independent_change"])
                >= utility_guardrail):
            movement = summary["movement"]["mean"]
            if not np.isfinite(movement) or movement < 0:
                raise ValueError("Movement must be finite and nonnegative")
            eligible.append((original, summary))
    if not eligible:
        raise ValueError("No endpoint passes both utility guardrails")
    strict = min(eligible, key=lambda pair: (-pair[1]["objective_value"], pair[1]["movement"]["mean"],
                                           _endpoint_order(pair[1]["endpoint"])))
    best_score = strict[1]["objective_value"]
    # Only machine-rounding slack, not an undeclared practical tolerance.
    near = [pair for pair in eligible if best_score - pair[1]["objective_value"] <= tolerance + 4 * np.finfo(float).eps]
    selected, summary = min(near, key=lambda pair: (pair[1]["movement"]["mean"], _endpoint_order(pair[1]["endpoint"])))
    result = deepcopy(selected)
    result["policy_summary"] = summary
    result["selection_policy"] = {
        "objective": objective, "utility_guardrail": float(utility_guardrail),
        "guardrail_heads": ["biased independent-regime", "independent-trained independent-regime"],
        "tolerance": float(tolerance), "tolerance_role": "explicit DEV parsimony; not statistical equivalence",
        "strict_maximum_endpoint": strict[1]["endpoint"], "strict_maximum_value": float(best_score),
        "selected_objective_value": float(summary["objective_value"]),
        "objective_sacrifice": float(best_score - summary["objective_value"]),
        "eligible_endpoints": len(eligible), "within_tolerance_endpoints": len(near),
        "tie_rule": "minimum movement, then numeric endpoint, then endpoint name"}
    return result
