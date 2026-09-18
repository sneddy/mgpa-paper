"""Participant bootstrap with maxima recomputed in each draw."""
from collections import defaultdict

import numpy as np

METRICS = ("S", "Tf", "Tr", "M")

def _population(cells):
    """Index disjoint EVAL participants by fit repetition."""
    if not cells:
        raise ValueError("Cannot aggregate empty result cells")
    by_seed = defaultdict(dict)
    for cell in cells:
        fitted_seed = cell["job"]["seed"]
        for row in cell["evaluation"]["participant_summary"]:
            person = str(row["participant"])
            if person in by_seed[fitted_seed]:
                raise ValueError(f"Duplicate EVAL participant within a fit repetition: {person}")
            by_seed[fitted_seed][person] = row
    seeds = sorted(by_seed)
    people = sorted(by_seed[seeds[0]])
    if any(sorted(rows) != people for rows in by_seed.values()):
        raise ValueError("Different participant populations between fit repetitions")
    if not people:
        raise ValueError("Cannot aggregate empty participant results")
    families = sorted(by_seed[seeds[0]][people[0]]["source_auroc_by_family"])
    if not families:
        raise ValueError("Cannot aggregate an empty source-reader bank")
    if any(sorted(row["source_auroc_by_family"]) != families for rows in by_seed.values() for row in rows.values()):
        raise ValueError("Inconsistent source-reader bank")
    return by_seed, seeds, people, families


def _values(population, metric, *, source_families=None):
    """Keep the reader axis until the cohort mean has been computed."""
    by_seed, seeds, people, families = population
    if metric == "S":
        families = families if source_families is None else list(source_families)
        if not families or any(f not in population[3] for f in families):
            raise ValueError("Requested source-reader family is missing")
        values = np.array([[[by_seed[s][p]["source_auroc_by_family"][f] for f in families]
                            for p in people] for s in seeds], dtype=float)
        statistic = lambda a: a.mean(axis=-2).max(axis=-1)
    else:
        def value(row):
            """Read one participant's prespecified scalar metric."""
            return row.get(metric)
        if any(value(by_seed[s][p]) is None for s in seeds for p in people):
            return None, None
        values = np.array([[value(by_seed[s][p]) for p in people] for s in seeds], dtype=float)
        statistic = lambda a: a.mean(axis=-1)
    if not np.isfinite(values).all():
        raise ValueError(f"Nonfinite {metric} values")
    return values, statistic


def _interval(values, statistic, seeds, people, *, resamples, seed, reference=None):
    """Resample participants jointly across all fitted maps."""
    if type(resamples) is not int or resamples < 1 or type(seed) is not int:
        raise ValueError("Bootstrap requires positive integer resamples and an integer seed")
    per_seed = statistic(values)
    if reference is not None:
        per_seed = per_seed - statistic(reference)
    rng, draws = np.random.default_rng(seed), []
    for start in range(0, resamples, 256):
        index = rng.integers(len(people), size=(min(256, resamples-start), len(people)))
        current = statistic(values[:, index])
        if reference is not None:
            current = current - statistic(reference[:, index])
        draws.append(current.mean(axis=0))
    low, high = np.quantile(np.concatenate(draws), [.025, .975])
    return {"mean": float(per_seed.mean()), "ci95": [float(low), float(high)],
            "per_seed": {str(s): float(v) for s, v in zip(seeds, per_seed)},
            "seed_std": float(per_seed.std(ddof=1)) if len(seeds)>1 else None}


def metric_summary(cells, metric, *, resamples=20000, seed=1101, source_families=None):
    """Estimate a scalar and paired-participant percentile interval."""
    population = _population(cells)
    values, statistic = _values(population, metric, source_families=source_families)
    if values is None:
        return None
    return _interval(values, statistic, population[1], population[2], resamples=resamples, seed=seed)


def paired_contrast(cells, reference_cells, metric, *, resamples=20000, seed=1101):
    """Method-minus-reference; recompute each source MAX inside every draw."""
    population, reference = _population(cells), _population(reference_cells)
    if population[2:] != reference[2:]:
        raise ValueError("Paired contrasts require identical participants and source readers")
    values, statistic = _values(population, metric)
    reference_values, _ = _values(reference, metric)
    if values is None or reference_values is None:
        return None
    seeds = population[1]
    if seeds != reference[1]:
        if len(seeds) != 1 and len(reference[1]) != 1:
            raise ValueError("Different stochastic fit seeds cannot be paired")
        seeds = reference[1] if len(seeds) == 1 else seeds
        values = np.broadcast_to(values, (len(seeds), *values.shape[1:]))
        reference_values = np.broadcast_to(reference_values, (len(seeds), *reference_values.shape[1:]))
    return _interval(values, statistic, seeds, population[2], reference=reference_values,
                     resamples=resamples, seed=seed)


def ensemble_summary(groups, metric, *, resamples=20000, seed=1101, reference_cells=None):
    """Average fixed orientations after each reader maximum, with shared draws."""
    populations = [_population(cells) for cells in groups]
    first = populations[0]
    if any(p[1:] != first[1:] for p in populations[1:]):
        raise ValueError("Orientations must share fit seeds, participants and reader bank")
    values = np.stack([_values(p, metric)[0] for p in populations])
    statistic = _values(first, metric)[1]
    reference = None
    if reference_cells is not None:
        ref = _population(reference_cells)
        if ref[2:] != first[2:]:
            raise ValueError("Reference cohort and reader bank differ")
        reference = _values(ref, metric)[0]
    def estimate(v, ref=None):
        """Average orientation-specific statistics and subtract the shared reference."""
        result = statistic(v).mean(axis=(0,1))
        return result if ref is None else result - statistic(ref).mean(axis=0)
    point = estimate(values, reference)
    rng = np.random.default_rng(seed)
    draws = []
    for start in range(0, resamples, 256):
        indices = rng.integers(len(first[2]), size=(min(256,resamples-start),len(first[2])))
        draws.append(estimate(values[:,:,indices], None if reference is None else reference[:,indices]))
    low, high = np.quantile(np.concatenate(draws), [.025,.975])
    return {'mean':float(point),'ci95':[float(low),float(high)],'orientations':len(groups)}


def aggregate(cells, *, resamples=20000, seed=1101):
    """Equal people and repetitions; source MAX after cohort averaging per seed.

    One participant is resampled jointly across repetitions. Intervals are
    conditional on fitted maps/readers, not retraining uncertainty.
    """
    population = _population(cells)
    _, seeds, people, _ = population
    result = {"participants": len(people), "participant_ids": people, "fit_seeds": seeds,
              "bootstrap_resamples": resamples, "bootstrap_seed": seed,
              "ci_scope": "paired participant resampling conditional on fitted maps/readers; repetitions are not extra participants"}
    for metric in METRICS:
        values, statistic = _values(population, metric)
        if values is not None:
            result[metric] = _interval(values, statistic, seeds, people, resamples=resamples, seed=seed)
    return result
