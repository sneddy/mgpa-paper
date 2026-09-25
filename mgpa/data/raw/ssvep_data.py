from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
from scipy.io import loadmat, whosmat
from scipy.signal import resample_poly
from sklearn.model_selection import StratifiedKFold


CHANNEL_NAMES: tuple[str, ...] = ("POz", "PO3", "PO4", "PO5", "PO6", "Oz", "O1", "O2")
SOURCE_NAMES: tuple[str, ...] = ("dry", "wet")
EXPECTED_TRIAL_SHAPE = (8, 710, 2, 10, 12)
FREQUENCIES = np.asarray(
    [9.25, 11.25, 13.25, 9.75, 11.75, 13.75, 10.25, 12.25, 14.25, 10.75, 12.75, 14.75],
    dtype=np.float64,
)
PHASES = np.asarray([0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.5, 1.5, 1.5])


def _scalar_text(value: object) -> str:
    """Normalize one MATLAB text scalar for the trial index."""
    current = value
    while isinstance(current, np.ndarray) and current.size == 1:
        current = current.item()
    return str(current).strip()


def subject_file(root: Path, subject: int) -> Path:
    """Resolve the canonical SSVEP MATLAB filename for one participant."""
    return root / f"S{subject:03d}.mat"


def load_subject_metadata(root: Path) -> pd.DataFrame:
    """Read participant sex and acquisition order for stratified splitting."""
    path = root / "Subjects_Information.mat"
    if not path.exists():
        raise FileNotFoundError(path)
    table = loadmat(path)["Subjects_Information"]
    if table.shape != (103, 11):
        raise AssertionError(f"Unexpected Subjects_Information shape: {table.shape}")
    headers = [_scalar_text(value) for value in table[0]]
    rows: list[dict[str, object]] = []
    for subject, values in enumerate(table[1:], start=1):
        record = {header: _scalar_text(value) for header, value in zip(headers, values)}
        rows.append(
            {
                "subject": subject,
                "subject_id": f"S{subject:03d}",
                "gender": record["Gender"].lower(),
                "age": int(float(record["Age"])),
                "handedness": record["Handedness"].lower(),
                "first_electrode": record["Headband to wear first"].lower(),
            }
        )
    metadata = pd.DataFrame(rows)
    if set(metadata["first_electrode"]) != {"dry", "wet"}:
        raise AssertionError("Could not recover dry/wet-first acquisition order.")
    return metadata


def build_trial_index(root: Path) -> pd.DataFrame:
    """Build a metadata-only index in MATLAB axis order.

    No signal arrays are loaded here.  Every row corresponds to exactly one
    ``(subject, electrode, block, target)`` trial.
    """

    subjects = load_subject_metadata(root)
    rows: list[dict[str, object]] = []
    for row in subjects.itertuples(index=False):
        path = subject_file(root, int(row.subject))
        if not path.exists():
            raise FileNotFoundError(path)
        for source, source_name in enumerate(SOURCE_NAMES):
            for block in range(10):
                for target in range(12):
                    rows.append(
                        {
                            "sample_id": f"{row.subject_id}/e{source}/b{block:02d}/t{target:02d}",
                            "subject": int(row.subject),
                            "subject_id": row.subject_id,
                            "gender": row.gender,
                            "first_electrode": row.first_electrode,
                            "source": source,
                            "source_name": source_name,
                            "block": block,
                            "target": target,
                            "frequency_hz": float(FREQUENCIES[target]),
                            "frequency_binary": int(FREQUENCIES[target] >= 12.0),
                            "phase": int(round(PHASES[target] * 2.0)) // 1,
                            "phase_class": int(round(PHASES[target] * 2.0)) // 1,
                            "path": str(path),
                        }
                    )
    index = pd.DataFrame(rows)
    index["phase_class"] = index["phase_class"].map({0: 0, 1: 1, 2: 2, 3: 3}).astype(int)
    audit_trial_index(index)
    return index


def audit_trial_index(index: pd.DataFrame) -> None:
    """Check complete, unique source-by-frequency trial metadata."""
    required = {
        "sample_id",
        "subject",
        "source",
        "block",
        "target",
        "frequency_binary",
        "phase_class",
    }
    missing = required.difference(index.columns)
    if missing:
        raise AssertionError(f"Trial index missing columns: {sorted(missing)}")
    if len(index) != 102 * 2 * 10 * 12:
        raise AssertionError(f"Expected 24,480 trials, found {len(index):,}.")
    if index["sample_id"].duplicated().any():
        raise AssertionError("sample_id must be unique.")
    counts = index.groupby(["subject", "source", "block", "target"], observed=True).size()
    if len(counts) != 102 * 2 * 10 * 12 or not (counts == 1).all():
        raise AssertionError("Incomplete subject × source × block × target factorial.")
    source_target = index.groupby(["source", "target"], observed=True).size().unstack(fill_value=0)
    if source_target.shape != (2, 12) or (source_target <= 0).any().any():
        raise AssertionError("Every source-target cell must have support.")


def audit_raw_files(root: Path, *, check_finite: bool = True) -> pd.DataFrame:
    """Verify the required SSVEP tensors and optional finite values."""
    records: list[dict[str, object]] = []
    for subject in range(1, 103):
        path = subject_file(root, subject)
        variables = {name: shape for name, shape, _dtype in whosmat(path)}
        shape = tuple(variables.get("data", ()))
        if shape != EXPECTED_TRIAL_SHAPE:
            raise AssertionError(f"{path.name}: expected {EXPECTED_TRIAL_SHAPE}, got {shape}")
        finite: bool | None = None
        if check_finite:
            values = loadmat(path, variable_names=["data"])["data"]
            finite = bool(np.isfinite(values).all())
            if not finite:
                raise AssertionError(f"Non-finite values in {path}")
        records.append(
            {
                "subject": subject,
                "file": path.name,
                "shape": str(shape),
                "finite_checked": bool(check_finite),
                "finite": finite,
            }
        )
    return pd.DataFrame(records)


def audit_impedance(root: Path) -> pd.DataFrame:
    """Summarize recorded electrode impedances without changing observations."""
    path = root / "Impedance.mat"
    impedance = np.asarray(loadmat(path)["Impedance"], dtype=np.float64)
    if impedance.shape != (8, 10, 2, 102):
        raise AssertionError(f"Unexpected impedance shape: {impedance.shape}")
    means = impedance.mean(axis=(0, 1, 3))
    # The observed values resolve an ambiguity in the prose distributed with
    # the data: index 0 is dry and index 1 is wet.
    if not (means[0] > means[1] and abs(means[0] - 261.67) < 5 and abs(means[1] - 19.63) < 5):
        raise AssertionError(f"Impedance indices do not match published means: {means}")
    return pd.DataFrame(
        {
            "source": [0, 1],
            "source_name": list(SOURCE_NAMES),
            "mean_kohm": means,
            "expected_mean_kohm": [261.67, 19.63],
        }
    )


def load_trials(index: pd.DataFrame) -> np.ndarray:
    """Load indexed trials as ``[trial, channel, sample]`` efficiently."""

    output = np.empty((len(index), 8, 710), dtype=np.float32)
    positions = pd.Series(np.arange(len(index)), index=index.index)
    for path, group in index.groupby("path", sort=False):
        data = np.asarray(loadmat(path, variable_names=["data"])["data"], dtype=np.float32)
        for row in group.itertuples():
            output[positions.loc[row.Index]] = data[:, :, int(row.source), int(row.block), int(row.target)]
    if not np.isfinite(output).all():
        raise AssertionError("Loaded trial tensor contains non-finite values.")
    return output


def preprocess_response(
    trials: np.ndarray,
    *,
    start: int = 160,
    stop: int = 660,
    source_sfreq: int = 250,
    target_sfreq: int = 200,
    eps: float = 1e-6,
) -> np.ndarray:
    """Crop the 2 s response, polyphase-resample, and normalize per channel."""

    trials = np.asarray(trials, dtype=np.float32)
    if trials.ndim != 3 or trials.shape[1:] != (8, 710):
        raise ValueError(f"Expected [trial, 8, 710], got {trials.shape}")
    cropped = trials[:, :, start:stop]
    if cropped.shape[-1] != 500:
        raise AssertionError(f"Primary crop must contain 500 samples, got {cropped.shape[-1]}")
    resampled = resample_poly(cropped, target_sfreq, source_sfreq, axis=-1).astype(np.float32)
    if resampled.shape[-1] != 400:
        raise AssertionError(f"Encoder input must contain exactly 400 samples, got {resampled.shape[-1]}")
    mean = resampled.mean(axis=-1, keepdims=True)
    scale = np.maximum(resampled.std(axis=-1, keepdims=True), eps)
    normalized = (resampled - mean) / scale
    if not np.isfinite(normalized).all():
        raise AssertionError("Preprocessed response contains non-finite values.")
    return normalized.astype(np.float32)


def prestimulus(trials: np.ndarray) -> np.ndarray:
    """Select the recorded prestimulus interval used for transport fitting."""
    trials = np.asarray(trials, dtype=np.float32)
    return trials[..., :125]


@dataclass(frozen=True)
class OuterSplit:
    """Participant identities for the independent adaptation and reader roles."""
    rotation: int
    adapter_fit_subjects: tuple[int, ...]
    adapter_validation_subjects: tuple[int, ...]
    usage_subjects: tuple[int, ...]
    test_subjects: tuple[int, ...]

    @property
    def adaptation_subjects(self) -> tuple[int, ...]:
        """Combine adapter-fit and adapter-validation participants."""
        return tuple(sorted(self.adapter_fit_subjects + self.adapter_validation_subjects))

    def role_for(self, subject: int) -> str:
        """Return a participant's unique role within this rotation."""
        if subject in self.adapter_fit_subjects:
            return "adapter_fit"
        if subject in self.adapter_validation_subjects:
            return "adapter_validation"
        if subject in self.usage_subjects:
            return "usage"
        if subject in self.test_subjects:
            return "test"
        raise KeyError(subject)


def make_outer_splits(subject_metadata: pd.DataFrame, *, seed: int, n_splits: int = 5) -> list[OuterSplit]:
    """Five deterministic rotations stratified by order and gender."""

    subjects = subject_metadata.sort_values("subject").reset_index(drop=True)
    strata = subjects["first_electrode"].astype(str) + "|" + subjects["gender"].astype(str)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_subjects: list[tuple[int, ...]] = []
    for _train, test in splitter.split(subjects, strata):
        fold_subjects.append(tuple(sorted(subjects.iloc[test]["subject"].astype(int).tolist())))

    splits: list[OuterSplit] = []
    for test_fold in range(n_splits):
        split = OuterSplit(
            rotation=test_fold,
            test_subjects=fold_subjects[test_fold],
            usage_subjects=fold_subjects[(test_fold + 1) % n_splits],
            adapter_validation_subjects=fold_subjects[(test_fold + 2) % n_splits],
            adapter_fit_subjects=tuple(
                sorted(fold_subjects[(test_fold + 3) % n_splits] + fold_subjects[(test_fold + 4) % n_splits])
            ),
        )
        all_sets = [
            set(split.adapter_fit_subjects),
            set(split.adapter_validation_subjects),
            set(split.usage_subjects),
            set(split.test_subjects),
        ]
        if any(left & right for i, left in enumerate(all_sets) for right in all_sets[i + 1 :]):
            raise AssertionError("Outer split roles are not subject-disjoint.")
        if set.union(*all_sets) != set(subjects["subject"].astype(int)):
            raise AssertionError("Outer split does not cover every participant exactly once.")
        splits.append(split)
    return splits


def split_audit_table(index: pd.DataFrame, splits: Sequence[OuterSplit]) -> pd.DataFrame:
    """Tabulate the participant and observation membership of each rotation."""
    records: list[dict[str, object]] = []
    for split in splits:
        for role, subject_ids in (
            ("adapter_fit", split.adapter_fit_subjects),
            ("adapter_validation", split.adapter_validation_subjects),
            ("usage", split.usage_subjects),
            ("test", split.test_subjects),
        ):
            part = index[index["subject"].isin(subject_ids)]
            cells = part.groupby(["source", "target"], observed=True).size()
            records.append(
                {
                    "rotation": split.rotation,
                    "role": role,
                    "n_subjects": len(subject_ids),
                    "n_trials": len(part),
                    "all_source_target_cells_supported": bool(len(cells) == 24 and (cells > 0).all()),
                    "subject_hash": hashlib.sha256(
                        ",".join(map(str, sorted(subject_ids))).encode("utf-8")
                    ).hexdigest()[:16],
                }
            )
    audit = pd.DataFrame(records)
    if not audit["all_source_target_cells_supported"].all():
        raise AssertionError("At least one fold role lacks source-target support.")
    return audit


def deterministic_subsample(
    index: pd.DataFrame, n: int, *, seed: int, group_columns: Sequence[str] = ("source", "target")
) -> pd.DataFrame:
    """Draw a reproducible group-balanced calibration subset."""
    if len(index) <= n:
        return index.copy()
    rng = np.random.default_rng(seed)
    groups = list(index.groupby(list(group_columns), sort=True, observed=True))
    per_group = max(1, n // len(groups))
    selected: list[int] = []
    for _key, group in groups:
        size = min(len(group), per_group)
        selected.extend(rng.choice(group.index.to_numpy(), size=size, replace=False).tolist())
    remaining = n - len(selected)
    if remaining > 0:
        pool = index.index.difference(selected).to_numpy()
        selected.extend(rng.choice(pool, size=min(remaining, len(pool)), replace=False).tolist())
    return index.loc[sorted(selected)].copy()
