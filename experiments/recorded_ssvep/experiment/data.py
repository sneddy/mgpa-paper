"""recorded_ssvep data boundary; the runner opens only the requested roles."""
from mgpa.data.interfaces import CONTEXTS, fit_bank as _fit_bank, load_dataset, transform_role

DATASET = "recorded_ssvep"
CONTEXT_IDS = CONTEXTS[DATASET]


def load(manifest, roles):
    """Load only the requested roles under this experiment's data contract."""
    return load_dataset(DATASET, manifest, roles)


def fitting(role):
    """Expose the experiment's source-only fitting bank."""
    return _fit_bank(DATASET, role)


def transform(model, role, *, endpoint=None, source_aware=False):
    """Replay the map through this experiment's paired or token interface."""
    return transform_role(model, DATASET, role, endpoint=endpoint, source_aware=source_aware)
