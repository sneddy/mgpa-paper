"""Independent HEAD-fitted readers and participant-level evaluation."""

from .evaluation import EvaluationConfig, Evaluator
from .reporting import aggregate, metric_summary, paired_contrast

__all__ = ["EvaluationConfig", "Evaluator", "aggregate", "metric_summary", "paired_contrast"]
