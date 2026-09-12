"""Inspection of an instance before it is explained.

An explanation is only meaningful when the instance is one the model handles
well and when the segmentation carries some structure. The report below puts
both conditions in front of the user before any search starts. It gathers three
groups of facts, namely the shape and statistics of the series, the confidence
and the class distribution the model assigns to it, and the geometry of the
segmentation together with a diagnosis of how well the breakpoints follow the
series.

The segment separation index merits a separate note. It compares the variance that
remains inside the segments with the total variance of the series, in the
manner of a one-way analysis of variance. A value near one says the breakpoints
captured almost every level change, so the segments are homogeneous and the
partition is informative. A value near zero says the segments are no more
homogeneous than arbitrary windows would be, which usually points at a penalty
that is too large or at a cost function that does not match the signal.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .segmentation import Segmentation

__all__ = ["describe_instance", "instance_report"]


def _series_statistics(instance: np.ndarray) -> Dict[str, float]:
    flat = instance.reshape(-1)
    differences = np.diff(instance, axis=1)
    return {
        "minimum": float(np.min(flat)),
        "maximum": float(np.max(flat)),
        "mean": float(np.mean(flat)),
        "median": float(np.median(flat)),
        "std": float(np.std(flat)),
        "range": float(np.ptp(flat)),
        "mean_abs_step": float(np.mean(np.abs(differences))) if differences.size else 0.0,
        "n_turning_points": int(
            np.sum(np.sign(differences[:, 1:]) != np.sign(differences[:, :-1]))
        )
        if differences.shape[1] > 1
        else 0,
    }


def _segment_separation(instance: np.ndarray, segmentation: Segmentation) -> float:
    """Share of the variance of the series explained by the segment means."""
    total, within = 0.0, 0.0
    for channel in range(segmentation.n_channels):
        series = instance[channel]
        total += float(np.sum((series - series.mean()) ** 2))
        for segment in segmentation:
            if segment.channel != channel:
                continue
            window = series[segment.slice()]
            within += float(np.sum((window - window.mean()) ** 2))
    if total <= 0:
        return 0.0
    return float(max(0.0, min(1.0, 1.0 - within / total)))


def describe_instance(
    instance: np.ndarray,
    segmentation: Segmentation,
    probabilities: Optional[np.ndarray] = None,
    label: Optional[int] = None,
    class_names=None,
    true_label=None,
) -> Dict[str, object]:
    """Gather every fact the report prints, as a dictionary.

    Parameters
    ----------
    instance : array
        The series, shaped ``(n_timestamps,)`` or ``(n_channels, n_timestamps)``.
    segmentation : Segmentation
        The partition to describe.
    probabilities : array, optional
        Class probabilities of the instance. The section about the model is
        omitted when they are not given.
    label : int, optional
        Position of the explained class. The most probable one is used by
        default.
    class_names : sequence, optional
        Labels in the order of the probability columns, used for reporting.
    true_label : optional
        Ground truth label, which turns on the correctness verdict.

    Returns
    -------
    dict
        The same information :func:`instance_report` formats, so the values can
        be logged or tabulated instead of printed.
    """
    instance = np.asarray(instance, dtype=float)
    if instance.ndim == 1:
        instance = instance.reshape(1, -1)
    lengths = segmentation.lengths

    facts: Dict[str, object] = {
        "n_channels": int(instance.shape[0]),
        "n_timestamps": int(instance.shape[1]),
        "series": _series_statistics(instance),
        "segmentation": {
            "method": segmentation.method,
            "n_segments": segmentation.n_segments,
            "shortest": int(lengths.min()),
            "longest": int(lengths.max()),
            "mean_length": float(lengths.mean()),
            "coverage": segmentation.coverage,
            "separation": _segment_separation(instance, segmentation),
            "params": segmentation.params,
        },
    }

    if probabilities is not None:
        probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
        position = int(np.argmax(probabilities)) if label is None else int(label)
        ordered = np.argsort(-probabilities)
        runner_up = int(ordered[1]) if probabilities.size > 1 else position
        names = (
            [str(name) for name in class_names]
            if class_names is not None
            else [str(k) for k in range(probabilities.size)]
        )
        entropy = float(
            -np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, None)))
        )
        facts["model"] = {
            "predicted": names[position],
            "probability": float(probabilities[position]),
            "runner_up": names[runner_up],
            "runner_up_probability": float(probabilities[runner_up]),
            "margin": float(probabilities[position] - probabilities[runner_up]),
            "entropy": entropy,
            "normalized_entropy": entropy / float(np.log(max(probabilities.size, 2))),
            "true_label": None if true_label is None else str(true_label),
            "correct": None
            if true_label is None
            else bool(str(true_label) == names[position]),
        }
    return facts


def instance_report(facts: Dict[str, object], width: int = 78) -> str:
    """Format the facts collected by :func:`describe_instance` for a console.

    Parameters
    ----------
    facts : dict
        The dictionary returned by :func:`describe_instance`.
    width : int, default=78
        Width of the rules that separate the sections.

    Returns
    -------
    str
        The report, ready to be printed.
    """
    series = facts["series"]
    seg = facts["segmentation"]
    lines = ["=" * width]
    lines.append("Instance under study")
    lines.append("=" * width)

    lines.append(
        f"  shape          {facts['n_channels']} channel(s) of "
        f"{facts['n_timestamps']} time points"
    )
    lines.append(
        f"  range          [{series['minimum']:.3f}, {series['maximum']:.3f}], "
        f"span {series['range']:.3f}"
    )
    lines.append(
        f"  center         mean {series['mean']:.3f}, median {series['median']:.3f}, "
        f"deviation {series['std']:.3f}"
    )
    lines.append(
        f"  roughness      mean absolute step {series['mean_abs_step']:.4f}, "
        f"{series['n_turning_points']} turning points"
    )

    if "model" in facts:
        model = facts["model"]
        lines.append("-" * width)
        verdict = ""
        if model["correct"] is not None:
            verdict = ", correct" if model["correct"] else ", MISCLASSIFIED"
        lines.append(
            f"  prediction     class {model['predicted']} at probability "
            f"{model['probability']:.4f}{verdict}"
        )
        lines.append(
            f"  runner up      class {model['runner_up']} at "
            f"{model['runner_up_probability']:.4f}, margin {model['margin']:.4f}"
        )
        lines.append(
            f"  uncertainty    entropy {model['entropy']:.4f}, normalized "
            f"{model['normalized_entropy']:.3f}"
        )
        if model["margin"] < 0.1:
            lines.append(
                "  note           the decision is close to a boundary, so the "
                "explanation may be unstable"
            )
        if model["correct"] is False:
            lines.append(
                "  note           SOFI assumes the prediction is one the model "
                "handles well, and this instance is misclassified"
            )

    lines.append("-" * width)
    lines.append(
        f"  segmentation   {seg['method']}, {seg['n_segments']} segments covering "
        f"{seg['coverage'] * 100:.0f}% of the series"
    )
    lines.append(
        f"  lengths        shortest {seg['shortest']}, longest {seg['longest']}, "
        f"mean {seg['mean_length']:.1f}"
    )
    lines.append(
        f"  separation     {seg['separation']:.3f} of the variance is explained by "
        "the segment means"
    )
    if seg["separation"] < 0.2:
        lines.append(
            "  note           the segments are barely more homogeneous than "
            "arbitrary windows, so consider a smaller penalty or another cost"
        )
    if seg["n_segments"] < 2:
        lines.append(
            "  note           a single segment leaves nothing to rank, so the "
            "explanation will be trivial"
        )
    if seg["n_segments"] > 25:
        lines.append(
            "  note           many segments make the search slower and the "
            "explanation harder to read"
        )
    lines.append("=" * width)
    return "\n".join(lines)
