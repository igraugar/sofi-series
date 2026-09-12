"""Container returned by the explainer.

An explanation holds the ranked segments, both perturbation curves on the raw
and on the reported scale, the scores derived from them, and enough context for
a reader to know what the ranking means. That last point matters more than it
may appear, since the reading of a ranking depends on the operator that produced
it. A class agnostic substitution answers which segments the prediction relies
on, whereas a class directed one answers which segments move the model away from
the current class fastest. The summary states which of the two applies rather
than leaving it to the reader.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .objective import curve_auc
from .segmentation import Segmentation

__all__ = ["SOFIExplanation", "aggregate_explanations", "noise_onset"]


def noise_onset(scores: Sequence[float], tolerance: float = 0.05) -> Optional[int]:
    """Step after which a perturbation curve recovers instead of falling further.

    Parameters
    ----------
    scores : sequence of float
        A perturbation curve, starting with the unperturbed response.
    tolerance : float, default=0.05
        Fraction of the range of the curve the recovery must exceed before the
        tail is declared uninformative, which keeps a numerical wobble from
        being read as a recovery.

    Returns
    -------
    int or None
        Position of the minimum when a systematic recovery follows it.


    Once the informative segments are gone, marginalizing what remains often
    pushes the response back towards its original state, since the substituted
    values are typical of the training data rather than hostile to the class.
    The tail of the ranking beyond that point carries no evidence. ``None`` says
    no systematic recovery occurred.
    """
    values = np.asarray(scores, dtype=float)
    if values.size < 3:
        return None
    boundary = int(np.argmin(values))
    if boundary >= values.size - 1:
        return None
    span = float(np.max(values) - np.min(values))
    rise = float(values[-1] - values[boundary])
    if rise < max(float(tolerance) * span, 1e-9):
        return None
    return boundary


@dataclass
class SOFIExplanation:
    """Result of one SOFI run on one instance.

    Attributes
    ----------
    order : list of int
        Positions of the segments, from the most to the least relevant.
    morf_scores, lerf_scores : list of float
        Probability the model assigns to the explained class along the ranking
        and along its reverse, each starting with the unperturbed one. No
        rescaling is applied, so a value of 0.4 is a probability of 0.4.
    ds : float
        Degradation score, the area between the two curves, which the search
        maximizes.
    segmentation : Segmentation
        The partition whose parts the ranking permutes.
    label : int
        Position of the explained class among the probability columns.
    class_name : str
        Readable name of that class.
    sparsity_point : int or None
        Step at which the MoRF order first changes the predicted class, and
        ``None`` when the class survives every marginalization.
    sparsity_rate : float
        That step over the segment count, in the unit interval, so that lower
        values denote sparser explanations. A ranking that never changes the
        class receives one.
    sparsity_probability : float
        Probability of the explained class at that step.
    lerf_sparsity_point : int or None
        The same step for the reversed ranking, which should arrive far later.
    perturbation : dict
        Summary of the operator, holding its name, its family, its detail line
        and the reading that follows from it.
    modularity_gap : float or None
        Largest departure from the modularity assumption along the ranking, and
        ``None`` when it was not measured.
    instance : ndarray
        The series that was explained.
    states : ndarray
        The series along the cumulative marginalization, one row per step, the
        first of them untouched.
    history : list of dict
        One record per proposal of the search.
    n_iterations, n_restarts_used, n_evaluations, n_model_calls : int
        Proposals made, restarts used, rankings evaluated and queries sent to
        the model.
    runtime : float
        Seconds spent on the run.
    noise_tolerance : float
        Fraction of the range of the curve a recovery must exceed before the
        tail is declared uninformative.
    source : SOFIExplainer
        The explainer that produced the result. The animation consults it to
        rebuild the states of the LeRF pass, which are not carried here. It is
        dropped when an explanation is pickled, since a fitted model is not
        always serializable and the figures that need it can be redrawn from a
        fresh run.
    """

    segmentation: Segmentation
    order: List[int]
    morf_scores: List[float]
    lerf_scores: List[float]
    ds: float
    label: int
    class_name: str
    sparsity_point: Optional[int] = None
    sparsity_rate: float = 1.0
    sparsity_probability: float = 0.0
    lerf_sparsity_point: Optional[int] = None
    perturbation: Dict[str, object] = field(default_factory=dict)
    modularity_gap: Optional[float] = None
    instance: Optional[np.ndarray] = field(default=None, repr=False)
    states: Optional[np.ndarray] = field(default=None, repr=False)
    history: List[Dict] = field(default_factory=list, repr=False)
    n_iterations: int = 0
    n_restarts_used: int = 0
    n_evaluations: int = 0
    n_model_calls: int = 0
    runtime: float = 0.0
    noise_tolerance: float = 0.05
    source: object = field(default=None, repr=False)

    # ------------------------------------------------------------- accessors
    @property
    def n_segments(self) -> int:
        return self.segmentation.n_segments

    @property
    def segment_names(self) -> List[str]:
        """Labels of the segments in the order of the segmentation."""
        return self.segmentation.labels

    @property
    def ranking(self) -> List[str]:
        """Segments sorted from the most to the least important, by name."""
        return [self.segmentation[position].label for position in self.order]

    @property
    def segments(self) -> List:
        """Segment objects in ranked order."""
        return [self.segmentation[position] for position in self.order]

    def top(self, k: int = 5) -> List[str]:
        """First ``k`` segments of the ranking.

        Parameters
        ----------
        k : int, default=5
            Number of segments returned, counted from the most important one.
        """
        return self.ranking[: int(k)]

    @property
    def scores(self) -> List[float]:
        """Alias of ``morf_scores``."""
        return self.morf_scores

    @property
    def baseline_probability(self) -> float:
        """Probability of the explained class before any marginalization."""
        return float(self.morf_scores[0])

    @property
    def auc_morf(self) -> float:
        """Mean height of the MoRF curve."""
        return curve_auc(self.morf_scores)

    @property
    def auc_lerf(self) -> float:
        """Mean height of the LeRF curve."""
        return curve_auc(self.lerf_scores)

    @property
    def drops(self) -> np.ndarray:
        """Degradation attributable to each step of the MoRF curve."""
        return -np.diff(np.asarray(self.morf_scores, dtype=float))

    @property
    def lerf_sparsity_rate(self) -> float:
        """Where the reverse order changes the class, on the sparsity scale.

        Marginalizing from the least relevant segment onwards should need far
        more segments to change the decision than marginalizing from the most
        relevant one, so the separation between this rate and ``sparsity_rate``
        quantifies the advantage of the ranking over its reverse.
        """
        if self.lerf_sparsity_point is None:
            return 1.0
        return self.lerf_sparsity_point / float(self.n_segments)

    @property
    def class_directed(self) -> bool:
        """Whether the operator that produced the ranking consulted the class."""
        return bool(self.perturbation.get("class_directed", False))

    @property
    def interpretation(self) -> str:
        """Sentence stating what the ranking of this run means."""
        return str(
            self.perturbation.get(
                "interpretation",
                "Segments are ranked by how much the prediction relies on them.",
            )
        )

    # ----------------------------------------------------------- noise region
    @property
    def noise_onset(self) -> Optional[int]:
        """Start of the region where the probability recovers instead of falling."""
        return noise_onset(self.morf_scores, self.noise_tolerance)

    @property
    def noise_features(self) -> List[str]:
        """Segments of the ranking that fall inside the noise region."""
        boundary = self.noise_onset
        return [] if boundary is None else self.ranking[boundary:]

    @property
    def importances(self) -> np.ndarray:
        """Rank based importance of every segment, in the unit interval.

        The weight of a segment follows its position in the ranking and nothing
        else, so it inherits the meaning of the ranking without claiming that a
        segment carries an independent score. Segments inside the noise region
        receive no weight, since the ranking says nothing about them. The values
        are in the order of the segmentation.
        """
        boundary = self.noise_onset
        informative = self.n_segments if boundary is None else boundary
        weights = np.zeros(self.n_segments, dtype=float)
        if informative <= 0:
            return weights
        raw = np.arange(informative, 0, -1, dtype=float)
        raw = raw / raw.sum()
        for rank, position in enumerate(self.order[:informative]):
            weights[position] = raw[rank]
        return weights

    # -------------------------------------------------------------- reporting
    def to_frame(self):
        """Tabular view of the ranking, one row per marginalization step."""
        import pandas as pd

        weights = self.importances
        return pd.DataFrame(
            {
                "step": np.arange(1, self.n_segments + 1),
                "segment": self.ranking,
                "channel": [s.channel for s in self.segments],
                "start": [s.start for s in self.segments],
                "end": [s.end for s in self.segments],
                "probability": np.asarray(self.morf_scores[1:], dtype=float),
                "drop": self.drops,
                "importance": [weights[p] for p in self.order],
                "noise": [
                    name in set(self.noise_features) for name in self.ranking
                ],
            }
        )

    @property
    def statistics(self) -> Dict[str, object]:
        """Every figure of the run gathered in one dictionary."""
        return {
            "n_segments": self.n_segments,
            "segmentation": self.segmentation.method,
            "operator": self.perturbation.get("operator"),
            "class_directed": self.class_directed,
            "explained_class": self.class_name,
            "unperturbed_probability": self.baseline_probability,
            "degradation_score": self.ds,
            "auc_morf": self.auc_morf,
            "auc_lerf": self.auc_lerf,
            "sparsity_point": self.sparsity_point,
            "sparsity_rate": self.sparsity_rate,
            "lerf_sparsity_point": self.lerf_sparsity_point,
            "lerf_sparsity_rate": self.lerf_sparsity_rate,
            "noise_onset": self.noise_onset,
            "modularity_gap": self.modularity_gap,
            "n_iterations": self.n_iterations,
            "n_restarts_used": self.n_restarts_used,
            "n_evaluations": self.n_evaluations,
            "n_model_calls": self.n_model_calls,
            "runtime": self.runtime,
        }

    def summary(self, top_k: Optional[int] = None) -> str:
        """Report of the run, meant to be printed to the console.

        Parameters
        ----------
        top_k : int, optional
            Number of ranked segments listed. Every segment is listed by
            default, and the count of those left out closes the line otherwise.
        """
        listed = self.ranking if top_k is None else self.ranking[: int(top_k)]
        listing = ", ".join(
            f"{position + 1}. {name}" for position, name in enumerate(listed)
        )
        remaining = len(self.order) - len(listed)
        if remaining > 0:
            listing += f", and {remaining} more"

        boundary = self.noise_onset
        if boundary is None:
            noise = "none detected"
        else:
            noise = f"from step {boundary}, {', '.join(self.noise_features)}"

        if self.sparsity_point is None:
            flip = f"never, over {self.n_segments} segments"
        else:
            flip = (
                f"after {self.sparsity_point} of {self.n_segments} segments, "
                f"rate {self.sparsity_rate:.3f}"
            )
        if self.lerf_sparsity_point is None:
            flip += ", never under LeRF"
        else:
            flip += f", against step {self.lerf_sparsity_point} under LeRF"

        if self.modularity_gap is None:
            modular = "not evaluated"
        elif self.modularity_gap < 0.02:
            modular = f"{self.modularity_gap:.4f}, the greedy ranking is near optimal"
        else:
            modular = f"{self.modularity_gap:.4f}, segment effects are conditional"

        rows = [
            ("explained class", self.class_name),
            ("segmentation", f"{self.segmentation.method}, {self.n_segments} segments"),
            ("marginalization", str(self.perturbation.get("detail", ""))),
            ("reading", self.interpretation),
            ("degradation score", f"{self.ds:.4f}"),
            ("area under MoRF", f"{self.auc_morf:.4f}"),
            ("area under LeRF", f"{self.auc_lerf:.4f}"),
            ("unperturbed probability", f"{self.baseline_probability:.4f}"),
            ("class change", flip),
            ("noise region", noise),
            ("modularity gap", modular),
            (
                "search",
                f"{self.n_iterations} iterations, {self.n_restarts_used} restarts, "
                f"{self.n_evaluations} evaluations, {self.n_model_calls} model queries",
            ),
            ("ranking", listing),
        ]
        width = max(len(label) for label, _ in rows)
        lines = ["SOFI explanation"]
        lines += [f"  {label.ljust(width)}   {value}" for label, value in rows]
        return "\n".join(lines)

    # ---------------------------------------------------------------- figures
    def plot_segmentation(self, **kwargs):
        """Draw the series with the breakpoints that divide it.

        Parameters
        ----------
        **kwargs : dict
            Settings of :func:`~sofits.plotting.plot_segmentation`, such as
            ``style``, ``marker_color``, ``linewidth``, ``save_path`` or ``ax``.

        Returns
        -------
        matplotlib Axes
            The axes the series was drawn on.
        """
        from .plotting import plot_segmentation

        return plot_segmentation(self.instance, self.segmentation, **kwargs)

    def plot_degradation_curves(self, **kwargs):
        """Draw the two degradation curves and the area between them.

        Parameters
        ----------
        **kwargs : dict
            Settings of :func:`~sofits.plotting.plot_degradation_curves`, such as
            ``linewidth``, ``markersize``, ``morf_color``, ``lerf_color``,
            ``title``, ``save_path`` or ``ax``.

        Returns
        -------
        matplotlib Axes
            The axes the curves were drawn on.
        """
        from .plotting import plot_degradation_curves

        return plot_degradation_curves(self, **kwargs)

    def plot_explanation(self, **kwargs):
        """Draw the explanation as two panels.

        The left panel carries the series with the segments the ranking retains
        and the right panel the curves from which the ranking was obtained. The
        animation closes on this same figure, so the two agree by construction.

        Parameters
        ----------
        **kwargs : dict
            Settings of :func:`~sofits.plotting.plot_explanation`, such as
            ``title``, ``figsize``, ``linewidth``, ``markersize``,
            ``marker_color`` or ``save_path``.

        Returns
        -------
        matplotlib Figure
            The figure holding the two panels.
        """
        from .plotting import plot_explanation

        return plot_explanation(self, **kwargs)

    def plot_marginalization(self, **kwargs):
        """Draw the series along the cumulative marginalization of the ranking.

        Parameters
        ----------
        **kwargs : dict
            Settings of :func:`~sofits.plotting.plot_marginalization`, such as
            ``steps``, ``ncols``, ``panel_size``, ``title``, ``linewidth`` or
            ``save_path``.

        Returns
        -------
        matplotlib Figure
            The figure holding the panels.
        """
        from .plotting import plot_marginalization

        return plot_marginalization(self, **kwargs)

    def plot_animation(self, save_path=None, **kwargs):
        """Render the two passes of the explanation as an animated GIF.

        The animation is returned rather than only written, so a notebook cell
        that ends on the call displays it. The closing frame is the figure
        :meth:`plot_explanation` produces, since both are drawn from this result
        with the same segments and the same operator.

        Parameters
        ----------
        save_path : str or Path, optional
            Destination file. A path without an extension receives ``.gif``.
            Nothing is written when it is left out.
        **kwargs : dict
            Settings of :func:`~sofits.animation.animate_marginalization`, such
            as ``fps``, ``step_ms``, ``dpi``, ``linewidth`` or ``progress``.

        Returns
        -------
        Animation
            The encoded GIF, displayed by a notebook and written by ``save``.
        """
        from .animation import animate_marginalization

        return animate_marginalization(
            self, explainer=self.source, save_path=save_path, **kwargs
        )

    def __getstate__(self):
        # the explainer holds a fitted model, which is not always serializable,
        # so it is left out and the figures that need it are redrawn from a run
        state = dict(self.__dict__)
        state["source"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.summary()

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"SOFIExplanation(class={self.class_name}, segments={self.n_segments}, "
            f"ds={self.ds:.3f}, sparsity={self.sparsity_rate:.3f})"
        )


def aggregate_explanations(explanations: Sequence[SOFIExplanation]):
    """Summarize explanations produced on a segmentation shared by every instance.

    Parameters
    ----------
    explanations : sequence of SOFIExplanation
        Explanations to pool, all of them built on the same partition.

    Returns
    -------
    DataFrame
        One row per segment with its mean rank, the deviation of that rank and
        how often the segment reached the first three positions, sorted from the
        most to the least important.


    The mean rank orders the segments by their average position and the
    frequency column reports how often a segment reaches the first three
    positions. Instances segmented individually cannot be pooled this way, since
    their segment identities do not correspond, and the function says so.
    """
    import pandas as pd

    if len(explanations) == 0:
        raise ValueError("At least one explanation is required.")
    widths = {explanation.n_segments for explanation in explanations}
    if len(widths) > 1:
        raise ValueError(
            "The explanations do not share a segmentation, so their segments "
            "cannot be pooled. Aggregation needs a segmentation shared by every "
            "instance, such as expert breakpoints or uniform binning."
        )

    n_segments = explanations[0].n_segments
    ranks = np.zeros((len(explanations), n_segments), dtype=float)
    for row, explanation in enumerate(explanations):
        for rank, position in enumerate(explanation.order):
            ranks[row, position] = rank + 1

    frame = pd.DataFrame(
        {
            "segment": explanations[0].segmentation.labels,
            "mean_rank": ranks.mean(axis=0),
            "std_rank": ranks.std(axis=0),
            "top3_frequency": (ranks <= 3).mean(axis=0),
        }
    )
    return frame.sort_values("mean_rank").reset_index(drop=True)
