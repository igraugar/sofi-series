"""Operators that neutralize a segment of a time series.

Marginalizing a tabular feature is easy, since one statistic of the training
column carries no instance level information. A time series segment is harder.
A flat constant is itself a shape, and a model may read it as evidence for some
class, which leaves the perturbed series informative and the degradation curve
misleading. The operators below therefore span three levels of sophistication,
and the right one is problem dependent, as the paper concludes.

Two families exist and they carry different readings of the resulting ranking.

**Class-agnostic operators** erase the content of a segment without steering the
prediction anywhere. The substituted values come from the training
distribution, from the segment itself or from noise, and none of them consults
the class being explained. A ranking obtained this way answers the question the
method was designed for, namely which segments the model relies on. These
operators are the default.

**Class-directed operators** choose the substitution so that the response of the
model falls as fast as possible, either through a search over constants or by
importing evidence from the other classes. They degrade the prediction sooner
and produce sparser explanations, and the reading changes accordingly. A
ranking obtained this way answers which segments move the model away from the
current class most quickly, which is a counterfactual question rather than a
reliance question. Every explanation states which family produced it, so the
two readings are never confused.

Every operator can request several replicas. The response is then averaged over
the draws, which approximates an expectation instead of the output at one
arbitrary point, at a cost that grows linearly with the draw count.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from .segmentation import Segment

__all__ = [
    "PerturbationContext",
    "Perturbation",
    "ConstantPerturbation",
    "RandomConstantPerturbation",
    "NoisePerturbation",
    "SegmentMeanPerturbation",
    "LinearPerturbation",
    "BackgroundPerturbation",
    "LineSearchPerturbation",
    "OppositeClassMeanPerturbation",
    "NearestUnlikeNeighborPerturbation",
    "make_perturbation",
    "PERTURBATIONS",
    "CLASS_AGNOSTIC",
    "CLASS_DIRECTED",
    "DEFAULT_CANDIDATES",
]


@dataclass
class PerturbationContext:
    """Everything an operator may consult before it starts substituting.

    Attributes
    ----------
    instance : ndarray
        The series being explained, shaped ``(n_channels, n_timestamps)``.
    label : int
        Position of the class whose probability the search tracks, normally the
        class the model predicts for the unperturbed instance.
    wrapper : ClassifierWrapper
        Access to the black box, needed by the operators that search.
    X_train : ndarray, optional
        Training instances shaped ``(n, n_channels, n_timestamps)``, the only
        source of substitution values. The data being explained never
        contributes statistics.
    y_train : ndarray, optional
        Training labels, needed by the class-directed operators alone.
    rng : Generator
        Source of randomness, so that a run is reproducible.
    """

    instance: np.ndarray
    label: int
    wrapper: object
    X_train: Optional[np.ndarray] = None
    y_train: Optional[np.ndarray] = None
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    def require_training(self, operator: str) -> np.ndarray:
        """Training data, or a readable error naming the operator that needs it.

        Parameters
        ----------
        operator : str
            Name of the operator, used in the message.
        """
        if self.X_train is None:
            raise ValueError(
                f"The '{operator}' operator needs training data, since it is the "
                "only admissible source of substitution values. Pass X_train to "
                "the explainer."
            )
        return np.asarray(self.X_train, dtype=float)

    def require_labels(self, operator: str) -> np.ndarray:
        """Training labels, or a readable error naming the operator that needs them.

        Parameters
        ----------
        operator : str
            Name of the operator, used in the message.
        """
        if self.y_train is None:
            raise ValueError(
                f"The '{operator}' operator needs training labels, since it "
                "draws its substitution from the classes other than the one "
                "being explained. Pass y_train to the explainer."
            )
        return np.asarray(self.y_train)

    @property
    def statistics(self) -> Dict[str, float]:
        """Statistics of the training data pooled over every channel."""
        data = self.require_training("statistics")
        return {
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "mean": float(np.mean(data)),
            "median": float(np.median(data)),
            "std": float(np.std(data)),
        }

    @property
    def channel_statistics(self) -> Dict[str, np.ndarray]:
        """The same statistics computed for each channel separately.

        Channels of a multivariate series often carry different physical
        quantities on different scales, so one pooled constant would neutralize
        one channel while it introduced a gross outlier in another. Every
        constant operator therefore reads the statistic of the channel the
        segment belongs to, which reduces to the pooled value when the series is
        univariate.
        """
        data = self.require_training("channel statistics")
        axes = (0, 2)
        return {
            "min": np.min(data, axis=axes),
            "max": np.max(data, axis=axes),
            "mean": np.mean(data, axis=axes),
            "median": np.median(data, axis=axes),
            "std": np.std(data, axis=axes),
        }


class Perturbation:
    """Base class of every substitution operator.

    Subclasses implement :meth:`_fit`, which prepares whatever the operator
    needs, and :meth:`replacement`, which returns the values that replace one
    segment on one replica.

    Parameters
    ----------
    n_replicas : int, default=1
        Number of independent substitutions averaged at every step. Operators
        that draw randomly benefit from several replicas, and deterministic
        ones ignore the argument.
    taper : int, default=0
        Number of time points over which the substituted values are blended
        into their neighbors at each end of a segment. A constant substitution
        introduces step discontinuities at the breakpoints, and convolutional
        models react to sharp edges, so part of the measured degradation may be
        a boundary artifact rather than lost evidence. The default of zero
        reproduces the published behavior.
    """

    name = "perturbation"
    class_directed = False
    interpretation = (
        "Segments are ranked by how much the prediction relies on them."
    )

    def __init__(self, n_replicas: int = 1, taper: int = 0):
        if int(n_replicas) < 1:
            raise ValueError("n_replicas must be a positive integer.")
        if int(taper) < 0:
            raise ValueError("taper must be zero or a positive integer.")
        self.n_replicas = int(n_replicas)
        self.taper = int(taper)
        self.context: Optional[PerturbationContext] = None

    # ------------------------------------------------------------------ fit
    def fit(self, context: PerturbationContext) -> "Perturbation":
        """Prepare the operator for one instance.

        Parameters
        ----------
        context : PerturbationContext
            Everything the operator may consult, namely the instance, the class
            being explained, the classifier, the training data and its labels,
            and the source of randomness.

        Returns
        -------
        Perturbation
            The operator itself, so that the call chains.
        """
        self.context = context
        self._fit(context)
        return self

    def _fit(self, context: PerturbationContext) -> None:  # pragma: no cover - hook
        return None

    # -------------------------------------------------------------- applying
    def replacement(self, segment: Segment, replica: int = 0) -> np.ndarray:
        """Values that replace one segment on one replica.

        Parameters
        ----------
        segment : Segment
            The segment being neutralized.
        replica : int, default=0
            Which of the ``n_replicas`` independent substitutions is wanted.
            Deterministic operators ignore it.

        Returns
        -------
        ndarray
            One value per time point of the segment.
        """
        raise NotImplementedError

    def warn_if_taper_is_idle(self, segmentation) -> None:
        """Say so when the taper is too wide to apply anywhere.

        Blending needs room on both sides of a segment, so a taper wider than
        half the shortest segment is silently skipped. Announcing it prevents a
        reader from crediting the taper with a result it never touched.

        Parameters
        ----------
        segmentation : Segmentation
            The partition the operator will be applied to.
        """
        if self.taper <= 0:
            return
        shortest = int(min(segment.length for segment in segmentation))
        if shortest <= 2 * self.taper:
            warnings.warn(
                f"taper={self.taper} needs segments longer than {2 * self.taper} "
                f"time points and the shortest one holds {shortest}, so the "
                "blending is skipped there and the substitution stays abrupt.",
                stacklevel=3,
            )

    def apply(self, work: np.ndarray, segment: Segment, replica: int = 0) -> None:
        """Substitute one segment of ``work``, in place.

        Parameters
        ----------
        work : ndarray
            A single instance shaped ``(n_channels, n_timestamps)``, which the
            objective mutates cumulatively along a ranking.
        segment : Segment
            The segment to neutralize.
        replica : int, default=0
            Which of the independent substitutions is applied.
        """
        values = np.asarray(self.replacement(segment, replica), dtype=float)
        span = segment.slice()
        if self.taper > 0 and segment.length > 2 * self.taper:
            original = self.context.instance[segment.channel, span]
            weights = np.ones(segment.length, dtype=float)
            ramp = np.linspace(0.0, 1.0, self.taper + 2)[1:-1]
            weights[: self.taper] = ramp
            weights[-self.taper :] = ramp[::-1]
            values = weights * values + (1.0 - weights) * original
        work[segment.channel, span] = values

    # -------------------------------------------------------------- reporting
    def describe(self) -> str:
        return self.name

    def summary(self) -> Dict[str, object]:
        return {
            "operator": self.name,
            "class_directed": self.class_directed,
            "n_replicas": self.n_replicas,
            "taper": self.taper,
            "detail": self.describe(),
            "interpretation": self.interpretation,
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}({self.describe()})"


# ------------------------------------------------------- class agnostic
class ConstantPerturbation(Perturbation):
    """Replace a segment with one constant estimated on the training data.

    This is the family used in the paper. The statistic is global, so it is the
    same for every segment and every channel, which keeps the substituted level
    typical of the data the model was trained on. The training mean is the most
    reliable choice on the benchmark problems and it is the default of the
    package.

    Parameters
    ----------
    statistic : {"mean", "min", "max", "median", "zero"} or float
        Value that replaces the segment. A float is used verbatim, which covers
        a constant chosen from domain knowledge.
    n_replicas : int, default=1
        Number of independent substitutions averaged at every step.
    taper : int, default=0
        Number of time points over which the substituted values are blended into
        their neighbors at each end of a segment, which removes the step
        discontinuity a flat substitution introduces at a breakpoint. Zero
        reproduces the published behavior.
    """

    name = "constant"

    def __init__(self, statistic="mean", n_replicas: int = 1, taper: int = 0):
        super().__init__(n_replicas=n_replicas, taper=taper)
        self.statistic = statistic
        self.value_: Optional[float] = None

    def _fit(self, context: PerturbationContext) -> None:
        n_channels = np.asarray(context.instance).shape[0]
        if isinstance(self.statistic, (int, float, np.floating, np.integer)):
            self.values_ = np.full(n_channels, float(self.statistic))
        elif str(self.statistic).lower() == "zero":
            self.values_ = np.zeros(n_channels)
        else:
            key = str(self.statistic).lower()
            stats = context.channel_statistics
            if key not in stats:
                raise ValueError(
                    f"Unknown constant statistic '{self.statistic}'. Choose among "
                    "'mean', 'min', 'max', 'median', 'zero', or pass a number."
                )
            values = np.asarray(stats[key], dtype=float).reshape(-1)
            if values.size < n_channels:
                values = np.resize(values, n_channels)
            self.values_ = values[:n_channels]
        self.value_ = float(np.mean(self.values_))

    def replacement(self, segment: Segment, replica: int = 0) -> np.ndarray:
        """The constant of the channel the segment belongs to.

        Parameters
        ----------
        segment : Segment
            The segment being neutralized.
        replica : int, default=0
            Which of the independent substitutions is wanted.

        Returns
        -------
        ndarray
            One value per time point of the segment.
        """
        channel = min(segment.channel, self.values_.size - 1)
        return np.full(segment.length, float(self.values_[channel]), dtype=float)

    def describe(self) -> str:
        if self.values_.size > 1 and float(np.ptp(self.values_)) > 1e-9:
            return (
                f"constant {self.statistic} per channel, from "
                f"{self.values_.min():.4f} to {self.values_.max():.4f}"
            )
        return f"constant {self.statistic} at {self.value_:.4f}"


class RandomConstantPerturbation(Perturbation):
    """Replace a segment with a constant drawn uniformly from the training range.

    A fresh value is drawn for every segment and every replica, so the operator
    carries no systematic level of its own. Several replicas are recommended,
    since a single draw is one arbitrary point of the range.

    Parameters
    ----------
    n_replicas : int, default=5
        Number of independent substitutions averaged at every step.
    taper : int, default=0
        Number of time points over which the substituted values are blended into
        their neighbors at each end of a segment, which removes the step
        discontinuity a flat substitution introduces at a breakpoint. Zero
        reproduces the published behavior.
    """

    name = "random"

    def __init__(self, n_replicas: int = 5, taper: int = 0):
        super().__init__(n_replicas=n_replicas, taper=taper)
        self.draws_: Dict[int, np.ndarray] = {}

    def _fit(self, context: PerturbationContext) -> None:
        stats = context.channel_statistics
        self.low_ = np.asarray(stats["min"], dtype=float).reshape(-1)
        self.high_ = np.asarray(stats["max"], dtype=float).reshape(-1)
        self.draws_ = {}
        self._rng = context.rng

    def replacement(self, segment: Segment, replica: int = 0) -> np.ndarray:
        """One uniform draw, fixed per segment and replica.

        Parameters
        ----------
        segment : Segment
            The segment being neutralized.
        replica : int, default=0
            Which of the independent substitutions is wanted.

        Returns
        -------
        ndarray
            One value per time point of the segment.
        """
        key = (segment.index, replica)
        if key not in self.draws_:
            channel = min(segment.channel, self.low_.size - 1)
            self.draws_[key] = float(
                self._rng.uniform(self.low_[channel], self.high_[channel])
            )
        return np.full(segment.length, self.draws_[key], dtype=float)

    def describe(self) -> str:
        return f"uniform draw on [{self.low_.min():.3f}, {self.high_.max():.3f}]"


class NoisePerturbation(Perturbation):
    """Replace a segment with Gaussian noise matched to the training data.

    Level and dispersion follow the training set, so the substituted stretch
    stays plausible in scale while carrying no temporal structure. The draws are
    fixed once per segment and replica, so competing rankings meet identical
    conditions.

    Parameters
    ----------
    n_replicas : int, default=5
        Number of independent draws averaged at every step.
    taper : int, default=0
        Number of time points over which the substituted values are blended into
        their neighbors at each end of a segment.
    scale : float, default=1.0
        Multiplier applied to the deviation of the training data, so that the
        noise can be made milder or harsher than the signal it replaces.
    """

    name = "noise"

    def __init__(self, n_replicas: int = 5, taper: int = 0, scale: float = 1.0):
        super().__init__(n_replicas=n_replicas, taper=taper)
        self.scale = float(scale)
        self.draws_: Dict[tuple, np.ndarray] = {}

    def _fit(self, context: PerturbationContext) -> None:
        stats = context.channel_statistics
        self.center_ = np.asarray(stats["mean"], dtype=float).reshape(-1)
        self.spread_ = np.asarray(stats["std"], dtype=float).reshape(-1) * self.scale
        self.draws_ = {}
        self._rng = context.rng

    def replacement(self, segment: Segment, replica: int = 0) -> np.ndarray:
        """Gaussian noise, fixed per segment and replica.

        Parameters
        ----------
        segment : Segment
            The segment being neutralized.
        replica : int, default=0
            Which of the independent substitutions is wanted.

        Returns
        -------
        ndarray
            One value per time point of the segment.
        """
        key = (segment.index, replica)
        if key not in self.draws_:
            channel = min(segment.channel, self.center_.size - 1)
            self.draws_[key] = self._rng.normal(
                self.center_[channel], self.spread_[channel], size=segment.length
            )
        return self.draws_[key]

    def describe(self) -> str:
        return (
            f"Gaussian noise per channel, spread from {self.spread_.min():.3f} "
            f"to {self.spread_.max():.3f}"
        )


class SegmentMeanPerturbation(Perturbation):
    """Flatten a segment onto its own average.

    Level and position are preserved and the shape is destroyed, which isolates
    the contribution of the temporal pattern from the contribution of the
    magnitude. The substitution is local to the instance, so no training
    statistic enters, and the perturbed series stays close to the data manifold.
    """

    name = "segment_mean"

    def _fit(self, context: PerturbationContext) -> None:
        self.instance_ = np.asarray(context.instance, dtype=float)

    def replacement(self, segment: Segment, replica: int = 0) -> np.ndarray:
        """The average of the segment itself.

        Parameters
        ----------
        segment : Segment
            The segment being neutralized.
        replica : int, default=0
            Which of the independent substitutions is wanted.

        Returns
        -------
        ndarray
            One value per time point of the segment.
        """
        window = self.instance_[segment.channel, segment.slice()]
        return np.full(segment.length, float(np.mean(window)), dtype=float)

    def describe(self) -> str:
        return "the average of the segment itself"


class LinearPerturbation(Perturbation):
    """Replace a segment with the straight line joining its two endpoints.

    Continuity at the breakpoints is preserved, so no step discontinuity is
    introduced, which removes the boundary artifact that a flat constant can
    provoke in convolutional models. Everything the segment contributed beyond
    a linear trend is removed.
    """

    name = "linear"

    def _fit(self, context: PerturbationContext) -> None:
        self.instance_ = np.asarray(context.instance, dtype=float)

    def replacement(self, segment: Segment, replica: int = 0) -> np.ndarray:
        """The straight line joining the endpoints of the segment.

        Parameters
        ----------
        segment : Segment
            The segment being neutralized.
        replica : int, default=0
            Which of the independent substitutions is wanted.

        Returns
        -------
        ndarray
            One value per time point of the segment.
        """
        series = self.instance_[segment.channel]
        left = series[segment.start - 1] if segment.start > 0 else series[segment.start]
        right = (
            series[segment.end] if segment.end < series.size else series[segment.end - 1]
        )
        return np.linspace(float(left), float(right), segment.length)

    def describe(self) -> str:
        return "linear interpolation between the segment endpoints"


class BackgroundPerturbation(Perturbation):
    """Replace a segment with the same window of training instances.

    The substituted stretch is a real piece of a real series, so the perturbed
    instance stays on the data manifold and the model is never asked about an
    input unlike anything it saw. Averaging over several draws approximates the
    expectation of the response when the segment carries no instance specific
    information, which is the definition marginalization is built on.

    Parameters
    ----------
    n_replicas : int, default=10
        Number of training instances drawn. Larger values smooth the
        expectation and raise the cost linearly.
    exclude_label : bool, default=False
        Whether instances of the explained class are excluded from the draws.
        Leaving it false keeps the operator class-agnostic, which is the
        recommended setting.

    taper : int, default=0
        Number of time points over which the substituted values are blended into
        their neighbors at each end of a segment.
    """

    name = "background"

    def __init__(self, n_replicas: int = 10, taper: int = 0, exclude_label: bool = False):
        super().__init__(n_replicas=n_replicas, taper=taper)
        self.exclude_label = bool(exclude_label)
        self.class_directed = bool(exclude_label)

    def _fit(self, context: PerturbationContext) -> None:
        data = context.require_training(self.name)
        if self.exclude_label:
            labels = context.require_labels(self.name)
            keep = np.asarray(labels) != labels[0].__class__(context.label)
            if not keep.any():
                raise ValueError(
                    "Every training instance belongs to the class being "
                    "explained, so no background outside it exists."
                )
            data = data[keep]
        size = min(self.n_replicas, len(data))
        positions = context.rng.choice(len(data), size=size, replace=False)
        self.pool_ = data[np.sort(positions)]
        self.n_replicas = size

    def replacement(self, segment: Segment, replica: int = 0) -> np.ndarray:
        """The matching window of one training instance.

        Parameters
        ----------
        segment : Segment
            The segment being neutralized.
        replica : int, default=0
            Which of the independent substitutions is wanted.

        Returns
        -------
        ndarray
            One value per time point of the segment.
        """
        source = self.pool_[replica % len(self.pool_)]
        channel = min(segment.channel, source.shape[0] - 1)
        return source[channel, segment.slice()]

    def describe(self) -> str:
        scope = "instances outside the explained class" if self.exclude_label else "training instances"
        return f"the matching window of {len(self.pool_)} {scope}"


# -------------------------------------------------------- class directed
_DIRECTED_READING = (
    "Segments are ranked by how quickly they move the model away from the "
    "explained class, which is a counterfactual reading rather than a reliance "
    "reading."
)


class LineSearchPerturbation(Perturbation):
    """Constant found by the golden section search of the paper.

    The interval spanned by the training data is narrowed according to the
    golden ratio, and the constant retained is the one whose flat series
    receives the lowest probability for the explained class. The extremes, the
    training mean, a uniform draw and the midpoint of the final interval are
    evaluated alongside the outcome of the search, exactly as in Algorithm 2.

    Under ``admissible``, the search is restricted to constants whose flat
    series is not assigned the explained class at all. The substitution is then
    known in advance to carry no evidence for that class, which is the strongest
    guarantee a constant can offer. When no such constant exists, the operator
    says so and falls back to the plain minimizer, and the fact that the model
    reads a flat line as the explained class is itself worth knowing.

    Parameters
    ----------
    tolerance : float, default=1e-3
        Width of the interval below which the search stops.
    max_iterations : int, default=100
        Upper bound on the narrowing steps.
    admissible : bool, default=False
        Whether constants assigned to the explained class are rejected.
    taper : int, default=0
        Number of time points over which the substituted values are blended into
        their neighbors at each end of a segment.
    """

    name = "line_search"
    class_directed = True
    interpretation = _DIRECTED_READING

    def __init__(
        self,
        tolerance: float = 1e-3,
        max_iterations: int = 100,
        admissible: bool = False,
        taper: int = 0,
    ):
        super().__init__(n_replicas=1, taper=taper)
        self.tolerance = float(tolerance)
        self.max_iterations = int(max_iterations)
        self.admissible = bool(admissible)

    def _fit(self, context: PerturbationContext) -> None:
        stats = context.statistics
        instance = np.asarray(context.instance, dtype=float)
        wrapper, label = context.wrapper, int(context.label)
        self.fallback_ = False

        def probe(value: float):
            candidate = np.full_like(instance, float(value))
            probabilities = wrapper.predict_proba(candidate[None, ...])[0]
            return float(probabilities[label]), int(np.argmax(probabilities))

        phi = (1.0 + 5.0 ** 0.5) / 2.0
        low, high = stats["min"], stats["max"]
        c = high - (high - low) / phi
        d = low + (high - low) / phi
        f_c, _ = probe(c)
        f_d, _ = probe(d)

        for _ in range(self.max_iterations):
            if abs(high - low) < self.tolerance:
                break
            if f_c < f_d:
                high, d, f_d = d, c, f_c
                c = high - (high - low) / phi
                f_c, _ = probe(c)
            else:
                low, c, f_c = c, d, f_d
                d = low + (high - low) / phi
                f_d, _ = probe(d)

        candidates = [
            stats["min"],
            stats["max"],
            stats["mean"],
            float(context.rng.uniform(stats["min"], stats["max"])),
            0.5 * (low + high),
        ]
        scored = [(value,) + probe(value) for value in candidates]

        if self.admissible:
            outside = [item for item in scored if item[2] != label]
            if outside:
                self.value_ = float(min(outside, key=lambda item: item[1])[0])
                self.escaped_ = True
                self.probability_ = float(min(outside, key=lambda item: item[1])[1])
                return
            self.fallback_ = True

        best = min(scored, key=lambda item: item[1])
        self.value_ = float(best[0])
        self.probability_ = float(best[1])
        self.escaped_ = bool(best[2] != label)

    def replacement(self, segment: Segment, replica: int = 0) -> np.ndarray:
        """The constant found by the golden section search.

        Parameters
        ----------
        segment : Segment
            The segment being neutralized.
        replica : int, default=0
            Which of the independent substitutions is wanted.

        Returns
        -------
        ndarray
            One value per time point of the segment.
        """
        return np.full(segment.length, self.value_, dtype=float)

    def describe(self) -> str:
        verdict = "leaves the explained class" if self.escaped_ else "stays inside the explained class"
        note = ", no admissible constant existed" if self.fallback_ else ""
        return (
            f"constant {self.value_:.4f} from the golden section search, which "
            f"{verdict} at probability {self.probability_:.3f}{note}"
        )


class OppositeClassMeanPerturbation(Perturbation):
    """Replace a segment with the average series of the other classes.

    The paper describes this as the starting point of the constant search. Used
    on its own, it imports the level and shape typical of everything the
    explained class is not, so the perturbed instance drifts toward the
    alternatives rather than toward nothing.
    """

    name = "opposite_mean"
    class_directed = True
    interpretation = _DIRECTED_READING

    def _fit(self, context: PerturbationContext) -> None:
        data = context.require_training(self.name)
        labels = np.asarray(context.require_labels(self.name))
        codes = _as_positions(labels, context)
        keep = codes != int(context.label)
        if not keep.any():
            raise ValueError(
                "Every training instance belongs to the explained class, so the "
                "average of the other classes does not exist."
            )
        self.template_ = data[keep].mean(axis=0)
        self.n_used_ = int(keep.sum())

    def replacement(self, segment: Segment, replica: int = 0) -> np.ndarray:
        """The matching window of the mean of the other classes.

        Parameters
        ----------
        segment : Segment
            The segment being neutralized.
        replica : int, default=0
            Which of the independent substitutions is wanted.

        Returns
        -------
        ndarray
            One value per time point of the segment.
        """
        channel = min(segment.channel, self.template_.shape[0] - 1)
        return self.template_[channel, segment.slice()]

    def describe(self) -> str:
        return f"the mean series of {self.n_used_} instances outside the explained class"


class NearestUnlikeNeighborPerturbation(Perturbation):
    """Replace a segment with the same window of the nearest unlike neighbor.

    The donor is the training instance of another class closest to the one being
    explained under a Euclidean distance. Every substituted stretch is therefore
    real, plausible and drawn from a series the model assigns elsewhere, which
    makes the degradation both fast and realistic. The operator follows the
    logic of counterfactual explanation methods for time series, and its ranking
    should be read accordingly.
    """

    name = "nearest_unlike"
    class_directed = True
    interpretation = _DIRECTED_READING

    def _fit(self, context: PerturbationContext) -> None:
        data = context.require_training(self.name)
        labels = np.asarray(context.require_labels(self.name))
        codes = _as_positions(labels, context)
        keep = codes != int(context.label)
        if not keep.any():
            raise ValueError(
                "No training instance outside the explained class was found, so "
                "the nearest unlike neighbor does not exist."
            )
        pool = data[keep]
        instance = np.asarray(context.instance, dtype=float)
        distances = np.linalg.norm(
            pool.reshape(len(pool), -1) - instance.reshape(1, -1), axis=1
        )
        self.position_ = int(np.argmin(distances))
        self.template_ = pool[self.position_]
        self.distance_ = float(distances[self.position_])

    def replacement(self, segment: Segment, replica: int = 0) -> np.ndarray:
        """The matching window of the nearest unlike neighbor.

        Parameters
        ----------
        segment : Segment
            The segment being neutralized.
        replica : int, default=0
            Which of the independent substitutions is wanted.

        Returns
        -------
        ndarray
            One value per time point of the segment.
        """
        channel = min(segment.channel, self.template_.shape[0] - 1)
        return self.template_[channel, segment.slice()]

    def describe(self) -> str:
        return f"the matching window of the nearest unlike neighbor at distance {self.distance_:.3f}"


def _as_positions(labels: np.ndarray, context: PerturbationContext) -> np.ndarray:
    """Map training labels onto probability column positions."""
    classes = getattr(context.wrapper, "classes", None)
    if classes is not None:
        lookup = {str(value): position for position, value in enumerate(classes)}
        mapped = [lookup.get(str(value)) for value in labels]
        if all(item is not None for item in mapped):
            return np.array(mapped, dtype=int)
    try:
        return labels.astype(int)
    except (TypeError, ValueError):
        _, codes = np.unique(labels, return_inverse=True)
        return codes


# ------------------------------------------------------------------ registry
PERTURBATIONS = {
    "zero": lambda **kw: ConstantPerturbation("zero", **kw),
    "mean": lambda **kw: ConstantPerturbation("mean", **kw),
    "median": lambda **kw: ConstantPerturbation("median", **kw),
    "min": lambda **kw: ConstantPerturbation("min", **kw),
    "max": lambda **kw: ConstantPerturbation("max", **kw),
    "random": RandomConstantPerturbation,
    "noise": NoisePerturbation,
    "segment_mean": SegmentMeanPerturbation,
    "linear": LinearPerturbation,
    "background": BackgroundPerturbation,
    "line_search": LineSearchPerturbation,
    "admissible_line_search": lambda **kw: LineSearchPerturbation(admissible=True, **kw),
    "opposite_mean": OppositeClassMeanPerturbation,
    "nearest_unlike": NearestUnlikeNeighborPerturbation,
}

CLASS_AGNOSTIC = (
    "zero",
    "mean",
    "median",
    "min",
    "max",
    "random",
    "noise",
    "segment_mean",
    "linear",
    "background",
)

CLASS_DIRECTED = (
    "line_search",
    "admissible_line_search",
    "opposite_mean",
    "nearest_unlike",
)

# candidates explored under perturbation="auto", all of them cheap and all of
# them class agnostic, so the reading of the ranking never changes silently
DEFAULT_CANDIDATES = ("mean", "min", "max", "linear", "segment_mean")

# aliases kept so that the settings of the paper can be named as published
_ALIASES = {
    "constant_min": "min",
    "constant_max": "max",
    "constant_mean": "mean",
    "local_search": "line_search",
    "nun": "nearest_unlike",
}


def make_perturbation(spec, **kwargs) -> Perturbation:
    """Resolve the ``marginalization`` argument of the explainer.

    Parameters
    ----------
    spec : Perturbation, str or float
        A ready operator, which is returned untouched, the name of a registered
        one, listed in ``PERTURBATIONS``, or a number, which becomes a constant
        substitution at that value.
    **kwargs : dict
        Arguments of the operator, such as ``n_replicas`` and ``taper``.

    Returns
    -------
    Perturbation
        The operator, not yet fitted on any instance.
    """
    if isinstance(spec, Perturbation):
        return spec
    if isinstance(spec, (int, float, np.floating, np.integer)):
        return ConstantPerturbation(float(spec), **kwargs)
    key = _ALIASES.get(str(spec).lower(), str(spec).lower())
    if key not in PERTURBATIONS:
        raise ValueError(
            f"Unknown perturbation '{spec}'. Registered operators are "
            f"{sorted(PERTURBATIONS)}, and a number is read as a constant."
        )
    return PERTURBATIONS[key](**kwargs)
