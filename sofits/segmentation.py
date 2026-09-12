"""Segments of a time series, the units SOFI ranks.

A segment is a contiguous run of time points inside one channel. The
:class:`Segmentation` object holds the breakpoints of every channel, the
integer mask that assigns each time point to a segment, and the metadata used
by the report and the figures.

Four ways of obtaining a segmentation are supported. Expert breakpoints are
declared directly, uniform binning splits the series into equal parts, change
point detection delegates to ``ruptures``, and an explicit mask covers
groupings that are neither contiguous nor uniform. The change point methods are
the five compared in the paper, namely PELT, dynamic programming, binary
segmentation, bottom-up segmentation and sliding windows.

Segmentation always applies to one instance. A multivariate series receives one
set of breakpoints per channel unless ``shared`` asks for a single set derived
from every channel at once, which suits synchronized signals.
"""

from __future__ import annotations

import warnings

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

__all__ = [
    "Segment",
    "Segmentation",
    "manual_segmentation",
    "uniform_segmentation",
    "change_point_segmentation",
    "segmentation_from_mask",
    "build_segmentation",
    "CHANGE_POINT_METHODS",
]

CHANGE_POINT_METHODS = ("pelt", "dynp", "binseg", "bottomup", "window")

_ALIASES = {
    "dynamic": "dynp",
    "dynamic_programming": "dynp",
    "binary": "binseg",
    "bottom_up": "bottomup",
    "sliding_window": "window",
}


@dataclass(frozen=True)
class Segment:
    """One contiguous run of time points inside a single channel.

    Attributes
    ----------
    index : int
        Position of the segment inside its segmentation, which is also the
        position a ranking permutes and the number in its label.
    channel : int
        Channel the segment belongs to, always zero for a univariate series.
    start : int
        First time point of the segment, included.
    end : int
        Time point that closes the segment, excluded, following the convention
        of a Python slice.
    """

    index: int
    channel: int
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start

    @property
    def label(self) -> str:
        return f"s{self.index}"

    def slice(self) -> slice:
        return slice(self.start, self.end)


class Segmentation:
    """Partition of one instance into the units SOFI ranks.

    Parameters
    ----------
    segments : sequence of Segment
        Segments in the order used internally, which is also the order of the
        positions a ranking permutes.
    n_channels, n_timestamps : int
        Shape of the instance the segmentation belongs to.
    method : str
        Name of the procedure that produced the breakpoints, kept for the
        report and the figures.
    params : dict, optional
        Arguments handed to that procedure.
    """

    def __init__(
        self,
        segments: Sequence[Segment],
        n_channels: int,
        n_timestamps: int,
        method: str = "manual",
        params: Optional[Dict] = None,
    ):
        if len(segments) == 0:
            raise ValueError("A segmentation must hold at least one segment.")
        self.segments = [
            Segment(position, s.channel, s.start, s.end)
            for position, s in enumerate(segments)
        ]
        self.n_channels = int(n_channels)
        self.n_timestamps = int(n_timestamps)
        self.method = str(method)
        self.params = dict(params or {})

        mask = np.full((self.n_channels, self.n_timestamps), -1, dtype=int)
        for segment in self.segments:
            covered = mask[segment.channel, segment.slice()]
            if np.any(covered >= 0):
                raise ValueError(
                    "Two segments overlap on channel "
                    f"{segment.channel} around time point {segment.start}, which "
                    "makes the perturbation order ambiguous."
                )
            mask[segment.channel, segment.slice()] = segment.index
        self.mask = mask

    # ------------------------------------------------------------- accessors
    def __len__(self) -> int:
        return len(self.segments)

    def __iter__(self):
        return iter(self.segments)

    def __getitem__(self, position: int) -> Segment:
        return self.segments[int(position)]

    @property
    def n_segments(self) -> int:
        return len(self.segments)

    @property
    def labels(self) -> List[str]:
        return [segment.label for segment in self.segments]

    @property
    def lengths(self) -> np.ndarray:
        return np.array([segment.length for segment in self.segments], dtype=int)

    @property
    def coverage(self) -> float:
        """Fraction of the time points that belong to some segment."""
        return float(np.mean(self.mask >= 0))

    def breakpoints(self, channel: int = 0) -> List[int]:
        """End positions of the segments of one channel, in time order.

        Parameters
        ----------
        channel : int, default=0
            Channel whose breakpoints are returned.

        Returns
        -------
        list of int
            The exclusive end of every segment, the last of them being the
            length of the series.
        """
        ends = sorted(s.end for s in self.segments if s.channel == channel)
        return [int(end) for end in ends]

    def breakpoints_by_channel(self) -> Dict[int, List[int]]:
        return {
            channel: self.breakpoints(channel) for channel in range(self.n_channels)
        }

    def describe(self):
        """One row per segment, with its channel, span and length."""
        import pandas as pd

        return pd.DataFrame(
            {
                "segment": self.labels,
                "channel": [s.channel for s in self.segments],
                "start": [s.start for s in self.segments],
                "end": [s.end for s in self.segments],
                "length": [s.length for s in self.segments],
            }
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Segmentation(method={self.method}, n_segments={self.n_segments}, "
            f"n_channels={self.n_channels}, n_timestamps={self.n_timestamps})"
        )


# ---------------------------------------------------------------- builders
def _as_instance(x) -> np.ndarray:
    """Return one instance shaped (n_channels, n_timestamps)."""
    array = np.asarray(x, dtype=float)
    if array.ndim == 1:
        return array.reshape(1, -1)
    if array.ndim == 2:
        return array
    if array.ndim == 3 and array.shape[0] == 1:
        return array[0]
    raise ValueError(
        "An instance must be shaped (n_timestamps,) or (n_channels, n_timestamps), "
        f"and the array received has shape {array.shape}."
    )


def _segments_from_ends(ends: Sequence[int], channel: int, n_timestamps: int, offset: int):
    """Turn end positions into segments, closing the series when needed."""
    stray = [int(end) for end in ends if not 0 < int(end) <= n_timestamps]
    if stray:
        raise ValueError(
            f"The breakpoints {stray} fall outside a series of {n_timestamps} "
            "time points. A breakpoint is the exclusive end of a segment, so it "
            "must lie between one and the length of the series."
        )
    ordered = sorted({int(end) for end in ends})
    if not ordered or ordered[-1] != n_timestamps:
        ordered.append(n_timestamps)
    segments = []
    start = 0
    for end in ordered:
        if end > start:
            segments.append(Segment(offset + len(segments), channel, start, end))
            start = end
    return segments


def manual_segmentation(
    breakpoints: Union[Sequence[int], Dict[int, Sequence[int]]],
    n_timestamps: int,
    n_channels: int = 1,
) -> Segmentation:
    """Segmentation from breakpoints supplied by a domain expert.

    Parameters
    ----------
    breakpoints : sequence of int or dict
        End positions of the segments, exclusive, in the convention of
        ``ruptures``. A sequence applies to every channel, and a dict maps a
        channel onto its own positions. The closing position may be omitted.
    n_timestamps, n_channels : int
        Shape of the instances the segmentation will be applied to.

    Examples
    --------
    >>> manual_segmentation([40, 80], n_timestamps=128).n_segments
    3
    """
    if not isinstance(breakpoints, dict):
        breakpoints = {channel: breakpoints for channel in range(n_channels)}

    segments: List[Segment] = []
    for channel in range(n_channels):
        ends = breakpoints.get(channel, [n_timestamps])
        segments.extend(_segments_from_ends(ends, channel, n_timestamps, len(segments)))
    return Segmentation(
        segments,
        n_channels,
        n_timestamps,
        method="manual",
        params={"breakpoints": breakpoints},
    )


def uniform_segmentation(
    n_segments: int, n_timestamps: int, n_channels: int = 1
) -> Segmentation:
    """Split the series into parts of nearly equal length.

    The remainder is spread over the leading segments, so the lengths differ by
    at most one time point. The partition ignores the structure of the series
    entirely, which makes it the weakest option for a single instance and the
    natural one when several instances must share a vocabulary of segments.

    Parameters
    ----------
    n_segments : int
        Number of parts, which cannot exceed the length of the series.
    n_timestamps : int
        Length of the series the partition applies to.
    n_channels : int, default=1
        Number of channels. Every channel receives the same cuts, so the segment
        count of the partition is ``n_segments * n_channels``.

    Returns
    -------
    Segmentation
        The partition.
    """
    n_segments = int(n_segments)
    if n_segments < 1:
        raise ValueError("n_segments must be a positive integer.")
    if n_segments > n_timestamps:
        raise ValueError(
            f"{n_segments} segments cannot fit in a series of {n_timestamps} points."
        )
    edges = np.linspace(0, n_timestamps, n_segments + 1).round().astype(int)
    segments: List[Segment] = []
    for channel in range(n_channels):
        for position in range(n_segments):
            segments.append(
                Segment(len(segments), channel, int(edges[position]), int(edges[position + 1]))
            )
    return Segmentation(
        segments,
        n_channels,
        n_timestamps,
        method="uniform",
        params={"n_segments": n_segments},
    )


def segmentation_from_mask(mask, n_timestamps: Optional[int] = None) -> Segmentation:
    """Segmentation described by an integer mask.

    Every distinct non-negative value becomes one segment. A value that appears
    in several disjoint runs raises an error, since SOFI perturbs contiguous
    spans. Negative entries stay outside every segment and are never perturbed.

    Parameters
    ----------
    mask : array
        Integer array shaped ``(n_timestamps,)`` or ``(n_channels,
        n_timestamps)``, holding one value per time point.
    n_timestamps : int, optional
        Expected length of the series, checked against the width of the mask
        when it is given.

    Returns
    -------
    Segmentation
        The partition the mask describes.
    """
    array = np.asarray(mask)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    n_channels, length = array.shape
    if n_timestamps is not None and length != n_timestamps:
        raise ValueError("The mask width does not match the length of the series.")

    segments: List[Segment] = []
    for channel in range(n_channels):
        row = array[channel]
        for value in sorted(v for v in np.unique(row) if v >= 0):
            positions = np.flatnonzero(row == value)
            if positions.size == 0:
                continue
            if positions[-1] - positions[0] + 1 != positions.size:
                raise ValueError(
                    f"Mask value {value} on channel {channel} covers a set of time "
                    "points that is not contiguous."
                )
            segments.append(
                Segment(len(segments), channel, int(positions[0]), int(positions[-1]) + 1)
            )
    return Segmentation(segments, n_channels, length, method="mask")


def change_point_segmentation(
    x,
    method: str = "pelt",
    n_segments: Optional[int] = None,
    penalty: Optional[float] = 3.0,
    cost_model: str = "l2",
    min_size: int = 2,
    jump: int = 1,
    width: int = 40,
    shared: bool = False,
) -> Segmentation:
    """Segmentation obtained with a change point detection algorithm.

    Parameters
    ----------
    x : array
        One instance, shaped ``(n_timestamps,)`` or ``(n_channels, n_timestamps)``.
    method : {"pelt", "dynp", "binseg", "bottomup", "window"}, default="pelt"
        Detection procedure. The paper found PELT to give the most informative
        segments on the benchmark data, so it is the default.
    n_segments : int, optional
        Requested number of segments. Every method except PELT needs it, and
        PELT ignores it in favor of ``penalty``. The value is translated into
        ``n_segments - 1`` breakpoints.
    penalty : float, optional
        Penalty of the PELT objective. Larger values yield fewer segments.
    cost_model : str, default="l2"
        Cost function handed to ``ruptures``, such as ``l1``, ``l2`` or ``rbf``.
    min_size : int, default=2
        Shortest admissible segment.
    jump : int, default=1
        Grid on which candidate breakpoints are considered.
    width : int, default=40
        Width of the sliding window, used by the ``window`` method alone.
    shared : bool, default=False
        Whether one set of breakpoints is derived from every channel at once.
        The default detects change points channel by channel, which suits
        signals that are not synchronized.

    Notes
    -----
    ``ruptures`` is an optional dependency, installed through the ``segment``
    extra. The error raised when it is absent names the command that installs it.
    """
    try:
        import ruptures as rpt
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError(
            "Change point segmentation needs the ruptures package. Install it "
            "with 'pip install sofits[segment]' or supply the breakpoints "
            "yourself through segmentation='manual'."
        ) from error

    key = _ALIASES.get(str(method).lower(), str(method).lower())
    if key == "pelt" and (penalty is None or float(penalty) < 0):
        raise ValueError(
            "PELT is driven by a non-negative penalty, since the penalty is the "
            f"cost charged for opening a segment, and {penalty} is not one. Pass "
            "a larger value for fewer segments and a smaller one for more."
        )
    if int(min_size) < 1:
        raise ValueError("min_size must be a positive integer.")
    if key not in CHANGE_POINT_METHODS:
        raise ValueError(
            f"Unknown segmentation method '{method}'. Choose one of "
            f"{CHANGE_POINT_METHODS}, or supply breakpoints manually."
        )

    instance = _as_instance(x)
    n_channels, n_timestamps = instance.shape

    def detect(signal: np.ndarray) -> List[int]:
        signal = np.asarray(signal, dtype=float)
        if signal.ndim == 1:
            signal = signal.reshape(-1, 1)
        else:
            signal = signal.T
        if key == "pelt":
            algorithm = rpt.Pelt(model=cost_model, min_size=min_size, jump=jump)
            return list(algorithm.fit(signal).predict(pen=float(penalty)))
        n_bkps = _required_breakpoints(n_segments, key)
        if key == "dynp":
            algorithm = rpt.Dynp(model=cost_model, min_size=min_size, jump=jump)
        elif key == "binseg":
            algorithm = rpt.Binseg(model=cost_model, min_size=min_size, jump=jump)
        elif key == "bottomup":
            algorithm = rpt.BottomUp(model=cost_model, min_size=min_size, jump=jump)
        else:
            algorithm = rpt.Window(width=int(width), model=cost_model, min_size=min_size, jump=jump)
        return list(algorithm.fit(signal).predict(n_bkps=n_bkps))

    params = {
        "method": key,
        "n_segments": n_segments,
        "penalty": penalty,
        "cost_model": cost_model,
        "min_size": min_size,
        "width": width,
        "shared": shared,
    }

    def detect_safely(signal):
        try:
            return detect(signal)
        except Exception as error:  # pragma: no cover - depends on ruptures
            raise ValueError(
                f"The '{key}' procedure failed on this series, {type(error).__name__}, "
                f"{error}. A shorter series, a larger min_size or a wide window "
                "can leave no admissible breakpoint."
            ) from error

    segments: List[Segment] = []
    if shared or n_channels == 1:
        ends = detect_safely(instance)
        for channel in range(n_channels):
            segments.extend(
                _segments_from_ends(ends, channel, n_timestamps, len(segments))
            )
    else:
        for channel in range(n_channels):
            ends = detect_safely(instance[channel])
            segments.extend(
                _segments_from_ends(ends, channel, n_timestamps, len(segments))
            )

    segmentation = Segmentation(
        segments, n_channels, n_timestamps, method=key, params=params
    )

    # sliding windows in particular cannot always place as many breakpoints as
    # were asked for, since a wide window leaves too little room on a short
    # series. Silently returning a coarser partition would misrepresent what the
    # explainer is about to rank, so the shortfall is announced
    if n_segments is not None and key != "pelt":
        obtained = len(segmentation.breakpoints(0))
        if obtained != int(n_segments):
            warnings.warn(
                f"The '{key}' method placed {obtained} segments where "
                f"{n_segments} were requested. A shorter series, a larger "
                "min_size or a wide window leaves no room for more.",
                stacklevel=2,
            )
    return segmentation


def _required_breakpoints(n_segments: Optional[int], method: str) -> int:
    if n_segments is None:
        raise ValueError(
            f"The '{method}' method needs the number of segments, since it cannot "
            "decide how many change points to place on its own. Pass n_segments, "
            "or use method='pelt', which is driven by a penalty instead."
        )
    n_segments = int(n_segments)
    if n_segments < 2:
        raise ValueError("n_segments must be at least two for change point detection.")
    return n_segments - 1


def build_segmentation(x, segmentation="pelt", **kwargs) -> Segmentation:
    """Resolve the ``segmentation`` argument of the explainer.

    Accepted values are a ready :class:`Segmentation`, the name of a change
    point method, ``"uniform"``, ``"manual"`` together with ``breakpoints``,
    ``"mask"`` together with ``mask``, or a sequence of breakpoints.
    """
    instance = _as_instance(x)
    n_channels, n_timestamps = instance.shape

    if isinstance(segmentation, Segmentation):
        if segmentation.n_timestamps != n_timestamps:
            raise ValueError(
                "The segmentation was built for a series of "
                f"{segmentation.n_timestamps} points and the instance has "
                f"{n_timestamps}."
            )
        return segmentation

    if not isinstance(segmentation, str):
        return manual_segmentation(list(segmentation), n_timestamps, n_channels)

    name = segmentation.lower()
    if name == "uniform":
        return uniform_segmentation(
            kwargs.get("n_segments", 5), n_timestamps, n_channels
        )
    if name == "manual":
        breakpoints = kwargs.get("breakpoints")
        if breakpoints is None:
            raise ValueError(
                "segmentation='manual' needs the breakpoints argument, which "
                "holds the end position of every segment."
            )
        return manual_segmentation(breakpoints, n_timestamps, n_channels)
    if name == "mask":
        mask = kwargs.get("mask")
        if mask is None:
            raise ValueError("segmentation='mask' needs the mask argument.")
        return segmentation_from_mask(mask, n_timestamps)

    accepted = {
        "method",
        "n_segments",
        "penalty",
        "cost_model",
        "min_size",
        "jump",
        "width",
        "shared",
    }
    options = {k: v for k, v in kwargs.items() if k in accepted}
    options["method"] = name
    return change_point_segmentation(instance, **options)
