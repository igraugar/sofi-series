"""Figures of a SOFI explanation.

Every figure is produced by a single call and is configured by the arguments of
that call alone. Nothing is stored between calls, so the appearance of a figure
follows from its own arguments and from the defaults declared in its signature.
Colors, line thickness, marker size, titles, axis labels, the theme, the font
scale, the size of the panels and the destination file are all arguments, and a
figure that departs from the defaults does so without affecting any other.

The routines below cover one explanation at a time, which is the ordinary use of
the package. Figures that place several runs side by side belong to
:mod:`sofits.experiment`, since they serve comparisons rather than the reading
of a single decision.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

__all__ = [
    "plot_segmentation",
    "plot_degradation_curves",
    "plot_marginalized_series",
    "plot_explanation",
    "plot_marginalization",
]

# Defaults of the arguments below. They are read when a function is defined, so
# assigning to them afterwards changes nothing and a figure is never governed by
# a setting left behind by an earlier call.
MORF_COLOR = "#03719c"
LERF_COLOR = "#1A1A1A"
SERIES_COLOR = "tab:blue"
RELEVANT_COLOR = "tab:blue"
MARGINALIZED_COLOR = "#9e9e9e"
ORIGINAL_COLOR = "#bdbdbd"
AREA_COLOR = "gray"
BREAKPOINT_COLOR = "black"
THEME = "whitegrid"
FONT_SCALE = 1.15

# Horizontal space between the panels of a figure that holds several of them, as
# a fraction of the width of one panel. It keeps a panel clear of the axis label
# of the one beside it.
WSPACE = 0.05

_PROBABILITY_LABEL = "Predicted probability"

# Vertical space, in inches, left between the heading of a figure and the titles
# of the panels beneath it.
_HEADING_PAD = 0.125

# Space left between the label of the horizontal axis and the box of the legend
# beneath it, in inches. An absolute distance rather than a fraction of the
# height of the axes, so that a short panel and a tall one separate the two by
# the same amount.
_LEGEND_GAP = 0.13


# ------------------------------------------------------------------ internals
@contextmanager
def _theme_context(theme, font_scale):
    """Apply a theme and a font scale to one figure and to nothing else."""
    import matplotlib as mpl

    overrides = {}
    if theme is not None or font_scale is not None:
        try:
            import seaborn as sns

            if theme is not None:
                overrides.update(sns.axes_style(str(theme)))
            if font_scale is not None:
                overrides.update(
                    sns.plotting_context("notebook", font_scale=float(font_scale))
                )
        except ImportError:  # pragma: no cover - seaborn is a dependency
            if font_scale is not None:
                size = 10.0 * float(font_scale)
                overrides.update(
                    {
                        "font.size": size,
                        "axes.titlesize": 1.2 * size,
                        "axes.labelsize": size,
                        "xtick.labelsize": 0.9 * size,
                        "ytick.labelsize": 0.9 * size,
                        "legend.fontsize": 0.9 * size,
                    }
                )
    with mpl.rc_context(overrides):
        yield


def _new_axes(ax, figsize):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=figsize, layout="constrained")
    return ax


def _place_legend(ax, ncol: int = 2, **kwargs):
    """Put the legend of an axes underneath it, in a single row."""
    return ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.26),
        ncol=int(ncol),
        frameon=False,
        borderaxespad=0.0,
        handlelength=2.0,
        columnspacing=2.0,
        **kwargs,
    )


def _align_legends(figure, gap: Optional[float] = None, passes: int = 5):
    """Hold every legend of a figure a fixed distance below its axis label.

    A legend anchored as a fraction of the height of its axes drifts towards the
    label of the horizontal axis as the panel grows shorter, which is what
    happens as soon as several panels share a figure. The distance is therefore
    measured on the rendered figure and converted back into the coordinates of
    each axes. Rendering moves the axes in turn, so the measurement is repeated
    until it settles.
    """
    gap = _LEGEND_GAP if gap is None else float(gap)
    for _ in range(max(1, int(passes))):
        figure.canvas.draw()
        moved = False
        for ax in figure.axes:
            legend = ax.get_legend()
            if legend is None:
                continue
            panel = ax.get_window_extent()
            label = ax.xaxis.get_label()
            bottom = (
                label.get_window_extent().y0
                if label.get_text()
                else ax.get_xticklabels()[0].get_window_extent().y0
                if ax.get_xticklabels()
                else panel.y0
            )
            target = bottom - gap * figure.dpi
            legend.set_bbox_to_anchor(
                (0.5, (target - panel.y0) / panel.height), transform=ax.transAxes
            )
            moved = True
        if not moved:
            break
    figure.canvas.draw()
    return figure


def _add_heading(figure, title: str, wspace: float = WSPACE):
    """Place the heading of a figure of several panels, space them and align
    their legends."""
    if not title:
        raise ValueError(
            "A figure of several panels needs a heading, since the panels are "
            "read together and a reader needs to know what they have in common."
        )
    figure.get_layout_engine().set(h_pad=_HEADING_PAD, wspace=float(wspace))
    figure.suptitle(title)
    return _align_legends(figure)


def _save(figure, save_path, dpi: int = 300, default_suffix: str = ".pdf"):
    """Write a figure to disk when a destination was given.

    A path without an extension receives ``default_suffix``, and the enclosing
    directory is created when it does not exist.
    """
    if save_path is None:
        return None
    path = Path(str(save_path)).expanduser()
    if path.suffix == "":
        path = path.with_suffix(default_suffix)
    if path.parent != Path(""):
        path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=int(dpi), bbox_inches="tight")
    return str(path)


def _limits(values, low_margin: float = 0.12, high_margin: float = 0.12):
    span = float(np.ptp(values)) or 1.0
    return float(np.min(values)) - low_margin * span, float(np.max(values)) + high_margin * span


def _annotate(ax, text: str):
    ax.text(
        0.98,
        0.96,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=dict(
            facecolor="white", alpha=0.9, edgecolor="#001f3f", boxstyle="round,pad=0.35"
        ),
    )


# --------------------------------------------------------------- segmentation
def plot_segmentation(
    instance,
    segmentation,
    channel: int = 0,
    ax=None,
    title: Optional[str] = None,
    style: str = "dots",
    figsize=(6.6, 3.4),
    linewidth: float = 1.8,
    markersize: float = 8.0,
    color: str = SERIES_COLOR,
    marker_color: str = BREAKPOINT_COLOR,
    line_color: str = "gray",
    xlabel: str = "Time",
    ylabel: str = "Value",
    theme: Optional[str] = THEME,
    font_scale: Optional[float] = FONT_SCALE,
    save_path=None,
    dpi: int = 300,
):
    """Draw one channel of a series together with its breakpoints.

    Parameters
    ----------
    instance : array
        The series, shaped ``(n_timestamps,)`` or ``(n_channels, n_timestamps)``.
    segmentation : Segmentation
        Partition whose breakpoints are drawn.
    channel : int, default=0
        Channel shown, for a multivariate series.
    ax : matplotlib Axes, optional
        Target axes. A new figure is created when omitted.
    title : str, optional
        Heading of the figure. A default one names the method and the count.
    style : {"dots", "lines"}, default="dots"
        How a breakpoint is marked. Markers placed on the curve itself stay
        legible at any granularity and keep the eye on the signal, whereas a
        dashed vertical rule separates the segments plainly and crowds a series
        divided into many parts.
    figsize : tuple, default=(6.6, 3.4)
        Size of the figure created when no axes is supplied.
    linewidth : float, default=1.8
        Thickness of the series.
    markersize : float, default=8.0
        Size of the breakpoint markers under ``style="dots"``.
    color : str, default="tab:blue"
        Color of the series.
    marker_color : str, default="black"
        Color of the breakpoint markers, dark so that they read against the
        series rather than compete with it.
    line_color : str, default="gray"
        Color of the vertical rules under ``style="lines"``.
    xlabel, ylabel : str
        Labels of the two axes.
    theme : str, optional
        Name of the seaborn style applied to this figure alone. ``None`` keeps
        the matplotlib settings untouched.
    font_scale : float, optional
        Multiplier applied to every font of this figure alone.
    save_path : str or Path, optional
        Destination file. A path without an extension receives ``.pdf``.
    dpi : int, default=300
        Resolution used when the figure is written to a raster format.

    Returns
    -------
    matplotlib Axes
        The axes the series was drawn on.
    """
    if style not in ("lines", "dots"):
        raise ValueError("style must be 'dots' or 'lines'.")

    instance = np.asarray(instance, dtype=float)
    if instance.ndim == 1:
        instance = instance.reshape(1, -1)
    series = instance[channel]

    with _theme_context(theme, font_scale):
        ax = _new_axes(ax, figsize)
        ax.plot(series, color=color, linewidth=linewidth, zorder=3)

        cuts = sorted(
            {segment.start for segment in segmentation if segment.channel == channel}
            | {segment.end for segment in segmentation if segment.channel == channel}
        )
        if style == "lines":
            for position in cuts:
                ax.axvline(
                    position,
                    color=line_color,
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.75,
                    zorder=2,
                )
        else:
            inner = [position for position in cuts if 0 < position < len(series)]
            ax.plot(
                inner,
                series[inner],
                linestyle="none",
                marker="o",
                markersize=markersize,
                markerfacecolor=marker_color,
                markeredgecolor="white",
                markeredgewidth=1.0,
                zorder=4,
            )

        ax.set_xlim(-0.02 * len(series), 1.02 * len(series))
        ax.set_ylim(*_limits(series))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(
            title
            if title is not None
            else f"{segmentation.method} with {segmentation.n_segments} segments"
        )
        _save(ax.get_figure(), save_path, dpi)
    return ax


# ---------------------------------------------------------- degradation curves
def plot_degradation_curves(
    explanation,
    ax=None,
    title: str = "Degradation curves",
    annotate_ds: bool = True,
    legend: bool = True,
    mark_sparsity: bool = True,
    figsize=(6.6, 4.2),
    linewidth: float = 1.8,
    markersize: float = 8.0,
    morf_color: str = MORF_COLOR,
    lerf_color: str = LERF_COLOR,
    area_color: str = AREA_COLOR,
    area_alpha: float = 0.22,
    xlabel: str = "Marginalized segments",
    ylabel: str = _PROBABILITY_LABEL,
    theme: Optional[str] = THEME,
    font_scale: Optional[float] = FONT_SCALE,
    save_path=None,
    dpi: int = 300,
):
    """Draw the MoRF and the LeRF curves together with their enclosed area.

    Both curves report the probability the model assigns to the class it
    predicted before any perturbation, with no rescaling, so the vertical axis is
    a probability and the value at the first step is the confidence of the model.
    The shaded region between them is the degradation score, which places the
    quantity maximized by the search in the figure itself. A star on each curve
    marks the step at which the predicted class first changes, and their
    separation quantifies how much earlier the ranking induces the change of
    class than its reverse.

    Parameters
    ----------
    explanation : SOFIExplanation
        Result returned by the explainer.
    ax : matplotlib Axes, optional
        Target axes. A new figure is created when omitted.
    title : str, default="Degradation curves"
        Heading of the figure.
    annotate_ds : bool, default=True
        Whether the degradation score is written inside the axes.
    legend : bool, default=True
        Whether the two curves are labeled, in a single row underneath the axes.
    mark_sparsity : bool, default=True
        Whether the class change of each curve receives a star.
    figsize : tuple, default=(6.6, 4.2)
        Size of the figure created when no axes is supplied.
    linewidth : float, default=1.8
        Thickness of both curves.
    markersize : float, default=8.0
        Size of the square marker placed on every step. The star that marks a
        change of class is drawn larger, so it stays distinguishable from an
        ordinary step.
    morf_color, lerf_color : str
        Color of each curve.
    area_color : str, default="gray"
        Color of the region between the curves.
    area_alpha : float, default=0.22
        Opacity of that region.
    xlabel, ylabel : str
        Labels of the two axes.
    theme : str, optional
        Name of the seaborn style applied to this figure alone.
    font_scale : float, optional
        Multiplier applied to every font of this figure alone.
    save_path : str or Path, optional
        Destination file. A path without an extension receives ``.pdf``.
    dpi : int, default=300
        Resolution used when the figure is written to a raster format.

    Returns
    -------
    matplotlib Axes
        The axes the curves were drawn on.
    """
    morf = np.asarray(explanation.morf_scores, dtype=float)
    lerf = np.asarray(explanation.lerf_scores, dtype=float)
    steps = np.arange(len(morf))
    owns_figure = ax is None

    with _theme_context(theme, font_scale):
        ax = _new_axes(ax, figsize)

        ax.fill_between(steps, morf, lerf, color=area_color, alpha=area_alpha, zorder=2)
        ax.plot(
            steps,
            lerf,
            marker="s",
            markersize=markersize,
            markeredgecolor="white",
            markeredgewidth=0.7,
            color=lerf_color,
            linewidth=linewidth,
            linestyle="-",
            label="LeRF",
            zorder=3,
        )
        ax.plot(
            steps,
            morf,
            marker="s",
            markersize=markersize,
            markeredgecolor="white",
            markeredgewidth=0.7,
            color=morf_color,
            linewidth=linewidth,
            label="MoRF",
            zorder=4,
        )

        if mark_sparsity:
            marks = (
                (explanation.sparsity_point, morf, morf_color),
                (explanation.lerf_sparsity_point, lerf, lerf_color),
            )
            for point, curve, curve_color in marks:
                if point is None:
                    continue
                ax.plot(
                    [int(point)],
                    [curve[int(point)]],
                    marker="*",
                    markersize=2.4 * markersize,
                    color=curve_color,
                    markeredgecolor="white",
                    markeredgewidth=0.8,
                    zorder=5,
                )

        # the reported curves live inside the unit interval, so the axis is
        # fixed, with a margin that leaves room for the annotation
        ax.set_ylim(-0.05, 1.22)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xlim(-0.4, len(morf) - 0.6)
        ax.set_xticks(steps[:: max(1, int(np.ceil(len(morf) / 11)))])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        if legend:
            _place_legend(ax, ncol=2)
        if annotate_ds:
            _annotate(ax, f"DS: {explanation.ds:.3f}")
        if owns_figure:
            _align_legends(ax.get_figure())
        _save(ax.get_figure(), save_path, dpi)
    return ax


def plot_marginalized_series(
    explanation,
    ax=None,
    series=None,
    shaded=(),
    shade_color: str = RELEVANT_COLOR,
    probability: Optional[float] = None,
    title: str = "Relevant segments",
    channel: int = 0,
    figsize=(6.6, 3.6),
    linewidth: float = 1.8,
    markersize: float = 8.0,
    marker_color: str = BREAKPOINT_COLOR,
    color: str = SERIES_COLOR,
    original_color: str = ORIGINAL_COLOR,
    shade_alpha: float = 0.30,
    limits=None,
    xlabel: str = "Time",
    ylabel: str = "Value",
    theme: Optional[str] = THEME,
    font_scale: Optional[float] = FONT_SCALE,
    save_path=None,
    dpi: int = 300,
):
    """Draw one state of the series with a set of segments marked.

    The left panel of :func:`plot_explanation` and every frame of the animation
    are drawn by this routine, so the two cannot drift apart. The untouched
    series is laid behind in gray, the state under discussion is drawn over it,
    the breakpoint markers follow that state, and the segments given are marked.

    Parameters
    ----------
    explanation : SOFIExplanation
        The run the segments belong to.
    ax : matplotlib Axes, optional
        Target axes. A new figure is created when it is omitted.
    series : array, optional
        The state to draw. The untouched series is used when it is omitted.
    shaded : sequence of Segment, default=()
        Segments marked on this panel.
    shade_color : str, default="tab:blue"
        Color of those marks.
    probability : float, optional
        Probability written in the corner. Nothing is written when it is left
        out.
    title : str, default="Relevant segments"
        Title of the panel.
    channel : int, default=0
        Channel drawn, for a multivariate series.
    figsize : tuple, default=(6.6, 3.6)
        Size of the figure created when no axes is supplied.
    linewidth : float, default=1.8
        Thickness of the series.
    markersize : float, default=8.0
        Size of the breakpoint markers.
    marker_color : str, default="black"
        Color of the breakpoint markers.
    color : str, default="tab:blue"
        Color of the state under discussion.
    original_color : str, default="#bdbdbd"
        Color of the untouched series drawn behind it.
    shade_alpha : float, default=0.30
        Opacity of the marks, so that the series stays visible underneath them.
    limits : tuple, optional
        Vertical range. It is derived from the series when it is left out, and
        passing it keeps a sequence of panels on one scale.
    xlabel, ylabel : str
        Labels of the two axes.
    theme : str, optional
        Name of the seaborn style applied to this figure alone.
    font_scale : float, optional
        Multiplier applied to every font of this figure alone.
    save_path : str or Path, optional
        Destination file. A path without an extension receives ``.pdf``.
    dpi : int, default=300
        Resolution used when the figure is written to a raster format.

    Returns
    -------
    matplotlib Axes
        The axes the panel was drawn on.
    """
    original = np.asarray(explanation.instance, dtype=float)[channel]
    values = original if series is None else np.asarray(series, dtype=float)
    if values.ndim > 1:
        values = values[channel]

    with _theme_context(theme, font_scale):
        ax = _new_axes(ax, figsize)
        for segment in shaded:
            ax.axvspan(
                segment.start,
                segment.end,
                facecolor=shade_color,
                alpha=shade_alpha,
                zorder=1,
            )
        ax.plot(original, color=original_color, linewidth=linewidth, zorder=2)
        ax.plot(values, color=color, linewidth=linewidth, zorder=3)

        cuts = np.array(
            [
                segment.start
                for segment in explanation.segmentation
                if segment.channel == channel and 0 < segment.start < values.size
            ],
            dtype=int,
        )
        if cuts.size:
            ax.plot(
                cuts,
                values[cuts],
                linestyle="none",
                marker="o",
                markersize=markersize,
                markerfacecolor=marker_color,
                markeredgecolor="white",
                markeredgewidth=0.8,
                zorder=4,
            )

        if limits is None:
            limits = _limits(original, 0.12, 0.22)
        ax.set_xlim(-0.02 * values.size, 1.02 * values.size)
        ax.set_ylim(*limits)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        if probability is not None:
            _annotate(ax, f"Probability: {probability:.3f}")
        _save(ax.get_figure(), save_path, dpi)
    return ax


# ----------------------------------------------------------------- explanation
_SERIES_KEYS = (
    "linewidth",
    "markersize",
    "marker_color",
    "color",
    "original_color",
    "shade_color",
    "shade_alpha",
)
_CURVE_KEYS = (
    "linewidth",
    "markersize",
    "morf_color",
    "lerf_color",
    "area_color",
    "area_alpha",
    "annotate_ds",
    "legend",
    "mark_sparsity",
)


def plot_explanation(
    explanation,
    figsize=(11.5, 4.2),
    title: str = "Sparseness Optimized Feature Importance",
    channel: int = 0,
    series_title: str = "Relevant segments",
    curve_title: str = "MoRF and LeRF curves",
    wspace: float = WSPACE,
    theme: Optional[str] = THEME,
    font_scale: Optional[float] = FONT_SCALE,
    save_path=None,
    dpi: int = 300,
    **kwargs,
):
    """Draw the explanation as two panels.

    The left panel carries the untouched series with the segments the ranking
    retains, namely the smallest set whose cumulative marginalization changes the
    predicted class. The right panel carries the two degradation curves from
    which that ranking was obtained.

    Parameters
    ----------
    explanation : SOFIExplanation
        The run to draw.
    figsize : tuple, default=(11.5, 4.2)
        Size of the figure holding the two panels.
    title : str, default="Sparseness Optimized Feature Importance"
        Heading placed above them.
    channel : int, default=0
        Channel drawn in the left panel, for a multivariate series.
    series_title, curve_title : str
        Titles of the two panels.
    wspace : float, default=0.05
        Horizontal space between the two panels, as a fraction of the width of
        one of them, which keeps the left panel clear of the axis label of the
        right one.
    theme : str, optional
        Name of the seaborn style applied to this figure alone.
    font_scale : float, optional
        Multiplier applied to every font of this figure alone.
    save_path : str or Path, optional
        Destination file. A path without an extension receives ``.pdf``.
    dpi : int, default=300
        Resolution used when the figure is written to a raster format.
    **kwargs : dict
        Settings of the two panels, namely ``linewidth``, ``markersize``,
        ``marker_color``, ``color``, ``original_color``, ``shade_color``,
        ``shade_alpha``, ``morf_color``, ``lerf_color``, ``area_color``,
        ``area_alpha``, ``annotate_ds``, ``legend`` and ``mark_sparsity``. Each
        is passed to the panel that accepts it.

    Returns
    -------
    matplotlib Figure
        The figure holding the two panels.
    """
    import matplotlib.pyplot as plt

    unknown = set(kwargs) - set(_SERIES_KEYS) - set(_CURVE_KEYS)
    if unknown:
        raise TypeError(f"unexpected settings {sorted(unknown)}")

    with _theme_context(theme, font_scale):
        figure, (left, right) = plt.subplots(1, 2, figsize=figsize, layout="constrained")
        depth = explanation.sparsity_point or 0
        plot_marginalized_series(
            explanation,
            ax=left,
            shaded=[explanation.segmentation[p] for p in explanation.order[:depth]],
            probability=explanation.baseline_probability,
            title=series_title,
            channel=channel,
            theme=None,
            font_scale=None,
            **{key: value for key, value in kwargs.items() if key in _SERIES_KEYS},
        )
        plot_degradation_curves(
            explanation,
            ax=right,
            title=curve_title,
            theme=None,
            font_scale=None,
            **{key: value for key, value in kwargs.items() if key in _CURVE_KEYS},
        )
        _add_heading(figure, title, wspace)
        _save(figure, save_path, dpi)
    return figure


# ------------------------------------------------ cumulative marginalization
def plot_marginalization(
    explanation,
    steps: Optional[Sequence[int]] = None,
    ncols: int = 3,
    figsize=None,
    panel_size=(4.4, 2.6),
    channel: int = 0,
    title: str = "Cumulative marginalization snapshots",
    wspace: float = WSPACE,
    linewidth: float = 1.8,
    color: str = SERIES_COLOR,
    original_color: str = "#b0b0b0",
    shade_color: str = "#d9d9d9",
    shade_alpha: float = 0.75,
    xlabel: str = "Time",
    ylabel: str = "Value",
    theme: Optional[str] = THEME,
    font_scale: Optional[float] = FONT_SCALE,
    save_path=None,
    dpi: int = 300,
):
    """Show the series along the cumulative marginalization of the ranking.

    Segments are never marginalized in isolation, so an explanation cannot be
    drawn by shading each of them with a score of its own. What the ranking
    prescribes is a sequence of states, each one obtained by marginalizing the
    next segment on top of everything already removed, and this figure shows that
    sequence. Every panel draws the original series in gray underneath the state
    reached at that step, shades the region marginalized so far, and reports the
    probability that survives it.

    Parameters
    ----------
    explanation : SOFIExplanation
        Result returned by the explainer, which always carries the states.
    steps : sequence of int, optional
        Steps drawn, counted from zero for the untouched series. Six steps spread
        over the whole marginalization are chosen when it is omitted, which fills
        a grid of two rows by three columns.
    ncols : int, default=3
        Number of panels per row.
    figsize : tuple, optional
        Size of the whole figure. It is derived from ``panel_size`` when omitted.
    panel_size : tuple, default=(4.4, 2.6)
        Size of one panel, from which the size of the figure is derived.
    channel : int, default=0
        Channel shown, for a multivariate series.
    title : str, default="Cumulative marginalization snapshots"
        Heading placed above the grid.
    wspace : float, default=0.05
        Horizontal space between the panels, as a fraction of the width of one
        of them, which keeps a panel clear of the axis label of its neighbor.
    linewidth : float, default=1.8
        Thickness of the series.
    color : str, default="tab:blue"
        Color of the marginalized series.
    original_color : str, default="#b0b0b0"
        Color of the untouched series drawn behind it.
    shade_color : str, default="#d9d9d9"
        Color of the region already marginalized.
    shade_alpha : float, default=0.75
        Opacity of that region.
    xlabel, ylabel : str
        Labels of the two axes.
    theme : str, optional
        Name of the seaborn style applied to this figure alone.
    font_scale : float, optional
        Multiplier applied to every font of this figure alone.
    save_path : str or Path, optional
        Destination file. A path without an extension receives ``.pdf``.
    dpi : int, default=300
        Resolution used when the figure is written to a raster format.

    Returns
    -------
    matplotlib Figure
        The figure holding the panels.
    """
    import matplotlib.pyplot as plt

    if explanation.states is None:
        raise ValueError(
            "The explanation carries no marginalized states, so the sequence "
            "cannot be drawn. Explanations built by SOFIExplainer carry them."
        )

    states = np.asarray(explanation.states, dtype=float)
    original = states[0]
    n_states = len(states)

    if steps is None:
        # six snapshots spread over the whole marginalization, which fills a grid
        # of two rows by three columns
        count = min(6, n_states)
        steps = sorted(
            set(np.linspace(0, n_states - 1, count).round().astype(int).tolist())
        )
    steps = [int(step) for step in steps]
    if any(step < 0 or step >= n_states for step in steps):
        raise ValueError(f"Steps must lie between 0 and {n_states - 1}.")

    ncols = max(1, int(ncols))
    nrows = int(np.ceil(len(steps) / ncols))
    if figsize is None:
        figsize = (panel_size[0] * ncols, panel_size[1] * nrows)

    values = original[channel]
    low, high = _limits(states[:, channel])

    with _theme_context(theme, font_scale):
        figure, axes = plt.subplots(
            nrows, ncols, figsize=figsize, layout="constrained", sharex=True, sharey=True
        )
        flat = np.atleast_1d(axes).ravel()

        for step, ax in zip(steps, flat):
            for position in explanation.order[:step]:
                segment = explanation.segmentation[position]
                if segment.channel != channel:
                    continue
                ax.axvspan(
                    segment.start,
                    segment.end,
                    facecolor=shade_color,
                    alpha=shade_alpha,
                    zorder=1,
                )
            ax.plot(values, color=original_color, linewidth=1.2, zorder=2)
            ax.plot(states[step][channel], color=color, linewidth=linewidth, zorder=3)

            probability = explanation.morf_scores[step]
            heading = (
                "Using all segments"
                if step == 0
                else f"After {step} segment{'s' if step > 1 else ''}"
            )
            ax.set_title(f"{heading}, probability={probability:.3f}")
            ax.set_xlim(0, values.size)
            ax.set_ylim(low, high)
        for ax in flat[len(steps):]:
            ax.axis("off")

        for ax in flat[-ncols:]:
            ax.set_xlabel(xlabel)
        for position in range(0, len(flat), ncols):
            flat[position].set_ylabel(ylabel)

        _add_heading(figure, title, wspace)
        _save(figure, save_path, dpi)
    return figure
