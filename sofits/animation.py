"""Animation of a SOFI explanation.

The two curves rest on two different orderings, the LeRF one being the reverse
of the MoRF one, so they are built one after the other rather than together. The
LeRF pass runs first, marginalizing the least relevant segment of the ranking
and working towards the most relevant one. The panel is then cleared and the
MoRF pass runs in the opposite direction. Once both curves are complete, the
area between them is shaded, the degradation score is written, and the series is
redrawn carrying the segments that constitute the explanation, which is the
figure :func:`sofits.plotting.plot_explanation` produces.

While a pass runs, the panel marks the segment being marginalized at that step
and nothing else, in gray, so the eye follows one removal at a time and the
series is never buried under accumulated shading. The segments already dealt
with need no mark, since the flat stretch that replaced them identifies them.

Judging a segment belongs to the ranking rather than to either pass, and it is
settled only once both curves exist. The closing frame therefore marks the
segments the ranking places first in blue over the untouched series.

The animation is returned as an object that a notebook renders in the cell that
produced it, and that writes itself to a file when a destination is given. The
method :meth:`sofits.SOFIExplanation.plot_animation` is the ordinary way in.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

import matplotlib.pyplot as plt

from . import plotting

__all__ = ["Animation", "animate_marginalization"]


class Animation:
    """An animated GIF held in memory, which a notebook displays on its own.

    The object is returned by :func:`animate_marginalization`. A notebook cell
    that ends on it shows the animation, :meth:`save` writes it to disk, and
    ``bytes(animation)`` hands over the encoded data.

    Attributes
    ----------
    data : bytes
        The encoded GIF.
    path : str or None
        Where it was written, when a destination was given.
    n_frames : int
        Number of frames it holds.
    """

    def __init__(self, data: bytes, path: Optional[str] = None, n_frames: int = 0):
        self.data = data
        self.path = path
        self.n_frames = int(n_frames)

    def save(self, path) -> str:
        """Write the animation to disk.

        Parameters
        ----------
        path : str or Path
            Destination file. A path without an extension receives ``.gif``.

        Returns
        -------
        str
            The path the animation was written to.
        """
        target = Path(str(path)).expanduser()
        if target.suffix == "":
            target = target.with_suffix(".gif")
        if target.parent != Path(""):
            target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.data)
        self.path = str(target)
        return self.path

    def _repr_html_(self) -> str:
        encoded = base64.b64encode(self.data).decode("ascii")
        return (
            f'<img src="data:image/gif;base64,{encoded}" '
            'style="max-width:100%;" alt="SOFI marginalization" />'
        )

    def __bytes__(self) -> bytes:
        return self.data

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        where = "held in memory" if self.path is None else f"written to {self.path}"
        return f"Animation({self.n_frames} frames, {len(self.data) / 1024:.0f} KiB, {where})"


def _progress(total: int, enabled: bool, description: str):
    """A progress bar over the frames, or a silent stand in without tqdm."""

    class _Silent:
        def update(self, _n=1):
            pass

        def close(self):
            pass

    if not enabled:
        return _Silent()
    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover - tqdm is a dependency
        return _Silent()
    return tqdm(total=int(total), desc=description, leave=False)


def _pass_states(objective, order):
    """Series and probabilities along one marginalization order."""
    scores, _ = objective.curve(order)
    return objective.marginalized_series(order), np.asarray(scores, dtype=float)


def _draw_curve_panel(
    ax,
    explanation,
    morf,
    lerf,
    n_morf,
    n_lerf,
    fill,
    show_ds,
    linewidth,
    markersize,
    morf_color,
    lerf_color,
):
    """One frame of the right panel, with each curve revealed to its own depth."""
    ax.cla()
    steps = np.arange(len(morf))

    if fill:
        ax.fill_between(steps, morf, lerf, color="gray", alpha=0.22, zorder=2)

    ax.plot(
        steps[:n_lerf],
        lerf[:n_lerf],
        marker="s",
        markersize=markersize,
        markeredgecolor="white",
        markeredgewidth=0.7,
        color=lerf_color,
        linewidth=linewidth,
        label="LeRF",
        zorder=3,
    )
    ax.plot(
        steps[:n_morf],
        morf[:n_morf],
        marker="s",
        markersize=markersize,
        markeredgecolor="white",
        markeredgewidth=0.7,
        color=morf_color,
        linewidth=linewidth,
        label="MoRF",
        zorder=4,
    )

    for point, curve, depth, color in (
        (explanation.sparsity_point, morf, n_morf, morf_color),
        (explanation.lerf_sparsity_point, lerf, n_lerf, lerf_color),
    ):
        if point is not None and point < depth:
            ax.plot(
                [int(point)],
                [curve[int(point)]],
                marker="*",
                markersize=2.4 * markersize,
                color=color,
                markeredgecolor="white",
                markeredgewidth=0.7,
                zorder=5,
            )

    ax.set_ylim(-0.05, 1.22)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlim(-0.4, len(morf) - 0.6)
    ax.set_xticks(steps[:: max(1, int(np.ceil(len(morf) / 11)))])
    ax.set_xlabel("Marginalized segments")
    ax.set_ylabel("Predicted probability")
    ax.set_title("MoRF and LeRF curves")

    plotting._place_legend(ax, ncol=2)

    if show_ds:
        ax.text(
            0.98,
            0.96,
            f"DS: {explanation.ds:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox=dict(
                facecolor="white",
                alpha=0.9,
                edgecolor="#001f3f",
                boxstyle="round,pad=0.35",
            ),
        )


def animate_marginalization(
    explanation,
    explainer=None,
    save_path=None,
    title: str = "Sparseness Optimized Feature Importance",
    channel: int = 0,
    figsize=(13.0, 5.4),
    dpi: int = 110,
    linewidth: float = 1.4,
    curve_markersize: float = 8.0,
    segment_markersize: float = 8.0,
    marker_color: str = plotting.BREAKPOINT_COLOR,
    series_color: str = plotting.SERIES_COLOR,
    original_color: str = plotting.ORIGINAL_COLOR,
    morf_color: str = plotting.MORF_COLOR,
    lerf_color: str = plotting.LERF_COLOR,
    shade_alpha: float = 0.30,
    step_ms: int = 700,
    start_hold_ms: int = 1000,
    phase_hold_ms: int = 1400,
    end_hold_ms: int = 2600,
    wspace: float = plotting.WSPACE,
    theme=plotting.THEME,
    font_scale=plotting.FONT_SCALE,
    progress: bool = True,
) -> Animation:
    """Render the two passes of a SOFI explanation as an animated GIF.

    The result is returned rather than only written, so a notebook cell that
    ends on the call displays the animation, and ``save_path`` writes the same
    data to disk.

    Parameters
    ----------
    explanation : SOFIExplanation
        Result returned by the explainer, which carries the states of the MoRF
        pass and both curves.
    explainer : SOFIExplainer, optional
        Explainer that produced the result. It is used to rebuild the states of
        the LeRF pass, which the explanation does not carry. Without it the
        animation shows the MoRF pass alone and the LeRF curve appears complete
        at the start of it.
    save_path : str or Path, optional
        Destination file. A path without an extension receives ``.gif``. Nothing
        is written when it is left out.
    title : str, default="Sparseness Optimized Feature Importance"
        Heading placed above both panels.
    channel : int, default=0
        Channel shown in the left panel, for a multivariate series.
    figsize : tuple, default=(13.0, 5.4)
        Size of the figure holding the two panels.
    dpi : int, default=110
        Resolution of every frame.
    linewidth : float, default=1.4
        Thickness of the series and of both curves.
    curve_markersize : float, default=8.0
        Size of the square markers on the curves. The stars that mark a change of
        class scale with it.
    segment_markersize : float, default=8.0
        Size of the breakpoint markers on the series.
    marker_color : str, default="black"
        Color of the breakpoint markers.
    series_color, original_color : str
        Color of the marginalized series and of the untouched one behind it.
    morf_color, lerf_color : str
        Color of each curve.
    shade_alpha : float, default=0.30
        Opacity of the shading that marks a segment, so that the series stays
        visible underneath it.
    step_ms : int, default=700
        Time each marginalization step is held on screen.
    start_hold_ms : int, default=1000
        Time the untouched series is held at the start of each pass.
    phase_hold_ms : int, default=1400
        Time the completed LeRF pass is held before the panel is cleared.
    end_hold_ms : int, default=2600
        Time the explanation is held, with the shaded area and the score.
    wspace : float, default=0.05
        Horizontal space between the two panels, as a fraction of the width of
        one of them, which keeps the left panel clear of the axis label of the
        right one.
    theme : str, optional
        Name of the seaborn style applied to this animation alone.
    font_scale : float, optional
        Multiplier applied to every font of this animation alone.
    progress : bool, default=True
        Whether a bar follows the rendering, which takes a moment on a long
        ranking.

    Returns
    -------
    Animation
        The encoded GIF, displayed by a notebook and written by ``save``.
    """
    if explanation.states is None:
        raise ValueError(
            "The explanation carries no marginalized states, so the left panel "
            "cannot be drawn."
        )

    morf = np.asarray(explanation.morf_scores, dtype=float)
    lerf = np.asarray(explanation.lerf_scores, dtype=float)
    morf_states = np.asarray(explanation.states, dtype=float)
    morf_order = list(explanation.order)
    lerf_order = morf_order[::-1]

    # the explanation carries the states of the MoRF pass alone, so the LeRF ones
    # are rebuilt when the explainer is at hand
    lerf_states = None
    if explainer is not None:
        objective, _ = explainer.build_objective(
            explanation.instance,
            segmentation=explanation.segmentation,
            label=explanation.label,
        )
        lerf_states, rebuilt = _pass_states(objective, lerf_order)
        if not np.allclose(rebuilt, lerf, atol=1e-6):
            raise ValueError(
                "The LeRF curve rebuilt from the explainer disagrees with the one "
                "carried by the explanation, so they describe different runs. "
                "Pass the explainer that produced this explanation."
            )

    span = float(np.ptp(morf_states[:, channel])) or 1.0
    limits = (
        float(np.min(morf_states[:, channel])) - 0.12 * span,
        float(np.max(morf_states[:, channel])) + 0.22 * span,
    )

    n_frames = len(morf) + 1 + (len(lerf) + 1 if lerf_states is not None else 0)
    bar = _progress(n_frames, progress, "Rendering frames")

    frames: List[Image.Image] = []
    durations: List[int] = []

    with plotting._theme_context(theme, font_scale):
        figure, (left, right) = plt.subplots(1, 2, figsize=figsize, layout="constrained")
        plotting._add_heading(figure, title, wspace)

        def capture(duration):
            # the legends are held a fixed distance below their axis labels,
            # which is measured on the rendered frame
            plotting._align_legends(figure)
            buffer = io.BytesIO()
            figure.savefig(buffer, format="png", dpi=dpi)
            buffer.seek(0)
            frames.append(Image.open(buffer).convert("RGB"))
            durations.append(duration)
            bar.update(1)

        def run_pass(order, states, scores, heading, n_morf, n_lerf, growing):
            """Draw one pass, revealing the curve named by ``growing`` step by step."""
            for step in range(len(scores)):
                # the segment marginalized at this step, marked for this frame alone
                shaded = [explanation.segmentation[order[step - 1]]] if step > 0 else []
                left.cla()
                plotting.plot_marginalized_series(
                    explanation, ax=left, series=states[step][channel], shaded=shaded,
                    shade_color=plotting.MARGINALIZED_COLOR, probability=scores[step],
                    title=heading, channel=channel, limits=limits, linewidth=linewidth,
                    markersize=segment_markersize, marker_color=marker_color,
                    color=series_color, original_color=original_color,
                    shade_alpha=shade_alpha, theme=None, font_scale=None,
                )
                _draw_curve_panel(
                    right, explanation, morf, lerf,
                    n_morf=step + 1 if growing == "morf" else n_morf,
                    n_lerf=step + 1 if growing == "lerf" else n_lerf,
                    fill=False, show_ds=False, linewidth=linewidth,
                    markersize=curve_markersize,
                    morf_color=morf_color, lerf_color=lerf_color,
                )
                capture(start_hold_ms if step == 0 else step_ms)

        try:
            if lerf_states is not None:
                run_pass(
                    lerf_order, lerf_states, lerf, "Least relevant segments first",
                    n_morf=0, n_lerf=0, growing="lerf",
                )
                capture(phase_hold_ms)

            run_pass(
                morf_order, morf_states, morf, "Most relevant segments first",
                n_morf=0, n_lerf=len(lerf) if lerf_states is not None else 0,
                growing="morf",
            )

            # the closing frame keeps the segments that constitute the explanation
            # and returns the untouched series behind them, with the score between
            # the curves
            depth = explanation.sparsity_point or 0
            left.cla()
            plotting.plot_marginalized_series(
                explanation, ax=left,
                shaded=[explanation.segmentation[p] for p in morf_order[:depth]],
                probability=explanation.baseline_probability, title="Relevant segments",
                channel=channel, limits=limits, linewidth=linewidth,
                markersize=segment_markersize, marker_color=marker_color,
                color=series_color, original_color=original_color,
                shade_alpha=shade_alpha, theme=None, font_scale=None,
            )
            _draw_curve_panel(
                right, explanation, morf, lerf, len(morf), len(lerf), fill=True,
                show_ds=True, linewidth=linewidth, markersize=curve_markersize,
                morf_color=morf_color, lerf_color=lerf_color,
            )
            capture(end_hold_ms)
        finally:
            bar.close()
            plt.close(figure)

    buffer = io.BytesIO()
    frames[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    animation = Animation(buffer.getvalue(), n_frames=len(frames))
    if save_path is not None:
        animation.save(save_path)
    return animation
