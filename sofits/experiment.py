"""Experimentation utilities for SOFI.

Nothing in this module is needed to explain a decision. The explainer and the
explanation cover that case on their own, one instance at a time, which is how
the package is used in practice. What is gathered here serves the other
situation, namely a study that compares many runs against one another, and it is
kept apart so that the ordinary interface stays small.

Three concerns live here. The selection of the instances a study runs on, which
in a real application is settled by the expert rather than by a rule. The
comparisons themselves, over batches of instances, segmentation procedures,
marginalization operators and explainers. The grid figures, which place one
panel per run so that the outcomes can be read side by side.

:class:`Experiment` gathers all of it behind one object bound to an explainer.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .baselines import compare_rankings as _compare_rankings
from .explanation import aggregate_explanations
from .metrics import robustness_score
from .model import ClassifierWrapper
from .perturbation import make_perturbation
from .plotting import (
    FONT_SCALE,
    THEME,
    WSPACE,
    _add_heading,
    _save,
    _theme_context,
    plot_degradation_curves,
    plot_marginalized_series,
    plot_segmentation,
)

__all__ = [
    "Experiment",
    "select_reliable_instances",
    "plot_segmentation_grid",
    "plot_explanation_grid",
    "plot_degradation_grid",
]


# ------------------------------------------------------------ instance choice
def select_reliable_instances(
    model,
    X,
    y,
    output_fn=None,
    output: str = "auto",
    channels_first="auto",
    classes: Optional[Sequence] = None,
    return_mask: bool = False,
):
    """Keep the instances a classifier gets right, for experiments alone.

    This helper exists for studies that must run over many instances and need a
    defensible way of choosing them. In a real application the instance to
    explain is the one the expert brings, and no filter is applied to it. The
    premise of the method is that the response deteriorates as segments are
    marginalized, which only holds for instances the model handles well, so a
    study that pools results over a whole test set states its criterion and this
    function covers the usual one.

    Parameters
    ----------
    model : object
        A fitted classifier or an already built :class:`ClassifierWrapper`.
    X : array
        Instances shaped ``(n, n_timestamps)`` or ``(n, n_channels,
        n_timestamps)``.
    y : array
        Their labels, compared against the prediction after both have been
        mapped onto the positions of the probability columns.
    output_fn : str or callable, optional
        Method of the model that produces the outputs, as in
        :class:`ClassifierWrapper`. Ignored when a wrapper is passed.
    output : {"auto", "proba", "logits"}, default="auto"
        Whether the model already returns probabilities. Ignored when a wrapper
        is passed.
    channels_first : {"auto", True, False}, default="auto"
        Whether the model expects the channel axis before the time axis. Ignored
        when a wrapper is passed.
    classes : sequence, optional
        Class labels in the order of the probability columns, used to compare
        the predictions with the labels given.
    return_mask : bool, default=False
        Whether the boolean mask is returned as well, which is how the matching
        targets are selected.

    Returns
    -------
    ndarray
        The retained instances, and the mask when it is requested.
    """
    data = np.asarray(X, dtype=float)
    if data.ndim == 2:
        data = data[:, None, :]
    elif data.ndim != 3:
        raise ValueError(
            "A dataset must be shaped (n_instances, n_timestamps) or "
            f"(n_instances, n_channels, n_timestamps), and the input has shape "
            f"{np.asarray(X).shape}."
        )
    if len(data) == 0:
        raise ValueError("A dataset cannot be empty.")

    labels = np.asarray(y)
    wrapper = (
        model
        if isinstance(model, ClassifierWrapper)
        else ClassifierWrapper(
            model,
            output_fn=output_fn,
            output=output,
            channels_first=channels_first,
            classes=classes,
        )
    )
    if wrapper.n_classes is None:
        wrapper.calibrate(data[: min(4, len(data))])

    predicted = np.argmax(wrapper.predict_proba(data), axis=1)
    names = [str(name) for name in (wrapper.classes or range(wrapper.n_classes))]
    lookup = {name: position for position, name in enumerate(names)}
    truth = np.array([lookup.get(str(value), -1) for value in labels], dtype=int)
    if np.any(truth < 0):
        # labels that do not match the declared classes are compared as integers
        truth = labels.astype(int)

    mask = predicted == truth
    return (data[mask], mask) if return_mask else data[mask]


# ------------------------------------------------------------- grid internals
def _labeled(items):
    """Read a mapping, a sequence of pairs or a sequence of explanations."""
    if isinstance(items, dict):
        pairs = list(items.items())
    else:
        pairs = []
        for item in items:
            if isinstance(item, tuple):
                pairs.append((item[0], item[1]))
            else:
                pairs.append((getattr(item, "class_name", None), item))
    if not pairs:
        raise ValueError("At least one entry is required.")
    return pairs


def _grid(n_items, ncols, figsize, panel_size, share):
    import matplotlib.pyplot as plt

    ncols = max(1, int(ncols))
    nrows = int(np.ceil(n_items / ncols))
    if figsize is None:
        figsize = (panel_size[0] * ncols, panel_size[1] * nrows)
    figure, axes = plt.subplots(
        nrows, ncols, figsize=figsize, layout="constrained", sharex=share, sharey=share
    )
    return figure, np.atleast_1d(axes).ravel()


# ----------------------------------------------------------------- grid plots
def plot_segmentation_grid(
    instance,
    segmentations,
    ncols: int = 3,
    figsize=None,
    panel_size=(4.6, 2.9),
    channel: int = 0,
    title: str = "Segmentation procedures",
    style: str = "dots",
    wspace: float = WSPACE,
    theme: Optional[str] = THEME,
    font_scale: Optional[float] = FONT_SCALE,
    save_path=None,
    dpi: int = 300,
    **kwargs,
):
    """Draw one instance under several segmentation procedures.

    Every panel holds the same series with the breakpoints of one procedure and
    no segment is singled out, so the vocabularies the explainer would rank are
    compared before any ranking exists.

    Parameters
    ----------
    instance : array
        The series every panel shows.
    segmentations : dict or sequence
        Mapping from a heading onto a ``Segmentation``, or a sequence of them, in
        which case the method name becomes the heading.
    ncols : int, default=3
        Number of panels per row.
    figsize : tuple, optional
        Size of the whole figure. It is derived from ``panel_size`` when omitted.
    panel_size : tuple, default=(4.6, 2.9)
        Size of one panel, from which the size of the figure is derived.
    channel : int, default=0
        Channel drawn in every panel, for a multivariate series.
    title : str, default="Segmentation procedures"
        Heading placed above the grid.
    style : {"dots", "lines"}, default="dots"
        How breakpoints are marked in every panel.
    wspace : float, default=0.05
        Horizontal space between the panels, as a fraction of the width of one
        of them, which keeps a panel clear of the axis label of its neighbor.
    theme : str, optional
        Name of the seaborn style applied to this figure alone.
    font_scale : float, optional
        Multiplier applied to every font of this figure alone.
    save_path : str or Path, optional
        Destination file. A path without an extension receives ``.pdf``.
    dpi : int, default=300
        Resolution used when the figure is written to a raster format.
    **kwargs : dict
        Further arguments of :func:`~sofits.plotting.plot_segmentation`, such as
        ``marker_color``, ``color``, ``linewidth`` or ``markersize``.

    Returns
    -------
    matplotlib Figure
        The figure holding the panels.
    """
    items = (
        list(segmentations.items())
        if isinstance(segmentations, dict)
        else [(segmentation.method, segmentation) for segmentation in segmentations]
    )
    if not items:
        raise ValueError("At least one segmentation is required.")

    with _theme_context(theme, font_scale):
        figure, flat = _grid(len(items), ncols, figsize, panel_size, share=True)
        for (heading, segmentation), ax in zip(items, flat):
            plot_segmentation(
                instance,
                segmentation,
                channel=channel,
                ax=ax,
                style=style,
                title=f"{heading} with {segmentation.n_segments} segments",
                theme=None,
                font_scale=None,
                **kwargs,
            )
        for ax in flat[len(items):]:
            ax.axis("off")
        _add_heading(figure, title, wspace)
        _save(figure, save_path, dpi)
    return figure


def plot_explanation_grid(
    explanations,
    title: str = "SOFI explanations",
    ncols: int = 2,
    figsize=None,
    panel_size=(5.0, 3.2),
    channel: int = 0,
    wspace: float = WSPACE,
    theme: Optional[str] = THEME,
    font_scale: Optional[float] = FONT_SCALE,
    save_path=None,
    dpi: int = 300,
    **kwargs,
):
    """Draw the relevant segments of several explanations on one grid.

    Each panel is the left panel of :func:`~sofits.plotting.plot_explanation`,
    namely the series with the segments the ranking retains, so the grid answers
    which parts of the series each run held responsible for the decision.

    Parameters
    ----------
    explanations : dict or sequence
        Mapping from a heading onto a ``SOFIExplanation``, a sequence of pairs
        holding a heading and an explanation, or a sequence of explanations.
    title : str, default="SOFI explanations"
        Heading of the grid.
    ncols : int, default=2
        Number of panels per row.
    figsize : tuple, optional
        Size of the figure. It is derived from ``panel_size`` when omitted.
    panel_size : tuple, default=(5.0, 3.2)
        Size of one panel, used to derive the size of the figure.
    channel : int, default=0
        Channel drawn in every panel, for a multivariate series.
    wspace : float, default=0.05
        Horizontal space between the panels, as a fraction of the width of one
        of them, which keeps a panel clear of the axis label of its neighbor.
    theme : str, optional
        Name of the seaborn style applied to this figure alone.
    font_scale : float, optional
        Multiplier applied to every font of this figure alone.
    save_path : str or Path, optional
        Destination file. A path without an extension receives ``.pdf``.
    dpi : int, default=300
        Resolution used when the figure is written to a raster format.
    **kwargs : dict
        Further arguments of
        :func:`~sofits.plotting.plot_marginalized_series`, such as ``linewidth``,
        ``markersize``, ``marker_color``, ``color`` or ``shade_color``.

    Returns
    -------
    matplotlib Figure
        The figure holding the panels.
    """
    items = _labeled(explanations)

    with _theme_context(theme, font_scale):
        figure, flat = _grid(len(items), ncols, figsize, panel_size, share=False)
        for (heading, explanation), ax in zip(items, flat):
            depth = explanation.sparsity_point or 0
            plot_marginalized_series(
                explanation,
                ax=ax,
                shaded=[
                    explanation.segmentation[position]
                    for position in explanation.order[:depth]
                ],
                probability=explanation.baseline_probability,
                title=heading if heading is not None else "Relevant segments",
                channel=channel,
                theme=None,
                font_scale=None,
                **kwargs,
            )
        for ax in flat[len(items):]:
            ax.axis("off")
        _add_heading(figure, title, wspace)
        _save(figure, save_path, dpi)
    return figure


def plot_degradation_grid(
    explanations,
    title: str = "Degradation curves",
    ncols: int = 2,
    figsize=None,
    panel_size=(5.0, 3.7),
    wspace: float = WSPACE,
    theme: Optional[str] = THEME,
    font_scale: Optional[float] = FONT_SCALE,
    save_path=None,
    dpi: int = 300,
    **kwargs,
):
    """Draw the MoRF and LeRF curves of several explanations on one grid.

    Each panel is the right panel of
    :func:`~sofits.plotting.plot_explanation`, so the grid answers how fast each
    run degrades the response and how wide the area between the curves is.

    Parameters
    ----------
    explanations : dict or sequence
        Mapping from a heading onto a ``SOFIExplanation``, a sequence of pairs
        holding a heading and an explanation, or a sequence of explanations.
    title : str, default="Degradation curves"
        Heading of the grid.
    ncols : int, default=2
        Number of panels per row.
    figsize : tuple, optional
        Size of the figure. It is derived from ``panel_size`` when omitted.
    panel_size : tuple, default=(5.0, 3.7)
        Size of one panel, used to derive the size of the figure.
    wspace : float, default=0.05
        Horizontal space between the panels, as a fraction of the width of one
        of them, which keeps a panel clear of the axis label of its neighbor.
    theme : str, optional
        Name of the seaborn style applied to this figure alone.
    font_scale : float, optional
        Multiplier applied to every font of this figure alone.
    save_path : str or Path, optional
        Destination file. A path without an extension receives ``.pdf``.
    dpi : int, default=300
        Resolution used when the figure is written to a raster format.
    **kwargs : dict
        Further arguments of
        :func:`~sofits.plotting.plot_degradation_curves`, such as ``linewidth``,
        ``markersize``, ``morf_color``, ``lerf_color`` or ``annotate_ds``.

    Returns
    -------
    matplotlib Figure
        The figure holding the panels.
    """
    items = _labeled(explanations)

    with _theme_context(theme, font_scale):
        figure, flat = _grid(len(items), ncols, figsize, panel_size, share=False)
        for (heading, explanation), ax in zip(items, flat):
            plot_degradation_curves(
                explanation,
                ax=ax,
                title=heading if heading is not None else "Degradation curves",
                theme=None,
                font_scale=None,
                **kwargs,
            )
        for ax in flat[len(items):]:
            ax.axis("off")
        _add_heading(figure, title, wspace)
        _save(figure, save_path, dpi)
    return figure


# ------------------------------------------------------------------ the class
class Experiment:
    """Comparisons of SOFI runs over batches, and the figures that display them.

    An experiment is bound to one explainer, whose segmentation and operator
    define the protocol every comparison is run under. Each comparison returns a
    mapping from a heading onto an explanation, which the grid figures and
    :meth:`summarize` accept directly.

    Parameters
    ----------
    explainer : SOFIExplainer, optional
        The explainer the comparisons are run with. It is required by every
        method that produces explanations and may be left out when the object is
        used for its figures alone.
    random_state : int or Generator, optional
        Seed of the random orderings a comparison draws, so that a study is
        reproducible.

    Examples
    --------
    >>> study = Experiment(explainer, random_state=42)                # doctest: +SKIP
    >>> runs = study.compare_marginalizations(instance, ["mean", "noise"])
    >>> study.plot_degradation_grid(runs, ncols=2)
    >>> study.summarize(runs)
    """

    def __init__(self, explainer=None, random_state=None):
        self.explainer = explainer
        self.random_state = random_state

    # ------------------------------------------------------------- selection
    def select_reliable_instances(self, X, y, **kwargs):
        """Keep the instances the classifier of the explainer gets right.

        The criterion serves a study that must run over many instances. In a
        real application the instance to explain is the one the expert brings.

        Parameters
        ----------
        X, y : array
            The instances and their labels.
        **kwargs : dict
            Further arguments of :func:`select_reliable_instances`, such as
            ``return_mask``.

        Returns
        -------
        ndarray
            The retained instances, and the mask when it is requested.
        """
        return select_reliable_instances(self._explainer().wrapper, X, y, **kwargs)

    # ----------------------------------------------------------- comparisons
    def explain_batch(self, X, **kwargs):
        """Explain several instances, one search each.

        Parameters
        ----------
        X : array
            The instances, shaped ``(n, n_timestamps)`` or ``(n, n_channels,
            n_timestamps)``.
        **kwargs : dict
            Further arguments of ``SOFIExplainer.explain_batch``.

        Returns
        -------
        list of SOFIExplanation
            One explanation per instance.
        """
        return self._explainer().explain_batch(X, **kwargs)

    def compare_segmentations(self, instance, segmentations, **kwargs):
        """Explain one instance under several partitions of it.

        Parameters
        ----------
        instance : array
            The series every run explains.
        segmentations : dict
            Maps a heading onto a ``Segmentation``.
        **kwargs : dict
            Further arguments of ``SOFIExplainer.explain``, applied to every run
            so that they all meet the same protocol.

        Returns
        -------
        dict
            Maps each heading onto the explanation obtained under it.
        """
        explainer = self._explainer()
        return {
            name: explainer.explain(instance, segmentation=segmentation, **kwargs)
            for name, segmentation in dict(segmentations).items()
        }

    def compare_marginalizations(self, instance, operators, segmentation=None, **kwargs):
        """Explain one instance under several marginalization operators.

        Parameters
        ----------
        instance : array
            The series every run explains.
        operators : sequence or dict
            Operator names, or a mapping from a heading onto an operator. An
            operator is a name, a number, a ready ``Perturbation`` or a
            dictionary holding a name under ``"marginalization"`` together with
            the arguments of the operator, such as ``n_replicas`` or ``taper``.
        segmentation : Segmentation, optional
            Partition every run ranks. The one of the explainer is used when it
            is omitted, and passing it keeps the runs comparable.
        **kwargs : dict
            Further arguments of ``SOFIExplainer.explain``.

        Returns
        -------
        dict
            Maps each heading onto the explanation obtained under it.
        """
        explainer = self._explainer()
        items = (
            dict(operators).items()
            if isinstance(operators, dict)
            else [(str(name), name) for name in operators]
        )
        if segmentation is None:
            segmentation = explainer.segment(instance)

        results = {}
        for name, spec in items:
            if isinstance(spec, dict):
                settings = dict(spec)
                chosen = settings.pop("marginalization", None)
                if chosen is None:
                    chosen = settings.pop("name", None)
                if chosen is None:
                    raise ValueError(
                        f"The operator of '{name}' names no substitution. Give it "
                        "under the key 'marginalization'."
                    )
                operator = make_perturbation(chosen, **settings)
            else:
                operator = spec
            results[name] = explainer.explain(
                instance, segmentation=segmentation, marginalization=operator, **kwargs
            )
        return results

    def compare_explainers(self, instance, explainers, **kwargs):
        """Explain one instance with several explainers, each reading its model.

        Parameters
        ----------
        instance : array
            The series every explainer reads.
        explainers : dict
            Maps a heading onto a ``SOFIExplainer``.
        **kwargs : dict
            Further arguments of ``SOFIExplainer.explain``.

        Returns
        -------
        dict
            Maps each heading onto the explanation that explainer produced.
        """
        return {
            name: explainer.explain(instance, **kwargs)
            for name, explainer in dict(explainers).items()
        }

    def compare_rankings(self, instance, rankings, **kwargs):
        """Score rankings obtained elsewhere under the protocol of the explainer.

        Parameters
        ----------
        instance : array
            The series every ranking refers to.
        rankings : dict
            Maps a method name onto an order of segment positions.
        **kwargs : dict
            Further arguments of ``SOFIExplainer.score_ranking``, such as
            ``segmentation`` or ``marginalization``, applied to every ranking.

        Returns
        -------
        DataFrame
            One row per method, sorted from the most to the least faithful.
        """
        return _compare_rankings(self._explainer(), instance, rankings, **kwargs)

    # -------------------------------------------------------------- reporting
    @staticmethod
    def summarize(explanations, top_k: int = 3):
        """Gather the outcome of several runs in one table.

        Parameters
        ----------
        explanations : dict or sequence
            The runs to tabulate, as accepted by the grid figures.
        top_k : int, default=3
            Number of leading segments listed for each run.

        Returns
        -------
        DataFrame
            One row per run with its degradation score, its sparsity rate, the
            cost of the search and the leading segments of its ranking.
        """
        import pandas as pd

        rows = []
        for name, explanation in _labeled(explanations):
            rows.append(
                {
                    "run": name,
                    "predicted": explanation.class_name,
                    "confidence": explanation.baseline_probability,
                    "ds": explanation.ds,
                    "sparsity_rate": explanation.sparsity_rate,
                    "n_segments": explanation.n_segments,
                    "model_queries": explanation.n_model_calls,
                    "runtime": explanation.runtime,
                    f"top_{top_k}": ", ".join(explanation.top(top_k)),
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def aggregate(explanations):
        """Pool explanations built on a partition shared by every instance.

        Parameters
        ----------
        explanations : sequence of SOFIExplanation
            Explanations to pool, all of them on the same partition.

        Returns
        -------
        DataFrame
            One row per segment with its mean rank, the deviation of that rank
            and how often it reached the first three positions.
        """
        return aggregate_explanations(list(explanations))

    @staticmethod
    def robustness(explanations, X, p: float = 0.8) -> float:
        """Agreement between the ranking of an instance and that of its neighbor.

        Parameters
        ----------
        explanations : sequence of SOFIExplanation
            One explanation per instance, aligned with the rows of ``X``.
        X : array
            The instances the explanations belong to.
        p : float, default=0.8
            Persistence of the rank biased overlap, which decides how much
            weight the leading positions carry.

        Returns
        -------
        float
            Mean similarity over the instances, in the unit interval.
        """
        rankings = [explanation.ranking for explanation in explanations]
        return robustness_score(rankings, np.asarray(X, dtype=float), p=p)

    # ---------------------------------------------------------------- figures
    plot_segmentation_grid = staticmethod(plot_segmentation_grid)
    plot_explanation_grid = staticmethod(plot_explanation_grid)
    plot_degradation_grid = staticmethod(plot_degradation_grid)

    # -------------------------------------------------------------- internals
    def _explainer(self):
        if self.explainer is None:
            raise ValueError(
                "This experiment carries no explainer, so no explanation can be "
                "produced. Build it as Experiment(explainer)."
            )
        return self.explainer

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Experiment(explainer={self.explainer!r})"
