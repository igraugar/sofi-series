"""Sparseness Optimized Feature Importance for time series classification."""

from __future__ import annotations

import time
import warnings
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from .explanation import SOFIExplanation
from .model import ClassifierWrapper
from .objective import Objective, modularity_gap
from .perturbation import DEFAULT_CANDIDATES, PerturbationContext, make_perturbation
from .report import describe_instance, instance_report
from .search import greedy_ranking, hill_climbing
from .segmentation import Segmentation, build_segmentation

__all__ = ["SOFIExplainer"]


def _finite(array: np.ndarray, what: str) -> np.ndarray:
    """Refuse arrays holding values that are not finite.

    A missing or infinite value propagates silently through the perturbation
    curves and leaves a degradation score that looks ordinary while meaning
    nothing, so it is refused where it enters rather than diagnosed later.
    """
    if not np.all(np.isfinite(array)):
        raise ValueError(
            f"{what} holds values that are not finite. Impute or drop them "
            "before explaining, since a curve computed on them carries no "
            "information."
        )
    return array


def _as_instance(x) -> np.ndarray:
    array = np.asarray(x, dtype=float)
    if array.ndim == 1:
        instance = array.reshape(1, -1)
    elif array.ndim == 2:
        instance = array
    elif array.ndim == 3 and array.shape[0] == 1:
        instance = array[0]
    else:
        raise ValueError(
            "An instance must be shaped (n_timestamps,), (n_channels, "
            f"n_timestamps) or (1, n_channels, n_timestamps), and the input has "
            f"shape {array.shape}."
        )
    if instance.size == 0:
        raise ValueError("An instance cannot be empty.")
    return _finite(instance, "The instance")


def _as_dataset(X) -> np.ndarray:
    array = np.asarray(X, dtype=float)
    if array.ndim == 2:
        data = array[:, None, :]
    elif array.ndim == 3:
        data = array
    else:
        raise ValueError(
            "A dataset must be shaped (n_instances, n_timestamps) or "
            f"(n_instances, n_channels, n_timestamps), and the input has shape "
            f"{array.shape}."
        )
    if len(data) == 0:
        raise ValueError("A dataset cannot be empty.")
    return data


def _budget(value, name: str, minimum: int = 0) -> int:
    """Read a non-negative search budget, refusing anything else."""
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer, and {value!r} is not.") from None
    if number < minimum:
        raise ValueError(f"{name} must be at least {minimum}, and {number} is not.")
    return number


def _progress(iterable, enabled: bool, description: str):
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover - optional
        return iterable
    return tqdm(iterable, desc=description)


class SOFIExplainer:
    """Post hoc explainer that optimizes the sparseness of segment rankings.

    SOFI searches for an order of time series segments whose cumulative
    marginalization degrades the response of a classifier as fast as possible. A
    ranking is judged through its degradation score, namely the integral between
    the curve obtained when the least relevant segments are removed first and the
    one obtained when the most relevant segments are removed first. Maximizing
    that integral rewards sparsity and correctness at once, and the outcome is
    the smallest set of segments that changes the decision of the model.

    The explainer is agnostic to the classifier, which it queries for class
    probabilities alone, and agnostic to the segmentation, which may come from
    domain experts or from a change point detection algorithm. Both choices
    belong to the user and both are reported alongside the explanation, since a
    ranking of segments means nothing without the segments it ranks.

    Segmentation applies to one instance at a time, since change points differ
    from series to series, so a ranking describes one decision. Rankings can be
    pooled across instances only when the segmentation is shared by all of them,
    which happens with expert breakpoints and with uniform binning.

    Parameters
    ----------
    model : object
        Fitted classifier. Learners of ``tsai``, estimators exposing
        ``predict_proba``, Keras models and PyTorch modules are recognized
        without further declaration.
    X_train : array
        Training instances shaped ``(n, n_timestamps)`` or ``(n, n_channels,
        n_timestamps)``. They are the only source of marginalization values, and
        the instances explained afterwards never contribute statistics.
    y_train : array, optional
        Training labels, needed by the class directed operators alone.
    output_fn : str or callable, optional
        Method of the model that produces the outputs. A string names an
        attribute, such as ``"predict_proba"`` or ``"forward"``, and a callable
        receives the batch directly. The detection resolves it when the argument
        is left out.
    output : {"auto", "proba", "logits"}, default="auto"
        Whether the model already returns probabilities. Under ``"auto"`` a probe
        batch decides, and a softmax is applied only when the raw output leaves
        the probability simplex.
    channels_first : {"auto", True, False}, default="auto"
        Whether the model expects the channel axis before the time axis.
    classes : sequence, optional
        Class labels in the order of the probability columns, for reporting.
    segmentation : str, Segmentation or sequence, default="pelt"
        How an instance is divided. See ``segmentation_params`` for the
        arguments of each procedure.
    segmentation_params : dict, optional
        Arguments of the segmentation, such as ``n_segments`` for every method
        except PELT, ``penalty`` for PELT, ``cost_model``, ``min_size``,
        ``width`` and ``shared``.
    marginalization : str, Perturbation, number or sequence, default="mean"
        Operator that neutralizes a segment. ``"auto"`` runs the search once per
        candidate operator and keeps the best outcome, and an explicit sequence
        of names restricts those candidates.
    marginalization_params : dict, optional
        Arguments of the operator, such as ``n_replicas`` and ``taper``.
    initialization : {"greedy", "sequential", "random"} or sequence, default="greedy"
        Starting ranking. ``"greedy"`` reuses the ordering produced by the
        reference sweep, at no extra cost. ``"sequential"`` runs the greedy
        selection of Algorithm 3, which is stronger and quadratic in the segment
        count. An explicit sequence lets prior knowledge enter the search.
    max_iterations : int, default=200
        Budget of proposed swaps across all restarts. Zero evaluates the initial
        ranking without any search.
    patience : int, optional
        Consecutive swaps without improvement tolerated before a restart or the
        end of the search. ``None`` sets it to the whole budget.
    n_restarts : int, default=0
        Restarts from a random ranking granted once the patience expires.
    accept_equal : bool, default=False
        Whether swaps that leave the score unchanged are accepted.
    check_modularity : bool, default=True
        Whether the departure from the modularity assumption is measured, which
        says whether the greedy ranking is already optimal.
    random_state : int or Generator, optional
        Seed of the operator draws, of the swap operator and of the restarts.
    verbose : bool, default=True
        Whether the one line notice naming the resolved backend, output
        convention and channel layout is printed when the explainer is built.
    progress : bool, default=False
        Whether a progress bar follows the search. It is off by default, since a
        local explanation is quick and a bar per call would bury the output of a
        notebook, and it is worth turning on for long searches or large batches.

    Examples
    --------
    >>> explainer = SOFIExplainer(model, X_train, random_state=42)
    >>> explainer.inspect(X_test[0])
    >>> explanation = explainer.explain(X_test[0])
    >>> print(explanation.summary())
    """

    def __init__(
        self,
        model,
        X_train,
        y_train=None,
        output_fn=None,
        output: str = "auto",
        channels_first="auto",
        classes: Optional[Sequence] = None,
        segmentation: Union[str, Segmentation, Sequence] = "pelt",
        segmentation_params: Optional[Dict] = None,
        marginalization="mean",
        marginalization_params: Optional[Dict] = None,
        initialization: Union[str, Sequence] = "greedy",
        max_iterations: int = 200,
        patience: Optional[int] = None,
        n_restarts: int = 0,
        accept_equal: bool = False,
        check_modularity: bool = True,
        random_state=None,
        verbose: bool = True,
        progress: bool = False,
    ):
        self.wrapper = ClassifierWrapper(
            model,
            output_fn=output_fn,
            output=output,
            channels_first=channels_first,
            classes=classes,
            verbose=verbose,
        )
        if X_train is None:
            raise ValueError(
                "Training data is required, since it is the only source of "
                "marginalization values."
            )
        self.X_train = _finite(_as_dataset(X_train), "The training data")
        self.y_train = None if y_train is None else np.asarray(y_train)
        if self.y_train is not None and len(self.X_train) != len(self.y_train):
            raise ValueError("X_train and y_train hold a different number of instances.")

        self.segmentation = segmentation
        self.segmentation_params = dict(segmentation_params or {})
        self.marginalization = marginalization
        self.marginalization_params = dict(marginalization_params or {})
        self.initialization = initialization
        self.max_iterations = _budget(max_iterations, "max_iterations")
        self.patience = None if patience is None else _budget(patience, "patience", 1)
        self.n_restarts = _budget(n_restarts, "n_restarts")
        self.accept_equal = bool(accept_equal)
        self.check_modularity = bool(check_modularity)
        self.random_state = random_state
        self.verbose = bool(verbose)
        self.progress = bool(progress)

        self.wrapper.calibrate(self.X_train[: min(4, len(self.X_train))])

    # ------------------------------------------------------------- inspection
    def _check_instance(self, instance: np.ndarray) -> None:
        """Refuse an instance whose shape the training data cannot explain.

        Marginalization values come from the training data, so a series of a
        different length or a different number of channels has no admissible
        substitution and the model would receive an input it was never fitted
        for. Saying so here turns a confusing failure inside the estimator into
        a readable one.
        """
        expected = self.X_train.shape[1:]
        if instance.shape != expected:
            raise ValueError(
                f"The instance is shaped {instance.shape} while the training "
                f"data holds series shaped {expected}. Both must agree, since "
                "the marginalization values are drawn from the training data."
            )

    def segment(self, x, segmentation=None, **overrides) -> Segmentation:
        """Segmentation of one instance, without running any search.

        The procedure and its arguments default to the ones the explainer was
        built with, and either can be overridden for a single call, which is how
        several partitions of the same instance are obtained without building
        several explainers.

        Parameters
        ----------
        x : array
            The instance to divide, shaped ``(n_timestamps,)`` or
            ``(n_channels, n_timestamps)``.
        segmentation : str, Segmentation or sequence, optional
            Procedure for this call alone. ``method="binseg"`` is accepted as a
            synonym, so the name of a change point procedure can be passed the
            way the procedures themselves take it.
        **overrides : dict
            Arguments of the procedure, such as ``n_segments``, ``penalty``,
            ``cost_model``, ``min_size``, ``width`` or ``shared``. They override
            the ones the explainer was built with, for this call alone.

        Returns
        -------
        Segmentation
            The partition, which can be passed to :meth:`explain`.
        """
        instance = _as_instance(x)
        params = dict(self.segmentation_params)
        params.update(overrides)
        declared = params.pop("method", None)
        kind = segmentation if segmentation is not None else declared
        return build_segmentation(instance, self.segmentation if kind is None else kind, **params)

    def inspect(
        self,
        x,
        true_label=None,
        plot: bool = True,
        printout: bool = True,
        segmentation: Optional[Segmentation] = None,
        ax=None,
        channel: int = 0,
        style: str = "dots",
        **plot_kwargs,
    ) -> Dict[str, object]:
        """Describe an instance and its segmentation before explaining it.

        The report gathers the shape and statistics of the series, the confidence
        of the model with its runner up class, and the geometry of the
        segmentation together with the share of variance its segment means
        explain. Warnings appear when the decision sits near a class boundary,
        when the instance is misclassified, or when the breakpoints barely
        separate the series, since none of those yields a ranking worth reading.

        Parameters
        ----------
        x : array
            The instance to describe.
        true_label : optional
            Ground truth label of the instance. When it is given, the report
            says whether the prediction is correct and warns when it is not.
        plot : bool, default=True
            Whether the series and its breakpoints are drawn.
        printout : bool, default=True
            Whether the report is printed. Setting it to ``False`` returns the
            same facts without writing anything, which suits logging.
        segmentation : Segmentation, optional
            Partition to describe. The one of the explainer is used when it is
            omitted.
        ax : matplotlib Axes, optional
            Target axes of the figure. A new one is created when it is omitted.
        channel : int, default=0
            Channel drawn, for a multivariate series.
        style : {"dots", "lines"}, default="dots"
            How a breakpoint is marked, as described under
            :func:`~sofits.plotting.plot_segmentation`.
        **plot_kwargs : dict
            Further arguments of :func:`~sofits.plotting.plot_segmentation`,
            such as ``marker_color`` or ``linewidth``, so that the figure
            follows the same settings as every other one.

        Returns
        -------
        dict
            The facts the report formats, so that they can be logged or
            tabulated instead of printed.
        """
        instance = _as_instance(x)
        self._check_instance(instance)
        seg = (
            self.segment(instance)
            if segmentation is None
            else self._check_segmentation(instance, segmentation)
        )
        probabilities = self.wrapper.predict_proba(instance[None, ...])[0]
        facts = describe_instance(
            instance,
            seg,
            probabilities=probabilities,
            class_names=self.wrapper.classes,
            true_label=true_label,
        )
        if printout:
            print(instance_report(facts))
        if plot:
            from .plotting import plot_segmentation

            plot_segmentation(
                instance, seg, channel=channel, ax=ax, style=style, **plot_kwargs
            )
        facts["segmentation_object"] = seg
        return facts

    # ---------------------------------------------------------------- explain
    @staticmethod
    def _check_segmentation(instance: np.ndarray, seg: Segmentation) -> Segmentation:
        """Refuse a segmentation that was built for a different shape."""
        if seg.n_timestamps != instance.shape[1]:
            raise ValueError(
                f"The segmentation covers {seg.n_timestamps} time points and the "
                f"instance holds {instance.shape[1]}. A segmentation belongs to "
                "one series length, so build a new one or reuse the instance it "
                "was made for."
            )
        if seg.n_channels != instance.shape[0]:
            raise ValueError(
                f"The segmentation spans {seg.n_channels} channels and the "
                f"instance holds {instance.shape[0]}."
            )
        return seg

    def explain(
        self,
        x,
        segmentation: Optional[Segmentation] = None,
        marginalization=None,
        label: Optional[int] = None,
        initialization=None,
        max_iterations: Optional[int] = None,
        patience: Optional[int] = None,
        n_restarts: Optional[int] = None,
        random_state=None,
        verbose: Optional[bool] = None,
    ) -> SOFIExplanation:
        """Explain the prediction made for one instance.

        Parameters
        ----------
        x : array
            The series to explain, shaped ``(n_timestamps,)`` or
            ``(n_channels, n_timestamps)``.
        segmentation : Segmentation, optional
            Partition to rank. It is computed from the settings of the explainer
            when omitted, which is the usual case.
        marginalization : optional
            Operator for this call alone, overriding the one of the explainer.
        label : int, optional
            Position of the class whose probability the search tracks. The
            predicted class is used by default, which makes the explanation
            independent of the ground truth.
        initialization : {"greedy", "sequential", "random"} or sequence, optional
            Starting ranking for this call alone.
        max_iterations : int, optional
            Budget of proposed swaps for this call alone. Zero evaluates the
            initial ranking without searching.
        patience : int, optional
            Swaps without improvement tolerated for this call alone.
        n_restarts : int, optional
            Restarts granted for this call alone.
        random_state : int or Generator, optional
            Seed for this call alone.
        verbose : bool, optional
            Whether a progress bar follows this search.

        Returns
        -------
        SOFIExplanation
            The ranking, both curves, the scores and the counters of the run.
        """
        instance = _as_instance(x)
        self._check_instance(instance)
        seg = (
            self.segment(instance)
            if segmentation is None
            else self._check_segmentation(instance, segmentation)
        )
        spec = self.marginalization if marginalization is None else marginalization
        show = self.progress if verbose is None else bool(verbose)
        seed = self.random_state if random_state is None else random_state

        settings = dict(
            label=label,
            initialization=initialization,
            max_iterations=max_iterations,
            patience=patience,
            n_restarts=n_restarts,
            seed=seed,
        )
        if isinstance(spec, str) and spec.lower() == "auto":
            spec = list(DEFAULT_CANDIDATES)
        if isinstance(spec, (list, tuple)):
            return self._explain_best_of(instance, seg, list(spec), verbose=show, **settings)
        return self._explain_once(instance, seg, spec, verbose=show, **settings)

    def explain_batch(self, X, labels=None, **kwargs) -> List[SOFIExplanation]:
        """Explain several instances, one search each.

        Every instance receives its own segmentation, since change points are
        instance specific. The explanations can be pooled afterwards only when
        the segmentation is shared, which ``aggregate_explanations`` checks.

        Parameters
        ----------
        X : array
            Instances shaped ``(n, n_timestamps)`` or ``(n, n_channels,
            n_timestamps)``.
        labels : sequence of int, optional
            Class whose probability is tracked for each instance. The predicted
            class of each of them is used by default.
        **kwargs : dict
            Further arguments of :meth:`explain`, applied to every instance.

        Returns
        -------
        list of SOFIExplanation
            One explanation per instance, in the order they were given.
        """
        data = _as_dataset(X)
        show = kwargs.pop("verbose", self.progress)
        explanations = []
        for position in _progress(range(len(data)), show, "SOFI explanations"):
            explanations.append(
                self.explain(
                    data[position],
                    label=None if labels is None else int(labels[position]),
                    verbose=False,
                    **kwargs,
                )
            )
        return explanations

    def score_ranking(
        self,
        x,
        ranking: Sequence[int],
        segmentation: Optional[Segmentation] = None,
        marginalization=None,
        label: Optional[int] = None,
    ) -> SOFIExplanation:
        """Evaluate a ranking produced elsewhere under the same protocol.

        The degradation score and the sparsity rate become common ground for
        comparison, so any attribution method that ends in an order of segments
        can be measured against SOFI on identical terms.

        Parameters
        ----------
        x : array
            The instance the ranking refers to.
        ranking : sequence of int
            Positions of the segments, from the most to the least relevant. It
            must be a permutation of the segments of the partition.
        segmentation : Segmentation, optional
            Partition the ranking refers to. The one of the explainer is used
            when it is omitted.
        marginalization : optional
            Operator for this call alone.
        label : int, optional
            Position of the class whose probability is tracked.

        Returns
        -------
        SOFIExplanation
            The same object :meth:`explain` returns, with no search performed.
        """
        instance = _as_instance(x)
        self._check_instance(instance)
        seg = (
            self.segment(instance)
            if segmentation is None
            else self._check_segmentation(instance, segmentation)
        )
        return self._explain_once(
            instance,
            seg,
            self.marginalization if marginalization is None else marginalization,
            label=label,
            initialization=list(ranking),
            max_iterations=0,
            patience=None,
            n_restarts=0,
            seed=self.random_state,
            verbose=False,
        )

    # ------------------------------------------------------------- internals
    def build_objective(self, x, segmentation=None, marginalization=None, label=None, random_state=None):
        """Bind an objective to one instance, without running any search.

        The objective is the piece that drives the model through cumulative
        marginalization, so it is what a custom search or an external baseline
        needs. Everything the explainer was configured with is honored.

        Parameters
        ----------
        x : array
            The instance to bind the objective to.
        segmentation : Segmentation, optional
            Partition to rank. The one of the explainer is used when it is
            omitted.
        marginalization : optional
            Operator for this call alone.
        label : int, optional
            Position of the class whose probability is tracked.
        random_state : int or Generator, optional
            Seed of the draws the operator makes.

        Returns
        -------
        tuple
            The objective and the operator already fitted on the instance.
        """
        instance = _as_instance(x)
        self._check_instance(instance)
        seg = (
            self.segment(instance)
            if segmentation is None
            else self._check_segmentation(instance, segmentation)
        )
        spec = self.marginalization if marginalization is None else marginalization
        rng = np.random.default_rng(
            self.random_state if random_state is None else random_state
        )
        return self._build_objective(instance, seg, spec, label, rng)

    def _build_objective(self, instance, seg, spec, label, rng):
        """Fit the operator on the instance and bind an objective to it."""
        operator = make_perturbation(spec, **self.marginalization_params)
        probe = self.wrapper.predict_proba(instance[None, ...])[0]
        if label is None:
            position = int(np.argmax(probe))
        else:
            position = int(label)
            if not 0 <= position < probe.size:
                raise ValueError(
                    f"label={label} names no class of this model, which returns "
                    f"{probe.size} of them. Pass the position of a column of the "
                    "probability matrix."
                )
        context = PerturbationContext(
            instance=instance,
            label=position,
            wrapper=self.wrapper,
            X_train=self.X_train,
            y_train=self.y_train,
            rng=rng,
        )
        operator.fit(context)
        operator.warn_if_taper_is_idle(seg)
        objective = Objective(self.wrapper, instance, seg, operator, label=position)
        return objective, operator

    def _initial_order(self, objective, initialization, rng) -> List[int]:
        spec = self.initialization if initialization is None else initialization
        n = objective.n_segments
        if isinstance(spec, str):
            key = spec.lower()
            if key == "random":
                return [int(p) for p in rng.permutation(n)]
            if key == "sequential":
                return greedy_ranking(objective)
            if key == "greedy":
                return objective.greedy_order()
            raise ValueError(
                "initialization must be 'greedy', 'sequential', 'random' or a ranking."
            )
        order = [int(position) for position in spec]
        if sorted(order) != list(range(n)):
            raise ValueError(
                f"An explicit ranking must be a permutation of the {n} segments."
            )
        return order

    def _explain_once(
        self,
        instance,
        seg,
        spec,
        label,
        initialization,
        max_iterations,
        patience,
        n_restarts,
        seed,
        verbose,
    ) -> SOFIExplanation:
        started = time.time()
        rng = np.random.default_rng(seed)
        objective, operator = self._build_objective(instance, seg, spec, label, rng)
        calls_before = objective.n_calls_

        start = self._initial_order(objective, initialization, rng)
        budget = self.max_iterations if max_iterations is None else int(max_iterations)
        result = hill_climbing(
            n_segments=objective.n_segments,
            evaluate=objective.evaluate,
            initial_order=start,
            max_iterations=budget,
            patience=self.patience if patience is None else patience,
            n_restarts=self.n_restarts if n_restarts is None else int(n_restarts),
            accept_equal=self.accept_equal,
            rng=rng,
            verbose=verbose,
        )
        evaluation = result["evaluation"]
        point, rate, _ = objective.sparsity(evaluation)
        lerf_point = objective.flip_point(list(evaluation.order)[::-1])
        gap = modularity_gap(objective) if self.check_modularity else None

        # the series and the states it passes through are small next to the
        # queries already spent, and every figure needs them, so they are always
        # carried rather than hidden behind a flag
        states = objective.marginalized_series(evaluation.order)

        return SOFIExplanation(
            segmentation=seg,
            order=[int(p) for p in evaluation.order],
            morf_scores=[float(v) for v in evaluation.morf],
            lerf_scores=[float(v) for v in evaluation.lerf],
            ds=float(evaluation.ds),
            label=int(objective.label),
            class_name=self.wrapper.class_name(objective.label),
            sparsity_point=point,
            sparsity_rate=rate,
            sparsity_probability=float(
                evaluation.morf[point] if point else evaluation.morf[-1]
            ),
            lerf_sparsity_point=lerf_point,
            perturbation=operator.summary(),
            modularity_gap=gap,
            instance=instance.copy(),
            states=states,
            history=result["history"],
            n_iterations=result["n_iterations"],
            n_restarts_used=result["n_restarts_used"],
            n_evaluations=result["n_evaluations"],
            n_model_calls=objective.n_calls_ - calls_before,
            runtime=time.time() - started,
            source=self,
        )

    def _explain_best_of(self, instance, seg, candidates, verbose=False, **kwargs) -> SOFIExplanation:
        """Run one search per candidate operator and keep the best outcome.

        No single operator fits every problem, so the choice is treated as a
        hyperparameter resolved per instance. The criterion is the degradation
        score, and ties keep the earlier candidate, which favors the cheaper
        operators listed first.
        """
        best, trials = None, []
        for spec in _progress(candidates, verbose, "SOFI operators"):
            try:
                explanation = self._explain_once(instance, seg, spec, verbose=False, **kwargs)
            except (ValueError, ImportError) as error:
                warnings.warn(
                    f"The '{spec}' operator was skipped, {type(error).__name__}, {error}",
                    stacklevel=3,
                )
                continue
            trials.append(
                {
                    "operator": explanation.perturbation.get("operator"),
                    "detail": explanation.perturbation.get("detail"),
                    "ds": explanation.ds,
                    "sparsity_rate": explanation.sparsity_rate,
                }
            )
            if best is None or explanation.ds > best.ds:
                best = explanation
        if best is None:
            raise ValueError(
                "Every candidate operator failed, so no explanation could be "
                "produced. Inspect the warnings above for the reasons."
            )
        best.perturbation = dict(best.perturbation)
        best.perturbation["trials"] = trials
        return best

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"SOFIExplainer(segmentation={self.segmentation!r}, "
            f"marginalization={self.marginalization!r}, backend={self.wrapper.backend})"
        )
