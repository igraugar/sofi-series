"""SOFI for time series classification.

Sparseness Optimized Feature Importance is a model agnostic post hoc explainer.
An explanation is an order of time series segments whose cumulative perturbation
degrades the response of a classifier as fast as possible, and its quality is
the area between the two perturbation curves that order induces. Maximizing that
area rewards sparsity and faithfulness at once, so the explanation isolates the
smallest set of segments that changes the decision.

The segments are the unit of interpretation and they belong to the user. Expert
breakpoints are the ideal case, since a cardiologist reads an ECG in terms of
waves and intervals rather than time points, and change point detection stands
in when no domain knowledge is available.

Two objects cover the ordinary use. ``SOFIExplainer`` is configured once and
explains one instance at a time, and the ``SOFIExplanation`` it returns reports
and draws itself. ``Experiment`` sits beside them and serves studies that
compare many runs, which is the only setting where the instances are chosen by a
rule rather than brought by an expert.
"""

from .baselines import (
    compare_rankings,
    feature_occlusion,
    random_ranking,
    ranking_from_scores,
    segment_scores,
)
from .data import TimeSeriesDataset, load_dataset, load_local
from .explainer import SOFIExplainer
from .explanation import SOFIExplanation, aggregate_explanations, noise_onset
from .metrics import (
    fold_change,
    nearest_neighbors,
    rank_biased_overlap,
    robustness_score,
)
from .model import ClassifierWrapper
from .objective import (
    Objective,
    RankingEvaluation,
    curve_auc,
    degradation_score,
    modularity_gap,
)
from .perturbation import (
    CLASS_AGNOSTIC,
    CLASS_DIRECTED,
    BackgroundPerturbation,
    ConstantPerturbation,
    LinearPerturbation,
    LineSearchPerturbation,
    NearestUnlikeNeighborPerturbation,
    NoisePerturbation,
    OppositeClassMeanPerturbation,
    Perturbation,
    PerturbationContext,
    RandomConstantPerturbation,
    SegmentMeanPerturbation,
    make_perturbation,
)
from .animation import Animation, animate_marginalization
from .plotting import (
    plot_degradation_curves,
    plot_explanation,
    plot_marginalization,
    plot_marginalized_series,
    plot_segmentation,
)
from .experiment import (
    Experiment,
    plot_degradation_grid,
    plot_explanation_grid,
    plot_segmentation_grid,
    select_reliable_instances,
)
from .report import describe_instance, instance_report
from .search import greedy_ranking, hill_climbing
from .segmentation import (
    CHANGE_POINT_METHODS,
    Segment,
    Segmentation,
    change_point_segmentation,
    manual_segmentation,
    segmentation_from_mask,
    uniform_segmentation,
)

__version__ = "1.0.0"

__all__ = [
    # the explainer and its result
    "SOFIExplainer",
    "SOFIExplanation",
    "ClassifierWrapper",
    # segments
    "Segment",
    "Segmentation",
    "CHANGE_POINT_METHODS",
    "manual_segmentation",
    "uniform_segmentation",
    "change_point_segmentation",
    "segmentation_from_mask",
    # marginalization
    "Perturbation",
    "PerturbationContext",
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
    "CLASS_AGNOSTIC",
    "CLASS_DIRECTED",
    # figures of one explanation
    "plot_segmentation",
    "plot_degradation_curves",
    "plot_explanation",
    "plot_marginalization",
    "plot_marginalized_series",
    "Animation",
    "animate_marginalization",
    # experimentation
    "Experiment",
    "select_reliable_instances",
    "plot_segmentation_grid",
    "plot_explanation_grid",
    "plot_degradation_grid",
    "aggregate_explanations",
    "compare_rankings",
    "segment_scores",
    "ranking_from_scores",
    "feature_occlusion",
    "random_ranking",
    "rank_biased_overlap",
    "robustness_score",
    "nearest_neighbors",
    "fold_change",
    # data
    "TimeSeriesDataset",
    "load_dataset",
    "load_local",
    # building blocks
    "Objective",
    "RankingEvaluation",
    "curve_auc",
    "degradation_score",
    "modularity_gap",
    "hill_climbing",
    "greedy_ranking",
    "noise_onset",
    "describe_instance",
    "instance_report",
    "__version__",
]
