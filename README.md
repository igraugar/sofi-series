# Sparseness Optimized Feature Importance for Time Series Classification

Sparseness Optimized Feature Importance (SOFI) is a model agnostic, declarative post hoc explainer. An explanation takes the form of a ranking of time series segments, and its quality is the degradation score obtained after cumulative marginalization. The implementation supports univariate and multivariate problems, ranks segments defined by domain experts as readily as segments discovered automatically, and works with any classifier that yields class probabilities.

## Installation

```bash
git clone https://github.com/<user>/sofits.git
cd sofits
pip install -e .
```

The package requires Python 3.9 or later. Everything needed to segment a series, run the search, draw every figure and render the animation is installed with it, namely `numpy`, `pandas`, `ruptures`, `matplotlib`, `seaborn`, `pillow` and `tqdm`. No further installation is required for any function documented below.

Neither PyTorch, nor Keras, nor `tsai` is a dependency. The package works with all three and imports none of them until the model handed to it requires one. The notebook trains one model per framework, so it asks for them explicitly.

```bash
pip install -e ".[demo]"      # what SOFI_demo.ipynb needs, on top of the package itself
```

## Quick start

```python
from sofits import SOFIExplainer, load_dataset

data = load_dataset("CBF")                    # bundled, no network needed
instance = data.X_test[5]                     # the series the expert wants explained

explainer = SOFIExplainer(model, data.X_train, random_state=42)
explainer.inspect(instance)                   # read this before explaining
explanation = explainer.explain(instance)

print(explanation.summary())
explanation.plot_explanation(save_path="explanation.pdf")
explanation.plot_animation(save_path="explanation.gif")
```

Two objects cover the whole interface. The explainer is configured once and produces explanations, and an explanation reports and draws itself. A third object, `Experiment`, serves studies that compare many runs and is described under [Experimentation](#experimentation).

`SOFI_demo.ipynb` follows the same division. Its first part configures an explainer, explains one instance and draws every figure an explanation offers. Its second part runs the comparisons that a study needs, over segmentation procedures, marginalization operators and competing explainers.

Cylinder Bell Funnel carries the notebook, and BasicMotions, a six channel problem of four classes recorded from a wrist sensor, ships alongside it for the multivariate path. Both are bundled, so nothing has to be downloaded.

## The degradation score

Two curves are derived from a ranking. The MoRF curve marginalizes the most relevant segments first and should collapse immediately, since a sparse explanation concentrates the response of the model in a handful of segments. The LeRF curve marginalizes the least relevant segments first and should stay flat for as long as possible, since the segments declared irrelevant must indeed be the ones whose removal leaves the response untouched. Both curves share their first point, the unperturbed response, and their last point, the response once every segment has been neutralized. The degradation score is the area between the two curves, and the search maximizes it. A single curve cannot capture both properties. The MoRF branch alone rewards sparsity while saying nothing about the tail of the ranking, and the LeRF branch alone rewards correctness at the tail while saying nothing about the head.

Both curves report the probability the model assigns to the class it predicted before any perturbation, as in region perturbation, and nothing else. Ground truth never enters, which keeps explanations independent of the error of the model, and no rescaling is applied, so a value of 0.4 on a curve is a probability of 0.4. The first point of a curve is therefore the confidence of the model on the untouched instance rather than a fixed one, which lets a reader see how certain the model was before anything was removed and judge the collapse against it. Curves from different instances are compared on the same axis because a probability means the same thing everywhere, and no anchor has to be agreed on first.

The objective is built from a reference sweep that marginalizes every segment on its own and then follows the resulting greedy ordering cumulatively. That sweep supplies the starting ranking of the search and establishes whether the response reacts to marginalization at all. When it never moves, a warning says so, because the degradation score then carries no information.

## The sparsity rate

The sparsity rate is the number of segments that must be marginalized before the predicted class changes, divided by the segment count, so lower values denote sparser explanations. A ranking whose cumulative marginalization never changes the class receives a rate of one and a position of `None`, and the two outcomes are reported separately rather than conflated. The same measure applied to the reversed ranking is reported by `lerf_sparsity_rate`, and their separation quantifies how much earlier the ranking induces the change of class than its reverse. The degradation curves mark both events with a star.

## Declaring the segments

Segments are the unit of interpretation and they belong to the user. A cardiologist reads an electrocardiogram in terms of the P wave, the QRS complex and the ST segment, so an explanation phrased in those terms is meaningful while one phrased in time points is not. Nine ways of declaring them are available.

| `segmentation` | What it does | Key arguments |
| --- | --- | --- |
| `"pelt"` | Change points from a globally optimal penalized cost, the default | `penalty`, `cost_model`, `min_size` |
| `"dynp"` | Exact optimal segmentation for a fixed count | `n_segments` |
| `"binseg"` | Greedy recursive splitting | `n_segments` |
| `"bottomup"` | Fine initial partition merged upwards | `n_segments` |
| `"window"` | Statistics compared across sliding windows | `n_segments`, `width` |
| `"uniform"` | Equal parts, the same for every instance | `n_segments` |
| `"manual"` | Breakpoints supplied by an expert | `breakpoints` |
| `"mask"` | An arbitrary integer mask | `mask` |
| a sequence | Read directly as the breakpoints | none |

```python
from sofits import SOFIExplainer, manual_segmentation

# an electrocardiogram divided the way a clinician reads it
segments = manual_segmentation([40, 62, 95, 140], n_timestamps=187)
explainer = SOFIExplainer(model, X_train, segmentation=segments)
```

PELT is driven by a penalty rather than by a count, so it decides for itself how many breakpoints the series deserves, while the other change point procedures take the count as given. A sliding window cannot always place as many breakpoints as it is asked for, since a wide window leaves too little room on a short series, and the shortfall is announced rather than passed over in silence.

Change point detection applies to one instance at a time, since the structure of one series says nothing about the structure of another. Rankings from different instances therefore describe different partitions and cannot be pooled. Uniform binning and expert breakpoints are shared by construction, and only those two allow the explanations of a batch to be summarized together.

A multivariate series receives one set of breakpoints per channel, so a segment is a channel and an interval together and a ranking may mix channels, which lets an explanation say where as well as when. Passing `shared=True` derives a single set of breakpoints from every channel at once, which suits synchronized signals.

## Marginalizing a segment

Marginalizing a tabular feature is easy, since one statistic of the training column carries no instance level information. A time series segment is harder. A flat constant is itself a shape, and a model may read it as evidence for some class, which leaves the marginalized series informative and the curve misleading. Fourteen operators are available in two families, and the right one is problem dependent.

**Class agnostic operators** erase the content of a segment without steering the prediction anywhere. A ranking obtained this way answers which segments the prediction relies on, which is the question the method was designed for. These are the default.

| Name | Substitution |
| --- | --- |
| `"mean"` | The training mean, the default and the most reliable constant across the benchmarks |
| `"min"`, `"max"`, `"median"`, `"zero"` | Other constants of the training data |
| `"random"` | A fresh uniform draw per segment and per replica |
| `"noise"` | Gaussian noise matched to the training level and dispersion |
| `"segment_mean"` | The average of the segment itself, which erases shape and keeps level |
| `"linear"` | The line joining the segment endpoints, which introduces no discontinuity |
| `"background"` | The matching window of several training instances, averaged over the draws |

Every constant reads the statistic of the channel a segment belongs to rather than a value pooled over all of them. Channels of a multivariate series often carry different physical quantities on different scales, so a pooled constant would neutralize one channel while introducing a gross outlier in another. The two coincide when the series is univariate.

**Class directed operators** choose the substitution so that the response falls as fast as possible. They produce sparser explanations and a different reading, since a ranking obtained this way answers which segments move the model away from the current class fastest, which is a counterfactual question rather than a reliance question. Every explanation states which family produced it, in `explanation.interpretation` and in the printed summary, so the two readings are never confused.

| Name | Substitution |
| --- | --- |
| `"line_search"` | The constant found by the golden section search of the paper |
| `"admissible_line_search"` | The same search restricted to constants whose flat series is not assigned the explained class, so the substitution is known in advance to carry no evidence for it |
| `"opposite_mean"` | The matching window of the mean series of the other classes |
| `"nearest_unlike"` | The matching window of the nearest training instance of another class |

Two further controls apply to every operator. `n_replicas` averages the response over several substitutions, which approximates the expectation the method is defined on rather than the output at one arbitrary point, at a cost that grows linearly with the sample. `taper` blends the substituted values into their neighbors over a few time points, which removes the step discontinuity a flat constant introduces at a breakpoint. Convolutional models react strongly to sharp edges, so part of a measured degradation can be a boundary artifact rather than lost evidence, and the default of zero reproduces the published behavior.

No single operator fits every problem. Setting `marginalization="auto"` runs the search once per candidate and keeps the highest degradation score, which treats the choice as a per instance hyperparameter. The candidates are the cheap class agnostic operators, so the reading of the ranking never changes without the user asking, and `explanation.perturbation["trials"]` reports what each of them achieved.

## The noise region

Once the informative segments are gone, marginalizing what remains often pushes the response back towards its original state. The MoRF curve then climbs after its minimum, and the tail of the ranking that follows that minimum carries no evidence. The region is reported by `noise_onset` and `noise_features`, and it also appears in the printed summary. Segments inside it receive no importance weight, since the ranking says nothing about them.

## The search

Hill climbing with a local operator that swaps two randomly selected positions of the current ranking. A candidate is accepted when it raises the degradation score. The starting ranking is either the greedy ordering already produced by the reference sweep, at no extra cost, the sequential selection of Algorithm 3, which is stronger and quadratic in the segment count, or a random one. An explicit ranking is also accepted, which is how prior knowledge enters the procedure. The run ends after a budget of iterations or once the patience expires, and restarts from a random ranking replace an early stop when `n_restarts` is positive.

`modularity_gap` reports the largest departure from the modularity assumption of Theorem 3 along the greedy order. A value near zero says the greedy ranking is already optimal and the search has nothing left to find, and larger values say the effect of marginalizing a segment depends on what was marginalized before, which is the ordinary situation and the reason the search exists.

## Declaring the classifier

A fitted model is always required, since the explanation describes that model and nothing else. Four families are recognized without importing their libraries, so neither PyTorch nor Keras becomes a dependency of the package.

| Family | Recognized through |
| --- | --- |
| `tsai` | `get_X_preds`, from which the probabilities are read |
| scikit-learn, `aeon`, `sktime` | `predict_proba` |
| Keras and TensorFlow | the call interface of the model |
| PyTorch | `torch.nn.Module`, evaluated under `no_grad` on the device of its parameters |

Keras is tested before PyTorch on purpose. Under the PyTorch backend a Keras model also inherits from `torch.nn.Module`, so the reverse order would drive it through the PyTorch calling convention and bypass its own preprocessing.

Two conventions are resolved once, on a probe batch drawn from the training data, and then held fixed. The output convention states whether the raw output already lies on the probability simplex, and a softmax is applied only when it does not. The channel layout states whether the model expects the channel axis before the time axis, as PyTorch does, or after it, as Keras does. Both are detected by default and both can be stated explicitly when the probe would be ambiguous.

```python
SOFIExplainer(torch_module, X_train)                                    # detected
SOFIExplainer(keras_model, X_train)                                     # detected
SOFIExplainer(tsai_learner, X_train)                                    # detected
SOFIExplainer(model, X_train, output="logits", channels_first=True)     # declared conventions
SOFIExplainer(model, X_train, output_fn="forward", output="logits")     # declared method
```

`output_fn` names the method of the model that produces the outputs, either as the name of an attribute or as a callable. It covers the cases where the detection would pick the wrong method or where none carries a familiar name, and the conversion of the input stays the one the framework of the model expects. A model exposing hard labels alone is rejected at construction, since probability degradation is the quantity SOFI measures and failing early with a readable message is preferable to failing inside the search.

## Choosing the instance

The instance to explain is the one the expert brings. SOFI explains the decision made on that series, and no filter is applied to it.

`inspect` describes it before the search runs. It reports the shape and the statistics of the series, the confidence of the model with its runner up class, and the geometry of the segmentation, closing with a separation index that compares the variance left inside the segments with the total variance in the manner of a one way analysis of variance. Warnings appear when the decision sits near a class boundary, when the instance is misclassified, and when the breakpoints barely separate the series. None of those three yields a ranking worth reading, so the check is worth running first.

```python
facts = explainer.inspect(instance, true_label=y_true)
```

A study that must run over many instances faces a different question, since it needs a stated criterion for the instances it pools. `select_reliable_instances` covers the usual one and belongs to the experimentation interface described below. It has no place in an ordinary application.

## Loading data

`load_dataset` tries three sources in turn and any of them alone suffices. A local directory holding `X_train.npy`, `y_train.npy`, `X_valid.npy` and `y_valid.npy`, or a pair of `.ts` files in the format of the UCR and UEA archives, needs nothing beyond `numpy` and is the path to prefer when the data must travel with the code. The `tsai` loader is used when that package is installed. A direct download of the archive of the dataset is the last resort, cached on disk so the network is consulted once.

```python
load_dataset("CBF")                              # bundled
load_dataset("BasicMotions")                     # bundled, multivariate
load_dataset("ECG200")                           # tsai or download
load_dataset("MyProblem", directory="~/data")    # your own files
```

## Drawing an explanation

An explanation carries the instance, the segments, both curves and the states the series passed through, so it draws its own figures with no further arguments. Every method accepts the settings of the figure it produces and writes it to disk through `save_path`.

| Method | Figure |
| --- | --- |
| `plot_segmentation()` | The series with the breakpoints that divide it |
| `plot_degradation_curves()` | The MoRF and LeRF curves and the area between them |
| `plot_explanation()` | Two panels, the segments the ranking retains beside the curves |
| `plot_marginalization()` | Snapshots of the series along the cumulative marginalization |
| `plot_animation()` | The two passes as an animated GIF, displayed in the notebook cell |

```python
explanation.plot_segmentation(style="dots", marker_color="black")
explanation.plot_degradation_curves(morf_color="#03719c", linewidth=1.4)
explanation.plot_explanation(figsize=(11.5, 4.2), save_path="explanation.pdf")
explanation.plot_marginalization(ncols=3, panel_size=(4.6, 3.1))
explanation.plot_animation(save_path="explanation.gif")
```

The same figures are available as functions of `sofits`, namely `plot_segmentation`, `plot_degradation_curves`, `plot_explanation`, `plot_marginalization` and `plot_marginalized_series`, which are what the methods above call. The functions take the explanation as their first argument and are useful when a figure must be assembled inside a layout of your own, through the `ax` argument.

Every visual property is an argument of the call that draws the figure, and nothing is stored between calls. Colors, line thickness, marker size, titles, axis labels, panel sizes, the theme and the font scale all have defaults declared in the signature of each function, so a session needs no preparation and a figure that departs from those defaults affects no other figure.

```python
explanation.plot_degradation_curves()                                    # the defaults
explanation.plot_degradation_curves(lerf_color="#865321", linewidth=1.4, # this figure alone
                                    markersize=7, figsize=(5.6, 4.0),
                                    theme="ticks", font_scale=1.3)
```

`save_path` writes the figure where it points. A path without an extension receives `.pdf` for a static figure and `.gif` for the animation, and the enclosing directory is created when it does not exist. The animation is returned as an object that a notebook displays in the cell that produced it, whether or not a destination was given, and `save` writes it afterwards.

```python
animation = explanation.plot_animation()     # rendered in the cell, with a progress bar
animation.save("explanation.gif")            # written when you want to keep it
```

The closing frame of the animation is the figure `plot_explanation` produces, since both are drawn from the same result with the same segments and the same operator. The left panel of each is drawn by `plot_marginalized_series`, so the two cannot drift apart.

## Reading the marginalization

Segments are marginalized cumulatively, so the effect of a segment depends on what was already removed and no independent score describes it. `plot_marginalization` therefore shows the process instead of summarizing it. The untouched series is drawn in gray, the series after the first steps of the ranking have been removed is drawn in color, the region already marginalized is shaded behind them, and the probability that survives each step heads its panel, so the collapse of the MoRF curve can be read off the series itself.

The animation follows the same process one frame at a time. The two curves rest on two different orderings, the LeRF one being the reverse of the MoRF one, so they are built one after the other rather than together. The LeRF pass runs first, marginalizing the least relevant segment and working towards the most relevant one, then the panel is cleared and the MoRF pass runs in the opposite direction. Once both curves are complete, the area between them is shaded, the degradation score is written, and the series is redrawn carrying the segments that constitute the explanation.

While a pass runs, the panel marks the segment being marginalized at that step and nothing else, in gray, so the eye follows one removal at a time and the series is never buried under accumulated shading. The segments already dealt with need no mark, since the flat stretch that replaced them identifies them on its own. Judging a segment belongs to the ranking rather than to either pass, and it is settled only once both curves exist, so the closing frame marks the segments the ranking places first, namely the smallest set whose cumulative marginalization changes the predicted class. Those are the segments responsible for the decision.

The explainer is consulted while the animation is built, because the states of the LeRF pass have to be rebuilt and the explanation carries only those of the MoRF pass. `plot_animation` passes it for you. The free function `animate_marginalization` accepts it as its second argument, and leaving it out shows the MoRF pass alone, with the LeRF curve already complete when it starts.

## Experimentation

A study compares many runs against one another, which is a different activity from explaining a decision and is kept apart from it. `Experiment` is bound to an explainer, whose segmentation and operator define the protocol every comparison runs under, and it gathers the instance selection, the comparisons themselves and the grid figures that display them.

```python
from sofits import Experiment

study = Experiment(explainer, random_state=42)

reliable, mask = study.select_reliable_instances(X_test, y_test, return_mask=True)
runs = study.compare_marginalizations(instance, ["mean", "linear", "noise", "background"])

study.plot_degradation_grid(runs, ncols=2, save_path="operators.pdf")
study.summarize(runs)
```

| Method | What it does |
| --- | --- |
| `select_reliable_instances(X, y)` | Keeps the instances the classifier gets right, which is the usual criterion for the batch a study pools. Experiments only, since in an application the expert supplies the instance |
| `explain_batch(X)` | One search per instance, each with its own segmentation |
| `compare_segmentations(instance, segmentations)` | One explanation per partition of the same instance |
| `compare_marginalizations(instance, operators)` | One explanation per operator, on a shared partition |
| `compare_explainers(instance, explainers)` | One explanation per explainer, so that models can be set against one another |
| `compare_rankings(instance, rankings)` | Scores rankings obtained elsewhere under the same protocol |
| `summarize(explanations)` | One row per run with its degradation score, its sparsity rate, the cost of the search and the leading segments |
| `aggregate(explanations)` | Mean rank of every segment over a batch, for a partition shared by every instance |
| `robustness(explanations, X)` | Agreement between the ranking of an instance and that of its nearest neighbor |

Each comparison returns a mapping from a heading onto an explanation, which the three grid figures accept directly. They are the experimentation counterpart of the figures of a single explanation, one panel per run.

| Figure | What it draws |
| --- | --- |
| `plot_explanation_grid(explanations)` | The relevant segments of every run, the left panel of `plot_explanation` |
| `plot_degradation_grid(explanations)` | The MoRF and LeRF curves of every run, the right panel of `plot_explanation` |
| `plot_segmentation_grid(instance, segmentations)` | One instance under several partitions, with no segment singled out |

The three are also importable as functions of `sofits` and take the same arguments as the methods. Like every other figure, they accept `save_path` and the settings of the panels they hold.

`compare_rankings` establishes the degradation score as a common ground between methods. `explainer.score_ranking` evaluates a ranking produced elsewhere under the protocol of the explainer, and `explainer.build_objective` returns the objective bound to an instance without running any search, which is what a custom optimizer needs. Any attribution method that ends in an ordering of segments can be measured this way, and a point wise saliency map enters the same table once `segment_scores` has averaged it inside each segment.

```python
from sofits import Experiment, feature_occlusion, random_ranking, ranking_from_scores, segment_scores

occlusion = ranking_from_scores(feature_occlusion(explainer, instance, segmentation))
gradients = ranking_from_scores(segment_scores(saliency_map, segmentation))
table = Experiment(explainer).compare_rankings(
    instance,
    {
        "SOFI": explanation.order,
        "Occlusion": occlusion,
        "Gradients": gradients,
        "Random": random_ranking(segmentation.n_segments, 42),
    },
    segmentation=segmentation,
)
```

Feature occlusion is implemented in the package, since it needs only the classifier and it is exactly the reference sweep the search starts from, so `feature_occlusion(explainer, instance)` returns the drop each segment causes on its own. That connection carries a theoretical reading. Under the modularity assumption of Theorem 3 the contributions of the segments simply add up, and the optimal ranking is then exactly the one obtained by sorting those individual drops, which is what occlusion produces, so a searched ranking that coincides with the occlusion ranking behaves as the modular case predicts. The converse does not follow, since the two orderings can agree by coincidence, and `modularity_gap` settles the question directly.

`robustness` measures how far the ranking of an instance agrees with the ranking of its nearest neighbor, through rank biased overlap. That measure rests on the assumption that locally similar series deserve similar explanations, which is common yet not correlated with faithfulness or sparsity, so the number is informative alongside the others and misleading on its own.

## Cost and reproducibility

Building the objective for an instance costs two sweeps of the segments, one to marginalize each segment alone, which produces the greedy ordering, and one to follow that ordering cumulatively. Every candidate proposed afterwards requires the two curves, so an evaluation costs twice a single curve. A curve issues one batched query, and a candidate produced by a swap at positions `i` and `j` (with `i < j`) reuses the first `i` points of the MoRF curve of the incumbent and the first `n - 1 - j` points of its LeRF curve, since the reversed ranking is affected at mirrored positions. Only the affected tails are recomputed, and the result is identical to a full evaluation. Replicas multiply every query by their count.

Every run is reproducible from `random_state`, which controls the operator draws, the swap operator and the restarts. No internal state is modified during a search, so repeated calls on the same instance return the same explanation.

## Reference of `SOFIExplainer`

| Parameter | Default | Accepted values | Rationale |
| --- | --- | --- | --- |
| `model` | required | A fitted classifier of any of the four families, or any object whose output method is named through `output_fn`. | The explainer is agnostic to the model family and only queries its outputs. A model is always required, since the explanation describes it. |
| `X_train` | required | Array shaped `(n, n_timestamps)` or `(n, n_channels, n_timestamps)`. | Source of the marginalization values. The instances explained afterwards never contribute statistics. |
| `y_train` | `None` | `None` or an array of labels. | Needed by the class directed operators alone, which draw their substitution from the classes other than the one being explained. |
| `output_fn` | `None` | `None`, the name of a method of the model, or a callable receiving the batch. | Names the method that produces the outputs, for the cases where the detection would pick the wrong one. A callable takes full responsibility for the conversion of the input. |
| `output` | `"auto"` | `"auto"`, `"proba"`, `"logits"`. | Whether the model already returns probabilities. `"auto"` lets a probe batch decide, and a softmax is applied only when the raw output leaves the probability simplex. |
| `channels_first` | `"auto"` | `"auto"`, `True`, `False`. | Whether the model expects the channel axis before the time axis. `"auto"` tries both orientations on the probe. |
| `classes` | `None` | `None` or a sequence of labels. | Class labels in the order of the probability columns, used for reporting alone. Read from the estimator when available. |
| `segmentation` | `"pelt"` | Any value of the table above, or a ready `Segmentation`. | Declares which units are ranked. |
| `segmentation_params` | `None` | Dict of arguments of the procedure. | Holds `n_segments`, `penalty`, `cost_model`, `min_size`, `width` and `shared`. |
| `marginalization` | `"mean"` | Any operator name, a number, a `Perturbation`, `"auto"`, or a sequence of names. | Declares what neutralizing a segment means. A number is read as a constant, and `"auto"` resolves the choice per instance. |
| `marginalization_params` | `None` | Dict of arguments of the operator. | Holds `n_replicas` and `taper`, among others. |
| `initialization` | `"greedy"` | `"greedy"`, `"sequential"`, `"random"`, or a sequence of positions. | `"greedy"` starts from the ordering produced by the reference sweep, which is already available. `"sequential"` runs Algorithm 3, which is stronger and quadratic. An explicit sequence lets prior knowledge enter the search. |
| `max_iterations` | `200` | Non-negative integer. | Budget of proposed swaps across all restarts. Zero evaluates the initial ranking without any search. |
| `patience` | `None` | `None` or a positive integer. | Consecutive swaps without improvement tolerated before a restart or the end of the search. `None` sets it to the whole budget. |
| `n_restarts` | `0` | Integer of at least zero. | Restarts from a random ranking granted once the patience expires. |
| `accept_equal` | `False` | `False`, `True`. | Whether swaps that leave the score unchanged are accepted, which lets the search drift along plateaus. |
| `check_modularity` | `True` | `True`, `False`. | Whether the departure from the modularity assumption is measured, which costs one extra curve. |
| `random_state` | `None` | `None`, an integer, or a `numpy` `Generator`. | Seed of the operator draws, of the swap operator and of the restarts. |
| `verbose` | `True` | `True`, `False`. | Whether the notice naming the resolved backend, output convention and channel layout is printed when the explainer is built. |
| `progress` | `False` | `True`, `False`. | Whether a progress bar follows the search. Off by default, since a local explanation is quick and a bar per call would bury the output of a notebook. |

## Reference of `explain`

| Parameter | Default | Accepted values | Rationale |
| --- | --- | --- | --- |
| `x` | required | Array shaped `(n_timestamps,)`, `(n_channels, n_timestamps)` or `(1, n_channels, n_timestamps)`. | The instance to explain. Segmentation applies to it alone, so a ranking describes one decision. |
| `segmentation` | `None` | `None` or a ready `Segmentation`. | Partition to rank. It is computed from the settings of the explainer when omitted, which is the usual case. A shared partition is how several instances become comparable. |
| `marginalization` | `None` | Any value accepted by the constructor. | Operator for this call alone. |
| `label` | `None` | `None` or a class position. | Class whose probability the search tracks. The predicted class is used by default, which makes the explanation independent of the ground truth. |
| `initialization` | `None` | `"greedy"`, `"sequential"`, `"random"`, or a sequence of positions. | Starting ranking for this call alone. |
| `max_iterations` | `None` | `None` or a non-negative integer. | Budget of proposed swaps for this call alone. Zero evaluates the initial ranking without searching. |
| `patience` | `None` | `None` or a positive integer. | Swaps without improvement tolerated for this call alone. |
| `n_restarts` | `None` | `None` or an integer of at least zero. | Restarts granted for this call alone. |
| `random_state` | `None` | `None`, an integer, or a `numpy` `Generator`. | Seed for this call alone. |
| `verbose` | `None` | `None`, `True`, `False`. | Whether a progress bar follows this search. The setting of the explainer applies when it is left out. |

`explain_batch` runs one search per instance and `score_ranking` evaluates a ranking without searching.

## Reference of `SOFIExplanation`

| Attribute | Meaning |
| --- | --- |
| `ranking`, `order`, `segments` | Segments from the most to the least important, by name, by position and as objects |
| `top(k)` | First `k` segments of the ranking |
| `morf_scores`, `lerf_scores` | Probability of the explained class along the ranking and along its reverse |
| `ds` | Degradation score, namely the area between the two curves, which the search maximizes |
| `baseline_probability` | Probability of the explained class before any perturbation, the first point of both curves |
| `scores` | Alias of `morf_scores` |
| `auc_morf`, `auc_lerf` | Mean height of each curve |
| `drops` | Degradation attributable to each step of the MoRF curve |
| `sparsity_point`, `sparsity_rate`, `sparsity_probability` | Where the MoRF order changes the predicted class, and how confident the model was there |
| `lerf_sparsity_point`, `lerf_sparsity_rate` | Where the reversed order changes it, which should be far later |
| `importances` | Rank based importance in the unit interval, with the noise region excluded |
| `noise_onset`, `noise_features` | Start of the region where the probability recovers, and the segments it holds |
| `interpretation`, `class_directed` | What this ranking means, given the operator that produced it |
| `modularity_gap` | Departure from the assumption of Theorem 3 |
| `states` | The series along the cumulative marginalization, one row per step, always carried |
| `statistics` | Every figure of the run gathered in one dictionary |
| `summary(top_k=None)` | Report meant to be printed, listing the whole ranking, also returned by `print(explanation)` |
| `to_frame()` | Tabular view of the ranking, one row per marginalization step |
| `plot_segmentation()`, `plot_degradation_curves()`, `plot_explanation()`, `plot_marginalization()`, `plot_animation()` | The figures described under "Drawing an explanation" |
| `source` | The explainer that produced the result, consulted by `plot_animation` and dropped when the explanation is pickled |
| `history` | One record per iteration of the search |
| `n_iterations`, `n_restarts_used`, `n_evaluations`, `n_model_calls` | Counters describing the run |

## Reference of the figures

Every function below takes `save_path` and `dpi`, which write the figure, and `theme` and `font_scale`, which govern its appearance without touching any other figure. The remaining arguments are those of the panels each one draws.

| Function | Principal arguments |
| --- | --- |
| `plot_segmentation(instance, segmentation)` | `style` (`"dots"` or `"lines"`), `channel`, `color`, `marker_color`, `linewidth`, `markersize`, `title`, `figsize`, `ax` |
| `plot_degradation_curves(explanation)` | `morf_color`, `lerf_color`, `area_color`, `linewidth`, `markersize`, `annotate_ds`, `legend`, `mark_sparsity`, `title`, `figsize`, `ax` |
| `plot_explanation(explanation)` | `channel`, `series_title`, `curve_title`, `figsize`, `wspace`, and the settings of the two panels it holds |
| `plot_marginalization(explanation)` | `steps`, `ncols`, `panel_size`, `wspace`, `channel`, `color`, `shade_color`, `linewidth`, `title` |
| `plot_marginalized_series(explanation)` | `series`, `shaded`, `shade_color`, `probability`, `limits`, `channel`, `title`, `ax` |
| `animate_marginalization(explanation, explainer)` | `step_ms`, `start_hold_ms`, `phase_hold_ms`, `end_hold_ms`, `dpi`, `progress`, `wspace`, and the colors of the two panels |
| `plot_explanation_grid`, `plot_degradation_grid`, `plot_segmentation_grid` | `ncols`, `panel_size`, `figsize`, `wspace`, `title`, and the settings of the panels they hold |

`steps` of `plot_marginalization` names the steps drawn, counted from zero for the untouched series, and six steps spread over the whole marginalization are chosen when it is omitted. `wspace` sets the horizontal space between the panels of any figure that holds several of them, as a fraction of the width of one panel, and its default keeps a panel clear of the axis label of its neighbor. `channel` selects the channel drawn from a multivariate series wherever it appears. `ax` places a figure inside a layout assembled elsewhere, and a new figure is created when it is left out.

## Advanced building blocks

`SOFIExplainer` is a thin orchestrator over five pieces, all importable from `sofits` for anyone who wants to reuse part of the machinery outside the explainer, for instance to plug a custom search into the existing objective.

| Name | What it does |
| --- | --- |
| `ClassifierWrapper` | Reduces any classifier to one probability function, resolving the output convention and the channel layout. |
| `Segmentation` | Holds the breakpoints, the mask and the segment metadata, and reports them through `describe()`. |
| `Perturbation` | Computes and applies the values that neutralize a segment, under any of the fourteen operators. |
| `Objective` | Turns a ranking into a `RankingEvaluation` by driving the model through cumulative marginalization, and holds the reference sweep the search starts from. |
| `hill_climbing`, `greedy_ranking` | The two searches, decoupled from the objective. Hill climbing accepts any `evaluate` callable that returns an object exposing `.order` and `.ds`. |
| `curve_auc`, `degradation_score`, `modularity_gap` | The scoring functions applied to any pair of MoRF and LeRF curves, independent of how they were produced. |
| `RankingEvaluation` | The dataclass passed between the objective and the search. |
| `noise_onset` | The free function behind `SOFIExplanation.noise_onset`, usable directly on any curve. |

Most users never need to import these directly. `SOFIExplainer.explain` and `score_ranking` cover the ordinary usage.

## Citation

```bibtex
@article{grau2026sofits,
  title   = {Sparseness-Optimized Feature Importance for Time Series Classification},
  author  = {Grau, Isel and N{\'a}poles, Gonzalo and Jastrzebska, Agnieszka and Salgueiro, Yamisleydi},
  journal = {IEEE Access},
  volume  = {14},
  pages   = {29874--29893},
  year    = {2026},
  doi     = {10.1109/ACCESS.2026.3667092}
}

@inproceedings{grau2024sofi,
  title     = {Sparseness-Optimized Feature Importance},
  author    = {Grau, Isel and N{\'a}poles, Gonzalo},
  booktitle = {Explainable Artificial Intelligence. xAI 2024},
  series    = {Communications in Computer and Information Science},
  volume    = {2154},
  pages     = {393--415},
  publisher = {Springer},
  year      = {2024},
  doi       = {10.1007/978-3-031-63797-1_20}
}
```

## License

MIT. See `LICENSE`.
