---
layout: post
title: "XGBoost 3.3.0 Release"
date:   2026-7-21
author: Jiaming Yuan and Rory Mitchell
categories:
comments: true
---

We are pleased to announce XGBoost 3.3.0. This release adds expectile regression,
enables categorical feature support by default, expands SHAP support for
vector-leaf multi-output trees, and includes a broad set of performance and
scalability improvements across CPU, GPU, distributed training, Python, JVM, and
the build system. In particular, GPU quantile sketching uses significantly less
memory for training.

XGBoost 3.3.0 is available from the usual package channels. The complete release
notes are available in the
[XGBoost documentation](https://xgboost.readthedocs.io/en/latest/changes/v3.3.0.html),
and release artifacts are available on
[GitHub](https://github.com/dmlc/xgboost/releases/tag/v3.3.0).

<p>
  <img src="/images/blog/xgboost-3-3-larger-datasets.png" alt="Train even larger datasets with XGBoost 3.3" style="display: block; max-width: 480px; width: 100%; height: auto; margin: 0 auto;">
</p>

## Faster quantile sketching and distributed training

Quantile sketching received a substantial refactor in 3.3.0. The internals are
simpler, the weighted quantile sketch implementation is improved, and GPU
quantile sketching uses significantly less memory.

These changes matter because quantile construction is a central part of the
histogram-based training pipeline. Reducing memory use and simplifying the
implementation improves robustness for large datasets, wide datasets, and GPU
workloads where memory pressure often decides whether a job fits.

Distributed quantile construction is also more efficient in this release, using
tree reductions to improve scalability. XGBoost no longer requires all-to-all
collective connections for distributed training, and users can now customize the
worker port when deploying distributed jobs.

## Expectile regression

This release adds expectile regression, giving practitioners another tool for
risk-sensitive forecasting, uncertainty-aware modeling, and applications where
overprediction and underprediction carry different costs. It is available
through the new `reg:expectileerror` objective, the `expectile` metric, and the
`expectile_alpha` parameter.

Expectile regression is useful when users want an asymmetric squared-loss
analogue of quantile regression. It gives a smooth objective for estimating
different parts of the conditional response distribution, while retaining a
squared-error style penalty.

Multiple expectiles are supported, allowing users to train models for several
asymmetric targets in a single configuration. Unlike quantile regression,
expectile curves do not suffer from crossing, which makes them easier to use
when estimating multiple levels of the response distribution together. This
complements XGBoost's existing regression objectives.

## Categorical features are on by default

Categorical feature support is now enabled by default. Users no longer need to
set `enable_categorical=True` in the common case. The parameter remains available
for users who need to disable categorical handling explicitly.

This change reflects the maturity of categorical support in XGBoost and makes
the Python and dataframe workflows a little more natural. Dataframes with
categorical columns can be passed into XGBoost without an extra opt-in step,
making the default behavior match what many users already expect from modern
tabular machine learning libraries.

The CPU `hist` tree method also gained one-hot categorical split support for the
work-in-progress vector-leaf model. Vector-leaf trees are still under active
development, but this release continues to close gaps in the feature set.

## SHAP support for vector-leaf trees

XGBoost 3.3.0 expands SHAP support for vector-leaf multi-output trees. Exact SHAP
contribution and interaction prediction are now available for vector-leaf
multi-output trees on both CPU and GPU.

This is an important step for multi-output models. Vector-leaf trees can model
relationships across targets directly by storing a vector of weights in each
leaf, but users also need the same interpretability tools they rely on for
single-output models. With this release, multi-output workflows gain a more
complete explanation story.

The TreeSHAP implementation has also been updated for improved numerical
stability and faster execution with `QuadratureTreeSHAP` [\[1\]](#references).

## CPU and GPU performance improvements

On CPUs, histogram building has been optimized for wide datasets with column
block tiling. XGBoost can also detect CPU cache sizes through Linux sysfs on
aarch64, improving its ability to choose efficient execution strategies on
modern ARM platforms.

On GPUs, sampling now uses Philox for faster random number generation. This
release also continues the wider work on reducing GPU memory pressure and
improving the efficiency of the training pipeline.

Training with sampling is now more reproducible within a single session. XGBoost
uses a local random number generator and serializes the RNG state in the model
configuration, so pickled models can retain that state across save/load
boundaries. This helps when training multiple sampled models in the same process.

## Python and Spark updates

The minimum supported Python version is now Python 3.12. This lets the project
remove older compatibility paths and focus testing on actively supported Python
versions.

The Python package also adds PySpark support for Spark Connect ML. This improves
compatibility with the newer Spark client-server workflow and keeps the XGBoost
Spark integration moving with the broader Spark ecosystem.

The release includes dataframe validation improvements as well. XGBoost now
requires `Enum` support from Polars and validates that pandas column names are
unique, helping users catch ambiguous feature inputs earlier.

Several Python fixes and cleanups are included: default verbose behavior is now
consistent between `XGBClassifier.fit` and `XGBRegressor.fit`, `python -OO`
crashes from missing docstrings have been fixed, boolean indicator features are
handled in `trees_to_dataframe`, and validation messages around feature
information and deprecated functions have been improved.

## JVM, build, and platform updates

The JVM packages now document Spark 4.0 compatibility. Regressor and ranker
pipelines support columnar input, Java packages include automatic module names,
and `xgboost4j` supports FreeBSD.

On the build and platform side, XGBoost 3.3.0 adds support for Visual Studio
2026, updates CUDA Toolkit support, and fixes a number of CMake, macOS, and
packaging issues. Linux packaging now uses a versioned shared object.

## Deprecation notice

The `gblinear` booster is deprecated in XGBoost 3.3.0 and will be removed in a
future release. The linear model feature in XGBoost has been a niche feature and
has not been as actively maintained as the tree algorithms. It also does not
support or align with many capabilities in modern XGBoost, such as categorical
splits, interaction constraints, monotone constraints, and TreeSHAP-style
explanations.

XGBoost is primarily a decision tree library, and we have decided to focus
development on the tree algorithms and on keeping the interface consistent for
the features users rely on most. Users relying on `gblinear` should start
planning migrations to other estimators or workflows.

## Upgrade

For Python users, upgrade with:

```sh
pip install --upgrade xgboost
```

For conda users:

```sh
conda install -c conda-forge xgboost
```

Please see the
[installation guide](https://xgboost.readthedocs.io/en/latest/install.html) for
platform-specific instructions, and the
[3.3.0 release notes](https://xgboost.readthedocs.io/en/latest/changes/v3.3.0.html)
for the full list of changes.

## References

[1] Wettenstein R, Mitchell R, Yu P. (2026) Quadrature-TreeSHAP: Depth-Independent TreeSHAP and Shapley Interactions. [arXiv:2605.04497](https://arxiv.org/abs/2605.04497).

Many thanks to everyone who contributed code, reviews, testing, documentation,
bug reports, and release work for XGBoost 3.3.0.
