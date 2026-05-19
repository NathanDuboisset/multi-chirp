"""Back-compat surface for `building.scaling`.

Shared training primitives live in `building.training`; scaling-experiment-
specific code lives in `building.scaling.runner`. Existing notebook imports
like `from building.scaling import load_dataset_catalog, run_experiments`
keep working because every public name from both modules is re-exported here.
"""

from __future__ import annotations

from building.training import (
    FIT_VERBOSE,
    PREFETCH_BUFFER,
    SHUFFLE_BUFFER_CAP,
    ClassEntry,
    ClassSplit,
    DatasetCatalog,
    DatasetMeta,
    RunMetrics,
    build_dataset_from_catalog,
    build_grouped_dataset_from_catalog,
    cleanup_waveform_cache,
    collect_predictions,
    compute_metrics,
    configure_tf_for_long_runs,
    load_dataset_catalog,
    model_factory,
)

from .runner import (
    BaseRunResult,
    BaselineRunResult,
    BaselineSummary,
    RunResult,
    ScalingResult,
    ScalingRunConfig,
    load_results,
    plot_summary,
    print_baselines,
    run_experiments,
    summarize_results,
)

__all__ = [
    # from training
    "ClassSplit",
    "ClassEntry",
    "DatasetCatalog",
    "DatasetMeta",
    "RunMetrics",
    "FIT_VERBOSE",
    "SHUFFLE_BUFFER_CAP",
    "PREFETCH_BUFFER",
    "configure_tf_for_long_runs",
    "cleanup_waveform_cache",
    "load_dataset_catalog",
    "model_factory",
    "build_dataset_from_catalog",
    "build_grouped_dataset_from_catalog",
    "collect_predictions",
    "compute_metrics",
    # from runner
    "ScalingRunConfig",
    "BaseRunResult",
    "BaselineRunResult",
    "ScalingResult",
    "RunResult",
    "BaselineSummary",
    "load_results",
    "run_experiments",
    "print_baselines",
    "summarize_results",
    "plot_summary",
]
