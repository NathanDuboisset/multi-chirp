from __future__ import annotations

from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Union, Annotated
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
import pyrootutils
import tensorflow as tf

ROOT = pyrootutils.setup_root(
    search_from=__file__,
    indicator="pyproject.toml",
    pythonpath=True,
    dotenv=True,
)

FIT_VERBOSE = 2
# Per-class shuffle buffer cap. Each unit ≈ 47872 × 4 B = 192 KB of resident
# RAM, held for the duration of the epoch. At k=10 classes the total is
# SHUFFLE_BUFFER_CAP × 10 × 192 KB. Keep small; the file-level shuffle in
# dataset.py already provides bulk randomness.
SHUFFLE_BUFFER_CAP = 256
PREFETCH_BUFFER = 2


def configure_tf_for_long_runs() -> None:
    try:
        tf.config.optimizer.set_jit(False)
    except Exception:
        pass
    try:
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass


configure_tf_for_long_runs()


class AugmentConfig(BaseModel):
    """Waveform-level augmentations applied only to the training split,
    *before* the mel spectrogram is computed."""

    enabled: bool = True
    polarity_flip_prob: float = 0.5
    time_shift_max_seconds: float = 0.1
    gaussian_std: float = 0.005
    gain_min: float = 0.9
    gain_max: float = 1.1


class ScalingRunConfig(BaseModel):
    collection: str
    build_model: str
    epochs: int
    patience: int
    batch_size: int
    seed: int
    threshold: float
    models_dir: Path
    results_file: Path
    input_repr: Literal["time", "mel"] = "time"
    augment: AugmentConfig = Field(default_factory=AugmentConfig)
    # Per scaling sample, fold this many random unchosen target species into
    # the "other" class, sampled uniformly alongside the existing non_target
    # bucket. Keeps "other" diversity comparable across k values.
    extras_into_other: int = 4


@dataclass
class ClassSplit:
    """Per-class, per-split cached unbatched dataset (features only, no label)."""

    ds: tf.data.Dataset
    count: int


@dataclass
class ClassEntry:
    name: str
    global_idx: int
    train: ClassSplit
    val: ClassSplit
    test: ClassSplit


class DatasetCatalog(BaseModel):
    """Replaces DatasetArrays: holds per-class cached TF datasets instead of numpy blobs."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    class_names: list[str]
    entries: list[ClassEntry]  # parallel to class_names


class RunMetrics(BaseModel):
    recall_mean: float
    recall_std: float
    precision_mean: float
    precision_std: float
    f1_mean: float
    f1_std: float
    top1_accuracy: float
    loss: float


class BaseRunResult(BaseModel):
    collection: str
    build_model: str
    epochs_trained: int
    model_path: str
    timestamp: str
    config: dict[str, Any]


class BaselineRunResult(BaseRunResult):
    run_type: Literal["baseline"]
    target_class: str
    target_idx: int
    metrics: RunMetrics


class ScalingResult(BaseRunResult):
    run_type: Literal["scaling"]
    k: int
    sample_idx: int
    chosen_class_indices: list[int]
    chosen_classes: list[str]
    n_labels: int
    extra_other_indices: list[int] = []
    extra_other_classes: list[str] = []
    metrics: RunMetrics


RunResult = Annotated[
    Union[BaselineRunResult, ScalingResult], Field(discriminator="run_type")
]
_run_result_adapter = TypeAdapter(RunResult)


class BaselineSummary(BaseModel):
    recall: float
    precision: float
    f1: float
    top1_accuracy: float
    loss: float


@dataclass
class DatasetMeta:
    n_each: int
    n_classes: int


_CACHE_BATCH = 64  # batch size used only during cache-population pass


def _feature_config_hash() -> str:
    """Hash the waveform-cache config (sample rate + target length).

    The cache stores raw waveform now; mel STFT runs at training time, so
    changing mel parameters does NOT invalidate the cache."""
    from building.utils import SAMPLE_RATE, TARGET_AUDIO_LEN_MEL

    cfg: dict[str, Any] = {
        "sample_rate": SAMPLE_RATE,
        "target_audio_len": TARGET_AUDIO_LEN_MEL,
    }
    payload = json.dumps(cfg, sort_keys=True).encode()
    return hashlib.sha1(payload).hexdigest()[:8]


def load_dataset_catalog(
    collection: str,
    input_repr: Literal["time", "mel"] = "time",
    # batch_size / seed kept as kwargs so old call-sites don't break
    batch_size: int = 32,  # noqa: ARG001 – ignored, kept for API compat
    seed: int = 42,  # noqa: ARG001 – ignored, kept for API compat
) -> DatasetCatalog:
    """Build a per-class cached DatasetCatalog.

    On the first call for a given (collection, input_repr) the audio files are
    read, features are computed and written to
    ``<root>/.cache/<collection>/<input_repr>_<cfghash>/<split>/<class>/``.
    The cfg-hash suffix invalidates the cache automatically when feature
    extraction parameters change. All subsequent calls re-use the on-disk
    cache — no audio decoding, no numpy blobs in RAM.
    """
    from building.utils import SAMPLE_RATE, fix_audio_length_mel

    keras = tf.keras
    dataset_root = ROOT / "datasets" / collection
    cfg_hash = _feature_config_hash()
    # Cache raw waveform (same shape regardless of input_repr) so augmentation
    # can run on the waveform *before* the mel STFT.
    cache_root = ROOT / ".cache" / collection / f"waveform_{cfg_hash}"
    _ = input_repr  # kept for API compat; no longer affects the cache.

    def _feature_fn(audio_batch: tf.Tensor) -> tf.Tensor:
        # fix_audio_length_mel returns [B, T] (no channel dim) — exactly
        # what we want for the unified waveform cache.
        return fix_audio_length_mel(audio_batch)
    _ = SAMPLE_RATE  # consumed by audio_dataset_from_directory below

    # Discover classes (sorted so global_idx is stable across calls).
    training_dir = dataset_root / "training"
    class_names = sorted(d.name for d in training_dir.iterdir() if d.is_dir())

    def _build_class_split(split_dir_name: str, class_name: str) -> ClassSplit:
        split_dir = dataset_root / split_dir_name
        cache_dir = cache_root / split_dir_name / class_name
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = str(cache_dir / "data")
        count_file = cache_dir / "count.json"
        cache_index = Path(cache_path + ".index")

        # Remove stale lockfiles left by interrupted previous runs.
        for lockfile in cache_dir.glob("*.lockfile"):
            lockfile.unlink(missing_ok=True)

        # Load only this class's audio files; label (always 0) is discarded.
        ds_raw = keras.utils.audio_dataset_from_directory(
            split_dir,
            labels="inferred",
            class_names=[class_name],
            sampling_rate=SAMPLE_RATE,
            batch_size=_CACHE_BATCH,
            shuffle=False,
        )
        ds: tf.data.Dataset = (
            ds_raw.map(lambda x, _: _feature_fn(x), num_parallel_calls=tf.data.AUTOTUNE)
            .unbatch()
            .cache(cache_path)
        )

        cache_complete = count_file.exists() and cache_index.exists()
        if not cache_complete:
            # Wipe any partial cache shards so TF starts from a clean slate.
            for stale in cache_dir.glob("data*"):
                stale.unlink(missing_ok=True)
            count_file.unlink(missing_ok=True)

        if cache_complete:
            count = json.loads(count_file.read_text())["count"]
        else:
            print(f"  caching {split_dir_name}/{class_name} ...", end=" ", flush=True)
            count = sum(1 for _ in ds)  # triggers the cache-write pass
            count_file.write_text(json.dumps({"count": count}))
            print(f"{count} samples")

        return ClassSplit(ds=ds, count=count)

    entries: list[ClassEntry] = []
    for global_idx, class_name in enumerate(class_names):
        entries.append(
            ClassEntry(
                name=class_name,
                global_idx=global_idx,
                train=_build_class_split("training", class_name),
                val=_build_class_split("validation", class_name),
                test=_build_class_split("testing", class_name),
            )
        )

    return DatasetCatalog(class_names=class_names, entries=entries)


# Backward-compat alias so old notebooks keep working unchanged.
load_full_arrays = load_dataset_catalog


def model_factory(
    name: str, input_repr: Literal["time", "mel"] = "time"
) -> Callable[[int], tf.keras.Model]:
    if input_repr == "time":
        from building.time_models import get_time_model, TARGET_AUDIO_LEN

        time_model = get_time_model(name)
        return lambda n_classes: time_model(n_classes, TARGET_AUDIO_LEN)
    elif input_repr == "mel":
        from building.mel_models import get_mel_model

        mel_model = get_mel_model(name)
        return lambda n_classes: mel_model(n_classes)
    raise ValueError(f"Unknown model: {name}")


def _make_augment_fn(cfg: AugmentConfig) -> Callable[[tf.Tensor], tf.Tensor]:
    """Build a per-sample waveform augmentation function. Input/output: [T]."""
    from building.utils import SAMPLE_RATE

    max_shift = int(cfg.time_shift_max_seconds * SAMPLE_RATE)
    flip_prob = float(cfg.polarity_flip_prob)
    gain_lo = float(cfg.gain_min)
    gain_hi = float(cfg.gain_max)
    noise_std = float(cfg.gaussian_std)

    def augment(audio: tf.Tensor) -> tf.Tensor:
        # polarity flip
        if flip_prob > 0:
            flip = tf.cast(
                tf.random.uniform([], 0.0, 1.0) < flip_prob, audio.dtype
            )
            audio = audio * (1.0 - 2.0 * flip)
        # random gain
        if gain_lo != 1.0 or gain_hi != 1.0:
            audio = audio * tf.random.uniform(
                [], gain_lo, gain_hi, dtype=audio.dtype
            )
        # additive gaussian
        if noise_std > 0:
            audio = audio + tf.random.normal(
                tf.shape(audio), stddev=noise_std, dtype=audio.dtype
            )
        # time shift (wrap-around; <0.1 s on 3 s clip is negligible)
        if max_shift > 0:
            shift = tf.random.uniform(
                [], -max_shift, max_shift + 1, dtype=tf.int32
            )
            audio = tf.roll(audio, shift=shift, axis=0)
        return audio

    return augment


def _build_dataset_from_catalog(
    catalog: DatasetCatalog,
    chosen_idxs: list[int],
    non_target_idx: int,
    batch_size: int,
    rng: np.random.Generator,
    split: Literal["train", "val", "test"],
    input_repr: Literal["time", "mel"] = "time",
    augment: AugmentConfig | None = None,
    extra_other_idxs: list[int] | None = None,
) -> tuple[tf.data.Dataset, DatasetMeta]:
    """Build a batched TF dataset for one experiment from the on-disk catalog.

    Train: each class is shuffled independently and repeated, then interleaved
    via ``sample_from_datasets`` so every batch is class-balanced regardless of
    class size. Epoch length is bounded to ``min(counts) * n_classes`` so smaller
    classes see all their data and larger classes see a fresh subset each epoch
    (reshuffle_each_iteration=True).
    Val/Test: full concatenation, no resampling.

    The "other" class is the union of ``non_target_idx`` and any
    ``extra_other_idxs`` (unchosen target species folded in). For training the
    sub-buckets are mixed uniformly via ``sample_from_datasets``; for val/test
    they are simply concatenated under the same label.
    No numpy arrays are allocated; RAM usage is O(batch_size × sample_size).
    """
    extras = list(extra_other_idxs or [])
    other_idxs = [non_target_idx] + extras

    def _get_split(e: ClassEntry) -> ClassSplit:
        if split == "train":
            return e.train
        if split == "val":
            return e.val
        return e.test

    target_class_splits = [_get_split(catalog.entries[i]) for i in chosen_idxs]
    other_class_splits = [_get_split(catalog.entries[i]) for i in other_idxs]
    target_counts = [cs.count for cs in target_class_splits]
    other_counts = [cs.count for cs in other_class_splits]

    if any(c == 0 for c in target_counts):
        bad = [chosen_idxs[i] for i, c in enumerate(target_counts) if c == 0]
        raise ValueError(f"Class(es) {bad} have no samples in split '{split}'")
    if any(c == 0 for c in other_counts):
        bad = [other_idxs[i] for i, c in enumerate(other_counts) if c == 0]
        raise ValueError(
            f"Other-bucket class(es) {bad} have no samples in split '{split}'"
        )

    n_classes = len(chosen_idxs) + 1  # targets + 1 merged "other"
    other_label = len(chosen_idxs)
    do_shuffle = split == "train"

    if do_shuffle:
        n_each = min(target_counts)
        epoch_len = n_each * n_classes
        parts: list[tf.data.Dataset] = []
        for local_label, (cs, count) in enumerate(
            zip(target_class_splits, target_counts)
        ):
            lbl = tf.constant(local_label, dtype=tf.int32)
            shuffle_buf = min(count, SHUFFLE_BUFFER_CAP)
            seed = int(rng.integers(0, 2**31 - 1))
            ds = (
                cs.ds.shuffle(
                    shuffle_buf, seed=seed, reshuffle_each_iteration=True
                )
                .repeat()
                .map(lambda x, label=lbl: (x, label))
            )
            parts.append(ds)

        # Build the merged "other" stream: each sub-bucket shuffled+repeated,
        # then mixed uniformly so the audioset/empty/extra-species sources are
        # represented evenly within the "other" share of every batch.
        other_sub_parts: list[tf.data.Dataset] = []
        for cs, count in zip(other_class_splits, other_counts):
            shuffle_buf = min(count, SHUFFLE_BUFFER_CAP)
            seed = int(rng.integers(0, 2**31 - 1))
            other_sub_parts.append(
                cs.ds.shuffle(
                    shuffle_buf, seed=seed, reshuffle_each_iteration=True
                ).repeat()
            )
        if len(other_sub_parts) == 1:
            other_stream = other_sub_parts[0]
        else:
            sub_weights = [1.0 / len(other_sub_parts)] * len(other_sub_parts)
            other_stream = tf.data.Dataset.sample_from_datasets(
                other_sub_parts,
                weights=sub_weights,
                seed=int(rng.integers(0, 2**31 - 1)),
                stop_on_empty_dataset=False,
            )
        olbl = tf.constant(other_label, dtype=tf.int32)
        parts.append(other_stream.map(lambda x, label=olbl: (x, label)))

        weights = [1.0 / n_classes] * n_classes
        combined = tf.data.Dataset.sample_from_datasets(
            parts,
            weights=weights,
            seed=int(rng.integers(0, 2**31 - 1)),
            stop_on_empty_dataset=False,
        ).take(epoch_len)
    else:
        n_each = max(max(target_counts), sum(other_counts))
        parts = []
        for local_label, cs in enumerate(target_class_splits):
            lbl = tf.constant(local_label, dtype=tf.int32)
            parts.append(cs.ds.map(lambda x, label=lbl: (x, label)))
        olbl = tf.constant(other_label, dtype=tf.int32)
        other_ds = other_class_splits[0].ds.map(
            lambda x, label=olbl: (x, label)
        )
        for cs in other_class_splits[1:]:
            other_ds = other_ds.concatenate(
                cs.ds.map(lambda x, label=olbl: (x, label))
            )
        parts.append(other_ds)
        combined = parts[0]
        for p in parts[1:]:
            combined = combined.concatenate(p)

    # Per-sample waveform augmentation (train split only). Runs *before* the
    # mel STFT so polarity flip / shift / noise affect the spectrogram.
    if do_shuffle and augment is not None and augment.enabled:
        aug_fn = _make_augment_fn(augment)
        combined = combined.map(
            lambda x, y: (aug_fn(x), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    combined = combined.batch(batch_size)

    if input_repr == "mel":
        from building.utils import (
            create_log_mel_spectrogram,
            TARGET_FRAMES_MEL,
            NUM_MEL_BINS_MEL,
        )

        def _to_features(xb: tf.Tensor, yb: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
            spec = create_log_mel_spectrogram(xb)
            spec = tf.ensure_shape(
                spec, [None, TARGET_FRAMES_MEL, NUM_MEL_BINS_MEL]
            )
            return tf.expand_dims(spec, -1), tf.one_hot(
                yb, n_classes, dtype=tf.float32
            )
    else:

        def _to_features(xb: tf.Tensor, yb: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
            return tf.expand_dims(xb, -1), tf.one_hot(
                yb, n_classes, dtype=tf.float32
            )

    combined = combined.map(
        _to_features, num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(PREFETCH_BUFFER)
    return combined, DatasetMeta(n_each=int(n_each), n_classes=n_classes)


def _collect_predictions(
    model: tf.keras.Model, ds: tf.data.Dataset
) -> tuple[np.ndarray, np.ndarray]:
    y_true, y_pred = [], []
    for xb, yb in ds:
        y_true.append(yb.numpy())
        y_pred.append(model(xb, training=False).numpy())
    return np.concatenate(y_true), np.concatenate(y_pred)


def _compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: float, loss: float
) -> RunMetrics:
    recalls, precisions, f1s = [], [], []
    for c in range(y_true.shape[1]):
        true_c = y_true[:, c] >= 0.5
        pred_c = y_pred[:, c] >= threshold
        rec = float(np.mean(pred_c[true_c])) if true_c.any() else None
        prec = float(np.mean(true_c[pred_c])) if pred_c.any() else None
        if rec is not None:
            recalls.append(rec)
        if prec is not None:
            precisions.append(prec)
        if rec is not None and prec is not None and (rec + prec) > 0:
            f1s.append(2 * rec * prec / (rec + prec))
    return RunMetrics(
        recall_mean=float(np.mean(recalls)) if recalls else 0.0,
        recall_std=float(np.std(recalls)) if recalls else 0.0,
        precision_mean=float(np.mean(precisions)) if precisions else 0.0,
        precision_std=float(np.std(precisions)) if precisions else 0.0,
        f1_mean=float(np.mean(f1s)) if f1s else 0.0,
        f1_std=float(np.std(f1s)) if f1s else 0.0,
        top1_accuracy=float(np.mean(np.argmax(y_true, 1) == np.argmax(y_pred, 1))),
        loss=loss,
    )


def load_results(results_file: Path) -> list[RunResult]:
    if not results_file.exists():
        return []
    with results_file.open("r", encoding="utf-8") as f:
        return [_run_result_adapter.validate_json(line) for line in f if line.strip()]


def run_experiments(
    catalog: DatasetCatalog,
    config: ScalingRunConfig,
    n_samples: int = 3,
    k_values: Iterable[int] = range(2, 6),
    *,
    run_baseline: bool = True,
    run_scaling: bool = True,
) -> list[RunResult]:
    rng = np.random.default_rng(config.seed)
    model_builder = model_factory(config.build_model, config.input_repr)
    config.models_dir.mkdir(parents=True, exist_ok=True)
    config.results_file.parent.mkdir(parents=True, exist_ok=True)

    if "non_target" not in catalog.class_names:
        raise ValueError("Dataset must contain a 'non_target' class.")
    non_target_idx = catalog.class_names.index("non_target")
    existing = load_results(config.results_file)
    produced: list[RunResult] = []

    def is_done(run_type: Literal["baseline", "scaling"], **keys: object) -> bool:
        return any(
            r.run_type == run_type
            and all(getattr(r, k, None) == v for k, v in keys.items())
            for r in existing
        )

    def fit_eval(
        chosen_idxs: list[int],
        model_name: str,
        run_type: Literal["baseline", "scaling"],
        extra: dict[str, Any],
        extra_other_idxs: list[int] | None = None,
    ) -> None:
        # Drop any state lingering from a previous experiment before we start.
        tf.keras.backend.clear_session()
        gc.collect()
        model = history = None
        train_ds = val_ds = test_ds = None
        try:
            train_ds, meta = _build_dataset_from_catalog(
                catalog,
                chosen_idxs,
                non_target_idx,
                config.batch_size,
                rng,
                split="train",
                input_repr=config.input_repr,
                augment=config.augment,
                extra_other_idxs=extra_other_idxs,
            )
            val_ds, _ = _build_dataset_from_catalog(
                catalog,
                chosen_idxs,
                non_target_idx,
                config.batch_size,
                rng,
                split="val",
                input_repr=config.input_repr,
                augment=None,
                extra_other_idxs=extra_other_idxs,
            )
            test_ds, _ = _build_dataset_from_catalog(
                catalog,
                chosen_idxs,
                non_target_idx,
                config.batch_size,
                rng,
                split="test",
                input_repr=config.input_repr,
                augment=None,
                extra_other_idxs=extra_other_idxs,
            )
            print(
                f"[{run_type}] training_samples={meta.n_each} n_classes={meta.n_classes}"
            )

            model = model_builder(meta.n_classes)
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=config.epochs,
                verbose=FIT_VERBOSE,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss",
                        patience=config.patience,
                        restore_best_weights=True,
                    )
                ],
            )
            test_loss = float(model.evaluate(test_ds, verbose=0)[0])  # ty:ignore[not-subscriptable]
            y_true, y_pred = _collect_predictions(model, test_ds)
            m = _compute_metrics(y_true, y_pred, config.threshold, test_loss)

            model_path = (
                config.models_dir
                / f"scaling_{len(chosen_idxs)}"
                / f"{model_name}.keras"
            )
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(model_path)

            common: dict[str, Any] = {
                "collection": config.collection,
                "build_model": config.build_model,
                "epochs_trained": len(history.history.get("loss", [])),
                "model_path": str(model_path),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "config": config.model_dump(mode="json"),
                **extra,
            }
            res_cls = BaselineRunResult if run_type == "baseline" else ScalingResult
            res = res_cls(run_type=run_type, metrics=m, **common)  # ty:ignore[invalid-argument-type]
            with config.results_file.open("a", encoding="utf-8") as f:
                f.write(res.model_dump_json() + "\n")
            existing.append(res)
            produced.append(res)
        finally:
            # Drop dataset references *before* clear_session so iterator /
            # shuffle-buffer state isn't pinned across the session teardown.
            del model, history, train_ds, val_ds, test_ds
            tf.keras.backend.clear_session()
            gc.collect()

    if run_baseline:
        species_classes = [
            (i, n) for i, n in enumerate(catalog.class_names) if n != "non_target"
        ]
        for target_idx, target_name in species_classes:
            if is_done("baseline", target_class=target_name):
                continue
            print(f"[baseline] target={target_name}")
            fit_eval(
                [target_idx],
                f"sample_{target_idx}",
                "baseline",
                {"target_class": target_name, "target_idx": target_idx},
            )

    if run_scaling:
        target_idxs_pool = [
            i for i, n in enumerate(catalog.class_names) if n != "non_target"
        ]
        for k in k_values:
            if k < 2 or k > len(target_idxs_pool):
                continue
            for sample_idx in range(n_samples):
                if is_done("scaling", k=k, sample_idx=sample_idx):
                    continue
                chosen_idxs = rng.choice(
                    target_idxs_pool, size=k, replace=False
                ).tolist()
                unchosen = [i for i in target_idxs_pool if i not in chosen_idxs]
                n_extras = min(config.extras_into_other, len(unchosen))
                extra_other_idxs = (
                    rng.choice(unchosen, size=n_extras, replace=False).tolist()
                    if n_extras > 0
                    else []
                )
                chosen_names = [catalog.class_names[i] for i in chosen_idxs]
                extra_names = [catalog.class_names[i] for i in extra_other_idxs]
                print(
                    f"[scaling] k={k} sample={sample_idx + 1}/{n_samples} "
                    f"chosen={chosen_names} extras_into_other={extra_names}"
                )
                fit_eval(
                    chosen_idxs,
                    f"sample_{'_'.join(str(i) for i in chosen_idxs)}",
                    "scaling",
                    {
                        "k": k,
                        "sample_idx": sample_idx,
                        "chosen_class_indices": chosen_idxs,
                        "chosen_classes": chosen_names,
                        "n_labels": k + 1,
                        "extra_other_indices": extra_other_idxs,
                        "extra_other_classes": extra_names,
                    },
                    extra_other_idxs=extra_other_idxs,
                )

    return produced


def print_baselines(catalog: DatasetCatalog, results_file: Path) -> None:
    try:
        df = pd.read_json(results_file, lines=True)
    except Exception as e:
        print(f"Could not read results file: {e}")
        return

    baseline_df = df[df["run_type"] == "baseline"].copy()
    class_names = list(catalog.class_names)

    # Calculate the longest class name
    max_cls_len = max((len(str(cls)) for cls in class_names), default=0)
    col_width = max(max_cls_len + 2, 15)

    header = f"{'Target Class':<{col_width}} | {'Precision':>9} | {'Recall':>9} | {'Epochs':>6} | {'Timestamp'}"
    print(header)
    print("-" * len(header))

    for cls in class_names:
        cls_str = f"'{cls}'"
        matches = baseline_df[baseline_df["target_class"] == cls]

        if matches.empty:
            continue

        row = matches.iloc[-1]
        metrics = row["metrics"]

        prec = metrics["precision_mean"]
        prec_std = metrics["precision_std"]

        rec = metrics["recall_mean"]
        rec_std = metrics["recall_std"]

        f1 = metrics["f1_mean"]
        f1_std = metrics["f1_std"]

        epochs = row["epochs_trained"]
        timestamp = row["timestamp"]
        print(
            f"{cls_str:<{col_width}} | {prec:>9.4f} ± {prec_std:>9.4f} | {rec:>9.4f} ± {rec_std:>9.4f} | {f1:>9.4f} ± {f1_std:>9.4f} | {epochs:>6.0f} "
        )


def summarize_results(results_file: Path) -> tuple[BaselineSummary, pd.DataFrame]:
    rows = load_results(results_file)
    empty_df = pd.DataFrame(
        columns=[
            "k",
            "recall_mean",
            "recall_std",
            "precision_mean",
            "precision_std",
            "f1_mean",
            "f1_std",
            "top1_acc_mean",
            "top1_acc_std",
            "loss_mean",
            "loss_std",
        ]
    )
    if not rows:
        return BaselineSummary(
            recall=0.0, precision=0.0, f1=0.0, top1_accuracy=0.0, loss=0.0
        ), empty_df

    bl = [r for r in rows if isinstance(r, BaselineRunResult)]
    baseline = BaselineSummary(
        recall=float(np.mean([r.metrics.recall_mean for r in bl])) if bl else 0.0,
        precision=float(np.mean([r.metrics.precision_mean for r in bl])) if bl else 0.0,
        f1=float(np.mean([r.metrics.f1_mean for r in bl])) if bl else 0.0,
        top1_accuracy=float(np.mean([r.metrics.top1_accuracy for r in bl]))
        if bl
        else 0.0,
        loss=float(np.mean([r.metrics.loss for r in bl])) if bl else 0.0,
    )

    sc = [r for r in rows if isinstance(r, ScalingResult)]
    if not sc:
        return baseline, empty_df

    df = pd.DataFrame(
        [
            {
                "k": r.k,
                "recall_mean": r.metrics.recall_mean,
                "precision_mean": r.metrics.precision_mean,
                "f1_mean": r.metrics.f1_mean,
                "top1_accuracy": r.metrics.top1_accuracy,
                "loss": r.metrics.loss,
            }
            for r in sc
        ]
    )
    summary = (
        df.groupby("k", as_index=False)
        .agg(
            recall_mean=("recall_mean", "mean"),
            recall_std=("recall_mean", "std"),
            precision_mean=("precision_mean", "mean"),
            precision_std=("precision_mean", "std"),
            f1_mean=("f1_mean", "mean"),
            f1_std=("f1_mean", "std"),
            top1_acc_mean=("top1_accuracy", "mean"),
            top1_acc_std=("top1_accuracy", "std"),
            loss_mean=("loss", "mean"),
            loss_std=("loss", "std"),
        )
        .fillna(0.0)
        .sort_values("k")
    )
    return baseline, summary


def plot_summary(
    summary_df: pd.DataFrame, baseline: BaselineSummary | None = None
) -> None:
    import matplotlib.pyplot as plt

    if summary_df.empty:
        print("No scaling runs to plot.")
        return

    x = summary_df["k"].to_numpy()
    x_rand = np.linspace(x.min() if baseline is None else 1, x.max(), 300)
    rand_curve = 1.0 / (x_rand + 1)

    # (title, col_mean, col_std, baseline_val, color, marker, show_rand_curve)
    metrics = [
        (
            "Macro Recall",
            "recall_mean",
            "recall_std",
            baseline.recall if baseline else None,
            "steelblue",
            "o",
            True,
        ),
        (
            "Macro Precision",
            "precision_mean",
            "precision_std",
            baseline.precision if baseline else None,
            "darkorange",
            "s",
            True,
        ),
        (
            "Macro F1",
            "f1_mean",
            "f1_std",
            baseline.f1 if baseline else None,
            "mediumpurple",
            "D",
            True,
        ),
        (
            "Top-1 Accuracy",
            "top1_acc_mean",
            "top1_acc_std",
            baseline.top1_accuracy if baseline else None,
            "seagreen",
            "^",
            True,
        ),
        (
            "Test Loss",
            "loss_mean",
            "loss_std",
            baseline.loss if baseline else None,
            "crimson",
            "v",
            False,
        ),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Scaling results by k", fontsize=13, fontweight="bold")
    flat_axes = axes.flatten()
    flat_axes[-1].set_visible(False)

    for ax, (title, col_mean, col_std, bl_val, color, marker, show_rand) in zip(
        flat_axes, metrics
    ):
        mean = summary_df[col_mean].to_numpy()
        std = summary_df[col_std].to_numpy()

        if show_rand:
            ax.plot(
                x_rand,
                rand_curve,
                color="gray",
                linestyle=":",
                linewidth=1.2,
                label="Random (1/(k+1))",
            )
        if bl_val is not None:
            ax.scatter(
                [1],
                [bl_val],
                color=color,
                marker="*",
                s=120,
                zorder=5,
                label=f"Baseline ({bl_val:.3f})",
            )
        ax.plot(x, mean, marker=marker, color=color, label=title)
        ax.fill_between(x, mean - std, mean + std, alpha=0.18, color=color)

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("k  (target classes + 1 non-target)")
        ax.set_ylabel("Loss" if not show_rand else "Score")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()
