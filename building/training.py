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
# Per-class shuffle buffer cap. Each unit ≈ 48000 × 4 B = 187 KB of resident
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


@dataclass
class DatasetMeta:
    epoch_samples: int
    n_classes: int
    class_weights: dict[int, float] | None = None


_CACHE_BATCH = 1  # one file per batch so a corrupt clip only drops itself


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


def cleanup_waveform_cache(collection: str) -> None:
    """Wipe the waveform cache for a collection.

    Useful at the end of training to reclaim disk space and to guarantee the
    next run rebuilds from the current files on disk (the cache key is keyed
    only on the feature config, not the file-set, so it doesn't auto-invalidate
    when the dataset folder is edited).
    """
    import shutil

    cache_root = ROOT / ".cache" / collection
    if not cache_root.exists():
        print(f"[cleanup] no cache at {cache_root}")
        return
    size_mb = sum(p.stat().st_size for p in cache_root.rglob("*") if p.is_file()) / 1e6
    shutil.rmtree(cache_root)
    print(f"[cleanup] removed {cache_root} ({size_mb:,.1f} MB freed)")


def load_dataset_catalog(collection: str) -> DatasetCatalog:
    """Per-class cached waveform DatasetCatalog. Cache invalidates via cfg hash."""
    from building.utils import SAMPLE_RATE, fix_audio_length_mel

    keras = tf.keras
    dataset_root = ROOT / "datasets" / collection
    cfg_hash = _feature_config_hash()
    cache_root = ROOT / ".cache" / collection / f"waveform_{cfg_hash}"

    def _feature_fn(audio_batch: tf.Tensor) -> tf.Tensor:
        return fix_audio_length_mel(audio_batch)

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
            verbose=False,
        )
        ds: tf.data.Dataset = (
            ds_raw
            # AudioSet downloads occasionally contain corrupt headers; skip
            # them rather than aborting the whole class. log_warning prints
            # the offending file path so it can be removed at leisure.
            .apply(tf.data.experimental.ignore_errors(log_warning=True))
            .map(
                tf.autograph.experimental.do_not_convert(lambda x, _: _feature_fn(x)),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
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

        # ignore_errors() makes cardinality UNKNOWN, which then propagates
        # all the way to Keras and triggers spurious "ran out of data"
        # warnings. We know the real count from the cache, so assert it.
        ds = ds.apply(tf.data.experimental.assert_cardinality(count))
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


def model_factory(name: str, **kwargs) -> Callable[[int], tf.keras.Model]:
    from building.models import build_model

    return lambda n_classes: build_model(name, n_classes, **kwargs)


def build_dataset_from_catalog(
    catalog: DatasetCatalog,
    chosen_idxs: list[int],
    non_target_idx: int,
    batch_size: int,
    rng: np.random.Generator,
    split: Literal["train", "val", "test"],
    input_repr: Literal["time", "mel"] = "time",
    augment: bool = False,
    extra_other_idxs: list[int] | None = None,
) -> tuple[tf.data.Dataset, DatasetMeta]:
    """Build a batched TF dataset for one experiment from the on-disk catalog.

    Train: every class's per-class waveform dataset is concatenated under the
    natural distribution and globally shuffled. One epoch = one pass over every
    training sample. Per-class inverse-frequency weights (mean-normalised to 1)
    are emitted as the ``sample_weight`` element of each ``(x, y, sw)`` batch
    tuple so Keras reweights the loss without further plumbing.
    Val/Test: full concatenation, no resampling, no sample weights — eval loss
    reflects the natural distribution.

    The "other" class is the union of ``non_target_idx`` and any
    ``extra_other_idxs`` (unchosen target species folded in); sub-buckets are
    concatenated under the same label.
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

    # AutoGraph can't introspect lambdas/closures defined here when this module
    # is reloaded inside a notebook, so wrap every map fn with do_not_convert.
    nag = tf.autograph.experimental.do_not_convert

    # Build per-class labelled streams. counts_by_label is the natural epoch
    # budget per class — what we'd like to see in one full pass.
    counts_by_label: dict[int, int] = {}
    raw_per_class: list[tf.data.Dataset] = []
    for local_label, (cs, count) in enumerate(
        zip(target_class_splits, target_counts)
    ):
        lbl = tf.constant(local_label, dtype=tf.int32)
        raw_per_class.append(cs.ds.map(nag(lambda x, label=lbl: (x, label))))
        counts_by_label[local_label] = count

    olbl = tf.constant(other_label, dtype=tf.int32)
    other_ds = other_class_splits[0].ds.map(nag(lambda x, label=olbl: (x, label)))
    for cs in other_class_splits[1:]:
        other_ds = other_ds.concatenate(
            cs.ds.map(nag(lambda x, label=olbl: (x, label)))
        )
    raw_per_class.append(other_ds)
    counts_by_label[other_label] = sum(other_counts)

    total_samples = sum(counts_by_label.values())

    if do_shuffle:
        # Weighted interleave: each per-class stream is shuffled within its
        # own buffer (cheap, since per-class buffer fits well under
        # SHUFFLE_BUFFER_CAP) and repeated; sample_from_datasets mixes them
        # in natural-distribution proportions every batch. Epoch length is
        # pinned to total_samples so cardinality is known.
        per_class_streams: list[tf.data.Dataset] = []
        for lbl_idx, ds_lbl in enumerate(raw_per_class):
            count = counts_by_label[lbl_idx]
            shuffle_buf = min(count, SHUFFLE_BUFFER_CAP)
            seed = int(rng.integers(0, 2**31 - 1))
            per_class_streams.append(
                ds_lbl.shuffle(
                    shuffle_buf, seed=seed, reshuffle_each_iteration=True
                ).repeat()
            )
        weights = [counts_by_label[i] / total_samples for i in range(n_classes)]
        combined = tf.data.Dataset.sample_from_datasets(
            per_class_streams,
            weights=weights,
            seed=int(rng.integers(0, 2**31 - 1)),
            stop_on_empty_dataset=False,
        ).take(total_samples)
    else:
        combined = raw_per_class[0]
        for p in raw_per_class[1:]:
            combined = combined.concatenate(p)

    # Mean-normalised inverse-frequency weights: w_c = N / (K * n_c), then
    # rescaled so the mean weight = 1 (keeps the loss scale comparable).
    raw_weights = {
        lbl: total_samples / (n_classes * count)
        for lbl, count in counts_by_label.items()
    }
    mean_w = sum(raw_weights.values()) / len(raw_weights)
    class_weights = {lbl: w / mean_w for lbl, w in raw_weights.items()}

    # Per-sample waveform augmentation (train split only). Runs *before* the
    # mel STFT so polarity flip / shift / noise affect the spectrogram.
    if do_shuffle and augment:
        from building.data.augmentation import augment_tf

        combined = combined.map(
            nag(lambda x, y: (augment_tf(x), y)),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    combined = combined.batch(batch_size)

    cw_tensor = tf.constant(
        [class_weights[i] for i in range(n_classes)], dtype=tf.float32
    )

    if input_repr == "mel":
        from building.utils import (
            create_log_mel_spectrogram,
            TARGET_FRAMES_MEL,
            NUM_MEL_BINS_MEL,
        )

        @nag
        def _featurize(xb: tf.Tensor) -> tf.Tensor:
            spec = create_log_mel_spectrogram(xb)
            spec = tf.ensure_shape(
                spec, [None, TARGET_FRAMES_MEL, NUM_MEL_BINS_MEL]
            )
            return tf.expand_dims(spec, -1)
    else:

        @nag
        def _featurize(xb: tf.Tensor) -> tf.Tensor:
            return tf.expand_dims(xb, -1)

    # Emit sample_weight for train + val (Keras consumes the 3-tuple
    # natively and EarlyStopping then monitors a class-balanced val_loss).
    # Test stays as a 2-tuple so the downstream metric helpers iterate it
    # as (xb, yb) and report the honest natural-distribution test loss.
    weighted = split in ("train", "val")
    if weighted:
        @nag
        def _to_features(xb: tf.Tensor, yb: tf.Tensor):
            return (
                _featurize(xb),
                tf.one_hot(yb, n_classes, dtype=tf.float32),
                tf.gather(cw_tensor, yb),
            )
    else:
        @nag
        def _to_features(xb: tf.Tensor, yb: tf.Tensor):
            return _featurize(xb), tf.one_hot(yb, n_classes, dtype=tf.float32)

    combined = combined.map(
        _to_features, num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(PREFETCH_BUFFER)
    return combined, DatasetMeta(
        epoch_samples=int(total_samples),
        n_classes=n_classes,
        class_weights=class_weights if weighted else None,
    )


def build_grouped_dataset_from_catalog(
    catalog: DatasetCatalog,
    label_groups: list[list[int]],
    batch_size: int,
    rng: np.random.Generator,
    split: Literal["train", "val", "test"],
    input_repr: Literal["time", "mel"] = "time",
    augment: bool = False,
) -> tuple[tf.data.Dataset, DatasetMeta]:
    """Like `build_dataset_from_catalog` but with explicit label groups.

    Each group is a list of catalog class indices; all classes in group i
    receive label i. Within a group, per-class TF datasets are concatenated;
    across groups, the train split is mixed via ``sample_from_datasets`` in
    natural-distribution proportions (same scheme as the existing builder).

    Used by the cascading pipeline: stage 1 pools every bird folder under one
    label and uses ``no_bird`` as the other; stage 2 reuses the same catalog
    with one group per target species and a single ``non_target_bird`` group.
    """
    if not label_groups or any(len(g) == 0 for g in label_groups):
        raise ValueError("label_groups must be a non-empty list of non-empty groups")

    def _get_split(e: ClassEntry) -> ClassSplit:
        if split == "train":
            return e.train
        if split == "val":
            return e.val
        return e.test

    n_classes = len(label_groups)
    do_shuffle = split == "train"
    nag = tf.autograph.experimental.do_not_convert

    counts_by_label: dict[int, int] = {}
    raw_per_label: list[tf.data.Dataset] = []
    for lbl_idx, group in enumerate(label_groups):
        splits = [_get_split(catalog.entries[i]) for i in group]
        counts = [s.count for s in splits]
        if any(c == 0 for c in counts):
            bad = [group[i] for i, c in enumerate(counts) if c == 0]
            raise ValueError(
                f"Class(es) {bad} have no samples in split '{split}'"
            )
        lbl = tf.constant(lbl_idx, dtype=tf.int32)
        ds = splits[0].ds.map(nag(lambda x, label=lbl: (x, label)))
        for s in splits[1:]:
            ds = ds.concatenate(
                s.ds.map(nag(lambda x, label=lbl: (x, label)))
            )
        raw_per_label.append(ds)
        counts_by_label[lbl_idx] = sum(counts)

    total_samples = sum(counts_by_label.values())

    if do_shuffle:
        per_class_streams: list[tf.data.Dataset] = []
        for lbl_idx, ds_lbl in enumerate(raw_per_label):
            count = counts_by_label[lbl_idx]
            shuffle_buf = min(count, SHUFFLE_BUFFER_CAP)
            seed = int(rng.integers(0, 2**31 - 1))
            per_class_streams.append(
                ds_lbl.shuffle(
                    shuffle_buf, seed=seed, reshuffle_each_iteration=True
                ).repeat()
            )
        weights = [counts_by_label[i] / total_samples for i in range(n_classes)]
        combined = tf.data.Dataset.sample_from_datasets(
            per_class_streams,
            weights=weights,
            seed=int(rng.integers(0, 2**31 - 1)),
            stop_on_empty_dataset=False,
        ).take(total_samples)
    else:
        combined = raw_per_label[0]
        for p in raw_per_label[1:]:
            combined = combined.concatenate(p)

    raw_weights = {
        lbl: total_samples / (n_classes * count)
        for lbl, count in counts_by_label.items()
    }
    mean_w = sum(raw_weights.values()) / len(raw_weights)
    class_weights = {lbl: w / mean_w for lbl, w in raw_weights.items()}

    if do_shuffle and augment:
        from building.data.augmentation import augment_tf

        combined = combined.map(
            nag(lambda x, y: (augment_tf(x), y)),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    combined = combined.batch(batch_size)

    cw_tensor = tf.constant(
        [class_weights[i] for i in range(n_classes)], dtype=tf.float32
    )

    if input_repr == "mel":
        from building.utils import (
            create_log_mel_spectrogram,
            TARGET_FRAMES_MEL,
            NUM_MEL_BINS_MEL,
        )

        @nag
        def _featurize(xb: tf.Tensor) -> tf.Tensor:
            spec = create_log_mel_spectrogram(xb)
            spec = tf.ensure_shape(
                spec, [None, TARGET_FRAMES_MEL, NUM_MEL_BINS_MEL]
            )
            return tf.expand_dims(spec, -1)
    else:

        @nag
        def _featurize(xb: tf.Tensor) -> tf.Tensor:
            return tf.expand_dims(xb, -1)

    weighted = split in ("train", "val")
    if weighted:
        @nag
        def _to_features(xb: tf.Tensor, yb: tf.Tensor):
            return (
                _featurize(xb),
                tf.one_hot(yb, n_classes, dtype=tf.float32),
                tf.gather(cw_tensor, yb),
            )
    else:
        @nag
        def _to_features(xb: tf.Tensor, yb: tf.Tensor):
            return _featurize(xb), tf.one_hot(yb, n_classes, dtype=tf.float32)

    combined = combined.map(
        _to_features, num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(PREFETCH_BUFFER)
    return combined, DatasetMeta(
        epoch_samples=int(total_samples),
        n_classes=n_classes,
        class_weights=class_weights if weighted else None,
    )


def collect_predictions(
    model: tf.keras.Model, ds: tf.data.Dataset
) -> tuple[np.ndarray, np.ndarray]:
    y_true, y_pred = [], []
    for xb, yb in ds:
        y_true.append(yb.numpy())
        y_pred.append(model(xb, training=False).numpy())
    return np.concatenate(y_true), np.concatenate(y_pred)


def compute_metrics(
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
