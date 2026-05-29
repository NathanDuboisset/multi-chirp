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
SHUFFLE_BUFFER_CAP = 256  # per-class; file-level shuffle in dataset.py handles bulk randomness
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
    model_config = ConfigDict(arbitrary_types_allowed=True)
    class_names: list[str]
    entries: list[ClassEntry]  # parallel to class_names


class PerClassMetricsLite(BaseModel):
    support: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    f2: float
    auc: float | None = None


class MacroBlock(BaseModel):
    precision: float
    recall: float
    f1: float
    f2: float
    auc: float | None = None


class RunMetrics(BaseModel):
    recall_mean: float
    recall_std: float
    precision_mean: float
    precision_std: float
    f1_mean: float
    f1_std: float
    top1_accuracy: float
    loss: float
    # optional for backwards-compat with old jsonl rows
    per_class: dict[str, PerClassMetricsLite] | None = None
    macro_targets: MacroBlock | None = None
    macro_all: MacroBlock | None = None
    threshold: float | None = None


@dataclass
class DatasetMeta:
    epoch_samples: int
    n_classes: int
    class_weights: dict[int, float] | None = None


_CACHE_BATCH = 1  # one file per batch so a corrupt clip only drops itself


def _feature_config_hash() -> str:
    # mel STFT runs at train time so mel params don't invalidate the cache
    from building.utils import SAMPLE_RATE, TARGET_AUDIO_LEN_MEL

    cfg: dict[str, Any] = {
        "sample_rate": SAMPLE_RATE,
        "target_audio_len": TARGET_AUDIO_LEN_MEL,
    }
    payload = json.dumps(cfg, sort_keys=True).encode()
    return hashlib.sha1(payload).hexdigest()[:8]


def cleanup_waveform_cache(collection: str) -> None:
    import shutil

    cache_root = ROOT / ".cache" / collection
    if not cache_root.exists():
        print(f"[cleanup] no cache at {cache_root}")
        return
    size_mb = sum(p.stat().st_size for p in cache_root.rglob("*") if p.is_file()) / 1e6
    shutil.rmtree(cache_root)
    print(f"[cleanup] removed {cache_root} ({size_mb:,.1f} MB freed)")


def load_dataset_catalog(collection: str) -> DatasetCatalog:
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

        for lockfile in cache_dir.glob("*.lockfile"):
            lockfile.unlink(missing_ok=True)

        # label (always 0) is discarded; we only need this class's waveforms
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
            # AudioSet occasionally has corrupt headers; skip rather than abort
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

        # ignore_errors() makes cardinality UNKNOWN; reassert to silence Keras
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

    # AutoGraph breaks on reloaded notebook closures
    nag = tf.autograph.experimental.do_not_convert

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
        # choose_from_datasets with a numpy-shuffled label schedule visits each
        # sample exactly once per epoch at natural class proportions
        per_class_streams: list[tf.data.Dataset] = []
        for lbl_idx, ds_lbl in enumerate(raw_per_class):
            count = counts_by_label[lbl_idx]
            shuffle_buf = min(count, SHUFFLE_BUFFER_CAP)
            seed = int(rng.integers(0, 2**31 - 1))
            per_class_streams.append(
                ds_lbl.shuffle(
                    shuffle_buf, seed=seed, reshuffle_each_iteration=True
                )
            )
        schedule = np.concatenate(
            [
                np.full(counts_by_label[i], i, dtype=np.int64)
                for i in range(n_classes)
            ]
        )
        np.random.default_rng(int(rng.integers(0, 2**31 - 1))).shuffle(schedule)
        schedule_ds = tf.data.Dataset.from_tensor_slices(schedule)
        combined = tf.data.Dataset.choose_from_datasets(
            per_class_streams, schedule_ds
        )
        # reassert cardinality to silence Keras' "ran out of data" warning
        combined = combined.apply(
            tf.data.experimental.assert_cardinality(int(total_samples))
        )
    else:
        combined = raw_per_class[0]
        for p in raw_per_class[1:]:
            combined = combined.concatenate(p)

    # inverse-frequency weights w_c = N / (K * n_c), rescaled to mean=1
    raw_weights = {
        lbl: total_samples / (n_classes * count)
        for lbl, count in counts_by_label.items()
    }
    mean_w = sum(raw_weights.values()) / len(raw_weights)
    class_weights = {lbl: w / mean_w for lbl, w in raw_weights.items()}

    # augmentation runs before _featurize so polarity/shift/noise also affect
    # the downstream mel STFT
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

    # train + val emit sample_weight so EarlyStopping sees class-balanced
    # val_loss; test stays a 2-tuple for natural-distribution loss
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
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
    loss: float,
    class_names: list[str] | None = None,
    non_target_names: Iterable[str] = (),
) -> RunMetrics:
    # legacy recall_mean / std / ... fields keep averaging over *every* class
    # so old jsonl consumers continue to work
    from sklearn.metrics import roc_auc_score

    n_classes = y_true.shape[1]
    if class_names is not None and len(class_names) != n_classes:
        raise ValueError(
            f"class_names has {len(class_names)} entries but y_true has "
            f"{n_classes} columns."
        )

    recalls, precisions, f1s = [], [], []
    per_class: dict[str, PerClassMetricsLite] = {}
    for c in range(n_classes):
        true_c = y_true[:, c] >= 0.5
        pred_c = y_pred[:, c] >= threshold
        tp = int((pred_c & true_c).sum())
        fp = int((pred_c & ~true_c).sum())
        fn = int((~pred_c & true_c).sum())
        tn = int((~pred_c & ~true_c).sum())
        rec_v = tp / (tp + fn) if (tp + fn) else None
        prec_v = tp / (tp + fp) if (tp + fp) else None
        if rec_v is not None:
            recalls.append(rec_v)
        if prec_v is not None:
            precisions.append(prec_v)
        if rec_v is not None and prec_v is not None and (rec_v + prec_v) > 0:
            f1s.append(2 * rec_v * prec_v / (rec_v + prec_v))

        if class_names is not None:
            p = prec_v if prec_v is not None else 0.0
            r = rec_v if rec_v is not None else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) else 0.0
            d2 = 4 * p + r
            f2 = 5 * p * r / d2 if d2 else 0.0
            n = tp + fp + fn + tn
            acc = (tp + tn) / n if n else 0.0
            auc: float | None = None
            ys = y_pred[:, c]
            if true_c.any() and not true_c.all():
                auc = float(roc_auc_score(true_c.astype(int), ys))
            per_class[class_names[c]] = PerClassMetricsLite(
                support=int(true_c.sum()),
                accuracy=float(acc),
                precision=float(p),
                recall=float(r),
                f1=float(f1),
                f2=float(f2),
                auc=auc,
            )

    macro_targets: MacroBlock | None = None
    macro_all: MacroBlock | None = None
    if class_names is not None:
        nt = set(non_target_names)

        def _aggregate(names: list[str]) -> MacroBlock | None:
            if not names:
                return None
            ms = [per_class[n] for n in names]
            aucs = [m.auc for m in ms]
            return MacroBlock(
                precision=float(np.mean([m.precision for m in ms])),
                recall=float(np.mean([m.recall for m in ms])),
                f1=float(np.mean([m.f1 for m in ms])),
                f2=float(np.mean([m.f2 for m in ms])),
                auc=(
                    float(np.mean(aucs))
                    if aucs and all(a is not None for a in aucs)
                    else None
                ),
            )

        macro_all = _aggregate(class_names)
        target_names = [n for n in class_names if n not in nt]
        macro_targets = _aggregate(target_names)

    return RunMetrics(
        recall_mean=float(np.mean(recalls)) if recalls else 0.0,
        recall_std=float(np.std(recalls)) if recalls else 0.0,
        precision_mean=float(np.mean(precisions)) if precisions else 0.0,
        precision_std=float(np.std(precisions)) if precisions else 0.0,
        f1_mean=float(np.mean(f1s)) if f1s else 0.0,
        f1_std=float(np.std(f1s)) if f1s else 0.0,
        top1_accuracy=float(np.mean(np.argmax(y_true, 1) == np.argmax(y_pred, 1))),
        loss=loss,
        per_class=per_class or None,
        macro_targets=macro_targets,
        macro_all=macro_all,
        threshold=float(threshold) if class_names is not None else None,
    )
