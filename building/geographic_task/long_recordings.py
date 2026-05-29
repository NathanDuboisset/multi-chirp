from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import pyrootutils
from tqdm.auto import tqdm
from xenocanto3 import AsyncXenoCantoClient, Recording

from building.analysis import _run_birdnet
from building.data.download import POOL_SIZE

ROOT = pyrootutils.setup_root(
    search_from=__file__,
    indicator="pyproject.toml",
    pythonpath=True,
    dotenv=True,
)

LONG_REC_ROOT = ROOT / "raw_dataset" / "long_recordings"
XC_DL_INTERVAL = 0.2
WINDOW_SEC = 3.0
SAMPLE_RATE = 16_000


def long_rec_dir(lat: float, lon: float, radius_km: float) -> Path:
    return LONG_REC_ROOT / f"r{radius_km:g}km_lat{lat:.4f}_lon{lon:.4f}"


def pick_longest(
    area_df: pd.DataFrame,
    target_species: Sequence[str],
    n_per_species: int,
) -> pd.DataFrame:
    df = area_df.dropna(subset=["file", "length_seconds"]).copy()
    df = df[(df["length_seconds"] > 0) & df["scientific_name"].isin(target_species)]
    return (
        df.sort_values("length_seconds", ascending=False)
        .groupby("scientific_name", sort=False)
        .head(n_per_species)
        .reset_index(drop=True)
    )


async def download_long(
    picks: pd.DataFrame,
    out_dir: Path,
    pool_size: int = POOL_SIZE,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    dl_lock = asyncio.Lock()
    dl_last = [0.0]
    sem = asyncio.Semaphore(pool_size)
    pbar = tqdm(total=len(picks), desc="download long XC")

    async with AsyncXenoCantoClient(api_key=os.getenv("XC_API_KEY", "demo")) as xc:

        async def _fetch(row) -> None:
            xc_id = int(row.id)
            audio_path = out_dir / f"XC{xc_id}.mp3"
            meta_path = out_dir / f"XC{xc_id}.meta.json"
            if not audio_path.exists():

                async with dl_lock:
                    now = asyncio.get_event_loop().time()
                    wait = XC_DL_INTERVAL - (now - dl_last[0])
                    if wait > 0:
                        await asyncio.sleep(wait)
                    dl_last[0] = asyncio.get_event_loop().time()
                async with sem:
                    try:
                        await xc.download(
                            Recording(id=xc_id, file=row.file),
                            audio_path,
                            overwrite=False,
                        )
                    except Exception as e:
                        print(f"[XC{xc_id}] download failed: {e}")
                        pbar.update(1)
                        return
            if not meta_path.exists():
                meta_path.write_text(
                    json.dumps(
                        {
                            "xc_id": xc_id,
                            "scientific_name": row.scientific_name,
                            "en": row.en,
                            "length_seconds": float(row.length_seconds),
                            "lat": row.lat,
                            "lon": row.lon,
                            "q": row.q,
                            "loc": row.loc,
                            "date": str(row.date),
                            "also": list(row.also) if row.also else [],
                            "file": row.file,
                        },
                        ensure_ascii=False,
                    )
                )
            pbar.update(1)

        await asyncio.gather(*[_fetch(row) for row in picks.itertuples()])
    pbar.close()
    _write_index(out_dir)


def _write_index(out_dir: Path) -> None:
    rows = []
    for meta_path in sorted(out_dir.glob("XC*.meta.json")):
        m = json.loads(meta_path.read_text())
        rows.append(
            {
                "xc_id": m["xc_id"],
                "scientific_name": m["scientific_name"],
                "length_seconds": m["length_seconds"],
                "audio_file": f"XC{m['xc_id']}.mp3",
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "index.csv", index=False)


def birdnet_long(
    audio_dir: Path,
    target_species: Sequence[str],
    min_confidence: float = 0.5,
    overwrite: bool = False,
) -> None:
    audio_dir.mkdir(parents=True, exist_ok=True)
    target_set = set(target_species)
    cache_key = sorted(target_set)
    files = sorted(audio_dir.glob("XC*.mp3"))
    for f in tqdm(files, desc="BirdNET long"):
        jsonl_path = f.with_suffix(".birdnet.jsonl")
        if jsonl_path.exists() and not overwrite:
            try:
                header = json.loads(jsonl_path.read_text().splitlines()[0])
                if (
                    float(header.get("min_confidence", 1.0)) <= min_confidence
                    and header.get("targets") == cache_key
                ):
                    continue
            except Exception:
                pass
        try:
            dets = _run_birdnet(f, min_confidence)
        except Exception as e:
            print(f"[{f.name}] BirdNET failed: {e}")
            continue
        dets = [d for d in dets if d.get("scientific_name") in target_set]
        with jsonl_path.open("w") as fh:
            fh.write(
                json.dumps(
                    {"min_confidence": min_confidence, "targets": cache_key, "schema": "header"}
                )
                + "\n"
            )
            for d in dets:
                fh.write(
                    json.dumps(
                        {
                            "scientific_name": d.get("scientific_name", ""),
                            "common_name": d.get("common_name", ""),
                            "confidence": float(d.get("confidence", 0.0)),
                            "start_time": float(d.get("start_time", 0.0)),
                            "end_time": float(d.get("end_time", 0.0)),
                        }
                    )
                    + "\n"
                )


def load_birdnet_jsonl(jsonl_path: Path) -> pd.DataFrame:
    if not jsonl_path.exists():
        return pd.DataFrame(
            columns=["scientific_name", "common_name", "confidence", "start_time", "end_time"]
        )
    rows = []
    for line in jsonl_path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("schema") == "header":
            continue
        rows.append(d)
    return pd.DataFrame(rows)


@dataclass
class TFLiteRunner:
    interp: object
    inp: dict
    out: dict
    in_scale: float
    in_zp: int
    out_scale: float
    out_zp: int
    in_dtype: np.dtype
    out_dtype: np.dtype
    in_shape: tuple

    @classmethod
    def load(cls, tflite_path: Path) -> "TFLiteRunner":
        import tensorflow as tf

        interp = tf.lite.Interpreter(model_path=str(tflite_path))
        interp.allocate_tensors()
        inp = interp.get_input_details()[0]
        out = interp.get_output_details()[0]
        return cls(
            interp=interp,
            inp=inp,
            out=out,
            in_scale=float(inp["quantization"][0]),
            in_zp=int(inp["quantization"][1]),
            out_scale=float(out["quantization"][0]),
            out_zp=int(out["quantization"][1]),
            in_dtype=inp["dtype"],
            out_dtype=out["dtype"],
            in_shape=tuple(inp["shape"]),
        )

    def predict(self, wave: np.ndarray) -> np.ndarray:
        x = wave.astype(np.float32).reshape(self.in_shape[1:])
        if self.in_dtype == np.float32:
            xq = x[None, ...].astype(np.float32)
        elif self.in_dtype in (np.int8, np.uint8):
            qmin, qmax = (-128, 127) if self.in_dtype == np.int8 else (0, 255)
            xq = np.clip(np.round(x / self.in_scale) + self.in_zp, qmin, qmax).astype(
                self.in_dtype
            )[None, ...]
        elif self.in_dtype == np.int16:
            xq = np.clip(np.round(x / self.in_scale) + self.in_zp, -32768, 32767).astype(
                np.int16
            )[None, ...]
        else:
            raise ValueError(f"unsupported tflite input dtype {self.in_dtype}")
        self.interp.set_tensor(self.inp["index"], xq)
        self.interp.invoke()
        raw = self.interp.get_tensor(self.out["index"])
        if self.out_dtype == np.float32:
            return raw.astype(np.float32).reshape(-1)
        return ((raw.astype(np.float32) - self.out_zp) * self.out_scale).reshape(-1)


def predict_recording(
    runner: TFLiteRunner,
    audio_path: Path,
    label_names: Sequence[str],
) -> pd.DataFrame:
    import librosa

    wave, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    win = int(WINDOW_SEC * SAMPLE_RATE)
    n = wave.shape[0] // win
    rows = []
    for i in range(n):
        probs = runner.predict(wave[i * win : (i + 1) * win])
        if probs.shape[0] != len(label_names):
            raise ValueError(
                f"model has {probs.shape[0]} outputs, but {len(label_names)} label names given"
            )
        start = i * WINDOW_SEC
        row = {"window_idx": i, "start_sec": start, "end_sec": start + WINDOW_SEC,
               "minute": int(start // 60)}
        for name, p in zip(label_names, probs):
            row[name] = float(p)
        rows.append(row)
    return pd.DataFrame(rows)


def model_per_minute(
    win_df: pd.DataFrame,
    target_labels: Sequence[str],
    threshold: float | Mapping[str, float],
) -> pd.DataFrame:
    g = win_df.groupby("minute")[list(target_labels)].mean()
    if isinstance(threshold, Mapping):
        thr_vec = np.array([float(threshold[n]) for n in target_labels])
    else:
        thr_vec = np.full(len(target_labels), float(threshold))
    probs = g.to_numpy()
    passes = probs >= thr_vec[None, :]
    masked = np.where(passes, probs, -np.inf)
    best_idx = masked.argmax(axis=1)
    any_pass = passes.any(axis=1)
    species = np.array(list(target_labels))[best_idx]
    out = pd.DataFrame(index=g.index.copy())
    out["max_target_prob"] = probs.max(axis=1)
    out["predicted_species"] = pd.Series(
        np.where(any_pass, species, None), index=g.index
    )
    out["bird"] = any_pass
    return out.reset_index()


def birdnet_per_minute(
    det_df: pd.DataFrame,
    target_species: Sequence[str],
    n_minutes: int,
) -> pd.DataFrame:
    target_underscored = {s.replace(" ", "_"): s for s in target_species}
    minutes = []
    for m in range(n_minutes):
        rows = det_df[
            (det_df["start_time"] >= m * 60.0) & (det_df["start_time"] < (m + 1) * 60.0)
        ]
        if rows.empty:
            minutes.append({"minute": m, "bird": False, "predicted_species": None})
            continue
        best = rows.sort_values("confidence", ascending=False).iloc[0]
        sci_under = (best["scientific_name"] or "").replace(" ", "_")
        minutes.append(
            {
                "minute": m,
                "bird": True,
                "predicted_species": sci_under if sci_under in target_underscored else None,
                "any_species": sci_under,
            }
        )
    return pd.DataFrame(minutes)


@dataclass
class MinuteStats:
    n_minutes: int
    n_model_bird: int
    n_birdnet_bird: int
    n_model_and_birdnet_bird: int
    n_correct_species: int
    n_model_bird_predictions: int
    n_missed_birds: int

    @property
    def precision_bird(self) -> float | None:
        return self.n_model_and_birdnet_bird / self.n_model_bird if self.n_model_bird else None

    @property
    def correct_species_rate(self) -> float | None:
        return (
            self.n_correct_species / self.n_model_bird_predictions
            if self.n_model_bird_predictions
            else None
        )

    @property
    def miss_rate(self) -> float | None:
        return self.n_missed_birds / self.n_birdnet_bird if self.n_birdnet_bird else None


def confuse_minutes(model_min: pd.DataFrame, birdnet_min: pd.DataFrame) -> MinuteStats:
    j = model_min.merge(birdnet_min, on="minute", suffixes=("_model", "_birdnet"))
    return MinuteStats(
        n_minutes=len(j),
        n_model_bird=int(j["bird_model"].sum()),
        n_birdnet_bird=int(j["bird_birdnet"].sum()),
        n_model_and_birdnet_bird=int((j["bird_model"] & j["bird_birdnet"]).sum()),
        n_model_bird_predictions=int(j["predicted_species_model"].notna().sum()),
        n_correct_species=int(
            (
                j["predicted_species_model"].notna()
                & (j["predicted_species_model"] == j["predicted_species_birdnet"])
            ).sum()
        ),
        n_missed_birds=int((j["bird_birdnet"] & ~j["bird_model"]).sum()),
    )


@dataclass
class RecordingReport:
    xc_id: int
    audio_path: Path
    length_seconds: float
    stats: MinuteStats
    model_min: pd.DataFrame
    birdnet_min: pd.DataFrame


def evaluate_long_recordings(
    audio_dir: Path,
    tflite_path: Path,
    label_names: Sequence[str],
    target_species: Sequence[str],
    threshold: float | Mapping[str, float] = 0.5,
) -> tuple[list[RecordingReport], MinuteStats]:
    runner = TFLiteRunner.load(tflite_path)
    target_labels = [n for n in label_names if n in {s.replace(" ", "_") for s in target_species}]
    if not target_labels:
        raise ValueError(
            f"none of {label_names} match target_species {target_species} "
            "(expected underscored scientific names)"
        )

    files = sorted(audio_dir.glob("XC*.mp3"))
    reports: list[RecordingReport] = []
    totals = MinuteStats(0, 0, 0, 0, 0, 0, 0)

    for f in tqdm(files, desc="INT8 over long"):
        meta = json.loads(f.with_suffix(".meta.json").read_text())
        win_df = predict_recording(runner, f, label_names)
        if win_df.empty:
            continue
        n_minutes = int(win_df["minute"].max()) + 1
        m_min = (
            pd.DataFrame({"minute": range(n_minutes)})
            .merge(model_per_minute(win_df, target_labels, threshold), on="minute", how="left")
            .fillna({"bird": False, "max_target_prob": 0.0})
        )
        b_min = birdnet_per_minute(
            load_birdnet_jsonl(f.with_suffix(".birdnet.jsonl")), target_species, n_minutes
        )
        stats = confuse_minutes(m_min, b_min)

        reports.append(
            RecordingReport(
                xc_id=int(meta["xc_id"]),
                audio_path=f,
                length_seconds=float(meta["length_seconds"]),
                stats=stats,
                model_min=m_min,
                birdnet_min=b_min,
            )
        )
        totals = MinuteStats(
            n_minutes=totals.n_minutes + stats.n_minutes,
            n_model_bird=totals.n_model_bird + stats.n_model_bird,
            n_birdnet_bird=totals.n_birdnet_bird + stats.n_birdnet_bird,
            n_model_and_birdnet_bird=totals.n_model_and_birdnet_bird + stats.n_model_and_birdnet_bird,
            n_correct_species=totals.n_correct_species + stats.n_correct_species,
            n_model_bird_predictions=totals.n_model_bird_predictions + stats.n_model_bird_predictions,
            n_missed_birds=totals.n_missed_birds + stats.n_missed_birds,
        )
    return reports, totals


def format_stats(s: MinuteStats) -> str:
    def pct(x: float | None) -> str:
        return f"{100 * x:6.2f}%" if x is not None else "  n/a "

    return (
        f"  minutes scored          : {s.n_minutes}\n"
        f"  model said bird         : {s.n_model_bird}\n"
        f"  BirdNET said bird       : {s.n_birdnet_bird}\n"
        f"  P(bird | model bird)    : {pct(s.precision_bird)}\n"
        f"  P(correct species | s)  : {pct(s.correct_species_rate)}\n"
        f"  P(miss | BirdNET bird)  : {pct(s.miss_rate)}\n"
    )
