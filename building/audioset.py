"""Download AudioSet ambient clips by YouTube ID, filtered via the official CSVs."""

from __future__ import annotations

import asyncio
import csv
import random
import re
import shutil
import subprocess
import tempfile
import urllib.request
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Literal

import numpy as np
import pyrootutils
import resampy
import soundfile as sf
from pydantic import BaseModel, ConfigDict
from tqdm.auto import tqdm

from building.download import CLIP_DURATION, RAW_DATASET_DIR, TARGET_SAMPLE_RATE

ROOT = pyrootutils.setup_root(
    search_from=__file__,
    indicator="pyproject.toml",
    pythonpath=True,
    dotenv=True,
)


FOCUS_CLASSES: list[str] = [
    "Speech",
    "Male speech, man speaking",
    "Female speech, woman speaking",
    "Conversation",
    "Narration, monologue",
    "Children playing",
    "Crowd",
    "Laughter",
    "Applause",
    "Cheering",
    "Whistling",
    "Singing",
    "Walk, footsteps",
    "Door",
    "Knock",
    "Typing",
    "Computer keyboard",
    "Aircraft",
    "Helicopter",
    "Car",
    "Motorcycle",
    "Truck",
    "Bus",
    "Train",
    "Traffic noise, roadway noise",
    "Engine",
    "Light engine (high frequency)",
    "Chainsaw",
    "Lawn mower",
    "Power tool",
    "Drill",
    "Hammer",
    "Sawing",
    "Jackhammer",
    "Church bell",
    "Bell",
    "Alarm",
    "Siren",
    "Dog",
    "Cat",
    "Bee, wasp, etc.",
    "Cricket",
    "Insect",
    "Mosquito",
    "Fly, housefly",
    "Frog",
    "Croak",
    "Rain",
    "Rain on surface",
    "Raindrop",
    "Thunder",
    "Thunderstorm",
    "Wind",
    "Wind noise (microphone)",
    "Howl",
    "Rustling leaves",
    "Stream",
    "Waterfall",
    "Ocean",
    "Fire",
    "Outside, rural or natural",
    "Outside, urban or manmade",
    "Inside, small room",
    "Inside, large room or hall",
]


AUDIOSET_CSV_BASE = "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv"
CSV_CACHE_DIR = RAW_DATASET_DIR / "audioset" / "_csv"


def _slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return s or "unknown"


class AudioSetConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    split: Literal["unbalanced_train", "balanced_train", "eval"] = "unbalanced_train"
    clips_per_class: int = 250
    max_total_clips: int = 5000
    target_sample_rate: int = TARGET_SAMPLE_RATE
    target_clip_duration: float = CLIP_DURATION
    processed_dir: Path = RAW_DATASET_DIR / "audioset"
    window_hop_s: float = 0.5
    min_rms: float = 5e-3
    num_workers: int = 6
    yt_dlp_timeout_s: int = 90
    shuffle_seed: int = 0
    cookies_from_browser: str | None = "chrome"  # firefox|chrome|chromium|brave|edge|None


def _loudest_window(
    audio: np.ndarray, sr: int, dur: float, hop_s: float, min_rms: float
) -> np.ndarray | None:
    n = int(dur * sr)
    if len(audio) < sr:
        return None
    if len(audio) <= n:
        pad = n - len(audio)
        left = pad // 2
        padded = np.pad(audio, (left, pad - left)).astype(np.float32)
        if float(np.sqrt(np.mean(padded**2))) < min_rms:
            return None
        return padded
    hop = max(1, int(hop_s * sr))
    best_start, best_rms = 0, -1.0
    for start in range(0, len(audio) - n + 1, hop):
        rms = float(np.sqrt(np.mean(audio[start : start + n] ** 2)))
        if rms > best_rms:
            best_rms = rms
            best_start = start
    if best_rms < min_rms:
        return None
    return audio[best_start : best_start + n].astype(np.float32)


def _existing_keys(class_dir: Path) -> set[str]:
    if not class_dir.exists():
        return set()
    return {p.stem for p in class_dir.glob("*.wav")}


def _failed_log_path(class_dir: Path) -> Path:
    return class_dir / "_failed.txt"


def _failed_keys(class_dir: Path) -> set[str]:
    path = _failed_log_path(class_dir)
    if not path.exists():
        return set()
    with path.open() as f:
        return {line.strip() for line in f if line.strip()}


def _record_failure(class_dir: Path, ytid: str) -> None:
    class_dir.mkdir(parents=True, exist_ok=True)
    with _failed_log_path(class_dir).open("a") as f:
        f.write(f"{ytid}\n")


def _ensure_csv(name: str) -> Path:
    CSV_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dst = CSV_CACHE_DIR / name
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    url = f"{AUDIOSET_CSV_BASE}/{name}"
    tmp = dst.with_suffix(dst.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dst)
    return dst


def _load_mid_to_focus() -> dict[str, str]:
    """Map AudioSet machine-id (mid) -> focus class display name."""
    path = _ensure_csv("class_labels_indices.csv")
    focus_lookup = {c.lower(): c for c in FOCUS_CLASSES}
    out: dict[str, str] = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            disp = row["display_name"].strip().strip('"')
            if disp.lower() in focus_lookup:
                out[row["mid"]] = focus_lookup[disp.lower()]
    return out


def _iter_segments(split: str, focus_mids: dict[str, str]):
    """Yield (ytid, start_s, end_s, focus_class) for rows whose labels match focus."""
    fname = {
        "unbalanced_train": "unbalanced_train_segments.csv",
        "balanced_train": "balanced_train_segments.csv",
        "eval": "eval_segments.csv",
    }[split]
    path = _ensure_csv(fname)
    with path.open() as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            # Format: YTID, start_seconds, end_seconds, "mid,mid,..."
            # The label list is quoted; split carefully.
            try:
                head, labels_q = line.rstrip("\n").split(', "', 1)
                labels = labels_q.rstrip('"').split(",")
                ytid, start_s, end_s = [x.strip() for x in head.split(",")]
            except ValueError:
                continue
            matched = next((focus_mids[m] for m in labels if m in focus_mids), None)
            if matched is None:
                continue
            yield ytid, float(start_s), float(end_s), matched


def _ytdlp_fetch(
    ytid: str,
    start_s: float,
    end_s: float,
    out_wav: Path,
    timeout_s: int,
    cookies_from_browser: str | None = None,
) -> bool:
    """Download a YouTube clip segment as WAV. Returns True on success."""
    # Pad by 0.5s either side so loudest-window has room.
    s = max(0.0, start_s - 0.5)
    e = end_s + 0.5
    cmd = [
        "yt-dlp",
        "-f", "bestaudio",
        "--no-playlist",
        "--quiet",
        "--no-warnings",
        "--no-progress",
        "--no-call-home",
        "--download-sections", f"*{s}-{e}",
        "--force-keyframes-at-cuts",
        "-x",
        "--audio-format", "wav",
        "-o", str(out_wav.with_suffix(".%(ext)s")),
        f"https://www.youtube.com/watch?v={ytid}",
    ]
    if cookies_from_browser:
        cmd[1:1] = ["--cookies-from-browser", cookies_from_browser]
    try:
        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False
    # yt-dlp post-processors sometimes return non-zero even when the WAV is fine,
    # so trust the file, not the exit code.
    return out_wav.exists() and out_wav.stat().st_size > 1024


def _process_clip(
    ytid: str,
    start_s: float,
    end_s: float,
    cfg: AudioSetConfig,
) -> np.ndarray | None:
    """Download + decode + window. Returns the final mono target-sr clip, or None."""
    with tempfile.TemporaryDirectory(prefix="as_yt_") as td:
        tmp_wav = Path(td) / f"{ytid}.wav"
        if not _ytdlp_fetch(
            ytid,
            start_s,
            end_s,
            tmp_wav,
            cfg.yt_dlp_timeout_s,
            cfg.cookies_from_browser,
        ):
            return None
        try:
            arr, sr = sf.read(str(tmp_wav), dtype="float32", always_2d=False)
        except Exception:
            return None
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=-1)
    sr = int(sr)
    if sr != cfg.target_sample_rate:
        arr = resampy.resample(arr, sr, cfg.target_sample_rate).astype(np.float32)
        sr = cfg.target_sample_rate
    return _loudest_window(
        arr, sr, cfg.target_clip_duration, cfg.window_hop_s, cfg.min_rms
    )


def _download_class(
    cls: str,
    candidates: list[tuple[str, float, float]],
    have: int,
    cfg: AudioSetConfig,
    seen_ids: set[str],
) -> int:
    """Download up to (clips_per_class - have) clips for a single focus class.

    Returns number of new clips written.
    """
    key = _slugify(cls)
    target = cfg.clips_per_class - have
    if target <= 0:
        print(f"[{cls}] {have} clips, skipping.")
        return 0
    pending = [c for c in candidates if c[0] not in seen_ids]
    if not pending:
        print(f"[{cls}] no pending recordings.")
        return 0
    rng = random.Random(cfg.shuffle_seed + abs(hash(cls)) % 100000)
    rng.shuffle(pending)

    class_dir = cfg.processed_dir / key
    class_dir.mkdir(parents=True, exist_ok=True)

    pbar_recs = tqdm(
        total=len(pending),
        desc=f"[{cls}] recs",
        position=0,
        leave=True,
        dynamic_ncols=True,
    )
    pbar_clips = tqdm(
        total=cfg.clips_per_class,
        initial=have,
        desc=f"[{cls}] clips",
        position=1,
        leave=True,
        dynamic_ncols=True,
    )

    written = 0
    cand_iter = iter(pending)
    inflight: dict[Future, str] = {}

    def _submit_next(executor: ThreadPoolExecutor) -> bool:
        for ytid, s, e in cand_iter:
            if ytid in seen_ids:
                continue
            fut = executor.submit(_process_clip, ytid, s, e, cfg)
            inflight[fut] = ytid
            return True
        return False

    try:
        with ThreadPoolExecutor(max_workers=cfg.num_workers) as ex:
            for _ in range(cfg.num_workers):
                if written >= target:
                    break
                if not _submit_next(ex):
                    break

            while inflight and written < target:
                done, _ = wait(inflight.keys(), return_when=FIRST_COMPLETED)
                for fut in done:
                    ytid = inflight.pop(fut)
                    try:
                        window = fut.result()
                    except Exception:
                        window = None
                    pbar_recs.update(1)
                    if window is not None and written < target:
                        out_path = class_dir / f"{ytid}.wav"
                        sf.write(
                            str(out_path),
                            window,
                            cfg.target_sample_rate,
                            subtype="PCM_16",
                        )
                        seen_ids.add(ytid)
                        written += 1
                        pbar_clips.update(1)
                    elif window is None:
                        _record_failure(class_dir, ytid)
                        seen_ids.add(ytid)
                    if written < target:
                        _submit_next(ex)
    finally:
        pbar_clips.close()
        pbar_recs.close()

    return written


def stream_download_audioset(cfg: AudioSetConfig) -> dict[str, int]:
    if shutil.which("yt-dlp") is None or shutil.which("ffmpeg") is None:
        raise RuntimeError("yt-dlp and ffmpeg must be installed and on PATH.")

    cfg.processed_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    seen: dict[str, set[str]] = {}
    for cls in FOCUS_CLASSES:
        key = _slugify(cls)
        class_dir = cfg.processed_dir / key
        existing = _existing_keys(class_dir)
        counts[key] = len(existing)
        seen[key] = existing | _failed_keys(class_dir)
    total = sum(counts.values())

    if total >= cfg.max_total_clips:
        print(f"[audioset] global cap {cfg.max_total_clips} already reached ({total} on disk).")
        return counts

    print(
        f"[audioset] split={cfg.split} per_class={cfg.clips_per_class} "
        f"global={cfg.max_total_clips} on_disk={total} workers={cfg.num_workers}"
    )

    focus_mids = _load_mid_to_focus()
    if not focus_mids:
        raise RuntimeError("No focus classes matched the AudioSet ontology.")

    # Bucket candidates by focus class (one pass through the big CSV).
    by_class: dict[str, list[tuple[str, float, float]]] = {
        _slugify(c): [] for c in FOCUS_CLASSES
    }
    print(f"[audioset] indexing {cfg.split} segments...")
    for ytid, s, e, matched in _iter_segments(cfg.split, focus_mids):
        by_class[_slugify(matched)].append((ytid, s, e))
    print(
        "[audioset] candidates per class: "
        + ", ".join(f"{c}={len(by_class[_slugify(c)])}" for c in FOCUS_CLASSES[:5])
        + ", ..."
    )

    for cls in FOCUS_CLASSES:
        if total >= cfg.max_total_clips:
            print(f"[audioset] global cap {cfg.max_total_clips} reached.")
            break
        key = _slugify(cls)
        written = _download_class(cls, by_class[key], counts[key], cfg, seen[key])
        counts[key] += written
        total += written

    print(f"[audioset] done. {total} clips on disk total.")
    width = max(len(c) for c in FOCUS_CLASSES)
    for cls in FOCUS_CLASSES:
        key = _slugify(cls)
        n = counts.get(key, 0)
        flag = "  ZERO" if n == 0 else ("  CAP" if n >= cfg.clips_per_class else "")
        print(f"  {cls.ljust(width)}  {n:4d}/{cfg.clips_per_class}{flag}")
    return counts


async def stream_download_audioset_async(cfg: AudioSetConfig) -> dict[str, int]:
    return await asyncio.to_thread(stream_download_audioset, cfg)
