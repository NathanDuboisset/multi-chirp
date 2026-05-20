# Paper TODO — TinyChirp-caliber writeup

Reference: `notebooks_thesis/2407.21453v2.pdf` (TinyChirp, IEEE IS2 2024). Goal: empirical, reproducible, on-device numbers — not just val accuracy curves.

## Story / contributions (decide first)
- [ ] One-sentence pitch: what does multi-chirp do that TinyChirp doesn't? (multi-species? geographic scaling? new architectures: LEAF / SincNet / mel-CNN / 1D-CNN?)
- [ ] Pick 2–4 concrete claims the paper proves. Every figure/table must support one.
- [ ] Decide scope: single-species binary (like TinyChirp) vs. multi-species vs. geographic-task. Don't mix in the writeup.

## Datasets
- [ ] Lock the dataset version (commit hash of `raw_dataset/` build script + manifest of clip IDs).
- [ ] Per-source clip counts table: Xeno-Canto, Macaulay (eBird), AudioSet, BirdNET-no-bird — by class and by split.
- [ ] **Group-aware splitting audit**: confirm no recording leaks across train/val/test (just fixed in `building/data/dataset.py`). Add a unit test or notebook cell that asserts disjoint recording IDs per split.
- [ ] Document label-curation pipeline (BirdNET confidence threshold, segment length, padding, downsampling rate).
- [ ] Class-imbalance numbers + how it's handled (round-robin pooling, per-species cap).
- [ ] Pilot analysis: average STFT of target vs non-target (TinyChirp Fig. 2 analogue) to justify SR and bandpass choices.

## Baselines (must-have for credibility)
- [ ] **Signal-processing baseline**: bandpass + power threshold (TinyChirp `Baseline`). Tune `t_low` / `t_high` on train, report on test.
- [ ] **BirdNET** (off-the-shelf): inference on the same test set, same classes — anchors absolute accuracy.
- [ ] **Random / majority-class** baselines for the imbalance sanity check.
- [ ] At least one published TinyML model re-implemented or cited with matching input (e.g. CNN-Mel from TinyChirp).

## Models to benchmark
- [ ] Spectrogram-input: `models/mel_cnn.py` (current), one SqueezeNet-Mel variant.
- [ ] Time-series-input: `models/cnn1d.py`, `models/sincnet.py`, `models/leaf.py`.
- [ ] (Optional) tiny transformer on raw waveform — TinyChirp's Transformer-Time was their best time-series model.
- [ ] Per-model card: input shape, param count, layer table (mirror TinyChirp Table II).

## Training protocol
- [ ] Fixed seed, fixed split (SPLIT_SEED=42 already), fixed augmentation pipeline — document each.
- [ ] Same optimizer/LR schedule/epochs across all models, or document why not.
- [ ] Early stopping on val F2 (recall-weighted — matches bioacoustic ops).
- [ ] Log everything (loss, val curves, confusion matrices) to disk per run, not just stdout.

## Predictive metrics
- [ ] Accuracy, Precision, Recall, F1, **F2** (recall-weighted, standard in bioacoustics).
- [ ] ROC + AUC per model, single figure.
- [ ] Threshold sweep plot (TinyChirp Fig. 5): metric vs decision threshold `t`.
- [ ] Confusion matrix at the F2-optimal threshold.
- [ ] **Test set is held out** — pick threshold on val, then evaluate once on test. No peeking.

## On-device measurements (this is what makes it a TinyML paper)
- [ ] Pick target hardware. TinyChirp used nRF52840DK (Cortex-M4, 1 MB Flash / 256 kB RAM). State ours.
- [ ] **RAM (peak activation)** per model — measure, don't estimate.
- [ ] **Flash (weights + code)** per model.
- [ ] **Inference latency** per 3 s clip, on-device.
- [ ] **Preprocessing latency** (mel-spectrogram cost) — TinyChirp's key finding was that mel preprocessing dominates.
- [ ] **Energy per inference** (mJ) — ampere-meter on the dev board, or a calibrated estimate. Idle power separately.
- [ ] Table: Model | RAM | Flash | Latency-infer | Latency-prep | Power | Energy/inf (TinyChirp Table VI).
- [ ] Flag any model that OOMs and report it (TinyChirp did this for SqueezeNet-Mel/Time).

## Compression / optimization
- [ ] Post-training quantization (int8 weights + activations). Report accuracy delta.
- [ ] Quantization-aware training as a comparison if PTQ degrades.
- [ ] Partial convolution / streaming inference for the 1D-CNN — TinyChirp's main systems contribution.
- [ ] Pruning if it helps (optional).

## Two-stage decision strategy (TinyChirp's main systems idea — worth replicating)
- [ ] Stage 1: cheap signal-processing pre-screen, discards silent/non-bird audio.
- [ ] Stage 2: TinyML model verifies positives.
- [ ] Report deployment lifetime extension: e.g. "2 weeks → N weeks on the same SD card / battery" using measured power + storage-per-positive.

## Ablations / sensitivity
- [ ] Sample rate: 16 kHz vs 22.05 / 32 kHz — impact on accuracy and energy.
- [ ] Segment length: 1 s / 3 s / 5 s.
- [ ] Mel bins / FFT size.
- [ ] Train-set size scaling (this is what `building/scaling/` is for — make it a figure).
- [ ] Geographic generalisation: train on region A, test on region B (this is what `building/geographic_task/` and `geographic_scale/` are for — also a figure).

## Reproducibility
- [ ] Public repo with: data manifest, training scripts, model weights, on-device firmware, eval notebooks.
- [ ] One command per result. Pin `uv.lock` / `Cargo.lock` (already done).
- [ ] Hardware setup photo + wiring for the energy measurement.

## Writeup structure (mirror TinyChirp, ~10 pages IEEE)
- [ ] Abstract + contributions list.
- [ ] Background & related work (TinyML on audio, bioacoustic ML, BirdNET).
- [ ] Monitoring scenario / use-case.
- [ ] Evaluation metrics (define F2 and why it matters).
- [ ] Methodology: data acquisition, preprocessing, pilot analysis.
- [ ] Baseline + decision strategy.
- [ ] Models (spectrogram, time-series, learnable front-ends — LEAF/SincNet is our angle).
- [ ] Optimizations (PTQ, partial conv).
- [ ] Performance evaluation: predictive + on-device, side-by-side.
- [ ] Conclusion: deployment lifetime extension as the headline number.

## Risks / things to watch
- [ ] Recording-level leakage (now fixed — verify in the regenerated dataset).
- [ ] AudioSet ytid offsets (`_30`, `_80` suffix) — same video can produce multiple clips, also leakage risk (now grouped).
- [ ] BirdNET-labelled "no bird" clips: confirm they're really no-bird, not just low-confidence bird (mislabel risk).
- [ ] Class collapse on imbalanced data — check per-class recall, not just macro.
- [ ] On-device numbers measured in `release` builds with same compiler flags as deployment.
