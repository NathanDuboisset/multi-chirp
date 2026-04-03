## TinyChirp on MicroFlow

Trying to expand tinychirp to more bird species
Using microflow and ariel os as the infrastructure, for a full Rust code

- MicroFlow repo: `https://github.com/matteocarnelos/microflow-rs`
- MicroFlow paper: `https://arxiv.org/pdf/2409.19432`

Based of the following :

- TinyChirp repo: `https://github.com/TinyPART/TinyChirp`
- TinyChirp paper: `https://arxiv.org/abs/2407.21453`

## How to build models

uv from astral (https://docs.astral.sh/uv/) is used to manage python deps. Since torch and tensorflow can be conflicting on versions, there is two folders to generate, each having its own env,
In a folder, run 
```
uv sync
```
then run notebooks with the .venv created in that folder

### How to run

From the repo root:

```bash
laze build -b {your-board-ariel-id} run
```
This builds the firmware, runs the TinyChirp model on test clips and prints predictions + latency over serial.

### Check RAM / Flash usage

Binary path after build:

```text
build/bin/{your-board-ariel-id}/cargo/thumbv8m.main-none-eabihf/release/tiny-chrip-microflow
```

Example:

```bash
runtime_file_path=build/bin/{your-board-ariel-id}/cargo/thumbv8m.main-none-eabihf/release/tiny-chrip-microflow

arm-none-eabi-size "$runtime_file_path"
nm --print-size --size-sort --demangle=rust --radix=d "$runtime_file_path"
```

## Dataset pipeline (`building/`)

Multi-class bird sound datasets built from [Xeno-canto](https://xeno-canto.org/).

### Setup

```bash
cd building
uv sync
echo "XC_API_KEY=your_key_here" > ../.env
```

### Steps

1. **Select species** — `taxonomy.py` downloads the eBird taxonomy and queries XC for recording counts. Species with <100 recordings are excluded. Four collections are assembled:
   - `diff_species` — N species from different genera
   - `diff_genus` — N species, same genus
   - `diff_family` — N species, different genera, same family
   - `diff_order` — N species, different families, same order

2. **Download** — `download.py` fetches MP3s from XC API v3 at original sample rate into `raw_dataset/<Genus_species>/`.

3. **Validate & split** — `dataset.py` runs BirdNET (threshold 0.92) on each recording, extracts confirmed 3 s clips at **16 kHz**, and assembles:
   ```
   datasets/<collection>/
       training/<Species>/
       validation/<Species>/
       testing/<Species>/
   ```

4. **Train** — use `time_models.py` (CNN-1D, SincNet, BiLSTM) or `mel_models.py` (CNN-2D, MobileNetV2, CRNN). All models take `n_classes` as argument and output sigmoid probabilities.

5. **Evaluate** — `evaluate.py` provides `evaluate_multiclass()` with per-class AUC, confusion matrix, and classification report.