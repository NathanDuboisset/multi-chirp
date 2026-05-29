## TinyChirp on MicroFlow

Expanding [TinyChirp](https://github.com/TinyPART/TinyChirp) ([paper](https://arxiv.org/abs/2407.21453)) to more bird species, deployed in Rust via [MicroFlow](https://github.com/matteocarnelos/microflow-rs) ([paper](https://arxiv.org/pdf/2409.19432)) on Ariel OS.

## Layout

- `building/`: Python (datasets, training, PTQ, eval)
- `src/`, `Cargo.toml`, `laze-project.yml`: Rust firmware
- `datasets/<collection>/{training,validation,testing}/<Class>/`
- `models/<collection>/<model>.{keras,tflite}`
- `results/<collection>/<model>.{json,npz}`

## Setup

- `cd building && uv sync` ([uv](https://docs.astral.sh/uv/))
- `building/.env`:
  - `XC_API_KEY=...` ([xeno-canto](https://xeno-canto.org/account/api))
  - `EBIRD_API_KEY=...` ([ebird](https://ebird.org/api/keygen), required)

## Pipelines

Each pipeline is driven by its own notebooks; instructions live there.

- `building/scaling/`: taxonomic scaling sweep (k = 2..N)
- `building/geographic_task/`: place + target species, multi-label model, scored on long recordings
- `building/geographic_scale/`: scaling sweep rooted in a place

Models: `cnn1d`, `sincnet`, `leaf`, `transformer`, `squeezenet_time`, `mel_cnn`, `mel_cnn_2`, `squeezenet_mel` (see `building.models.available_models()`).

## Deploy

```bash
laze build -b {board-id} run
```

Binary size:

```bash
runtime=build/bin/{board-id}/cargo/thumbv8m.main-none-eabihf/release/tiny-chrip-microflow
arm-none-eabi-size "$runtime"
nm --print-size --size-sort --demangle=rust --radix=d "$runtime"
```
