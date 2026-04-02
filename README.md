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