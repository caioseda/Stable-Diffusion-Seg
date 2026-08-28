# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

SDSeg (Stable Diffusion Segmentation): a MICCAI 2024 paper's official implementation, reusing a Stable-Diffusion-style
latent diffusion model (autoencoder + denoising UNet) for biomedical image segmentation with a single-step reverse
process. This fork extends the original with a VFSS (videofluoroscopic swallowing study) dataset and a temporal
context module for using multi-frame windows as conditioning (`ldm/models/diffusion/videoSDSeg.py`,
`ldm/modules/temporal_modules/tcm.py`) — this part is actively evolving and not yet reflected in `docs/`.

## Environment

Conda envs already exist locally: `sdseg` (primary, Python 3.8, PyTorch 1.11/CUDA 11.3), plus `sdseg-cpython` and
`vfss`. To create from scratch:

```bash
conda env create -f environment.yaml
conda activate sdseg
pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
pip install -e git+https://github.com/openai/CLIP.git@main#egg=clip
pip install -e .
```

There is no test suite, linter, or CI config in this repo — do not invent one unless asked.

## Commands

Training (background, logged to `nohup/`, tensorboard-enabled automatically):

```bash
nohup python -u main.py --base configs/SDSeg/<config>.yaml -t --gpus 0, --name <experiment_name> > nohup/<experiment_name>.log 2>&1 &
tail -f nohup/<experiment_name>.log
tensorboard --logdir logs --port 6006 --host 0.0.0.0
```

`commands.md` in the repo root is the user's personal scratchpad of recent training invocations for this fork's VFSS
work — check it for currently-running/likely-relevant configs, but it's not authoritative documentation.

Inference / evaluation, after training (see "Important Files" below for what to edit first):

```bash
python -u scripts/slice2seg.py --dataset cvc                       # single inference pass
python -u scripts/slice2seg.py --dataset cvc --times 10 --save_results  # stability evaluation, saves to ./outputs/
```
`scripts/slice2seg.py` requires manually editing hardcoded run/checkpoint paths before use. Stability results are
analyzed in `scripts/stability_evaluation.ipynb`.

Downloading pretrained weights (autoencoder + conditioning stage, and the LSUN-churches UNet used to init training):

```bash
bash scripts/download_first_stages_f8.sh
bash scripts/download_models_lsun_churches.sh
```

## Architecture

### Inheritance chain: `SDSeg <- LatentDiffusion <- DDPM`

The whole model is defined in one file per variant:
- `ldm/models/diffusion/SDSeg.py` — original 2D single-frame version.
- `ldm/models/diffusion/videoSDSeg.py` — same class hierarchy, plus a `TemporalContextBlock` for window-of-frames
  conditioning (VFSS work). The two files are near-duplicates by design (see the module docstring in each); check
  which one a config's `model.target` points at before editing.

Layered responsibilities:
- `DDPM`: generic pixel-space diffusion — noise schedule, `q_sample`/`p_sample`, `p_losses`, EMA, training loop hooks.
- `LatentDiffusion`: moves diffusion into autoencoder latent space, adds a conditioning stage, latent-space sampling
  and decoding, and the split-input (patch-based) fold/unfold machinery for large images.
- `SDSeg`: reinterprets latents as segmentation masks — target latent is the mask, condition latent is the input
  image, evaluation is Dice/IoU (`log_dice`), logging is mask-focused (`log_images`).

Two authoritative deep-dive docs exist for this and should be read before non-trivial changes to the model classes:
- `docs/DIFFUSION_CLASS_FLOW.md` — method-by-method reference across all three classes, with a recommended reading
  order and line-number anchors.
- `docs/IMPLEMENTATION_TRICKS.md` — the *why* behind non-obvious choices (EMA-as-eval-model, `scale_factor` latent
  normalization, frozen first-stage vs. trainable conditioner, the extra `loss_seg` term, single-step "direct"
  inference, concat-conditioning channel doubling, etc). Read this before assuming something is dead code or a bug.
- `docs/SDSeg_FUNCTIONS.md` / `docs/UNET_MODEL.md` — narrower references for the `SDSeg` class alone and for
  `ldm/modules/diffusionmodules/openaimodel.py::UNetModel` respectively.

Key non-obvious facts (see `docs/IMPLEMENTATION_TRICKS.md` for the full reasoning):
- Evaluation/logging run inside `ema_scope()`, so monitored metrics reflect EMA weights, not raw training weights.
- Checkpoints are usually loaded with `load_only_unet: true` — only the UNet transfers; first-stage/conditioner are
  built fresh from their own configs.
- The training monitor is `val_avg_dice` (set in `log_dice`), not diffusion loss — checkpoint selection is
  segmentation-quality-driven.
- SDSeg's single-step "direct" inference (predict `x0` from one UNet call at `t = num_timesteps - 1`) is the paper's
  core trick; `ddim`/`plms` iterative sampling also exist for comparison.

### Config-driven instantiation (OmegaConf + `ldm/util.py:instantiate_from_config`)

Every YAML under `configs/SDSeg/*.yaml` has `model` / `data` / `lightning` sections. Nested `target`/`params` dicts
are resolved recursively via `instantiate_from_config` — new model/data/module classes just need to be importable by
their dotted path, no registry to update. `configs/latent-diffusion/` and `configs/autoencoder/` hold configs for the
pre-SDSeg-refactor / autoencoder-pretraining stages.

### Training entrypoint: `main.py`

Standard latent-diffusion-style Lightning entrypoint: `get_parser()` builds the CLI, `DataModuleFromConfig` wraps the
`data.train`/`validation`/`test` configs into a `LightningDataModule`, and custom callbacks
(`SetupCallback`, `ImageLogger`, `CUDACallback`) are wired from the config's `lightning.callbacks` section.
`ImageLogger.log_dice_frequency` controls how often `SDSeg.log_dice` runs during training.

### Data layer: `ldm/data/`

Each dataset module (`vfss_new.py`, `cvc.py`, `sts3d.py`, `synapse.py`, `refuge2.py`, `kseg.py`, ...) exposes
Train/Val/Test dataset classes referenced by config `target` paths. `ldm/data/vfss_new.py::VFSSWindowImageDataset` is
the current VFSS loader — it reads a CSV frame table (`inca-video-frame-dataset.csv`) and can return an odd-sized
temporal window of frames (`window_size`) rather than a single image, feeding the temporal conditioning path in
`videoSDSeg.py`. Raw dataset preprocessing scripts live under `data/<dataset>/`.

### Notebooks

`notebooks/` holds exploratory/debugging notebooks (training forward-step inspection, TCM shape debugging, VFSS mask
range debugging, stability evaluation). These are scratch/debug artifacts, not documentation — don't treat their
contents as settled design decisions.
