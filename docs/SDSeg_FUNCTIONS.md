# SDSeg Function Reference

This document explains the methods defined on the rewritten `SDSeg` class in `ldm/models/diffusion/SDSeg.py`.

Scope:

- This is about the `SDSeg` class itself, starting at `class SDSeg(LatentDiffusion)`.
- It does not try to re-document every inherited method from `LatentDiffusion` or `DDPM`.
- The method you asked about as `log_image` is named `log_images` in the code.

Source anchor:

- `ldm/models/diffusion/SDSeg.py:1428`

## High-Level Mental Model

`SDSeg` is a segmentation version of latent diffusion:

1. The ground-truth segmentation mask is encoded into latent space.
2. Noise is added to that latent.
3. The UNet predicts the noise while being conditioned on the input image latent.
4. The predicted clean latent is decoded back into a segmentation mask.

In the common binary setup used by this repo:

- class `0` = background
- class `1` = foreground
- the image latent is passed through `c_concat`
- the class id is passed through `c_crossattn`

## Method Map

| Method | What it does |
| --- | --- |
| `__init__` | Builds the segmentation diffusion model and stores `num_classes`. |
| `init_from_ckpt` | Loads pretrained weights, usually only the diffusion UNet. |
| `training_step` | Runs one train step and logs training losses. |
| `on_train_batch_start` | Optionally estimates latent std on the first batch and sets `scale_factor`. |
| `get_denoise_row_from_list` | Decodes a list of latent states into image and latent grids for logging. |
| `get_input` | Converts a batch into segmentation latent, conditioning latent, raw tensors, and class ids. |
| `shared_step` | Pulls model inputs from the batch and calls `forward`. |
| `forward` | Samples diffusion timesteps, packages conditioning, and calls `p_losses`. |
| `get_loss_seg_regression` | Reconstructs the clean latent from predicted noise and compares it to the target latent. |
| `p_losses` | Computes the full SDSeg training loss. |
| `log_dice` | Runs inference and computes Dice/IoU metrics. |
| `prepare_latent_to_log` | Rearranges latent tensors into a grid for visualization. |
| `log_images` | Produces qualitative debug visualizations for inputs, latents, sampling, and denoising. |
| `configure_optimizers` | Builds the optimizer and optional scheduler. |

## Detailed Method Explanations

### `__init__(...)`

Location: `ldm/models/diffusion/SDSeg.py:1441`

What it does:

- Calls the parent `LatentDiffusion` constructor.
- Stores `self.num_classes`.

Why it matters:

- `num_classes=2` is the binary segmentation case used most often here.
- Multi-class inference and logging branches later depend on this value.

### `init_from_ckpt(path, ignore_keys=list(), only_model=True)`

Location: `ldm/models/diffusion/SDSeg.py:1450`

What it does:

- Loads model weights from a checkpoint.
- With `only_model=True`, it loads only the diffusion UNet weights.
- With `only_model=False`, it loads the full SDSeg pipeline.

Important behavior:

- The default path is segmentation fine-tuning from a pretrained diffusion UNet.
- Non-UNet checkpoint entries are removed when `only_model=True`.
- If a UNet tensor shape mismatches in a known 4D convolution case, it zero-fills extra channels instead of failing.
- Label embedding parameters are preserved separately if the current model has them.

Why it exists:

- SDSeg often starts from Stable Diffusion style pretrained weights rather than training from scratch.

### `training_step(batch, batch_idx)`

Location: `ldm/models/diffusion/SDSeg.py:1529`

What it does:

- Calls `shared_step(batch)`.
- Removes `train/loss_vlb` from progress-bar logging.
- Logs the remaining train metrics and global step.
- Logs learning rate when a scheduler is active.

Why it is different from the parent:

- It mostly exists to customize logging behavior, not to change the training objective.

### `on_train_batch_start(batch, batch_idx, dataloader_idx)`

Location: `ldm/models/diffusion/SDSeg.py:1551`

What it does:

- Runs only on the first batch of the first epoch.
- If `scale_by_std` is enabled, it encodes the input once, measures latent standard deviation, and sets:
  `scale_factor = 1 / std`.
- Initializes `val_avg_dice` logging with `0`.

Why it matters:

- Different datasets can produce latents with different scales.
- This hook normalizes latent magnitude early so diffusion training is more stable.

### `get_denoise_row_from_list(samples, desc='', force_no_decoder_quantization=False)`

Location: `ldm/models/diffusion/SDSeg.py:1574`

What it does:

- Takes a list of latent tensors from denoising steps.
- Decodes each latent into image space.
- Builds:
  - an image-space grid
  - a latent-space grid

When it is used:

- Mostly by `log_images` when plotting denoising or progressive sampling rows.

### `get_input(batch, k, return_first_stage_outputs=False, force_c_encode=False, cond_key=None, return_original_cond=False, bs=None)`

Location: `ldm/models/diffusion/SDSeg.py:1601`

What it does:

- Pulls the segmentation tensor from the batch using `DDPM.get_input`.
- Extracts `class_id`.
- Encodes the segmentation mask into latent space as `z`.
- Builds the conditioning tensor `c`, usually from the input image.
- Optionally returns decoded reconstructions and the original conditioning tensor.

Main outputs:

- `z`: segmentation latent target
- `c`: conditioning latent
- `x`: original segmentation tensor
- `cls_id`: per-sample class id

Important detail:

- In normal SDSeg training, the segmentation mask is the diffusion target and the image latent is the condition.

### `shared_step(batch, **kwargs)`

Location: `ldm/models/diffusion/SDSeg.py:1667`

What it does:

- Calls `get_input`.
- Sends the resulting tensors into `forward`.

Notes:

- This is a thin adapter between the dataloader batch and the loss path.

### `forward(x, c, cls_id, *args, **kwargs)`

Location: `ldm/models/diffusion/SDSeg.py:1673`

What it does:

- Samples a random diffusion timestep `t` for each batch item.
- Converts conditioning into the hybrid dict expected by the UNet:
  - `c_concat=[c]`
  - `c_crossattn=[cls_id]`
- Calls `p_losses(...)`.

Why it matters:

- This is where SDSeg turns raw batch-derived tensors into the actual diffusion training call.

### `get_loss_seg_regression(x_start, x_noisy, t, model_output, seg_label=None)`

Location: `ldm/models/diffusion/SDSeg.py:1693`

What it does:

- Reconstructs the clean latent mask with `predict_start_from_noise(...)`.
- Compares that reconstructed latent to the target latent `x_start`.

Important detail:

- Despite the name, this is not a Dice loss and not a pixel-space mask loss.
- It is a latent-space reconstruction loss.
- `seg_label` is accepted but not actually used inside the method.

### `p_losses(x_start, cond, t, seg_label, noise=None)`

Location: `ldm/models/diffusion/SDSeg.py:1704`

What it does:

- Samples or receives diffusion noise.
- Creates `x_noisy = q_sample(x_start, t, noise)`.
- Runs the UNet with `apply_model`.
- Computes:
  - `loss_seg`: latent mask reconstruction loss
  - `loss_simple`: standard diffusion prediction loss
  - `loss_vlb`: VLB-style auxiliary term
- Builds the final loss.

Final loss structure in practice:

`loss = l_simple_weight * weighted_noise_loss + original_elbo_weight * loss_vlb + mean(loss_seg)`

Important detail:

- The segmentation-specific part is `loss_seg`.
- `seg_label` is passed through but is not directly used in this implementation.

## Deep Dive: `log_dice`

Location: `ldm/models/diffusion/SDSeg.py:1763`

This is the main evaluation function of SDSeg.

### What `log_dice` returns

It returns two dictionaries:

- `metrics_dict`: scalar Dice and IoU metrics
- `seg_label_dict`: image/latent grids for logging debug examples

### What happens first

If `data is None`, the method automatically uses:

- `self.trainer.datamodule.datasets["test"]`

and wraps it in a `DataLoader(batch_size=1)`.

So this method can be called in two ways:

- from training/validation, letting the trainer provide test data
- from inference scripts, by passing a dataloader explicitly

### Supported inference modes

Inside `log_dice`, the nested helper `get_dice(...)` can run three sampler modes:

- `direct`
- `ddim`
- `plms`

In the current outer flow, the method actually evaluates:

- `direct`
- `ddim`

and does not call `plms` by default.

### What `direct` means

`direct` is the SDSeg single-step idea.

The code:

1. starts from random latent noise
2. uses the final diffusion timestep `t = num_timesteps - 1`
3. predicts noise once with the UNet
4. reconstructs the clean latent with `predict_start_from_noise`

So there is no iterative denoising loop in this mode.

### What `ddim` means

`ddim` uses the `DDIMSampler` and runs an iterative reverse process for `ddim_steps` steps.

This is mainly used as a comparison against the direct one-step path.

### Binary 2D path, step by step

For the common 2D binary case, the important branch is the `else` block starting near `ldm/models/diffusion/SDSeg.py:1956`.

For each sample:

1. Load `image` and `label`.
2. Resize both to `256 x 256`.
3. Build the conditioning dict:
   - `c_concat` gets the encoded input image latent.
   - `c_crossattn` is `[None]` for binary inference here.
4. Generate one latent prediction:
   - by direct one-step prediction, or
   - by DDIM sampling.
5. Decode the predicted latent with `decode_first_stage`.
6. Convert decoded values from `[-1, 1]` to `[0, 1]`.
7. Average channels.
8. Threshold at `0.5`.
9. Compare predicted foreground against ground truth foreground.
10. Accumulate Dice and IoU.

### How binary predictions are converted into masks

The binary conversion is the most important part to understand:

```python
x_samples_ddim = self.decode_first_stage(samples_pred[0])
x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
x_samples_ddim = x_samples_ddim.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
out_p = rearrange(x_samples_ddim.squeeze(0).cpu().numpy(), 'c h w -> h w c')
out = (out_p > 0.5)
```

Meaning:

- decode latent to image-like output
- rescale to probability-like range
- average channels
- threshold at `0.5`
- treat the result as a binary mask

### How Dice and IoU are computed

For each foreground class index from `1` to `num_classes - 1`, the method computes:

- `dice_score(prediction == idx, label == idx)`
- `iou_score(prediction == idx, label == idx)`

For binary segmentation, this means only class `1` is measured.

### What gets saved when `save_dir` is set

The method can save visual outputs to disk:

- predicted mask images
- logits/probability-like visualizations
- combined debug images

The exact saved set differs between the 3D and 2D branches.

### What gets logged for debugging

The method also collects example tensors for later visualization:

- predicted latents
- label latents
- predicted mask images
- label images
- logits visualizations
- conditioning images

These are packed into `seg_label_dict`.

### Why `log_dice` matters

This method is where SDSeg turns a latent diffusion model into segmentation metrics that you can actually evaluate.

If you want to understand:

- how inference is performed
- how one-step prediction differs from DDIM
- how masks are decoded and thresholded
- where Dice comes from

this is the most important method in the class.

## Deep Dive: `prepare_latent_to_log`

Location: `ldm/models/diffusion/SDSeg.py:2160`

What it does:

- Adds a singleton dimension.
- Rearranges latent tensors into a tiled grid.
- Returns a grid suitable for TensorBoard or image loggers.

Why it exists:

- Latents are not directly human-readable, so the logger needs them packed into a visual grid format.

## Deep Dive: `log_images`

Location: `ldm/models/diffusion/SDSeg.py:2169`

This is the main qualitative visualization method.

If `log_dice` tells you how well the model performs numerically, `log_images` shows what the model is doing visually.

### Important naming note

The code method is `log_images`, plural, not `log_image`.

### What `log_images` returns

It returns a dictionary of tensors keyed by names such as:

- `inputs`
- `latent`
- `reconstruction`
- `conditioning`
- `conditioning_latent`
- `diffusion_row`
- `diffusion_row_latent`
- `samples`
- `samples_latent`
- `progressive_row`
- `progressive_row_latent`

Depending on flags, some keys may be absent.

### What it reads from the batch

It calls:

`get_input(..., return_first_stage_outputs=True, force_c_encode=True, return_original_cond=True, bs=N)`

So it gets:

- `z`: segmentation latent
- `c`: conditioning latent
- `cls_id`
- `x`: original segmentation mask tensor
- `xrec`: reconstruction of `z`
- `xc`: original conditioning tensor

### Base visual outputs

The method always logs these core views:

- `inputs`: original segmentation masks
- `latent`: latent version of the segmentation masks, arranged as a grid
- `reconstruction`: decoded latent reconstruction of the masks

If the conditioning input is image-like, it also logs:

- `conditioning`
- `conditioning_latent`

### Diffusion row

When `plot_diffusion_rows=True`, the method visualizes the forward noising process:

1. Start from the clean latent `z`.
2. At selected timesteps, sample noise with `q_sample`.
3. Decode each noisy latent.
4. Build image and latent grids.

This answers:

- what the mask latent looks like as noise increases
- whether the first-stage decoder still produces interpretable structure at intermediate steps

### Sample outputs

When `sample=True`, the method runs generation using `sample_log(...)` under EMA weights.

It then logs:

- `samples`: decoded sampled outputs
- `samples_latent`: sampled latent outputs

If `plot_denoise_rows=True`, it also logs the full reverse denoising trajectory via `get_denoise_row_from_list(...)`.

### Progressive rows

When `plot_progressive_rows=True`, the method calls `progressive_denoising(...)` and logs a longer stepwise reverse process:

- `progressive_row`
- `progressive_row_latent`

This is useful when you want to inspect how structure appears across denoising steps.

### Important implementation detail: `ddim_steps`

The function signature suggests that `ddim_steps` controls the number of DDIM steps, but the code does this:

```python
use_ddim = ddim_steps is not None
if use_ddim:
    ddim_steps = self.num_timesteps // 5
```

So if you pass any non-`None` value, it is replaced by `self.num_timesteps // 5`.

That means:

- `ddim_steps` acts more like an on/off switch here
- the exact numeric argument is not honored in this implementation

### Why `log_images` matters

Use `log_images` when you want to inspect:

- whether mask latents reconstruct correctly
- whether conditioning latents look sensible
- what forward diffusion is doing
- what reverse denoising is producing
- whether failures come from the autoencoder, the condition encoder, or the UNet sampling path

## `configure_optimizers()`

Location: `ldm/models/diffusion/SDSeg.py:2293`

What it does:

- Creates an `AdamW` optimizer.
- Adds the major UNet blocks.
- Optionally adds:
  - `label_emb` parameters with `100x` learning rate
  - condition-stage parameters
  - `logvar`
- Optionally builds a `LambdaLR` scheduler from config.

Why it matters:

- This controls what parts of SDSeg actually learn during training.
- In multi-class mode, label embedding parameters are treated specially.

## End-to-End Flow Summary

If you want the shortest summary of SDSeg:

1. `get_input` encodes the mask target and the conditioning image.
2. `forward` packages the condition and samples random diffusion timesteps.
3. `p_losses` computes diffusion loss plus latent reconstruction loss.
4. `log_dice` runs inference and computes Dice/IoU.
5. `log_images` produces visual debugging outputs.

## Most Important Methods To Read First

If you only want the core behavior, read these in order:

1. `get_input`
2. `forward`
3. `p_losses`
4. `log_dice`
5. `log_images`
