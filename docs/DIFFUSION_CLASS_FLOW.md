# Diffusion Class Flow

This document explains how the three main classes in `ldm/models/diffusion/SDSeg.py` fit together:

1. `DDPM`
2. `LatentDiffusion`
3. `SDSeg`

The goal is not just to list methods, but to make the understanding flow from the most general diffusion logic to the segmentation-specific implementation.

## The Big Picture

These classes form an inheritance chain:

`SDSeg -> LatentDiffusion -> DDPM`

You can read them as layers of specialization:

- `DDPM` defines the base diffusion process.
- `LatentDiffusion` moves diffusion from image space into latent space and adds conditioning.
- `SDSeg` adapts latent diffusion for segmentation.

If you understand the classes in that order, the file becomes much easier to reason about.

## What Each Class Is Responsible For

### `DDPM`

Core question it answers:

"How do I train and sample a diffusion model at all?"

It provides:

- diffusion schedule construction
- forward noising
- reverse denoising
- loss computation
- training and validation hooks
- generic image logging

### `LatentDiffusion`

Core question it answers:

"How do I run diffusion on autoencoder latents instead of raw pixels, and how do I condition it on something else?"

It adds:

- a first-stage autoencoder
- a conditioning encoder
- latent encoding and decoding
- conditional UNet calls
- latent-space sampling and logging

### `SDSeg`

Core question it answers:

"How do I use latent diffusion to predict segmentation masks from input images?"

It changes:

- what the target `x` means
- what the condition `c` means
- how batch input is unpacked
- how training loss is augmented for segmentation
- how evaluation computes Dice and IoU
- how qualitative logs are tailored for segmentation debugging

## The Fastest Way To Read The Code

If you are new to this file, this is the best reading order:

1. `DDPM.__init__`
2. `DDPM.register_schedule`
3. `DDPM.q_sample`
4. `DDPM.p_losses`
5. `DDPM.p_sample`
6. `LatentDiffusion.get_input`
7. `LatentDiffusion.apply_model`
8. `LatentDiffusion.p_losses`
9. `SDSeg.get_input`
10. `SDSeg.p_losses`
11. `SDSeg.log_dice`
12. `SDSeg.log_images`

## End-to-End Flow

### Training Flow

At a high level, training goes like this:

1. Read a batch from the dataset.
2. Convert the batch into the model input format.
3. Encode into latent space when needed.
4. Sample a random diffusion timestep.
5. Add noise to the target latent.
6. Ask the UNet to predict the noise or clean target.
7. Compute the loss.
8. Log metrics and update the optimizer.

How the three classes divide that work:

- `DDPM` handles steps 4 to 8 in the general case.
- `LatentDiffusion` changes step 2 and step 3 by introducing first-stage and conditioning encoders.
- `SDSeg` changes step 2 and step 7 for segmentation-specific training.

### Inference Flow

At a high level, inference goes like this:

1. Build the conditioning input.
2. Start from noise in latent space.
3. Reverse the diffusion process directly or iteratively.
4. Decode the final latent.
5. Convert the decoded output into a task-specific result.

How the three classes divide that work:

- `DDPM` defines the generic reverse diffusion sampler.
- `LatentDiffusion` samples latents and decodes them back to image space.
- `SDSeg` turns decoded outputs into masks and computes Dice/IoU.

## `DDPM` Reference

Source range:

- `ldm/models/diffusion/SDSeg.py:58-434`

### Role

`DDPM` is the base diffusion model in image space. It does not know about autoencoders, latent spaces, segmentation masks, or image conditioning. It only knows how to:

- define a noise schedule
- corrupt clean samples
- train a denoiser
- reverse the diffusion process

### Method Groups

#### Setup and State

##### `__init__(...)`

Location: `ldm/models/diffusion/SDSeg.py:60`

What it does:

- stores the core diffusion configuration
- builds the wrapped diffusion UNet through `DiffusionWrapper`
- optionally enables EMA
- optionally loads a checkpoint
- registers the diffusion schedule
- initializes learned log variance if enabled

Why it matters:

- this is where the model decides whether it predicts `eps` or `x0`
- it also defines image size, channels, and the training objective style

##### `register_schedule(...)`

Location: `ldm/models/diffusion/SDSeg.py:130`

What it does:

- creates the beta schedule
- derives all alpha-related cumulative products
- stores buffers used by forward and reverse diffusion
- prepares `lvlb_weights`

Why it matters:

- nearly every diffusion math helper later depends on these buffers

##### `ema_scope(context=None)`

Location: `ldm/models/diffusion/SDSeg.py:185`

What it does:

- temporarily swaps current weights for EMA weights
- restores normal weights afterward

Why it matters:

- evaluation and logging often use EMA weights because they are usually more stable

##### `init_from_ckpt(path, ignore_keys=list(), only_model=False)`

Location: `ldm/models/diffusion/SDSeg.py:199`

What it does:

- loads a checkpoint
- optionally ignores key prefixes
- loads either the full module or only the wrapped diffusion model

#### Forward Diffusion Math

##### `q_mean_variance(x_start, t)`

Location: `ldm/models/diffusion/SDSeg.py:217`

What it does:

- returns the mean and variance of `q(x_t | x_0)`

##### `predict_start_from_noise(x_t, t, noise)`

Location: `ldm/models/diffusion/SDSeg.py:229`

What it does:

- reconstructs an estimate of `x_0` from noisy `x_t` and predicted noise

Why it matters:

- this is one of the core reverse-diffusion formulas

##### `q_posterior(x_start, x_t, t)`

Location: `ldm/models/diffusion/SDSeg.py:235`

What it does:

- computes the posterior distribution `q(x_{t-1} | x_t, x_0)`

##### `q_sample(x_start, t, noise=None)`

Location: `ldm/models/diffusion/SDSeg.py:287`

What it does:

- adds diffusion noise to `x_start` to produce `x_t`

This is the function that turns a clean sample into a noisy training sample.

#### Reverse Diffusion and Sampling

##### `p_mean_variance(x, t, clip_denoised)`

Location: `ldm/models/diffusion/SDSeg.py:244`

What it does:

- runs the UNet on `x_t`
- reconstructs `x_0`
- builds the reverse-step Gaussian parameters

##### `p_sample(x, t, clip_denoised=True, repeat_noise=False)`

Location: `ldm/models/diffusion/SDSeg.py:257`

What it does:

- performs one reverse diffusion step

##### `p_sample_loop(shape, return_intermediates=False)`

Location: `ldm/models/diffusion/SDSeg.py:266`

What it does:

- starts from random noise
- repeatedly applies `p_sample`
- optionally collects intermediate states

##### `sample(batch_size=16, return_intermediates=False)`

Location: `ldm/models/diffusion/SDSeg.py:281`

What it does:

- convenience wrapper around `p_sample_loop`

#### Loss and Training Path

##### `get_loss(pred, target, mean=True)`

Location: `ldm/models/diffusion/SDSeg.py:292`

What it does:

- computes either L1 or L2 loss

##### `p_losses(x_start, t, noise=None)`

Location: `ldm/models/diffusion/SDSeg.py:307`

What it does:

- samples noise
- creates `x_noisy`
- runs the denoiser
- compares prediction to the chosen target
- computes `loss_simple`, `loss_vlb`, and total loss

This is the main `DDPM` training objective.

##### `forward(x, *args, **kwargs)`

Location: `ldm/models/diffusion/SDSeg.py:336`

What it does:

- samples a random timestep
- delegates to `p_losses`

##### `get_input(batch, k)`

Location: `ldm/models/diffusion/SDSeg.py:342`

What it does:

- extracts a tensor from the batch
- ensures channel-last inputs become channel-first

##### `shared_step(batch)`

Location: `ldm/models/diffusion/SDSeg.py:350`

What it does:

- gets the model input from the batch
- runs one forward loss computation

##### `training_step(batch, batch_idx)`

Location: `ldm/models/diffusion/SDSeg.py:355`

What it does:

- calls `shared_step`
- logs train metrics
- logs learning rate when a scheduler is enabled

##### `validation_step(batch, batch_idx)`

Location: `ldm/models/diffusion/SDSeg.py:371`

What it does:

- evaluates once with current weights
- evaluates again under EMA weights
- logs both sets of metrics

##### `on_train_batch_end(...)`

Location: `ldm/models/diffusion/SDSeg.py:379`

What it does:

- updates the EMA model after each train batch

#### Logging and Visualization

##### `_get_rows_from_list(samples)`

Location: `ldm/models/diffusion/SDSeg.py:383`

What it does:

- converts a list of image tensors into a tiled grid

##### `log_images(batch, N=8, n_row=2, sample=True, return_keys=None, **kwargs)`

Location: `ldm/models/diffusion/SDSeg.py:391`

What it does:

- logs input images
- logs forward diffusion snapshots
- logs reverse denoising samples

This is the generic image-space debug logger for base `DDPM`.

##### `configure_optimizers()`

Location: `ldm/models/diffusion/SDSeg.py:428`

What it does:

- builds an `AdamW` optimizer
- includes learned log variance if enabled

### `DDPM` In One Sentence

`DDPM` is the minimal diffusion engine: schedule, noising, denoising, loss, sampling, and logging.

## `LatentDiffusion` Reference

Source range:

- `ldm/models/diffusion/SDSeg.py:437-1426`

### Role

`LatentDiffusion` takes the generic `DDPM` logic and makes it practical by operating in latent space instead of raw image space.

That means:

- a first-stage model encodes inputs into latents
- diffusion runs on those latents
- a condition encoder produces the information the UNet conditions on
- the final latent is decoded back to image space

### What It Adds Conceptually

Compared with `DDPM`, `LatentDiffusion` introduces three big ideas:

1. First-stage autoencoding
2. Conditional generation
3. Latent-space sampling and decoding

### Method Groups

#### Setup and Model Construction

##### `__init__(...)`

Location: `ldm/models/diffusion/SDSeg.py:439`

What it does:

- defines conditioning configuration
- decides the conditioning key style
- calls the base `DDPM` constructor
- builds the first-stage model
- builds the conditioning model
- handles latent scaling
- optionally loads a checkpoint

Why it matters:

- this is where the class stops being a plain diffusion model and becomes a latent conditional diffusion model

##### `make_cond_schedule()`

Location: `ldm/models/diffusion/SDSeg.py:484`

What it does:

- creates a reduced conditioning timestep schedule

Why it matters:

- it supports the optional "shortened conditioning schedule" path

##### `on_train_batch_start(batch, batch_idx, dataloader_idx)`

Location: `ldm/models/diffusion/SDSeg.py:491`

What it does:

- optionally computes latent standard deviation on the very first batch
- sets `scale_factor = 1 / std`

##### `register_schedule(...)`

Location: `ldm/models/diffusion/SDSeg.py:506`

What it does:

- calls `DDPM.register_schedule`
- additionally prepares the conditioning schedule if needed

##### `instantiate_first_stage(config)`

Location: `ldm/models/diffusion/SDSeg.py:515`

What it does:

- builds the first-stage autoencoder
- freezes it
- keeps it permanently in eval mode

##### `instantiate_cond_stage(config)`

Location: `ldm/models/diffusion/SDSeg.py:522`

What it does:

- builds the conditioning model
- either freezes it or leaves it trainable
- supports special modes:
  - same as first stage
  - unconditional

#### Latent Encoding and Decoding

##### `_get_denoise_row_from_list(samples, desc='', force_no_decoder_quantization=False)`

Location: `ldm/models/diffusion/SDSeg.py:543`

What it does:

- decodes a list of latent states into a visualization grid

##### `get_first_stage_encoding(encoder_posterior)`

Location: `ldm/models/diffusion/SDSeg.py:555`

What it does:

- turns encoder output into the latent tensor used by diffusion
- samples from a KL posterior when needed
- applies `scale_factor`

##### `decode_first_stage(z, predict_cids=False, force_not_quantize=False)`

Location: `ldm/models/diffusion/SDSeg.py:740`

What it does:

- converts a latent tensor back to image space
- undoes latent scaling
- supports patch-based decode when split input mode is enabled

##### `differentiable_decode_first_stage(z, predict_cids=False, force_not_quantize=False)`

Location: `ldm/models/diffusion/SDSeg.py:800`

What it does:

- same purpose as `decode_first_stage`
- without the `@torch.no_grad()` wrapper

Use this when gradients through the decoder are needed.

##### `encode_first_stage(x)`

Location: `ldm/models/diffusion/SDSeg.py:860`

What it does:

- encodes an image tensor into latent space
- supports patch-based encode when split input mode is enabled

#### Conditioning

##### `get_learned_conditioning(c)`

Location: `ldm/models/diffusion/SDSeg.py:573`

What it does:

- runs the conditioning model
- returns the representation the UNet will consume

Important detail:

- if the conditioning encoder returns a Gaussian distribution, this method uses its mode

##### `get_input(batch, k, return_first_stage_outputs=False, force_c_encode=False, cond_key=None, return_original_cond=False, bs=None)`

Location: `ldm/models/diffusion/SDSeg.py:688`

What it does:

- reads the target tensor from the batch
- encodes it into latent `z`
- builds condition `c`
- optionally returns reconstructions and original conditioning input

This is the bridge from dataloader output to latent diffusion input.

##### `forward(x, c, *args, **kwargs)`

Location: `ldm/models/diffusion/SDSeg.py:904`

What it does:

- samples a timestep
- optionally re-encodes trainable conditioning
- optionally shortens conditioning schedule
- delegates to `p_losses`

#### Split-Input Helpers

These exist to support patch-wise encoding, decoding, and model application on large images.

##### `meshgrid(h, w)`

Location: `ldm/models/diffusion/SDSeg.py:598`

What it does:

- creates a coordinate grid

##### `delta_border(h, w)`

Location: `ldm/models/diffusion/SDSeg.py:605`

What it does:

- computes normalized distance-to-border values

##### `get_weighting(h, w, Ly, Lx, device)`

Location: `ldm/models/diffusion/SDSeg.py:619`

What it does:

- builds blending weights for overlapping patches

##### `get_fold_unfold(x, kernel_size, stride, uf=1, df=1)`

Location: `ldm/models/diffusion/SDSeg.py:635`

What it does:

- constructs `Fold` and `Unfold` helpers
- prepares overlap normalization and weighting

##### `_rescale_annotations(bboxes, crop_coordinates)`

Location: `ldm/models/diffusion/SDSeg.py:915`

What it does:

- rescales bounding boxes for cropped regions

This is mainly relevant to bounding-box conditioning paths.

#### Applying the Conditional UNet

##### `apply_model(x_noisy, t, cond, return_ids=False)`

Location: `ldm/models/diffusion/SDSeg.py:925`

What it does:

- converts conditioning into the format expected by `DiffusionWrapper`
- optionally applies the model patch by patch
- stitches patch outputs back together when split-input mode is active

Why it matters:

- this is the main place where latent input, timestep, and condition are finally handed to the wrapped diffusion model

#### Loss and Sampling

##### `_predict_eps_from_xstart(x_t, t, pred_xstart)`

Location: `ldm/models/diffusion/SDSeg.py:1028`

What it does:

- recovers epsilon from `x_t` and predicted `x_0`

##### `_prior_bpd(x_start)`

Location: `ldm/models/diffusion/SDSeg.py:1032`

What it does:

- computes the prior KL term in bits-per-dimension

##### `p_losses(x_start, cond, t, noise=None)`

Location: `ldm/models/diffusion/SDSeg.py:1046`

What it does:

- runs the conditional training objective in latent space
- computes the standard diffusion loss terms

Compared with `DDPM.p_losses`, the main change is that the model is now conditioned and runs on latents.

##### `p_mean_variance(x, c, t, clip_denoised, return_codebook_ids=False, quantize_denoised=False, return_x0=False, score_corrector=None, corrector_kwargs=None)`

Location: `ldm/models/diffusion/SDSeg.py:1081`

What it does:

- runs the conditional denoiser
- reconstructs `x_0`
- optionally quantizes
- optionally returns logits or `x_0`

##### `p_sample(...)`

Location: `ldm/models/diffusion/SDSeg.py:1113`

What it does:

- performs one conditional reverse step in latent space

##### `progressive_denoising(...)`

Location: `ldm/models/diffusion/SDSeg.py:1144`

What it does:

- runs reverse denoising while collecting intermediate `x_0` estimates

This is mainly useful for visualization.

##### `p_sample_loop(...)`

Location: `ldm/models/diffusion/SDSeg.py:1200`

What it does:

- iteratively samples a latent while conditioned on `cond`

##### `sample(cond, batch_size=16, return_intermediates=False, x_T=None, ...)`

Location: `ldm/models/diffusion/SDSeg.py:1251`

What it does:

- convenience wrapper around the conditional sample loop

##### `sample_log(cond, batch_size, ddim, ddim_steps, **kwargs)`

Location: `ldm/models/diffusion/SDSeg.py:1269`

What it does:

- chooses between DDIM sampling and the native sampling loop
- returns final samples plus intermediates

#### Logging and Visualization

##### `log_images(batch, N=8, n_row=4, sample=True, ddim_steps=200, ddim_eta=1., return_keys=None, quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True, plot_diffusion_rows=True, **kwargs)`

Location: `ldm/models/diffusion/SDSeg.py:1285`

What it does:

- logs inputs and reconstructions
- logs conditioning views
- logs diffusion rows
- logs sampled outputs
- optionally logs denoising rows
- optionally logs inpainting and outpainting
- optionally logs progressive denoising

This is the generic latent-diffusion visualization path that SDSeg later customizes.

##### `configure_optimizers()`

Location: `ldm/models/diffusion/SDSeg.py:1395`

What it does:

- builds the optimizer
- optionally includes trainable conditioning parameters
- optionally includes learned log variance
- optionally adds a scheduler

##### `to_rgb(x)`

Location: `ldm/models/diffusion/SDSeg.py:1420`

What it does:

- colorizes non-RGB tensor maps for visualization

### `LatentDiffusion` In One Sentence

`LatentDiffusion` is `DDPM` plus autoencoder latents, conditioning, latent-space sampling, and richer visualization.

## `SDSeg` Reference

Source range:

- `ldm/models/diffusion/SDSeg.py:1428-2299`

### Role

`SDSeg` takes `LatentDiffusion` and reinterprets it for segmentation:

- the diffusion target is a segmentation mask latent
- the condition is the input image latent
- evaluation is segmentation-specific
- logging is segmentation-specific

### What Changes Relative To `LatentDiffusion`

`SDSeg` does not replace the entire latent diffusion pipeline. It mostly overrides the places where task semantics matter:

- checkpoint loading
- training logging
- batch unpacking
- conditioning packaging for class ids
- segmentation-specific loss augmentation
- Dice and IoU evaluation
- segmentation-focused logging
- optimizer configuration

### Method Groups

#### Setup and Training Hooks

##### `__init__(first_stage_config, cond_stage_config, load_only_unet=True, num_classes=2, *args, **kwargs)`

Location: `ldm/models/diffusion/SDSeg.py:1441`

What it does:

- calls the parent constructor
- stores `num_classes`

##### `init_from_ckpt(path, ignore_keys=list(), only_model=True)`

Location: `ldm/models/diffusion/SDSeg.py:1450`

What it does:

- usually loads only the diffusion UNet from a checkpoint
- can also restore the full pipeline
- handles a known channel-mismatch case by zero-filling extra channels

##### `training_step(batch, batch_idx)`

Location: `ldm/models/diffusion/SDSeg.py:1529`

What it does:

- runs one SDSeg train step
- removes `train/loss_vlb` from progress-bar logging

##### `on_train_batch_start(batch, batch_idx, dataloader_idx)`

Location: `ldm/models/diffusion/SDSeg.py:1551`

What it does:

- initializes latent scaling from the first batch when `scale_by_std` is enabled
- initializes `val_avg_dice` logging

#### Segmentation Input and Forward Path

##### `get_denoise_row_from_list(samples, desc='', force_no_decoder_quantization=False)`

Location: `ldm/models/diffusion/SDSeg.py:1574`

What it does:

- similar to the parent helper
- returns both decoded image grids and latent grids

##### `get_input(batch, k, return_first_stage_outputs=False, force_c_encode=False, cond_key=None, return_original_cond=False, bs=None)`

Location: `ldm/models/diffusion/SDSeg.py:1601`

What it does:

- encodes the segmentation mask into latent `z`
- encodes the input image into condition `c`
- extracts `class_id`
- optionally returns reconstructions and original conditioning input

This is the main place where the segmentation interpretation becomes explicit.

##### `shared_step(batch, **kwargs)`

Location: `ldm/models/diffusion/SDSeg.py:1667`

What it does:

- gets segmentation-specific inputs
- calls `forward`

##### `forward(x, c, cls_id, *args, **kwargs)`

Location: `ldm/models/diffusion/SDSeg.py:1673`

What it does:

- samples a timestep
- packages the condition as hybrid conditioning:
  - `c_concat=[c]`
  - `c_crossattn=[cls_id]`
- delegates to `p_losses`

#### Segmentation Loss

##### `get_loss_seg_regression(x_start, x_noisy, t, model_output, seg_label=None)`

Location: `ldm/models/diffusion/SDSeg.py:1693`

What it does:

- reconstructs a clean latent mask estimate from the predicted noise
- compares it against the target latent mask

Important detail:

- despite the name, this is not a Dice loss
- it is a latent reconstruction loss

##### `p_losses(x_start, cond, t, seg_label, noise=None)`

Location: `ldm/models/diffusion/SDSeg.py:1704`

What it does:

- computes the standard diffusion noise-prediction loss
- computes an extra latent segmentation reconstruction loss
- combines them with the inherited VLB-style term

This is the main SDSeg training objective.

### Deep Dive: `log_dice`

##### `log_dice(data=None, save_dir=None, ddim_steps=50)`

Location: `ldm/models/diffusion/SDSeg.py:1763`

What it does:

- runs segmentation inference
- computes Dice and IoU
- logs debug examples

This is the main evaluation entry point for SDSeg.

How it flows:

1. Build or receive a dataloader.
2. Choose an inference style.
3. Create conditioning from the input image.
4. Predict a segmentation latent.
5. Decode that latent.
6. Convert the decoded output into a mask.
7. Compute Dice and IoU.
8. Aggregate metrics and debug grids.

Inference styles supported internally:

- `direct`
- `ddim`
- `plms`

The outer method currently evaluates `direct` and `ddim`.

Binary 2D interpretation:

- decode the predicted latent
- map values from `[-1, 1]` to `[0, 1]`
- average channels
- threshold at `0.5`
- compare foreground class `1` against the ground truth

Why it matters:

- this is where SDSeg stops being "a diffusion model" and becomes "a segmentation model with measurable Dice"

#### Segmentation Logging

##### `prepare_latent_to_log(latent)`

Location: `ldm/models/diffusion/SDSeg.py:2160`

What it does:

- arranges latent tensors into a logger-friendly grid

##### `log_images(batch, N=8, n_row=4, sample=True, ddim_steps=True, ddim_eta=1., return_keys=None, quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True, plot_diffusion_rows=True, **kwargs)`

Location: `ldm/models/diffusion/SDSeg.py:2169`

What it does:

- logs segmentation inputs
- logs segmentation latents
- logs reconstructions
- logs conditioning and conditioning latents
- logs forward diffusion rows
- logs sampled outputs
- logs latent samples
- logs progressive denoising

What makes it different from the parent `LatentDiffusion.log_images`:

- it logs segmentation latents explicitly
- it returns both image-space and latent-space visualization artifacts
- it is tailored for mask debugging rather than generic image generation

Important implementation detail:

- the method is named `log_images`, not `log_image`
- if `ddim_steps` is not `None`, it gets replaced internally by `self.num_timesteps // 5`

#### Optimization

##### `configure_optimizers()`

Location: `ldm/models/diffusion/SDSeg.py:2293`

What it does:

- optimizes the major UNet blocks explicitly
- may give `label_emb` a larger learning rate
- may include condition-stage parameters
- may include learned log variance
- may add a scheduler

### `SDSeg` In One Sentence

`SDSeg` is `LatentDiffusion` specialized so the target is a segmentation-mask latent and the outputs are evaluated as masks with Dice and IoU.

## How The Understanding Should Flow

If you want the cleanest mental progression, think of the classes like this:

### Step 1: Understand `DDPM`

Learn:

- what `x_start`, `x_t`, and `t` mean
- how `q_sample` corrupts data
- how `p_sample` reverses corruption
- how `p_losses` trains the denoiser

Until that makes sense, nothing above it will feel intuitive.

### Step 2: Understand What `LatentDiffusion` Changes

Then replace the meaning of `x`:

- in `DDPM`, `x` is the sample in image space
- in `LatentDiffusion`, `x` is the encoded latent of that sample

Then add conditioning:

- the model no longer denoises blindly
- it denoises while looking at a condition representation

### Step 3: Understand What `SDSeg` Reinterprets

Finally replace the meaning of the target and condition:

- target latent = segmentation mask latent
- condition latent = input image latent

Now the full segmentation story becomes:

1. encode a mask to latent space
2. add noise
3. condition on the image latent
4. denoise back toward the mask latent
5. decode the result into a mask
6. measure Dice and IoU

## Most Important Method Pairs Across Classes

These pairs are useful because each later class builds directly on the earlier one.

### Input Path

- `DDPM.get_input`
- `LatentDiffusion.get_input`
- `SDSeg.get_input`

This shows how the meaning of model input evolves from plain tensor to latent target plus condition plus class id.

### Training Objective

- `DDPM.p_losses`
- `LatentDiffusion.p_losses`
- `SDSeg.p_losses`

This shows how the objective evolves from generic diffusion loss to conditional latent diffusion loss to segmentation-aware latent loss.

### Logging

- `DDPM.log_images`
- `LatentDiffusion.log_images`
- `SDSeg.log_images`

This shows how visualization evolves from generic diffusion snapshots to latent diffusion visualization to segmentation-specific debugging.

## Practical Summary

If someone asks "where should I look for what?" in this file:

- look at `DDPM` for diffusion math and the generic training loop
- look at `LatentDiffusion` for autoencoder latents and conditioning
- look at `SDSeg` for segmentation semantics, Dice/IoU evaluation, and segmentation logging

## Related Doc

There is also a focused SDSeg-only reference:

- `docs/SDSeg_FUNCTIONS.md`

Use this file when you want the class hierarchy and flow.
Use the SDSeg-only file when you want a tighter task-specific reference for the segmentation class itself.
