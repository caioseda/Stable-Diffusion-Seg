# UNet Model Reference

This document explains the UNet used by SDSeg, implemented in:

- `ldm/modules/diffusionmodules/openaimodel.py`

The main class is:

- `UNetModel`

This is the denoising network inside the diffusion model. During training and inference, it is the model that looks at a noisy latent `x_t`, the diffusion timestep `t`, and optional conditioning, and predicts what should be removed or reconstructed.

## Scope

This document focuses on:

- `UNetModel`
- the blocks it is built from
- how timestep conditioning works
- how attention works here
- how skip connections are assembled
- how SDSeg uses this UNet in practice

It does not try to document every other architecture in the repo. In particular:

- `EncoderUNetModel` is not the main SDSeg denoiser and is out of scope here

## Big Picture

The diffusion pipeline in this repository works like this:

1. The first-stage model encodes a mask into a latent.
2. Noise is added to that latent.
3. `UNetModel` receives the noisy latent.
4. `UNetModel` also receives the timestep embedding.
5. It may also receive conditioning information.
6. It predicts either the noise or the clean target, depending on diffusion parameterization.

So the UNet is the actual denoiser at the center of the diffusion process.

## Where The UNet Sits In SDSeg

In the SDSeg stack:

- `DDPM` defines diffusion math
- `LatentDiffusion` defines latent-space diffusion
- `SDSeg` defines segmentation-specific semantics
- `UNetModel` is the neural network used by all of them for denoising

In code, the UNet is wrapped by `DiffusionWrapper` and then called through methods like:

- `LatentDiffusion.apply_model(...)`
- `SDSeg.p_losses(...)`
- `SDSeg.log_dice(...)`

## What Input The UNet Actually Sees

From the point of view of `UNetModel.forward(...)`, the inputs are:

- `x`: the noisy latent tensor
- `timesteps`: the diffusion timestep for each batch item
- `context`: optional cross-attention context
- `y`: optional class labels

Source:

- `ldm/modules/diffusionmodules/openaimodel.py:713`

Important detail:

- In the SDSeg configs in this repo, conditioning is usually `concat`, not cross-attention.
- That means the conditioning image latent is concatenated into the input channels before the UNet sees it.
- So in the common SDSeg setup here, `context` is usually not the main mechanism. The main mechanism is extra input channels.

## How SDSeg Configures The UNet

A representative SDSeg config is:

- `configs/SDSeg/cvc-ldm-kl-8.yaml`

Relevant values there:

- `image_size: 32`
- `in_channels: 8`
- `out_channels: 4`
- `model_channels: 192`
- `attention_resolutions: [1, 2, 4, 8]`
- `num_res_blocks: 2`
- `channel_mult: [1, 2, 2, 4, 4]`
- `num_heads: 8`
- `use_scale_shift_norm: True`
- `resblock_updown: True`
- `dropout: 0.2`

Why `in_channels` is `8`:

- the segmentation latent has 4 channels
- the conditioning image latent also has 4 channels
- concat conditioning stacks them together
- so the UNet input becomes 8 channels

Why `out_channels` is `4`:

- the UNet predicts a latent-shaped output matching the target latent channel count

## High-Level Architecture

`UNetModel` has the usual encoder-bottleneck-decoder structure:

1. Input blocks
2. Middle block
3. Output blocks
4. Final output projection

With skip connections:

- each encoder stage output is saved
- decoder stages concatenate the saved encoder features back in

This is the core U-shape.

## Architecture Walkthrough

### 1. Timestep Embedding

Before any image features are processed, the timestep is embedded.

Source:

- `ldm/modules/diffusionmodules/openaimodel.py:509-514`
- `ldm/modules/diffusionmodules/openaimodel.py:726-727`

Flow:

1. `timestep_embedding(...)` creates a sinusoidal-style timestep representation.
2. `self.time_embed` maps it through two linear layers with `SiLU`.
3. The result becomes the global conditioning vector `emb`.

If the UNet is class-conditional:

- `label_emb(y)` is added to `emb`

Source:

- `ldm/modules/diffusionmodules/openaimodel.py:516-517`
- `ldm/modules/diffusionmodules/openaimodel.py:729-732`

Meaning:

- every residual block can adapt its computation based on the current diffusion timestep
- optionally it can also adapt based on class label

### 2. Input Blocks

Source:

- `ldm/modules/diffusionmodules/openaimodel.py:519-590`

The input side starts with:

- one initial `3x3` convolution from `in_channels -> model_channels`

Then for each resolution level:

- `num_res_blocks` residual blocks
- optional attention or spatial transformer blocks
- optional downsampling between levels

The current number of channels grows according to `channel_mult`.

Example with:

- `model_channels = 192`
- `channel_mult = [1, 2, 2, 4, 4]`

The channel widths across levels become:

- `192`
- `384`
- `384`
- `768`
- `768`

At each stage:

- the resulting tensor is pushed into `hs`
- later the decoder will reuse it as a skip connection

### 3. Middle Block

Source:

- `ldm/modules/diffusionmodules/openaimodel.py:600-626`

The bottleneck is:

1. `ResBlock`
2. `AttentionBlock` or `SpatialTransformer`
3. `ResBlock`

This is the most compressed representation in the U-shape.

It allows:

- local feature processing through residual blocks
- global interaction through attention

### 4. Output Blocks

Source:

- `ldm/modules/diffusionmodules/openaimodel.py:629-683`

The decoder mirrors the encoder.

For each output stage:

1. Pop one skip feature from `hs`
2. Concatenate it with the current decoder state
3. Apply a `ResBlock`
4. Optionally apply attention
5. Optionally upsample when moving to a higher spatial resolution

Important detail:

- concatenation happens before the block:
  `h = th.cat([h, hs.pop()], dim=1)`

So the decoder does not just receive its own current features. It also receives the corresponding encoder features directly.

### 5. Final Projection

Source:

- `ldm/modules/diffusionmodules/openaimodel.py:685-689`

The output head is:

1. normalization
2. `SiLU`
3. zero-initialized `3x3` convolution

The zero initialization is intentional:

- it makes the final residual-style output start near zero
- this tends to stabilize training

If `predict_codebook_ids` is enabled, the model can instead output codebook logits:

- `ldm/modules/diffusionmodules/openaimodel.py:690-695`

That path is not the standard SDSeg use case.

## The Main Building Blocks

## `TimestepEmbedSequential`

Source:

- `ldm/modules/diffusionmodules/openaimodel.py:75`

What it does:

- behaves like `nn.Sequential`
- but passes timestep embeddings into layers that know how to use them
- also passes `context` into `SpatialTransformer`

Why it matters:

- this is how one container can mix:
  - normal conv layers
  - timestep-aware residual blocks
  - transformer blocks with context

In practice:

- if a child is a `TimestepBlock`, it gets `(x, emb)`
- if a child is a `SpatialTransformer`, it gets `(x, context)`
- otherwise it gets just `x`

## `ResBlock`

Source:

- `ldm/modules/diffusionmodules/openaimodel.py:166`

This is the most important local-processing block in the UNet.

### Structure

A `ResBlock` contains:

- input normalization
- `SiLU`
- `3x3` convolution
- timestep projection
- output normalization
- `SiLU`
- dropout
- zero-initialized `3x3` convolution
- skip connection

### What the timestep does

The timestep embedding is projected by:

- `self.emb_layers`

and then injected into feature processing.

There are two modes:

- additive mode
- scale-shift normalization mode

If `use_scale_shift_norm=False`:

- the timestep embedding is added to the feature tensor

If `use_scale_shift_norm=True`:

- the timestep embedding is split into scale and shift
- it modulates the normalized activations in a FiLM-like way

Source:

- `ldm/modules/diffusionmodules/openaimodel.py:266-276`

This repo’s SDSeg configs use:

- `use_scale_shift_norm: True`

So the timestep conditioning is not just simple addition. It actively modulates normalized features.

### Upsampling and downsampling inside `ResBlock`

`ResBlock` can optionally perform:

- upsampling
- downsampling

Source:

- `ldm/modules/diffusionmodules/openaimodel.py:210-219`

This is used when:

- `resblock_updown=True`

That is the SDSeg default in the configs shown above.

Meaning:

- resolution changes are handled inside residual blocks instead of by plain standalone up/downsample modules

## `AttentionBlock`

Source:

- `ldm/modules/diffusionmodules/openaimodel.py:280`

This block performs spatial self-attention.

Flow:

1. normalize features
2. produce Q, K, V by `1x1` convolution
3. flatten spatial positions into a sequence
4. compute attention between positions
5. project back
6. add a residual connection

Why it matters:

- convolutions are local
- attention lets each spatial position interact with every other spatial position

This helps the UNet reason about global structure.

## `QKVAttentionLegacy` and `QKVAttention`

Source:

- `ldm/modules/diffusionmodules/openaimodel.py:350`
- `ldm/modules/diffusionmodules/openaimodel.py:382`

These are two implementations of multi-head attention over flattened spatial tokens.

Difference:

- they split and arrange Q/K/V in slightly different orders

From the user perspective:

- both serve the same role
- they are low-level internals for the attention block

## `SpatialTransformer`

Source:

- `ldm/modules/attention.py:218`

This is the more modern transformer-style attention block optionally used instead of `AttentionBlock`.

Structure:

1. normalize image features
2. project channels to transformer dimension
3. flatten spatial map into a token sequence
4. run one or more transformer blocks
5. reshape back to image format
6. project back and add residual connection

Each transformer block contains:

- self-attention
- cross-attention
- feed-forward network

Source:

- `ldm/modules/attention.py:196-215`

Important detail:

- if no external `context` is given, cross-attention falls back to self-attention
- if `context` is given, the block becomes genuinely conditional

## `Upsample` and `Downsample`

Source:

- `ldm/modules/diffusionmodules/openaimodel.py:92`
- `ldm/modules/diffusionmodules/openaimodel.py:137`

These change spatial resolution between stages.

`Upsample`:

- nearest-neighbor resize
- optional convolution afterward

`Downsample`:

- either strided convolution
- or average pooling

In SDSeg configs:

- `resblock_updown=True`

So resolution change often happens inside `ResBlock` rather than through these plain modules alone.

## Forward Pass Walkthrough

Source:

- `ldm/modules/diffusionmodules/openaimodel.py:713-746`

The forward pass is:

1. Build the timestep embedding.
2. If class-conditional, add label embedding.
3. Cast input to the model dtype.
4. Pass through all input blocks, saving each output in `hs`.
5. Pass through the middle block.
6. For each output block:
   - pop one skip feature
   - concatenate it with current features
   - process through the block
7. Cast back to the original input dtype.
8. Return either:
   - `id_predictor(h)`, or
   - `self.out(h)`

The key line for skip connections is:

```python
h = th.cat([h, hs.pop()], dim=1)
```

That is the central U-Net idea in this implementation.

## Attention Resolution Logic

The UNet does not apply attention at every location by default. It checks:

- whether the current downsampling factor `ds` is in `attention_resolutions`

Source:

- `ldm/modules/diffusionmodules/openaimodel.py:544`
- `ldm/modules/diffusionmodules/openaimodel.py:645`

In the SDSeg configs:

- `attention_resolutions: [1, 2, 4, 8]`

Given a latent input size of `32 x 32`, that means attention is used at:

- `32 x 32`
- `16 x 16`
- `8 x 8`
- `4 x 4`

This gives the model a lot of global interaction across the latent feature hierarchy.

## Conditioning Modes And What They Mean For The UNet

The UNet supports several conditioning styles through `DiffusionWrapper`, but the most relevant are:

### 1. Concat conditioning

Meaning:

- condition features are concatenated into input channels before the UNet processes them

In SDSeg this is the typical path:

- noisy segmentation latent: 4 channels
- image conditioning latent: 4 channels
- total input to UNet: 8 channels

### 2. Cross-attention conditioning

Meaning:

- external context tokens are passed through transformer attention

This matters when:

- `use_spatial_transformer=True`
- `context_dim` is set

That path is supported by the generic UNet implementation, even if the common SDSeg configs shown here mainly use concat conditioning.

### 3. Class conditioning

Meaning:

- labels are embedded by `label_emb`
- the label embedding is added to the timestep embedding

This is simpler than cross-attention:

- it changes the global conditioning vector, not the spatial feature map directly

## How To Read The UNet In The Repo

If you want to understand it efficiently, read in this order:

1. `UNetModel.__init__`
2. `UNetModel.forward`
3. `TimestepEmbedSequential`
4. `ResBlock`
5. `AttentionBlock`
6. `SpatialTransformer`
7. `Upsample` and `Downsample`

That order goes from overall structure to the internal blocks.

## Mental Model For SDSeg

For SDSeg specifically, the simplest mental model is:

1. Start with a noisy segmentation latent.
2. Concatenate the image latent next to it along channels.
3. Encode the diffusion timestep.
4. Run through a U-Net with residual blocks, attention, and skip connections.
5. Produce a 4-channel latent prediction.
6. Let the diffusion code interpret that output as predicted noise or predicted clean latent.

So the UNet is not directly outputting a final binary mask.

It is outputting a latent-space denoising prediction that the surrounding diffusion code later converts into a mask.

## Most Important Takeaways

- `UNetModel` is the denoiser at the center of SDSeg.
- The common SDSeg setup here uses concat conditioning, so `in_channels` is doubled from `4` to `8`.
- Timestep conditioning is injected into every residual block.
- Skip connections are implemented by concatenating saved encoder features into decoder features.
- Attention is applied at multiple latent resolutions.
- The UNet predicts latent-space outputs, not final segmentation masks directly.

## Related Docs

- `docs/DIFFUSION_CLASS_FLOW.md`
- `docs/SDSeg_FUNCTIONS.md`

Use this file for the denoiser architecture.
Use the diffusion class flow doc for how the UNet fits into `DDPM`, `LatentDiffusion`, and `SDSeg`.
