# Sparsh × I-JEPA: route from tactile image to loss

Follow steps 0 → 8 in order. Each step gives:

- **Go to**: links in reading order (click to open the file at that line),
- **Essential code**: a trimmed excerpt of the original (`...` = skipped lines),
- **Shapes** and a short **note** only where needed.

The same pipeline as one runnable file, with sections numbered the same way: [sparsh_ijepa_lean.py](sparsh_ijepa_lean.py).
It was checked against the original code. Given the same input, weights and seeds, it gives identical preprocessed tensors, identical masks, encoder and predictor outputs equal up to float rounding (≤ 3e-6), the same loss and the same EMA update.

Shapes are for the pretraining config: ViT-Base, input 320×240, patch 16 → grid 20×15 = **300 patches**, batch B = 64.

## Overview

```
 raw tactile frames (.pkl, one file per sequence)
      │ 1. Dataset: background subtraction, resize, augment, stack frames t and t-5
      ▼
 x (B, 6, 320, 240) ──────────────────────────────┬──────────────────────────────┐
      │ 3. sample_masks                           │                              │
      │    encoder_masks   1 × (B, N_ctx)         │                              │
      │    predictor_masks 4 × (B, N_tgt)         │                              │
      ▼                                           │                              ▼
 4. CONTEXT ENCODER (ViT-B, trained)              │      6. TARGET ENCODER (EMA copy, no grad)
    patch embed + pos → keep context → 12 blocks  │         patch embed + pos → 12 blocks (all 300)
      │ (B, N_ctx, 768)                           │           │ (B, 300, 768) → layer_norm
      ▼                                           │           │ keep target patches
 5. PREDICTOR (ViT 192-d, 6 blocks)  ◄────────────┘           │ (4B, N_tgt, 768)
    context tokens + mask tokens at target positions          │
      │ (4B, N_tgt, 768)                                      │
      └───────────────────► 7. smooth L1 loss ◄───────────────┘
                                    │ backward + optimizer step (context encoder, predictor)
                                    ▼
                       8. EMA: target ← m·target + (1−m)·context
```

---

## Step 0: Config → objects

**Go to**

1. [train.py:211 `main`](../sparsh/train.py#L211): Hydra entry point, launched as `python train.py +experiment=ijepa_vit paths=... wandb=...`
2. [config/experiment/ijepa_vit.yaml](../sparsh/config/experiment/ijepa_vit.yaml): selects `data: vision_based`, `model: ijepa_vit`, and overrides sizes
3. [config/model/ijepa_vit.yaml](../sparsh/config/model/ijepa_vit.yaml): `_target_: tactile_ssl.algorithm.IJEPAModule` and the masking parameters
4. [config/data/vision_based.yaml](../sparsh/config/data/vision_based.yaml): one dataset block per sensor (DIGIT, GelSight'17, GelSight Mini)
5. [train.py:199-206](../sparsh/train.py#L199-L206): build the dataloaders, instantiate the model, `trainer.fit`

**Essential code**

```python
# train.py — train()
train_dataloader, val_dataloader = get_dataloaders(cfg)
trainer = Trainer(wandb_logger=wandb, **cfg.trainer)
model = hydra.utils.instantiate(cfg.model)      # builds encoder + predictor first, then IJEPAModule(...)
trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=cfg.ckpt_path)
```

**Effective values** (model yaml, then overridden by the experiment yaml)

| Config key                                      | Value                                                                                                                                                                                            | Builds                                                                                               |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------- |
| `model._target_`                                | `tactile_ssl.algorithm.IJEPAModule`                                                                                                                                                              | [ijepa.py:24](../sparsh/tactile_ssl/algorithm/ijepa.py#L24)                                          |
| `model.encoder`                                 | `vit_base`, `img_size [320,240]`, `in_chans 6`, `patch_size 16`, `pos_embed_fn sinusoidal`                                                                                                       | context encoder, [vision_transformer.py:557](../sparsh/tactile_ssl/model/vision_transformer.py#L557) |
| `model.predictor`                               | `vit_predictor`, `input_dim 768`, `embed_dim 192` (depth 6, 12 heads by default)                                                                                                                 | [vision_transformer.py:505](../sparsh/tactile_ssl/model/vision_transformer.py#L505)                  |
| masking                                         | `encoder_mask_scale [0.85,1.0]`, `predictor_mask_scale [0.15,0.2]`, `aspect_ratio [0.75,1.5]`, `min_keep_num_patches 10`, `allow_mask_overlap false`, 1 context mask / 4 target masks (defaults) | `sample_masks` (Step 3)                                                                              |
| `model.moving_average_decay`                    | `[0.996, 1.0]`                                                                                                                                                                                   | EMA momentum ramp (Step 8)                                                                           |
| `num_frames`, `frame_stride`, `data_out_format` | `2`, `5`, `concat_ch_img`                                                                                                                                                                        | the 6-channel input (Step 1)                                                                         |
| `data.sensor[*].dataset._target_`               | `VisionTactileBackboneDataset`                                                                                                                                                                   | one dataset per sequence, joined with `ConcatDataset`                                                |

---

## Step 1: Raw image → input tensor

**Go to**

1. [train.py:70 `get_dataloaders_vision_based`](../sparsh/train.py#L70): one dataset per `object/dataset_i.pkl`, all concatenated
2. [vision_tactile.py:93 `__getitem__`](../sparsh/tactile_ssl/data/vision_tactile.py#L93)
3. [vision_tactile.py:112 `_get_tactile_images`](../sparsh/tactile_ssl/data/vision_tactile.py#L112)
4. [digit/utils.py:125 `load_sample_from_buf`](../sparsh/tactile_ssl/data/digit/utils.py#L125) → [digit/utils.py:51 `compute_diff`](../sparsh/tactile_ssl/data/digit/utils.py#L51)
5. [digit/utils.py:79 `get_resize_transform`](../sparsh/tactile_ssl/data/digit/utils.py#L79)

**Essential code**

```python
# vision_tactile.py — __getitem__ + _get_tactile_images
idx += self.frames_concat_idx[-1]                # frames_concat_idx = [0, 5]
do_flip = random.random() < self.p_flip          # drawn once per sample, applied to both frames
do_crop = random.random() < self.p_crop
do_rot  = random.random() < self.p_rot
...
for i in self.frames_concat_idx:                 # frame t, then frame t-5
    frame_index = np.clip(ref_frame - i, 0, self.total_frames - 1)
    image = self.loader(self.frames[frame_index], self.bg)   # load_sample_from_buf
    image = self.transform_resize(image)                     # Resize(320, 240) + ToTensor → [0, 1]
    if do_flip: image = TF.hflip(image)
    if do_crop: ...                              # center crop 60–90 %, resize back
    if do_rot:  ...                              # rotate ±15°, center crop the black corners away, resize back
    sample_images.append(image)
output = torch.cat(sample_images, dim=0)         # out_format "concat_ch_img"
inputs["image"] = output
```

```python
# digit/utils.py — load_sample_from_buf + compute_diff
img = load_bin_image(io_buf)                     # bytes → H×W×3 uint8
img = compute_diff(img, img_bg, offset=0.5)      # (img − bg)/255 + 0.5, clipped to [0, 1]
if h < w:
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)   # landscape → portrait
...                                              # center crop to h/w = 4/3
return Image.fromarray(img).convert("RGB")
```

**Shapes**: frame H×W×3 uint8 → `(3, 320, 240)` in [0, 1] → `(6, 320, 240)` → DataLoader → `batch["image"]`: `(B, 6, 320, 240)`

**Note**

- Channels 0:3 hold frame t and channels 3:6 hold frame t−5.
- The background is a no-contact image from the same sensor ([`get_bg_img`](../sparsh/tactile_ssl/data/digit/utils.py#L89)). GelSight'17 data has no background removal.

---

## Step 2: Training loop → `IJEPAModule.training_step`

**Go to**

1. [trainer.py:232](../sparsh/tactile_ssl/trainer/trainer.py#L232): epoch loop in `fit`
2. [trainer.py:266 `train_loop`](../sparsh/tactile_ssl/trainer/trainer.py#L266): for each batch, run the forward + backward, step the optimizer, then call `on_train_batch_end`
3. [trainer.py:418 `training_step`](../sparsh/tactile_ssl/trainer/trainer.py#L418): calls `module.training_step`, then backpropagates `outputs["loss"]`
4. [ijepa.py:331 `IJEPAModule.training_step`](../sparsh/tactile_ssl/algorithm/ijepa.py#L331)

**Essential code**

```python
# ijepa.py — training_step
self.step = self.step + 1
self.generator.manual_seed(self.step)            # the masks are reproducible per step
x = batch["image"]                               # (B, 6, 320, 240)
encoder_masks, predictor_masks = self.sample_masks(x)        # Step 3
loss = self.forward(x, encoder_masks, predictor_masks)       # Steps 4–7
...                                              # online probe, skip
output["loss"] = loss
```

---

## Step 3: Masks

**Go to**

1. [ijepa.py:239 `sample_masks`](../sparsh/tactile_ssl/algorithm/ijepa.py#L239)
2. [ijepa.py:137 `_sample_block_size`](../sparsh/tactile_ssl/algorithm/ijepa.py#L137)
3. [ijepa.py:156 `_sample_block_mask`](../sparsh/tactile_ssl/algorithm/ijepa.py#L156)

**Essential code**

```python
# sample_masks
height, width = image_height // patch_size, image_width // patch_size    # 20, 15
predictor_size = self._sample_block_size(height, width, self.predictor_mask_scale, self.aspect_ratio)
encoder_size   = self._sample_block_size(height, width, self.encoder_mask_scale, (1.0, 1.0))
for _ in range(batch_size):                      # block sizes are shared by the whole batch, positions are per sample
    for _ in range(self.num_predictor_masks):    # 4 target blocks
        mask, mask_complement = self._sample_block_mask(height, width, predictor_size)
        masks_predictor.append(mask); masks_complement.append(mask_complement)
    acceptable_regions = masks_complement        # allow_mask_overlap = False
    for _ in range(self.num_encoder_masks):      # 1 context block
        mask, _ = self._sample_block_mask(height, width, encoder_size, acceptable_regions)
...                                              # truncate all masks to the shortest one in the batch, then collate
return encoder_masks, predictor_masks
```

```python
# _sample_block_size — a single random draw r sets both the scale and the aspect ratio
mask_scale   = min_s + r * (max_s - min_s);    max_keep = int(height * width * mask_scale)
aspect_ratio = min_ar + r * (max_ar - min_ar)
h = int(round(math.sqrt(max_keep * aspect_ratio)))   # then clipped to the grid
w = int(round(math.sqrt(max_keep / aspect_ratio)))

# _sample_block_mask
top  = torch.randint(0, height - h + 1, (1,), generator=self.generator)
left = torch.randint(0, width - w + 1, (1,), generator=self.generator)
mask = torch.zeros((height, width)); mask[top:top+h, left:left+w] = 1
constrain_mask(mask)                             # mask *= complement of every target block (context only)
mask = torch.nonzero(mask.flatten())             # → flat patch indices in [0, 300)
```

**Shapes**

- `encoder_masks`: list of 1 tensor `(B, N_ctx)`
- `predictor_masks`: list of 4 tensors `(B, N_tgt)`
- Each target block covers about 15–20 % of the grid. The context is one large block (85–100 % of the grid before clipping) minus all 4 target blocks.
- Sampled example (batch of 3): N_ctx = 55–116 and N_tgt = 48–63. With B = 64 the numbers are a bit smaller, because every mask is cut to the shortest one in the batch.

### Helper used in Steps 4–6: `apply_masks`

[utils/\_\_init\_\_.py:25](../sparsh/tactile_ssl/utils/__init__.py#L25)

```python
for mask in masks:                               # list of (B, K) index tensors
    mask_keep = einops.repeat(mask, "b n -> b n d", d=x.size(-1))
    all_x.append(torch.gather(x, dim=-2, index=mask_keep))
return torch.cat(all_x, dim=0)                   # (len(masks)·B, K, D); row k·B + b = mask k of sample b
```

---

## Step 4: Context encoder

**Go to**

1. [ijepa.py:314](../sparsh/tactile_ssl/algorithm/ijepa.py#L314): `context_features = self.context_encoder(x, encoder_masks)`
2. [vision_transformer.py:420 `forward`](../sparsh/tactile_ssl/model/vision_transformer.py#L420) → [:341 `forward_features`](../sparsh/tactile_ssl/model/vision_transformer.py#L341)
3. [vision_transformer.py:300 `prepare_tokens_with_masks`](../sparsh/tactile_ssl/model/vision_transformer.py#L300)
4. [patch_embed.py:84 `PatchEmbed.forward`](../sparsh/tactile_ssl/model/layers/patch_embed.py#L84)
5. [patch_embed.py:183 `SinusoidalEmbed.forward`](../sparsh/tactile_ssl/model/layers/patch_embed.py#L183)
6. [utils/\_\_init\_\_.py:25 `apply_masks`](../sparsh/tactile_ssl/utils/__init__.py#L25)
7. [block.py:89 `Block.forward`](../sparsh/tactile_ssl/model/layers/block.py#L89): treat it as a black box (pre-norm attention + MLP)

**Essential code**

```python
# vision_transformer.py — prepare_tokens_with_masks → forward_features → forward
pos_encoding = self.interpolate_pos_encoding(x.shape, ...)   # = SinusoidalEmbed output, (1, 300, 768)
x = self.patch_embed(x)                          # Conv2d(6, 768, kernel=16, stride=16) → (B, 300, 768)
x = x + pos_encoding
if masks is not None:
    x = apply_masks(x, masks)                    # (B, N_ctx, 768)
for blk in self.blocks:                          # 12 × Block
    x = blk(x)
x_norm = self.norm(x)
return x_norm                                    # "x_norm_patchtokens"
```

**Shapes**: `(B, 6, 320, 240)` → `(B, 300, 768)` → `(B, N_ctx, 768)`

**Note**

- Masking happens after the patch and position embeddings: patches outside the context are dropped, not replaced by mask tokens.
- There is no CLS token.
- The position embedding is fixed, not learned: `[sin(y·f), cos(y·f), sin(x·f), cos(x·f)]` for each patch at grid position (y, x).

---

## Step 5: Predictor

**Go to**

1. [ijepa.py:315](../sparsh/tactile_ssl/algorithm/ijepa.py#L315): `self.predictor(context_features, encoder_masks, predictor_masks)`
2. [vision_transformer.py:425 `VisionTransformerPredictor.__init__`](../sparsh/tactile_ssl/model/vision_transformer.py#L425): adds `input_projection`, `output_projection` and `mask_token`
3. [vision_transformer.py:448 `VisionTransformerPredictor.forward`](../sparsh/tactile_ssl/model/vision_transformer.py#L448)

**Essential code**

```python
x = self.input_projection(x)                     # 768 → 192: (B, N_ctx, 192)
pos_emb_x = self.pos_embed(x.device)             # sinusoidal, (300, 192)
pos_emb_x = einops.repeat(pos_emb_x, "n c -> b n c", b=batch_size)
x = x + apply_masks(pos_emb_x, context_masks)    # add the positions back to the context tokens
pos_emb_masked = apply_masks(pos_emb_x, masks)   # target positions: (4B, N_tgt, 192)
...
prediction_tokens = mask_token + pos_emb_masked  # one shared learnable token plus each target position
x = einops.repeat(x, "(p b) n c -> (p k b) n c", p=len(context_masks), k=len(masks))  # copy the context once per target
x = torch.cat([x, prediction_tokens], dim=1)     # (4B, N_ctx + N_tgt, 192)
for blk in self.blocks:                          # 6 × Block
    x = blk(x)
x = self.norm(x)
x = x[:, num_context_tokens:]                    # keep only the target tokens
x = self.output_projection(x)                    # 192 → 768: (4B, N_tgt, 768)
```

**Note**

- Each of the 4 target blocks is predicted in its own sequence (the context plus that block's mask tokens). The 4 sequences are stacked along the batch dimension.

---

## Step 6: Target encoder

**Go to**

1. [ijepa.py:65](../sparsh/tactile_ssl/algorithm/ijepa.py#L65): `self.target_encoder = copy.deepcopy(self.context_encoder)`, with `requires_grad_(False)`
2. [ijepa.py:318-327](../sparsh/tactile_ssl/algorithm/ijepa.py#L318-L327)

**Essential code**

```python
with torch.no_grad():
    target_features = self.target_encoder(x)     # no mask: (B, 300, 768)
    target_features = F.layer_norm(target_features, [target_features.shape[-1]])   # extra non-affine LayerNorm
    masked_target_features_x = apply_masks(target_features, predictor_masks)      # (4B, N_tgt, 768)
    masked_target_features = einops.repeat(masked_target_features_x, ..., k=len(encoder_masks), ...)  # no-op with 1 context mask
```

**Note**

- This is the same `VisionTransformer` code path as Step 4, called with `masks=None`.
- The image is encoded first, and only then are the target patches selected. So each target feature has seen the whole image.

---

## Step 7: Loss

**Go to**: [ijepa.py:328-329](../sparsh/tactile_ssl/algorithm/ijepa.py#L328-L329)

```python
loss = F.smooth_l1_loss(context_prediction, masked_target_features.detach())
return loss.mean()
```

**Shapes**: `(4B, N_tgt, 768)` vs `(4B, N_tgt, 768)` → scalar. The rows line up because both sides come from `apply_masks(·, predictor_masks)`.

**After the loss**

1. Back in `training_step`, the online-probe loss is added. It has no effect on the encoders.
2. [trainer.py:435](../sparsh/tactile_ssl/trainer/trainer.py#L435): `fabric.backward(loss)`
3. [trainer.py:316-320](../sparsh/tactile_ssl/trainer/trainer.py#L316-L320): gradient clipping, then `optimizer.step()`. Only the context encoder and the predictor are updated.

---

## Step 8: EMA update of the target encoder

**Go to**

1. [trainer.py:328](../sparsh/tactile_ssl/trainer/trainer.py#L328): `module.on_train_batch_end(...)`, called after the optimizer step
2. [ijepa.py:109 `on_train_batch_end`](../sparsh/tactile_ssl/algorithm/ijepa.py#L109)
3. [utils/ema.py:12 `update_moving_average`](../sparsh/tactile_ssl/utils/ema.py#L12)
4. [ijepa.py:422-429](../sparsh/tactile_ssl/algorithm/ijepa.py#L422-L429): the momentum schedule

```python
# ijepa.py — on_train_batch_end
moving_average_decay = next(self.momentum_scheduler)    # linear ramp 0.996 → 1.0 over all training steps
update_moving_average(self.target_encoder, self.context_encoder, moving_average_decay)

# ema.py
ma_params.data = old * beta + (1.0 - beta) * new        # target ← m·target + (1−m)·context
```

---

## Skip these: they look related but are not on the I-JEPA path

| Where                                                                                                                                                        | What it is                                                                                             |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------ |
| [utils/masking.py](../sparsh/tactile_ssl/utils/masking.py)                                                                                                   | V-JEPA mask collator. I-JEPA builds its masks inside `ijepa.py`.                                       |
| [ijepa.py:344-371](../sparsh/tactile_ssl/algorithm/ijepa.py#L344-L371)                                                                                       | Online reconstruction probe. It only monitors training and decodes target features with gradients off. |
| [ijepa.py:196-237](../sparsh/tactile_ssl/algorithm/ijepa.py#L196-L237)                                                                                       | `get_masks_img` / `plot_masks`, used only for debug plots.                                             |
| [vision_transformer.py:252](../sparsh/tactile_ssl/model/vision_transformer.py#L252)                                                                          | `interpolate_pos_encoding`. Changes nothing at the native 320×240 size.                                |
| `vision_transformer.py`: `PatchEmbed3D`/video branches, register tokens, `block_chunks`, `forward_features_list`, `get_intermediate_layers`                  | Not used by I-JEPA pretraining.                                                                        |
| [block.py:211 `NestedTensorBlock`](../sparsh/tactile_ssl/model/layers/block.py#L211), `MemEffAttention`, drop-path helpers                                   | xFormers versions of the same Block and Attention. `drop_path_rate` is 0.                              |
| [ijepa.py:381 `configure_optimizers`](../sparsh/tactile_ssl/algorithm/ijepa.py#L381), [custom_scheduler.py](../sparsh/tactile_ssl/model/custom_scheduler.py) | Optimizer, LR and WD schedules. Only the momentum ramp at 422–429 matters here (Step 8).               |

## Lean file map

| Step                         | [sparsh_ijepa_lean.py](sparsh_ijepa_lean.py)                                                 |
| ---------------------------- | -------------------------------------------------------------------------------------------- |
| 0 Config                     | section 0, `CONFIG` constants                                                                |
| 1 Data                       | section 1, `compute_diff`, `load_sample`, `TactileDataset`                                   |
| 3 Masks                      | section 2, `sample_block_size`, `sample_block_mask`, `sample_masks`, `apply_masks`           |
| 4 Context / 6 Target encoder | sections 3–4, `PatchEmbed`, `sinusoidal_pos_embed`, `Block`, `VisionTransformer`             |
| 5 Predictor                  | section 5, `Predictor`                                                                       |
| 2, 6, 7, 8                   | section 6–8, `IJEPA.training_step`, `IJEPA.forward`, `IJEPA.ema_update`, `momentum_schedule` |
