"""
Sparsh I-JEPA, lean version: raw tactile frames -> batch -> masks -> context encoder
-> predictor -> target encoder -> loss -> EMA update.

Only the essential code paths of the original are kept: no Hydra, no video/register-token
branches, no xFormers, no online probes, no LR/WD schedulers. Every block is annotated with
`src:` pointing to the original code (paths relative to the repo root).

Values come from the composition of
    sparsh/config/default.yaml
    sparsh/config/experiment/ijepa_vit.yaml     (overrides the two below)
    sparsh/config/model/ijepa_vit.yaml
    sparsh/config/data/vision_based.yaml
i.e. the DIGIT / ViT-Base pretraining run.

Run:  python docs/sparsh_ijepa_lean.py   (needs torch, torchvision, einops, numpy, pillow)
"""

import copy
import math
import random

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# =====================================================================================
# 0. CONFIG  (the values that matter, resolved from the yaml files above)
# =====================================================================================
IMG_SIZE = (
    320,
    240,
)  # (H, W)                  experiment/ijepa_vit.yaml: model.encoder.img_size
PATCH_SIZE = (
    16  #                               model/ijepa_vit.yaml: encoder.patch_size
)
GRID = (IMG_SIZE[0] // PATCH_SIZE, IMG_SIZE[1] // PATCH_SIZE)  # (20, 15) -> 300 patches
NUM_FRAMES, FRAME_STRIDE = 2, 5  #               default.yaml: num_frames, frame_stride
IN_CHANS = (
    3 * NUM_FRAMES
)  # 6                   experiment/ijepa_vit.yaml: model.encoder.in_chans

ENC_DIM, ENC_DEPTH, ENC_HEADS = 768, 12, 12  #   tactile_ssl.model.vit_base
PRED_DIM, PRED_DEPTH, PRED_HEADS = (
    192,
    6,
    12,
)  # experiment: predictor.embed_dim / vit_predictor defaults

NUM_ENC_MASKS, NUM_PRED_MASKS = 1, 4  #          IJEPAModule defaults
ENC_MASK_SCALE = (
    0.85,
    1.0,
)  #                  model/ijepa_vit.yaml: encoder_mask_scale
PRED_MASK_SCALE = (
    0.15,
    0.2,
)  #                 model/ijepa_vit.yaml: predictor_mask_scale
ASPECT_RATIO = (0.75, 1.5)  #                    model/ijepa_vit.yaml: aspect_ratio
MIN_KEEP = (
    10  #                                 model/ijepa_vit.yaml: min_keep_num_patches
)
EMA_DECAY = (
    0.996,
    1.0,
)  #                      experiment/ijepa_vit.yaml: moving_average_decay

P_FLIP, P_CROP, P_ROT = 0.5, 0.2, 0.2  #         data/vision_based.yaml: transforms


# =====================================================================================
# 1. DATA: raw frame -> (6, H, W) tensor
# =====================================================================================
def compute_diff(img, bg, offset=0.5):
    # src: sparsh/tactile_ssl/data/digit/utils.py:51  (background subtraction, re-centred at 0.5)
    diff = (img.astype(np.int32) - bg.astype(np.int32)) / 255.0 + offset
    return np.uint8(np.clip(diff, 0.0, 1.0) * 255.0)


def load_sample(img, bg=None):
    # src: sparsh/tactile_ssl/data/digit/utils.py:125  load_sample_from_buf
    if bg is not None:
        img = compute_diff(img, bg)
    h, w, _ = img.shape
    if h < w:  # landscape -> portrait (cv2.ROTATE_90_CLOCKWISE)
        img = np.ascontiguousarray(np.rot90(img, k=-1))
        h, w, _ = img.shape
    r = 4 / 3  # enforce the 4:3 aspect ratio (crop is a no-op for DIGIT frames)
    if h / w != r:
        h2 = int(h / r)
        img = img[int((h - h2) / 2) : int((h + h2) / 2)]
    return Image.fromarray(img).convert("RGB")


class TactileDataset(Dataset):
    """src: sparsh/tactile_ssl/data/vision_tactile.py:29  VisionTactileBackboneDataset
    One instance per recorded sequence; train.py concatenates them with ConcatDataset."""

    def __init__(self, frames, bg=None):
        self.frames = frames  # list of HxWx3 uint8 images (decoded from the .pkl)
        self.bg = bg  # no-contact image of the same sensor (None for GelSight'17)
        self.frames_concat_idx = np.arange(
            0, NUM_FRAMES * FRAME_STRIDE, FRAME_STRIDE
        )  # [0, 5]
        self.resize = (
            transforms.Compose(  # src: digit/utils.py:79  get_resize_transform
                [
                    transforms.Resize(IMG_SIZE, antialias=True),
                    transforms.ToTensor(),
                ]  # -> [0, 1]
            )
        )

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        # src: vision_tactile.py:93  __getitem__
        ref = idx + self.frames_concat_idx[-1]
        do_flip, do_crop, do_rot = (
            random.random() < p for p in (P_FLIP, P_CROP, P_ROT)
        )
        return {"image": self._get_tactile_images(ref, do_flip, do_crop, do_rot)}

    def _get_tactile_images(self, ref, do_flip, do_crop, do_rot):
        # src: vision_tactile.py:112  same augmentation applied to both frames
        if do_crop:
            s = random.uniform(0.6, 0.9)
            crop_size = (int(IMG_SIZE[0] * s), int(IMG_SIZE[1] * s))
        if do_rot:
            angle = random.uniform(-15, 15)
            m = TF.rotate(
                torch.ones(1, *IMG_SIZE),
                15,
                interpolation=transforms.InterpolationMode.BILINEAR,
            )
            corner = min(
                torch.argmax(m[:, 0, :]).item(), torch.argmax(m[:, :, 0]).item()
            )
            rot_size = (
                IMG_SIZE[0] - corner,
                IMG_SIZE[1] - corner,
            )  # crop that removes black corners

        images = []
        for i in self.frames_concat_idx:  # frame t, then frame t-5
            frame = self.frames[int(np.clip(ref - i, 0, len(self.frames) - 1))]
            image = self.resize(load_sample(frame, self.bg))  # (3, 320, 240)
            if do_flip:
                image = TF.hflip(image)
            if do_crop:
                image = TF.resize(
                    TF.center_crop(image, crop_size), IMG_SIZE, antialias=True
                )
            if do_rot:
                image = TF.rotate(
                    image, angle, interpolation=transforms.InterpolationMode.BILINEAR
                )
                image = TF.resize(
                    TF.center_crop(image, rot_size), IMG_SIZE, antialias=True
                )
            images.append(image)
        return torch.cat(images, dim=0)  # out_format "concat_ch_img": (6, 320, 240)


# =====================================================================================
# 2. MASKS: which patch indices the context encoder sees / the predictor must predict
# =====================================================================================
def sample_block_size(g, scale, aspect_ratio):
    # src: sparsh/tactile_ssl/algorithm/ijepa.py:137  _sample_block_size
    H, W = GRID
    r = torch.rand(1, generator=g).item()  # one draw shared by scale and aspect ratio
    num = int(H * W * (scale[0] + r * (scale[1] - scale[0])))
    ar = aspect_ratio[0] + r * (aspect_ratio[1] - aspect_ratio[0])
    h = min(int(round(math.sqrt(num * ar))), H)
    w = min(int(round(math.sqrt(num / ar))), W)
    return h, w


def sample_block_mask(g, block_size, allowed=None):
    # src: ijepa.py:156  _sample_block_mask
    # (the original also relaxes `allowed` after 20 failed tries; omitted here)
    H, W = GRID
    h, w = block_size
    while True:
        top = torch.randint(0, H - h + 1, (1,), generator=g)
        left = torch.randint(0, W - w + 1, (1,), generator=g)
        block = torch.zeros(H, W, dtype=torch.int32)
        block[top : top + h, left : left + w] = 1
        keep = (
            block * allowed if allowed is not None else block
        )  # drop target patches from context
        idx = torch.nonzero(keep.flatten()).squeeze()  # flat patch indices
        if len(idx) > MIN_KEEP:
            return idx, block


def sample_masks(g, batch_size):
    # src: ijepa.py:239  sample_masks
    pred_size = sample_block_size(
        g, PRED_MASK_SCALE, ASPECT_RATIO
    )  # same size for whole batch
    enc_size = sample_block_size(g, ENC_MASK_SCALE, (1.0, 1.0))  # square context block

    all_pred, all_enc = [], []
    for _ in range(batch_size):
        pred, allowed = [], torch.ones(GRID, dtype=torch.int32)
        for _ in range(NUM_PRED_MASKS):
            idx, block = sample_block_mask(g, pred_size)
            pred.append(idx)
            allowed *= (
                1 - block
            )  # allow_mask_overlap=False: context never sees a target patch
        enc = [sample_block_mask(g, enc_size, allowed)[0] for _ in range(NUM_ENC_MASKS)]
        all_pred.append(pred)
        all_enc.append(enc)

    # truncate to the smallest mask in the batch so masks can be stacked
    n_tgt = min(len(m) for ms in all_pred for m in ms)
    n_ctx = min(len(m) for ms in all_enc for m in ms)
    predictor_masks = [
        torch.stack([ms[k][:n_tgt] for ms in all_pred]) for k in range(NUM_PRED_MASKS)
    ]
    encoder_masks = [
        torch.stack([ms[k][:n_ctx] for ms in all_enc]) for k in range(NUM_ENC_MASKS)
    ]
    return encoder_masks, predictor_masks  # 1 x (B, N_ctx), 4 x (B, N_tgt)


def apply_masks(x, masks):
    # src: sparsh/tactile_ssl/utils/__init__.py:25
    # x: (B, N, D), masks: list of M index tensors (B, K)  ->  (M*B, K, D), mask-major order
    return torch.cat(
        [
            torch.gather(x, 1, einops.repeat(m, "b k -> b k d", d=x.size(-1)))
            for m in masks
        ],
        dim=0,
    )


# =====================================================================================
# 3. MODEL BUILDING BLOCKS
# =====================================================================================
class PatchEmbed(nn.Module):
    # src: sparsh/tactile_ssl/model/layers/patch_embed.py:39
    def __init__(self, in_chans, dim):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, dim, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)

    def forward(self, x):  # (B, C, H, W) -> (B, N, D), row-major over the 20x15 grid
        return self.proj(x).flatten(2).transpose(1, 2)


def sinusoidal_pos_embed(dim):
    # src: patch_embed.py:133  SinusoidalEmbed (fixed, not learned)
    # per patch (y, x): [sin(y*f), cos(y*f), sin(x*f), cos(x*f)]
    num_bands = math.ceil(dim / (2 * len(GRID)))
    freqs = 10000 ** -torch.linspace(0, 1, num_bands + 1)[:-1]
    grid = torch.stack(
        torch.meshgrid(
            *[torch.arange(n, dtype=torch.float) for n in GRID], indexing="ij"
        ),
        -1,
    )
    features = grid.reshape(-1, len(GRID), 1) * freqs  # (N, 2, num_bands)
    return torch.cat([features.sin(), features.cos()], -1).flatten(-2)[
        :, :dim
    ]  # (N, dim)


class Block(nn.Module):
    """Standard pre-norm transformer block (treated as a black box in the route).
    src: sparsh/tactile_ssl/model/layers/block.py:89, attention.py:60, mlp.py:34, layer_scale.py"""

    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.norm1, self.norm2 = (
            nn.LayerNorm(dim, eps=1e-6),
            nn.LayerNorm(dim, eps=1e-6),
        )
        self.qkv, self.proj = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)
        )
        self.ls1, self.ls2 = (
            nn.Parameter(torch.ones(dim)),
            nn.Parameter(torch.ones(dim)),
        )  # LayerScale

    def attn(self, x):
        B, N, C = x.shape
        q, k, v = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        return self.proj(
            F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, C)
        )

    def forward(self, x):
        x = x + self.ls1 * self.attn(self.norm1(x))
        return x + self.ls2 * self.mlp(self.norm2(x))


# =====================================================================================
# 4. ENCODER (used twice: context encoder and its EMA copy, the target encoder)
# =====================================================================================
class VisionTransformer(nn.Module):
    # src: sparsh/tactile_ssl/model/vision_transformer.py:56   (built by vit_base, :557)
    def __init__(
        self, in_chans=IN_CHANS, dim=ENC_DIM, depth=ENC_DEPTH, num_heads=ENC_HEADS
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(in_chans, dim)
        self.register_buffer(
            "pos_embed", sinusoidal_pos_embed(dim), persistent=False
        )  # (N, D)
        self.blocks = nn.ModuleList([Block(dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x, masks=None):
        # src: vision_transformer.py:300 prepare_tokens_with_masks + :341 forward_features
        x = self.patch_embed(x) + self.pos_embed  # (B, 300, 768)
        if masks is not None:
            x = apply_masks(
                x, masks
            )  # (B, N_ctx, 768): drop everything outside the context
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)  # "x_norm_patchtokens"


# =====================================================================================
# 5. PREDICTOR
# =====================================================================================
class Predictor(nn.Module):
    # src: vision_transformer.py:425  VisionTransformerPredictor   (built by vit_predictor, :505)
    def __init__(
        self, input_dim=ENC_DIM, dim=PRED_DIM, depth=PRED_DEPTH, num_heads=PRED_HEADS
    ):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, dim)
        self.register_buffer(
            "pos_embed", sinusoidal_pos_embed(dim), persistent=False
        )  # (N, 192)
        self.mask_token = nn.Parameter(torch.zeros(1, dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.blocks = nn.ModuleList([Block(dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.output_projection = nn.Linear(dim, input_dim)

    def forward(self, ctx, encoder_masks, predictor_masks):
        # src: vision_transformer.py:448  forward   (assumes NUM_ENC_MASKS == 1)
        B, n_ctx, _ = ctx.shape
        pos = einops.repeat(self.pos_embed, "n d -> b n d", b=B)  # (B, 300, 192)

        x = self.input_projection(ctx)  # (B, N_ctx, 192)
        x = x + apply_masks(
            pos, encoder_masks
        )  # context tokens get their positions back

        queries = self.mask_token + apply_masks(
            pos, predictor_masks
        )  # (4B, N_tgt, 192)
        x = einops.repeat(
            x, "b n d -> (k b) n d", k=len(predictor_masks)
        )  # one context copy per target
        x = torch.cat([x, queries], dim=1)  # (4B, N_ctx + N_tgt, 192)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)[:, n_ctx:]  # keep only the target positions
        return self.output_projection(x)  # (4B, N_tgt, 768)


# =====================================================================================
# 6-8. I-JEPA: target encoder, loss, EMA
# =====================================================================================
class IJEPA(nn.Module):
    # src: sparsh/tactile_ssl/algorithm/ijepa.py:24  IJEPAModule
    def __init__(self):
        super().__init__()
        self.context_encoder = VisionTransformer()
        self.predictor = Predictor()
        self.target_encoder = copy.deepcopy(self.context_encoder).requires_grad_(
            False
        )  # ijepa.py:65
        self.generator = torch.Generator()

    def training_step(self, x, step):
        # src: ijepa.py:331  training_step  (minus the online probe)
        self.generator.manual_seed(step)
        encoder_masks, predictor_masks = sample_masks(self.generator, x.size(0))
        return self.forward(x, encoder_masks, predictor_masks)

    def forward(self, x, encoder_masks, predictor_masks):
        # src: ijepa.py:298  forward
        ctx = self.context_encoder(x, encoder_masks)  # (B, N_ctx, 768)
        pred = self.predictor(ctx, encoder_masks, predictor_masks)  # (4B, N_tgt, 768)
        with torch.no_grad():
            tgt = self.target_encoder(x)  # (B, 300, 768): full, unmasked image
            tgt = F.layer_norm(tgt, tgt.shape[-1:])  # extra non-affine LayerNorm
            tgt = apply_masks(tgt, predictor_masks)  # (4B, N_tgt, 768)
        return F.smooth_l1_loss(pred, tgt)  # ijepa.py:328

    @torch.no_grad()
    def ema_update(self, m):
        # src: ijepa.py:109 on_train_batch_end -> sparsh/tactile_ssl/utils/ema.py:12
        for p_tgt, p_ctx in zip(
            self.target_encoder.parameters(), self.context_encoder.parameters()
        ):
            p_tgt.mul_(m).add_(p_ctx.detach(), alpha=1.0 - m)  # tgt = m*tgt + (1-m)*ctx


def momentum_schedule(total_steps):
    # src: ijepa.py:422  linear ramp 0.996 -> 1.0 over the whole training
    m0, m1 = EMA_DECAY
    return (m0 + i * (m1 - m0) / total_steps for i in range(total_steps + 1))


# =====================================================================================
# Demo: two training steps on random "raw DIGIT frames"
# =====================================================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)
    frames = [
        np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8) for _ in range(20)
    ]  # landscape, like the .pkl
    bg = np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8)
    loader = DataLoader(
        TactileDataset(frames, bg), batch_size=4, shuffle=True, drop_last=True
    )

    model = IJEPA()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4
    )
    num_steps = 2
    momentum = momentum_schedule(num_steps)

    # src: sparsh/tactile_ssl/trainer/trainer.py:266 train_loop / :418 training_step
    for step, batch in zip(range(num_steps), loader):
        x = batch["image"]  # (B, 6, 320, 240)
        loss = model.training_step(x, step)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        model.ema_update(next(momentum))
        print(f"step {step}: x {tuple(x.shape)}  loss {loss.item():.4f}")
