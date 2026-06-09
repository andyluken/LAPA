"""
Smoke-test LAQ training on nuScenes mini CAM_FRONT frames.

Run data prep first:
  python ../data/prepare_nuscenes_laq.py

Then train:
  python train_nuscenes.py

Results (reconstruction PNGs + checkpoint) land in results_nuscenes_smoke/.
Each saved PNG is a 3-column grid: [frame_t | frame_t+offset | reconstruction].
"""

import os
os.environ["WANDB_MODE"] = "offline"  # no wandb account needed

import torch
torch.backends.cudnn.enabled = False  # work around cuDNN version mismatch on this system

from laq_model import LAQTrainer, LatentActionQuantization

# Reduced model for smoke test (production: dim=1024, depth=8, heads=16, code_seq_len=4)
laq = LatentActionQuantization(
    dim=256,
    quant_dim=32,
    codebook_size=4,   # 4 codes: batch_size=16 can activate all 4; 8 was too many
    image_size=256,
    patch_size=32,
    spatial_depth=2,
    temporal_depth=2,
    dim_head=64,
    heads=4,           # heads * dim_head = 256 = dim
    code_seq_len=1,
).cuda()

trainer = LAQTrainer(
    laq,
    folder="/tmp/nuscenes_laq_frames",
    offsets=3,          # ~250ms between frames at 12Hz sweep rate
    batch_size=16,      # larger batch → diverse codes activated per step
    grad_accum_every=1,
    train_on_images=False,
    use_ema=False,
    num_train_steps=2001,
    results_folder="results_nuscenes_smoke_v3",
    lr=1e-4,
    save_model_every=1000,
    save_results_every=200,
)

trainer.train()
