# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environments and Installation

Only the `lapa` env exists in practice (the `laq` env described below was never created). Use `lapa` for everything:

**LAQ / LAQ-AD stage (nuScenes driving experiments):**
```bash
conda activate lapa   # torch 2.12, accelerate 1.13, nuscenes-devkit 1.2.0, einops
cd laq_ad && pip install -e .   # installs laq_ad_model; laq_model imported via sys.path
```
Key deps: torch≥2.0, accelerate, einops, nuscenes-devkit==1.2.0, pyquaternion, ultralytics, wandb, opencv-python≥4.13. Use `cv2.DISOpticalFlow_create` — there is no `cv2.optflow` module in cv2 4.13+.

**Latent pretraining stage:**
```bash
conda activate lapa
pip install -r requirements.txt
```

## Common Commands

### nuScenes data preparation (run from repo root)
```bash
# mini (default)
python data/prepare_nuscenes_laq.py
python data/prepare_nuscenes_egomotion.py

# trainval
python data/prepare_nuscenes_laq.py \
    --dataroot /media/andy/Samsung_T7/nuScenes/Trainval \
    --version v1.0-trainval \
    --output /home/andy/Dataset/nuscenes_laq_frames
python data/prepare_nuscenes_egomotion.py --overwrite    # regenerate after trainval frames added

python data/prepare_nuscenes_heatmaps.py                 # YOLO detection heatmaps (optional)
python data/prepare_nuscenes_costmaps.py                 # BEV cost maps (optional, Phase 2)
```

### Training (run from `laq/` directory)
```bash
python train_nuscenes.py                      # 8-code baseline (no heatmap)
python train_nuscenes_egomotion_mag8.py       # Exp 1: magnitude-only gate — best NMI
python train_nuscenes_egomotion_absflow8.py   # Exp 2: |fx|+|fy| gate
python train_nuscenes_egomotion_additive8.py  # Exp 3: additive direction embedding
```

### Evaluation (run from `laq/` directory)
```bash
python eval_compare_checkpoints.py   # N-way checkpoint comparison: NMI, perplexity, costmap sim
python eval_latent_clusters.py       # bar chart: latent code × maneuver distribution
```

### Tests (require CUDA — NSVQ codebook buffers are GPU-only)
```bash
cd laq
pytest tests/
pytest tests/test_latent_action_quantization_backward_compat.py -v
```

### Latent pretraining inference (run from repo root)
```bash
python -m latent_pretraining.inference
```

## Architecture

### Two-stage pipeline
1. **LAQ** (`laq/`): frame-pair VQ-VAE that compresses (frame_t, frame_{t+offset}) into discrete codebook indices (latent action tokens).
2. **Latent Pretraining** (`latent_pretraining/`): LLaMA-based model trained on those tokens. Downstream stage; not currently being modified.

### LAQ module layout (`laq/laq_model/`)
| File | Role |
|---|---|
| `latent_action_quantization.py` | `LatentActionQuantization`: patch embed → spatial transformer → temporal transformer → NSVQ VQ → cross-attn decoder → pixel recon |
| `laq_trainer.py` | `LAQTrainer`: `accelerate`-backed training loop; `_unpack_batch` always returns 3-tuple `(video, heatmap_or_None, costmap_or_None)` — both train and validation callers must unpack all three |
| `nsvq.py` | NSVQ quantizer; codebook buffers hardcoded to `device='cuda'`; `replace_unused_codebooks` called on a step schedule inside `forward` |
| `heatmap_conditioning.py` | `apply_patch_heatmap(patch_tokens, heatmap, alpha, alpha_dir, abs_dir)` and `EgoMotionDirectionEmbedding`; handles 4D (single-channel) and 5D (3-channel) heatmaps |
| `data.py` | `ImageVideoDataset` (base), `HeatmapVideoDataset` (4D YOLO), `EgoMotionVideoDataset` (5D egomotion with backward-compat for old 2D caches) |
| `eval_utils.py` | Shared helpers: `load_nuscenes_pairs`, `maneuver_label`, `normalized_mutual_information`, `mean_intra_cluster_costmap_similarity` |
| `costmap_loss.py` | BEV cost-map regularizer — negative result at nuScenes-mini scale; documented and not called in the forward pass |

### nuScenes data layout
```
/home/andy/Dataset/nuscenes_laq_frames/<scene_token>/
    frame_0001.jpg
    frame_0001.egomotion.npy   # (3, 8, 8) float32 — [mag/peak, fx/peak, fy/peak]
    frame_0001.heatmap.npy     # (8, 8) float32 — YOLO detection saliency
    frame_0001.costmap.npy     # (32, 32) float32 — BEV collision risk
```
The last frame in each scene has no `.egomotion.npy` (no successor); the dataset falls back to zeros.

## Key Technical Notes

### NSVQ fragility
Any auxiliary loss gradient introduced during early training (< ~700 steps) collapses the codebook permanently. The `costmap_loss_weight` and `costmap_warmup_steps` attributes are commented out in `__init__` and `forward`; `_effective_costmap_loss_weight` is dead code kept for reference. The controlled isolation test (`.detach()` on the loss output) that proved the gradient is the cause is documented in `README.md`'s "Phase 2" section and `laq_model/costmap_loss.py`'s docstring. The hard-gate mechanism is unit-tested in `tests/test_latent_action_quantization_backward_compat.py`.

### Egomotion heatmap format (3-channel)
`(3, 8, 8)` float32: channel 0 = `mag/peak ∈ [0,1]`, channel 1 = `fx/peak ∈ [-1,1]`, channel 2 = `fy/peak ∈ [-1,1]`. All channels share the same peak normalization. Old 2D `(8, 8)` caches are auto-upgraded in `EgoMotionVideoDataset._load_egomotion` by zero-padding direction channels; run `--overwrite` to get real direction features.

### Heatmap conditioning is strictly additive
`heatmap=None` or `heatmap_alpha=0.0` → output identical to unconditioned baseline. Regression-tested in `tests/test_latent_action_quantization_backward_compat.py`.

### Signed direction gate causes codebook collapse
Gate `= 1 + alpha*mag + alpha_dir*(fx+fy)` can go below 1.0, suppressing patches and routing 60%+ of samples to one code (NMI=0.0754). Safe alternatives: `heatmap_abs_dir=True` (uses `|fx|+|fy|`, gate always ≥ 1) or `heatmap_dir_alpha=0.0` (magnitude only).

### Experiment results (nuScenes-mini, 8 codes, 3001 steps)
| Run | NMI | Notes |
|---|---|---|
| Baseline (no heatmap) | 0.2150 | |
| **Exp 1: mag gate only** (`heatmap_dir_alpha=0.0`) | **0.2415** | Best overall |
| Exp 2: abs-value direction (`heatmap_abs_dir=True`) | 0.2037 | Stable but no gain over mag-only |
| Exp 3: additive embedding (`heatmap_additive_dir=True`) | 0.1232 | Linear(2→512) underfits ~2,300 samples |
| Signed direction gate (earlier, collapsed) | 0.0754 | Gate goes below 1 → collapse |

Key insight: the SPATIAL PATTERN of flow magnitude alone distinguishes left/right turns — a right turn produces high magnitude on the LEFT patches and vice versa. Signed direction channels add instability without discriminability gain at this data scale.

## `laq_ad/` — Clean AD Implementation

Production autonomous-driving LAQ under `laq_ad/`. Install alongside `laq`:
```bash
conda activate lapa
cd laq_ad && pip install -e .
```

### Commands (run from `laq_ad/`)
```bash
python train_nuscenes_mini.py    # smoke-test: nuScenes-mini, [3,3] → 9 codes
python train_nuscenes.py         # full trainval: [3,3] → 9 codes
# Sweep all checkpoints in a results folder:
python eval_sweep_checkpoints.py --results results_laq_ad_trainval_3x3_canStrong
python eval_sweep_checkpoints.py --results results_laq_ad_mini --version v1.0-mini \
    --dataroot /home/andy/Dataset/Nuscenes/v1.0-mini
```

### What changed vs `laq/`
| Feature | laq/ | laq_ad/ |
|---|---|---|
| Quantizer | NSVQ (collapses under aux gradients) | **FSQ** (no collapse, no commitment loss) |
| Codebook size | 8 codes | `prod(levels)` — 9 codes ([3,3]) with independent heads |
| Auxiliary loss | none | **CAN bus**: predict yaw_rate + speed from latent |
| Data sampler | random | **ManeuverBalancedSampler** (oversamples turns) |
| Image loading | exported frames only | nuScenes API + frames_dir with egomotion caches |

### `laq_ad_model/` layout
| File | Role |
|---|---|
| `fsq.py` | `FSQ(levels)` — Mentzer 2023; straight-through rounding per dimension |
| `model.py` | `LAQADModel`: patch embed → transformers → CNN compress → FSQ → cross-attn decoder + CAN bus head + flow prediction head |
| `heatmap.py` | `apply_patch_heatmap` — magnitude-only gate (proven best in laq/ experiments) |
| `data.py` | `NuScenesLAQDataset` + `maneuver_balanced_sampler` + `CANBusCache` |
| `trainer.py` | `LAQADTrainer`: accelerate-backed, cosine LR with warmup, logs recon/CAN/perplexity |
| `eval_utils.py` | `normalized_mutual_information`, `print_summary` |

### FSQ collapse notes (hard-won, do not repeat)

Six distinct failure modes:

| Failure | Symptom | Cause | Fix |
|---------|---------|-------|-----|
| Saturation collapse | All → code 0 | z_pre drifts to -∞ (large negative weights), tanh gradient = 0 | weak `spread_reg` (mean penalty) keeps z_pre mean ≈ 0 |
| Posterior collapse | All → code 4 (center) | Decoder ignores z_q; z_pre → 0 (min-effort constant) | `spread_reg` std_penalty forces `std(z_pre) ≈ 1` → can't stay constant |
| Boundary attraction | High soft H but perplexity drops | entropy_reg on z_bounded pushes samples to code boundaries; soft entropy ≈ max but hard perplexity falls | **Remove entropy_reg entirely** (`entropy_reg_weight=0.0`) |
| Multi-D correlation collapse | Only 15-39 / 1200 codes used, NMI≈0.01 | All FSQ dims share one `project_in` linear → correlated → only 1D manifold visited | Per-dim independent heads (`nn.ModuleList`) + `levels=[3,3]` |
| Bimodal collapse (1D) | codes=2 by step 5000, NMI≈0.003 on trainval | CAN bus creates binary "moving vs stationary" pressure; bimodal z_pre at ±1 satisfies spread_reg (mean=0, std=1) but uses only 2 bins | **Use `levels=[3,3]`** (2D) with independent heads — each axis captures one CAN bus signal (yaw, speed) independently |
| Random axis assignment | ~50% of runs produce no left-turn code (NMI≈0.28 instead of 0.37) | Shared `Linear(2,64)` CAN head lets model swap which axis predicts yaw vs speed; both assignments satisfy the loss equally — random seed determines which emerges | **Per-axis `Linear(1,1)` heads**: axis 0 is FORCED to predict yaw_rate, axis 1 FORCED to predict speed. Assignment is fixed by construction regardless of seed. |
| spread_reg overrides flow signal | codes=4 (corners only), NMI≈0.002 despite 9 codes globally active | `spread_reg_weight=2.0` contributes 0.37 to loss (std_penalty at natural distribution std≈0.57) vs flow_loss=0.04 → 9× stronger than signal | **`spread_reg_weight=0.05`, `flow_prediction_weight=5.0`** — flow dominates 22× over spread |

### K-means upper bound (established, do not re-derive)

K-means on raw egomotion heatmaps (192D) achieves **NMI=0.37** with K=4 on nuScenes trainval. The flow signal IS highly maneuver-discriminative:
- Right turn: 99% pure (K=4), 98% pure (K=9)
- Left turn: 67-78% pure
- Stationary: 55-82% pure
- Straight: 74% pure with 1 dominant code

Any model achieving NMI << 0.10 is blocked by regularization or architecture, not data quality.

### Key diagnostic: spread_reg std_penalty vs natural distribution

The natural class distribution (49% straight → z_pre≈0, 16% left → z_pre≈−1, 17% right → z_pre≈+1, 18% stationary → z_pre≈0) has std≈0.57 per axis, not 1.0. The std_penalty `(0.57−1)² = 0.185` weighted at 2.0 = **0.37 loss contribution** — 9× stronger than the flow signal (0.04). This forces samples away from maneuver-optimal positions.

### Decoder shortcut is NOT an issue in laq_ad

Confirmed experimentally: removing reconstruction loss (`recon_loss_weight=0.0`) has ZERO effect on z_q learning — flow loss, code counts, NMI are identical with or without reconstruction. The reconstruction gradient through action_ctx is negligible from the start.

### Working config for trainval (confirmed NMI≈0.40, EXCEEDS K-means upper bound):
```python
LAQADModel(levels=[3,3], spread_reg_weight=0.05, entropy_reg_weight=0.0,
           can_bus_weight=1.0, covariance_reg_weight=1.0,
           flow_prediction_weight=5.0, recon_loss_weight=0.0)
# can_bus_weight=1.0: REQUIRED for reliable yaw-axis bootstrapping. 0.5 is seed-sensitive
#   (~50% of runs fail to separate left/right). 1.0 reaches NMI=0.40 in 26K steps
#   vs 50K with 0.5, and does so consistently across seeds.
# flow_prediction_weight=5.0: flow predicts (3,8,8) egomotion from z_q directly.
# spread_reg_weight=0.05: prevents mean drift without forcing std=1.
# covariance_reg: prevents correlated bimodal collapse.
# recon_loss_weight=0.0: reconstruction gradient is negligible (confirmed ablation).
# Results folder: results_laq_ad_trainval_3x3_canStrong2
# Best checkpoint: step 26K (NMI=0.3998)
#
# CRITICAL: CANBusHead uses per-axis Linear(1,1) heads, NOT shared Linear(fsq_dim,2).
#   Shared head has axis-assignment symmetry (~50% wrong-axis failure). Per-axis heads
#   fix assignment: axis0=yaw_rate, axis1=speed, axis N≥2 free (shaped by flow loss).
#   For fsq_dim > 2 (e.g. [3,3,3]), only the first 2 axes get CAN heads; axis 2 is free.
```

### Experiment results (nuScenes trainval, [3,3] FSQ)
| Run | Steps | NMI | Notes |
|---|---|---|---|
| All prior trainval runs (spread=2.0) | 10K | 0.001-0.008 | spread_reg 9× stronger than flow |
| flowStrong (spread=0.05, flow=5.0, can=0.05) | 10K | 0.2839 | spread weakened, flow dominant |
| canStrong, shared MLP head (can=0.5) | 26K | 0.3663 | exceeded K-means K=9 but seed-sensitive |
| canStrong, shared MLP head (can=0.5) | 50K | 0.2946 | bad seed — speed captured axis 0 instead of yaw |
| perAxisCAN (per-axis Linear(1,1), can=0.5) | 50K | 0.4007 | exceeds K-means K=4, but seed-sensitive |
| perAxisCAN re-run (can=0.5) | 55K | 0.3283 | bad seed — axis 0 didn't capture yaw |
| **canStrong2 (per-axis, can=1.0)** | **26K** | **0.3998** | **reliable; all 9 codes active** |
| K-means upper bound (K=4, raw flow) | — | 0.3734 | model surpasses this |
| K-means upper bound (K=9, raw flow) | — | 0.3586 | model surpasses this |

### canStrong2 per-code distribution at step 26K (current best):
| Code | n | Dominant | Purity |
|---|---|---|---|
| 0 | 1243 | **right turn** | 98.0% |
| 3 | 2443 | **left turn** | 82.8% |
| 6 | 214 | **left turn (sharp)** | 92.5% |
| 7 | 5533 | **stationary** | 95.9% |
| 2 | 6752 | straight | 83.1% |
| 5 | 7300 | straight | 76.6% |
| 1 | 5458 | straight+right | 54.7% str / 40.2% right |
| 4 | 2319 | mixed | 38.8% str / 40.2% left |
| 8 | 337 | mixed small | 62.3% str |

NMI plateaus at 0.39-0.40 from step 22K; training beyond 26K gives no further gain.
The NMI=0.3998 at step 26K equals the original 0.4007 (50K) — same performance ceiling,
reached in half the steps.

### Pending improvements
- **SigLIP NaFlex** spatial encoder (replaces simple patch embed; needs timm ≥ 1.0)
- Multi-scale temporal offsets (3/12/36 frames) in a single model
- Trajectory consistency metric (code → future waypoint distribution)
