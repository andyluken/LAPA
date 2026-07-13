# LAPA: Latent Action Pretraining from Videos
[[Project]](https://latentactionpretraining.github.io/)
[[Paper]](https://arxiv.org/abs/2410.11758)
[[Models]](https://huggingface.co/latent-action-pretraining/LAPA-7B-openx)

**News**

[2025.01.22] LAPA has been accepted to ICLR 2025! 

[2024.11.22] We release the weights for the LAQ pretrained on openx here: https://huggingface.co/latent-action-pretraining/LAPA-7B-openx/blob/main/laq_openx.pt. 

[2024.11.10] LAPA has won the [CoRL 2024 LangRob Workshop](https://sites.google.com/view/langrob-corl24/) **Best Paper Award** (among 75 accepted papers)! 🥳

**LAPA** 

- **Unsupervised approach** for pretraining Vision-Language-Action (VLA) models without ground-truth robot action labels.

- Outperforms the current state-of-the-art VLA model trained with ground-truth actions, building a new **SOTA VLA model**.

- Achieves over **30x** greater pretraining efficiency compared to conventional VLA pretraining.

<div align="center">
  <img src="./imgs/latent_action_pretraining.png"/>
</div>


## Getting Started 

```bash
conda create -n lapa python=3.10 -y
conda activate lapa
git clone https://github.com/LatentActionPretraining/LAPA.git
pip install -r requirements.txt 
mkdir lapa_checkpoints && cd lapa_checkpoints
```
Next, download the model checkpoint from [Huggingface](https://huggingface.co/latent-action-pretraining/LAPA-7B-openx) repository. Download, three files under `lapa_checkpoints` directory. 

```bash
wget https://huggingface.co/latent-action-pretraining/LAPA-7B-openx/resolve/main/tokenizer.model
wget https://huggingface.co/latent-action-pretraining/LAPA-7B-openx/resolve/main/vqgan
wget https://huggingface.co/latent-action-pretraining/LAPA-7B-openx/resolve/main/params
```

To run LAPA checkpoint which is pretrained on [Open-X Embodiment dataset](https://arxiv.org/abs/2310.08864), run the following command:
```bash
cd ..
python -m latent_pretraining.inference
```
This will generate the latent action conditioned on the input image and the natural language instruction.
You can change the input image and the instruction to a custom instance. **Note that the output space is the latent action space (which a space size of $8^4$), which is not the real action space**. To evaluate LAPA, fine-tuning is needed to map the latent space to the real action space (e.g. end-effector).

## Fine-tuning LAPA 
For fine-tuning LAPA on real world trajectories, you have to first preprocess the dataset to discretize the action space. We assume that there is a json file (`--input_path`) where the json file has the following row format:
```json
  {
    "id": "data/finetune_data/episode_0/step_0",
    "image": "data/finetune_data/episode_0/step_0.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nWhat action should the robot take to `pick up the milk and put it in the sink`"
      },
      {
        "from": "gpt",
        "raw_actions": [
          0.0004934221506118809,
          -0.00011252239346504211,
          -0.001941084861755371,
          0.013634951062806884,
          0.013678191591275368,
          -0.004913635449167675,
          0.0
        ],
        "states": {
          "eef_pos": [
            0.24725835025310516,
            -0.022094586864113808,
            0.9283081889152527
          ],
          "eef_euler": [
            3.1202197128587876,
            -0.7113159765223936,
            -0.10937155062330725
          ],
          "gripper_state": 0.0
        }
      }
    ]
  }
```
where `finetune_data` contains the images of fine-tuning trajectories.

Run the following commands to preprocess the fine-tuning dataset and fine-tune LAPA.
```bash
python data/finetune_preprocess.py --input_path "/path_to_json_file" --output_filename "data/real_finetune.jsonl" --csv_filename "data/real_finetune.csv"
./scripts/finetune_real.sh
```
We ran the experiments with 4 80GB-A100 GPUs. To change the number of GPUs being used, change the second index of `--mesh_dim` in the script to the number of GPUs.

For fine-tuning on SIMPLER rollout trajectories (100 trajecories), run the following command:
```bash
./scripts/finetune_simpler.sh
```

After finetuning, to deploy the model, run the following command:
```bash
python -m latent_pretraining.deploy --load_checkpoint "params::/path_to_the_finetuned_ckpt" --action_scale_file "data/real_finetune.csv"
```
where `load_checkpoint` includes the path to the finet-uned checkpoint and `action_scale_file` includes the path to the csv file constructed during data preprocessing of fine-tuning dataset.
 
## Latent Action Quantization 
We provide the code for latent action quantization pretraining.
```bash
conda create -n laq python=3.10 -y
conda activate laq
cd laq
pip install -e .
accelerate launch train_sthv2.py
```
Note that the current data loader code is based on something-something v2 dataset structure where the directory consists of multiple trajectories and each trajectory contain multiple images. To train on custom dataset, either change the data structure or modify the existing data loading code. 

After training, you can use the trained quantization model as an inverse dynamics model to obtain latent actions for training data. 

```bash
python inference_sthv2.py
```
Add arguments based on the training arguements. For the `input_file` argument, it should be a jsonl file which contains `id`, `image`, `instruction` keys as the metadata and `vision` which is the output of the vqgan model consisting of 256 discrete image tokens as the otuput.

## Heatmap-Conditioned LAQ (driving)
For driving video (e.g. nuScenes), `heatmap_lapa/` provides a reusable YOLO-detection-based
saliency heatmap generator, and the LAQ encoder can optionally be conditioned on it so the
latent-action codebook learns to represent dynamic agents (pedestrians, vehicles) rather than
static background, instead of just reconstruction error. This is additive: with no heatmap
supplied, `LatentActionQuantization` behaves exactly as before (see
`laq/tests/test_latent_action_quantization_backward_compat.py`).

```bash
# 1. Detect + generate saliency heatmaps standalone (sanity check / demo)
python -m heatmap_lapa.cli_demo --video heatmap_lapa/demo_videos/traffic.mp4

# 2. Prepare nuScenes frames (if not already done) and cache per-frame heatmaps
python data/prepare_nuscenes_laq.py
python data/prepare_nuscenes_heatmaps.py

# 3. Train the heatmap-conditioned LAQ model (controlled A/B pair with train_nuscenes.py)
cd laq
python train_nuscenes.py            # baseline
python train_nuscenes_heatmap.py    # heatmap-conditioned

# 4. Compare codebook usage and maneuver-cluster alignment between the two checkpoints
python eval_compare_checkpoints.py
```
On nuScenes-mini at the smoke-test model scale, heatmap conditioning improved codebook
perplexity from 1.75 to 3.17 (out of a max of 4) and the normalized mutual information between
codebook index and ground-truth driving maneuver from 0.048 to 0.120 — i.e. the latent codes
spread across the codebook more evenly and align better with actual driving behavior. This is
Phase 1 of a larger roadmap (cost-map latent regularization, a trajectory-heatmap decoder, and
optical-flow contrastive augmentation).

### Phase 2: Cost-Map Latent Regularization (negative result, documented)
Adds a second, optional regularizer that pulls together the latent actions of scenes sharing a
similar collision-risk profile (e.g. both involve a pedestrian about to cross), per the design
doc's "most important modification for safety." The risk signal is a real bird's-eye-view (BEV)
cost map: `heatmap_lapa/costmap.py` projects YOLO detections onto the ego ground plane (a
camera-ray / z=0-plane intersection), weights them by per-class collision risk (pedestrians and
cyclists outweigh vehicles), and boosts risk for objects on the drivable area, queried via
`nuscenes-devkit`'s raster `MapMask` (the mini download only ships the legacy raster maps, not
the vector map-expansion pack, so `MapMask` is used rather than the full `NuScenesMap` API).
Like Phase 1, the wiring is purely additive: `costmap_loss_weight=0.0` (the default) makes
`LatentActionQuantization` behave exactly as if cost-map regularization didn't exist, regardless
of what `costmap` tensor is passed (see the extended cases in
`laq/tests/test_latent_action_quantization_backward_compat.py`).

**The regularizer itself does not work on nuScenes-mini at this model scale, and an extensive,
controlled ablation pins down exactly why.** Summary (full story in
`laq_model/costmap_loss.py`'s docstring):

| Variant tried | Outcome |
|---|---|
| Pull-only loss, raw (unnormalized) latent distance | Collapsed the codebook to 1 entry at every weight 0.001–0.1 |
| Pull-only loss, L2-normalized latent distance | Fixed the scale-collapse exploit (clean monotonic weight-vs-perplexity curve), but still degraded codebook health at every weight tested, with no improvement to cost-map cluster alignment |
| Geometry-matching loss (Gram-matrix MSE, has built-in repulsion) | Still collapsed, within ~60 steps |
| + gradient-norm clipping disabled | No change |
| + linear or hard-gate warmup (held at weight 0 until the codebook stabilizes) | Recovered cleanly *while gated*, collapsed again as soon as the gate opened |
| + codebook_size 4 → 8 | Same collapse pattern, just at 8 codes |
| **Loss output `.detach()`-ed (zero gradient, everything else byte-identical)** | **Recovered cleanly — full codebook usage** |

The last row is the controlled isolation: the exact same forward pass, with the exact same
`return_continuous=True` plumbing, is healthy the moment the gradient is removed. This rules out
the loss formula, the weight, gradient clipping, training-phase timing, and codebook capacity —
the conflict is between *any* live gradient from a pairwise cost-map-similarity objective and
NSVQ's codebook-formation dynamics, at nuScenes-mini's scale (~2,300 training pairs,
codebook_size 4–8). The original LAPA paper's Stage-1 model uses 60k–970k trajectories; closing
this gap likely needs substantially more pretraining data, or an architecturally decoupled
auxiliary branch (which would no longer literally be "regularizing the action latent space," a
real divergence from the design doc's intent) — both out of scope here.

The code ships correct, unit-tested (including `laq/laq_model/latent_action_quantization.py`'s
`costmap_warmup_steps` hard-gate mechanism, useful at larger scale), and ready to retry on a
larger dataset:

```bash
# 1. Cache per-frame BEV cost maps (needs prepare_nuscenes_laq.py + prepare_nuscenes_heatmaps.py first)
python data/prepare_nuscenes_costmaps.py

# 2. Train with cost-map regularization on top of Phase 1
cd laq
python train_nuscenes_costmap.py --costmap-loss-weight 0.3 --costmap-warmup-steps 1500
```
`eval_compare_checkpoints.py` supports an arbitrary list of named checkpoints (see `CONFIGS` in
that file) and reports codebook perplexity, maneuver NMI, and mean intra-codebook-cluster
cost-map cosine similarity (the direct test of whether a regularizer pulled similar-risk scenes
into the same code) for each — add a Phase 2 checkpoint there if you train one that works at a
larger data scale.

Phase 2 needs `nuscenes-devkit` and `pyquaternion` (see `laq/setup.py`). Note:
`nuscenes-devkit==1.2.0`'s declared `numpy<2.0`/`Shapely~=2.0.3` pins are stale — it runs
correctly against this project's numpy 2.x / Shapely 2.x, and no downgrade is performed.

## Latent-Pretraining 
We provide the code to do latent pretraining from pretrained LWM checkpoint. First, download the [LWM-Chat-1M-Jax](https://huggingface.co/LargeWorldModel/LWM-Chat-1M-Jax) model under `lwm_checkpoints` directory. Then, download the pretraining dataset from this [link](https://huggingface.co/latent-action-pretraining/LAPA-7B-openx/resolve/main/latent_action_pretraining_openx.jsonl) under the `data` directory. Run the following command for latent pretraining:
```bash
./scripts/latent_pretrain_openx.sh
```
We experimented with 8 H100 GPUs for 34 hours. We have empirically observed that 70K steps with a batch size of 256 is enough to get decent performance on downstream tasks after fine-tuning.

## SIMPLER
As a reproducible simulation, we release the setup that we tested with. First, install packages required for our latent-pretraining and [SIMPLER](https://github.com/simpler-env/SimplerEnv) following the installation guide. 

The inference script is provided in `scripts/lapa_bridge.sh`.
## Acknowledgement 
The codebase is based on [Large-World-Model](https://github.com/LargeWorldModel/LWM) repository. For latent action quantization, we referred to [Phenaki](https://github.com/lucidrains/phenaki-pytorch) code. For deployment code, we referred to the [OpenVLA](https://github.com/openvla/openvla) code. For the SIMPLER evaluation code, we referred to the [SIMPLER](https://github.com/simpler-env/SimplerEnv) repository.


## Citation

If you use this codebase, or otherwise found our work valuable, please cite:
```
@article{ye2024latent,
  title={Latent Action Pretraining from Videos},
  author={Ye, Seonghyeon and Jang, Joel and Jeon, Byeongguk and Joo, Sejune and Yang, Jianwei and Peng, Baolin and Mandlekar, Ajay and Tan, Reuben and Chao, Yu-Wei and Lin, Bill Yuchen and others},
  journal={arXiv preprint arXiv:2410.11758},
  year={2024}
}
```

If you have additional question, feel free to send an email to latentactionpretraining@gmail.com.

## License

LAPA's code and model weights are released under the MIT License. 
