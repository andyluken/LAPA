#!/bin/bash
# Stage 3: Action fine-tuning on CARLA driving trajectories.
#
# Prerequisites:
#   1. Stage 2 checkpoint in results_nuscenes_stage2_200m/<uuid>/
#   2. Run data/prepare_carla_stage3.py to produce carla_stage3.jsonl
#   3. Activate the lapa conda env: conda activate lapa
#
# Usage:
#   bash scripts/train_carla_stage3.sh [stage2_uuid] [extra flags...]
#
#   stage2_uuid — subfolder name inside results_nuscenes_stage2_200m/
#                 Defaults to the latest checkpoint found automatically.
#
# Run from LAPA root:
#   bash scripts/train_carla_stage3.sh a8489aa27b29408581a5f4e6f94adc4e

set -e
cd "$(dirname "$0")/.."

# JAX 0.4.23 cuDNN bindings are incompatible with RTX 5080 (Blackwell, CC 12.0).
export JAX_PLATFORMS=cpu

JSONL_PATH="/home/andy/LAPA/data/carla_stage3.jsonl"
OUTPUT_DIR="results_carla_stage3"

# Resolve Stage 2 checkpoint
if [ -n "$1" ] && [ -d "results_nuscenes_stage2_200m/$1" ]; then
    STAGE2_UUID="$1"
    shift
else
    # Auto-detect: pick the most recently modified uuid directory
    STAGE2_UUID=$(ls -t results_nuscenes_stage2_200m/ 2>/dev/null | head -1)
    if [ -z "$STAGE2_UUID" ]; then
        echo "ERROR: No Stage 2 checkpoint found in results_nuscenes_stage2_200m/"
        echo "       Run bash scripts/train_nuscenes_stage2.sh 200m first."
        exit 1
    fi
fi

STAGE2_CKPT="results_nuscenes_stage2_200m/${STAGE2_UUID}/streaming_params"
if [ ! -f "$STAGE2_CKPT" ]; then
    echo "ERROR: Stage 2 checkpoint not found: $STAGE2_CKPT"
    exit 1
fi

if [ ! -f "$JSONL_PATH" ]; then
    echo "ERROR: $JSONL_PATH not found."
    echo "       Run: python data/prepare_carla_stage3.py --episodes-dir carla_episodes"
    exit 1
fi

echo "=== Stage 3 Action Fine-tuning ==="
echo "  Stage 2 ckpt : $STAGE2_CKPT"
echo "  JSONL data   : $JSONL_PATH ($(wc -l < "$JSONL_PATH") records)"
echo "  Output dir   : $OUTPUT_DIR"
echo ""

python3 -u -m latent_pretraining.train \
    --tokenizer.vocab_file="$(dirname "$0")/../lapa_checkpoints/tokenizer.model" \
    --modality='vision,action,delta' \
    --mesh_dim='1,1,1,1' \
    --use_data_sharded_loader=False \
    --load_llama_config='200m' \
    --update_llama_config="{'delta_vocab_size': 4, 'action_vocab_size': 256}" \
    --load_checkpoint=params::"${STAGE2_CKPT}" \
    --total_steps=2000 \
    --log_freq=10 \
    --save_model_freq=500 \
    --optimizer.adamw_optimizer.lr=2e-5 \
    --optimizer.adamw_optimizer.lr_warmup_steps=10 \
    --optimizer.adamw_optimizer.lr_decay_steps=100 \
    --train_dataset.type='json_vision_delta_action' \
    --train_dataset.delta_vision_action_processor.fields_from_example='fields' \
    --train_dataset.delta_vision_action_processor.n_tokens_per_action=7 \
    --train_dataset.delta_vision_action_processor.n_tokens_per_delta=1 \
    --train_dataset.delta_vision_action_processor.max_n_frames=1 \
    --train_dataset.delta_vision_action_processor.img_aug=False \
    --train_dataset.json_delta_action_dataset.path="$JSONL_PATH" \
    --train_dataset.json_delta_action_dataset.seq_length=384 \
    --train_dataset.json_delta_action_dataset.batch_size=4 \
    --train_dataset.json_delta_action_dataset.use_data_sharded_loader=False \
    --logger.output_dir="$OUTPUT_DIR" \
    --logger.online=False \
    "$@"
