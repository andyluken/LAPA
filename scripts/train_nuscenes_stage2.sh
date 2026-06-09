#!/bin/bash
# Stage 2: Latent pretraining on nuScenes mini driving video.
#
# Prerequisites:
#   1. Run data/prepare_nuscenes_stage2.py to produce /home/andy/LAPA/data/nuscenes_stage2.jsonl
#   2. Activate the lapa conda env: conda activate lapa
#
# Model size options (set LLAMA_CONFIG):
#   debug — hidden=256, 2 layers   (fastest, pipeline smoke test only)
#   200m  — hidden=1024, 14 layers (recommended first real run, ~3GB VRAM)
#   1b    — hidden=2048, 22 layers (~12GB VRAM)
#
# Run from LAPA root:
#   bash scripts/train_nuscenes_stage2.sh

set -e
cd "$(dirname "$0")/.."

# JAX 0.4.23 cuDNN bindings are incompatible with RTX 5080 (Blackwell, CC 12.0).
# Force CPU for now. To use GPU, upgrade: pip install "jax[cuda12]>=0.4.28"
export JAX_PLATFORMS=cpu

LLAMA_CONFIG="${1:-200m}"
JSONL_PATH="/home/andy/LAPA/data/nuscenes_stage2.jsonl"
OUTPUT_DIR="results_nuscenes_stage2_${LLAMA_CONFIG}"

if [ ! -f "$JSONL_PATH" ]; then
    echo "ERROR: $JSONL_PATH not found. Run data/prepare_nuscenes_stage2.py first."
    exit 1
fi

echo "=== Stage 2 Latent Pretraining ==="
echo "  Model config : $LLAMA_CONFIG"
echo "  JSONL data   : $JSONL_PATH ($(wc -l < "$JSONL_PATH") records)"
echo "  Output dir   : $OUTPUT_DIR"
echo ""

python3 -u -m latent_pretraining.train \
    "--tokenizer.vocab_file=$(dirname "$0")/../lapa_checkpoints/tokenizer.model" \
    --modality='vision,text,delta' \
    --mesh_dim='1,1,1,1' \
    --use_data_sharded_loader=False \
    --load_llama_config="$LLAMA_CONFIG" \
    "--update_llama_config={'delta_vocab_size': 4}" \
    --total_steps=2000 \
    --log_freq=10 \
    --save_model_freq=500 \
    --optimizer.adamw_optimizer.lr_warmup_steps=200 \
    --train_dataset.type='json_vision_delta' \
    --train_dataset.delta_vision_text_processor.fields_from_example='fields' \
    --train_dataset.delta_vision_text_processor.n_tokens_per_delta=1 \
    --train_dataset.delta_vision_text_processor.max_n_frames=1 \
    --train_dataset.delta_vision_text_processor.img_aug=False \
    "--train_dataset.json_delta_dataset.path=$JSONL_PATH" \
    --train_dataset.json_delta_dataset.seq_length=384 \
    --train_dataset.json_delta_dataset.batch_size=4 \
    --train_dataset.json_delta_dataset.use_data_sharded_loader=False \
    "--logger.output_dir=$OUTPUT_DIR" \
    --logger.online=False \
    "$@"
