#!/bin/bash
set +e

export WANDB_PROJECT="korean-hate-speech"

echo "=== ELECTRA FINAL ==="
echo "시작: $(date)"

mkdir -p models/electra_final_v2
mkdir -p logs

python src/main.py \
  --run_name "FINAL_electra_aug_lr1e5_ep10_ENSEMBLE" \
  --model_name "monologg/koelectra-small-discriminator" \
  --dataset_dir "./data/raw_aeda" \
  --lr 1e-5 \
  --batch_size 32 \
  --epochs 10 \
  --save_path "./models/electra_final_v2" \
  > logs/electra_final_v2.log 2>&1

echo "완료: $(date)"
