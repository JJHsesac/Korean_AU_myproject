#!/bin/bash
set +e

export WANDB_PROJECT="korean-hate-speech"

echo "=== KcBERT TAPT Fine-tuning FINAL ==="
echo "시작: $(date)"

mkdir -p models/kcbert_tapt_final
mkdir -p logs

python src/main.py \
  --run_name "FINAL_kcbert_tapt_ft_lr2e5_ep12_ENSEMBLE" \
  --model_name "./models/kcbert_tapt" \
  --dataset_dir "./data/raw_aeda" \
  --lr 2e-5 \
  --batch_size 32 \
  --epochs 12 \
  --save_path "./models/kcbert_tapt_final" \
  > logs/kcbert_tapt_final.log 2>&1

echo "완료: $(date)"
