#!/bin/bash
set +e

echo "=== TAPT 모델 Fine-tuning 시작 ==="
echo "시작: $(date)"

export WANDB_PROJECT="korean-hate-speech"  # 추가!

mkdir -p logs

python src/main.py \
  --run_name "kcbert_tapt_finetuned_lr2e5_bs32_ep12_v1" \
  --model_name "./models/kcbert_tapt" \
  --dataset_dir "./data/raw_aeda" \
  --lr 2e-5 \
  --batch_size 32 \
  --epochs 12 \
  > logs/kcbert_tapt_finetuned.log 2>&1

echo "완료: $(date)"
