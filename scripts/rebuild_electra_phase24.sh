#!/bin/bash
set +e

export WANDB_PROJECT="korean-hate-speech"

echo "=== ELECTRA Phase 2-4 재학습 ==="
echo "시작: $(date)"

mkdir -p models/electra_phase24
mkdir -p logs

python src/main.py \
  --run_name "electra_aug_lr1e5_bs32_ep10_rebuild" \
  --model_name "monologg/koelectra-small-discriminator" \
  --dataset_dir "./data/raw_aeda" \
  --lr 1e-5 \
  --batch_size 32 \
  --epochs 10 \
  > logs/electra_rebuild.log 2>&1

echo "완료: $(date)"
