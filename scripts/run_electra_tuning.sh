#!/bin/bash
set +e

echo "=== ELECTRA 하이퍼파라미터 튜닝 시작 ==="
echo "시작: $(date)"

export WANDB_PROJECT="korean-hate-speech"

mkdir -p logs

python src/main.py \
  --run_name "electra_aug_lr2e5_bs32_ep12_v2" \
  --model_name "monologg/koelectra-small-discriminator" \
  --dataset_dir "./data/raw_aeda" \
  --lr 2e-5 \
  --batch_size 32 \
  --epochs 12 \
  > logs/electra_tuned.log 2>&1

echo "완료: $(date)"
