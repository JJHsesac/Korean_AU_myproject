#!/bin/bash
set +e

export WANDB_PROJECT="korean-hate-speech"

echo "=== ELECTRA 재학습 (저장 포함) ==="
echo "시작: $(date)"

mkdir -p models/electra_final
mkdir -p logs

python src/main.py \
  --run_name "electra_aug_lr1e5_bs32_ep10_final" \
  --model_name "monologg/koelectra-small-discriminator" \
  --dataset_dir "./data/raw_aeda" \
  --lr 1e-5 \
  --batch_size 32 \
  --epochs 10 \
  --save_path "./models/electra_final" \
  > logs/electra_final.log 2>&1

# 수동 저장 추가
echo "모델 복사 중..."
if [ -d "./src/best_model" ]; then
  cp -r ./src/best_model/* ./models/electra_final/ 2>/dev/null
  echo "복사 완료"
fi

echo "완료: $(date)"
