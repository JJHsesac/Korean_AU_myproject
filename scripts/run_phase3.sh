#!/bin/bash
set +e

echo "=== Phase 3: KcBERT 하이퍼파라미터 튜닝 시작 ==="
echo "시작 시간: $(date)"
echo ""

mkdir -p logs

# 1. lr=2e-5 (baseline)
echo "1/2: lr=2e-5 실행 중..."
python src/main.py \
  --run_name "kcbert_aug_lr2e5_bs32_ep10_v2" \
  --model_name "beomi/kcbert-base" \
  --dataset_dir "./data/raw_aeda" \
  --lr 2e-5 \
  --batch_size 32 \
  --epochs 10 \
  > logs/kcbert_lr2e5_v2.log 2>&1
echo "lr=2e-5 완료 (종료 코드: $?)"

# 2. lr=3e-5 (higher)
echo "2/2: lr=3e-5 실행 중..."
python src/main.py \
  --run_name "kcbert_aug_lr3e5_bs32_ep10_v2" \
  --model_name "beomi/kcbert-base" \
  --dataset_dir "./data/raw_aeda" \
  --lr 3e-5 \
  --batch_size 32 \
  --epochs 10 \
  > logs/kcbert_lr3e5_v2.log 2>&1
echo "lr=3e-5 완료 (종료 코드: $?)"

echo ""
echo "=== Phase 3 완료 ==="
echo "종료 시간: $(date)"
