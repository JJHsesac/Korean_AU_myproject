#!/bin/bash
set +e

echo "=== Phase 3: Epoch 실험 (LR=2e-5) ==="
echo "시작: $(date)"

mkdir -p logs

# ep=12
echo "1/2: ep=12 실행 중..."
python src/main.py \
  --run_name "kcbert_aug_lr2e5_bs32_ep12_v2" \
  --model_name "beomi/kcbert-base" \
  --dataset_dir "./data/raw_aeda" \
  --lr 2e-5 \
  --batch_size 32 \
  --epochs 12 \
  > logs/kcbert_lr2e5_ep12.log 2>&1
echo "ep=12 완료 (종료 코드: $?)"

# ep=15
echo "2/2: ep=15 실행 중..."
python src/main.py \
  --run_name "kcbert_aug_lr2e5_bs32_ep15_v2" \
  --model_name "beomi/kcbert-base" \
  --dataset_dir "./data/raw_aeda" \
  --lr 2e-5 \
  --batch_size 32 \
  --epochs 15 \
  > logs/kcbert_lr2e5_ep15.log 2>&1
echo "ep=15 완료 (종료 코드: $?)"

echo ""
echo "=== Phase 3 Epoch 실험 완료 ==="
echo "종료: $(date)"

# 완료 신호 파일 생성
touch ~/PHASE3_DONE_$(date +%Y%m%d_%H%M).txt
