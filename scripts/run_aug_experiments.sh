#!/bin/bash

# 에러 발생 시에도 계속 진행
set +e

echo "=== 증강 데이터 실험 시작 ==="
echo "시작 시간: $(date)"

# 로그 디렉토리 생성
mkdir -p logs

# 1. BERT
echo "1/4: BERT 실행 중..."
python src/main.py \
  --run_name "bert_aug_lr2e5_bs32_ep10_v1" \
  --model_name "klue/bert-base" \
  --dataset_dir "./data/raw_aeda" \
  --lr 2e-5 \
  --batch_size 32 \
  --epochs 10 \
  > logs/bert_aug.log 2>&1
echo "BERT 완료 (종료 코드: $?)"

# 2. KcBERT
echo "2/4: KcBERT 실행 중..."
python src/main.py \
  --run_name "kcbert_aug_lr2e5_bs32_ep10_v1" \
  --model_name "beomi/kcbert-base" \
  --dataset_dir "./data/raw_aeda" \
  --lr 2e-5 \
  --batch_size 32 \
  --epochs 10 \
  > logs/kcbert_aug.log 2>&1
echo "KcBERT 완료 (종료 코드: $?)"

# 3. RoBERTa
echo "3/4: RoBERTa 실행 중..."
python src/main.py \
  --run_name "roberta_aug_lr1e5_bs16_ep10_v1" \
  --model_name "klue/roberta-large" \
  --dataset_dir "./data/raw_aeda" \
  --lr 1e-5 \
  --batch_size 16 \
  --epochs 10 \
  > logs/roberta_aug.log 2>&1
echo "RoBERTa 완료 (종료 코드: $?)"

# 4. ELECTRA
echo "4/4: ELECTRA 실행 중..."
python src/main.py \
  --run_name "electra_aug_lr1e5_bs32_ep10_v1" \
  --model_name "monologg/koelectra-small-discriminator" \
  --dataset_dir "./data/raw_aeda" \
  --lr 1e-5 \
  --batch_size 32 \
  --epochs 10 \
  > logs/electra_aug.log 2>&1
echo "ELECTRA 완료 (종료 코드: $?)"

echo "=== 모든 실험 완료 ==="
echo "종료 시간: $(date)"
