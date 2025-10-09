#!/bin/bash
set +e

echo "=== TAPT (Task-Adaptive Pre-Training) 시작 ==="
echo "시작: $(date)"

mkdir -p models/kcbert_tapt
mkdir -p logs

python src/tapt.py > logs/tapt.log 2>&1

echo "=== TAPT 완료 ==="
echo "종료: $(date)"
