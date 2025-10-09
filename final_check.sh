#!/bin/bash

echo "=== 파일 존재 확인 ==="
echo ""

files=(
    "results/all_experiments.csv"
    "results/complete_experiment_summary.png"
    "results/experiment_results.html"
    "results/ensemble_results.csv"
    "docs/EXPERIMENT_SUMMARY.md"
    "src/main.py"
    "src/model.py"
    "src/ensemble.py"
)

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file"
    else
        echo "❌ $file (누락)"
    fi
done

echo ""
echo "=== 디렉토리 확인 ==="
dirs=(
    "src"
    "scripts"
    "results"
    "docs"
    "data"
)

for dir in "${dirs[@]}"; do
    if [ -d "$dir" ]; then
        echo "✅ $dir/"
    else
        echo "❌ $dir/ (누락)"
    fi
done

echo ""
echo "=== 추가 문서 확인 ==="
files_extra=(
    "README.md"
    "requirements.txt"
    ".gitignore"
    "models/README.md"
    "data/README.md"
    "docs/ERRORS.md"
)

for file in "${files_extra[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file"
    else
        echo "❌ $file (누락)"
    fi
done

echo ""
echo "=== 완료 ==="
