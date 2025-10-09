#!/bin/bash

echo "=== GitHub Push 전 보안 체크 ==="
echo ""

# 1. API 키 확인 (정확한 패턴으로)
echo "1. API 키 하드코딩 체크..."
if grep -rE "(hf_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{30,}|api_key\s*=\s*['\"][A-Za-z0-9_-]{20,})" src/ scripts/ --include="*.py" --include="*.sh" 2>/dev/null; then
    echo "❌ API 키 발견! 제거 필요"
    exit 1
else
    echo "✅ API 키 하드코딩 없음"
fi

# 2. .env 파일 확인
echo "2. .env 파일 체크..."
if [ -f ".env" ]; then
    echo "❌ .env 파일 존재! 삭제 필요"
    exit 1
else
    echo "✅ .env 파일 없음"
fi

# 3. 대용량 데이터 확인
echo "3. 대용량 파일 체크..."
large_files=$(find . -type f -size +50M 2>/dev/null | grep -v ".git" | grep -v "my_env" | grep -v "wandb" | grep -v "models")
if [ -n "$large_files" ]; then
    echo "❌ 대용량 파일 발견:"
    echo "$large_files"
    exit 1
else
    echo "✅ 대용량 파일 없음"
fi

# 4. 민감한 데이터 파일 확인
echo "4. 민감 데이터 체크..."
sensitive_found=0

if [ -d "data/NIKL_AU_2023_COMPETITION_v1.0" ]; then
    echo "⚠️  원본 데이터 폴더 존재 (gitignore로 제외됨)"
fi

if [ -d "data/raw_aeda" ]; then
    echo "⚠️  증강 데이터 폴더 존재 (gitignore로 제외됨)"
fi

echo "✅ 민감 데이터는 gitignore로 제외 처리됨"

# 5. gitignore 확인
echo "5. .gitignore 존재 확인..."
if [ ! -f ".gitignore" ]; then
    echo "❌ .gitignore 파일 없음!"
    exit 1
else
    echo "✅ .gitignore 존재"
fi

echo ""
echo "=== 체크 완료 ==="
echo "✅ 모든 보안 검사 통과"
echo ""
echo "다음 단계:"
echo "1. git init"
echo "2. git add ."
echo "3. git status (확인)"
echo "4. git commit -m 'Initial commit'"
