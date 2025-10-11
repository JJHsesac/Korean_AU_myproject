# 프로젝트 회고 (Project Reflection)

## 💭 기술적 도전과 해결

### 1. Vocab 크기 불일치 에러
**문제:** Special token 17개 추가 시 토크나이저(30017) ≠ 모델(30000)
**해결:** `model.resize_token_embeddings(len(tokenizer))`
**배운 점:** Embedding layer 구조 이해, 모델 확장 방법 학습

### 2. ELECTRA Tensor Contiguous
**문제:** safetensors 저장 시 메모리 비연속 에러
**해결:** `param.data.contiguous()` 또는 `save_safetensors=False`
**배운 점:** PyTorch 메모리 관리, ELECTRA는 `use_fast=False` 필수

### 3. 체크포인트 관리
**문제:** 50+ 실험에서 최종 모델 찾기 어려움
**해결:** 명확한 네이밍 규칙, 토크나이저 함께 저장
**배운 점:** 대규모 실험 관리의 중요성

## 🎯 의사결정

### AEDA 선택: +1.66%p (가장 큰 향상)
### TAPT 도입: +0.14%p (작지만 일관된 개선)
### 앙상블 0.55:0.45: 성능과 다양성 균형

## 📊 최종 성과
- Dev: 0.9383 | Test: 0.9429 ⭐
- Test > Dev = 과적합 없음!

## 🚀 배운 점
- Transformer Fine-tuning 실전
- 체계적 실험 설계
- WandB 활용 실험 관리
- 문제 해결 및 디버깅 능력
