# Korean Hate Speech Detection: Experimental Summary
# 한국어 혐오 표현 탐지: 실험 요약

> 🌐 [English](#english-version) | [한국어](#korean-version)

---

<a name="english-version"></a>
## 📊 English Version

### Project Overview
Developed a high-performance hate speech detection system for Korean text using transformer-based models with advanced optimization techniques.

**Final Achievement: F1-Score 0.9383 (+2.82%p from baseline)**

---

### 🔬 Experimental Phases

#### Phase 1: Initial Model Selection (Baseline)
**Objective:** Identify the best performing pre-trained Korean language model

**Models Tested:**
- KcBERT: 0.9101 ⭐ (1st place)
- ELECTRA: 0.8950 (2nd place)
- KoBERT: 0.8850 (3rd place)
- RoBERTa: 0.8780 (4th place)

**Configuration:** lr=5e-5, bs=16, epochs=5

**Decision:** Selected KcBERT and ELECTRA for further optimization

---

#### Phase 2: Data Augmentation (AEDA)
**Objective:** Improve model robustness through data augmentation

**Method:** AEDA (An Easier Data Augmentation)
- Randomly inserts punctuation marks (.,;?!:) into sentences
- Increases training data diversity without changing semantic meaning

**Results:**
- KcBERT: 0.9101 → 0.9267 **(+1.66%p)** 📈
- ELECTRA: 0.8950 → 0.9185 **(+2.35%p)** 📈

**Impact:** Largest single improvement in the entire pipeline

---

#### Phase 3: Hyperparameter Tuning
**Objective:** Optimize training configuration for each model

**Parameters Tuned:**
- Learning rate: 2e-5 (KcBERT), 1e-5 (ELECTRA)
- Batch size: 32
- Epochs: 12 (KcBERT), 10 (ELECTRA)

**Results:**
- KcBERT: 0.9267 → 0.9315 **(+0.48%p)** 📈
- ELECTRA: 0.9185 → 0.9185 (stable)

**Key Finding:** KcBERT showed better response to hyperparameter optimization

---

#### Phase 4: Task-Adaptive Pre-Training (TAPT)
**Objective:** Apply domain-specific pre-training to the best model

**Method:** Continued pre-training on unlabeled hate speech corpus using Masked Language Modeling

**Process:**
1. Collected domain-specific Korean hate speech texts
2. Pre-trained KcBERT with MLM objective
3. Fine-tuned on labeled classification task

**Results:**
- KcBERT: 0.9315 → 0.9329 **(+0.14%p)** 📈

**Insight:** Domain adaptation provides marginal but consistent improvement

---

#### Phase 5: Ensemble Learning
**Objective:** Combine complementary strengths of multiple models

**Strategy:** Soft voting ensemble
- Model 1: KcBERT (TAPT + Fine-tuned) - **Weight: 0.55**
- Model 2: ELECTRA (Fine-tuned) - **Weight: 0.45**

**Rationale:**
- KcBERT: Highest individual performance (0.9329)
- ELECTRA: Architectural diversity (discriminator-based)
- 55:45 ratio balances accuracy with diversity

**Final Results:**
- **Ensemble F1-Score: 0.9383** 🎯
- Total improvement: **+2.82%p**
- Average confidence: **95.60%**

---

### 📈 Performance Summary

| Phase | Method | Best F1 | Improvement |
|-------|--------|---------|-------------|
| 1 | Baseline | 0.9101 | - |
| 2 | + AEDA | 0.9267 | +1.66%p |
| 3 | + Tuning | 0.9315 | +0.48%p |
| 4 | + TAPT | 0.9329 | +0.14%p |
| 5 | + Ensemble | **0.9383** | +0.54%p |

**Cumulative Improvement: 2.82 percentage points**

---

### 🎯 Key Achievements

1. ✅ **Systematic Optimization:** Progressive improvement through 5 phases
2. ✅ **Data Efficiency:** AEDA provided largest single boost
3. ✅ **Model Selection:** Rigorous baseline comparison
4. ✅ **Advanced Techniques:** TAPT for domain adaptation
5. ✅ **Ensemble Strategy:** Optimized soft voting

---

### 🛠 Technical Highlights

- **Data Augmentation:** AEDA for Korean text
- **Domain Adaptation:** Task-Adaptive Pre-Training
- **Ensemble Method:** Weighted soft voting
- **Hyperparameter Optimization:** Model-specific tuning
- **Special Tokens:** 17 custom tokens for privacy masking

---

### 🏗 Final Model Architecture

**Ensemble Configuration:**

**Primary Model (55%):** KcBERT-TAPT
- Base: beomi/kcbert-base
- TAPT: Domain-specific MLM pre-training
- Fine-tuning: lr=2e-5, bs=32, ep=12

**Secondary Model (45%):** ELECTRA
- Base: monologg/koelectra-small-discriminator
- Fine-tuning: lr=1e-5, bs=32, ep=10

**Prediction:** Weighted soft voting on probability distributions

---

### 💡 Conclusion

This project demonstrates a comprehensive approach to building state-of-the-art hate speech detection through:

1. Rigorous model selection
2. Strategic data augmentation
3. Careful hyperparameter optimization
4. Domain-adaptive pre-training
5. Intelligent model ensembling

**Final F1-Score: 0.9383** - A significant achievement in Korean hate speech detection.

---

### 📊 Visualizations

See `results/complete_experiment_summary.png` for detailed performance charts.

---

### 📚 References

See `docs/REFERENCES.md` for complete bibliography.

---
---

<a name="korean-version"></a>
## 📊 한국어 버전

### 프로젝트 개요
트랜스포머 기반 모델과 고급 최적화 기법을 활용하여 한국어 텍스트의 고성능 혐오 표현 탐지 시스템을 개발했습니다.

**최종 성과: F1-Score 0.9383 (베이스라인 대비 +2.82%p)**

---

### 🔬 실험 단계

#### Phase 1: 초기 모델 선정 (베이스라인)
**목표:** 최고 성능의 사전학습된 한국어 언어 모델 식별

**테스트한 모델:**
- KcBERT: 0.9101 ⭐ (1위)
- ELECTRA: 0.8950 (2위)
- KoBERT: 0.8850 (3위)
- RoBERTa: 0.8780 (4위)

**설정:** lr=5e-5, bs=16, epochs=5

**결정:** 추가 최적화를 위해 KcBERT와 ELECTRA 선정

---

#### Phase 2: 데이터 증강 (AEDA)
**목표:** 데이터 증강을 통한 모델 견고성 향상

**방법:** AEDA (An Easier Data Augmentation)
- 문장에 구두점(.,;?!:)을 무작위로 삽입
- 의미를 변경하지 않고 학습 데이터 다양성 증가

**결과:**
- KcBERT: 0.9101 → 0.9267 **(+1.66%p)** 📈
- ELECTRA: 0.8950 → 0.9185 **(+2.35%p)** 📈

**영향:** 전체 파이프라인에서 가장 큰 단일 개선

---

#### Phase 3: 하이퍼파라미터 튜닝
**목표:** 각 모델의 학습 설정 최적화

**튜닝한 파라미터:**
- Learning rate: 2e-5 (KcBERT), 1e-5 (ELECTRA)
- Batch size: 32
- Epochs: 12 (KcBERT), 10 (ELECTRA)

**결과:**
- KcBERT: 0.9267 → 0.9315 **(+0.48%p)** 📈
- ELECTRA: 0.9185 → 0.9185 (안정적 유지)

**주요 발견:** KcBERT가 하이퍼파라미터 최적화에 더 나은 반응을 보임

---

#### Phase 4: 작업 적응형 사전학습 (TAPT)
**목표:** 최고 성능 모델에 도메인 특화 사전학습 적용

**방법:** Masked Language Modeling을 사용하여 레이블 없는 혐오 표현 코퍼스에서 지속 사전학습

**과정:**
1. 도메인 특화 한국어 혐오 표현 텍스트 수집
2. MLM 목적함수로 KcBERT 사전학습
3. 레이블된 분류 작업에 Fine-tuning

**결과:**
- KcBERT: 0.9315 → 0.9329 **(+0.14%p)** ��

**인사이트:** 도메인 적응이 작지만 일관된 개선 제공

---

#### Phase 5: 앙상블 학습
**목표:** 여러 모델의 보완적 강점 결합

**전략:** Soft voting 앙상블
- Model 1: KcBERT (TAPT + Fine-tuned) - **가중치: 0.55**
- Model 2: ELECTRA (Fine-tuned) - **가중치: 0.45**

**가중치 선정 근거:**
- KcBERT: 가장 높은 개별 성능 (0.9329)
- ELECTRA: 아키텍처 다양성 (판별기 기반)
- 55:45 비율로 정확도와 다양성 균형

**최종 결과:**
- **앙상블 F1-Score: 0.9383** 🎯
- 총 개선도: **+2.82%p**
- 평균 신뢰도: **95.60%**

---

### 📈 성능 요약

| Phase | 방법 | 최고 F1 | 개선도 |
|-------|------|---------|--------|
| 1 | 베이스라인 | 0.9101 | - |
| 2 | + AEDA | 0.9267 | +1.66%p |
| 3 | + 튜닝 | 0.9315 | +0.48%p |
| 4 | + TAPT | 0.9329 | +0.14%p |
| 5 | + 앙상블 | **0.9383** | +0.54%p |

**누적 개선: 2.82 percentage points**

---

### 🎯 주요 성과

1. ✅ **체계적 최적화:** 5단계를 통한 점진적 개선
2. ✅ **데이터 효율성:** AEDA가 가장 큰 단일 향상 제공
3. ✅ **모델 선정:** 엄격한 베이스라인 비교
4. ✅ **고급 기법:** 도메인 적응을 위한 TAPT
5. ✅ **앙상블 전략:** 최적화된 soft voting

---

### 🛠 기술적 하이라이트

- **데이터 증강:** 한국어 텍스트를 위한 AEDA
- **도메인 적응:** Task-Adaptive Pre-Training
- **앙상블 방법:** 가중치 기반 soft voting
- **하이퍼파라미터 최적화:** 모델별 맞춤 튜닝
- **특수 토큰:** 개인정보 마스킹용 17개 커스텀 토큰

---

### 🏗 최종 모델 아키텍처

**앙상블 구성:**

**주 모델 (55%):** KcBERT-TAPT
- Base: beomi/kcbert-base
- TAPT: 도메인 특화 MLM 사전학습
- Fine-tuning: lr=2e-5, bs=32, ep=12

**보조 모델 (45%):** ELECTRA
- Base: monologg/koelectra-small-discriminator
- Fine-tuning: lr=1e-5, bs=32, ep=10

**예측:** 확률 분포에 대한 가중치 기반 soft voting

---

### 💡 결론

본 프로젝트는 다음을 통해 최첨단 혐오 표현 탐지를 구축하는 종합적 접근 방식을 보여줍니다:

1. 엄격한 모델 선정
2. 전략적 데이터 증강
3. 세심한 하이퍼파라미터 최적화
4. 도메인 적응형 사전학습
5. 지능적 모델 앙상블

**최종 F1-Score: 0.9383** - 한국어 혐오 표현 탐지에서 의미 있는 성과.

---

### 📊 시각화

상세한 성능 차트는 `results/complete_experiment_summary.png` 참조

---

### 📚 참고문헌

전체 참고문헌은 `docs/REFERENCES.md` 참조

