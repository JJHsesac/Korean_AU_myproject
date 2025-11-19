"""
test.csv 추론 결과를 WandB에 등록
"""

import wandb
import pandas as pd
import numpy as np

# test 예측 결과 로드
test_df = pd.read_csv("./results/test_predictions.csv")

# WandB 초기화
run = wandb.init(
    project="korean-hate-speech",
    name="FINAL_test_inference",
    job_type="test_prediction"
)

# 예측 통계
non_hate_count = (test_df['prediction'] == 0).sum()
hate_count = (test_df['prediction'] == 1).sum()
avg_confidence = test_df['confidence'].mean()

# Summary에 기록
wandb.summary.update({
    "total_samples": len(test_df),
    "non_hate_count": int(non_hate_count),
    "hate_count": int(hate_count),
    "non_hate_ratio": float(non_hate_count / len(test_df)),
    "hate_ratio": float(hate_count / len(test_df)),
    "avg_confidence": float(avg_confidence),
    "min_confidence": float(test_df['confidence'].min()),
    "max_confidence": float(test_df['confidence'].max()),
    "model1": "KcBERT_TAPT",
    "model2": "ELECTRA",
    "ensemble_weights": "0.55:0.45"
})

# Config
wandb.config.update({
    "dataset": "test.csv",
    "model1_path": "./models/kcbert_tapt_final_fixed",
    "model2_path": "./models/electra_final_v2_fixed",
    "ensemble_method": "soft_voting"
})

# Table로 샘플 데이터 업로드 (처음 100개)
sample_df = test_df.head(100)
table = wandb.Table(dataframe=sample_df)
wandb.log({"test_predictions_sample": table})

# 신뢰도 분포 히스토그램
wandb.log({
    "confidence_distribution": wandb.Histogram(test_df['confidence'])
})

print(f"\n✅ WandB 등록 완료!")
print(f"URL: https://wandb.ai/{run.entity}/{run.project}/runs/{run.id}")
print(f"\n=== 통계 ===")
print(f"총 샘플: {len(test_df)}개")
print(f"Non-Hate: {non_hate_count}개 ({non_hate_count/len(test_df)*100:.1f}%)")
print(f"Hate: {hate_count}개 ({hate_count/len(test_df)*100:.1f}%)")
print(f"평균 신뢰도: {avg_confidence:.2%}")

wandb.finish()
