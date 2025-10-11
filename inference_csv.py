"""
CSV 파일로 Inference 수행
"""

import pandas as pd
from inference import HateSpeechDetector
from tqdm import tqdm

def predict_csv(input_csv, output_csv, model1_path, model2_path):
    """
    CSV 파일의 텍스트에 대해 예측 수행
    
    Args:
        input_csv: 입력 CSV 파일 경로
        output_csv: 출력 CSV 파일 경로
        model1_path: KcBERT 모델 경로
        model2_path: ELECTRA 모델 경로
    """
    # 데이터 로드
    print(f"데이터 로딩: {input_csv}")
    df = pd.read_csv(input_csv)
    
    # input 컬럼 확인
    if 'input' not in df.columns:
        raise ValueError("CSV에 'input' 컬럼이 없습니다!")
    
    print(f"총 {len(df)}개 텍스트 예측 시작...")
    
    # 모델 초기화
    detector = HateSpeechDetector(model1_path, model2_path)
    
    # 예측 수행
    texts = df['input'].tolist()
    predictions, probs, confidence = detector.predict(texts)
    
    # 결과 저장
    df['prediction'] = predictions
    df['confidence'] = confidence
    df['prob_non_hate'] = probs[:, 0]
    df['prob_hate'] = probs[:, 1]
    
    # 결과 CSV 저장
    df.to_csv(output_csv, index=False)
    print(f"\n✅ 결과 저장: {output_csv}")
    
    # 간단한 통계
    print(f"\n=== 예측 결과 요약 ===")
    print(f"Non-Hate: {(predictions == 0).sum()}개")
    print(f"Hate: {(predictions == 1).sum()}개")
    print(f"평균 신뢰도: {confidence.mean():.2%}")
    
    return df


if __name__ == "__main__":
    # 예시: dev.csv 예측
    result_df = predict_csv(
        input_csv="./NIKL_AU_2023_COMPETITION_v1.0/dev.csv",
        output_csv="./results/dev_predictions.csv",
        model1_path="./models/kcbert_tapt_final_fixed",
        model2_path="./models/electra_final_v2_fixed"
    )
    
    # 정답과 비교 (dev.csv에 output 컬럼이 있는 경우)
    if 'output' in result_df.columns:
        from sklearn.metrics import f1_score, accuracy_score
        
        y_true = result_df['output']
        y_pred = result_df['prediction']
        
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='micro')
        
        print(f"\n=== 성능 평가 ===")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1-Score: {f1:.4f}")
