"""
CSV 예측 결과를 JSONL 형식으로 변환
NIKL 대회 제출 형식에 맞춤
"""

import pandas as pd
import json

def csv_to_jsonl(csv_path, jsonl_path, id_column='id', pred_column='prediction'):
    """
    CSV를 JSONL로 변환
    
    Args:
        csv_path: 입력 CSV 파일 경로
        jsonl_path: 출력 JSONL 파일 경로
        id_column: ID 컬럼명 (기본: 'id')
        pred_column: 예측 컬럼명 (기본: 'prediction')
    """
    # CSV 로드
    print(f"CSV 로딩: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"총 {len(df)}개 데이터")
    
    # 필수 컬럼 확인
    if id_column not in df.columns:
        # id 컬럼이 없으면 인덱스 사용
        df[id_column] = df.index
        print(f"⚠️  '{id_column}' 컬럼이 없어서 인덱스 사용")
    
    if pred_column not in df.columns:
        raise ValueError(f"'{pred_column}' 컬럼이 없습니다!")
    
    # JSONL 생성
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for idx, row in df.iterrows():
            json_obj = {
                "id": str(row[id_column]),
                "output": int(row[pred_column])
            }
            f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
    
    print(f"✅ JSONL 저장: {jsonl_path}")
    
    # 샘플 출력
    print(f"\n=== 샘플 (처음 5개) ===")
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            print(line.strip())
    
    # 통계
    print(f"\n=== 예측 분포 ===")
    print(f"Non-Hate (0): {(df[pred_column] == 0).sum()}개")
    print(f"Hate (1): {(df[pred_column] == 1).sum()}개")


if __name__ == "__main__":
    # test.csv 예측 결과를 JSONL로 변환
    csv_to_jsonl(
        csv_path="./results/test_predictions.csv",
        jsonl_path="./results/test_submission.jsonl",
        id_column='id',  # test.csv에 id 컬럼이 있다면
        pred_column='prediction'
    )
    
    print("\n🎉 제출 파일 준비 완료!")
    print("제출 파일: ./results/test_submission.jsonl")
