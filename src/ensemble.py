import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os

class TwoModelEnsemble:
    def __init__(self, model1_path, model2_path, weights=[0.55, 0.45]):
        """
        2개 모델 앙상블
        model1: KcBERT TAPT (더 높은 가중치)
        model2: ELECTRA Phase 2-4
        """
        self.device = torch.device('cpu')
    
        print("=== 모델 로딩 ===")
        print(f"Device: {self.device}\n")
    
        # Model 1: KcBERT TAPT
        print(f"1. KcBERT TAPT 로딩: {model1_path}")
        self.tokenizer1 = AutoTokenizer.from_pretrained(model1_path)
        self.model1 = AutoModelForSequenceClassification.from_pretrained(model1_path)
    
        # 모델과 토크나이저 크기 맞추기
        if len(self.tokenizer1) != self.model1.config.vocab_size:
            print(f"⚠️  크기 불일치 감지: 토크나이저 {len(self.tokenizer1)} vs 모델 {self.model1.config.vocab_size}")
            print("토크나이저를 모델 크기에 맞춤...")
            # 토크나이저를 모델의 vocab_size로 확장
            self.model1.resize_token_embeddings(self.model1.config.vocab_size)
    
        self.model1.to(self.device)
        self.model1.eval()
    
        # Model 2: ELECTRA
        print(f"2. ELECTRA 로딩: {model2_path}")
        self.tokenizer2 = AutoTokenizer.from_pretrained(model2_path)
        self.model2 = AutoModelForSequenceClassification.from_pretrained(model2_path)
        self.model2.to(self.device)
        self.model2.eval()
    
        # 가중치
        self.weights = weights
        print(f"\n가중치: KcBERT {weights[0]}, ELECTRA {weights[1]}")
        print("로딩 완료!\n")
        
    def predict(self, texts, batch_size=32):
        """Soft Voting 예측"""
        all_probs = []
        
        # Model 1 예측
        print("1/2: KcBERT 예측 중...")
        probs1 = self._predict_single(self.model1, self.tokenizer1, texts, batch_size)
        all_probs.append(probs1)
        
        # Model 2 예측
        print("2/2: ELECTRA 예측 중...")
        probs2 = self._predict_single(self.model2, self.tokenizer2, texts, batch_size)
        all_probs.append(probs2)
        
        # Weighted Soft Voting
        print("\n앙상블 집계 중...")
        weighted_probs = (
            all_probs[0] * self.weights[0] + 
            all_probs[1] * self.weights[1]
        )
        
        predictions = np.argmax(weighted_probs, axis=1)
        confidence = np.max(weighted_probs, axis=1)
        
        return predictions, confidence, weighted_probs
    
    def _predict_single(self, model, tokenizer, texts, batch_size):
        """단일 모델 예측"""
        encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )
        
        probs_list = []
        num_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in tqdm(range(0, len(texts), batch_size), total=num_batches):
            batch = {
                k: v[i:i+batch_size].to(self.device) 
                for k, v in encodings.items()
            }
            
            with torch.no_grad():
                outputs = model(**batch)
                probs = torch.softmax(outputs.logits, dim=-1)
                probs_list.append(probs.cpu().numpy())
        
        return np.vstack(probs_list)


def evaluate_ensemble(ensemble, test_df):
    """앙상블 평가"""
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    
    texts = test_df['input'].tolist()
    labels = test_df['output'].values
    
    predictions, confidence, probs = ensemble.predict(texts)
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='micro')
    
    print("\n" + "="*50)
    print("🎯 앙상블 최종 결과")
    print("="*50)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"\n평균 신뢰도: {confidence.mean():.4f}")
    print(f"최소 신뢰도: {confidence.min():.4f}")
    print(f"최대 신뢰도: {confidence.max():.4f}")
    
    print("\n" + "="*50)
    print("📊 분류 리포트")
    print("="*50)
    print(classification_report(
        labels, predictions, 
        target_names=['Non-Hate', 'Hate'],
        digits=4
    ))
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'predictions': predictions,
        'confidence': confidence
    }


if __name__ == "__main__":
    print("="*60)
    print("Phase 5: 2-Model Ensemble (KcBERT TAPT + ELECTRA)")
    print("="*60 + "\n")
    
    # 앙상블 생성
    ensemble = TwoModelEnsemble(
        model1_path="./models/kcbert_tapt_final_fixed",
        model2_path="./models/electra_final_v2_fixed",
        weights=[0.55, 0.45]
    )
    
    # 테스트 데이터로 평가
    test_df = pd.read_csv('./NIKL_AU_2023_COMPETITION_v1.0/dev.csv')
    
    results = evaluate_ensemble(ensemble, test_df)
    
    # 결과 저장
    test_df['ensemble_prediction'] = results['predictions']
    test_df['ensemble_confidence'] = results['confidence']
    test_df.to_csv('./ensemble_results.csv', index=False)
    
    print(f"\n✅ 결과 저장: ./ensemble_results.csv")
    print(f"✅ 최종 F1-Score: {results['f1_score']:.4f}")
