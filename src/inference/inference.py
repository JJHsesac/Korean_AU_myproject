import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

class HateSpeechDetector:
    def __init__(self, model1_path, model2_path, weights=[0.55, 0.45]):
        self.device = torch.device('cpu')
        self.weights = weights
        
        print("모델 로딩 중...")
        self.tokenizer1 = AutoTokenizer.from_pretrained(model1_path)
        self.model1 = AutoModelForSequenceClassification.from_pretrained(model1_path)
        self.model1.to(self.device)
        self.model1.eval()
        
        self.tokenizer2 = AutoTokenizer.from_pretrained(model2_path)
        self.model2 = AutoModelForSequenceClassification.from_pretrained(model2_path)
        self.model2.to(self.device)
        self.model2.eval()
        print("모델 로딩 완료!")
    
    def predict(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        
        encoding1 = self.tokenizer1(texts, truncation=True, padding=True, 
                                     max_length=512, return_tensors='pt')
        with torch.no_grad():
            outputs1 = self.model1(**encoding1)
            probs1 = torch.softmax(outputs1.logits, dim=-1).numpy()
        
        encoding2 = self.tokenizer2(texts, truncation=True, padding=True,
                                     max_length=512, return_tensors='pt')
        with torch.no_grad():
            outputs2 = self.model2(**encoding2)
            probs2 = torch.softmax(outputs2.logits, dim=-1).numpy()
        
        ensemble_probs = probs1 * self.weights[0] + probs2 * self.weights[1]
        predictions = np.argmax(ensemble_probs, axis=1)
        confidence = np.max(ensemble_probs, axis=1)
        
        return predictions, ensemble_probs, confidence
