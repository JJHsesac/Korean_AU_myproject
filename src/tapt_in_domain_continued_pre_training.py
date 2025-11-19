import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
from datasets import Dataset
import os

def prepare_mlm_data(csv_path):
    df = pd.read_csv(csv_path)
    texts = df['input'].tolist()
    return texts

def run_tapt(
    model_name="beomi/kcbert-base",
    data_path="./data/raw_aeda/train.csv",
    output_dir="./models/kcbert_tapt",
    epochs=3,
    batch_size=16,
    mlm_probability=0.15
):
    print(f"=== Task-Adaptive Pre-training (TAPT) : In-domain Continued Pre-training (혐오표현 학습 데이터셋 (AEDA 증강), 테스크와 동일한 도메인 특화 학습 표현 ===")
    print(f"모델: {model_name}")
    print(f"데이터: {data_path}")
    print(f"Epochs: {epochs}\n")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    texts = prepare_mlm_data(data_path)
    print(f"총 {len(texts)}개 문장\n")
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=512,
            padding=False
        )
    
    dataset = Dataset.from_dict({'text': texts})
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text']
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_probability
    )
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        logging_dir=f"{output_dir}/logs",
        report_to="wandb",
        run_name="kcbert_tapt_ep3_v1",
        seed=42,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
    )
    
    print("학습 시작...\n")
    trainer.train()
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\n저장 완료: {output_dir}")
    
    return output_dir

if __name__ == "__main__":
    output = run_tapt(
        model_name="beomi/kcbert-base",
        data_path="./data/raw_aeda/train.csv",
        output_dir="./models/kcbert_tapt",
        epochs=3,
        batch_size=16,
        mlm_probability=0.15
    )
    
    print(f"\n Conditional Task-Adaptive Pre-training (TAPT) 완료: {output}")
    # AEDA 증강 학습 데이터 (혐오표현 도메인)
    # 실제 테스크와 동일한 혐오표현 탐지 테스크 분포
    # Masked Language Modeling (MLM) - Text only without label
    # Task-Adaptive pre-training (TAPT) 방식으로 혐오표현 탐지 도메인에 모델 적응
    # 근거 : Don't Stop Pretraining (Gururangan et al., 2020)의 TAPT 정의
