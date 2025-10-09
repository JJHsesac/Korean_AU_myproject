import pytorch_lightning as pl
import os
import torch
from utils import compute_metrics
from data import prepare_dataset

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
)
from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback
from transformers.optimization import get_cosine_with_hard_restarts_schedule_with_warmup


def load_tokenizer_and_model_for_train(args):
    """학습(train)을 위한 사전학습(pretrained) 토크나이저와 모델을 huggingface에서 load"""
    # load model and tokenizer
    MODEL_NAME = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Special Token 추가
    special_tokens = [
        '&name&', '&location&', '&organization&', 
        '&account&', '&address&', '&number&',
        '&site&', '&email&', '&phone&', 
        '&url&', '&id&', '&product&',
        '&bank&', '&card&', '&date&',
        '&money&', '&company&'
    ]
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})

    # setting model hyperparameter
    model_config = AutoConfig.from_pretrained(MODEL_NAME)
    model_config.num_labels = 2
    print(model_config)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, config=model_config
    )
    print("--- Modeling Done ---")
    return tokenizer, model

def load_model_for_inference(model_name,model_dir):
    """추론(infer)에 필요한 모델과 토크나이저 load """
    # load tokenizer
    Tokenizer_NAME = model_name
    tokenizer = AutoTokenizer.from_pretrained(Tokenizer_NAME)

    ## load my model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, config=model_config
    )
    
    # 토크나이저 확장에 맞춰 임베딩 조정
    model.resize_token_embeddings(len(tokenizer))
    
    print(f"Special tokens added: {len(special_tokens)}")
    print("--- Modeling Done ---")
    return tokenizer, model


def load_trainer_for_train(args, model, hate_train_dataset, hate_valid_dataset):
    training_args = TrainingArguments(
        output_dir=args.save_path + "/results",
        save_total_limit=2,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_steps=args.save_step,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=8,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_dir=args.save_path + "logs",
        logging_steps=args.logging_step,
        evaluation_strategy="epoch",
        eval_steps=args.eval_step,
        load_best_model_at_end=True,
        report_to="wandb",
        run_name=args.run_name,
        save_strategy="epoch",
        save_safetensors=False,
    )

    # optimizer와 scheduler 정의 (이 부분이 누락되었음)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.weight_decay,
        amsgrad=False,
    )
    
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=len(hate_train_dataset) * args.epochs,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=hate_train_dataset,
        eval_dataset=hate_valid_dataset,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, scheduler)
    )
    
    return trainer


def train(args):
    """모델을 학습(train)하고 best model을 저장"""
    # fix a seed
    pl.seed_everything(seed=42, workers=False)

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # set model and tokenizer
    tokenizer, model = load_tokenizer_and_model_for_train(args)
    model.to(device)

    # set data
    hate_train_dataset, hate_valid_dataset, hate_test_dataset, test_dataset = (
        prepare_dataset(args.dataset_dir, tokenizer, args.max_len)
    )

    # set trainer
    trainer = load_trainer_for_train(
        args, model, hate_train_dataset, hate_valid_dataset
    )

    # train model
    print("--- Start train ---")
    trainer.train()
    print("--- Finish train ---")
    
    # 최종 모델 저장
    print("\n=== 최종 모델 저장 ===")
    final_save_path = os.path.join(args.save_path, "final_model")
    os.makedirs(final_save_path, exist_ok=True)
    
    try:
        # 텐서 contiguous 처리
        print("텐서 contiguous 처리 중...")
        for name, param in model.named_parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()
        
        for name, buffer in model.named_buffers():
            if not buffer.is_contiguous():
                buffer.data = buffer.data.contiguous()
        
        # 저장
        model.save_pretrained(final_save_path, safe_serialization=True)
        tokenizer.save_pretrained(final_save_path)
        print(f"✅ 저장 완료: {final_save_path}\n")
        
    except Exception as e:
        print(f"❌ Safe save failed: {e}")
        print("PyTorch 형식으로 재시도...")
        
        try:
            torch.save(model.state_dict(), f"{final_save_path}/pytorch_model.bin")
            tokenizer.save_pretrained(final_save_path)
            model.config.save_pretrained(final_save_path)
            print("✅ PyTorch 형식 저장 성공!")
        except Exception as e2:
            print(f"❌ All saves failed: {e2}")
    
    return trainer
    
