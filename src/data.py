import os
import pandas as pd
import torch


class hate_dataset(torch.utils.data.Dataset):
    """dataframe을 torch dataset class로 변환"""

    def __init__(self, hate_dataset, labels):
        self.dataset = hate_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.dataset.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def load_data(dataset_dir):
    """csv file을 dataframe으로 load"""
    dataset = pd.read_csv(dataset_dir)
    print("dataframe 의 형태")
    print("-" * 100)
    print(dataset.head())
    
    # ===== 라벨 검사 및 정제 (추가!) =====
    if 'output' in dataset.columns:
        # NaN이 아닌 값이 있을 때만 처리 (test는 NaN이라 건너뜀)
        if dataset['output'].notna().any():
            print(f"\n🔍 라벨 검사: {dataset_dir}")
            print(f"  원본 크기: {len(dataset)}")
            print(f"  라벨 종류: {sorted(dataset['output'].dropna().unique())}")
            print(f"  NaN 개수: {dataset['output'].isna().sum()}")
            
            # 0과 1만 남기기
            before = len(dataset)
            dataset = dataset[dataset['output'].isin([0, 1])].copy()
            dataset = dataset.dropna(subset=['output']).copy()
            dataset['output'] = dataset['output'].astype(int)
            after = len(dataset)
            
            removed = before - after
            if removed > 0:
                print(f"  ⚠️ 제거된 데이터: {removed}개")
            print(f"  ✅ 정제 후: {after}개\n")
    # ===== 여기까지 =====
    
    return dataset


def construct_tokenized_dataset(dataset, tokenizer, max_length):
    print("tokenizer 에 들어가는 데이터 형태")
    print(dataset["input"][:5])
    
    model_name = tokenizer.name_or_path
    print(f"토크나이저 모델명: {model_name}")
    
    return_token_type_ids = True
    if "roberta" in model_name.lower():
        return_token_type_ids = False
        print("RoBERTa 감지: return_token_type_ids=False 설정")

    tokenized_senetences = tokenizer(
        dataset["input"].tolist(),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
        return_token_type_ids=return_token_type_ids,
    )
    
     # 디버깅: 토크나이징 결과 확인
    print("토크나이징 결과 키들:", tokenized_senetences.keys())
    print("토크나이징 결과가 None인가?", tokenized_senetences is None)
    
    # ===== 토큰 ID 검증 추가 =====
    input_ids = tokenized_senetences['input_ids']
    vocab_size = tokenizer.vocab_size
    
    max_token_id = input_ids.max().item()
    min_token_id = input_ids.min().item()
    
    print(f"\n🔍 토큰 ID 검증:")
    print(f"  Vocab 크기: {vocab_size} (0~{vocab_size-1})")
    print(f"  실제 범위: {min_token_id}~{max_token_id}")
    
    # 범위 벗어난 토큰 찾기
    invalid_mask = (input_ids >= vocab_size) | (input_ids < 0)
    invalid_count = invalid_mask.sum().item()
    
    if invalid_count > 0:
        print(f"  ⚠️ 잘못된 토큰: {invalid_count}개")
        # [UNK] 토큰으로 교체
        unk_token_id = tokenizer.unk_token_id
        print(f"  🔧 [UNK]({unk_token_id})로 교체")
        input_ids[invalid_mask] = unk_token_id
        print(f"  ✅ 교체 완료! 새 최대값: {input_ids.max().item()}")
    else:
        print(f"  ✅ 모든 토큰 유효\n")
    # ===== 여기까지 =====
    
    return tokenized_senetences


def prepare_dataset(dataset_dir, tokenizer, max_len):
    """학습(train)과 평가(test)를 위한 데이터셋을 준비"""
    # load_data
    train_dataset = load_data(os.path.join(dataset_dir, "train.csv")) 
    valid_dataset = load_data(os.path.join(dataset_dir, "dev.csv"))
    test_dataset = load_data(os.path.join(dataset_dir, "test.csv"))
    print("--- data loading Done ---")
    
    # ===== 전체 요약 출력 (추가!) =====
    print("\n" + "="*50)
    print("📊 최종 데이터셋 요약")
    print("="*50)
    print(f"훈련 데이터: {len(train_dataset)}개")
    print(f"검증 데이터: {len(valid_dataset)}개")
    print(f"테스트 데이터: {len(test_dataset)}개")
    print("="*50 + "\n")
    # ===== 여기까지 =====
    
    # split label
    train_label = train_dataset["output"].values
    valid_label = valid_dataset["output"].values
    test_label = test_dataset["output"].values

    # tokenizing dataset
    tokenized_train = construct_tokenized_dataset(train_dataset, tokenizer, max_len)
    tokenized_valid = construct_tokenized_dataset(valid_dataset, tokenizer, max_len)
    tokenized_test = construct_tokenized_dataset(test_dataset, tokenizer, max_len)
    print("--- data tokenizing Done ---")

    # make dataset for pytorch.
    hate_train_dataset = hate_dataset(tokenized_train, train_label)
    hate_valid_dataset = hate_dataset(tokenized_valid, valid_label)
    hate_test_dataset = hate_dataset(tokenized_test, test_label)
    print("--- pytorch dataset class Done ---")

    return hate_train_dataset, hate_valid_dataset, hate_test_dataset, test_dataset