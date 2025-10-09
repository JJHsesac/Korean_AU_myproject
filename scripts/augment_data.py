import pandas as pd
import random
import os

def aeda_augmentation(text, num_aug=1):
    """AEDA: 랜덤 위치에 구두점 삽입"""
    punctuations = ['.', ',', '!', '?', ';']
    
    augmented_texts = []
    words = text.split()
    
    for _ in range(num_aug):
        new_words = words.copy()
        # 30% 확률로 구두점 삽입
        num_insertions = max(1, int(len(words) * 0.3))
        
        for _ in range(num_insertions):
            pos = random.randint(0, len(new_words))
            punct = random.choice(punctuations)
            new_words.insert(pos, punct)
        
        augmented_texts.append(' '.join(new_words))
    
    return augmented_texts

# 원본 데이터 로드
input_dir = './NIKL_AU_2023_COMPETITION_v1.0'
output_dir = './data/raw_aeda'

# 출력 디렉토리 생성
os.makedirs(output_dir, exist_ok=True)

print("=== AEDA 증강 시작 ===\n")

for filename in ['train.csv', 'dev.csv', 'test.csv']:
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)
    
    print(f"처리 중: {filename}")
    df = pd.read_csv(input_path)
    print(f"  원본: {len(df)}개")
    
    if filename == 'train.csv':
        # train만 증강 (2배)
        augmented_data = []
        for _, row in df.iterrows():
            # 원본 추가
            augmented_data.append(row)
            
            # 증강본 추가
            aug_text = aeda_augmentation(row['input'], num_aug=1)[0]
            aug_row = row.copy()
            aug_row['input'] = aug_text
            augmented_data.append(aug_row)
        
        df_aug = pd.DataFrame(augmented_data)
        print(f"  증강 후: {len(df_aug)}개 (2배)")
    else:
        # dev, test는 원본 그대로
        df_aug = df
        print(f"  증강 안함: {len(df_aug)}개")
    
    # 저장 (index=False로 인덱스 제거)
    df_aug.to_csv(output_path, index=False, encoding='utf-8')
    print(f"  저장 완료: {output_path}\n")

print("=== 증강 완료 ===")

# 결과 확인
print("\n최종 파일:")
for filename in ['train.csv', 'dev.csv', 'test.csv']:
    path = os.path.join(output_dir, filename)
    df_check = pd.read_csv(path)
    print(f"  {filename}: {len(df_check)}개")