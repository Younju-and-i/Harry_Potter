import re
import os
import pandas as pd
from metadata import metadata

def run():
    print("\n>>> [Step 2 & 3] 데이터 정제 및 고유명사 치환 시작")
    
    # 1. 경로 설정 및 초기화
    save_path = "./data"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"[디버그] {save_path} 폴더가 없어 생성했습니다.")

    # 2. 원본 로드
    file_path = './Harry_Potter_all_books_preprocessed.txt'
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    print(f"[디버그] 원본 로드 완료: {len(text)}자")

    # 3. 기본 정제
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\.(?=[A-Z])', '. ', text)

    # 4. 고유명사 및 가문 치환
    text_for_analysis = text.lower()
    
    # 복합어 치환 (Harry Potter -> Harry_Potter)
    for cat_name, items in metadata.items():
        for item in items:
            if " " in item:
                text = re.sub(r'\b' + re.escape(item) + r'\b', item.replace(" ", "_"), text, flags=re.IGNORECASE)

    # 가문 치환 (Potter -> Potter_family)
    for family in metadata["가문"]:
        text = re.sub(r'\b' + re.escape(family) + r's\b', family + "_family", text, flags=re.IGNORECASE)
        text = re.sub(r'\b' + re.escape(family) + r'\b', family + "_family", text, flags=re.IGNORECASE)

    # 5. CSV 저장 (덮어쓰기)
    # 분석용과 전처리용 텍스트를 담은 DataFrame 생성
    df = pd.DataFrame({
        'type': ['cleaned_text', 'analysis_text'],
        'content': [text, text_for_analysis]
    })
    
    csv_file = os.path.join(save_path, "step1_cleaned.csv")
    df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    
    print(f"[디버그] 치환 후 글자 수: {len(text)}자")
    print(f"[디버그] 중간 파일 저장 완료: {csv_file}")