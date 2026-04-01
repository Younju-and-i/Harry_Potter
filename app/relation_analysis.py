import re
import pandas as pd
import pickle
import os
from metadata import metadata

def run():
    print("\n>>> [Step 6] 최종 빈도 분석 및 결과 리포트")
    
    # 1. 데이터 로드
    csv1 = "./data/step1_cleaned.csv"
    pkl_file = "./data/final_data.pkl"
    
    if not os.path.exists(csv1) or not os.path.exists(pkl_file):
        print("[오류] 필요한 데이터 파일이 부족합니다.")
        return

    df_analysis = pd.read_csv(csv1)
    text_for_analysis = df_analysis[df_analysis['type'] == 'analysis_text']['content'].values[0]
    
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
        padded_data = data['padded_data']
        tokenizer = data['tokenizer']

    # 2. 빈도 분석 리포트
    for category, items in metadata.items():
        category_total = 0
        results = []
        for item in items:
            pattern = r'\b' + re.escape(item.lower()) + r'\b'
            count = len(re.findall(pattern, text_for_analysis))
            
            if category == "등장인물" and " " in item:
                first_name = item.split()[0].lower()
                last_name = item.split()[-1].lower()
                name_pattern = r'\b' + re.escape(first_name) + r'\b(?!\s' + re.escape(last_name) + r')'
                count += len(re.findall(name_pattern, text_for_analysis))
                
            category_total += count
            results.append((item, count))
        
        print(f"\n◈ {category} (총 {category_total}회)")
        results.sort(key=lambda x: x[1], reverse=True)
        for name, freq in results[:5]:
            if freq > 0: print(f"   - {name}: {freq}회")

    print("\n" + "="*40)
    print(f"최종 패딩 쉐이프: {padded_data.shape}")
    print(f"전체 단어 사전 크기: {len(tokenizer.word_index)}")
    print("="*40)