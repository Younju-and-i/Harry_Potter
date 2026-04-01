import re
import json
from metadata import metadata

# [1] 데이터 로드 및 기본 정제
with open('./Harry_Potter_all_books_preprocessed.txt', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('\n', ' ').strip()
text = re.sub(r'\s+', ' ', text)
text = re.sub(r'\.(?=[A-Z])', '. ', text)

# [2] 복합어 치환 및 빈도 분석용 텍스트 준비
text_for_analysis = text.lower()
for cat, items in metadata.items():
    for item in items:
        if " " in item:
            # 실제 텍스트 내에서 띄어쓰기를 언더바로 연결
            text = re.sub(r'\b' + re.escape(item) + r'\b', item.replace(" ", "_"), text, flags=re.IGNORECASE)

# [3] 빈도 분석 로직 (기존 Step 4 내용)
for category, items in metadata.items():
    category_total = 0
    results = []
    
    for item in items:
        # 풀네임 검색
        full_name_pattern = r'\b' + re.escape(item.lower()) + r'\b'
        count = len(re.findall(full_name_pattern, text_for_analysis))
        
        # 등장인물의 경우 성을 뗀 이름만 나오는 경우도 합산
        if category == "등장인물" and " " in item:
            first_name = item.split()[0].lower()
            only_first_name_pattern = r'\b' + re.escape(first_name) + r'\b(?!\s' + re.escape(item.split()[-1].lower()) + r')'
            count += len(re.findall(only_first_name_pattern, text_for_analysis))
            
        category_total += count
        results.append((item, count))
    
    print(f"\n◈ {category} (총 {category_total}회 등장)")
    results.sort(key=lambda x: x[1], reverse=True)
    for name, freq in results[:5]:
        if freq > 0:
            print(f"   - {name}: {freq}회")

# [4] 정제된 텍스트 저장
with open('./data/cleaned_harry.txt', 'w', encoding='utf-8') as f:
    f.write(text)

print("Step 1 완료: 정제된 텍스트가 cleaned_harry.txt로 저장되었습니다.")