import re
import nltk
import collections
import pickle
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==========================================
# STEP 1: 메타데이터 및 환경 설정
# ==========================================
# 분석할 고유명사 그룹과 인물-가문 매핑 데이터를 정의합니다.
metadata = {
    "등장인물": [
        "Harry Potter", "Ron Weasley", "Hermione Granger", "Albus Dumbledore", "Lord Voldemort", 
        "Severus Snape", "Rubeus Hagrid", "Draco Malfoy", "Neville Longbottom", "Sirius Black", 
        "Remus Lupin", "Minerva McGonagall", "Bellatrix Lestrange", "Lucius Malfoy", "Ginny Weasley", 
        "Molly Weasley", "Arthur Weasley", "Fred Weasley", "George Weasley", "Percy Weasley", 
        "Bill Weasley", "Charlie Weasley", "Luna Lovegood", "Cedric Diggory", "Cho Chang", 
        "Fleur Delacour", "Viktor Krum", "Alastor Moody", "Dolores Umbridge", "Cornelius Fudge", 
        "Rufus Scrimgeour", "Peter Pettigrew", "James Potter", "Lily Potter", "Sybill Trelawney", 
        "Gilderoy Lockhart", "Quirinus Quirrell", "Horace Slughorn", "Argus Filch", "Madam Pomfrey", 
        "Dobby", "Kreacher", "Griphook", "Garrick Ollivander", "Rita Skeeter", "Vernon Dursley", 
        "Petunia Dursley", "Dudley Dursley", "Aunt Marge", "Mrs. Figg"
    ],
    "가문": [
        "Potter", "Weasley", "Malfoy", "Black", "Lestrange", "Longbottom", "Granger", 
        "Lovegood", "Crouch", "Riddle", "Gaunt", "Dumbledore", "Diggory", "Prewett", 
        "Tonks", "Cattermole", "Dursley"
    ],
    "장소": [
        "Hogwarts", "Privet Drive", "The Burrow", "Diagon Alley", "Hogsmeade", "Godric's Hollow", 
        "Ministry of Magic", "Azkaban", "Platform Nine and Three-Quarters", "Gringotts", 
        "The Leaky Cauldron", "Grimmauld Place", "Forbidden Forest", "Hagrid's Hut", "Great Hall", 
        "Room of Requirement", "Chamber of Secrets", "Astronomy Tower", "Malfoy Manor", 
        "Shell Cottage", "St. Mungo's", "Little Whinging", "King's Cross Station"
    ],
    "주문": [
        "Expecto Patronum", "Expelliarmus", "Avada Kedavra", "Crucio", "Imperio", "Lumos", "Nox", 
        "Alohomora", "Wingardium Leviosa", "Stupefy", "Petrificus Totalus", "Accio", "Protego", 
        "Sectumsempra", "Riddikulus", "Confundo", "Morsmordre", "Prior Incantato"
    ],
    "사물/아이템": [
        "Philosopher's Stone", "Elder Wand", "Resurrection Stone", "Invisibility Cloak", 
        "The Marauder's Map", "Tom Riddle's Diary", "The Goblet of Fire", "Time-Turner", 
        "Sword of Gryffindor", "Sorting Hat", "Firebolt", "Nimbus 2000", "Salazar Slytherin's Locket", 
        "Helga Hufflepuff's Cup", "Rowena Ravenclaw's Diadem", "Marvolo Gaunt's Ring", "Pensieve", 
        "Deluminator", "Golden Snitch", "Mirror of Erised", "Naginis", "Monster Book of Monsters"
    ],
    "시간": [
        "Nineteen years later", "First Wizarding War", "Second Wizarding War", "Hogwarts Express", 
        "Halloween", "Christmas Break", "Easter Holidays", "O.W.L.s", "N.E.W.T.s", "Midnight", 
        "Dusk", "Dawn", "Summer Holidays", "Triwizard Tournament", "The Yule Ball"
    ],
    "감정": [
        "Fear", "Bravery", "Anger", "Grief", "Love", "Loyalty", "Hatred", "Regret", "Joy", 
        "Despair", "Hope", "Guilt", "Resentment", "Compassion"
    ],
    "관계": [
        "The Order of the Phoenix", "Death Eaters", "Dumbledore's Army", "Gryffindor", "Slytherin", 
        "Ravenclaw", "Hufflepuff", "Pure-blood", "Muggle-born", "Half-blood", "Mudblood", "Squib"
    ]
}

# 인물별 가문 매핑 (관계성 분석을 위한 기준 테이블)
characters_to_family = {
    "Harry Potter": "potter_family", "James Potter": "potter_family", "Lily Potter": "potter_family",
    "Draco Malfoy": "malfoy_family", "Lucius Malfoy": "malfoy_family",
    "Ron Weasley": "weasley_family", "Fred Weasley": "weasley_family", "George Weasley": "weasley_family",
    "Hermione Granger": "granger_family", "Sirius Black": "black_family",
    "Bellatrix Lestrange": "lestrange_family", "Neville Longbottom": "longbottom_family",
    "Luna Lovegood": "lovegood_family", "Cedric Diggory": "diggory_family",
    "Vernon Dursley": "dursley_family", "Tom Riddle": "riddle_family"
}

# 불용어 설정 (not, he, she 등 문맥 파악에 중요한 단어는 제외 리스트에서 뺌)
stop_words = set(stopwords.words('english'))
exclude_stops = {'not', 'no', 'never', 'he', 'she', 'him', 'her', 'they', 'them'}
custom_stops = stop_words - exclude_stops

# ==========================================
# STEP 2: 데이터 로드 및 텍스트 정제 (Cleaning)
# ==========================================
with open('./Harry_Potter_all_books_preprocessed.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(f"--- [Step 2: 데이터 로드 및 정제] ---")
print(f"정제 전 문자 수: {len(text)}자")

# 1. 줄바꿈 제거 및 연속 공백 축소
text = text.replace('\n', ' ')
text = re.sub(r'\s+', ' ', text)
# 2. 마침표 뒤 공백 강제 삽입 (NLTK 문장 분리 정확도 향상)
text = re.sub(r'\.(?=[A-Z])', '. ', text)

print(f"정제 후 문자 수: {len(text)}자")

# ==========================================
# STEP 3: 고유명사 치환 및 가문 처리 (Normalization)
# ==========================================
# 빈도 분석용 원본 텍스트 보존 (소문자 기준)
text_for_analysis = text.lower()

# 1. 복합 고유명사 치환 (Harry Potter -> Harry_Potter)
for cat_name, items in metadata.items():
    for item in items:
        if " " in item:
            # 단어 경계(\b)를 사용하여 정확히 일치하는 경우만 치환
            text = re.sub(r'\b' + re.escape(item) + r'\b', item.replace(" ", "_"), text, flags=re.IGNORECASE)

# 2. 가문 데이터 치환 (Potters/Potter -> Potter_family)
for family in metadata["가문"]:
    # 복수형(Potters) 먼저 치환 후 단수형 처리
    text = re.sub(r'\b' + re.escape(family) + r's\b', family + "_family", text, flags=re.IGNORECASE)
    text = re.sub(r'\b' + re.escape(family) + r'\b', family + "_family", text, flags=re.IGNORECASE)

print(f"\n--- [Step 3: 고유명사 치환 및 가문 처리] ---")
print(f"치환 후 문자 수: {len(text)}자")

# ==========================================
# STEP 4: 문장 분리 및 NLP 전처리 (Pipeline)
# ==========================================
sentences = sent_tokenize(text)
print(f"\n--- [Step 4: 문장 분리 및 전처리] ---")
print(f"분리된 총 문장 수: {len(sentences)}개")

preprocessed_docs = []
lemmatizer = WordNetLemmatizer()

for sent in sentences:
    # 단어 토큰화 및 품사 태깅 (대소문자 보존 상태에서 수행)
    tokens = word_tokenize(sent)
    if not tokens: continue
    
    tagged = pos_tag(tokens)
    sentence_tokens = []
    
    for word, pos in tagged:
        word_low = word.lower()
        # 불용어 필터링 및 최소 길이(2자 이상) 체크
        if word_low not in custom_stops and len(word_low) > 1:
            # 품사에 따른 표제어 추출 (동사 'v', 명사 'n')
            lemma = lemmatizer.lemmatize(word_low, pos='v' if pos.startswith('V') else 'n')
            
            # 보존 조건: 치환된 단어(_)이거나 명사(N), 형용사(J), 동사(V), 부사(R), 대명사(PRP)
            if "_" in word_low or pos.startswith(('N', 'J', 'V', 'R', 'PRP')):
                sentence_tokens.append(lemma)
    
    if sentence_tokens:
        preprocessed_docs.append(sentence_tokens)

print(f"최종 학습용 문장 수: {len(preprocessed_docs)}개")

# ==========================================
# STEP 5: 정수 인코딩 및 패딩 (Vectorization)
# ==========================================
# 딥러닝 모델 입력을 위한 숫자 시퀀스 변환
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(preprocessed_docs)
sequences = tokenizer.texts_to_sequences(preprocessed_docs)
# 패딩 길이는 50으로 설정 (소설 문맥 보존을 위해 상향)
padded_data = pad_sequences(sequences, maxlen=50, padding='post')

# ==========================================
# STEP 6: 빈도 분석 결과 출력
# ==========================================
print("\n--- [Step 6: 항목별 빈도 분석 결과] ---")

for category, items in metadata.items():
    category_total = 0
    results = []
    
    for item in items:
        # 풀네임/원형 검색
        pattern = r'\b' + re.escape(item.lower()) + r'\b'
        count = len(re.findall(pattern, text_for_analysis))
        
        # 등장인물의 경우 '성'을 제외한 '이름'만 나오는 경우도 추가 합산
        if category == "등장인물" and " " in item:
            first_name = item.split()[0].lower()
            last_name = item.split()[-1].lower()
            # 이름 뒤에 성이 오지 않는 경우만 카운트 (중복 방지)
            name_pattern = r'\b' + re.escape(first_name) + r'\b(?!\s' + re.escape(last_name) + r')'
            count += len(re.findall(name_pattern, text_for_analysis))
            
        category_total += count
        results.append((item, count))
    
    print(f"\n◈ {category} (총 {category_total}회 등장)")
    results.sort(key=lambda x: x[1], reverse=True)
    for name, freq in results[:5]: # 상위 5개 항목 출력
        if freq > 0:
            print(f"   - {name}: {freq}회")

# 최종 요약 정보 출력
print(f"\n[최종 완성] 패딩 데이터 쉐이프: {padded_data.shape}")
print(f"사용된 단어 사전(Vocal) 크기: {len(tokenizer.word_index)}")