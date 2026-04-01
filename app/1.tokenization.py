import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import collections

# 1. 메타데이터 정의
metadata = {
    "등장인물": ["Harry Potter", "Ron Weasley", "Hermione Granger", "Albus Dumbledore", "Lord Voldemort", "Severus Snape", "Rubeus Hagrid", "Draco Malfoy", "Neville Longbottom", "Sirius Black", "Remus Lupin", "Minerva McGonagall", "Bellatrix Lestrange", "Lucius Malfoy", "Ginny Weasley", "Molly Weasley", "Arthur Weasley", "Fred Weasley", "George Weasley", "Percy Weasley", "Bill Weasley", "Charlie Weasley", "Luna Lovegood", "Cedric Diggory", "Cho Chang", "Fleur Delacour", "Viktor Krum", "Alastor Moody", "Dolores Umbridge", "Cornelius Fudge", "Rufus Scrimgeour", "Peter Pettigrew", "James Potter", "Lily Potter", "Sybill Trelawney", "Gilderoy Lockhart", "Quirinus Quirrell", "Horace Slughorn", "Argus Filch", "Madam Pomfrey", "Dobby", "Kreacher", "Griphook", "Garrick Ollivander", "Rita Skeeter", "Vernon Dursley", "Petunia Dursley", "Dudley Dursley", "Aunt Marge", "Mrs. Figg"],
    "장소": ["Hogwarts", "Privet Drive", "The Burrow", "Diagon Alley", "Hogsmeade", "Godric's Hollow", "Ministry of Magic", "Azkaban", "Platform Nine and Three-Quarters", "Gringotts", "The Leaky Cauldron", "Grimmauld Place", "Forbidden Forest", "Hagrid's Hut", "Great Hall", "Room of Requirement", "Chamber of Secrets", "Astronomy Tower", "Malfoy Manor", "Shell Cottage", "St. Mungo's", "Little Whinging", "King's Cross Station"],
    "주문": ["Expecto Patronum", "Expelliarmus", "Avada Kedavra", "Crucio", "Imperio", "Lumos", "Nox", "Alohomora", "Wingardium Leviosa", "Stupefy", "Petrificus Totalus", "Accio", "Protego", "Sectumsempra", "Riddikulus", "Confundo", "Morsmordre", "Prior Incantato"],
    "사물/아이템": ["Philosopher's Stone", "Elder Wand", "Resurrection Stone", "Invisibility Cloak", "The Marauder's Map", "Tom Riddle's Diary", "The Goblet of Fire", "Time-Turner", "Sword of Gryffindor", "Sorting Hat", "Firebolt", "Nimbus 2000", "Salazar Slytherin's Locket", "Helga Hufflepuff's Cup", "Rowena Ravenclaw's Diadem", "Marvolo Gaunt's Ring", "Pensieve", "Deluminator", "Golden Snitch", "Mirror of Erised", "Naginis", "Monster Book of Monsters"],
    "시간": ["Nineteen years later", "First Wizarding War", "Second Wizarding War", "Hogwarts Express", "Halloween", "Christmas Break", "Easter Holidays", "O.W.L.s", "N.E.W.T.s", "Midnight", "Dusk", "Dawn", "Summer Holidays", "Triwizard Tournament", "The Yule Ball"],
    "감정": ["Fear", "Bravery", "Anger", "Grief", "Love", "Loyalty", "Hatred", "Regret", "Joy", "Despair", "Hope", "Guilt", "Resentment", "Compassion"],
    "관계": ["The Order of the Phoenix", "Death Eaters", "Dumbledore's Army", "Gryffindor", "Slytherin", "Ravenclaw", "Hufflepuff", "Pure-blood", "Muggle-born", "Half-blood", "Mudblood", "Squib"]
}

# 2. 고도화 세팅 (수정사항 1 반영: 부정어 및 대명사 보존)
stop_words = set(stopwords.words('english'))
exclude_stops = {'not', 'no', 'never', 'he', 'she', 'him', 'her', 'they', 'them'}
custom_stops = stop_words - exclude_stops

# [Step 1] 데이터 로드
with open('./Harry_Potter_all_books_preprocessed.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(f"--- [Step 1: 데이터 로드 및 정제] ---")
print(f"정제 전 문자 수: {len(text)}자")

# --- [추가된 정제 로직: 이 부분이 문장 분할의 핵심입니다] ---
# 1. 불필요한 줄바꿈(\n)을 공백으로 변환
text = text.replace('\n', ' ')
# 2. 여러 개의 공백을 하나로 축소
text = re.sub(r'\s+', ' ', text)
# 3. 마침표(.) 뒤에 공백이 없는 경우 강제로 삽입 (NLTK가 문장을 잘 나눌 수 있게 돕습니다)
text = re.sub(r'\.(?=[A-Z])', '. ', text)
# --------------------------------------------------------

print(f"정제 후 문자 수: {len(text)}자")

# [수정사항 2 반영] 복합어 치환 (Harry Potter -> Harry_Potter)
text_for_analysis = text.lower()
for cat in metadata.values():
    for item in cat:
        if " " in item:
            text = re.sub(r'\b' + re.escape(item) + r'\b', item.replace(" ", "_"), text, flags=re.IGNORECASE)

# [Step 2] 문장 분리 (이제 정제된 텍스트를 분리합니다)
sentences = sent_tokenize(text)
print(f"\n--- [Step 2: 문장 분리 확인] ---")
print(f"NLTK가 분리한 원본 문장 총 개수: {len(sentences)}개") # <-- 여기서 숫자가 크게 늘어야 합니다!

# [Step 3] 전처리 파이프라인
preprocessed_docs = []
lemmatizer = WordNetLemmatizer()
dropped_sentences = 0

for i, sent in enumerate(sentences):
    # POS 태깅 전 단어 토큰화
    tokens = word_tokenize(sent)
    if not tokens: continue
    
    tagged = pos_tag(tokens)
    sentence_tokens = []
    
    for word, pos in tagged:
        word_low = word.lower()
        # 불용어 제거 및 한 글자 단어 제외
        if word_low not in custom_stops and len(word_low) > 1:
            # 표제어 추출
            lemma = lemmatizer.lemmatize(word_low, pos='v' if pos.startswith('V') else 'n')
            # 고도화 조건: 복합어(_)이거나 주요 품사(N, J, V, R, PRP) 보존
            if "_" in word_low or pos.startswith(('N', 'J', 'V', 'R', 'PRP')):
                sentence_tokens.append(lemma)
    
    if sentence_tokens:
        preprocessed_docs.append(sentence_tokens)
    else:
        dropped_sentences += 1

print(f"\n--- [Step 3: 필터링 결과 확인] ---")
print(f"최종 리스트에 담긴 학습용 문장: {len(preprocessed_docs)}개")

# 4. 정수 인코딩 및 패딩 (수정사항 4 반영)
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(preprocessed_docs)
sequences = tokenizer.texts_to_sequences(preprocessed_docs)
padded_data = pad_sequences(sequences, maxlen=50, padding='post')

# --- [빈도 분석 섹션: 빈도 보존 및 로직 최적화] ---
print("\n--- [Step 4: 빈도 분석 결과] ---")

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

print(f"\n[알림] 패딩 길이: {len(padded_data[0])}, 전체 문장 수: {len(preprocessed_docs)}")

print(f"\n[최종 요약] 패딩 길이: {len(padded_data[0])}, 학습용 문장 수: {len(preprocessed_docs)}")