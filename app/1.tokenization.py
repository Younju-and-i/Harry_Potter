import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# NLTK 리소스 다운로드 (처음 실행 시 필요)
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')

# 1. 메타데이터 정의 (제공해주신 데이터)
metadata = {
    "등장인물": ["Harry Potter", "Ron Weasley", "Hermione Granger", "Albus Dumbledore", "Lord Voldemort", "Severus Snape", "Rubeus Hagrid", "Draco Malfoy", "Neville Longbottom", "Sirius Black", "Remus Lupin", "Minerva McGonagall", "Bellatrix Lestrange", "Lucius Malfoy", "Ginny Weasley", "Molly Weasley", "Arthur Weasley", "Fred Weasley", "George Weasley", "Percy Weasley", "Bill Weasley", "Charlie Weasley", "Luna Lovegood", "Cedric Diggory", "Cho Chang", "Fleur Delacour", "Viktor Krum", "Alastor Moody", "Dolores Umbridge", "Cornelius Fudge", "Rufus Scrimgeour", "Peter Pettigrew", "James Potter", "Lily Potter", "Sybill Trelawney", "Gilderoy Lockhart", "Quirinus Quirrell", "Horace Slughorn", "Argus Filch", "Madam Pomfrey", "Dobby", "Kreacher", "Griphook", "Garrick Ollivander", "Rita Skeeter", "Vernon Dursley", "Petunia Dursley", "Dudley Dursley", "Aunt Marge", "Mrs. Figg"],
    "장소": ["Hogwarts", "Privet Drive", "The Burrow", "Diagon Alley", "Hogsmeade", "Godric's Hollow", "Ministry of Magic", "Azkaban", "Platform Nine and Three-Quarters", "Gringotts", "The Leaky Cauldron", "Grimmauld Place", "Forbidden Forest", "Hagrid's Hut", "Great Hall", "Room of Requirement", "Chamber of Secrets", "Astronomy Tower", "Malfoy Manor", "Shell Cottage", "St. Mungo's", "Little Whinging", "King's Cross Station"],
    "주문": ["Expecto Patronum", "Expelliarmus", "Avada Kedavra", "Crucio", "Imperio", "Lumos", "Nox", "Alohomora", "Wingardium Leviosa", "Stupefy", "Petrificus Totalus", "Accio", "Protego", "Sectumsempra", "Riddikulus", "Confundo", "Morsmordre", "Prior Incantato"],
    "사물/아이템": ["Philosopher's Stone", "Elder Wand", "Resurrection Stone", "Invisibility Cloak", "The Marauder's Map", "Tom Riddle's Diary", "The Goblet of Fire", "Time-Turner", "Sword of Gryffindor", "Sorting Hat", "Firebolt", "Nimbus 2000", "Salazar Slytherin's Locket", "Helga Hufflepuff's Cup", "Rowena Ravenclaw's Diadem", "Marvolo Gaunt's Ring", "Pensieve", "Deluminator", "Golden Snitch", "Mirror of Erised", "Naginis", "Monster Book of Monsters"],
    "시간": ["Nineteen years later", "First Wizarding War", "Second Wizarding War", "Hogwarts Express", "Halloween", "Christmas Break", "Easter Holidays", "O.W.L.s", "N.E.W.T.s", "Midnight", "Dusk", "Dawn", "Summer Holidays", "Triwizard Tournament", "The Yule Ball"],
    "감정": ["Fear", "Bravery", "Anger", "Grief", "Love", "Loyalty", "Hatred", "Regret", "Joy", "Despair", "Hope", "Guilt", "Resentment", "Compassion"],
    "관계": ["The Order of the Phoenix", "Death Eaters", "Dumbledore's Army", "Gryffindor", "Slytherin", "Ravenclaw", "Hufflepuff", "Pure-blood", "Muggle-born", "Half-blood", "Mudblood", "Squib"]
}

# 빠른 검색을 위해 모든 메타데이터 단어를 소문자로 집합화
meta_words_set = set()
for cat in metadata.values():
    for item in cat:
        meta_words_set.update(item.lower().split())

# 2. 초기화 및 데이터 로드
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# 파일 로드 부분 수정
with open('./Harry_Potter_all_books_preprocessed.txt', 'r', encoding='utf-8') as f:
    text = f.read()
# 1. 불필요한 줄바꿈(\n)을 공백으로 변환
text = text.replace('\n', ' ')
# 2. 여러 개의 공백을 하나로 축소
text = re.sub(r'\s+', ' ', text)
# 3. 마침표(.) 뒤에 공백이 없는 경우 강제로 삽입 (구분 성능 향상)
text = re.sub(r'\.(?=[A-Z])', '. ', text)

# 3. 전처리 파이프라인 (문장 분리 추가!)
# 그 후 문장 분리 실행
# 이 과정이 있어야 sequences[0]이 전체 텍스트가 아닌 '첫 문장'이 됩니다.
sentences = sent_tokenize(text)
preprocessed_docs = []

for sent in sentences:
    # 단어 토큰화 및 소문자화
    words = word_tokenize(sent.lower())
    # 품사 태깅
    tagged = pos_tag(words)
    
    sentence_tokens = []
    for word, pos in tagged:
        # 불용어 제거 및 1글자 단어 제거
        if word not in stop_words and len(word) > 1:
            # 표제어 추출 (동사/명사 구분)
            lemma = lemmatizer.lemmatize(word, pos='v' if pos.startswith('V') else 'n')
            
            # 메타데이터 키워드이거나 명사(N), 형용사(J), 동사(V)인 경우만 보존
            if lemma in meta_words_set or pos.startswith(('N', 'J', 'V')):
                sentence_tokens.append(lemma)
    
    # 가공된 단어가 있는 문장만 리스트에 추가
    if sentence_tokens:
        preprocessed_docs.append(sentence_tokens)

# 4. 정수 인코딩
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(preprocessed_docs)

# 모든 문장을 숫자로 변환
sequences = tokenizer.texts_to_sequences(preprocessed_docs)

# 5. 패딩
# max_len을 20으로 설정 (문장당 단어 20개 유지)
max_len = 20
padded_data = pad_sequences(sequences, maxlen=max_len, padding='post')

# 결과 확인을 위한 복원
decoded_sentence = tokenizer.sequences_to_texts([sequences[0]])

print(f"원문 첫 문장: {sentences[0]}") 
print(f"전처리 후 단어들: {preprocessed_docs[0]}")
print(f"--- 분석 결과 요약 ---")
print(f"총 문장 수: {len(preprocessed_docs)}")
print(f"단어 집합 크기: {len(tokenizer.word_index)}")
print(f"인코딩 예시 (첫 문장 숫자): {sequences[0]}")
print(f"복원된 첫 문장 (단어): {decoded_sentence[0]}")
print(f"패딩 결과 예시 (첫 문장 20자로 맞춤):\n{padded_data[0]}")

# # 상위 20개 단어 인덱스 확인
# print(f"\n--- 단어 인덱스 상위 20개 ---")
# for word, index in list(tokenizer.word_index.items())[:20]:
#     print(f"{word}: {index}")