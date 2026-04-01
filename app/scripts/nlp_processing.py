import pandas as pd
import pickle
import os
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from .metadata import custom_stops

def run():
    print("\n>>> [Step 4 & 5] NLP 전처리 및 벡터화 시작")
    
    # 1. 데이터 로드
    csv_path = "./data/step1_cleaned.csv"
    if not os.path.exists(csv_path):
        print("[오류] 이전 단계의 CSV 파일을 찾을 수 없습니다.")
        return
        
    df = pd.read_csv(csv_path)
    text = df[df['type'] == 'cleaned_text']['content'].values[0]

    # 2. 문장 분리 및 전처리
    sentences = sent_tokenize(text)
    print(f"[디버그] 분리된 문장 수: {len(sentences)}개")

    preprocessed_docs = []
    lemmatizer = WordNetLemmatizer()

    for sent in sentences:
        tokens = word_tokenize(sent)
        if not tokens: continue
        
        tagged = pos_tag(tokens)
        sentence_tokens = []
        
        for word, pos in tagged:
            word_low = word.lower()
            if word_low not in custom_stops and len(word_low) > 1:
                lemma = lemmatizer.lemmatize(word_low, pos='v' if pos.startswith('V') else 'n')
                if "_" in word_low or pos.startswith(('N', 'J', 'V', 'R', 'PRP')):
                    sentence_tokens.append(lemma)
        
        if sentence_tokens:
            preprocessed_docs.append(sentence_tokens)

    print(f"[디버그] 유효 학습 문장 수: {len(preprocessed_docs)}개")
    print(f"[디버그] 전처리 샘플(1번 문장): {preprocessed_docs[0]}")

    # 3. 인코딩 및 패딩
    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(preprocessed_docs)
    sequences = tokenizer.texts_to_sequences(preprocessed_docs)
    padded_data = pad_sequences(sequences, maxlen=50, padding='post')

    # 4. pickle 저장 (패딩 데이터는 대용량이므로 pkl 유지, 리스트는 csv 병행)
    with open('./data/final_data.pkl', 'wb') as f:
        pickle.dump({'padded_data': padded_data, 'tokenizer': tokenizer}, f)
    
    # 전처리된 문장 리스트를 CSV로 저장
    docs_df = pd.DataFrame({'tokens': [",".join(doc) for doc in preprocessed_docs]})
    docs_df.to_csv("./data/step2_preprocessed_docs.csv", index=False, encoding='utf-8-sig')
    
    print(f"[디버그] 최종 데이터셋 쉐이프: {padded_data.shape}")
    print(f"[디버그] 결과 파일 저장 완료 (pkl & csv)")