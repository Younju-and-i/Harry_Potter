import nltk
import os
# 각 파일에서 run 함수를 가져옵니다 (파일명이 1_clean_and_token.py 라면 숫자로 시작해서 import가 안 될 수 있으니 파일명을 clean_token.py 등으로 바꾸는 것을 추천합니다)
import scripts.clean_token as step1
import scripts.nlp_processing as step2
import scripts.relation_analysis as step3

def setFolder():
    folder_path = "C:/nltk_data"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        nltk.data.path.append(folder_path) # NLTK가 새 폴더를 인식하게 경로 추가
        print(f"NLTK 폴더 생성 및 경로 추가 완료: {folder_path}")
    else:
        nltk.data.path.append(folder_path)
        print("NLTK 폴더가 이미 존재합니다.")

# def download_resources():
#     print("\n--- [NLTK 리소스 체크 및 다운로드] ---")
#     nltk.download('punkt_tab', download_dir="C:/nltk_data")
#     nltk.download('averaged_perceptron_tagger_eng', download_dir="C:/nltk_data")
#     nltk.download('wordnet', download_dir="C:/nltk_data")
#     nltk.download('stopwords', download_dir="C:/nltk_data")

if __name__ == '__main__':
    # 0. 초기 세팅
    setFolder()
    # download_resources()

    print("\n" + "="*30)
    print("🚀 해리포터 분석 파이프라인 시작")
    print("="*30)

    # 1. 데이터 정제 및 치환
    step1.run()

    # 2. NLP 전처리 (문장 분리, 태깅 등)
    step2.run()

    # 3. 공기행렬 기반 관계 분석
    step3.run()

    print("\n" + "="*30)
    print("✅ 모든 프로세스가 성공적으로 완료되었습니다!")
    print("="*30)