# main.py

from media_score import run_method1
from ml_model_1 import run_method2
from ml_model_2 import run_method3
from utils import ensemble

def main():
    # 테스트용 뉴스 데이터
    news_sample = {
        "title": "정부, 새로운 정책 발표",
        "content": "오늘 정부는 ...",
        "media": "연합뉴스"
    }

    # 각 방법별 점수 계산
    score1 = run_method1(news_sample)
    score2 = run_method2(news_sample)
    score3 = run_method3(news_sample)

    print(f"[Method1] 언론사 기반 점수: {score1}")
    print(f"[Method2] ML 모델1 점수: {score2}")
    print(f"[Method3] ML 모델2 점수: {score3}")

    # 앙상블 점수
    final_score = ensemble([score1, score2, score3])
    print(f"\n[Final] 최종 신뢰도 점수: {final_score}")

if __name__ == "__main__":
    main()
