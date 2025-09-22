# media_score.py

# 미리 정의된 언론사별 신뢰도 점수 (예시 값)
MEDIA_SCORES = {
    "연합뉴스": 0.9,
    "조선일보": 0.6,
    "한겨레": 0.8,
    "중앙일보": 0.7,
    "기타": 0.5,
    "마이너스" : -0.1
}

def get_media_score(media_name: str) -> float:
    """
    언론사 이름에 따라 신뢰도 점수를 반환
    """
    return MEDIA_SCORES.get(media_name, MEDIA_SCORES["기타"])

def run_method1(news: dict) -> float:
    """
    main.py파일의 news_sample 딕셔너리를 입력받아 news로 저장 후 실수 반환
    media에 news의 media 키값을 저장하고 get_media_score(media) 함수에 media를 넣어 반환
    """
    media = news.get("media", "기타")
    return get_media_score(media)
