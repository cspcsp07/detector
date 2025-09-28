import pandas as pd
import numpy as np

# --- 튜닝 가능한 설정 값 ---
# 1. 라벨별 신뢰도 기여 점수 (사용자 정의)
LABEL_SCORE_MAP = {
    0: 1.0,   # 사실
    1: 0.6,   # 분석
    2: -0.15, # 의견
    3: -0.85  # 미확인
}

# 2. 언론사 초기 신뢰도 가중치 (사전 지식)
INITIAL_SOURCE_WEIGHTS = {
    "MBC": 0.61,
    "JTBC": 0.59,
    "YTN": 0.55,
    "SBS": 0.54,
    "연합뉴스TV": 0.52,
    "KBS": 0.48,
    "MBN": 0.46,
    "한겨레": 0.45,
    "경향신문": 0.43,
    "TV조선": 0.42,
    "조선일보": 0.40,
    "중앙일보": 0.40,
    "동아일보": 0.39,
    "한국일보": 0.32,
    "오마이뉴스": 0.20,
    "채널A": 0.20,
    "문화일보": 0.12,
    "뉴데일리": 0.13,
    "지역신문": 0.39,
    "etc": 0.20  # 신뢰도 정보 미공개 매체의 기본값 (중간값)
}

# 3. 초기 가중치 신뢰도 (Alpha)
ALPHA = 0.7

# 4. 통계적 유의성을 위한 최소 기사 수
MIN_ARTICLES_SOURCE = 3
MIN_ARTICLES_AUTHOR = 3

# 5. 라벨링된 데이터 파일
INPUT_CSV = "comprehensive_test_data.csv"


def calculate_ultimate_weights(df: pd.DataFrame):
    """'라벨 점수제'와 '기자 가중 평균'을 모두 적용하여 최종 가중치를 계산합니다."""
    
    df_clean = df.dropna(subset=['label', 'author'])
    df_clean = df_clean[df_clean['label'].isin([0, 1, 2, 3])]
    df_clean['label'] = pd.to_numeric(df_clean['label'])

    # --- 1단계: 모든 기사의 개별 점수(lk) 계산 ---
    df_clean['article_score'] = df_clean['label'].map(LABEL_SCORE_MAP)

    # --- 2단계: 기자 신뢰도 계산 (이는 기사 가중치 'wk'가 됨) ---
    author_scores = df_clean.groupby('author').agg(
        article_count=('label', 'count'),
        avg_article_score=('article_score', 'mean')
    ).reset_index()
    author_scores = author_scores[author_scores['article_count'] >= MIN_ARTICLES_AUTHOR]
    
    # 기자 점수(-1 ~ +1)를 가중치로 사용하기 위해 0~1 범위로 정규화
    author_scores['credibility_score'] = (author_scores['avg_article_score'] + 1) / 2
    
    # 기자 점수를 딕셔너리로 변환하여 나중에 쉽게 찾아 쓸 수 있도록 함
    author_cred_dict = author_scores['credibility_score'].to_dict()

    # --- 3단계: 기자 점수를 'wk'로 사용하여 언론사 가중 평균 계산 ---
    # 원본 데이터에 기자별 신뢰도 점수(기사 가중치 'wk')를 추가
    df_clean['author_cred_weight'] = df_clean['author'].map(author_cred_dict)
    # 점수가 없는 기자(기사 수가 적어 필터링된)는 중립 가중치 0.5로 설정
    df_clean['author_cred_weight'] = df_clean['author_cred_weight'].fillna(0.5)

    # 가중 평균 공식을 계산하는 사용자 정의 함수
    def weighted_average(group):
        weights = group['author_cred_weight']  # wk
        values = group['article_score']       # lk
        return (weights * values).sum() / weights.sum()

    # 언론사별로 그룹화하여, 기사 수와 '가중 평균된 점수'를 계산
    source_scores = df_clean.groupby('source').agg(
        article_count=('label', 'count'),
        weighted_avg_score=('article_score', lambda x: weighted_average(df_clean.loc[x.index]))
    ).reset_index()

    # 계산된 점수(-1 ~ +1)를 0~1 범위의 observed_score로 정규화
    source_scores['observed_score'] = (source_scores['weighted_avg_score'] + 1) / 2
    
    # --- 4단계: 초기 가중치와 결합 (기존과 동일) ---
    df_initial = pd.DataFrame(INITIAL_SOURCE_WEIGHTS.items(), columns=['source', 'initial_weight'])
    df_merged = pd.merge(df_initial, source_scores, on='source', how='outer')
    df_merged['initial_weight'] = df_merged['initial_weight'].fillna(INITIAL_SOURCE_WEIGHTS['etc'])
    df_merged['article_count'] = df_merged['article_count'].fillna(0).astype(int)
    df_merged['observed_score'] = df_merged['observed_score'].fillna(df_merged['initial_weight'])
    
    df_merged['final_weight'] = np.where(
        df_merged['article_count'] >= MIN_ARTICLES_SOURCE,
        (df_merged['initial_weight'] * ALPHA) + (df_merged['observed_score'] * (1 - ALPHA)),
        df_merged['initial_weight']
    )
    
    source_final_sorted = df_merged.sort_values(by='final_weight', ascending=False)
    author_final_sorted = author_scores[['author', 'article_count', 'credibility_score']].sort_values(by='credibility_score', ascending=False)
    
    return source_final_sorted, author_final_sorted


if __name__ == "__main__":
    try:
        df_labeled = pd.read_csv(INPUT_CSV)
        source_ranking, author_ranking = calculate_ultimate_weights(df_labeled)

        print("--- 최종 언론사별 가중치 (DataFrame 형태) ---")
        source_ranking_output = source_ranking[['source', 'initial_weight', 'observed_score', 'article_count', 'final_weight']]
        print(source_ranking_output.to_string(index=False))

        print("\n\n--- 최종 기자별 가중치 (DataFrame 형태) ---")
        print(author_ranking.to_string(index=False))

        source_weight_dict = source_ranking.set_index('source')['final_weight'].to_dict()
        print("\n\n--- 최종 언론사 가중치 (Dictionary 형태) ---")
        print(source_weight_dict)

    except FileNotFoundError:
        print(f"[ERROR] '{INPUT_CSV}' 파일을 찾을 수 없습니다.")
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")