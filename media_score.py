import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime

# --- 튜닝 가능한 설정 값 ---
LABEL_SCORE_MAP = {
    0: 1.0, 1: 0.6, 2: -0.15, 3: -0.85
}
ALPHA = 0.7
MIN_ARTICLES_SOURCE = 3
MIN_ARTICLES_AUTHOR = 3

# --- 파일 경로 설정 ---
# 초기 가중치 파일 (스크립트와 같은 위치에 있다고 가정)
INITIAL_WEIGHTS_FILE = "source_initial_weights.csv" 
# 피드백 데이터를 저장하고 읽어올 폴더
FEEDBACK_FOLDER = "feedback_data"
# 업데이트된 가중치들을 저장할 폴더
WEIGHTS_FOLDER = "weights_history" 


class CredibilityModel:
    def __init__(self, initial_weights_csv_path):
        """모델을 초기화합니다."""
        
        weight_files = glob.glob(os.path.join(WEIGHTS_FOLDER, 'weights_*.csv'))
        
        load_path = initial_weights_csv_path
        if weight_files:
            latest_file = max(weight_files, key=os.path.getctime)
            if os.path.exists(latest_file):
                load_path = latest_file

        print(f"[INFO] 가중치 파일 로드: '{load_path}'")
        try:
            self.source_weights = pd.read_csv(load_path).set_index('source')
            if 'final_weight' not in self.source_weights.columns:
                self.source_weights['final_weight'] = self.source_weights['initial_weight']
        except FileNotFoundError:
            print(f"[ERROR] 가중치 파일을 찾을 수 없습니다: '{load_path}'")
            self.source_weights = pd.DataFrame()
            
        self.author_weights = pd.DataFrame()

    def update_with_feedback(self, feedback_data: pd.DataFrame):
        """새로운 라벨링 데이터를 피드백받아 기존 가중치를 업데이트합니다."""
        print("\n[INFO] 새로운 피드백 데이터로 가중치 업데이트를 시작합니다...")
        
        df_clean = feedback_data.dropna(subset=['label', 'author', 'source'])
        df_clean = df_clean[df_clean['label'].isin([0, 1, 2, 3])]
        df_clean['label'] = pd.to_numeric(df_clean['label'])
        df_clean['article_score'] = df_clean['label'].map(LABEL_SCORE_MAP)

        author_scores = df_clean.groupby('author').agg(
            article_count=('label', 'count'),
            avg_article_score=('article_score', 'mean')
        ).reset_index()
        author_scores['credibility_score'] = (author_scores['avg_article_score'] + 1) / 2
        self.author_weights = author_scores[author_scores['article_count'] >= MIN_ARTICLES_AUTHOR]
        print("[INFO] 기자 신뢰도 업데이트 완료.")

        df_current_weights = self.source_weights[['final_weight']].reset_index()
        df_current_weights.rename(columns={'final_weight': 'initial_weight'}, inplace=True)

        source_scores = df_clean.groupby('source').agg(
            article_count=('label', 'count'),
            avg_article_score=('article_score', 'mean')
        ).reset_index()
        source_scores['observed_score'] = (source_scores['avg_article_score'] + 1) / 2
        
        df_merged = df_current_weights.merge(source_scores, on='source', how='outer')
        
        etc_weight = df_current_weights[df_current_weights['source'] == 'etc']['initial_weight'].iloc[0] if 'etc' in df_current_weights['source'].values else 0.5
        df_merged['initial_weight'] = df_merged['initial_weight'].fillna(etc_weight)
        
        df_merged['article_count'] = df_merged['article_count'].fillna(0).astype(int)
        
        fill_values = df_merged.set_index('source')['initial_weight'].to_dict()
        df_merged['observed_score'] = df_merged['observed_score'].fillna(df_merged['source'].map(fill_values))
        
        df_merged['final_weight'] = np.where(
            df_merged['article_count'] >= MIN_ARTICLES_SOURCE,
            (df_merged['initial_weight'] * ALPHA) + (df_merged['observed_score'] * (1 - ALPHA)),
            df_merged['initial_weight']
        )
        
        self.source_weights = df_merged.set_index('source')
        print("[INFO] 언론사 신뢰도 업데이트 완료.")

    def get_rankings(self):
        """현재 모델의 언론사 및 기자 랭킹을 반환합니다."""
        cols_to_show = [col for col in ['initial_weight', 'final_weight', 'article_count'] if col in self.source_weights.columns]
        source_ranking = self.source_weights[cols_to_show].sort_values(by='final_weight', ascending=False)
        
        if not self.author_weights.empty:
            author_ranking = self.author_weights[['author', 'article_count', 'credibility_score']].sort_values(by='credibility_score', ascending=False)
        else:
            author_ranking = self.author_weights
        
        return source_ranking, author_ranking

    def save_weights_to_csv(self):
        """현재 가중치를 지정된 폴더에 버전 관리되는 CSV 파일로 저장합니다."""
        os.makedirs(WEIGHTS_FOLDER, exist_ok=True)
        
        today_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = os.path.join(WEIGHTS_FOLDER, f'weights_{today_str}.csv')
        
        df_to_save = self.source_weights.copy()
        
        cols_to_save = [col for col in ['initial_weight', 'final_weight', 'article_count'] if col in df_to_save.columns]
        df_to_save[cols_to_save].to_csv(output_filename, encoding='utf-8-sig')
        print(f"\n[SUCCESS] 업데이트된 가중치를 '{output_filename}' 파일로 저장했습니다.")



if __name__ == "__main__":
    model = CredibilityModel(INITIAL_WEIGHTS_FILE)

    print("\n--- Phase 1: 초기 로드된 언론사 가중치 ---")
    initial_source_ranking, _ = model.get_rankings()
    print(initial_source_ranking)
    
    # --- 여기가 핵심 수정 사항입니다 ---
    # 1. 피드백 폴더 안의 모든 CSV 파일을 찾습니다.
    feedback_files = glob.glob(os.path.join(FEEDBACK_FOLDER, '*.csv'))
    
    # 2. 피드백 파일이 존재하는 경우에만 업데이트를 진행합니다.
    if feedback_files:
        # 3. 가장 최신 파일을 찾습니다.
        latest_feedback_file = max(feedback_files, key=os.path.getctime)
        print(f"\n[INFO] 최신 피드백 데이터 파일 '{latest_feedback_file}'을 사용하여 업데이트를 시작합니다.")
        
        try:
            new_labeled_data = pd.read_csv(latest_feedback_file)
            model.update_with_feedback(new_labeled_data)

            print("\n--- Phase 2: 업데이트 후 최종 언론사 가중치 ---")
            final_source_ranking, final_author_ranking = model.get_rankings()
            
            print(final_source_ranking)
            print("\n\n--- 최종 기자별 가중치 ---")
            print(final_author_ranking)

            model.save_weights_to_csv()

        except FileNotFoundError:
            print(f"\n[ERROR] 피드백 데이터 파일 '{latest_feedback_file}'을 찾을 수 없습니다.")
        except Exception as e:
            print(f"피드백 데이터 처리 중 오류가 발생했습니다: {e}")
            
    else:
        # 4. 피드백 파일이 없으면 업데이트를 건너뜁니다.
        print(f"\n[INFO] 피드백 폴더 '{FEEDBACK_FOLDER}'에 학습할 데이터가 없어 업데이트를 건너뜁니다.")