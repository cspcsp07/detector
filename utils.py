# utils.py

def normalize(score: float) -> float:
    """
    점수를 0~1 범위로 보정 (여기서는 그대로 반환)
    """
    return max(0.0, min(1.0, score))

def ensemble(scores: list, weights: list = None) -> float:
    """
    앙상블: 단순 가중 평균
    scores: [method1, method2, method3]
    weights: 각 방법의 가중치 (기본값: 동일 가중치)
    """
    if weights is None:
        weights = [1/len(scores)] * len(scores)
    
    weighted_sum = sum(s * w for s, w in zip(scores, weights))
    return round(weighted_sum, 3)
