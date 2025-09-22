# utils.py

def normalize(score: float) -> float:
    return max(0.0, min(1.0, score))

def ensemble(scores: list, weights: list = None) -> float:

    if weights is None:
        weights = [1/len(scores)] * len(scores)
    
    weighted_sum = sum(s * w for s, w in zip(scores, weights))
    return round(weighted_sum, 3)
