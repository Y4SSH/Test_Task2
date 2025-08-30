import pandas as pd
import numpy as np

def score_distributions(original: pd.DataFrame, synthetic: pd.DataFrame) -> dict:
    s = {}

    def safe_std(x):
        return float(np.std(x, ddof=0)) if len(x) else 0.0

    # numeric moments
    for col in ["price", "quantity", "total_revenue"]:
        if col in original.columns and col in synthetic.columns:
            o_mean, s_mean = float(np.mean(original[col])), float(np.mean(synthetic[col]))
            o_std, s_std = safe_std(original[col]), safe_std(synthetic[col])
            s[f"{col}_mean_absdiff"] = abs(o_mean - s_mean)
            s[f"{col}_std_absdiff"] = abs(o_std - s_std)

    # categorical overlap
    if "category" in original.columns and "category" in synthetic.columns:
        o_dist = original["category"].value_counts(normalize=True)
        s_dist = synthetic["category"].value_counts(normalize=True)
        cats = set(o_dist.index) | set(s_dist.index)
        overlap = sum(min(o_dist.get(c, 0.0), s_dist.get(c, 0.0)) for c in cats)
        s["category_overlap"] = float(overlap)

    # rejection proxy (if synthetic got filtered in env)
    s["size_ratio"] = float(len(synthetic) / max(1, len(original)))

    # aggregate score (simple heuristic)
    agg = 0.0
    agg += (1.0 - min(1.0, s.get("price_mean_absdiff", 1.0) / (original["price"].mean() + 1e-6))) * 0.25
    agg += (1.0 - min(1.0, s.get("quantity_mean_absdiff", 1.0) / (original["quantity"].mean() + 1e-6))) * 0.2
    agg += s.get("category_overlap", 0.0) * 0.25
    agg += min(1.0, s.get("size_ratio", 0.0)) * 0.1
    s["aggregate_score"] = float(max(0.0, min(1.0, agg)))
    return s
