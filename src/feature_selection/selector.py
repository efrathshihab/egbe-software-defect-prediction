"""
Feature selection utilities for empirical comparison between full 65-metric space
and subset representations.
"""

from typing import List, Optional
import pandas as pd
from sklearn.feature_selection import mutual_info_classif


def select_metrics_subset(
    X: pd.DataFrame,
    subset_type: str = "full",
    k: Optional[int] = None,
    y: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Selects metric subset according to experimental configuration.
    """
    churn_metrics = [
        "Added_lines", "Del_lines", "Commits", "Churn", "COMM", "ADEV", 
        "DDEV", "EXP", "OWN", "MINOR", "HeuBug", "HeuBugFix"
    ]
    ownership_metrics = ["OWN", "MINOR", "ADEV", "DDEV", "EXP"]

    if subset_type == "full":
        return X.copy()
    
    elif subset_type == "churn":
        selected = [c for c in X.columns if any(ch.lower() in c.lower() for ch in churn_metrics)]
        return X[selected] if selected else X.copy()
        
    elif subset_type == "ownership":
        selected = [c for c in X.columns if c in ownership_metrics]
        return X[selected] if selected else X.copy()
        
    elif subset_type == "static":
        excluded = [c for c in X.columns if any(ch.lower() in c.lower() for ch in churn_metrics)]
        return X.drop(columns=excluded)
        
    elif subset_type == "mutual_info":
        if y is None or k is None:
            raise ValueError("y and k must be specified for mutual_info subset_type.")
        mi = mutual_info_classif(X, y, random_state=42)
        top_indices = mi.argsort()[-k:][::-1]
        selected_cols = X.columns[top_indices]
        return X[selected_cols]
        
    else:
        raise ValueError(f"Unknown subset_type: {subset_type}")
