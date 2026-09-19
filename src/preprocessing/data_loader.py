"""
Data loader and dataset utility functions for Apache JIRA defect prediction.
Handles loading 65-metric CSV files, identifying static, churn, and ownership features.
"""

import os
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np


def get_feature_groups() -> Dict[str, List[str]]:
    """Returns classification of metrics into static code, churn, and ownership dimensions."""
    churn_metrics = [
        "Added_lines", "Del_lines", "Commits", "Churn", "COMM", "ADEV", 
        "DDEV", "EXP", "OWN", "MINOR", "HeuBug", "HeuBugFix"
    ]
    ownership_metrics = ["OWN", "MINOR", "ADEV", "DDEV", "EXP"]
    return {
        "churn": churn_metrics,
        "ownership": ownership_metrics,
        "all_65": []
    }


def load_project_dataset(
    file_path: str,
    target_column: str = "RealBug",
    metadata_columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Loads an Apache project release dataset CSV.
    """
    if metadata_columns is None:
        metadata_columns = ["File", "file", "Release", "release", "Project", "project"]

    df = pd.read_csv(file_path)
    if target_column not in df.columns:
        for alt_col in ["RealBug", "realbug", "bug", "Bug", "Defective", "defective", "is_defective"]:
            if alt_col in df.columns:
                target_column = alt_col
                break

    y = (df[target_column] > 0).astype(int)
    cols_to_drop = [target_column] + [c for c in metadata_columns if c in df.columns]
    X = df.drop(columns=cols_to_drop).select_dtypes(include=[np.number])
    feature_names = list(X.columns)

    return X, y, feature_names


def split_chronological(
    releases_data: Dict[str, pd.DataFrame],
    target_column: str = "RealBug"
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Constructs leak-free chronological train-test split across release versions.
    """
    sorted_releases = sorted(releases_data.keys())
    train_dfs = [releases_data[r] for r in sorted_releases[:-1]]
    test_df = releases_data[sorted_releases[-1]]
    
    train_full = pd.concat(train_dfs, ignore_index=True)
    
    X_train, y_train, _ = load_project_dataset_from_df(train_full, target_column)
    X_test, y_test, _ = load_project_dataset_from_df(test_df, target_column)
    
    return X_train, y_train, X_test, y_test


def load_project_dataset_from_df(
    df: pd.DataFrame,
    target_column: str = "RealBug"
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Helper to extract features and labels from a DataFrame."""
    metadata_columns = ["File", "file", "Release", "release", "Project", "project"]
    y = (df[target_column] > 0).astype(int)
    cols_to_drop = [target_column] + [c for c in metadata_columns if c in df.columns]
    X = df.drop(columns=cols_to_drop).select_dtypes(include=[np.number])
    return X, y, list(X.columns)
