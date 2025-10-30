"""
Reusable preprocessing utilities for ECG heartbeat datasets.

This module centralizes common preprocessing steps derived from the
`notebooks/01_data_exploration.ipynb` analysis so new modeling notebooks
can import and reuse consistent logic.

Key capabilities:
- Load PTBDB (normal/abnormal) and MITBIH (train/test) CSV datasets
- Drop duplicates in PTBDB partitions
- Provide features/targets split with column 187 as target
- Compute class weights for imbalanced classification
- Provide stratified train/val split helpers
- Compute zero-padding start index per row (for diagnostics/feature eng)

Note: Signals are already normalized to [0, 1] per the source datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTETomek, SMOTEENN

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# ------------------------------
# Constants and label mappings
# ------------------------------

TARGET_COLUMN_INDEX: int = 187
FEATURE_COLUMN_RANGE: slice = slice(0, TARGET_COLUMN_INDEX)  # 0..186
RANDOM_STATE: int = 42

# MITBIH label mapping for reporting/plots (keep numeric labels for modeling)
MITBIH_LABELS_MAP: Dict[int, str] = {0: "N", 1: "S", 2: "V", 3: "F", 4: "Q"}
MITBIH_LABELS_TO_DESC: Dict[str, str] = {
    "N": "Normal",
    "S": "Supraventricular premature beat",
    "V": "Premature ventricular contraction",
    "F": "Fusion of V+N",
    "Q": "Unclassified",
}


# ------------------------------
# Sampling method registry and suffixes
# ------------------------------

# Internal method keys exposed by this module
_SAMPLING_REGISTRY = {
    "random_over": RandomOverSampler,
    "smote": SMOTE,
    "adasyn": ADASYN,
    "random_under": RandomUnderSampler,
    "tomek": TomekLinks,
    "smote_tomek": SMOTETomek,
    "smote_enn": SMOTEENN,
}

# File suffixes for saved processed datasets
SAMPLING_SUFFIXES: Dict[str, str] = {
    "none": "",  # no sampling
    "random_over": "_ro",
    "smote": "_sm",
    "adasyn": "_ad",
    "random_under": "_ru",
    "tomek": "_tk",
    "smote_tomek": "_st",
    "smote_enn": "_se",
}

# External names normalization (accepts names used elsewhere in the project)
_EXTERNAL_TO_INTERNAL: Dict[str, str] = {
    "no_sampling": "none",
    "none": "none",
    "": "none",
    "randomoversampler": "random_over",
    "random_over": "random_over",
    "smote": "smote",
    "adasyn": "adasyn",
    "randomundersampler": "random_under",
    "random_under": "random_under",
    "tomek": "tomek",
    "smotetomek": "smote_tomek",
    "smoteenn": "smote_enn",
}

OUTLIER_REMOVAL_SUFFIX = "_olr"


@dataclass
class DatasetSplit:
    X_train: pd.DataFrame
    X_val: Optional[pd.DataFrame]  # Renamed from X_test to X_val
    y_train: pd.Series
    y_val: Optional[pd.Series]  # Renamed from y_test to y_val
    class_weight: Optional[Dict[int, float]]

    # Note: Outlier removal (if enabled) is applied after splitting. Class
    # weights are computed on the final training labels so your training loop
    # can pass them directly to supported estimators.


def _load_csv(path: Union[str, Path]) -> pd.DataFrame:
    """Read a CSV without header where each row is a 1D signal of length 188.

    Column 187 is the target label.
    """
    return pd.read_csv(str(path), header=None)


def load_ptbdb(
    data_dir: Union[str, Path] = "../data/original",
    drop_duplicates: bool = True,
) -> pd.DataFrame:
    """Load and combine PTBDB normal/abnormal datasets into a single DataFrame.

    Returns a single DataFrame with features in columns 0..186 and target
    in 187.
    """
    data_dir = Path(data_dir)
    normal = _load_csv(data_dir / "ptbdb_normal.csv")
    abnormal = _load_csv(data_dir / "ptbdb_abnormal.csv")

    if drop_duplicates:
        # Duplicates were found in exploration; remove them
        normal = normal.drop_duplicates()
        abnormal = abnormal.drop_duplicates()

    df = pd.concat([abnormal, normal], axis=0, ignore_index=True)
    return df


def load_mitbih(
    data_dir: Union[str, Path] = "../data/original",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load MITBIH train and test DataFrames (kept as provided)."""
    data_dir = Path(data_dir)
    train = _load_csv(data_dir / "mitbih_train.csv")
    test = _load_csv(data_dir / "mitbih_test.csv")
    return train, test


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) where y is column 187 and X are columns 0..186."""
    X = df.iloc[:, FEATURE_COLUMN_RANGE]
    y = df.iloc[:, TARGET_COLUMN_INDEX].astype(int)
    return X, y


def compute_balanced_class_weight(y: Union[pd.Series, np.ndarray]) -> Dict[int, float]:
    """Compute class weights to counter class imbalance.
    Useful for many models."""
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {int(cls): float(w) for cls, w in zip(classes, weights)}


def prepare_mitbih(
    data_dir: Union[str, Path] = "../data/original",
    random_state: int = 42,
    remove_outliers: bool = False,
    whisker_k: float = 1.5,
    feature_reduction: bool = False,
) -> DatasetSplit:
    """Load MITBIH train/test and class weights.

    The original test set is kept for final evaluation. Train/validation splitting
    is handled by cross-validation methods.
    """
    train_df, test_df = load_mitbih(data_dir=data_dir)

    if feature_reduction:
        raise NotImplementedError()

    X_train, y_train = split_features_target(train_df)
    X_test, y_test = split_features_target(test_df)

    if remove_outliers:
        # Reassemble dfs to compute zero_pad and apply bounds
        train_df = pd.concat([X_train, y_train.rename("target")], axis=1)
        test_df = pd.concat([X_test, y_test.rename("target")], axis=1)

        zp_train = compute_zero_padding_feature(train_df)
        bounds = fit_zero_pad_whisker_bounds(train_df, zp_train, whisker_k=whisker_k)

        train_df = drop_zero_pad_extreme_val_with_bounds(train_df, bounds, zp_train)
        # won't remove outliers from test set to avoid data leakage

        # Split back to X/y
        X_train, y_train = split_features_target(train_df)
        X_test, y_test = split_features_target(test_df)

    weight_map = compute_balanced_class_weight(y_train)

    return DatasetSplit(
        X_train=X_train,
        X_val=X_test,  # Renamed from X_test to X_val
        y_train=y_train,
        y_val=y_test,  # Renamed from y_test to y_val
        class_weight=weight_map,
    )


def prepare_ptbdb(
    data_dir: Union[str, Path] = "../data/original",
    test_size: float = 0.2,
    random_state: int = 42,
    remove_outliers: bool = False,
    whisker_k: float = 1.5,
) -> DatasetSplit:
    """Load PTBDB and produce stratified train/test splits and class weights.

    Train/validation splitting is handled by cross-validation methods.
    """
    df = load_ptbdb(data_dir=data_dir, drop_duplicates=True)
    X, y = split_features_target(df)

    # Split train vs test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if remove_outliers:
        train_df = pd.concat([X_train, y_train.rename("target")], axis=1)
        test_df = pd.concat([X_test, y_test.rename("target")], axis=1)

        zp_train = compute_zero_padding_feature(train_df)
        bounds = fit_zero_pad_whisker_bounds(train_df, zp_train, whisker_k=whisker_k)

        train_df = drop_zero_pad_extreme_val_with_bounds(train_df, bounds, zp_train)
        # won't remove outliers from test set to avoid data leakage

        X_train, y_train = split_features_target(train_df)
        X_test, y_test = split_features_target(test_df)

    weight_map = compute_balanced_class_weight(y_train)

    return DatasetSplit(
        X_train=X_train,
        X_val=X_test,  # Renamed from X_test to X_val
        y_train=y_train,
        y_val=y_test,  # Renamed from y_test to y_val
        class_weight=weight_map,
    )


def find_zero_padding_start(sequence_row: Union[pd.Series, np.ndarray]) -> int:
    """Return the first index after the last non-zero value
    scanning from the end.

    This matches the exploratory notebook's logic to estimate the beginning of
    right-side zero-padding per row.
    """
    if isinstance(sequence_row, pd.Series):
        values = sequence_row.values
    else:
        values = sequence_row

    first_zero_index = 0
    for i in range(len(values) - 1, -1, -1):
        if values[i] != 0:
            first_zero_index = (i + 1) / 1.2  # pre-defined from dataset
            break
    return int(first_zero_index)


def compute_zero_padding_feature(df: pd.DataFrame) -> pd.Series:
    """Compute `zero_pad_start` for each row based on
    feature columns 0..186."""
    X = df.iloc[:, FEATURE_COLUMN_RANGE]
    return X.apply(lambda row: find_zero_padding_start(row), axis=1)


def compute_zero_pad_outlier_flag(
    df: pd.DataFrame,
    zero_pad_start: Optional[pd.Series] = None,
    whisker_k: float = 1.5,
    as_int: bool = True,
) -> pd.Series:
    """Flag rows whose `zero_pad_start` is outside class-wise Tukey whiskers.

    The whiskers are computed per target class over the `zero_pad_start` values
    (25th/75th percentiles and IQR).
    Rows with values < lower whisker or > upper whisker are
    flagged as outliers for their class.

    Args:
        df: DataFrame with features in columns 0..186 and target in 187.
        zero_pad_start: Optional precomputed Series aligned to df.index. If not
            provided, it will be computed from feature columns using
            `find_zero_padding_start`.
        whisker_k: Multiplier for IQR to define whiskers (default 1.5).
        as_int: If True, return 1/0; otherwise return boolean Series.

    Returns:
        A pandas Series indexed like df with outlier flags per row.
    """
    # Compute or validate zero_pad_start
    if zero_pad_start is None:
        zero_pad_start = compute_zero_padding_feature(df)
    else:
        if not zero_pad_start.index.equals(df.index):
            # Align to df index if needed
            zero_pad_start = zero_pad_start.reindex(df.index)

    target = df.iloc[:, TARGET_COLUMN_INDEX].astype(int)

    temp = pd.DataFrame(
        {
            "zero_pad_start": zero_pad_start,
            "target": target.values,
        }
    )

    # Compute class-wise whisker bounds via IQR
    quantiles = (
        temp.groupby("target")["zero_pad_start"].quantile([0.25, 0.75]).unstack()
    )
    quantiles = quantiles.rename(columns={0.25: "q1", 0.75: "q3"})
    quantiles["iqr"] = quantiles["q3"] - quantiles["q1"]
    quantiles["lower"] = quantiles["q1"] - whisker_k * quantiles["iqr"]
    quantiles["upper"] = quantiles["q3"] + whisker_k * quantiles["iqr"]

    bounds = quantiles[["lower", "upper"]]

    # Join bounds per row and compute flag
    joined = temp.join(bounds, on="target")
    flag = (joined["zero_pad_start"] < joined["lower"]) | (
        joined["zero_pad_start"] > joined["upper"]
    )

    return flag.astype(int) if as_int else flag


def fit_zero_pad_whisker_bounds(
    df: pd.DataFrame,
    zero_pad_start: Optional[pd.Series] = None,
    whisker_k: float = 1.5,
) -> pd.DataFrame:
    """Fit per-class Tukey whisker bounds for `zero_pad_start` on the given df.

    Returns a DataFrame indexed by class with columns `lower` and `upper`.
    """
    if zero_pad_start is None:
        zero_pad_start = compute_zero_padding_feature(df)
    target = df.iloc[:, TARGET_COLUMN_INDEX].astype(int)

    temp = pd.DataFrame(
        {
            "zero_pad_start": zero_pad_start,
            "target": target.values,
        }
    )

    quantiles = (
        temp.groupby("target")["zero_pad_start"].quantile([0.25, 0.75]).unstack()
    )
    quantiles = quantiles.rename(columns={0.25: "q1", 0.75: "q3"})
    quantiles["iqr"] = quantiles["q3"] - quantiles["q1"]
    quantiles["lower"] = quantiles["q1"] - whisker_k * quantiles["iqr"]
    quantiles["upper"] = quantiles["q3"] + whisker_k * quantiles["iqr"]
    return quantiles[["lower", "upper"]]


def drop_zero_pad_extreme_val_with_bounds(
    df: pd.DataFrame,
    bounds: pd.DataFrame,
    zero_pad_start: Optional[pd.Series] = None,
    target_class: int = 0,
) -> pd.DataFrame:
    """Drop rows whose `zero_pad_start` is outside class-specific extreme
    values.

    By default, only applies outlier removal to the target class (class 0 - normal).
    All other classes (abnormal heartbeats) are protected and kept unchanged.
    Rows with unseen classes (not present in `bounds`) are kept unchanged.

    Args:
        df: DataFrame with features and target
        bounds: Per-class bounds DataFrame with 'lower' and 'upper' columns
        zero_pad_start: Optional precomputed zero padding start values
        target_class: Class to apply outlier removal to (default 0). All other
                     classes will be preserved without outlier removal.
    """
    if zero_pad_start is None:
        zero_pad_start = compute_zero_padding_feature(df)
    target = df.iloc[:, TARGET_COLUMN_INDEX].astype(int)

    temp = pd.DataFrame(
        {"zero_pad_start": zero_pad_start, "target": target.values}, index=df.index
    )
    temp = temp.join(bounds, on="target", how="left")

    # Only apply outlier removal to the target class (default 0 - normal)
    # All other classes (abnormal heartbeats) are protected and kept
    keep_mask = (
        (temp["target"] != target_class)
        | temp["lower"].isna()
        | (
            (temp["zero_pad_start"] >= temp["lower"])
            & (temp["zero_pad_start"] <= temp["upper"])
        )
    )
    return df.loc[keep_mask]


def resample_training(
    split: DatasetSplit,
    method: str,
    **kwargs,
) -> DatasetSplit:
    """Apply an imbalanced-learn resampler to the training split only.

    Args:
        split: Existing dataset split returned by
        `prepare_mitbih`/`prepare_ptbdb`.
        method: One of the supported method keys:
            - "random_over": RandomOverSampler
            - "smote": SMOTE
            - "adasyn": ADASYN
            - "random_under": RandomUnderSampler
            - "tomek": TomekLinks
            - "smote_tomek": SMOTETomek
            - "smote_enn": SMOTEENN
        **kwargs: Parameters forwarded to the sampler constructor (e.g.,
            random_state, k_neighbors, sampling_strategy, etc.).

    Returns:
        A new `DatasetSplit` with resampled `X_train`/`y_train` and updated
        `class_weight`. Validation and test splits are untouched to prevent
        data leakage.
    """
    # Lazy imports to avoid imposing a hard dependency unless used
    registry = _SAMPLING_REGISTRY

    if method not in registry:
        available = ", ".join(sorted(registry.keys()))
        raise ValueError(
            f"Unknown resampling method '{method}'. \
                         Available: {available}"
        )

    SamplerClass = registry[method]
    sampler = SamplerClass(**kwargs)

    X_train = split.X_train
    y_train = split.y_train

    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)

    # Ensure pandas types/column names are preserved when possible
    if not isinstance(X_resampled, pd.DataFrame):
        X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
    if not isinstance(y_resampled, (pd.Series, pd.DataFrame)):
        y_resampled = pd.Series(
            y_resampled, name=y_train.name if y_train is not None else "target"
        )
    if isinstance(y_resampled, pd.DataFrame):
        # Some samplers may return a DataFrame; ensure it's a 1D Series
        y_resampled = y_resampled.iloc[:, 0]
    y_resampled = y_resampled.astype(int)

    new_weights = compute_balanced_class_weight(y_resampled)

    return DatasetSplit(
        X_train=X_resampled,
        X_val=split.X_val,
        y_train=y_resampled,
        y_val=split.y_val,
        class_weight=new_weights,
    )


def save_processed_dataset(
    split: DatasetSplit,
    data_dir: Union[str, Path] = "../data/processed/mitbih",
    sampling_suffix: Optional[str] = None,
) -> None:
    """Save dataset split to processed directory with optional training suffix.

    - Training files are saved with the provided suffix (e.g. `_sm`, `_ro`).
    - Validation files are saved once without any suffix since they are identical
      across sampling/outlier-removal variants. Existing validation files are
      not overwritten.

    Args:
        split: DatasetSplit object to save
        data_dir: Directory to save the datasets
        sampling_suffix: Optional suffix for sampling method (e.g., '_sm' for SMOTE, '_ro' for RandomOverSampler)
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    suffix = sampling_suffix if sampling_suffix else ""

    # Save training data (with suffix)
    split.X_train.to_csv(data_dir / f"X_train{suffix}.csv", index=False)
    split.y_train.to_csv(data_dir / f"y_train{suffix}.csv", index=False)

    # Save validation data (without suffix) only once if available
    if split.X_val is not None:
        x_val_path = data_dir / "X_val.csv"
        if not x_val_path.exists():
            split.X_val.to_csv(x_val_path, index=False)
    if split.y_val is not None:
        y_val_path = data_dir / "y_val.csv"
        if not y_val_path.exists():
            split.y_val.to_csv(y_val_path, index=False)


def load_processed_dataset(
    data_dir: Union[str, Path] = "../data/processed/mitbih",
    sampling_suffix: Optional[str] = None,
) -> DatasetSplit:
    """Load dataset split from processed directory with optional training suffix.

    - Training files are loaded with the provided suffix (e.g. `_sm`, `_ro`).
    - Validation files are loaded without suffix since they are shared across
      variants.

    Args:
        data_dir: Directory to load the datasets from
        sampling_suffix: Optional suffix for sampling method (e.g., '_sm' for SMOTE, '_ro' for RandomOverSampler)

    Returns:
        DatasetSplit object with loaded data
    """
    data_dir = Path(data_dir)
    suffix = sampling_suffix if sampling_suffix else ""

    # Load training data
    x_train_dataset_path = data_dir / f"X_train{suffix}.csv"
    print(f"Loading processed X_train dataset from: {x_train_dataset_path}")
    X_train = pd.read_csv(x_train_dataset_path)

    y_train_dataset_path = data_dir / f"y_train{suffix}.csv"
    print(f"Loading processed y_train dataset from: {y_train_dataset_path}")
    y_train = pd.read_csv(y_train_dataset_path).iloc[:, 0]  # First column

    # Load validation data without suffix (shared across variants)
    X_val_path = data_dir / "X_val.csv"
    y_val_path = data_dir / "y_val.csv"

    X_val = None
    y_val = None

    if X_val_path.exists():
        X_val = pd.read_csv(X_val_path)
    else:
        raise ValueError(f"Validation features file not found: {X_val_path}")
    if y_val_path.exists():
        y_val = pd.read_csv(y_val_path).iloc[:, 0]  # First column#
    else:
        raise ValueError(f"Validation labels file not found: {y_val_path}")

    # Compute class weights
    weight_map = compute_balanced_class_weight(y_train)

    return DatasetSplit(
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        class_weight=weight_map,
    )


# ------------------------------
# Helpers for processed dataset generation and reuse
# ------------------------------


def _normalize_sampling_method_name(method: Optional[str]) -> str:
    """Normalize external sampling method names to internal keys.

    Accepts names like "No_Sampling", "SMOTE", "RandomOverSampler" and maps
    them to internal keys like "none", "smote", "random_over".
    """
    if method is None:
        return "none"
    key = str(method).strip().lower().replace(" ", "")
    return _EXTERNAL_TO_INTERNAL.get(key, key)


def sampling_suffix(method: Optional[str]) -> str:
    """Return file suffix for a sampling method (internal or external name)."""
    internal = _normalize_sampling_method_name(method)
    return SAMPLING_SUFFIXES.get(internal, "")


def build_full_suffix(method: Optional[str], remove_outliers: bool) -> str:
    """Compose sampling + outlier suffix for filenames."""
    return (
        f"{sampling_suffix(method)}{OUTLIER_REMOVAL_SUFFIX if remove_outliers else ''}"
    )


def _dataset_with_suffix_exists(data_dir: Union[str, Path], suffix: str) -> bool:
    data_dir = Path(data_dir)
    expected = [
        data_dir / f"X_train{suffix}.csv",
        data_dir / f"y_train{suffix}.csv",
    ]
    return all(p.exists() for p in expected)


def _ensure_base_split_exists(data_dir: Union[str, Path]) -> DatasetSplit:
    """Ensure the base no-sampling/no-outlier processed split exists, create if not."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if _dataset_with_suffix_exists(data_dir, ""):
        return load_processed_dataset(data_dir=data_dir, sampling_suffix="")

    # Create from original and save as base
    base_split = prepare_mitbih(
        data_dir=str(Path(data_dir).parents[1] / "original"), remove_outliers=False
    )
    save_processed_dataset(base_split, data_dir=data_dir, sampling_suffix="")
    return base_split


def apply_outlier_removal_to_split(
    split: DatasetSplit, whisker_k: float = 1.5
) -> DatasetSplit:
    """Apply zero-pad-based outlier removal to training only and return new split."""
    train_df = pd.concat([split.X_train, split.y_train.rename("target")], axis=1)
    zp_train = compute_zero_padding_feature(train_df)
    bounds = fit_zero_pad_whisker_bounds(train_df, zp_train, whisker_k=whisker_k)
    filtered_train_df = drop_zero_pad_extreme_val_with_bounds(
        train_df, bounds, zp_train
    )
    X_train, y_train = split_features_target(filtered_train_df)
    return DatasetSplit(
        X_train=X_train,
        X_val=split.X_val,
        y_train=y_train,
        y_val=split.y_val,
        class_weight=compute_balanced_class_weight(y_train),
    )


def resample_split(split: DatasetSplit, method: str, **kwargs) -> DatasetSplit:
    """Resample training portion of the split using a registered method key."""
    internal = _normalize_sampling_method_name(method)
    if internal == "none":
        return split
    return resample_training(split, method=internal, **kwargs)


def generate_all_processed_datasets(
    data_dir: Union[str, Path] = "../data/processed/mitbih",
    only_once: bool = True,
) -> None:
    """Generate and save all combinations of processed datasets.

    - Base split comes from the existing processed split if present, otherwise
      it's created from original data and saved without suffix.
    - For each sampling method (including none) and with/without outlier
      removal, create derived datasets if they do not already exist.
    - Never overwrite existing CSVs.

    Execution is gated by a marker file when only_once=True.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    marker = data_dir / ".generation_complete"
    if only_once and marker.exists():
        return

    # Ensure base split exists first
    base_split = _ensure_base_split_exists(data_dir)

    # All methods including none
    all_methods = ["none"] + sorted(_SAMPLING_REGISTRY.keys())

    for remove_outliers in (False, True):
        if remove_outliers:
            split_no_sample = apply_outlier_removal_to_split(base_split)
        else:
            split_no_sample = base_split

        for method in all_methods:
            suffix = build_full_suffix(method, remove_outliers)
            if _dataset_with_suffix_exists(data_dir, suffix):
                continue

            if method == "none":
                out_split = split_no_sample
            else:
                out_split = resample_split(split_no_sample, method)

            save_processed_dataset(out_split, data_dir=data_dir, sampling_suffix=suffix)

    if only_once:
        marker.touch()
