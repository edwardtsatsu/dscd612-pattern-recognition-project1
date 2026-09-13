"""Train/test split and standardization, fit on training data only."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


@dataclass
class ExperimentData:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler
    label_encoder: LabelEncoder
    class_names: list[str]


def prepare_experiment_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.3,
    random_state: int = 42,
) -> ExperimentData:
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X.to_numpy(),
        y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded,
    )

    scaler = StandardScaler().fit(X_train_raw)
    X_train = scaler.transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    return ExperimentData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        scaler=scaler,
        label_encoder=label_encoder,
        class_names=list(label_encoder.classes_),
    )
