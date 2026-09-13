import pandas as pd
from pattern_reg.data_loading import load_dry_bean_dataset


def test_load_dry_bean_dataset_shape_and_classes():
    X, y = load_dry_bean_dataset()
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.shape[0] == y.shape[0]
    assert X.shape[0] > 10000
    assert X.shape[1] == 16
    assert set(y.unique()) == {
        "SEKER",
        "BARBUNYA",
        "BOMBAY",
        "CALI",
        "DERMASON",
        "HOROZ",
        "SIRA",
    }
    assert X.isnull().sum().sum() == 0
