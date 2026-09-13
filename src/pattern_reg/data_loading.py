"""Download and load the UCI Dry Bean Dataset."""
from __future__ import annotations

import io
import os
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

DRY_BEAN_URL = "https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip"
DEFAULT_RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
FEATURE_COLUMNS = [
    "Area",
    "Perimeter",
    "MajorAxisLength",
    "MinorAxisLength",
    "AspectRation",
    "Eccentricity",
    "ConvexArea",
    "EquivDiameter",
    "Extent",
    "Solidity",
    "roundness",
    "Compactness",
    "ShapeFactor1",
    "ShapeFactor2",
    "ShapeFactor3",
    "ShapeFactor4",
]


def download_dry_bean_dataset(dest_dir: Path = DEFAULT_RAW_DIR) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    xlsx_path = dest_dir / "Dry_Bean_Dataset.xlsx"
    if xlsx_path.exists():
        return xlsx_path

    with urllib.request.urlopen(DRY_BEAN_URL, timeout=60) as response:
        zip_bytes = response.read()

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        member = next(
            (name for name in archive.namelist() if name.lower().endswith(".xlsx")),
            None,
        )
        if member is None:
            members = archive.namelist()
            raise RuntimeError(
                f"No .xlsx file found in archive at {DRY_BEAN_URL}. "
                f"Archive contains: {members}"
            )

        tmp_path = xlsx_path.with_suffix(".xlsx.tmp")
        with archive.open(member) as source, open(tmp_path, "wb") as target:
            target.write(source.read())

        os.replace(tmp_path, xlsx_path)

    return xlsx_path


def load_dry_bean_dataset(
    raw_dir: Path = DEFAULT_RAW_DIR,
) -> tuple[pd.DataFrame, pd.Series]:
    xlsx_path = download_dry_bean_dataset(raw_dir)
    df = pd.read_excel(xlsx_path)
    df = df.dropna()
    X = df[FEATURE_COLUMNS].astype(float)
    y = df["Class"].astype(str)
    return X, y
