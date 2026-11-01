"""Descarga y carga del dataset Parkinson's Telemonitoring (UCI).

Fuente: Tsanas, A. & Little, M. (2009). UCI Machine Learning Repository.
https://doi.org/10.24432/C5ZS3N  -  Licencia CC BY 4.0
"""
from pathlib import Path

import pandas as pd

URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "parkinsons/telemonitoring/parkinsons_updrs.data"
)

# Carpeta data/ en la raíz del proyecto (un nivel arriba de src/)
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CSV_PATH = DATA_DIR / "parkinsons_updrs.csv"


def download_data(force: bool = False) -> Path:
    """Descarga el dataset a data/ si todavía no existe."""
    DATA_DIR.mkdir(exist_ok=True)
    if CSV_PATH.exists() and not force:
        print(f"Ya existe: {CSV_PATH}")
        return CSV_PATH
    print("Descargando dataset desde UCI...")
    df = pd.read_csv(URL)
    df.to_csv(CSV_PATH, index=False)
    print(f"Guardado en: {CSV_PATH}  ({len(df)} filas)")
    return CSV_PATH


def load_data() -> pd.DataFrame:
    """Carga el dataset con nombres de columnas limpios y ordenado por paciente y tiempo."""
    if not CSV_PATH.exists():
        download_data()
    df = pd.read_csv(CSV_PATH)
    df = df.rename(columns={"subject#": "subject_id"})
    df = df.sort_values(["subject_id", "test_time"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    download_data()
