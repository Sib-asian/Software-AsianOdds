"""
Data Preparation Utilities
===========================

Funzioni per preparare dati storici per training dei modelli AI.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import sqlite3
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def load_historical_data(
    db_path: str = "storico_analisi.csv",
    min_samples: int = 1000
) -> pd.DataFrame:
    """
    Carica dati storici da database/CSV.

    Args:
        db_path: Path al database o CSV
        min_samples: Minimo numero di sample richiesti

    Returns:
        DataFrame con dati storici
    """
    try:
        # Try CSV first
        if Path(db_path).suffix == ".csv":
            df = pd.read_csv(db_path)
            logger.info(f"✅ Loaded {len(df)} samples from CSV")

        # Try SQLite
        else:
            conn = sqlite3.connect(db_path)
            df = pd.read_sql_query(
                "SELECT * FROM predictions ORDER BY data DESC",
                conn
            )
            conn.close()
            logger.info(f"✅ Loaded {len(df)} samples from SQLite")

        if len(df) < min_samples:
            logger.warning(
                f"⚠️  Only {len(df)} samples available "
                f"(minimum: {min_samples})"
            )

        return df

    except Exception as e:
        logger.error(f"❌ Error loading historical data: {e}")
        return pd.DataFrame()


def prepare_training_data(
    df: pd.DataFrame,
    target_column: str = "outcome"
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepara dati per training.

    Args:
        df: DataFrame raw
        target_column: Nome colonna target

    Returns:
        (features_df, target_series)
    """
    # Remove rows with missing target
    df = df.dropna(subset=[target_column])

    # Split features and target
    target = df[target_column]
    features = df.drop(columns=[target_column])

    logger.info(f"✅ Prepared {len(df)} samples for training")
    logger.info(f"   Features: {len(features.columns)}")
    logger.info(f"   Target distribution: {target.value_counts().to_dict()}")

    return features, target


def create_synthetic_training_data(
    n_samples: int = 2000,
    bias: float = 0.10
) -> pd.DataFrame:
    """
    Crea dati sintetici per testing/development.

    Args:
        n_samples: Numero di sample
        bias: Bias del modello (es. 0.10 = ottimista +10%)

    Returns:
        DataFrame con dati sintetici
    """
    logger.info(f"📝 Creating {n_samples} synthetic samples (bias={bias:.1%})...")

    np.random.seed(42)
    data = []

    for i in range(n_samples):
        # True underlying probability
        true_prob = np.random.beta(2, 2)

        # Biased model probability
        prob_raw = min(true_prob * (1 + bias), 0.99)

        # Simulate outcome
        outcome = 1 if np.random.random() < true_prob else 0

        # Context
        context = {
            "league": np.random.choice(
                ["Serie A", "Serie B", "Premier League", "La Liga"],
                p=[0.4, 0.2, 0.25, 0.15]
            ),
            "market": "1x2",
            "data_quality": np.random.uniform(0.5, 1.0),
            "api_context": {
                "form_home": np.random.choice(["WWWWW", "WWWDL", "WDDLL", "LLLLL"]),
                "form_away": np.random.choice(["WWWWW", "WWWDL", "WDDLL", "LLLLL"]),
                "xg_home_last5": np.random.uniform(5, 12),
                "xg_away_last5": np.random.uniform(4, 10),
                "xga_home_last5": np.random.uniform(3, 8),
                "xga_away_last5": np.random.uniform(5, 10),
                "lineup_quality_home": np.random.uniform(0.7, 1.0),
                "lineup_quality_away": np.random.uniform(0.7, 1.0),
                "injuries_home": [],
                "injuries_away": []
            }
        }

        data.append({
            "prob_raw": prob_raw,
            "outcome": outcome,
            "context": context,
            "odds": 1.0 / prob_raw if prob_raw > 0 else 2.0
        })

    df = pd.DataFrame(data)
    logger.info(f"✅ Created synthetic dataset: {len(df)} samples")
    logger.info(f"   Actual win rate: {df['outcome'].mean():.1%}")
    logger.info(f"   Predicted avg: {df['prob_raw'].mean():.1%}")

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test
    df = create_synthetic_training_data(1000)
    print(df.head())
