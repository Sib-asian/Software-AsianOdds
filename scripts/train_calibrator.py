#!/usr/bin/env python3
"""
Utility per allenare e salvare il Probability Calibrator.

Supporta due modalità:
1. Caricamento da dataset storico (CSV/SQLite) se disponibile
2. Generazione di dati sintetici (default) per avere un modello funzionante
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_system.config import AIConfig  # pylint: disable=wrong-import-position
from ai_system.blocco_1_calibrator import ProbabilityCalibrator  # pylint: disable=wrong-import-position
from ai_system.utils.data_preparation import (  # pylint: disable=wrong-import-position
    load_historical_data,
    create_synthetic_training_data,
)


def load_dataset(path: Optional[str], min_samples: int, synthetic_samples: int, bias: float) -> pd.DataFrame:
    if path:
        dataset_path = Path(path)
        if dataset_path.exists():
            df = load_historical_data(str(dataset_path), min_samples=min_samples)
            if not df.empty:
                return df
            print(f"⚠️  Dataset '{path}' vuoto o insufficiente, passo ai dati sintetici.")
        else:
            print(f"⚠️  Dataset '{path}' non trovato, passo ai dati sintetici.")

    return create_synthetic_training_data(n_samples=synthetic_samples, bias=bias)


def main():
    parser = argparse.ArgumentParser(description="Train Probability Calibrator")
    parser.add_argument("--data-path", help="Path a CSV/SQLite con storico (opzionale)")
    parser.add_argument("--min-samples", type=int, default=1000, help="Minimo campioni richiesti per usare lo storico")
    parser.add_argument("--synthetic-samples", type=int, default=4000, help="Numero di campioni sintetici se manca lo storico")
    parser.add_argument("--synthetic-bias", type=float, default=0.12, help="Bias del modello sintetico (0.12 = +12%)")
    parser.add_argument("--validation-split", type=float, default=0.2, help="Frazione validation set")
    parser.add_argument("--output", default=None, help="Path custom per salvare il modello")
    parser.add_argument("--log-level", default="INFO", help="Log level (INFO/DEBUG)")
    args = parser.parse_args()

    load_dotenv()

    df = load_dataset(
        path=args.data_path,
        min_samples=args.min_samples,
        synthetic_samples=args.synthetic_samples,
        bias=args.synthetic_bias,
    )
    if df.empty:
        raise SystemExit("❌ Impossibile creare un dataset di training.")

    config = AIConfig(log_level=args.log_level)
    calibrator = ProbabilityCalibrator(config=config)

    metrics = calibrator.train(df, validation_split=args.validation_split)
    model_path = calibrator.save(args.output)

    summary = {
        "samples": len(df),
        "val_brier": metrics["brier_score"],
        "val_log_loss": metrics["log_loss"],
        "epochs": metrics["epochs_trained"],
        "model_path": model_path,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
