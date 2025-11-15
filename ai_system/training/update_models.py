"""
Batch training CLI per aggiornare i modelli AI senza server H24.

Uso:
    python -m ai_system.training.update_models \
        --data storico_analisi.csv \
        --allow-synthetic
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from ..analysis.regime_detector import RegimeDetector
from ..blocco_1_calibrator import ProbabilityCalibrator
from ..config import AIConfig
from ..utils.data_preparation import (
    create_synthetic_training_data,
    load_historical_data,
)

logger = logging.getLogger(__name__)


def _load_dataset(path: Path, min_samples: int, allow_synthetic: bool, synthetic_samples: int) -> pd.DataFrame:
    if path.exists():
        df = load_historical_data(str(path), min_samples=min_samples)
    else:
        logger.warning("⚠️  Dataset %s non trovato. Uso synthetic data.", path)
        df = pd.DataFrame()

    if len(df) < min_samples:
        if not allow_synthetic:
            raise RuntimeError(
                f"Dataset insufficiente ({len(df)}). "
                "Abilita --allow-synthetic per bootstrap automatico."
            )
        logger.warning(
            "⚠️  Solo %s campioni disponibili (min=%s). Genero dataset sintetico.",
            len(df),
            min_samples,
        )
        df = create_synthetic_training_data(synthetic_samples)

    return df


def _configure_fast_mode(config: AIConfig, fast: bool):
    if not fast:
        return
    config.calibrator_epochs = min(40, config.calibrator_epochs)
    config.calibrator_early_stopping_patience = 5
    logger.info("⏩ Fast mode attivo: epochs=%s", config.calibrator_epochs)


def train_calibrator(config: AIConfig, df: pd.DataFrame):
    calibrator = ProbabilityCalibrator(config)
    metrics = calibrator.train(df, validation_split=config.calibrator_validation_split)
    calibrator.save(config.models_dir / "calibrator.pth")
    logger.info("✅ Calibratore aggiornato: %s", metrics)


def train_regime_detector(config: AIConfig, df: pd.DataFrame, synthetic_samples: int) -> Path:
    detector = RegimeDetector(config, n_clusters=config.regime_clusters)

    samples = detector.build_dataset_from_history(df)
    if len(samples) < config.regime_min_samples:
        logger.warning(
            "⚠️  Campioni regime insufficienti (%s/%s). Uso dataset sintetico.",
            len(samples),
            config.regime_min_samples,
        )
        samples = detector.generate_synthetic_samples(synthetic_samples)

    detector.train(samples)
    model_path = config.models_dir / config.regime_detector_model_name
    detector.save(model_path)
    return model_path


def run_batch_update(
    data_path: Optional[str] = None,
    allow_synthetic: bool = True,
    synthetic_samples: int = 2500,
    fast: bool = False,
    min_samples: Optional[int] = None,
):
    config = AIConfig()
    if min_samples:
        config.min_samples_to_train = min_samples
    _configure_fast_mode(config, fast)

    dataset_path = Path(data_path or config.predictions_db)
    df = _load_dataset(dataset_path, config.min_samples_to_train, allow_synthetic, synthetic_samples)

    train_calibrator(config, df)
    regime_path = train_regime_detector(config, df, synthetic_samples)
    logger.info("✅ Regime detector salvato in %s", regime_path)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggiorna i modelli AI (batch).")
    parser.add_argument("--data", type=str, help="Path dataset storico (CSV o SQLite).")
    parser.add_argument("--min-samples", type=int, default=1000, help="Minimo campioni per training.")
    parser.add_argument("--synthetic-samples", type=int, default=2500, help="Campioni sintetici da generare se necessario.")
    parser.add_argument("--no-allow-synthetic", action="store_true", help="Disabilita fallback sintetico.")
    parser.add_argument("--fast", action="store_true", help="Riduce epochs per run veloce.")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    run_batch_update(
        data_path=args.data,
        allow_synthetic=not args.no_allow_synthetic,
        synthetic_samples=args.synthetic_samples,
        fast=args.fast,
        min_samples=args.min_samples,
    )


if __name__ == "__main__":
    main()
