"""
BLOCCO 1: Probability Calibrator
=================================

Calibra le probabilità del modello Dixon-Coles usando Neural Network.

Funzionalità:
- Multi-Layer Perceptron (MLP) per calibrazione
- Context-aware: usa dati API (injuries, form, etc)
- Corregge bias sistematici del modello
- Fornisce bande di incertezza
- Training su dati storici con actual outcomes

Input: probabilità raw Dixon-Coles + API context
Output: probabilità calibrata + incertezza
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import joblib
import json
from datetime import datetime

# Deep Learning
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("⚠️ PyTorch not available. Install with: pip install torch")

from .config import AIConfig

logger = logging.getLogger(__name__)


# Define classes only if PyTorch is available
if TORCH_AVAILABLE:
    class CalibratorMLP(nn.Module):
        """
        Multi-Layer Perceptron per calibrazione probabilità.

        Architecture:
        Input → Hidden Layers → Dropout → Output (sigmoid)
        """

        def __init__(
            self,
            input_size: int,
            hidden_layers: List[int],
            dropout: float = 0.2
        ):
            super().__init__()

            layers = []
            prev_size = input_size

            # Hidden layers
            for hidden_size in hidden_layers:
                layers.append(nn.Linear(prev_size, hidden_size))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                prev_size = hidden_size

            # Output layer (sigmoid for probability)
            layers.append(nn.Linear(prev_size, 1))
            layers.append(nn.Sigmoid())

            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x)
else:
    # Dummy class when PyTorch is not available
    class CalibratorMLP:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required for CalibratorMLP. Install with: pip install torch")


class ProbabilityCalibrator:
    """
    Calibratore di probabilità con Neural Network.

    Features utilizzate:
    - Probabilità raw Dixon-Coles
    - League encoding
    - Market type encoding
    - Team strength difference
    - API context (injuries, form, xG, etc)

    Output:
    - Probabilità calibrata
    - Uncertainty (standard deviation)
    - Calibration shift
    """

    def __init__(self, config: Optional[AIConfig] = None):
        """
        Inizializza calibratore.

        Args:
            config: Configurazione AI
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch is required for Probability Calibrator. "
                "Install with: pip install torch"
            )

        self.config = config or AIConfig()
        self.model: Optional[CalibratorMLP] = None
        self.scaler = None  # Per normalizzazione features
        self.feature_names: List[str] = []
        self.is_trained = False

        # Statistics
        self.train_history = {
            "loss": [],
            "val_loss": [],
            "epochs": 0
        }

        logger.info("✅ Probability Calibrator initialized")

    def extract_features(
        self,
        prob_raw: float,
        context: Dict[str, Any]
    ) -> np.ndarray:
        """
        Estrae features per calibrazione.

        Args:
            prob_raw: Probabilità raw (Dixon-Coles)
            context: Context da API Data Engine + match info

        Returns:
            Array di features normalizzate
        """
        features = []

        # Base probability
        features.append(prob_raw)

        # Probability derived features
        features.append(prob_raw ** 2)  # Non-linearity
        features.append(np.log(prob_raw + 1e-10))  # Log scale
        features.append(1 - prob_raw)  # Complement

        # League encoding (simple numerical)
        league_scores = {
            "serie a": 0.9,
            "premier league": 0.95,
            "la liga": 0.92,
            "bundesliga": 0.90,
            "ligue 1": 0.85,
            "champions league": 1.0,
            "serie b": 0.6,
        }
        league = context.get("league", "").lower()
        league_score = 0.5  # default
        for league_name, score in league_scores.items():
            if league_name in league:
                league_score = score
                break
        features.append(league_score)

        # Market type encoding
        market_scores = {
            "1x2": 0.8,
            "over/under": 0.7,
            "btts": 0.6,
            "asian handicap": 0.75
        }
        market = context.get("market", "1x2").lower()
        market_score = market_scores.get(market, 0.5)
        features.append(market_score)

        # API context features (se disponibili)
        api_context = context.get("api_context", {})

        # Injuries impact
        injuries_home = api_context.get("injuries_home", [])
        injuries_away = api_context.get("injuries_away", [])
        injury_impact = sum([inj.get("impact", 0.0) for inj in injuries_home + injuries_away])
        features.append(min(injury_impact, 1.0))  # Cap a 1.0

        # Form features (W-D-L encoding)
        form_home = api_context.get("form_home", "DDDDD")
        form_away = api_context.get("form_away", "DDDDD")

        def encode_form(form_string: str) -> float:
            """Encode form string W-D-L to score"""
            score = 0.0
            for char in form_string[-5:]:  # Last 5 games
                if char == 'W':
                    score += 1.0
                elif char == 'D':
                    score += 0.5
            return score / 5.0  # Normalize to 0-1

        form_home_score = encode_form(form_home)
        form_away_score = encode_form(form_away)
        form_diff = form_home_score - form_away_score  # -1 to +1
        features.append(form_home_score)
        features.append(form_away_score)
        features.append(form_diff)

        # xG features (se disponibili)
        xg_home = api_context.get("xg_home_last5", 0.0)
        xg_away = api_context.get("xg_away_last5", 0.0)
        xga_home = api_context.get("xga_home_last5", 0.0)  # Goals conceded
        xga_away = api_context.get("xga_away_last5", 0.0)

        # Normalize xG (typical range 0-15 for 5 games)
        features.append(min(xg_home / 15.0, 1.0))
        features.append(min(xg_away / 15.0, 1.0))
        features.append(min(xga_home / 15.0, 1.0))
        features.append(min(xga_away / 15.0, 1.0))

        # Lineup quality
        lineup_quality_home = api_context.get("lineup_quality_home", 0.85)
        lineup_quality_away = api_context.get("lineup_quality_away", 0.85)
        features.append(lineup_quality_home)
        features.append(lineup_quality_away)

        # Data quality from API Engine
        data_quality = context.get("data_quality", 0.5)
        features.append(data_quality)

        # Total features: 20
        return np.array(features, dtype=np.float32)

    def train(
        self,
        historical_data: pd.DataFrame,
        validation_split: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Addestra il calibratore su dati storici.

        Args:
            historical_data: DataFrame con colonne:
                - prob_raw: Probabilità Dixon-Coles
                - outcome: Risultato reale (0 o 1)
                - context: JSON con context (league, market, api_data, etc)
            validation_split: Frazione per validation (default da config)

        Returns:
            Dizionario con metriche di training
        """
        if len(historical_data) < self.config.min_samples_to_train:
            raise ValueError(
                f"Not enough training data: {len(historical_data)} samples "
                f"(minimum: {self.config.min_samples_to_train})"
            )

        logger.info(f"📚 Training calibrator on {len(historical_data)} samples...")

        # Extract features and targets
        X = []
        y = []

        for idx, row in historical_data.iterrows():
            try:
                # Parse context if JSON string
                context = row["context"]
                if isinstance(context, str):
                    context = json.loads(context)

                features = self.extract_features(row["prob_raw"], context)
                X.append(features)
                y.append(float(row["outcome"]))

            except Exception as e:
                logger.warning(f"⚠️ Skipping row {idx}: {e}")
                continue

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32).reshape(-1, 1)

        logger.info(f"✅ Extracted {len(X)} feature vectors (dim={X.shape[1]})")

        # Normalize features
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)

        # Train/validation split
        val_split = validation_split or self.config.calibrator_validation_split
        split_idx = int(len(X) * (1 - val_split))

        X_train = X[:split_idx]
        y_train = y[:split_idx]
        X_val = X[split_idx:]
        y_val = y[split_idx:]

        logger.info(
            f"📊 Split: train={len(X_train)}, val={len(X_val)}"
        )

        # Convert to PyTorch tensors
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.FloatTensor(y_train)
        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.FloatTensor(y_val)

        # Create DataLoader
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.calibrator_batch_size,
            shuffle=True
        )

        # Initialize model
        input_size = X.shape[1]
        self.model = CalibratorMLP(
            input_size=input_size,
            hidden_layers=self.config.calibrator_hidden_layers,
            dropout=self.config.calibrator_dropout
        )

        # Loss and optimizer
        criterion = nn.BCELoss()  # Binary Cross Entropy
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.calibrator_learning_rate
        )

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.config.calibrator_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation phase
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_t)
                val_loss = criterion(val_outputs, y_val_t).item()

            # Log progress
            self.train_history["loss"].append(train_loss)
            self.train_history["val_loss"].append(val_loss)
            self.train_history["epochs"] += 1

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{self.config.calibrator_epochs}: "
                    f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
                )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= self.config.calibrator_early_stopping_patience:
                    logger.info(f"🛑 Early stopping at epoch {epoch+1}")
                    break

        # Restore best model
        self.model.load_state_dict(self.best_model_state)
        self.is_trained = True

        # Calculate final metrics
        metrics = self._calculate_metrics(X_val_t, y_val_t)

        logger.info(f"✅ Training completed. Validation Brier score: {metrics['brier_score']:.4f}")

        return {
            "train_loss": self.train_history["loss"][-1],
            "val_loss": best_val_loss,
            "epochs_trained": self.train_history["epochs"],
            **metrics
        }

    def _calculate_metrics(
        self,
        X: torch.Tensor,
        y: torch.Tensor
    ) -> Dict[str, float]:
        """Calcola metriche di performance"""
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X).numpy()
            targets = y.numpy()

        # Brier score
        brier_score = np.mean((predictions - targets) ** 2)

        # Log loss
        epsilon = 1e-15
        predictions_clipped = np.clip(predictions, epsilon, 1 - epsilon)
        log_loss = -np.mean(
            targets * np.log(predictions_clipped) +
            (1 - targets) * np.log(1 - predictions_clipped)
        )

        # Calibration error (mean signed difference)
        calibration_error = np.mean(predictions - targets)

        return {
            "brier_score": float(brier_score),
            "log_loss": float(log_loss),
            "calibration_error": float(calibration_error)
        }

    def calibrate(
        self,
        prob_raw: float,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calibra una probabilità usando il modello trained.

        Args:
            prob_raw: Probabilità raw (Dixon-Coles)
            context: Context (league, market, api_context, etc)

        Returns:
            Dizionario con:
            - prob_calibrated: Probabilità calibrata
            - uncertainty: Banda di incertezza
            - calibration_shift: Quanto è stata corretta
            - raw_features: Features estratte (per debug)
        """
        if not self.is_trained:
            logger.warning("⚠️ Calibrator not trained, returning raw probability")
            return {
                "prob_calibrated": prob_raw,
                "prob_raw": float(prob_raw),
                "uncertainty": 0.10,  # Default uncertainty
                "calibration_shift": 0.0,
                "calibration_method": "rule-based",
                "data_quality": context.get("data_quality", 0.5),
                "note": "model_not_trained"
            }

        try:
            # Extract features
            features = self.extract_features(prob_raw, context)

            # Normalize
            features_normalized = self.scaler.transform(features.reshape(1, -1))

            # Convert to tensor
            features_tensor = torch.FloatTensor(features_normalized)

            # Predict
            self.model.eval()
            with torch.no_grad():
                prob_calibrated = self.model(features_tensor).item()

            # Apply bounds
            prob_calibrated = np.clip(
                prob_calibrated,
                self.config.min_probability,
                self.config.max_probability
            )

            # Calculate calibration shift
            calibration_shift = prob_calibrated - prob_raw

            # Cap calibration shift
            if abs(calibration_shift) > self.config.max_calibration_shift:
                logger.warning(
                    f"⚠️ Large calibration shift: {calibration_shift:.3f} "
                    f"(capping to ±{self.config.max_calibration_shift})"
                )
                calibration_shift = np.sign(calibration_shift) * self.config.max_calibration_shift
                prob_calibrated = prob_raw + calibration_shift

            # Estimate uncertainty (based on calibration shift magnitude)
            # Larger shifts = more uncertainty
            uncertainty = min(abs(calibration_shift) * 2.0, 0.15)

            return {
                "prob_calibrated": float(prob_calibrated),
                "uncertainty": float(uncertainty),
                "calibration_shift": float(calibration_shift),
                "prob_raw": float(prob_raw),
                "calibration_method": "ML-based",
                "data_quality": context.get("data_quality", 0.5)
            }

        except Exception as e:
            logger.error(f"❌ Error in calibration: {e}")
            return {
                "prob_calibrated": prob_raw,
                "prob_raw": float(prob_raw),
                "uncertainty": 0.20,  # High uncertainty on error
                "calibration_shift": 0.0,
                "calibration_method": "rule-based",
                "data_quality": context.get("data_quality", 0.5),
                "error": str(e)
            }

    def save(self, filepath: Optional[str] = None) -> str:
        """
        Salva il modello trained.

        Args:
            filepath: Path dove salvare (default: models/calibrator.pth)

        Returns:
            Path dove è stato salvato
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model")

        if filepath is None:
            filepath = self.config.models_dir / "calibrator.pth"

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save model state, scaler, and metadata
        save_dict = {
            "model_state": self.model.state_dict(),
            "scaler": self.scaler,
            "train_history": self.train_history,
            "config": {
                "hidden_layers": self.config.calibrator_hidden_layers,
                "dropout": self.config.calibrator_dropout
            },
            "timestamp": datetime.now().isoformat()
        }

        torch.save(save_dict, filepath)
        logger.info(f"💾 Model saved to {filepath}")

        return str(filepath)

    def load(self, filepath: Optional[str] = None):
        """
        Carica un modello salvato.

        Args:
            filepath: Path del modello (default: models/calibrator.pth)
        """
        if filepath is None:
            filepath = self.config.models_dir / "calibrator.pth"

        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        # Load checkpoint
        checkpoint = torch.load(filepath, weights_only=False)

        # Restore scaler
        self.scaler = checkpoint["scaler"]

        # Recreate model with same architecture
        config_saved = checkpoint["config"]
        input_size = len(self.scaler.mean_)  # Infer from scaler

        self.model = CalibratorMLP(
            input_size=input_size,
            hidden_layers=config_saved["hidden_layers"],
            dropout=config_saved["dropout"]
        )

        # Load weights
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

        # Restore history
        self.train_history = checkpoint["train_history"]
        self.is_trained = True

        logger.info(f"✅ Model loaded from {filepath}")
        logger.info(f"   Trained for {self.train_history['epochs']} epochs")


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def test_probability_calibrator():
    """Test del Probability Calibrator"""
    print("=" * 70)
    print("TEST: Probability Calibrator")
    print("=" * 70)

    if not TORCH_AVAILABLE:
        print("❌ PyTorch not available, skipping test")
        return

    # Create synthetic training data
    print("\nCreating synthetic training data...")
    np.random.seed(42)

    n_samples = 2000
    data = []

    for i in range(n_samples):
        # Generate synthetic probability (biased high)
        true_prob = np.random.beta(2, 2)  # True underlying probability
        prob_raw = min(true_prob * 1.15, 0.99)  # Biased +15% (optimistic)

        # Generate outcome
        outcome = 1 if np.random.random() < true_prob else 0

        # Context
        context = {
            "league": np.random.choice(["Serie A", "Serie B", "Premier League"]),
            "market": "1x2",
            "data_quality": np.random.uniform(0.5, 1.0),
            "api_context": {
                "form_home": "WWDWL",
                "form_away": "LDLLW",
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
            "context": context
        })

    df = pd.DataFrame(data)
    print(f"✅ Created {len(df)} samples")

    # Initialize calibrator
    print("\nInitializing calibrator...")
    calibrator = ProbabilityCalibrator()

    # Train
    print("\nTraining...")
    metrics = calibrator.train(df, validation_split=0.2)
    print(f"✅ Training completed")
    print(f"   Val Brier score: {metrics['brier_score']:.4f}")
    print(f"   Val Log loss: {metrics['log_loss']:.4f}")
    print(f"   Epochs: {metrics['epochs_trained']}")

    # Test calibration
    print("\nTest calibration on examples:")
    test_cases = [
        {"prob_raw": 0.65, "desc": "Inter vs Genoa (65%)"},
        {"prob_raw": 0.45, "desc": "Roma vs Lecce (45%)"},
        {"prob_raw": 0.80, "desc": "Napoli vs Empoli (80%)"},
    ]

    for case in test_cases:
        context = {
            "league": "Serie A",
            "market": "1x2",
            "data_quality": 0.85,
            "api_context": {
                "form_home": "WWWWW",
                "form_away": "LLLLD",
                "xg_home_last5": 10.0,
                "xg_away_last5": 5.0,
                "xga_home_last5": 3.0,
                "xga_away_last5": 9.0,
                "lineup_quality_home": 0.95,
                "lineup_quality_away": 0.80,
                "injuries_home": [],
                "injuries_away": []
            }
        }

        result = calibrator.calibrate(case["prob_raw"], context)
        print(f"\n  {case['desc']}")
        print(f"    Raw: {result['prob_raw']:.1%}")
        print(f"    Calibrated: {result['prob_calibrated']:.1%}")
        print(f"    Shift: {result['calibration_shift']:+.1%}")
        print(f"    Uncertainty: ±{result['uncertainty']:.1%}")

    # Save model
    print("\nSaving model...")
    path = calibrator.save()
    print(f"✅ Model saved to {path}")

    # Test load
    print("\nTesting model load...")
    calibrator2 = ProbabilityCalibrator()
    calibrator2.load(path)
    print(f"✅ Model loaded successfully")

    print("\n" + "=" * 70)
    print("✅ Probability Calibrator tests completed")
    print("=" * 70)


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    test_probability_calibrator()
