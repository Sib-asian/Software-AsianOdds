"""
LSTM Predictor
==============

Recurrent Neural Network per analisi sequenze temporali.
Cattura momentum, trend, e pattern evoluti nel tempo.

Vantaggi rispetto a modelli statici:
- Cattura form dinamica (in ascesa vs in discesa)
- Rileva momentum shifts
- Considera contesto stagionale
- Adattamento a cambiamenti (nuovo allenatore, mercato)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Any, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class LSTMNet(nn.Module):
    """
    LSTM Neural Network per predizioni sequenze match.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super(LSTMNet, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor [batch_size, sequence_length, input_size]

        Returns:
            Output tensor [batch_size, 1] - probability
        """
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)

        # Use last output
        last_output = lstm_out[:, -1, :]

        # Fully connected
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc3(out)
        out = self.sigmoid(out)

        return out


class LSTMPredictor:
    """
    Predittore LSTM per risultati match basato su sequenze storiche.

    Analizza ultime N partite per catturare momentum e trend.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Inizializza LSTM predictor.

        Args:
            config: Configurazione modello
        """
        self.config = config or self._get_default_config()

        # Model parameters
        self.sequence_length = self.config.get('sequence_length', 10)
        self.hidden_size = self.config.get('hidden_size', 64)
        self.num_layers = self.config.get('num_layers', 2)
        self.dropout = self.config.get('dropout', 0.2)

        # Feature configuration
        self.feature_names = self._get_feature_names()
        self.input_size = len(self.feature_names)

        # Model (will be initialized when training or loading)
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Training state
        self.is_trained = False
        self.training_history = []

        logger.info(f"✅ LSTM Predictor initialized (device: {self.device})")
        logger.info(f"   Sequence length: {self.sequence_length}")
        logger.info(f"   Input features: {self.input_size}")

    def _get_feature_names(self) -> List[str]:
        """
        Define features per ogni match nella sequenza.

        Features devono catturare stato match-by-match.
        """
        return [
            'result',  # 1=win, 0.5=draw, 0=loss
            'goals_scored',
            'goals_conceded',
            'xg',
            'xga',
            'shots',
            'shots_on_target',
            'possession',
            'is_home',  # 1=home, 0=away
            'opponent_strength',  # Estimated
        ]

    def prepare_sequence(
        self,
        match_history: List[Dict],
        current_match: Dict,
        api_context: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Prepara sequenza di features dalle ultime N partite.

        Args:
            match_history: Lista ultimi match della squadra
            current_match: Match da predire
            api_context: Contesto API (optional)

        Returns:
            Sequence array [sequence_length, input_size]
        """
        sequence = []

        # Use last N matches from history
        recent_matches = match_history[-self.sequence_length:]

        # Pad if needed
        while len(recent_matches) < self.sequence_length:
            # Pad with neutral match
            recent_matches.insert(0, self._get_neutral_match())

        # Extract features for each match
        for match in recent_matches:
            features = self._extract_match_features(match)
            sequence.append(features)

        return np.array(sequence)

    def _extract_match_features(self, match: Dict) -> List[float]:
        """
        Estrae features da singolo match storico.
        """
        features = []

        for feat_name in self.feature_names:
            if feat_name == 'result':
                # Encode result as continuous
                result = match.get('result', 'D')
                if result == 'W':
                    value = 1.0
                elif result == 'D':
                    value = 0.5
                else:
                    value = 0.0
                features.append(value)

            elif feat_name == 'goals_scored':
                features.append(match.get('goals_scored', 1.5))

            elif feat_name == 'goals_conceded':
                features.append(match.get('goals_conceded', 1.2))

            elif feat_name == 'xg':
                features.append(match.get('xg', 1.5))

            elif feat_name == 'xga':
                features.append(match.get('xga', 1.2))

            elif feat_name == 'shots':
                features.append(match.get('shots', 12))

            elif feat_name == 'shots_on_target':
                features.append(match.get('shots_on_target', 4))

            elif feat_name == 'possession':
                features.append(match.get('possession', 50.0) / 100.0)

            elif feat_name == 'is_home':
                features.append(1.0 if match.get('venue') == 'home' else 0.0)

            elif feat_name == 'opponent_strength':
                # Estimate from opponent name or league
                features.append(match.get('opponent_strength', 0.5))

            else:
                features.append(0.0)

        return features

    def _get_neutral_match(self) -> Dict:
        """Ritorna match neutro per padding"""
        return {
            'result': 'D',
            'goals_scored': 1.0,
            'goals_conceded': 1.0,
            'xg': 1.5,
            'xga': 1.5,
            'shots': 12,
            'shots_on_target': 4,
            'possession': 50.0,
            'venue': 'home',
            'opponent_strength': 0.5
        }

    def predict(
        self,
        match_history: List[Dict],
        current_match: Dict,
        api_context: Optional[Dict] = None
    ) -> float:
        """
        Predice probabilità vittoria basandosi su sequenza storica.

        Args:
            match_history: Ultimi match squadra casa
            current_match: Match da predire
            api_context: Contesto API (optional)

        Returns:
            Probabilità vittoria (0-1)
        """
        # Prepare sequence
        sequence = self.prepare_sequence(match_history, current_match, api_context)

        # Convert to tensor
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)

        if self.is_trained and self.model is not None:
            # Trained model prediction
            self.model.eval()
            with torch.no_grad():
                prob = self.model(sequence_tensor).item()
        else:
            # Rule-based fallback
            prob = self._rule_based_prediction(sequence)

        # Clamp
        prob = np.clip(prob, 0.05, 0.95)

        logger.debug(f"LSTM prediction: {prob:.3f}")

        return float(prob)

    def _rule_based_prediction(self, sequence: np.ndarray) -> float:
        """
        Predizione rule-based quando modello non trainato.

        Analizza trend nelle ultime partite.
        """
        # Extract results from sequence
        results = sequence[:, 0]  # First feature is 'result'

        # Recent form (weighted toward recent matches)
        weights = np.linspace(0.5, 1.5, len(results))
        weighted_form = np.average(results, weights=weights)

        # Trend (ultimi 3 vs precedenti)
        recent_form = np.mean(results[-3:])
        previous_form = np.mean(results[-6:-3]) if len(results) >= 6 else np.mean(results[:-3])
        trend = recent_form - previous_form

        # xG trend
        xg_values = sequence[:, 3]  # xG feature
        xg_trend = np.mean(xg_values[-3:]) - np.mean(xg_values[:-3])

        # Base probability from weighted form
        prob = 0.3 + weighted_form * 0.3

        # Adjust for trend
        prob += trend * 0.15

        # Adjust for xG trend
        prob += xg_trend * 0.10

        # Home advantage (if last match is home)
        if sequence[-1, 8] > 0.5:  # is_home feature
            prob += 0.05

        # Clamp
        prob = np.clip(prob, 0.1, 0.9)

        return prob

    def train(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        validation_split: float = 0.2,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001
    ):
        """
        Train LSTM model su sequenze storiche.

        Args:
            sequences: Training sequences [n_samples, seq_length, input_size]
            labels: Target labels [n_samples] (1=home win, 0=not)
            validation_split: Validation percentage
            epochs: Training epochs
            batch_size: Batch size
            learning_rate: Learning rate
        """
        logger.info(f"🎓 Training LSTM on {len(sequences)} sequences...")

        # Initialize model
        if self.model is None:
            self.model = LSTMNet(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout
            ).to(self.device)

        # Split train/val
        split_idx = int(len(sequences) * (1 - validation_split))
        X_train, X_val = sequences[:split_idx], sequences[split_idx:]
        y_train, y_val = labels[:split_idx], labels[split_idx:]

        # Convert to tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_val = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)

        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10

        for epoch in range(epochs):
            self.model.train()

            # Training
            total_loss = 0
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_train_loss = total_loss / (len(X_train) / batch_size)

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_loss = criterion(val_outputs, y_val).item()

                # Accuracy
                val_preds = (val_outputs > 0.5).float()
                val_accuracy = (val_preds == y_val).float().mean().item()

            # Log progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Train Loss: {avg_train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"Val Acc: {val_accuracy:.1%}"
                )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

            # Store history
            self.training_history.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy
            })

        self.is_trained = True

        logger.info(f"✅ LSTM training completed")
        logger.info(f"   Best val loss: {best_val_loss:.4f}")
        logger.info(f"   Final val accuracy: {val_accuracy:.1%}")

        return {
            'best_val_loss': best_val_loss,
            'final_val_accuracy': val_accuracy,
            'training_history': self.training_history
        }

    def save(self, filepath: str):
        """Salva modello su disco"""
        if self.model is None:
            logger.warning("⚠️  No model to save")
            return

        save_data = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'feature_names': self.feature_names,
            'input_size': self.input_size,
            'is_trained': self.is_trained,
            'training_history': self.training_history
        }

        torch.save(save_data, filepath)
        logger.info(f"💾 LSTM model saved to {filepath}")

    def load(self, filepath: str):
        """Carica modello da disco"""
        save_data = torch.load(filepath, map_location=self.device)

        self.config = save_data.get('config', self.config)
        self.feature_names = save_data['feature_names']
        self.input_size = save_data['input_size']
        self.is_trained = save_data['is_trained']
        self.training_history = save_data.get('training_history', [])

        # Recreate model
        self.model = LSTMNet(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)

        self.model.load_state_dict(save_data['model_state_dict'])
        self.model.eval()

        logger.info(f"📦 LSTM model loaded from {filepath}")

    def _get_default_config(self) -> Dict:
        """Configurazione di default"""
        return {
            'sequence_length': 10,
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.2
        }


if __name__ == "__main__":
    # Test LSTM predictor
    logging.basicConfig(level=logging.INFO)

    print("Testing LSTM Predictor...")
    print("=" * 70)

    predictor = LSTMPredictor()

    # Test prediction (senza training)
    match_history = [
        {'result': 'W', 'goals_scored': 2, 'goals_conceded': 0, 'xg': 2.1, 'xga': 0.8, 'venue': 'home'},
        {'result': 'W', 'goals_scored': 3, 'goals_conceded': 1, 'xg': 2.5, 'xga': 1.2, 'venue': 'away'},
        {'result': 'D', 'goals_scored': 1, 'goals_conceded': 1, 'xg': 1.8, 'xga': 1.5, 'venue': 'home'},
        {'result': 'W', 'goals_scored': 2, 'goals_conceded': 1, 'xg': 2.0, 'xga': 1.1, 'venue': 'away'},
        {'result': 'L', 'goals_scored': 0, 'goals_conceded': 2, 'xg': 1.2, 'xga': 2.3, 'venue': 'away'},
    ]

    current_match = {
        'home': 'Inter',
        'away': 'Napoli',
        'league': 'Serie A'
    }

    prob = predictor.predict(match_history, current_match)

    print(f"\n✅ Prediction completed")
    print(f"Match: {current_match['home']} vs {current_match['away']}")
    print(f"Probability (home win): {prob:.1%}")
    print(f"Model trained: {predictor.is_trained}")
    print(f"Sequence length: {predictor.sequence_length}")
    print(f"Features per match: {predictor.input_size}")

    print("\n" + "=" * 70)
    print("✅ LSTM Predictor test passed!")
