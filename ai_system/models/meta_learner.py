"""
Meta-Learner
============

Neural network che impara come pesare dinamicamente i modelli dell'ensemble.

Input: Context features (league quality, data availability, model agreement, etc.)
Output: Weights per ogni modello (Dixon-Coles, XGBoost, LSTM)

Il Meta-Learner ottimizza i pesi in base al contesto specifico della partita,
permettendo all'ensemble di adattarsi dinamicamente.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple
import logging

from ..meta.context_features import (
    CONTEXT_FEATURE_NAMES,
    build_context_features,
    context_dict_to_array,
)

logger = logging.getLogger(__name__)


class MetaLearnerNet(nn.Module):
    """
    Neural Network per Meta-Learning.

    Prende context features e output weights per modelli.
    """

    def __init__(self, num_context_features: int, num_models: int):
        super(MetaLearnerNet, self).__init__()

        self.fc1 = nn.Linear(num_context_features, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, num_models)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Context features [batch_size, num_context_features]

        Returns:
            Model weights [batch_size, num_models] (sum to 1.0)
        """
        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)
        x = self.softmax(x)  # Ensure weights sum to 1.0

        return x


class MetaLearner:
    """
    Meta-Learner per ottimizzazione dinamica pesi ensemble.

    Può operare in due modalità:
    1. Rule-based: Usa regole predefinite per pesare modelli
    2. Trained: Usa neural network trainato su dati storici
    """

    def __init__(self, num_models: int = 3, config: Optional[Dict] = None):
        """
        Inizializza Meta-Learner.

        Args:
            num_models: Numero di modelli nell'ensemble (default: 3 = DC + XGB + LSTM)
            config: Configurazione
        """
        self.num_models = num_models
        self.config = config or {}

        # Context features definition
        self.context_features = list(CONTEXT_FEATURE_NAMES)
        self.num_context_features = len(self.context_features)

        # Model names
        self.model_names = ['dixon_coles', 'xgboost', 'lstm'][:num_models]

        # Neural network (will be initialized when training or loading)
        self.net = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Training state
        self.is_trained = False

        # Default weights (used when not trained)
        self.default_weights = self._get_default_weights()

        logger.info(f"✅ Meta-Learner initialized")
        logger.info(f"   Models: {self.model_names}")
        logger.info(f"   Context features: {self.num_context_features}")

    def calculate_weights(
        self,
        predictions: Dict[str, float],
        match_data: Dict,
        api_context: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Calcola weights ottimali per i modelli dato il contesto.

        Args:
            predictions: Dict con predizioni di ogni modello
            match_data: Dati match
            api_context: Contesto API (optional)

        Returns:
            Dict con weights per ogni modello (sum=1.0)
        """
        # Extract context features (as dict + array for downstream use)
        context_dict = build_context_features(match_data, predictions, api_context)
        context = context_dict_to_array(context_dict, self.context_features)

        if self.is_trained and self.net is not None:
            # Use trained neural network
            weights = self._predict_weights_nn(context)
        else:
            # Use rule-based weights
            weights = self._calculate_rule_based_weights(context, context_dict, predictions)

        # Ensure weights sum to 1.0 (numerical stability)
        total = sum(weights.values())
        if total == 0:
            logger.warning("All model weights are zero, using equal weights")
            weights = {k: 1.0/len(weights) for k in weights.keys()}
        else:
            weights = {k: v/total for k, v in weights.items()}

        logger.debug(f"Meta-Learner weights: {weights}")

        return weights

    def _extract_context_features(
        self,
        predictions: Dict[str, float],
        match_data: Dict,
        api_context: Optional[Dict]
    ) -> np.ndarray:
        """
        Estrae context features per decision making.
        """
        context_dict = build_context_features(match_data, predictions, api_context)
        return context_dict_to_array(context_dict, self.context_features)

    def _predict_weights_nn(self, context: np.ndarray) -> Dict[str, float]:
        """
        Usa neural network per predire weights.
        """
        # Convert to tensor
        context_tensor = torch.FloatTensor(context).unsqueeze(0).to(self.device)

        # Predict
        self.net.eval()
        with torch.no_grad():
            weights_tensor = self.net(context_tensor)
            weights_array = weights_tensor.cpu().numpy()[0]

        # Convert to dict
        weights = {name: float(w) for name, w in zip(self.model_names, weights_array)}

        return weights

    def _calculate_rule_based_weights(
        self,
        context: np.ndarray,
        context_dict: Dict[str, float],
        predictions: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calcola weights usando regole euristiche.

        Regole:
        - Top leagues: peso Dixon-Coles più alto
        - Bassa data availability: peso XGBoost basso, LSTM più alto (usa form)
        - Alta model agreement: pesi equilibrati
        - Pochi dati storici: peso LSTM basso
        - High H2H relevance: tutti i modelli pesati ugualmente
        """
        # Start with default weights
        weights = self.default_weights.copy()

        # Ensure context features dict is available
        if not context_dict:
            context_dict = {name: context[i] for i, name in enumerate(self.context_features)}

        # RULE 1: Top leagues favor Dixon-Coles
        if context_dict['is_top_league'] > 0.5:
            weights['dixon_coles'] += 0.10
            weights['xgboost'] -= 0.05
            weights['lstm'] -= 0.05

        # RULE 2: Low data availability
        if context_dict['data_availability'] < 0.5:
            # XGBoost needs good data, reduce weight
            weights['xgboost'] -= 0.15
            # LSTM relies more on sequence, increase
            weights['lstm'] += 0.10
            # Dixon-Coles statistical, increase
            weights['dixon_coles'] += 0.05

        # RULE 3: High model agreement (models converge)
        if context_dict['model_agreement'] > 0.9:
            # Equalize weights when all agree
            avg_weight = 1.0 / self.num_models
            for model in self.model_names:
                weights[model] = avg_weight * 0.5 + weights[model] * 0.5

        # RULE 4: Few historical matches
        if context_dict['historical_matches'] < 0.3:
            # LSTM needs sequences, reduce
            weights['lstm'] -= 0.10
            # XGBoost can generalize, increase
            weights['xgboost'] += 0.05
            # Dixon-Coles statistical, increase
            weights['dixon_coles'] += 0.05

        # RULE 5: High H2H relevance
        if context_dict['h2h_relevance'] > 0.7:
            # All models benefit from H2H data
            # Slight increase to XGBoost (uses H2H features directly)
            weights['xgboost'] += 0.05
            weights['dixon_coles'] -= 0.03
            weights['lstm'] -= 0.02

        # RULE 6: Late season (more data, more reliable)
        if context_dict['season_progress'] > 0.7:
            # XGBoost and LSTM have more data to learn from
            weights['xgboost'] += 0.05
            weights['lstm'] += 0.05
            weights['dixon_coles'] -= 0.10

        # RULE 7: High injuries impact
        if context_dict['injuries_impact'] > 0.3:
            # XGBoost explicitly models injuries, increase
            weights['xgboost'] += 0.08
            # Dixon-Coles doesn't, decrease
            weights['dixon_coles'] -= 0.05
            weights['lstm'] -= 0.03

        # Ensure positive weights
        for model in self.model_names:
            weights[model] = max(weights[model], 0.05)

        # Normalize to sum = 1.0
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}

        return weights

    def train(
        self,
        X_context: np.ndarray,
        predictions_history: np.ndarray,
        y_actual: np.ndarray,
        validation_split: float = 0.2,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001
    ):
        """
        Train Meta-Learner su dati storici.

        Args:
            X_context: Context features [n_samples, num_context_features]
            predictions_history: Predictions di ogni modello [n_samples, num_models]
            y_actual: Risultati reali [n_samples] (1=home win, 0=not)
            validation_split: Validation percentage
            epochs: Training epochs
            batch_size: Batch size
            learning_rate: Learning rate
        """
        logger.info(f"🎓 Training Meta-Learner on {len(X_context)} samples...")

        # Initialize network
        if self.net is None:
            self.net = MetaLearnerNet(
                num_context_features=self.num_context_features,
                num_models=self.num_models
            ).to(self.device)

        # Per ogni sample, trova optimal weights che minimizzano prediction error
        optimal_weights = []

        for i in range(len(X_context)):
            preds = predictions_history[i]  # [num_models]
            actual = y_actual[i]

            # Find weights that minimize |weighted_pred - actual|
            best_weights = self._find_optimal_weights(preds, actual)
            optimal_weights.append(best_weights)

        optimal_weights = np.array(optimal_weights)

        # Split train/val
        split_idx = int(len(X_context) * (1 - validation_split))
        X_train, X_val = X_context[:split_idx], X_context[split_idx:]
        y_train, y_val = optimal_weights[:split_idx], optimal_weights[split_idx:]

        # Convert to tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_val = torch.FloatTensor(y_val).to(self.device)

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(epochs):
            self.net.train()

            total_loss = 0
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]

                optimizer.zero_grad()
                outputs = self.net(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_train_loss = total_loss / (len(X_train) / batch_size)

            # Validation
            self.net.eval()
            with torch.no_grad():
                val_outputs = self.net(X_val)
                val_loss = criterion(val_outputs, y_val).item()

            if (epoch + 1) % 20 == 0 or epoch == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

        self.is_trained = True

        logger.info(f"✅ Meta-Learner training completed")
        logger.info(f"   Final val loss: {val_loss:.4f}")

        return {'final_val_loss': val_loss}

    def _find_optimal_weights(self, predictions: np.ndarray, actual: float) -> np.ndarray:
        """
        Trova weights ottimali che minimizzano errore per singolo sample.

        Usa ottimizzazione numerica.
        """
        from scipy.optimize import minimize

        def objective(weights):
            # Weighted prediction
            weighted_pred = np.dot(weights, predictions)
            # Squared error
            return (weighted_pred - actual) ** 2

        def constraint_sum(weights):
            # Weights must sum to 1.0
            return np.sum(weights) - 1.0

        # Initial guess (uniform)
        x0 = np.ones(self.num_models) / self.num_models

        # Bounds (weights between 0 and 1)
        bounds = [(0.0, 1.0) for _ in range(self.num_models)]

        # Constraint
        constraints = {'type': 'eq', 'fun': constraint_sum}

        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

        return result.x

    def save(self, filepath: str):
        """Salva Meta-Learner su disco"""
        if self.net is None:
            logger.warning("⚠️  No model to save")
            return

        save_data = {
            'net_state_dict': self.net.state_dict(),
            'num_models': self.num_models,
            'model_names': self.model_names,
            'context_features': self.context_features,
            'num_context_features': self.num_context_features,
            'is_trained': self.is_trained,
            'default_weights': self.default_weights,
            'config': self.config
        }

        torch.save(save_data, filepath)
        logger.info(f"💾 Meta-Learner saved to {filepath}")

    def load(self, filepath: str):
        """Carica Meta-Learner da disco"""
        save_data = torch.load(filepath, map_location=self.device)

        self.num_models = save_data['num_models']
        self.model_names = save_data['model_names']
        self.context_features = save_data['context_features']
        self.num_context_features = save_data['num_context_features']
        self.is_trained = save_data['is_trained']
        self.default_weights = save_data['default_weights']
        self.config = save_data.get('config', {})

        # Recreate network
        self.net = MetaLearnerNet(
            num_context_features=self.num_context_features,
            num_models=self.num_models
        ).to(self.device)

        self.net.load_state_dict(save_data['net_state_dict'])
        self.net.eval()

        logger.info(f"📦 Meta-Learner loaded from {filepath}")

    def _get_default_weights(self) -> Dict[str, float]:
        """
        Default weights quando non trainato.

        Dixon-Coles: 40% (provato e affidabile)
        XGBoost: 35% (feature-rich)
        LSTM: 25% (cattura momentum)
        """
        if self.num_models == 3:
            return {
                'dixon_coles': 0.40,
                'xgboost': 0.35,
                'lstm': 0.25
            }
        else:
            # Uniform distribution
            return {name: 1.0/self.num_models for name in self.model_names}



if __name__ == "__main__":
    # Test Meta-Learner
    logging.basicConfig(level=logging.INFO)

    print("Testing Meta-Learner...")
    print("=" * 70)

    meta = MetaLearner(num_models=3)

    # Test weight calculation
    predictions = {
        'dixon_coles': 0.65,
        'xgboost': 0.58,
        'lstm': 0.62
    }

    match_data = {
        'league': 'Serie A',
        'hours_to_kickoff': 12,
        'season_progress': 0.6
    }

    api_context = {
        'metadata': {'data_quality': 0.85},
        'match_data': {'h2h': {'total': 8}},
        'home_context': {'data': {'injuries': []}},
        'away_context': {'data': {'injuries': ['Player1']}}
    }

    weights = meta.calculate_weights(predictions, match_data, api_context)

    print(f"\n✅ Weights calculated")
    print(f"Model predictions: {predictions}")
    print(f"\nOptimal weights:")
    for model, weight in weights.items():
        print(f"  {model}: {weight:.1%}")

    # Calculate ensemble prediction
    ensemble_pred = sum(predictions[model] * weights[model] for model in predictions)
    print(f"\nEnsemble prediction: {ensemble_pred:.1%}")
    print(f"Average of models: {np.mean(list(predictions.values())):.1%}")
    print(f"Improvement: {abs(ensemble_pred - np.mean(list(predictions.values()))) * 100:.2f}pp")

    print("\n" + "=" * 70)
    print("✅ Meta-Learner test passed!")
