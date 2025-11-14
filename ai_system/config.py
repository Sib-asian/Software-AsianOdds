"""
AI System Configuration
=======================

Configurazione centralizzata per tutti i blocchi AI.
Tutti i parametri sono documentati e con valori di default ottimizzati.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path


@dataclass
class AIConfig:
    """Configurazione principale del sistema AI"""

    # ============================================================
    # PATHS & DIRECTORIES
    # ============================================================

    base_dir: Path = field(default_factory=lambda: Path(__file__).parent)
    models_dir: Path = field(default_factory=lambda: Path(__file__).parent / "models")
    cache_dir: Path = field(default_factory=lambda: Path(__file__).parent / "cache")
    logs_dir: Path = field(default_factory=lambda: Path(__file__).parent / "logs")

    # Database paths
    api_cache_db: str = "api_cache.db"
    predictions_db: str = "storico_analisi.csv"

    # ============================================================
    # ENSEMBLE META-MODEL
    # ============================================================

    # Enable ensemble (combines Dixon-Coles + XGBoost + LSTM)
    use_ensemble: bool = True  # Set False to use only Dixon-Coles

    # Load trained models on startup
    ensemble_load_models: bool = True

    # Ensemble models directory
    ensemble_models_dir: str = "ai_system/models"

    # ============================================================
    # LIVE MONITORING & NOTIFICATIONS
    # ============================================================

    # Telegram notifications
    telegram_enabled: bool = True
    telegram_bot_token: str = ""  # Set via env var TELEGRAM_BOT_TOKEN
    telegram_chat_id: str = ""    # Set via env var TELEGRAM_CHAT_ID

    # Notification thresholds
    telegram_min_ev: float = 5.0           # Min EV% per inviare notifica
    telegram_min_confidence: float = 60.0  # Min confidence per inviare
    telegram_rate_limit_seconds: int = 3   # Secondi tra messaggi

    # Live monitoring settings
    live_monitoring_enabled: bool = False  # Enable auto-monitoring
    live_update_interval: int = 60         # Secondi tra aggiornamenti (default: 60s)
    live_min_ev_alert: float = 8.0         # Min EV per alert live (default: 8%)

    # Daily reports
    telegram_daily_report_enabled: bool = True
    telegram_daily_report_time: str = "22:00"  # HH:MM format

    # ============================================================
    # SENTIMENT ANALYSIS - HUGGING FACE API
    # ============================================================

    # Hugging Face API (FREE - optional key for better rate limits)
    # Get free API key at: https://huggingface.co/settings/tokens
    huggingface_api_key: str = ""  # Set via env var HUGGINGFACE_API_KEY
    huggingface_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"

    # Enable sentiment analysis from social media & news
    sentiment_enabled: bool = True
    sentiment_sources: List[str] = field(default_factory=lambda: ['news', 'twitter', 'reddit'])
    sentiment_hours_before_match: int = 48  # Hours to analyze before match

    # ============================================================
    # BLOCCO 0: API DATA ENGINE
    # ============================================================

    # API Usage Strategy
    api_daily_budget: int = 100  # API-Football daily limit
    api_reserved_monitoring: int = 30  # Per odds tracking
    api_reserved_enrichment: int = 50  # Per data collection
    api_emergency_buffer: int = 20  # Riserva

      # Cache settings
      api_cache_ttl: int = 86400  # 24 hours
      api_cache_max_entries: int = 10000

      # Match importance thresholds
      high_importance_threshold: float = 0.75  # Usa API premium
      medium_importance_threshold: float = 0.50  # Solo free API

      # Data quality thresholds
      min_data_quality: float = 0.30  # Below this = fallback
      good_data_quality: float = 0.70  # Above this = confident
      excellent_data_quality: float = 0.90  # Above this = very confident

      # StatsBomb Open Data
      statsbomb_enabled: bool = True
      statsbomb_max_matches: int = 5
      statsbomb_cache_hours: int = 6
      statsbomb_min_matches: int = 2

    # ============================================================
    # BLOCCO 1: PROBABILITY CALIBRATOR
    # ============================================================

    # Neural Network Architecture
    calibrator_hidden_layers: List[int] = field(default_factory=lambda: [64, 32, 16])
    calibrator_dropout: float = 0.2
    calibrator_activation: str = "relu"
    calibrator_output_activation: str = "sigmoid"

    # Training parameters
    calibrator_epochs: int = 100
    calibrator_batch_size: int = 32
    calibrator_learning_rate: float = 0.001
    calibrator_validation_split: float = 0.2
    calibrator_early_stopping_patience: int = 10

    # Feature configuration
    calibrator_use_api_context: bool = True
    calibrator_context_weight: float = 0.3  # How much to weight API context

    # Calibration bounds (prevent extreme adjustments)
    max_calibration_shift: float = 0.15  # Max ±15% adjustment
    min_probability: float = 0.01  # Prevent 0%
    max_probability: float = 0.99  # Prevent 100%

    # ============================================================
    # BLOCCO 2: CONFIDENCE SCORER
    # ============================================================

    # Random Forest parameters
    confidence_n_estimators: int = 100
    confidence_max_depth: int = 10
    confidence_min_samples_split: int = 5
    confidence_min_samples_leaf: int = 2

    # Confidence level thresholds
    confidence_very_high: float = 85.0  # 85-100
    confidence_high: float = 70.0       # 70-85
    confidence_medium: float = 50.0     # 50-70
    confidence_low: float = 30.0        # 30-50
    # Below 30 = very low

    # Feature weights (must sum to 1.0)
    confidence_weights: Dict[str, float] = field(default_factory=lambda: {
        "model_agreement": 0.30,      # Agreement tra modelli
        "data_completeness": 0.25,    # API data quality
        "odds_stability": 0.20,       # Volatilità quote
        "historical_accuracy": 0.15,  # Performance storica
        "api_freshness": 0.10         # Freschezza dati
    })

    # Minimum requirements
    min_confidence_to_bet: float = 50.0  # Non scommettere sotto 50

    # ============================================================
    # BLOCCO 3: VALUE DETECTOR
    # ============================================================

    # XGBoost parameters
    value_n_estimators: int = 200
    value_max_depth: int = 6
    value_learning_rate: float = 0.1
    value_subsample: float = 0.8
    value_colsample_bytree: float = 0.8

    # Value classification thresholds
    value_true_value_threshold: float = 70.0    # 70+ = TRUE VALUE
    value_uncertain_threshold: float = 40.0     # 40-70 = UNCERTAIN
    # Below 40 = TRAP

    # Expected Value thresholds
    min_ev_to_bet: float = 0.03  # Minimo +3% EV
    good_ev_threshold: float = 0.08  # +8% = good value
    excellent_ev_threshold: float = 0.15  # +15% = excellent value

    # Odds movement detection
    sharp_money_threshold: float = -0.05  # Drop di 5%+ = sharp money
    odds_drift_threshold: float = 0.05    # Rise di 5%+ = public money

    # ============================================================
    # BLOCCO 4: SMART KELLY OPTIMIZER
    # ============================================================

    # Kelly Criterion base
    kelly_default_fraction: float = 0.25  # Fractional Kelly standard
    kelly_aggressive_fraction: float = 0.35  # Per high confidence
    kelly_conservative_fraction: float = 0.15  # Per low confidence

    # Dynamic adjustment factors
    kelly_confidence_multiplier: Dict[str, float] = field(default_factory=lambda: {
        "very_high": 1.4,   # 85+ confidence → Kelly × 1.4
        "high": 1.2,        # 70-85 → Kelly × 1.2
        "medium": 1.0,      # 50-70 → Kelly × 1.0
        "low": 0.7,         # 30-50 → Kelly × 0.7
        "very_low": 0.5     # <30 → Kelly × 0.5
    })

    kelly_api_quality_multiplier: Dict[str, float] = field(default_factory=lambda: {
        "excellent": 1.2,   # API quality > 0.9
        "good": 1.0,        # API quality 0.7-0.9
        "medium": 0.8,      # API quality 0.5-0.7
        "poor": 0.6         # API quality < 0.5
    })

    # Stake limits
    min_stake_pct: float = 0.5   # Min 0.5% bankroll
    max_stake_pct: float = 5.0   # Max 5% bankroll
    absolute_min_stake: float = 5.0  # Min €5
    absolute_max_stake: float = 100.0  # Max €100

    # Correlation penalty
    correlation_penalty_enabled: bool = True
    max_correlation_exposure: float = 0.15  # Max 15% bankroll su bets correlate

    # ============================================================
    # BLOCCO 5: RISK MANAGER
    # ============================================================

    # Portfolio limits
    max_active_bets: int = 10
    max_daily_bets: int = 5
    max_same_league_exposure: float = 0.30  # Max 30% su stessa lega
    max_same_team_exposure: float = 0.20    # Max 20% su stessa squadra

    # Filter thresholds
    min_value_score_to_bet: float = 60.0
    min_confidence_to_bet: float = 50.0
    min_ev_to_bet: float = 0.03

    # Red flags (se presenti >= N, skip bet)
    max_red_flags_allowed: int = 2

    # Red flag definitions
    red_flag_injury_impact: float = 0.15  # Infortunio > 15% impatto
    red_flag_form_losses: int = 4  # 4+ sconfitte ultime 5
    red_flag_lineup_weak: float = 0.70  # Lineup < 70% strength
    red_flag_odds_suspicious: float = 0.10  # Movimento quote sospetto >10%

    # Green flags (bonus per bet)
    green_flag_sharp_money: bool = True
    green_flag_strong_form: int = 4  # 4+ vittorie ultime 5
    green_flag_high_confidence: float = 85.0

    # Bankroll protection
    max_daily_loss_pct: float = 0.10  # Stop se perdi 10% in un giorno
    stop_loss_trigger: bool = True

    # ============================================================
    # BLOCCO 6: ODDS MOVEMENT TRACKER
    # ============================================================

    # LSTM parameters
    odds_lstm_units: int = 64
    odds_lstm_layers: int = 2
    odds_lstm_dropout: float = 0.2
    odds_lookback_window: int = 24  # Ultime 24 osservazioni

    # Monitoring settings
    odds_monitoring_enabled: bool = True
    odds_check_interval_minutes: int = 30  # Check ogni 30 min
    odds_min_data_points: int = 5  # Minimo 5 osservazioni per previsione

    # Timing recommendations
    odds_bet_now_threshold: float = -0.08  # Sharp drop > 8% → BET NOW
    odds_wait_threshold: float = 0.05      # Rising > 5% → WAIT
    odds_watch_threshold: float = 0.02     # Stable ±2% → WATCH

    # Urgency levels
    urgency_high_time_hours: float = 2.0   # <2h to kickoff = HIGH
    urgency_medium_time_hours: float = 6.0 # 2-6h = MEDIUM
    # >6h = LOW

      # Volume analysis
      volume_spike_threshold: float = 2.0  # Volume 2× media = spike
      volume_sharp_threshold: float = 3.0  # Volume 3× media = sharp money

      # Chronos forecasting (time-series transformer)
      chronos_enabled: bool = True
      chronos_model_name: str = "amazon/chronos-t5-small"
      chronos_prediction_length: int = 1
      chronos_quantiles: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
      chronos_device: str = "cpu"

      # TheOddsAPI integration
      theodds_enabled: bool = True
      theodds_api_key: str = ""
      theodds_regions: str = "eu"
      theodds_markets: List[str] = field(default_factory=lambda: ["h2h"])
      theodds_primary_market: str = "h2h"
      theodds_odds_format: str = "decimal"
      theodds_date_format: str = "iso"
      theodds_auto_refresh: bool = True
      theodds_history_window_hours: int = 6
      theodds_sport_mapping: Dict[str, str] = field(default_factory=lambda: {
          "serie a": "soccer_italy_serie_a",
          "premier league": "soccer_epl",
          "la liga": "soccer_spain_la_liga",
          "bundesliga": "soccer_germany_bundesliga",
          "ligue 1": "soccer_france_ligue_one",
          "champions league": "soccer_uefa_champs_league",
          "europa league": "soccer_uefa_europa_league",
      })

    # ============================================================
    # TRAINING & VALIDATION
    # ============================================================

    # Train/validation/test split
    train_split: float = 0.70
    validation_split: float = 0.15
    test_split: float = 0.15

    # Time series validation
    use_time_series_split: bool = True
    n_splits: int = 5  # Per time series cross-validation

    # Minimum data requirements
    min_samples_to_train: int = 1000  # Minimo 1000 match per training
    min_samples_per_class: int = 50   # Minimo 50 sample per classe

    # Feature engineering
    feature_scaling: str = "standard"  # "standard", "minmax", "robust"
    handle_missing: str = "mean"       # "mean", "median", "drop"
    outlier_std_threshold: float = 3.0 # Remove outlier > 3 std

    # Model persistence
    save_models: bool = True
    model_version_control: bool = True
    keep_n_versions: int = 5  # Mantieni ultime 5 versioni

    # ============================================================
    # LOGGING & MONITORING
    # ============================================================

    # Logging levels
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    log_to_file: bool = True
    log_to_console: bool = True

    # Performance tracking
    track_predictions: bool = True
    track_api_usage: bool = True
    track_model_performance: bool = True

    # Metrics to track
    tracked_metrics: List[str] = field(default_factory=lambda: [
        "brier_score",
        "log_loss",
        "roi",
        "win_rate",
        "sharpe_ratio",
        "max_drawdown",
        "api_calls_per_day",
        "cache_hit_rate"
    ])

    # Alerts
    alert_on_poor_performance: bool = True
    alert_on_api_quota_high: bool = True
    alert_on_api_quota_pct: float = 0.80  # Alert at 80% quota

    # ============================================================
    # MLFLOW INTEGRATION
    # ============================================================

    use_mlflow: bool = True
    mlflow_tracking_uri: str = "file:./mlruns"
    mlflow_experiment_name: str = "AsianOdds_AI_System"
    mlflow_log_frequency: int = 10  # Log ogni 10 predizioni

    # ============================================================
    # UTILITIES
    # ============================================================

    # Random seed for reproducibility
    random_seed: int = 42

    # Multiprocessing
    use_multiprocessing: bool = False  # Set True se hai CPU potente
    n_jobs: int = -1  # -1 = usa tutti i core

    # GPU acceleration
    use_gpu: bool = False  # Set True se hai GPU CUDA
    gpu_device: str = "cuda:0"

    def __post_init__(self):
        """Validazione e creazione directories"""
        # Crea directories se non esistono
        for dir_path in [self.models_dir, self.cache_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Load Telegram credentials from environment if not set
        if not self.telegram_bot_token:
            self.telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        if not self.telegram_chat_id:
            self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        if not self.theodds_api_key:
            self.theodds_api_key = os.getenv("THEODDS_API_KEY", "")

        # Valida che i pesi sommino a 1.0
        weight_sum = sum(self.confidence_weights.values())
        if abs(weight_sum - 1.0) > 1e-6:  # More reasonable precision tolerance
            raise ValueError(
                f"Confidence weights must sum to 1.0, got {weight_sum}. "
                f"Weights: {self.confidence_weights}"
            )

        # Valida thresholds
        if not (0 < self.min_confidence_to_bet < 100):
            raise ValueError(f"min_confidence_to_bet must be 0-100, got {self.min_confidence_to_bet}")

        if not (0 < self.min_ev_to_bet < 1):
            raise ValueError(f"min_ev_to_bet must be 0-1, got {self.min_ev_to_bet}")

        # Valida Kelly fractions
        if not (0 < self.kelly_default_fraction < 1):
            raise ValueError(f"kelly_default_fraction must be 0-1, got {self.kelly_default_fraction}")

        # Valida splits
        split_sum = self.train_split + self.validation_split + self.test_split
        if abs(split_sum - 1.0) > 0.001:
            raise ValueError(f"Train/val/test splits must sum to 1.0, got {split_sum}")

    def to_dict(self) -> Dict:
        """Converte config in dizionario per logging"""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }

    def save_to_file(self, filepath: str):
        """Salva configurazione su file JSON"""
        import json
        with open(filepath, 'w') as f:
            # Convert Path objects to strings for JSON serialization
            config_dict = {}
            for k, v in self.to_dict().items():
                if isinstance(v, Path):
                    config_dict[k] = str(v)
                else:
                    config_dict[k] = v
            json.dump(config_dict, f, indent=2)

    @classmethod
    def load_from_file(cls, filepath: str):
        """Carica configurazione da file JSON"""
        import json
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)


# ============================================================
# DEFAULT CONFIG INSTANCE
# ============================================================

# Istanza globale di default
DEFAULT_CONFIG = AIConfig()


# ============================================================
# CONFIG PRESETS
# ============================================================

def get_conservative_config() -> AIConfig:
    """Configurazione conservativa (basso rischio)"""
    config = AIConfig()
    config.kelly_default_fraction = 0.15
    config.min_confidence_to_bet = 70.0
    config.min_value_score_to_bet = 70.0
    config.min_ev_to_bet = 0.05
    config.max_stake_pct = 3.0
    config.max_red_flags_allowed = 1
    return config


def get_aggressive_config() -> AIConfig:
    """Configurazione aggressiva (alto rischio/reward)"""
    config = AIConfig()
    config.kelly_default_fraction = 0.35
    config.min_confidence_to_bet = 40.0
    config.min_value_score_to_bet = 50.0
    config.min_ev_to_bet = 0.02
    config.max_stake_pct = 7.0
    config.max_red_flags_allowed = 3
    return config


def get_research_config() -> AIConfig:
    """Configurazione per ricerca/testing (logging verboso)"""
    config = AIConfig()
    config.log_level = "DEBUG"
    config.track_predictions = True
    config.track_api_usage = True
    config.track_model_performance = True
    config.mlflow_log_frequency = 1  # Log ogni predizione
    return config


# ============================================================
# VALIDATION UTILITIES
# ============================================================

def validate_config(config: AIConfig) -> List[str]:
    """
    Valida una configurazione e ritorna lista di warning/errori

    Returns:
        Lista di messaggi di warning (vuota se tutto ok)
    """
    warnings = []

    # Check Kelly fractions
    if config.kelly_aggressive_fraction > 0.5:
        warnings.append(
            f"Kelly aggressive fraction molto alta ({config.kelly_aggressive_fraction}). "
            "Rischio di overbetting!"
        )

    # Check thresholds consistency
    if config.min_confidence_to_bet < 30:
        warnings.append(
            f"min_confidence_to_bet molto basso ({config.min_confidence_to_bet}). "
            "Potresti scommettere su predizioni incerte."
        )

    if config.min_ev_to_bet < 0.01:
        warnings.append(
            f"min_ev_to_bet molto basso ({config.min_ev_to_bet}). "
            "Rischio di scommettere su value marginale."
        )

    # Check API budget
    total_reserved = (
        config.api_reserved_monitoring +
        config.api_reserved_enrichment +
        config.api_emergency_buffer
    )
    if total_reserved > config.api_daily_budget:
        warnings.append(
            f"API budget allocation error: reserved {total_reserved} but daily budget is {config.api_daily_budget}"
        )

    # Check stake limits
    if config.max_stake_pct > 10:
        warnings.append(
            f"max_stake_pct molto alto ({config.max_stake_pct}%). "
            "Rischio di rovina su singola bet!"
        )

    return warnings


if __name__ == "__main__":
    # Test configurazione
    print("Testing AI Configuration...")
    print("=" * 70)

    # Test default config
    config = AIConfig()
    warnings = validate_config(config)
    print(f"✅ Default config loaded")
    print(f"   Models dir: {config.models_dir}")
    print(f"   Kelly fraction: {config.kelly_default_fraction}")
    print(f"   Min confidence: {config.min_confidence_to_bet}")

    if warnings:
        print(f"\n⚠️  Warnings ({len(warnings)}):")
        for w in warnings:
            print(f"   - {w}")
    else:
        print(f"\n✅ No warnings")

    # Test presets
    print(f"\n{'=' * 70}")
    print("Testing presets...")

    conservative = get_conservative_config()
    print(f"\n📘 Conservative config:")
    print(f"   Kelly: {conservative.kelly_default_fraction}")
    print(f"   Min confidence: {conservative.min_confidence_to_bet}")

    aggressive = get_aggressive_config()
    print(f"\n📕 Aggressive config:")
    print(f"   Kelly: {aggressive.kelly_default_fraction}")
    print(f"   Min confidence: {aggressive.min_confidence_to_bet}")

    print(f"\n{'=' * 70}")
    print("✅ Configuration tests passed!")
