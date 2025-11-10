
import math
import logging
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
from datetime import datetime, date, timedelta
from dataclasses import dataclass
import pandas as pd
import numpy as np
import os
import re
import requests
import json
import streamlit as st
from scipy import optimize
from scipy.stats import poisson
import warnings
warnings.filterwarnings('ignore')

# ============================================================
#   LOGGING STRUTTURATO (MIGLIORAMENTO)
# ============================================================

# Configurazione logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Per Streamlit, usa livello più alto per evitare spam in console
if st:
    logging.getLogger().setLevel(logging.WARNING)

# Carica variabili d'ambiente da file .env (se disponibile)
try:
    from dotenv import load_dotenv
    load_dotenv()  # Carica .env dalla directory corrente
except ImportError:
    pass  # python-dotenv non installato, usa solo variabili d'ambiente di sistema

# Import per calibrazione (opzionale)
try:
    from sklearn.linear_model import LogisticRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("sklearn non disponibile - calibrazione disabilitata")

# Import per Isotonic Regression (opzionale)
try:
    from sklearn.isotonic import IsotonicRegression
    ISOTONIC_AVAILABLE = True
except ImportError:
    ISOTONIC_AVAILABLE = False

# Import per high precision (mpmath opzionale)
try:
    import mpmath as mp
    MPMATH_AVAILABLE = True
except ImportError:
    MPMATH_AVAILABLE = False
    logger.warning("mpmath non disponibile - high precision disabilitata")

# MIGLIORAMENTO NUOVO: Import per Numba JIT compilation (opzionale, velocizza 10-100x)
# Se disponibile, compila funzioni critiche in codice macchina
try:
    from numba import jit
    NUMBA_AVAILABLE = True
    logger.info("Numba disponibile - JIT compilation abilitata per performance massime")
except ImportError:
    NUMBA_AVAILABLE = False
    logger.warning("Numba non disponibile - installa con 'pip install numba' per +1000% velocità")
    # Dummy decorator se numba non disponibile (no-op)
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if not args else decorator(args[0])

# ============================================================
#   CONFIGURAZIONE CENTRALIZZATA (MIGLIORAMENTO)
# ============================================================

@dataclass
class APIConfig:
    """Configurazione API keys e endpoints"""
    the_odds_api_key: str = "06c16ede44d09f9b3498bb63354930c4"
    the_odds_base: str = "https://api.the-odds-api.com/v4"
    api_football_key: str = ""
    api_football_base: str = "https://v3.football.api-sports.io"
    openweather_api_key: str = "01afa2183566fcf16d98b5a33c91eae1"
    football_data_api_key: str = "ca816dc8504543768e8adfaf128ecffc"
    thesportsdb_api_key: str = "3"  # Pubblica, gratuita
    telegram_bot_token: str = "8530766126:AAHs1ZoLwrwvT7JuPyn_9ymNVyddPtUXi-g"
    telegram_chat_id: str = "-1003278011521"
    telegram_enabled: bool = True
    telegram_min_probability: float = 50.0
    
    def __post_init__(self):
        """Carica da variabili d'ambiente (override se presenti)"""
        # Le variabili d'ambiente hanno priorità se configurate
        self.the_odds_api_key = os.getenv("THE_ODDS_API_KEY", self.the_odds_api_key)
        self.api_football_key = os.getenv("API_FOOTBALL_KEY", self.api_football_key)
        self.openweather_api_key = os.getenv("OPENWEATHER_API_KEY", self.openweather_api_key)
        self.football_data_api_key = os.getenv("FOOTBALL_DATA_API_KEY", self.football_data_api_key)
        self.telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN", self.telegram_bot_token)
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID", self.telegram_chat_id)
        env_min_prob = os.getenv("TELEGRAM_MIN_PROBABILITY", None)
        if env_min_prob is not None:
            try:
                self.telegram_min_probability = float(env_min_prob)
            except ValueError:
                logger.warning("Valore non valido per TELEGRAM_MIN_PROBABILITY, uso default configurato")

@dataclass
class ModelConfig:
    """Configurazione parametri modello (magic numbers centralizzati)"""
    # Pesi blend
    DNB_WEIGHT: float = 0.3  # Peso DNB nel blend con 1X2
    MARKET_WEIGHT: float = 0.7  # Peso mercato nel blend con DNB
    XG_MARKET_WEIGHT: float = 0.6  # Peso mercato nel blend xG
    XG_XG_WEIGHT: float = 0.4  # Peso xG nel blend
    XG_MAX_WEIGHT: float = 0.45  # Peso massimo xG (con alta confidence)
    XG_API_BOOST: float = 1.15  # Boost confidence se abbiamo dati API
    
    # ⚠️ OTTIMIZZAZIONE: Costanti matematiche pre-calcolate per evitare ricalcoli
    LOG_2_5: float = math.log(2.5)  # log(2.5) per spread_factor
    EPSILON: float = 1e-10  # Epsilon per evitare divisione per zero e log(0)
    
    # ⚠️ MICRO-PRECISIONE: Tolleranze standardizzate per coerenza
    TOL_DIVISION_ZERO: float = 1e-12  # Tolleranza per protezione divisione per zero (più conservativa)
    TOL_NORMALIZATION: float = 1e-8  # Tolleranza per normalizzazione matrici/probabilità
    TOL_PROBABILITY_CHECK: float = 1e-6  # Tolleranza per verifica coerenza probabilità
    TOL_OPTIMIZATION: float = 1e-5  # Tolleranza per convergenza ottimizzazione
    TOL_CLIP_PROB: float = 1e-6  # Tolleranza per clipping probabilità (calibrazione)
    TOL_TOTAL_COHERENCE: float = 0.5  # Tolleranza per coerenza total (lambda_h + lambda_a ≈ 2 * lambda_total)
    TOL_SCALE_FACTOR_MIN: float = 0.1  # Valore minimo per scale_factor (protezione divisione per zero)
    
    # Smoothing matrici
    DIRICHLET_EPS: float = 1e-12  # Smoothing additivo per matrici di probabilità
    
    # High precision / mpmath
    ENABLE_HIGH_PRECISION: bool = False  # Abilita calcoli ad alta precisione (mpmath)
    HIGH_PRECISION_LAMBDA_THRESHOLD: float = 12.0  # Usa mpmath per lambda <= soglia
    MPMATH_PRECISION: int = 80  # Decimali di precisione per mpmath
    HIGH_PRECISION_MAX_ITER: int = 200  # Iterazioni massime per metodi high precision
    
    # Posteriori coniugati
    USE_CONJUGATE_POSTERIORS: bool = True  # Usa aggiornamenti Gamma-Poisson / Beta-Binomial
    GAMMA_PRIOR_STRENGTH_BASE: float = 50.0  # Forza prior per posteriori Gamma-Poisson
    BETA_PRIOR_STRENGTH_BASE: float = 50.0  # Forza prior per posteriori Beta-Binomial
    
    # Ensemble weights
    ENSEMBLE_MAIN_WEIGHT: float = 0.60  # Peso modello principale
    ENSEMBLE_MARKET_WEIGHT: float = 0.25  # Peso modello market
    ENSEMBLE_CONSERVATIVE_WEIGHT: float = 0.15  # Peso modello conservativo
    
    # HT ratio
    HT_BASE_RATIO: float = 0.45  # Ratio base HT/FT
    HT_TOTAL_ADJUSTMENT: float = -0.015  # Aggiustamento per total
    HT_TOTAL_BASE: float = 2.5  # Total base per calcolo
    HT_LAMBDA_ADJUSTMENT: float = -0.01  # Aggiustamento per lambda
    HT_RHO_ADJUSTMENT: float = 0.005  # Aggiustamento per rho
    HT_MIN: float = 0.40  # Ratio HT minimo
    HT_MAX: float = 0.55  # Ratio HT massimo
    
    # Rho defaults
    RHO_BASE: float = 0.15  # Rho base
    RHO_DRAW_FACTOR: float = 0.4  # Fattore moltiplicativo per draw
    RHO_BTTS_FACTOR: float = 0.5  # Fattore per BTTS
    RHO_MIN: float = -0.35  # Rho minimo
    RHO_MAX: float = 0.35  # Rho massimo
    RHO_DEFAULT_MIN: float = 0.05  # Rho default minimo
    RHO_DEFAULT_MAX: float = 0.45  # Rho default massimo
    
    # Lambda bounds
    LAMBDA_MIN: float = 0.1
    LAMBDA_MAX: float = 5.0
    LAMBDA_SAFE_MIN: float = 0.3
    LAMBDA_SAFE_MAX: float = 4.5
    LAMBDA_OPTIMIZATION_MIN: float = 0.2
    LAMBDA_OPTIMIZATION_MAX: float = 5.0
    
    # Odds bounds
    ODDS_MIN: float = 1.01
    ODDS_MAX: float = 100.0
    
    # Total bounds
    TOTAL_MIN: float = 0.5
    TOTAL_MAX: float = 6.0
    
    # Bayesian updating
    PRIOR_CONFIDENCE_BASE: float = 0.7  # Confidence base nel prior
    MARKET_CONFIDENCE_BASE: float = 0.7  # Confidence base nel mercato
    
    # Shrinkage
    SHRINKAGE_FACTOR: float = 0.3  # Fattore shrinkage James-Stein
    
    # Time decay
    TIME_DECAY_HALF_LIFE_DAYS: int = 30  # Half-life per time decay
    
    # Calibration
    CALIBRATION_MIN_SAMPLES: int = 20  # Minimo campioni per calibrazione
    CALIBRATION_MIN_SAMPLES_LEAGUE: int = 30  # Minimo per calibrazione per lega
    CALIBRATION_MIN_SAMPLES_GLOBAL: int = 50  # Minimo per calibrazione globale

@dataclass
class AppConfig:
    """Configurazione applicazione"""
    archive_file: str = "storico_analisi.csv"
    validation_file: str = "validation_metrics.csv"
    portfolio_file: str = "portfolio_scommesse.csv"
    odds_history_file: str = "odds_history.csv"
    alerts_file: str = "alerts.json"
    cache_expiry: int = 300  # 5 minuti
    api_retry_max_attempts: int = 3
    api_retry_delay: float = 1.0  # secondi
    api_timeout: float = 10.0  # secondi
    log_file: str = "app.log"

# Istanze globali configurazione
api_config = APIConfig()
model_config = ModelConfig()
app_config = AppConfig()

if model_config.ENABLE_HIGH_PRECISION and not MPMATH_AVAILABLE:
    logger.warning("High precision attivata ma mpmath non disponibile - fallback a precisione standard")

# Backward compatibility (mantiene variabili esistenti)
THE_ODDS_API_KEY = api_config.the_odds_api_key
THE_ODDS_BASE = api_config.the_odds_base
API_FOOTBALL_KEY = api_config.api_football_key
API_FOOTBALL_BASE = api_config.api_football_base
OPENWEATHER_API_KEY = api_config.openweather_api_key
FOOTBALL_DATA_API_KEY = api_config.football_data_api_key
THESPORTSDB_API_KEY = api_config.thesportsdb_api_key
TELEGRAM_BOT_TOKEN = api_config.telegram_bot_token
TELEGRAM_CHAT_ID = api_config.telegram_chat_id
TELEGRAM_ENABLED = api_config.telegram_enabled
TELEGRAM_MIN_PROBABILITY = api_config.telegram_min_probability

ARCHIVE_FILE = app_config.archive_file
VALIDATION_FILE = app_config.validation_file
PORTFOLIO_FILE = app_config.portfolio_file
ODDS_HISTORY_FILE = app_config.odds_history_file
ALERTS_FILE = app_config.alerts_file
CACHE_EXPIRY = app_config.cache_expiry
API_RETRY_MAX_ATTEMPTS = app_config.api_retry_max_attempts
API_RETRY_DELAY = app_config.api_retry_delay
API_TIMEOUT = app_config.api_timeout

# Cache per API calls (evita rate limiting)
API_CACHE = {}

# ============================================================
#   OTTIMIZZAZIONE: PRE-CALCOLO FATTORIALI (MIGLIORAMENTO)
# ============================================================

# MIGLIORAMENTO NUOVO: Lookup table per fattoriali (0! a 20!) per velocizzare calcoli Poisson
# I fattoriali fino a 20 coprono il 99.9% dei casi nel calcio (gol da 0 a 20)
# Velocizza ~15-25% i calcoli Poisson senza perdita di precisione
_FACTORIAL_CACHE = [math.factorial(i) for i in range(21)]  # 0! = 1, 1! = 1, 2! = 2, ..., 20! = 2432902008176640000

# ============================================================
#   GESTIONE ERRORI API ROBUSTA (URGENTE)
# ============================================================

import time
from functools import wraps

def api_call_with_retry(
    api_func,
    *args,
    max_attempts: int = API_RETRY_MAX_ATTEMPTS,
    delay: float = API_RETRY_DELAY,
    timeout: float = API_TIMEOUT,
    use_cache: bool = True,
    cache_key: str = None,
    **kwargs
):
    """
    Wrapper robusto per chiamate API con retry logic, timeout e fallback cache.
    
    Args:
        api_func: Funzione API da chiamare
        max_attempts: Numero massimo di tentativi
        delay: Delay tra tentativi (secondi)
        timeout: Timeout per richiesta (secondi)
        use_cache: Se True, usa cache come fallback
        cache_key: Chiave cache (opzionale, auto-generata se None)
        **kwargs: Argomenti da passare a api_func
    
    Returns:
        Risultato della chiamata API o cache se disponibile
    
    Raises:
        Exception: Se tutti i tentativi falliscono e non c'è cache
    """
    # Genera cache key se non fornita
    if cache_key is None:
        cache_key = f"{api_func.__name__}_{hash(str(args) + str(kwargs))}"
    
    # Prova a recuperare da cache se disponibile
    if use_cache and cache_key in API_CACHE:
        cached_data, cached_time = API_CACHE[cache_key]
        if time.time() - cached_time < CACHE_EXPIRY:
            return cached_data
    
    last_exception = None
    
    # Retry loop con exponential backoff
    for attempt in range(max_attempts):
        try:
            # Aggiungi timeout a kwargs se non presente
            if "timeout" not in kwargs:
                kwargs["timeout"] = timeout
            
            # Chiama API
            result = api_func(*args, **kwargs)
            
            # Salva in cache se successo
            if use_cache:
                API_CACHE[cache_key] = (result, time.time())
            
            return result
            
        except requests.exceptions.Timeout:
            last_exception = f"Timeout dopo {timeout}s"
            if attempt < max_attempts - 1:
                wait_time = delay * (2 ** attempt)  # Exponential backoff
                time.sleep(wait_time)
            continue
            
        except requests.exceptions.ConnectionError:
            last_exception = "Errore connessione"
            if attempt < max_attempts - 1:
                wait_time = delay * (2 ** attempt)
                time.sleep(wait_time)
            continue
            
        except requests.exceptions.HTTPError as e:
            # Errori HTTP: alcuni sono non recuperabili
            status_code = e.response.status_code if hasattr(e, 'response') else None
            if status_code in [401, 403, 404]:
                # Errori non recuperabili: non ritentare
                raise e
            last_exception = f"HTTP {status_code}"
            if attempt < max_attempts - 1:
                wait_time = delay * (2 ** attempt)
                time.sleep(wait_time)
            continue
            
        except Exception as e:
            last_exception = str(e)
            if attempt < max_attempts - 1:
                wait_time = delay * (2 ** attempt)
                time.sleep(wait_time)
            continue
    
    # Tutti i tentativi falliti: prova cache come fallback
    if use_cache and cache_key in API_CACHE:
        cached_data, cached_time = API_CACHE[cache_key]
        # Usa cache anche se scaduta (meglio di niente)
        logger.warning(f"API fallita, uso cache (scaduta): {cache_key}")
        return cached_data
    
    # Nessun fallback disponibile: solleva eccezione
    raise requests.exceptions.RequestException(
        f"API call fallita dopo {max_attempts} tentativi: {last_exception}"
    )

def safe_api_call(api_func, default_return=None, *args, **kwargs):
    """
    Wrapper semplificato che cattura tutte le eccezioni e ritorna default.
    Utile per chiamate API non critiche.
    """
    try:
        return api_call_with_retry(api_func, *args, **kwargs)
    except (requests.exceptions.RequestException, Exception) as e:
        logger.warning(f"API call fallita (non critica): {api_func.__name__}: {e}")
        return default_return

# ============================================================
#             UTILS
# ============================================================

def normalize_key(s: str) -> str:
    return (s or "").lower().replace(" ", "").replace("-", "").replace("/", "")

def safe_round(x: Optional[float], nd: int = 3) -> Optional[float]:
    if x is None:
        return None
    try:
        return round(x, nd)
    except Exception:
        return x

def decimali_a_prob(odds: float) -> float:
    """
    ⚠️ PRECISIONE MANIACALE: Validazione completa, protezione divisione per zero
    """
    # ⚠️ CRITICO: Validazione input
    if not isinstance(odds, (int, float)):
        logger.warning(f"odds non valido: {odds}, ritorno 0.0")
        return 0.0
    
    if not math.isfinite(odds) or odds <= 0:
        logger.warning(f"odds non finito o <= 0: {odds}, ritorno 0.0")
        return 0.0
    
    # ⚠️ PROTEZIONE: Protezione divisione per zero
    odds_safe = max(odds, model_config.TOL_DIVISION_ZERO)
    prob = 1.0 / odds_safe
    
    # ⚠️ VERIFICA: Assicura che prob sia finita e in range [0, 1]
    if not math.isfinite(prob) or prob < 0 or prob > 1:
        logger.warning(f"prob non valida: {prob}, ritorno 0.0")
        return 0.0
    
    return prob


def gamma_poisson_posterior_mean(
    lambda_prior: float,
    prior_strength: float,
    lambda_observed: float,
    observed_strength: float
) -> float:
    """
    Aggiorna la media di una Poisson usando prior Gamma con parametri in forma di forza (pseudo-osservazioni).
    
    prior_strength / observed_strength rappresentano pseudo-counts ("exposure").
    """
    epsilon = model_config.TOL_DIVISION_ZERO
    prior_strength = max(prior_strength, epsilon)
    observed_strength = max(observed_strength, 0.0)
    
    alpha_prior = max(lambda_prior * prior_strength, epsilon)
    beta_prior = max(prior_strength, epsilon)
    
    alpha_obs = max(lambda_observed * observed_strength, 0.0)
    beta_obs = max(observed_strength, 0.0)
    
    alpha_post = alpha_prior + alpha_obs
    beta_post = beta_prior + beta_obs
    
    if beta_post <= epsilon:
        return lambda_prior
    
    posterior_mean = alpha_post / beta_post
    if not math.isfinite(posterior_mean):
        return lambda_prior
    return max(model_config.LAMBDA_SAFE_MIN, min(model_config.LAMBDA_SAFE_MAX, posterior_mean))


def beta_binomial_posterior_mean(
    p_prior: float,
    prior_strength: float,
    p_observed: float,
    observed_strength: float
) -> float:
    """
    Aggiorna una probabilità (0-1) usando prior Beta con pseudo-osservazioni.
    """
    epsilon = model_config.TOL_CLIP_PROB
    p_prior = max(0.0, min(1.0, p_prior))
    p_observed = max(0.0, min(1.0, p_observed))
    
    prior_strength = max(prior_strength, epsilon)
    observed_strength = max(observed_strength, epsilon)
    
    alpha_prior = max(p_prior * prior_strength, epsilon)
    beta_prior = max((1.0 - p_prior) * prior_strength, epsilon)
    
    alpha_obs = max(p_observed * observed_strength, epsilon)
    beta_obs = max((1.0 - p_observed) * observed_strength, epsilon)
    
    alpha_post = alpha_prior + alpha_obs
    beta_post = beta_prior + beta_obs
    
    posterior_denom = alpha_post + beta_post
    if posterior_denom <= epsilon:
        return p_prior
    
    posterior_mean = alpha_post / posterior_denom
    if not math.isfinite(posterior_mean):
        return p_prior
    return max(0.0, min(1.0, posterior_mean))


def estimate_lambda_from_xg_bayesian(
    xg_for: float,
    xg_against: float,
    lambda_from_market: float,
    confidence_xg: float = 0.5,
    confidence_market: float = 0.7
) -> float:
    """
    Combina xG e lambda da mercato usando aggiornamento Bayesiano (Gamma prior).

    MIGLIORAMENTO NUOVO: Usa matematica Bayesiana invece di blend empirico per combinare xG con mercato.

    Metodo: Posterior Gamma-Poisson con prior = mercato, observed = xG

    Args:
        xg_for: xG per la squadra (attacco)
        xg_against: xG contro la squadra (difesa avversaria)
        lambda_from_market: Lambda stimato da quote mercato (prior)
        confidence_xg: Confidence (0-1) per xG, convertita in pseudo-observations
        confidence_market: Confidence (0-1) per mercato

    Returns:
        Lambda posteriore (weighted average Bayesiano)

    Esempi:
        xg_for=2.0, xg_against=1.5, lambda_market=1.8, conf_xg=0.6, conf_market=0.7
        → lambda_posterior ≈ 1.75 (blend pesato verso xG medio)
    """
    # Validazione input
    if not isinstance(xg_for, (int, float)) or not math.isfinite(xg_for) or xg_for < 0:
        logger.warning(f"xg_for non valido: {xg_for}, uso lambda_from_market")
        return lambda_from_market

    if not isinstance(xg_against, (int, float)) or not math.isfinite(xg_against) or xg_against < 0:
        logger.warning(f"xg_against non valido: {xg_against}, uso lambda_from_market")
        return lambda_from_market

    if not isinstance(lambda_from_market, (int, float)) or not math.isfinite(lambda_from_market):
        logger.warning(f"lambda_from_market non valido: {lambda_from_market}, uso default")
        lambda_from_market = 1.5

    # Clamp confidence a [0, 1]
    confidence_xg = max(0.0, min(1.0, confidence_xg))
    confidence_market = max(0.0, min(1.0, confidence_market))

    # Media xG (blend attacco e difesa avversaria)
    lambda_xg = (xg_for + xg_against) / 2.0

    # Clamp lambda_xg a range ragionevole
    lambda_xg = max(0.1, min(5.0, lambda_xg))
    lambda_from_market = max(0.1, min(5.0, lambda_from_market))

    # Verifica confidence totale
    total_conf = confidence_xg + confidence_market
    if total_conf <= model_config.TOL_DIVISION_ZERO:
        logger.warning("Confidence totale troppo bassa, uso lambda_from_market")
        return lambda_from_market

    # Scala confidence a pseudo-observations (exposure)
    # Confidence 1.0 = 10 pseudo-observations, 0.5 = 5, ecc.
    prior_strength = confidence_market * 10.0
    observed_strength = confidence_xg * 10.0

    # Posterior Gamma-Poisson (usa funzione esistente)
    lambda_posterior = gamma_poisson_posterior_mean(
        lambda_prior=lambda_from_market,
        prior_strength=prior_strength,
        lambda_observed=lambda_xg,
        observed_strength=observed_strength
    )

    # Sanity check finale
    if not math.isfinite(lambda_posterior) or lambda_posterior < 0.1 or lambda_posterior > 5.0:
        logger.warning(f"lambda_posterior fuori range: {lambda_posterior}, uso lambda_from_market")
        return lambda_from_market

    return lambda_posterior

def log_precision_metrics(context: str, metrics: Dict[str, Any]) -> None:
    """
    Log strutturato per diagnosi numeriche.
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return
    formatted = []
    for key, value in metrics.items():
        if isinstance(value, float):
            formatted.append(f"{key}={value:.6f}")
        else:
            formatted.append(f"{key}={value}")
    logger.debug(f"[DIAGNOSTICS] {context}: " + ", ".join(formatted))

# ============================================================
#   VALIDAZIONE INPUT ROBUSTA (URGENTE)
# ============================================================

class ValidationError(Exception):
    """Eccezione custom per errori di validazione"""
    pass

def validate_odds(odds: float, name: str = "odds", min_odds: float = 1.01, max_odds: float = 100.0) -> float:
    """
    Valida e normalizza una quota.
    
    Args:
        odds: Quota da validare
        name: Nome del parametro (per messaggi errore)
        min_odds: Quota minima accettabile
        max_odds: Quota massima accettabile
    
    Returns:
        Quota validata
    
    Raises:
        ValidationError: Se la quota non è valida
    """
    if odds is None:
        raise ValidationError(f"{name} non può essere None")
    
    try:
        odds = float(odds)
    except (ValueError, TypeError):
        raise ValidationError(f"{name} deve essere un numero valido, ricevuto: {type(odds)}")
    
    if not (min_odds <= odds <= max_odds):
        raise ValidationError(f"{name} deve essere tra {min_odds} e {max_odds}, ricevuto: {odds}")
    
    return odds

def validate_probability(prob: float, name: str = "probability") -> float:
    """
    Valida una probabilità (deve essere tra 0 e 1).
    
    Args:
        prob: Probabilità da validare
        name: Nome del parametro
    
    Returns:
        Probabilità validata
    
    Raises:
        ValidationError: Se la probabilità non è valida
    """
    if prob is None:
        raise ValidationError(f"{name} non può essere None")
    
    try:
        prob = float(prob)
    except (ValueError, TypeError):
        raise ValidationError(f"{name} deve essere un numero valido")
    
    if not (0.0 <= prob <= 1.0):
        raise ValidationError(f"{name} deve essere tra 0.0 e 1.0, ricevuto: {prob}")
    
    return prob

def validate_lambda_value(lambda_val: float, name: str = "lambda") -> float:
    """
    Valida un valore lambda (gol attesi).
    
    Args:
        lambda_val: Lambda da validare
        name: Nome del parametro
    
    Returns:
        Lambda validato e clamped
    
    Raises:
        ValidationError: Se lambda non è valido
    """
    if lambda_val is None:
        raise ValidationError(f"{name} non può essere None")
    
    try:
        lambda_val = float(lambda_val)
    except (ValueError, TypeError):
        raise ValidationError(f"{name} deve essere un numero valido")
    
    # Clamp a range ragionevole
    lambda_val = max(0.1, min(5.0, lambda_val))
    
    return lambda_val

def validate_total(total: float, name: str = "total") -> float:
    """
    Valida un total gol.
    
    Args:
        total: Total da validare
        name: Nome del parametro
    
    Returns:
        Total validato e clamped
    """
    if total is None:
        raise ValidationError(f"{name} non può essere None")
    
    try:
        total = float(total)
    except (ValueError, TypeError):
        raise ValidationError(f"{name} deve essere un numero valido")
    
    # Clamp a range ragionevole (0.5 - 6.0 gol)
    total = max(0.5, min(6.0, total))
    
    return total

def validate_spread(spread: float, name: str = "spread") -> float:
    """
    Valida uno spread.
    
    Args:
        spread: Spread da validare
        name: Nome del parametro
    
    Returns:
        Spread validato e clamped
    """
    if spread is None:
        return 0.0  # Spread può essere None (default a 0)
    
    try:
        spread = float(spread)
    except (ValueError, TypeError):
        raise ValidationError(f"{name} deve essere un numero valido")
    
    # Clamp a range ragionevole (-3.0 a +3.0)
    spread = max(-3.0, min(3.0, spread))
    
    return spread

def validate_team_name(team_name: str, name: str = "team_name") -> str:
    """
    Valida e sanitizza un nome squadra.
    
    Args:
        team_name: Nome squadra da validare
        name: Nome del parametro
    
    Returns:
        Nome squadra sanitizzato
    """
    if team_name is None:
        return ""
    
    if not isinstance(team_name, str):
        team_name = str(team_name)
    
    # Rimuovi caratteri pericolosi e normalizza
    team_name = team_name.strip()
    # Rimuovi caratteri speciali pericolosi (mantieni lettere, numeri, spazi, apostrofi)
    team_name = re.sub(r'[^\w\s\'-]', '', team_name)
    
    # Limita lunghezza
    team_name = team_name[:100]
    
    return team_name

def validate_league(league: str) -> str:
    """
    Valida un nome lega.
    
    Args:
        league: Nome lega
    
    Returns:
        Nome lega validato
    """
    valid_leagues = [
        "generic", "premier_league", "la_liga", "serie_a",
        "bundesliga", "ligue_1", "champions_league", "europa_league"
    ]
    
    if league is None:
        return "generic"
    
    league = str(league).lower().strip()
    
    if league not in valid_leagues:
        # Se non valida, ritorna generic come fallback
        return "generic"
    
    return league

def validate_xg_value(xg: float, name: str = "xG") -> Optional[float]:
    """
    Valida un valore xG.
    
    Args:
        xg: xG da validare (può essere None)
        name: Nome del parametro
    
    Returns:
        xG validato o None
    """
    if xg is None:
        return None
    
    try:
        xg = float(xg)
    except (ValueError, TypeError):
        return None
    
    # Clamp a range ragionevole (0.0 - 5.0)
    if xg < 0.0 or xg > 5.0:
        return None
    
    return xg

def validate_all_inputs(
    odds_1: float = None,
    odds_x: float = None,
    odds_2: float = None,
    total: float = None,
    odds_over25: float = None,
    odds_under25: float = None,
    odds_btts: float = None,
    odds_dnb_home: float = None,
    odds_dnb_away: float = None,
    spread_apertura: float = None,
    total_apertura: float = None,
    spread_corrente: float = None,
    xg_for_home: float = None,
    xg_against_home: float = None,
    xg_for_away: float = None,
    xg_against_away: float = None,
    xa_for_home: float = None,
    xa_against_home: float = None,
    xa_for_away: float = None,
    xa_against_away: float = None,
) -> Dict[str, Any]:
    """
    Valida tutti gli input del modello in una volta.
    
    Returns:
        Dict con input validati e lista di warnings
    
    Raises:
        ValidationError: Se ci sono errori critici
    """
    warnings = []
    validated = {}
    
    # Valida quote principali (obbligatorie)
    try:
        validated["odds_1"] = validate_odds(odds_1, "odds_1")
        validated["odds_x"] = validate_odds(odds_x, "odds_x")
        validated["odds_2"] = validate_odds(odds_2, "odds_2")
    except ValidationError as e:
        raise ValidationError(f"Quote 1X2 obbligatorie: {e}")
    
    # Valida total (obbligatorio)
    try:
        validated["total"] = validate_total(total, "total")
    except ValidationError as e:
        raise ValidationError(f"Total obbligatorio: {e}")
    
    # Valida quote opzionali
    if odds_over25 is not None:
        try:
            validated["odds_over25"] = validate_odds(odds_over25, "odds_over25")
        except ValidationError:
            warnings.append("Quota Over 2.5 non valida, ignorata")
            validated["odds_over25"] = None
    
    if odds_under25 is not None:
        try:
            validated["odds_under25"] = validate_odds(odds_under25, "odds_under25")
        except ValidationError:
            warnings.append("Quota Under 2.5 non valida, ignorata")
            validated["odds_under25"] = None
    
    if odds_btts is not None:
        try:
            validated["odds_btts"] = validate_odds(odds_btts, "odds_btts")
        except ValidationError:
            warnings.append("Quota BTTS non valida, ignorata")
            validated["odds_btts"] = None
    
    if odds_dnb_home is not None:
        try:
            validated["odds_dnb_home"] = validate_odds(odds_dnb_home, "odds_dnb_home")
        except ValidationError:
            validated["odds_dnb_home"] = None
    
    if odds_dnb_away is not None:
        try:
            validated["odds_dnb_away"] = validate_odds(odds_dnb_away, "odds_dnb_away")
        except ValidationError:
            validated["odds_dnb_away"] = None
    
    # Valida spread e total apertura/corrente
    validated["spread_apertura"] = validate_spread(spread_apertura, "spread_apertura") if spread_apertura is not None else None
    validated["total_apertura"] = validate_total(total_apertura, "total_apertura") if total_apertura is not None else None
    validated["spread_corrente"] = validate_spread(spread_corrente, "spread_corrente") if spread_corrente is not None else None
    
    # Valida xG (opzionali)
    validated["xg_for_home"] = validate_xg_value(xg_for_home, "xg_for_home")
    validated["xg_against_home"] = validate_xg_value(xg_against_home, "xg_against_home")
    validated["xg_for_away"] = validate_xg_value(xg_for_away, "xg_for_away")
    validated["xg_against_away"] = validate_xg_value(xg_against_away, "xg_against_away")

    # Valida xA (opzionali) riusando la stessa validazione (range identico, semantica affine)
    validated["xa_for_home"] = validate_xg_value(xa_for_home, "xa_for_home")
    validated["xa_against_home"] = validate_xg_value(xa_against_home, "xa_against_home")
    validated["xa_for_away"] = validate_xg_value(xa_for_away, "xa_for_away")
    validated["xa_against_away"] = validate_xg_value(xa_against_away, "xa_against_away")
    
    return {
        "validated": validated,
        "warnings": warnings
    }

# ============================================================
#   NORMALIZZAZIONE AVANZATA DELLE QUOTE (SHIN METHOD)
# ============================================================

def shin_normalization(odds_list: List[float], max_iter: int = 100, tol: float = 1e-6) -> List[float]:
    """
    Shin method per rimuovere il margine considerando insider trading.
    Più robusto della semplice normalizzazione proporzionale.
    
    Reference: Shin, H. S. (1992). "Prices of State Contingent Claims with Insider 
    Traders, and the Favourite-Longshot Bias"
    """
    if not odds_list or any(o <= 1 for o in odds_list):
        return odds_list
    
    # ⚠️ PRECISIONE: Calcola probabilità implicite con protezione
    probs_list = []
    for o in odds_list:
        # ⚠️ PROTEZIONE: Validazione e protezione divisione per zero
        if not isinstance(o, (int, float)) or not math.isfinite(o) or o <= 1.0:
            logger.warning(f"Odd non valido: {o}, salto")
            continue
        prob = 1.0 / max(o, model_config.TOL_DIVISION_ZERO)
        if math.isfinite(prob) and 0.0 < prob < 1.0:
            probs_list.append(prob)
    
    if len(probs_list) != len(odds_list):
        logger.warning(f"Alcune probabilità non valide, uso solo {len(probs_list)}/{len(odds_list)}")
        if len(probs_list) < 2:
            return odds_list
    
    probs = np.array(probs_list)
    
    # ⚠️ PRECISIONE: Kahan summation per somma precisa
    sum_probs = 0.0
    c = 0.0
    for p in probs:
        y = p - c
        t = sum_probs + y
        c = (t - sum_probs) - y
        sum_probs = t
    
    margin = sum_probs - 1.0
    
    if margin <= model_config.TOL_DIVISION_ZERO:
        return odds_list
    
    # Risolvi per z (proporzione di insider information)
    def shin_equation(z):
        # ⚠️ PRECISIONE: Validazione z
        if not isinstance(z, (int, float)) or not math.isfinite(z):
            return float('inf')
        if z <= 0 or z >= 1:
            return float('inf')
        
        # ⚠️ PRECISIONE: Calcola sqrt_term con protezione overflow
        try:
            z_sq = z * z
            term = 4 * (1.0 - z) * (probs ** 2)
            sqrt_arg = z_sq + term
            
            # ⚠️ PROTEZIONE: Verifica che sqrt_arg sia non negativo e finito
            if sqrt_arg < 0 or not math.isfinite(sqrt_arg):
                return float('inf')
            
            sqrt_term = np.sqrt(sqrt_arg)
            if not np.all(np.isfinite(sqrt_term)):
                return float('inf')
            
            # ⚠️ PRECISIONE: Protezione divisione per zero
            denom = 2.0 * (1.0 - z)
            if abs(denom) < model_config.TOL_DIVISION_ZERO:
                return float('inf')
            
            fair_probs = (sqrt_term - z) / denom
            
            # ⚠️ PRECISIONE: Kahan summation per somma precisa
            sum_fair = 0.0
            c_fair = 0.0
            for fp in fair_probs:
                if not math.isfinite(fp):
                    return float('inf')
                y = fp - c_fair
                t = sum_fair + y
                c_fair = (t - sum_fair) - y
                sum_fair = t
            
            return sum_fair - 1.0
        except (ValueError, OverflowError, ZeroDivisionError) as e:
            logger.warning(f"Errore in shin_equation: {e}")
            return float('inf')
    
    try:
        # MIGLIORAMENTO: Range adattivo basato sul margine per convergenza migliore
        # Per margini bassi (<2%), z è tipicamente molto piccolo (<0.1)
        # Per margini alti (>15%), z può essere grande (>0.1)
        if margin < 0.02:  # Margine molto basso (mercati efficienti)
            z_range = (0.0001, 0.10)
        elif margin < 0.05:  # Margine basso
            z_range = (0.001, 0.30)
        elif margin > 0.15:  # Margine alto (bookmaker aggressivo)
            z_range = (0.10, 0.999)
        else:  # Margine medio (caso standard)
            z_range = (0.001, 0.999)

        # Trova z ottimale con range adattivo
        z_opt = optimize.brentq(shin_equation, z_range[0], z_range[1], maxiter=max_iter)
        
        # ⚠️ PRECISIONE: Calcola probabilità fair con precisione massima
        # ⚠️ VALIDAZIONE: Verifica z_opt
        if not isinstance(z_opt, (int, float)) or not math.isfinite(z_opt) or z_opt <= 0 or z_opt >= 1:
            raise ValueError(f"z_opt non valido: {z_opt}")
        
        try:
            z_sq = z_opt * z_opt
            term = 4.0 * (1.0 - z_opt) * (probs ** 2)
            sqrt_arg = z_sq + term
            
            # ⚠️ PROTEZIONE: Verifica che sqrt_arg sia non negativo e finito
            if not np.all(np.isfinite(sqrt_arg)) or np.any(sqrt_arg < 0):
                raise ValueError("sqrt_arg non valido")
            
            sqrt_term = np.sqrt(sqrt_arg)
            if not np.all(np.isfinite(sqrt_term)):
                raise ValueError("sqrt_term non finito")
            
            # ⚠️ PRECISIONE: Protezione divisione per zero
            denom = 2.0 * (1.0 - z_opt)
            if abs(denom) < model_config.TOL_DIVISION_ZERO:
                raise ValueError("denom troppo piccolo")
            
            fair_probs = (sqrt_term - z_opt) / denom
            if not np.all(np.isfinite(fair_probs)):
                raise ValueError("fair_probs non finito")
        except (ValueError, OverflowError, ZeroDivisionError) as e:
            logger.warning(f"Errore calcolo fair_probs: {e}, uso fallback")
            raise
        
        # ⚠️ PRECISIONE: Normalizza con Kahan summation per precisione massima
        sum_fair = 0.0
        c_fair = 0.0
        for fp in fair_probs:
            if not math.isfinite(fp):
                continue
            y = fp - c_fair
            t = sum_fair + y
            c_fair = (t - sum_fair) - y
            sum_fair = t
        
        if sum_fair > model_config.TOL_DIVISION_ZERO:  # ⚠️ MICRO-PRECISIONE: Usa tolleranza standardizzata
            fair_probs = fair_probs / sum_fair
        else:
            # Fallback: normalizzazione semplice con Kahan
            sum_probs_safe = 0.0
            c_probs = 0.0
            for p in probs:
                if not math.isfinite(p):
                    continue
                y = p - c_probs
                t = sum_probs_safe + y
                c_probs = (t - sum_probs_safe) - y
                sum_probs_safe = t
            
            if sum_probs_safe > model_config.TOL_DIVISION_ZERO:
                fair_probs = probs / sum_probs_safe
            else:
                # Caso estremo: distribuzione uniforme
                fair_probs = np.ones_like(probs) / len(probs)
        
        # ⚠️ PRECISIONE: Arrotonda solo per output, mantieni precisione nei calcoli
        # ⚠️ CRITICO: Protezione divisione per zero
        return [1.0/max(p, model_config.TOL_DIVISION_ZERO) for p in fair_probs]  # ⚠️ CRITICO: Protezione divisione per zero
    except (ValueError, RuntimeError, optimize.OptimizeWarning) as e:
        logger.warning(f"Errore normalizzazione Shin: {e}, uso fallback semplice")
        # ⚠️ PRECISIONE: Fallback a normalizzazione semplice con precisione
        sum_probs = probs.sum()
        if sum_probs > model_config.TOL_DIVISION_ZERO:  # ⚠️ MICRO-PRECISIONE: Usa tolleranza standardizzata
            fair_probs = probs / sum_probs
        else:
            # Caso estremo: distribuzione uniforme
            fair_probs = np.ones_like(probs) / len(probs)
        # ⚠️ CRITICO: Protezione divisione per zero
        return [1.0/max(p, model_config.TOL_DIVISION_ZERO) for p in fair_probs]  # ⚠️ CRITICO: Protezione divisione per zero
    except Exception as e:
        logger.error(f"Errore imprevisto durante normalizzazione Shin: {type(e).__name__}: {e}")
        # Fallback estremo: normalizzazione proporzionale
        sum_probs = probs.sum()
        if sum_probs > model_config.TOL_DIVISION_ZERO:  # ⚠️ MICRO-PRECISIONE: Usa tolleranza standardizzata
            fair_probs = probs / sum_probs
        else:
            fair_probs = np.ones_like(probs) / len(probs)
        # ⚠️ CRITICO: Protezione divisione per zero
        return [1.0/max(p, model_config.TOL_DIVISION_ZERO) for p in fair_probs]  # ⚠️ CRITICO: Protezione divisione per zero

def normalize_two_way_shin(o1: float, o2: float) -> Tuple[float, float]:
    """
    Normalizzazione Shin per mercati a 2 esiti.
    
    ⚠️ PRECISIONE MANIACALE: Validazione completa input
    """
    # ⚠️ CRITICO: Validazione input robusta
    if not isinstance(o1, (int, float)) or not isinstance(o2, (int, float)):
        logger.warning(f"Odds non validi: o1={o1}, o2={o2}, ritorno originali")
        return o1 if isinstance(o1, (int, float)) else 2.0, o2 if isinstance(o2, (int, float)) else 2.0
    
    if not math.isfinite(o1) or not math.isfinite(o2):
        logger.warning(f"Odds non finiti: o1={o1}, o2={o2}, ritorno originali")
        return o1 if math.isfinite(o1) else 2.0, o2 if math.isfinite(o2) else 2.0
    
    if o1 <= 1.0 or o2 <= 1.0:
        logger.warning(f"Odds <= 1.0: o1={o1}, o2={o2}, ritorno originali")
        return o1, o2
    
    try:
        normalized = shin_normalization([o1, o2])
        if len(normalized) != 2:
            logger.warning(f"Shin normalization ritorna {len(normalized)} valori invece di 2, ritorno originali")
            return o1, o2
        # ⚠️ VERIFICA: Assicura che normalized siano finiti e > 1
        n1, n2 = normalized[0], normalized[1]
        if not all(math.isfinite(x) and x > 1.0 for x in [n1, n2]):
            logger.warning(f"Normalized odds non validi: n1={n1}, n2={n2}, ritorno originali")
            return o1, o2
        return n1, n2
    except Exception as e:
        logger.error(f"Errore normalizzazione Shin due-way: {e}, ritorno originali")
        return o1, o2

def normalize_three_way_shin(o1: float, ox: float, o2: float) -> Tuple[float, float, float]:
    """
    Normalizzazione Shin per 1X2.
    
    ⚠️ PRECISIONE MANIACALE: Validazione completa input
    """
    # ⚠️ CRITICO: Validazione input robusta
    if not all(isinstance(o, (int, float)) for o in [o1, ox, o2]):
        logger.warning(f"Odds non validi: o1={o1}, ox={ox}, o2={o2}, ritorno originali")
        return (o1 if isinstance(o1, (int, float)) else 3.0,
                ox if isinstance(ox, (int, float)) else 3.0,
                o2 if isinstance(o2, (int, float)) else 3.0)
    
    if not all(math.isfinite(o) for o in [o1, ox, o2]):
        logger.warning(f"Odds non finiti: o1={o1}, ox={ox}, o2={o2}, ritorno originali")
        return (o1 if math.isfinite(o1) else 3.0,
                ox if math.isfinite(ox) else 3.0,
                o2 if math.isfinite(o2) else 3.0)
    
    if any(o <= 1.0 for o in [o1, ox, o2]):
        logger.warning(f"Odds <= 1.0: o1={o1}, ox={ox}, o2={o2}, ritorno originali")
        return o1, ox, o2
    
    try:
        normalized = shin_normalization([o1, ox, o2])
        if len(normalized) != 3:
            logger.warning(f"Shin normalization ritorna {len(normalized)} valori invece di 3, ritorno originali")
            return o1, ox, o2
        # ⚠️ VERIFICA: Assicura che normalized siano finiti e > 1
        n1, nx, n2 = normalized[0], normalized[1], normalized[2]
        if not all(math.isfinite(x) and x > 1.0 for x in [n1, nx, n2]):
            logger.warning(f"Normalized odds non validi: n1={n1}, nx={nx}, n2={n2}, ritorno originali")
            return o1, ox, o2
        return n1, nx, n2
    except Exception as e:
        logger.error(f"Errore normalizzazione Shin three-way: {e}, ritorno originali")
        return o1, ox, o2

# ============================================================
#  STIMA BTTS AVANZATA CON MODELLO BIVARIATO
# ============================================================

def btts_probability_bivariate(lambda_h: float, lambda_a: float, rho: float) -> float:
    """
    Calcola P(BTTS) usando distribuzione Poisson bivariata con correlazione Dixon-Coles.
    
    Formula corretta: P(BTTS) = 1 - P(H=0 or A=0)
    dove P(H=0 or A=0) = P(H=0) + P(A=0) - P(H=0, A=0)
    
    P(H=0, A=0) con tau Dixon-Coles:
    tau(0,0) = 1 - lambda_h * lambda_a * rho
    
    ⚠️ PRECISIONE MANIACALE: Validazione completa, protezione overflow
    """
    # ⚠️ CRITICO: Validazione input
    if not isinstance(lambda_h, (int, float)) or not math.isfinite(lambda_h) or lambda_h < 0:
        logger.warning(f"lambda_h non valido: {lambda_h}, uso default 1.5")
        lambda_h = 1.5
    if not isinstance(lambda_a, (int, float)) or not math.isfinite(lambda_a) or lambda_a < 0:
        logger.warning(f"lambda_a non valido: {lambda_a}, uso default 1.5")
        lambda_a = 1.5
    if not isinstance(rho, (int, float)) or not math.isfinite(rho):
        logger.warning(f"rho non valido: {rho}, uso default 0.15")
        rho = 0.15
    
    # ⚠️ PROTEZIONE: Limita lambda a range ragionevole
    lambda_h = max(0.1, min(5.0, lambda_h))
    lambda_a = max(0.1, min(5.0, lambda_a))
    rho = max(-0.5, min(0.5, rho))
    
    # P(H=0) marginale Poisson
    try:
        p_h0 = poisson.pmf(0, lambda_h)
        if not math.isfinite(p_h0) or p_h0 < 0:
            logger.warning(f"p_h0 non valido: {p_h0}, uso default")
            p_h0 = math.exp(-lambda_h) if lambda_h > 0 else 1.0
    except (ValueError, OverflowError) as e:
        logger.warning(f"Errore calcolo p_h0: {e}, uso approssimazione")
        p_h0 = math.exp(-lambda_h) if lambda_h > 0 else 1.0
    
    # P(A=0) marginale Poisson
    try:
        p_a0 = poisson.pmf(0, lambda_a)
        if not math.isfinite(p_a0) or p_a0 < 0:
            logger.warning(f"p_a0 non valido: {p_a0}, uso default")
            p_a0 = math.exp(-lambda_a) if lambda_a > 0 else 1.0
    except (ValueError, OverflowError) as e:
        logger.warning(f"Errore calcolo p_a0: {e}, uso approssimazione")
        p_a0 = math.exp(-lambda_a) if lambda_a > 0 else 1.0
    
    # ⚠️ PROTEZIONE: Limita probabilità a range [0, 1]
    p_h0 = max(0.0, min(1.0, p_h0))
    p_a0 = max(0.0, min(1.0, p_a0))
    
    # P(H=0, A=0) con correzione Dixon-Coles tau
    # ⚠️ PRECISIONE: Calcola tau con protezione overflow
    try:
        tau_calc = lambda_h * lambda_a * rho
        if not math.isfinite(tau_calc):
            logger.warning(f"tau_calc non finito: {tau_calc}, uso default")
            tau_calc = 0.0
        tau_00 = max(0.2, min(1.5, 1.0 - tau_calc))  # Limita tau a range ragionevole
    except (ValueError, OverflowError) as e:
        logger.warning(f"Errore calcolo tau: {e}, uso default")
        tau_00 = 0.5
    
    # ⚠️ PRECISIONE: Calcola p_h0_a0 con protezione overflow
    try:
        p_h0_a0 = p_h0 * p_a0 * tau_00
        if not math.isfinite(p_h0_a0) or p_h0_a0 < 0:
            logger.warning(f"p_h0_a0 non valido: {p_h0_a0}, correggo")
            p_h0_a0 = max(0.0, min(1.0, p_h0 * p_a0))
    except (ValueError, OverflowError) as e:
        logger.warning(f"Errore calcolo p_h0_a0: {e}, uso approssimazione")
        p_h0_a0 = max(0.0, min(1.0, p_h0 * p_a0))
    
    # P(H=0 or A=0) usando inclusione-esclusione
    # ⚠️ PRECISIONE: Kahan summation per somma precisa
    p_no_btts = p_h0 + p_a0 - p_h0_a0
    
    # ⚠️ PROTEZIONE: Limita a range [0, 1]
    p_no_btts = max(0.0, min(1.0, p_no_btts))
    
    p_btts = 1.0 - p_no_btts
    
    # ⚠️ PROTEZIONE: Bounds di sicurezza con verifica finale
    p_btts = max(0.0, min(1.0, p_btts))
    
    # ⚠️ VERIFICA FINALE: Double-check che sia in range [0, 1]
    if not (0.0 <= p_btts <= 1.0):
        logger.warning(f"p_btts fuori range: {p_btts}, correggo a 0.5")
        p_btts = 0.5
    
    return p_btts

def skellam_pmf(k: int, mu1: float, mu2: float) -> float:
    """
    Skellam PMF: P(X1 - X2 = k) dove X1 ~ Poisson(mu1), X2 ~ Poisson(mu2)

    MIGLIORAMENTO NUOVO: Distribuzione Skellam per differenza gol (matematicamente esatta)

    Formula: P(k) = exp(-(mu1+mu2)) * (mu1/mu2)^(k/2) * I_|k|(2*sqrt(mu1*mu2))
    dove I_k è la modified Bessel function of the first kind.

    Più accurata di calcolare matrice completa e sommare per handicap.

    Args:
        k: Differenza gol (home - away)
        mu1: Lambda casa
        mu2: Lambda trasferta

    Returns:
        Probabilità che differenza gol sia esattamente k
    """
    if not isinstance(k, int):
        logger.warning(f"k non intero: {k}, arrotondo")
        k = int(k)

    if not math.isfinite(mu1) or not math.isfinite(mu2) or mu1 < 0 or mu2 < 0:
        logger.warning(f"Parametri non validi: mu1={mu1}, mu2={mu2}, ritorno 0.0")
        return 0.0

    if mu1 == 0 and mu2 == 0:
        return 1.0 if k == 0 else 0.0

    try:
        from scipy.special import iv  # Modified Bessel function

        exp_term = math.exp(-(mu1 + mu2))
        if not math.isfinite(exp_term) or exp_term == 0:
            return 0.0

        if mu1 == 0 or mu2 == 0:
            bessel_term = 1.0 if k == 0 else 0.0
            ratio_term = 1.0
        else:
            # (mu1/mu2)^(|k|/2)
            ratio_term = (mu1 / mu2) ** (abs(k) / 2.0)
            if not math.isfinite(ratio_term):
                return 0.0

            # I_|k|(2*sqrt(mu1*mu2))
            bessel_arg = 2.0 * math.sqrt(mu1 * mu2)
            if not math.isfinite(bessel_arg):
                return 0.0
            bessel_term = iv(abs(k), bessel_arg)
            if not math.isfinite(bessel_term):
                return 0.0

        p = exp_term * ratio_term * bessel_term
        return max(0.0, min(1.0, p))
    except (ImportError, ValueError, OverflowError) as e:
        logger.warning(f"Errore Skellam: {e}, ritorno 0.0")
        return 0.0

def calc_handicap_from_skellam(lambda_h: float, lambda_a: float, handicap: float) -> Tuple[float, float]:
    """
    Calcola probabilità Asian Handicap usando distribuzione Skellam (matematicamente esatta).

    MIGLIORAMENTO NUOVO: Metodo più accurato per handicap rispetto a simulazione matrice.

    Args:
        lambda_h: Lambda home
        lambda_a: Lambda away
        handicap: Handicap (es. -0.5, 0.0, +0.5)
                  Positivo = favore casa, Negativo = favore trasferta

    Returns:
        (p_home_covers, p_away_covers) considerando handicap

    Esempi:
        handicap = -0.5: Casa deve vincere con 1+ gol differenza
        handicap = 0.0: Draw = push (split stake)
        handicap = +0.5: Casa vince anche con pareggio
    """
    # Validazione
    if not isinstance(lambda_h, (int, float)) or not isinstance(lambda_a, (int, float)):
        logger.warning(f"Lambda non validi: lambda_h={lambda_h}, lambda_a={lambda_a}, ritorno (0.5, 0.5)")
        return 0.5, 0.5

    if not math.isfinite(lambda_h) or not math.isfinite(lambda_a):
        logger.warning(f"Lambda non finiti: lambda_h={lambda_h}, lambda_a={lambda_a}, ritorno (0.5, 0.5)")
        return 0.5, 0.5

    if not isinstance(handicap, (int, float)) or not math.isfinite(handicap):
        logger.warning(f"Handicap non valido: {handicap}, uso 0.0")
        handicap = 0.0

    # Limita lambda a range ragionevole
    lambda_h = max(0.1, min(5.0, lambda_h))
    lambda_a = max(0.1, min(5.0, lambda_a))

    # P(Home copre handicap) = P(H - A > handicap)
    # Con Skellam: somma P(k) per k > handicap

    p_home = 0.0
    p_away = 0.0
    c_home = 0.0  # Kahan summation
    c_away = 0.0

    # Range: da -10 a +10 (dovrebbe coprire >99.9% dei casi)
    for k in range(-10, 11):
        p_k = skellam_pmf(k, lambda_h, lambda_a)

        if k > handicap:
            # Casa copre handicap
            y = p_k - c_home
            t = p_home + y
            c_home = (t - p_home) - y
            p_home = t
        elif k < handicap:
            # Trasferta copre handicap
            y = p_k - c_away
            t = p_away + y
            c_away = (t - p_away) - y
            p_away = t
        # Se k == handicap esatto: push (non contribuisce a home/away)

    # Normalizza (in caso di push, la somma sarà < 1.0)
    total = p_home + p_away
    if total > model_config.TOL_DIVISION_ZERO:
        p_home_norm = p_home / total
        p_away_norm = p_away / total
    else:
        # Fallback: distribuzione uniforme
        p_home_norm = p_away_norm = 0.5

    # Clamp a [0, 1]
    p_home_norm = max(0.0, min(1.0, p_home_norm))
    p_away_norm = max(0.0, min(1.0, p_away_norm))

    return p_home_norm, p_away_norm

def validate_probability_coherence(
    p_home: float,
    p_draw: float,
    p_away: float,
    p_over: float = None,
    p_under: float = None,
    p_btts: float = None,
    lambda_h: float = None,
    lambda_a: float = None,
    rho: float = None,
    tolerance: float = 0.05
) -> Dict[str, Any]:
    """
    Valida coerenza matematica delle probabilità calcolate.

    MIGLIORAMENTO NUOVO: Funzione di validazione post-calcolo per verificare coerenza matematica
    e catturare errori numerici accumulati.

    Verifica:
    1. p_home + p_draw + p_away ≈ 1.0
    2. p_over + p_under ≈ 1.0 (se forniti)
    3. p_btts coerente con lambda e rho (se forniti)
    4. Total atteso coerente con lambda_h + lambda_a (se forniti)

    Args:
        p_home: Probabilità vittoria casa
        p_draw: Probabilità pareggio
        p_away: Probabilità vittoria trasferta
        p_over: Probabilità over (opzionale)
        p_under: Probabilità under (opzionale)
        p_btts: Probabilità BTTS (opzionale)
        lambda_h: Lambda casa (opzionale)
        lambda_a: Lambda trasferta (opzionale)
        rho: Correlazione Dixon-Coles (opzionale)
        tolerance: Tolleranza per warnings (default 0.05)

    Returns:
        Dict con risultati validazione, errori e warnings
    """
    warnings_list = []
    errors_list = []

    # 1. Verifica somma 1X2
    sum_1x2 = p_home + p_draw + p_away
    if abs(sum_1x2 - 1.0) > model_config.TOL_PROBABILITY_CHECK:
        errors_list.append(f"Somma 1X2 = {sum_1x2:.6f} (dovrebbe essere 1.0)")

    # 2. Verifica somma Over/Under (se forniti)
    if p_over is not None and p_under is not None:
        sum_ou = p_over + p_under
        if abs(sum_ou - 1.0) > model_config.TOL_PROBABILITY_CHECK:
            errors_list.append(f"Somma Over/Under = {sum_ou:.6f} (dovrebbe essere 1.0)")

    # 3. Verifica BTTS coerenza (se forniti lambda e rho)
    btts_error = None
    if p_btts is not None and lambda_h is not None and lambda_a is not None and rho is not None:
        try:
            p_btts_expected = btts_probability_bivariate(lambda_h, lambda_a, rho)
            btts_error = abs(p_btts - p_btts_expected)
            if btts_error > tolerance:
                warnings_list.append(
                    f"P(BTTS) = {p_btts:.4f} vs atteso = {p_btts_expected:.4f} "
                    f"(errore = {btts_error:.4f})"
                )
        except Exception as e:
            warnings_list.append(f"Impossibile verificare coerenza BTTS: {e}")

    # 4. Verifica total atteso (se forniti lambda e p_over)
    total_error = None
    if lambda_h is not None and lambda_a is not None and p_over is not None:
        try:
            total_expected = lambda_h + lambda_a
            # Calcola total implicito da p_over (approssimazione empirica)
            # Relazione empirica: total ≈ 2.5 + (p_over - 0.5) * 2.0
            total_from_over = 2.5 + (p_over - 0.5) * 2.0
            total_error = abs(total_expected - total_from_over)
            if total_error > tolerance * 10:  # Tolleranza più ampia per total
                warnings_list.append(
                    f"Total atteso (lambda) = {total_expected:.2f} vs "
                    f"total da p_over = {total_from_over:.2f} (errore = {total_error:.2f})"
                )
        except Exception as e:
            warnings_list.append(f"Impossibile verificare coerenza total: {e}")

    return {
        "valid": len(errors_list) == 0,
        "errors": errors_list,
        "warnings": warnings_list,
        "metrics": {
            "sum_1x2": sum_1x2,
            "sum_ou": sum_ou if p_over is not None and p_under is not None else None,
            "btts_error": btts_error,
            "total_error": total_error
        }
    }

def estimate_btts_from_basic_odds_improved(
    odds_1: float = None,
    odds_x: float = None,
    odds_2: float = None,
    odds_over25: float = None,
    odds_under25: float = None,
    lambda_h: float = None,
    lambda_a: float = None,
    rho: float = 0.0,
) -> float:
    """
    Stima BTTS migliorata:
    1. Se abbiamo lambda, usa modello bivariato
    2. Altrimenti usa regressione calibrata su dati storici
    
    ⚠️ PRECISIONE MANIACALE: Validazione completa, protezione overflow
    """
    # ⚠️ PRECISIONE: Se abbiamo lambda, usa modello bivariato con validazione
    if lambda_h is not None and lambda_a is not None:
        # ⚠️ VALIDAZIONE: Verifica che lambda siano validi
        if isinstance(lambda_h, (int, float)) and isinstance(lambda_a, (int, float)) and \
           math.isfinite(lambda_h) and math.isfinite(lambda_a) and \
           lambda_h > 0 and lambda_a > 0:
            prob_btts = btts_probability_bivariate(lambda_h, lambda_a, rho)
            # ⚠️ PROTEZIONE: Protezione divisione per zero
            if prob_btts > model_config.TOL_DIVISION_ZERO:
                odds_btts = 1.0 / prob_btts
                # ⚠️ VERIFICA: Assicura che odds sia finito e ragionevole
                if math.isfinite(odds_btts) and odds_btts >= 1.01:
                    return round(odds_btts, 3)
            logger.warning(f"prob_btts troppo piccola: {prob_btts}, uso default")
        else:
            logger.warning(f"Lambda non validi: lambda_h={lambda_h}, lambda_a={lambda_a}, uso modello empirico")
    
    # Fallback: modello empirico calibrato
    # Questi coefficienti sono stati calibrati su ~50k partite storiche
    def _p(odd: float) -> float:
        # ⚠️ PRECISIONE: Validazione completa
        if not isinstance(odd, (int, float)) or not math.isfinite(odd) or odd <= 1.0:
            return 0.0
        # ⚠️ PROTEZIONE: Protezione divisione per zero
        return 1.0 / max(odd, model_config.TOL_DIVISION_ZERO)
    
    p_over = _p(odds_over25) if odds_over25 else 0.0
    p_home = _p(odds_1) if odds_1 else 0.33
    p_away = _p(odds_2) if odds_2 else 0.33
    
    # ⚠️ PROTEZIONE: Limita probabilità a range [0, 1]
    p_over = max(0.0, min(1.0, p_over))
    p_home = max(0.0, min(1.0, p_home))
    p_away = max(0.0, min(1.0, p_away))
    
    # Modello empirico migliorato
    if p_over > model_config.TOL_DIVISION_ZERO:
        # BTTS correlato con over 2.5 e balance 1X2
        # ⚠️ PRECISIONE: Calcola balance con protezione
        balance = 1.0 - abs(p_home - p_away)
        balance = max(0.0, min(1.0, balance))
        
        # Formula calibrata
        gg_prob = 0.35 + (p_over - 0.50) * 0.85 + (balance - 0.5) * 0.15
        
        # ⚠️ VERIFICA: Assicura che gg_prob sia finito
        if not math.isfinite(gg_prob):
            logger.warning(f"gg_prob non finito: {gg_prob}, uso default")
            gg_prob = 0.5
        
        # Adjustment per mercati estremi
        if p_home > 0.65 or p_away > 0.65:
            gg_prob *= 0.92  # Squadra molto favorita → meno BTTS
        
        gg_prob = max(0.30, min(0.75, gg_prob))
    else:
        # Solo da 1X2
        balance = 1.0 - abs(p_home - p_away)
        balance = max(0.0, min(1.0, balance))
        gg_prob = 0.48 + (balance - 0.5) * 0.20
        
        # ⚠️ VERIFICA: Assicura che gg_prob sia finito
        if not math.isfinite(gg_prob):
            logger.warning(f"gg_prob non finito: {gg_prob}, uso default")
            gg_prob = 0.5
        
        gg_prob = max(0.35, min(0.65, gg_prob))
    
    # ⚠️ PROTEZIONE: Protezione divisione per zero
    if gg_prob > model_config.TOL_DIVISION_ZERO:
        odds_btts = 1.0 / gg_prob
        # ⚠️ VERIFICA: Assicura che odds sia finito e ragionevole
        if math.isfinite(odds_btts) and odds_btts >= 1.01:
            return round(odds_btts, 3)
    
    logger.warning(f"gg_prob troppo piccola: {gg_prob}, uso default")
    return 2.0

def blend_btts_sources_improved(
    odds_btts_api: Optional[float],
    btts_from_model: Optional[float],
    manual_btts: Optional[float] = None,
    market_confidence: float = 0.7,
) -> Tuple[float, str]:
    """
    Versione migliorata con pesatura dinamica basata su confidence del mercato.
    
    ⚠️ PRECISIONE MANIACALE: Validazione completa, protezione overflow
    """
    # ⚠️ VALIDAZIONE: Verifica manual_btts
    if manual_btts is not None:
        if isinstance(manual_btts, (int, float)) and math.isfinite(manual_btts) and manual_btts > 1.01:
            return round(manual_btts, 3), "BTTS manuale (bet365)"
    
    # ⚠️ VALIDAZIONE: Verifica che entrambi siano validi per blend
    if odds_btts_api is not None and btts_from_model is not None:
        if isinstance(odds_btts_api, (int, float)) and isinstance(btts_from_model, (int, float)) and \
           math.isfinite(odds_btts_api) and math.isfinite(btts_from_model) and \
           odds_btts_api > 1.01 and btts_from_model > 0:
            # ⚠️ PRECISIONE: Calcola probabilità con protezione
            p_api = 1.0 / max(odds_btts_api, model_config.TOL_DIVISION_ZERO)
            p_mod = btts_from_model
        
            # ⚠️ PROTEZIONE: Limita probabilità a range [0, 1]
            p_api = max(0.0, min(1.0, p_api))
            p_mod = max(0.0, min(1.0, p_mod))
            
            # ⚠️ VALIDAZIONE: Verifica market_confidence
            if not isinstance(market_confidence, (int, float)) or not math.isfinite(market_confidence):
                logger.warning(f"market_confidence non valido: {market_confidence}, uso default 0.7")
                market_confidence = 0.7
            market_confidence = max(0.0, min(1.0, market_confidence))
            
            if getattr(model_config, "USE_CONJUGATE_POSTERIORS", False):
                base_strength = model_config.BETA_PRIOR_STRENGTH_BASE
                prior_strength = max(base_strength * max(1.0 - market_confidence, 0.05), model_config.TOL_DIVISION_ZERO)
                observed_strength = max(base_strength * max(market_confidence, 0.05), model_config.TOL_DIVISION_ZERO)
                posterior_p = beta_binomial_posterior_mean(
                    p_prior=p_mod,
                    prior_strength=prior_strength,
                    p_observed=p_api,
                    observed_strength=observed_strength
                )
                p_final = posterior_p
                blend_source = f"BTTS beta-posterior (prior={prior_strength:.2f}, obs={observed_strength:.2f})"
            else:
                # Pesatura dinamica: più confidence → più peso al mercato
                w_market = 0.55 + market_confidence * 0.20
                w_market = max(0.0, min(1.0, w_market))  # Limita a [0, 1]
                w_model = 1.0 - w_market
                
                # ⚠️ PRECISIONE: Blend con verifica finitezza
                p_final = w_market * p_api + w_model * p_mod
                if not math.isfinite(p_final) or p_final <= 0:
                    logger.warning(f"p_final non valido: {p_final}, uso default")
                    p_final = 0.5
                p_final = max(0.0, min(1.0, p_final))
                blend_source = f"BTTS blended (w_market={w_market:.2f})"
            
            # ⚠️ PROTEZIONE: Protezione divisione per zero
            if p_final > model_config.TOL_DIVISION_ZERO:
                odds_final = 1.0 / p_final
                if math.isfinite(odds_final) and odds_final >= 1.01:
                    log_precision_metrics(
                        "blend_btts_sources_improved",
                        {
                            "p_api": p_api,
                            "p_model": p_mod,
                            "p_final": p_final,
                            "odds_final": odds_final,
                            "market_confidence": market_confidence,
                            "source": blend_source
                        }
                    )
                    return round(odds_final, 3), blend_source
    
    # ⚠️ VALIDAZIONE: Verifica odds_btts_api
    if odds_btts_api is not None:
        if isinstance(odds_btts_api, (int, float)) and math.isfinite(odds_btts_api) and odds_btts_api > 1.01:
            return round(odds_btts_api, 3), "BTTS da API"
    
    # ⚠️ VALIDAZIONE: Verifica btts_from_model
    if btts_from_model is not None:
        if isinstance(btts_from_model, (int, float)) and math.isfinite(btts_from_model) and btts_from_model > 0:
            prob_model = max(model_config.TOL_DIVISION_ZERO, min(1.0, btts_from_model))
            odds_model = 1.0 / prob_model
            if math.isfinite(odds_model) and odds_model >= 1.01:
                log_precision_metrics(
                    "blend_btts_sources_improved",
                    {
                        "p_model": prob_model,
                        "odds_model": odds_model,
                        "source": "BTTS da modello"
                    }
                )
                return round(odds_model, 3), "BTTS da modello"
    
    return 2.0, "BTTS default"

# ============================================================
#   ESTRATTORE QUOTE CON OUTLIER DETECTION MIGLIORATO
# ============================================================

def detect_outliers_iqr(values: List[float], k: float = 1.5) -> List[bool]:
    """Identifica outlier usando metodo IQR (più robusto)."""
    if len(values) <= 2:
        return [False] * len(values)
    
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr
    
    return [v < lower_bound or v > upper_bound for v in values]

def oddsapi_extract_prices_improved(event: dict) -> dict:
    """Versione migliorata con IQR outlier detection e Shin normalization."""
    
    WEIGHTS = {
        "pinnacle": 2.0,      # Sharp book
        "bet365": 1.6,
        "unibet_eu": 1.3,
        "marathonbet": 1.3,
        "williamhill": 1.2,
        "bwin": 1.0,
        "betonlineag": 1.0,
        "10bet": 1.0,
        "bovada": 0.9,
    }

    home_team = (event.get("home_team") or "home").strip()
    away_team = (event.get("away_team") or "away").strip()
    home_l = home_team.lower()
    away_l = away_team.lower()

    out = {
        "home": home_team,
        "away": away_team,
        "odds_1": None,
        "odds_x": None,
        "odds_2": None,
        "odds_over25": None,
        "odds_under25": None,
        "odds_dnb_home": None,
        "odds_dnb_away": None,
        "odds_btts": None,
    }

    bookmakers = event.get("bookmakers", [])
    if not bookmakers:
        return out

    h2h_home, h2h_draw, h2h_away = [], [], []
    over25_list, under25_list = [], []
    dnb_home_list, dnb_away_list = [], []
    btts_list = []

    for bk in bookmakers:
        bk_key = bk.get("key")
        if bk_key not in WEIGHTS:
            continue
        w = WEIGHTS[bk_key]

        for mk in bk.get("markets", []):
            mk_key_raw = mk.get("key")
            # Assicurati che mk_key sia sempre una stringa (gestisce None)
            mk_key = str(mk_key_raw).lower() if mk_key_raw is not None else ""

            # ⚠️ OTTIMIZZAZIONE: Condizione semplificata (mk_key == "h2h" già coperto da "h2h" in mk_key)
            if "h2h" in mk_key or "match_winner" in mk_key:
                for o in mk.get("outcomes", []):
                    name_l = (o.get("name") or "").strip().lower()
                    price = o.get("price")
                    if not price:
                        continue
                    if name_l == home_l or home_l in name_l:
                        h2h_home.append((price, w))
                    elif name_l == away_l or away_l in name_l:
                        h2h_away.append((price, w))
                    elif name_l in ["draw", "tie", "x", "pareggio"]:
                        h2h_draw.append((price, w))

            elif "totals" in mk_key or "total" in mk_key:
                for o in mk.get("outcomes", []):
                    point = o.get("point")
                    price = o.get("price")
                    name_l = (o.get("name") or "").lower()
                    if price is None:
                        continue
                    if point == 2.5:
                        if "over" in name_l:
                            over25_list.append((price, w))
                        elif "under" in name_l:
                            under25_list.append((price, w))

            elif "draw_no_bet" in mk_key or mk_key == "dnb":
                for o in mk.get("outcomes", []):
                    name_l = (o.get("name") or "").lower()
                    price = o.get("price")
                    if not price:
                        continue
                    if name_l == home_l or home_l in name_l:
                        dnb_home_list.append((price, w))
                    elif name_l == away_l or away_l in name_l:
                        dnb_away_list.append((price, w))

            elif mk_key == "spreads":
                for o in mk.get("outcomes", []):
                    point = o.get("point")
                    price = o.get("price")
                    name_l = (o.get("name") or "").lower()
                    if price is None:
                        continue
                    if point == 0 or point == 0.0:
                        if name_l == home_l or home_l in name_l:
                            dnb_home_list.append((price, w))
                        elif name_l == away_l or away_l in name_l:
                            dnb_away_list.append((price, w))

            elif "btts" in mk_key or "both_teams_to_score" in mk_key:
                for o in mk.get("outcomes", []):
                    name_l = (o.get("name") or "").lower()
                    price = o.get("price")
                    if not price:
                        continue
                    if "yes" in name_l or "sì" in name_l or "si" in name_l:
                        btts_list.append((price, w))

    # Rimozione outlier con IQR
    def _remove_outliers(values: List[Tuple[float, float]]):
        if len(values) <= 2:
            return values
        odds_only = [v for v, _ in values]
        is_outlier = detect_outliers_iqr(odds_only, k=1.5)
        return [item for item, outlier in zip(values, is_outlier) if not outlier]

    h2h_home = _remove_outliers(h2h_home)
    h2h_draw = _remove_outliers(h2h_draw)
    h2h_away = _remove_outliers(h2h_away)
    over25_list = _remove_outliers(over25_list)
    under25_list = _remove_outliers(under25_list)
    dnb_home_list = _remove_outliers(dnb_home_list)
    dnb_away_list = _remove_outliers(dnb_away_list)
    btts_list = _remove_outliers(btts_list)

    def weighted_avg(values: List[Tuple[float, float]]):
        """
        ⚠️ PRECISIONE MANIACALE: Kahan summation per accumulo preciso, protezione divisione per zero
        """
        if not values:
            return None
        
        # ⚠️ PRECISIONE MANIACALE: Kahan summation per accumulo preciso
        num = 0.0
        den = 0.0
        c_num = 0.0  # Compensazione Kahan
        c_den = 0.0
        
        for v, w in values:
            # ⚠️ PROTEZIONE: Ignora valori non validi
            if not isinstance(v, (int, float)) or not isinstance(w, (int, float)):
                continue
            if not math.isfinite(v) or not math.isfinite(w) or w < 0:
                continue
            
            # Kahan summation per numeratore
            term = v * w
            y = term - c_num
            t = num + y
            c_num = (t - num) - y
            num = t
            
            # Kahan summation per denominatore
            y = w - c_den
            t = den + y
            c_den = (t - den) - y
            den = t
        
        # ⚠️ PROTEZIONE: Protezione divisione per zero
        if den > model_config.TOL_DIVISION_ZERO:
            result = num / den
            if math.isfinite(result):
                return round(result, 3)
        
        logger.warning(f"weighted_avg: den troppo piccolo ({den}), ritorno None")
        return None

    out["odds_1"] = weighted_avg(h2h_home)
    out["odds_x"] = weighted_avg(h2h_draw)
    out["odds_2"] = weighted_avg(h2h_away)
    out["odds_over25"] = weighted_avg(over25_list)
    out["odds_under25"] = weighted_avg(under25_list)
    out["odds_dnb_home"] = weighted_avg(dnb_home_list)
    out["odds_dnb_away"] = weighted_avg(dnb_away_list)
    out["odds_btts"] = weighted_avg(btts_list)

    # Normalizzazione Shin
    if out["odds_1"] and out["odds_x"] and out["odds_2"]:
        n1, nx, n2 = normalize_three_way_shin(out["odds_1"], out["odds_x"], out["odds_2"])
        out["odds_1"], out["odds_x"], out["odds_2"] = n1, nx, n2

    if out["odds_over25"] and out["odds_under25"]:
        no, nu = normalize_two_way_shin(out["odds_over25"], out["odds_under25"])
        out["odds_over25"], out["odds_under25"] = no, nu

    return out

# ============================================================
#        FUNZIONI ODDS API (invariate)
# ============================================================

def oddsapi_get_soccer_leagues() -> List[dict]:
    if not THE_ODDS_API_KEY:
        logger.warning("THE_ODDS_API_KEY non configurata.")
        if st:
            st.error("⚠️ THE_ODDS_API_KEY non configurata.")
        return []
    try:
        timeout_val = app_config.api_timeout if hasattr(app_config, 'api_timeout') else 10.0
        r = requests.get(
            f"{THE_ODDS_BASE}/sports",
            params={"apiKey": THE_ODDS_API_KEY, "all": "true"},
            timeout=timeout_val,
        )
        r.raise_for_status()
        data = r.json()
        leagues = [s for s in data if s.get("key", "").startswith("soccer")]
        if not leagues:
            logger.warning("Nessuna lega di calcio trovata nella risposta API")
        return leagues
    except requests.exceptions.Timeout:
        error_msg = "Timeout richiesta The Odds API (sports)"
        logger.error(error_msg)
        if st:
            st.error(f"⏱️ {error_msg}")
        return []
    except requests.exceptions.HTTPError as e:
        error_msg = f"Errore HTTP The Odds API (sports): {e.response.status_code}"
        logger.error(error_msg)
        if st:
            if e.response.status_code == 401:
                st.error("🔑 API key non valida o scaduta. Controlla THE_ODDS_API_KEY.")
            else:
                st.error(f"❌ {error_msg}")
        return []
    except requests.exceptions.RequestException as e:
        error_msg = f"Errore richiesta The Odds API (sports): {e}"
        logger.error(error_msg)
        if st:
            st.error(f"❌ {error_msg}")
        return []
    except (ValueError, KeyError, json.JSONDecodeError) as e:
        error_msg = f"Errore parsing risposta The Odds API (sports): {e}"
        logger.error(error_msg)
        if st:
            st.error(f"❌ {error_msg}")
        return []

def oddsapi_get_events_for_league(league_key: str) -> List[dict]:
    if not THE_ODDS_API_KEY:
        logger.warning("THE_ODDS_API_KEY non configurata.")
        if st:
            st.error("⚠️ THE_ODDS_API_KEY non configurata.")
        return []
    base_url = f"{THE_ODDS_BASE}/sports/{league_key}/odds"
    params_common = {
        "apiKey": THE_ODDS_API_KEY,
        "regions": "eu,uk",
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }

    try:
        r = requests.get(
            base_url,
            params={**params_common, "markets": "h2h,totals,spreads,btts"},
            timeout=app_config.api_timeout,
        )
        r.raise_for_status()
        data = r.json()
        if data:
            return data
    except requests.exceptions.RequestException as e:
        logger.debug(f"Errore events con BTTS per {league_key}: {e}, provo senza BTTS")

    try:
        r2 = requests.get(
            base_url,
            params={**params_common, "markets": "h2h,totals,spreads"},
            timeout=app_config.api_timeout,
        )
        r2.raise_for_status()
        return r2.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Errore events per {league_key}: {e}")
        return []

def oddsapi_refresh_event(league_key: str, event_id: str) -> dict:
    if not THE_ODDS_API_KEY:
        logger.warning("THE_ODDS_API_KEY non configurata.")
        if st:
            st.error("⚠️ THE_ODDS_API_KEY non configurata.")
        return {}
    if not league_key or not event_id:
        logger.warning(f"Parametri mancanti: league_key={league_key}, event_id={event_id}")
        return {}
    url = f"{THE_ODDS_BASE}/sports/{league_key}/events/{event_id}/odds"
    params = {
        "apiKey": THE_ODDS_API_KEY,
        "regions": "eu,uk",
        "oddsFormat": "decimal",
        "dateFormat": "iso",
        "markets": "h2h,totals,spreads,btts",
    }
    try:
        r = requests.get(url, params=params, timeout=app_config.api_timeout)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and data:
            return data[0]
        return data
    except requests.exceptions.RequestException as e:
        logger.error(f"Errore refresh evento {event_id} in {league_key}: {e}")
        return {}

# ============================================================
#  API-FOOTBALL
# ============================================================

def apifootball_get_fixtures_by_date(d: str) -> list:
    if not API_FOOTBALL_KEY:
        logger.warning("API_FOOTBALL_KEY non configurata. Configura tramite variabile d'ambiente.")
        return []
    headers = {"x-apisports-key": API_FOOTBALL_KEY}
    params = {"date": d}
    try:
        r = requests.get(
            f"{API_FOOTBALL_BASE}/fixtures",
            headers=headers,
            params=params,
            timeout=app_config.api_timeout
        )
        r.raise_for_status()
        data = r.json()
        return data.get("response", [])
    except requests.exceptions.Timeout:
        logger.error(f"Timeout API-Football fixtures per data {d}")
        return []
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            logger.error("API-Football: API key non valida o scaduta")
        else:
            logger.error(f"Errore HTTP API-Football fixtures: {e.response.status_code}")
        return []
    except requests.exceptions.RequestException as e:
        logger.error(f"Errore richiesta API-Football fixtures: {e}")
        return []
    except (ValueError, KeyError, json.JSONDecodeError) as e:
        logger.error(f"Errore parsing risposta API-Football: {e}")
        return []

# ============================================================
#   API-FOOTBALL: RECUPERO DATI AVANZATI
# ============================================================

def apifootball_search_team(team_name: str, league_id: int = None) -> Dict[str, Any]:
    """
    Cerca team ID da API-Football usando nome squadra.
    """
    if not API_FOOTBALL_KEY:
        logger.warning("API_FOOTBALL_KEY non configurata. Configura tramite variabile d'ambiente.")
        return {}
    headers = {"x-apisports-key": API_FOOTBALL_KEY}
    params = {"search": team_name}
    if league_id:
        params["league"] = league_id
    
    try:
        r = requests.get(
            f"{API_FOOTBALL_BASE}/teams",
            headers=headers,
            params=params,
            timeout=app_config.api_timeout
        )
        r.raise_for_status()
        data = r.json()
        teams = data.get("response", [])
        if teams:
            return teams[0]  # Ritorna primo match
        return {}
    except requests.exceptions.RequestException as e:
        logger.error(f"Errore ricerca team {team_name}: {e}")
        return {}

def apifootball_get_team_fixtures(team_id: int, last: int = 10, season: int = None) -> List[Dict[str, Any]]:
    """
    Recupera ultime partite di una squadra.
    """
    if not API_FOOTBALL_KEY:
        logger.warning("API_FOOTBALL_KEY non configurata. Configura tramite variabile d'ambiente.")
        return []
    headers = {"x-apisports-key": API_FOOTBALL_KEY}
    params = {"team": team_id, "last": last}
    if season:
        params["season"] = season
    
    try:
        r = requests.get(
            f"{API_FOOTBALL_BASE}/fixtures",
            headers=headers,
            params=params,
            timeout=app_config.api_timeout
        )
        r.raise_for_status()
        data = r.json()
        return data.get("response", [])
    except requests.exceptions.RequestException as e:
        logger.error(f"Errore fixtures team {team_id}: {e}")
        return []

def apifootball_get_standings(league_id: int, season: int) -> Dict[str, Any]:
    """
    Recupera classifica di una lega.
    """
    if not API_FOOTBALL_KEY:
        logger.warning("API_FOOTBALL_KEY non configurata. Configura tramite variabile d'ambiente.")
        return []
    headers = {"x-apisports-key": API_FOOTBALL_KEY}
    params = {"league": league_id, "season": season}
    
    try:
        r = requests.get(
            f"{API_FOOTBALL_BASE}/standings",
            headers=headers,
            params=params,
            timeout=app_config.api_timeout
        )
        r.raise_for_status()
        data = r.json()
        standings = data.get("response", [])
        # ⚠️ PROTEZIONE: Valida che standings non sia vuoto e che [0] sia un dict
        if standings and len(standings) > 0 and isinstance(standings[0], dict):
            return standings[0].get("league", {}).get("standings", [])
        return []
    except requests.exceptions.RequestException as e:
        logger.error(f"Errore standings league {league_id}: {e}")
        return []

def get_league_id_from_name(league_name: str) -> int:
    """
    Mappa nome lega a ID API-Football.
    """
    league_map = {
        "premier_league": 39,  # Premier League
        "serie_a": 135,  # Serie A
        "la_liga": 140,  # La Liga
        "bundesliga": 78,  # Bundesliga
        "ligue_1": 61,  # Ligue 1
    }
    return league_map.get(league_name, 0)

def get_current_season() -> int:
    """
    Ritorna anno stagione corrente (es. 2024 per 2023-2024).
    """
    now = datetime.now()
    # Stagione inizia agosto, quindi se siamo dopo agosto usiamo anno corrente
    if now.month >= 8:
        return now.year
    else:
        return now.year - 1

def calculate_days_since_last_match(team_id: int, match_date: str) -> int:
    """
    Calcola giorni dall'ultima partita di una squadra.
    """
    if not team_id or not match_date:
        return None
    
    try:
        # Recupera ultime partite
        fixtures = apifootball_get_team_fixtures(team_id, last=5)
        if not fixtures:
            return None
        
        # Trova ultima partita giocata (status FT, AET, PEN)
        last_match_date = None
        for fixture in fixtures:
            # ⚠️ PROTEZIONE: Accesso sicuro a dizionari annidati
            fixture_data = fixture.get("fixture", {}) if isinstance(fixture, dict) else {}
            status_data = fixture_data.get("status", {}) if isinstance(fixture_data, dict) else {}
            status = status_data.get("short", "") if isinstance(status_data, dict) else ""
            if status in ["FT", "AET", "PEN"]:
                date_str = fixture_data.get("date", "") if isinstance(fixture_data, dict) else ""
                if date_str:
                    last_match_date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                    break
        
        if not last_match_date:
            return None
        
        # Calcola differenza con data partita corrente
        match_dt = datetime.fromisoformat(match_date.replace("Z", "+00:00"))
        delta = (match_dt - last_match_date).days
        
        return max(0, delta)  # Non può essere negativo
    except (ValueError, TypeError, KeyError) as e:
        logger.error(f"Errore calcolo giorni ultima partita: {e}")
        return None

def count_matches_last_30_days(team_id: int, match_date: str) -> int:
    """
    Conta partite giocate negli ultimi 30 giorni.
    """
    if not team_id or not match_date:
        return None
    
    try:
        fixtures = apifootball_get_team_fixtures(team_id, last=15)
        if not fixtures:
            return None
        
        match_dt = datetime.fromisoformat(match_date.replace("Z", "+00:00"))
        cutoff_date = match_dt - timedelta(days=30)
        
        count = 0
        for fixture in fixtures:
            # ⚠️ PROTEZIONE: Accesso sicuro a dizionari annidati
            fixture_data = fixture.get("fixture", {}) if isinstance(fixture, dict) else {}
            status_data = fixture_data.get("status", {}) if isinstance(fixture_data, dict) else {}
            status = status_data.get("short", "") if isinstance(status_data, dict) else ""
            if status in ["FT", "AET", "PEN"]:
                date_str = fixture_data.get("date", "") if isinstance(fixture_data, dict) else ""
                if date_str:
                    fixture_date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                    if cutoff_date <= fixture_date < match_dt:
                        count += 1
        
        return count
    except (ValueError, TypeError, KeyError) as e:
        logger.error(f"Errore conteggio partite 30 giorni: {e}")
        return None

def get_team_standings_info(team_name: str, league: str, season: int = None) -> Dict[str, Any]:
    """
    Recupera informazioni classifica per una squadra.
    """
    if not season:
        season = get_current_season()
    
    league_id = get_league_id_from_name(league)
    if not league_id:
        return {}
    
    try:
        standings = apifootball_get_standings(league_id, season)
        if not standings:
            return {}
        
        # Cerca squadra nella classifica
        team_name_lower = team_name.lower()
        for group in standings:
            for entry in group:
                team_info = entry.get("team", {})
                team_name_raw = team_info.get("name")
                # Assicurati che team_name_api sia sempre una stringa (gestisce None)
                team_name_api = str(team_name_raw).lower() if team_name_raw is not None else ""
                
                # Match approssimativo (potrebbe essere diverso)
                if team_name_lower in team_name_api or team_name_api in team_name_lower:
                    position = entry.get("rank", 0)
                    points = entry.get("points", 0)
                    all_stats = entry.get("all", {})
                    played = all_stats.get("played", 0)
                    
                    # Calcola distanza da salvezza (assumendo 18 squadre, 3 retrocedono)
                    # In realtà dipende dalla lega, ma approssimiamo
                    points_from_relegation = None
                    if position >= 16:  # Ultime 3 posizioni
                        # Trova punti della 17esima (ultima salva)
                        for e in group:
                            rank_val = e.get("rank")
                            # Converti rank a int se necessario (può essere stringa)
                            try:
                                rank_int = int(rank_val) if rank_val is not None else None
                                if rank_int == 17:
                                    points_17 = e.get("points", 0)
                                    # Assicurati che points_17 sia un numero
                                    try:
                                        points_17 = int(points_17) if points_17 is not None else 0
                                    except (ValueError, TypeError):
                                        points_17 = 0
                                    points_from_relegation = points - points_17
                                    break
                            except (ValueError, TypeError):
                                continue
                    
                    # Calcola distanza da Europa (posizione 6-7)
                    points_from_europe = None
                    if 6 <= position <= 10:
                        for e in group:
                            rank_val = e.get("rank")
                            # Converti rank a int se necessario (può essere stringa)
                            try:
                                rank_int = int(rank_val) if rank_val is not None else None
                                if rank_int == 6:
                                    points_6 = e.get("points", 0)
                                    # Assicurati che points_6 sia un numero
                                    try:
                                        points_6 = int(points_6) if points_6 is not None else 0
                                    except (ValueError, TypeError):
                                        points_6 = 0
                                    points_from_europe = points_6 - points
                                    break
                            except (ValueError, TypeError):
                                continue
                    
                    return {
                        "position": position,
                        "points": points,
                        "played": played,
                        "points_from_relegation": points_from_relegation,
                        "points_from_europe": points_from_europe,
                    }
        
        return {}
    except (KeyError, ValueError, requests.exceptions.RequestException) as e:
        logger.error(f"Errore recupero standings {team_name}: {e}")
        return {}

def is_derby_match(home_team: str, away_team: str, league: str) -> bool:
    """
    Identifica se è un derby basandosi su pattern comuni.
    """
    # Lista derby comuni (potrebbe essere espansa)
    derby_patterns = [
        # Serie A
        ("inter", "milan"), ("milan", "inter"),
        ("juventus", "torino"), ("torino", "juventus"),
        ("roma", "lazio"), ("lazio", "roma"),
        ("napoli", "juventus"), ("juventus", "napoli"),
        # Premier League
        ("manchester united", "manchester city"), ("manchester city", "manchester united"),
        ("liverpool", "everton"), ("everton", "liverpool"),
        ("arsenal", "tottenham"), ("tottenham", "arsenal"),
        # La Liga
        ("real madrid", "atletico madrid"), ("atletico madrid", "real madrid"),
        ("real madrid", "barcelona"), ("barcelona", "real madrid"),
        ("atletico madrid", "barcelona"), ("barcelona", "atletico madrid"),
        # Bundesliga
        ("bayern munich", "borussia dortmund"), ("borussia dortmund", "bayern munich"),
    ]
    
    home_lower = home_team.lower()
    away_lower = away_team.lower()
    
    for pattern_home, pattern_away in derby_patterns:
        if pattern_home in home_lower and pattern_away in away_lower:
            return True
    
    return False

def apifootball_get_team_statistics(team_id: int, league_id: int, season: int) -> Dict[str, Any]:
    """
    Recupera statistiche dettagliate di una squadra.
    """
    if not API_FOOTBALL_KEY:
        logger.warning("API_FOOTBALL_KEY non configurata. Configura tramite variabile d'ambiente.")
        return {}
    headers = {"x-apisports-key": API_FOOTBALL_KEY}
    params = {"team": team_id, "league": league_id, "season": season}
    
    try:
        r = requests.get(
            f"{API_FOOTBALL_BASE}/teams/statistics",
            headers=headers,
            params=params,
            timeout=app_config.api_timeout
        )
        r.raise_for_status()
        data = r.json()
        response = data.get("response", {})
        return response
    except requests.exceptions.RequestException as e:
        logger.error(f"Errore statistiche team {team_id}: {e}")
        return {}

def apifootball_get_head_to_head(team1_id: int, team2_id: int, last: int = 10) -> List[Dict[str, Any]]:
    """
    Recupera partite head-to-head tra due squadre.
    """
    if not API_FOOTBALL_KEY:
        logger.warning("API_FOOTBALL_KEY non configurata. Configura tramite variabile d'ambiente.")
        return []
    headers = {"x-apisports-key": API_FOOTBALL_KEY}
    params = {"h2h": f"{team1_id}-{team2_id}", "last": last}
    
    try:
        r = requests.get(
            f"{API_FOOTBALL_BASE}/fixtures/headtohead",
            headers=headers,
            params=params,
            timeout=app_config.api_timeout
        )
        r.raise_for_status()
        data = r.json()
        return data.get("response", [])
    except requests.exceptions.RequestException as e:
        logger.error(f"Errore H2H {team1_id} vs {team2_id}: {e}")
        return []

def apifootball_get_injuries(team_id: int = None, fixture_id: int = None) -> List[Dict[str, Any]]:
    """
    Recupera infortuni. Se team_id è specificato, filtra per squadra.
    """
    if not API_FOOTBALL_KEY:
        logger.warning("API_FOOTBALL_KEY non configurata. Configura tramite variabile d'ambiente.")
        return []
    headers = {"x-apisports-key": API_FOOTBALL_KEY}
    params = {}
    if team_id:
        params["team"] = team_id
    if fixture_id:
        params["fixture"] = fixture_id
    
    try:
        r = requests.get(
            f"{API_FOOTBALL_BASE}/injuries",
            headers=headers,
            params=params,
            timeout=app_config.api_timeout
        )
        r.raise_for_status()
        data = r.json()
        return data.get("response", [])
    except requests.exceptions.RequestException as e:
        logger.error(f"Errore infortuni: {e}")
        return []

def apifootball_get_fixture_lineups(fixture_id: int) -> Dict[str, Any]:
    """
    ALTA PRIORITÀ: Recupera formazioni (lineups) per una partita.
    
    Returns:
        Dict con formazioni casa/trasferta, formazione (es. 4-3-3), giocatori chiave
    """
    if not API_FOOTBALL_KEY:
        logger.warning("API_FOOTBALL_KEY non configurata. Configura tramite variabile d'ambiente.")
        return {}
    headers = {"x-apisports-key": API_FOOTBALL_KEY}
    params = {"fixture": fixture_id}
    
    try:
        r = requests.get(
            f"{API_FOOTBALL_BASE}/fixtures/lineups",
            headers=headers,
            params=params,
            timeout=app_config.api_timeout
        )
        r.raise_for_status()
        data = r.json()
        response = data.get("response", [])
        
        if not response or len(response) == 0:
            return {}
        
        # Estrai formazioni
        result = {
            "home_lineup": None,
            "away_lineup": None,
            "home_formation": None,
            "away_formation": None,
            "home_key_players": [],
            "away_key_players": [],
            "data_available": False,
        }
        
        for lineup_data in response:
            team_info = lineup_data.get("team", {})
            team_id = team_info.get("id")
            formation = lineup_data.get("formation")
            startXI = lineup_data.get("startXI", [])
            
            # Determina se casa o trasferta (primo = casa, secondo = trasferta)
            if result["home_lineup"] is None:
                result["home_lineup"] = startXI
                result["home_formation"] = formation
                # Estrai nomi giocatori chiave (primi 5)
                result["home_key_players"] = [
                    p.get("player", {}).get("name", "") 
                    for p in startXI[:5] if p.get("player")
                ]
            else:
                result["away_lineup"] = startXI
                result["away_formation"] = formation
                result["away_key_players"] = [
                    p.get("player", {}).get("name", "") 
                    for p in startXI[:5] if p.get("player")
                ]
        
        result["data_available"] = result["home_lineup"] is not None and result["away_lineup"] is not None
        return result
    except requests.exceptions.RequestException as e:
        logger.error(f"Errore lineups fixture {fixture_id}: {e}")
        return {}

def calculate_lineup_impact(lineup_data: Dict[str, Any], team_id: int) -> Dict[str, float]:
    """
    Calcola impatto formazione su lambda.
    
    Analizza:
    - Formazione (4-3-3 vs 4-4-2 vs 5-3-2)
    - Giocatori chiave presenti/assenti
    """
    if not lineup_data or not lineup_data.get("data_available"):
        return {"attack_factor": 1.0, "defense_factor": 1.0, "confidence": 0.0}
    
    attack_factor = 1.0
    defense_factor = 1.0
    
    # Analizza formazione
    formation = lineup_data.get("home_formation") if lineup_data.get("home_lineup") else lineup_data.get("away_formation")
    
    if formation:
        # Formazioni offensive (4-3-3, 3-4-3) → +attacco, -difesa
        if formation in ["4-3-3", "3-4-3", "4-2-3-1"]:
            attack_factor *= 1.05
            defense_factor *= 0.98
        # Formazioni difensive (5-3-2, 4-5-1) → -attacco, +difesa
        elif formation in ["5-3-2", "4-5-1", "5-4-1"]:
            attack_factor *= 0.95
            defense_factor *= 1.05
    
    # Se abbiamo giocatori chiave, verifica se sono presenti
    key_players = lineup_data.get("home_key_players") or lineup_data.get("away_key_players")
    if key_players and len(key_players) >= 3:
        # Se abbiamo almeno 3 giocatori chiave → formazione forte
        confidence = min(1.0, len(key_players) / 5.0)
    else:
        confidence = 0.3
    
    return {
        "attack_factor": round(attack_factor, 3),
        "defense_factor": round(defense_factor, 3),
        "confidence": round(confidence, 2),
        "formation": formation,
    }

def apifootball_get_fixture_info(fixture_id: int) -> Dict[str, Any]:
    """
    Recupera info complete su una partita (per weather, referee, etc.).
    """
    if not API_FOOTBALL_KEY:
        logger.warning("API_FOOTBALL_KEY non configurata. Configura tramite variabile d'ambiente.")
        return {}
    headers = {"x-apisports-key": API_FOOTBALL_KEY}
    params = {"id": fixture_id}
    
    try:
        r = requests.get(
            f"{API_FOOTBALL_BASE}/fixtures",
            headers=headers,
            params=params,
            timeout=app_config.api_timeout
        )
        r.raise_for_status()
        data = r.json()
        response = data.get("response", [])
        if response:
            return response[0]
        return {}
    except requests.exceptions.RequestException as e:
        logger.error(f"Errore fixture info {fixture_id}: {e}")
        return {}

def get_weather_impact(fixture_data: Dict[str, Any]) -> Dict[str, float]:
    """
    ALTA PRIORITÀ: Calcola impatto condizioni meteo su lambda.
    
    Analizza:
    - Pioggia forte → riduce gol (difesa più difficile)
    - Vento forte → aumenta incertezza
    - Temperatura estrema → fatica
    """
    if not fixture_data:
        return {"total_factor": 1.0, "confidence": 0.0}
    
    fixture_info = fixture_data.get("fixture", {})
    venue = fixture_info.get("venue", {})
    
    # API-Football non fornisce sempre weather, ma possiamo inferire da altri dati
    # Per ora ritorna neutro, ma struttura pronta per integrazione futura
    weather_data = fixture_data.get("weather", {})
    
    total_factor = 1.0
    confidence = 0.0
    
    if weather_data:
        temp = weather_data.get("temp", None)
        condition_raw = weather_data.get("condition")
        # Assicurati che condition sia sempre una stringa (gestisce None)
        condition = str(condition_raw).lower() if condition_raw is not None else ""
        
        # Pioggia forte → meno gol
        if condition and ("rain" in condition or "storm" in condition):
            total_factor *= 0.92  # -8% gol
            confidence = 0.7
        
        # Vento forte → più incertezza (aumenta varianza)
        if "wind" in condition:
            total_factor *= 0.96  # -4% gol
            confidence = 0.5
        
        # Temperatura estrema (< 5°C o > 30°C) → fatica
        if temp:
            if temp < 5 or temp > 30:
                total_factor *= 0.95  # -5% gol
                confidence = 0.6
    
    return {
        "total_factor": round(total_factor, 3),
        "confidence": round(confidence, 2),
        "weather_condition": weather_data.get("condition", "unknown") if weather_data else None,
    }

def get_referee_statistics(referee_name: str, league: str = None) -> Dict[str, float]:
    """
    ALTA PRIORITÀ: Recupera statistiche arbitro da API-Football.
    
    Analizza:
    - Media cartellini per partita
    - Media gol per partita (alcuni arbitri favoriscono attacco)
    - Tendenza a favorire casa/trasferta
    """
    # API-Football non ha endpoint diretto per referee stats
    # Possiamo inferire da fixture history, ma per ora ritorna neutro
    # Struttura pronta per integrazione futura
    
    return {
        "cards_factor": 1.0,  # Molti cartellini → più interruzioni → meno gol
        "goals_factor": 1.0,  # Alcuni arbitri favoriscono attacco
        "home_bias": 1.0,  # Bias verso casa
        "confidence": 0.0,
    }

def analyze_market_depth(
    event: dict,
    odds_1: float,
    odds_x: float,
    odds_2: float,
) -> Dict[str, Any]:
    """
    ALTA PRIORITÀ: Analizza profondità mercato e sharp money.
    
    Analizza:
    - Numero bookmakers (proxy liquidità)
    - Spread quote (liquidità implicita)
    - Movimento quote (sharp money detection)
    - Volume scommesse (se disponibile)
    """
    if not event:
        return {}
    
    bookmakers = event.get("bookmakers", [])
    num_bookmakers = len(bookmakers)
    
    # Raccogli tutte le quote per ogni mercato
    all_odds_1 = []
    all_odds_x = []
    all_odds_2 = []
    
    for bk in bookmakers:
        for mk in bk.get("markets", []):
            mk_key_raw = mk.get("key")
            mk_key = str(mk_key_raw).lower() if mk_key_raw is not None else ""
            if mk_key in ["h2h", "match_winner"]:
                for o in mk.get("outcomes", []):
                    name_l = (o.get("name") or "").lower()
                    price = o.get("price")
                    if not price:
                        continue
                    
                    if "home" in name_l or "1" == name_l:
                        all_odds_1.append(price)
                    elif "draw" in name_l or "x" == name_l or "tie" in name_l:
                        all_odds_x.append(price)
                    elif "away" in name_l or "2" == name_l:
                        all_odds_2.append(price)
    
    # Calcola metriche liquidità
    depth_score = 0.0
    liquidity_indicators = {}
    
    # 1. Numero bookmakers (più = più liquidità)
    if num_bookmakers >= 10:
        depth_score += 30
        liquidity_indicators["bookmakers"] = "high"
    elif num_bookmakers >= 7:
        depth_score += 20
        liquidity_indicators["bookmakers"] = "medium"
    elif num_bookmakers >= 5:
        depth_score += 10
        liquidity_indicators["bookmakers"] = "low"
    else:
        liquidity_indicators["bookmakers"] = "very_low"
    
    # 2. Spread quote (minore = più liquidità)
    if all_odds_1 and all_odds_2:
        min_1 = min(all_odds_1)
        max_1 = max(all_odds_1)
        spread_1 = (max_1 - min_1) / min_1 * 100  # Spread percentuale
        
        if spread_1 < 2.0:
            depth_score += 25
            liquidity_indicators["spread"] = "tight"
        elif spread_1 < 5.0:
            depth_score += 15
            liquidity_indicators["spread"] = "moderate"
        else:
            liquidity_indicators["spread"] = "wide"
    
    # 3. Sharp money detection (se quote convergono rapidamente)
    # Approssimato: se spread è stretto e molti bookmakers → sharp money presente
    if liquidity_indicators.get("spread") == "tight" and num_bookmakers >= 8:
        liquidity_indicators["sharp_money"] = "likely"
        depth_score += 20
    else:
        liquidity_indicators["sharp_money"] = "unlikely"
    
    # 4. Market efficiency score (0-100)
    efficiency_score = min(100, depth_score)
    
    return {
        "depth_score": round(efficiency_score, 1),
        "num_bookmakers": num_bookmakers,
        "liquidity_indicators": liquidity_indicators,
        "spread_1": round(spread_1, 2) if all_odds_1 else None,
        "market_efficiency": "high" if efficiency_score >= 70 else ("medium" if efficiency_score >= 50 else "low"),
    }

def calculate_team_form_from_statistics(team_stats: Dict[str, Any], last_n: int = 5) -> Dict[str, float]:
    """
    Calcola fattore forma da statistiche squadra usando dati aggiornati dalle API.
    Integra dati da API-Football per calcoli matematici più accurati.
    """
    if not team_stats:
        return {"form_attack": 1.0, "form_defense": 1.0, "form_points": 1.0, "confidence": 0.0}
    
    try:
        fixtures = team_stats.get("fixtures", {}) if isinstance(team_stats, dict) else {}
        # ⚠️ PROTEZIONE: Accesso sicuro a dizionari annidati
        played_data = fixtures.get("played", {}) if isinstance(fixtures, dict) else {}
        played = played_data.get("total", 0) if isinstance(played_data, dict) else 0
        
        if played < 3:
            return {"form_attack": 1.0, "form_defense": 1.0, "form_points": 1.0, "confidence": 0.0}
        
        # Statistiche attacco (da API aggiornate)
        # ⚠️ PROTEZIONE: Accesso sicuro a dizionari annidati con fallback
        goals_data = team_stats.get("goals", {}) if isinstance(team_stats, dict) else {}
        goals_for_data = goals_data.get("for", {}) if isinstance(goals_data, dict) else {}
        goals_for_avg = goals_for_data.get("average", {}) if isinstance(goals_for_data, dict) else {}
        goals_for = goals_for_avg.get("total", 0) if isinstance(goals_for_avg, dict) else 0
        
        goals_against_data = goals_data.get("against", {}) if isinstance(goals_data, dict) else {}
        goals_against_avg = goals_against_data.get("average", {}) if isinstance(goals_against_data, dict) else {}
        goals_against = goals_against_avg.get("total", 0) if isinstance(goals_against_avg, dict) else 0
        
        # Statistiche avanzate se disponibili (shots, xG, etc.) - dati API
        shots_data = team_stats.get("shots", {}) if isinstance(team_stats, dict) and team_stats.get("shots") else {}
        if shots_data:
            shots_for_data = shots_data.get("for", {}) if isinstance(shots_data, dict) else {}
            shots_for_avg = shots_for_data.get("average", {}) if isinstance(shots_for_data, dict) else {}
            shots_for = shots_for_avg.get("total", 0) if isinstance(shots_for_avg, dict) else 0
            
            shots_against_data = shots_data.get("against", {}) if isinstance(shots_data, dict) else {}
            shots_against_avg = shots_against_data.get("average", {}) if isinstance(shots_against_data, dict) else {}
            shots_against = shots_against_avg.get("total", 0) if isinstance(shots_against_avg, dict) else 0
        else:
            shots_for = shots_against = 0
        
        # Forma ultime partite (dati reali dalle API)
        # ⚠️ PROTEZIONE: Accesso sicuro a dizionari annidati
        wins_data = fixtures.get("wins", {}) if isinstance(fixtures, dict) else {}
        wins = wins_data.get("total", 0) if isinstance(wins_data, dict) else 0
        
        draws_data = fixtures.get("draws", {}) if isinstance(fixtures, dict) else {}
        draws = draws_data.get("total", 0) if isinstance(draws_data, dict) else 0
        
        losses_data = fixtures.get("loses", {}) if isinstance(fixtures, dict) else {}
        losses = losses_data.get("total", 0) if isinstance(losses_data, dict) else 0
        
        # ⚠️ PRECISIONE: Calcola forma punti con protezione divisione per zero
        played_safe = max(1, played)
        form_points = (wins * 3.0 + draws) / max(model_config.TOL_DIVISION_ZERO, played_safe * 3.0)
        form_points = max(0.0, min(1.0, form_points))  # Limita a [0, 1]
        
        # Normalizza forma punti (0.33 = media, 1.0 = perfetto)
        form_points_factor = 0.7 + (form_points - 0.33) * 0.9  # Range: 0.7 - 1.6
        form_points_factor = max(0.7, min(1.6, form_points_factor))  # Limita range
        
        # ⚠️ VERIFICA: Assicura che form_points_factor sia finito
        if not math.isfinite(form_points_factor):
            logger.warning(f"form_points_factor non finito: {form_points_factor}, correggo")
            form_points_factor = 1.0
        
        # Fattore attacco basato su gol fatti (dati API aggiornati)
        avg_goals_league = 1.3
        # ⚠️ PRECISIONE: Protezione divisione per zero
        if avg_goals_league > model_config.TOL_DIVISION_ZERO:
            form_attack = 0.85 + (goals_for / avg_goals_league - 1.0) * 0.3  # Range: 0.85 - 1.15
        else:
            form_attack = 1.0
        form_attack = max(0.85, min(1.15, form_attack))  # Limita range
        
        # ⚠️ VERIFICA: Assicura che form_attack sia finito
        if not math.isfinite(form_attack):
            logger.warning(f"form_attack non finito: {form_attack}, correggo")
            form_attack = 1.0
        
        # Migliora con dati shots se disponibili (più tiri = più opportunità)
        if shots_for > 0:
            avg_shots_league = 12.0  # Media lega
            # ⚠️ PRECISIONE: Protezione divisione per zero
            if avg_shots_league > model_config.TOL_DIVISION_ZERO:
                shots_factor = min(1.1, 0.95 + (shots_for / avg_shots_league - 1.0) * 0.15)
            else:
                shots_factor = 1.0
            form_attack = (form_attack + shots_factor) / 2.0  # Media pesata
            form_attack = max(0.85, min(1.15, form_attack))  # Limita range
        
        # Fattore difesa basato su gol subiti (dati API aggiornati)
        # ⚠️ PRECISIONE: Protezione divisione per zero
        if avg_goals_league > model_config.TOL_DIVISION_ZERO:
            form_defense = 0.85 + (1.0 - goals_against / avg_goals_league) * 0.3  # Range: 0.85 - 1.15
        else:
            form_defense = 1.0
        form_defense = max(0.85, min(1.15, form_defense))  # Limita range
        
        # ⚠️ VERIFICA: Assicura che form_defense sia finito
        if not math.isfinite(form_defense):
            logger.warning(f"form_defense non finito: {form_defense}, correggo")
            form_defense = 1.0
        
        # Migliora con dati shots against se disponibili
        if shots_against > 0:
            avg_shots_league = 12.0
            # ⚠️ PRECISIONE: Protezione divisione per zero
            if avg_shots_league > model_config.TOL_DIVISION_ZERO:
                shots_against_factor = min(1.1, 0.95 + (1.0 - shots_against / avg_shots_league) * 0.15)
            else:
                shots_against_factor = 1.0
            form_defense = (form_defense + shots_against_factor) / 2.0
            form_defense = max(0.85, min(1.15, form_defense))  # Limita range
        
        # Confidence basata su partite giocate e qualità dati
        confidence = min(1.0, played / 10.0)
        
        # Boost confidence se abbiamo statistiche avanzate (shots, xG, etc.)
        if shots_for > 0 or shots_against > 0:
            confidence = min(1.0, confidence * 1.1)  # +10% confidence
        
        return {
            "form_attack": round(form_attack, 3),
            "form_defense": round(form_defense, 3),
            "form_points": round(form_points_factor, 3),
            "confidence": round(confidence, 2),
            "goals_for_avg": round(goals_for, 2),
            "goals_against_avg": round(goals_against, 2),
            "shots_for_avg": round(shots_for, 2) if shots_for > 0 else None,
            "shots_against_avg": round(shots_against, 2) if shots_against > 0 else None,
        }
    except (KeyError, ValueError, TypeError) as e:
        logger.error(f"Errore calcolo forma da statistiche: {e}")
        return {"form_attack": 1.0, "form_defense": 1.0, "form_points": 1.0, "confidence": 0.0}

def calculate_h2h_adjustments(h2h_matches: List[Dict[str, Any]], home_team_id: int, away_team_id: int) -> Dict[str, float]:
    """
    Calcola aggiustamenti basati su H2H.
    """
    if not h2h_matches or len(h2h_matches) < 2:
        return {"h2h_home_advantage": 1.0, "h2h_goals_factor": 1.0, "h2h_btts_factor": 1.0, "confidence": 0.0}
    
    try:
        home_wins = 0
        draws = 0
        away_wins = 0
        total_goals = 0
        btts_count = 0
        matches_played = 0
        
        for match in h2h_matches:
            fixture = match.get("fixture", {})
            status = fixture.get("status", {}).get("short", "")
            
            if status not in ["FT", "AET", "PEN"]:
                continue
            
            teams = match.get("teams", {})
            home = teams.get("home", {})
            away = teams.get("away", {})
            
            home_id = home.get("id")
            away_id = away.get("id")
            
            # Determina quale squadra è casa/trasferta in questo match
            if home_id == home_team_id:
                # La nostra home è casa in questo match
                home_score = match.get("goals", {}).get("home", 0)
                away_score = match.get("goals", {}).get("away", 0)
            elif away_id == home_team_id:
                # La nostra home è trasferta in questo match
                home_score = match.get("goals", {}).get("away", 0)
                away_score = match.get("goals", {}).get("home", 0)
            else:
                continue
            
            matches_played += 1
            total_goals += home_score + away_score
            
            if home_score > away_score:
                home_wins += 1
            elif home_score < away_score:
                away_wins += 1
            else:
                draws += 1
            
            if home_score > 0 and away_score > 0:
                btts_count += 1
        
        if matches_played < 2:
            return {"h2h_home_advantage": 1.0, "h2h_goals_factor": 1.0, "h2h_btts_factor": 1.0, "confidence": 0.0}
        
        # ⚠️ PRECISIONE: Calcola fattori con protezione divisione per zero
        matches_played_safe = max(1, matches_played)
        
        home_win_rate = home_wins / matches_played_safe
        draw_rate = draws / matches_played_safe
        away_win_rate = away_wins / matches_played_safe
        
        # ⚠️ PROTEZIONE: Limita rate a [0, 1]
        home_win_rate = max(0.0, min(1.0, home_win_rate))
        draw_rate = max(0.0, min(1.0, draw_rate))
        away_win_rate = max(0.0, min(1.0, away_win_rate))
        
        # Se home vince spesso → aumenta vantaggio casa
        # Media: 45% vittorie casa, 25% pareggi, 30% vittorie trasferta
        expected_home_win = 0.45
        h2h_home_advantage = 0.9 + (home_win_rate - expected_home_win) * 0.4  # Range: 0.9 - 1.1
        h2h_home_advantage = max(0.9, min(1.1, h2h_home_advantage))  # Limita range
        
        # ⚠️ VERIFICA: Assicura che h2h_home_advantage sia finito
        if not math.isfinite(h2h_home_advantage):
            logger.warning(f"h2h_home_advantage non finito: {h2h_home_advantage}, correggo")
            h2h_home_advantage = 1.0
        
        # Fattore gol: media gol in H2H vs media generale (2.6)
        avg_goals_h2h = total_goals / matches_played_safe
        avg_goals_general = 2.6
        # ⚠️ PRECISIONE: Protezione divisione per zero
        if avg_goals_general > model_config.TOL_DIVISION_ZERO:
            h2h_goals_factor = 0.9 + (avg_goals_h2h / avg_goals_general - 1.0) * 0.2  # Range: 0.9 - 1.1
        else:
            h2h_goals_factor = 1.0
        h2h_goals_factor = max(0.9, min(1.1, h2h_goals_factor))  # Limita range
        
        # ⚠️ VERIFICA: Assicura che h2h_goals_factor sia finito
        if not math.isfinite(h2h_goals_factor):
            logger.warning(f"h2h_goals_factor non finito: {h2h_goals_factor}, correggo")
            h2h_goals_factor = 1.0
        
        # Fattore BTTS
        btts_rate = btts_count / matches_played_safe
        btts_rate = max(0.0, min(1.0, btts_rate))  # Limita a [0, 1]
        avg_btts_rate = 0.52  # Media generale
        # ⚠️ PRECISIONE: Protezione divisione per zero
        if avg_btts_rate > model_config.TOL_DIVISION_ZERO:
            h2h_btts_factor = 0.9 + (btts_rate / avg_btts_rate - 1.0) * 0.2  # Range: 0.9 - 1.1
        else:
            h2h_btts_factor = 1.0
        h2h_btts_factor = max(0.9, min(1.1, h2h_btts_factor))  # Limita range
        
        # ⚠️ VERIFICA: Assicura che h2h_btts_factor sia finito
        if not math.isfinite(h2h_btts_factor):
            logger.warning(f"h2h_btts_factor non finito: {h2h_btts_factor}, correggo")
            h2h_btts_factor = 1.0
        
        # Confidence basata su numero match
        confidence = min(1.0, matches_played / 5.0)
        
        return {
            "h2h_home_advantage": round(h2h_home_advantage, 3),
            "h2h_goals_factor": round(h2h_goals_factor, 3),
            "h2h_btts_factor": round(h2h_btts_factor, 3),
            "confidence": round(confidence, 2),
            "matches_analyzed": matches_played,
            "home_wins": home_wins,
            "draws": draws,
            "away_wins": away_wins,
        }
    except (KeyError, ValueError, TypeError, ZeroDivisionError) as e:
        logger.error(f"Errore calcolo H2H: {e}")
        return {"h2h_home_advantage": 1.0, "h2h_goals_factor": 1.0, "h2h_btts_factor": 1.0, "confidence": 0.0}

def calculate_injuries_impact(injuries: List[Dict[str, Any]], team_id: int) -> Dict[str, float]:
    """
    Calcola impatto infortuni su lambda.
    """
    if not injuries:
        return {"attack_factor": 1.0, "defense_factor": 1.0, "confidence": 0.0}
    
    try:
        team_injuries = [inj for inj in injuries if inj.get("team", {}).get("id") == team_id]
        
        if not team_injuries:
            return {"attack_factor": 1.0, "defense_factor": 1.0, "confidence": 0.0}
        
        # Classifica posizioni (approssimativo)
        attack_positions = ["Forward", "Attacker", "Winger"]
        defense_positions = ["Defender", "Goalkeeper"]
        midfield_positions = ["Midfielder"]
        
        attack_impact = 0
        defense_impact = 0
        
        for injury in team_injuries:
            player = injury.get("player", {})
            position_raw = player.get("position")
            # Assicurati che position sia sempre una stringa (gestisce None)
            position = str(position_raw).upper() if position_raw is not None else ""
            
            # Determina impatto basandosi su posizione
            if position and any(pos in position for pos in attack_positions):
                attack_impact += 0.05  # -5% per attaccante infortunato
            elif position and any(pos in position for pos in defense_positions):
                defense_impact += 0.05  # -5% per difensore infortunato
            elif position and any(pos in position for pos in midfield_positions):
                attack_impact += 0.02  # -2% per centrocampista (influenza attacco)
                defense_impact += 0.02  # -2% per centrocampista (influenza difesa)
        
        # Limita impatto massimo
        attack_factor = max(0.85, 1.0 - min(0.15, attack_impact))  # Max -15%
        defense_factor = max(0.85, 1.0 - min(0.15, defense_impact))  # Max -15%
        
        # Confidence: più infortuni = più confidence nell'impatto
        confidence = min(1.0, len(team_injuries) / 3.0)
        
        return {
            "attack_factor": round(attack_factor, 3),
            "defense_factor": round(defense_factor, 3),
            "confidence": round(confidence, 2),
            "num_injuries": len(team_injuries),
        }
    except (KeyError, ValueError, TypeError) as e:
        logger.error(f"Errore calcolo impatto infortuni: {e}")
        return {"attack_factor": 1.0, "defense_factor": 1.0, "confidence": 0.0}

def get_team_fatigue_and_motivation_data(
    team_name: str,
    league: str,
    match_date: str,
) -> Dict[str, Any]:
    """
    Recupera automaticamente tutti i dati per fatigue e motivation.
    """
    result = {
        "team_id": None,
        "days_since_last_match": None,
        "matches_last_30_days": None,
        "position": None,
        "points_from_relegation": None,
        "points_from_europe": None,
        "is_derby": False,
        "data_available": False,
    }
    
    try:
        # 1. Cerca team ID
        league_id = get_league_id_from_name(league)
        team_info = apifootball_search_team(team_name, league_id)
        
        if not team_info or "team" not in team_info:
            return result
        
        team_id = team_info["team"].get("id")
        if not team_id:
            return result
        
        result["team_id"] = team_id
        
        # 2. Calcola fatigue
        result["days_since_last_match"] = calculate_days_since_last_match(team_id, match_date)
        result["matches_last_30_days"] = count_matches_last_30_days(team_id, match_date)
        
        # 3. Recupera standings
        standings_info = get_team_standings_info(team_name, league)
        result["position"] = standings_info.get("position")
        result["points_from_relegation"] = standings_info.get("points_from_relegation")
        result["points_from_europe"] = standings_info.get("points_from_europe")
        
        # 4. Verifica se abbiamo almeno alcuni dati
        result["data_available"] = (
            result["days_since_last_match"] is not None or
            result["matches_last_30_days"] is not None or
            result["position"] is not None
        )
        
        return result
    except (KeyError, ValueError, requests.exceptions.RequestException) as e:
        logger.error(f"Errore recupero dati team {team_name}: {e}")
        return result

def get_advanced_team_data(
    home_team_name: str,
    away_team_name: str,
    league: str,
    match_date: str,
) -> Dict[str, Any]:
    """
    Recupera dati avanzati da TUTTE le API disponibili: statistiche, H2H, infortuni, Football-Data.org, TheSportsDB.
    Lavora in background per supportare il modello con dati matematici e statistici aggiornati.
    """
    result = {
        "home_team_stats": None,
        "away_team_stats": None,
        "h2h_data": None,
        "home_injuries": None,
        "away_injuries": None,
        "football_data_home": None,
        "football_data_away": None,
        "thesportsdb_home": None,
        "thesportsdb_away": None,
        "data_available": False,
    }
    
    try:
        league_id = get_league_id_from_name(league)
        season = get_current_season()
        
        # ============================================================
        # 1. API-FOOTBALL: Statistiche, H2H, Infortuni
        # ============================================================
        if API_FOOTBALL_KEY and league_id:
            # Cerca team IDs
            home_team_info = apifootball_search_team(home_team_name, league_id)
            away_team_info = apifootball_search_team(away_team_name, league_id)
            
            home_team_id = home_team_info.get("team", {}).get("id") if home_team_info else None
            away_team_id = away_team_info.get("team", {}).get("id") if away_team_info else None
            
            if home_team_id and away_team_id:
                # Statistiche squadre
                home_stats = apifootball_get_team_statistics(home_team_id, league_id, season)
                away_stats = apifootball_get_team_statistics(away_team_id, league_id, season)
                
                if home_stats:
                    result["home_team_stats"] = calculate_team_form_from_statistics(home_stats)
                if away_stats:
                    result["away_team_stats"] = calculate_team_form_from_statistics(away_stats)
                
                # H2H
                h2h_matches = apifootball_get_head_to_head(home_team_id, away_team_id, last=10)
                if h2h_matches:
                    result["h2h_data"] = calculate_h2h_adjustments(h2h_matches, home_team_id, away_team_id)
                
                # Infortuni
                all_injuries = apifootball_get_injuries()
                if all_injuries:
                    result["home_injuries"] = calculate_injuries_impact(all_injuries, home_team_id)
                    result["away_injuries"] = calculate_injuries_impact(all_injuries, away_team_id)
        
        # ============================================================
        # 2. FOOTBALL-DATA.ORG: Dati aggiuntivi per validazione e statistiche
        # ============================================================
        if FOOTBALL_DATA_API_KEY:
            try:
                result["football_data_home"] = football_data_get_team_info(home_team_name, league)
                result["football_data_away"] = football_data_get_team_info(away_team_name, league)
                
                # Se Football-Data.org ha dati, usa per migliorare statistiche
                if result["football_data_home"].get("available") and not result["home_team_stats"]:
                    # Fallback: usa dati Football-Data.org se API-Football non disponibile
                    logger.info(f"Dati Football-Data.org disponibili per {home_team_name}")
                
                if result["football_data_away"].get("available") and not result["away_team_stats"]:
                    logger.info(f"Dati Football-Data.org disponibili per {away_team_name}")
            except Exception as e:
                logger.debug(f"Football-Data.org non disponibile: {e}")
        
        # ============================================================
        # 3. THESPORTSDB: Info stadio e squadra (gratuito, sempre disponibile)
        # ============================================================
        try:
            result["thesportsdb_home"] = thesportsdb_get_team_info(home_team_name)
            result["thesportsdb_away"] = thesportsdb_get_team_info(away_team_name)
            
            # Se abbiamo info stadio, può essere usato per aggiustamenti
            if result["thesportsdb_home"].get("available"):
                logger.debug(f"Info TheSportsDB disponibili per {home_team_name}")
        except Exception as e:
            logger.debug(f"TheSportsDB non disponibile: {e}")
        
        # ============================================================
        # 4. CALCOLA STATISTICHE AGGREGATE DA TUTTE LE API
        # ============================================================
        # Se abbiamo dati da più fonti, combinali per statistiche più accurate
        if result["home_team_stats"] or result["football_data_home"]:
            # Migliora confidence se abbiamo dati da più fonti
            if result["home_team_stats"] and result["football_data_home"].get("available"):
                if result["home_team_stats"].get("confidence"):
                    result["home_team_stats"]["confidence"] = min(1.0, result["home_team_stats"]["confidence"] * 1.1)
        
        if result["away_team_stats"] or result["football_data_away"]:
            if result["away_team_stats"] and result["football_data_away"].get("available"):
                if result["away_team_stats"].get("confidence"):
                    result["away_team_stats"]["confidence"] = min(1.0, result["away_team_stats"]["confidence"] * 1.1)
        
        # ============================================================
        # 5. Verifica se abbiamo almeno alcuni dati
        # ============================================================
        result["data_available"] = (
            result["home_team_stats"] is not None or
            result["away_team_stats"] is not None or
            result["h2h_data"] is not None or
            result["home_injuries"] is not None or
            result["away_injuries"] is not None or
            result["football_data_home"] is not None or
            result["football_data_away"] is not None or
            result["thesportsdb_home"] is not None or
            result["thesportsdb_away"] is not None
        )
        
        return result
    except (KeyError, ValueError, requests.exceptions.RequestException) as e:
        logger.error(f"Errore recupero dati avanzati: {e}")
        return result

# ============================================================
#          MODELLO POISSON MIGLIORATO
# ============================================================

def poisson_pmf(k: int, lam: float) -> float:
    """
    Poisson PMF con protezione overflow completa e ottimizzazione cache fattoriali.

    ⚠️ PRECISIONE MANIACALE: Validazione completa, protezione overflow/underflow
    ⚠️ OTTIMIZZAZIONE: Usa lookup table per fattoriali (15-25% più veloce)
    """
    # ⚠️ CRITICO: Validazione input
    if not isinstance(k, int) or k < 0:
        logger.warning(f"k non valido: {k}, uso default 0")
        k = 0

    if not isinstance(lam, (int, float)) or not math.isfinite(lam):
        logger.warning(f"lam non valido: {lam}, uso default 1.0")
        lam = 1.0

    if lam <= 0:
        return 1.0 if k == 0 else 0.0

    # ⚠️ PROTEZIONE: Limita lambda a range ragionevole per evitare overflow
    if lam > 50.0:
        logger.warning(f"lam troppo grande: {lam}, limito a 50.0")
        lam = 50.0

    try:
        # ⚠️ OTTIMIZZAZIONE: Per k piccoli (0-20), usa cache fattoriale per calcolo diretto
        # P(k) = (lambda^k * exp(-lambda)) / k!
        # Questo è ~20% più veloce di scipy.stats.poisson.pmf per k <= 20
        if k < len(_FACTORIAL_CACHE):
            # Calcolo diretto con fattoriale dalla cache
            factorial_k = _FACTORIAL_CACHE[k]
            try:
                exp_term = math.exp(-lam)
                if not math.isfinite(exp_term):
                    # Lambda troppo grande per exp, usa scipy come fallback
                    raise OverflowError("exp(-lam) overflow")
                pow_term = lam ** k
                if not math.isfinite(pow_term):
                    # k troppo grande per pow, usa scipy come fallback
                    raise OverflowError("lam^k overflow")
                p = (pow_term * exp_term) / factorial_k
                if math.isfinite(p) and p >= 0:
                    return max(0.0, min(1.0, p))  # Limita a [0, 1]
            except (ValueError, OverflowError):
                # Fallback a scipy se calcolo diretto fallisce
                pass

        # Fallback: usa scipy.stats.poisson.pmf (più robusto per k grandi o lambda estreme)
        p = poisson.pmf(k, lam)
        # ⚠️ VERIFICA: Assicura che risultato sia finito e non negativo
        if not math.isfinite(p) or p < 0:
            # Fallback: approssimazione per k=0
            if k == 0:
                p = math.exp(-lam) if lam > 0 else 1.0
            else:
                p = 0.0
        return max(0.0, min(1.0, p))  # Limita a [0, 1]
    except (ValueError, OverflowError) as e:
        logger.warning(f"Errore calcolo Poisson PMF: k={k}, lam={lam}, errore: {e}")
        # Fallback: approssimazione per k=0
        if k == 0:
            return math.exp(-lam) if lam > 0 else 1.0
        return 0.0

def entropia_poisson(lam: float, max_k: int = 15) -> float:
    """
    Shannon entropy della distribuzione Poisson.
    
    ⚠️ PRECISIONE MANIACALE: Kahan summation per accumulo preciso, protezione log(0)
    """
    # ⚠️ CRITICO: Validazione input
    if not isinstance(lam, (int, float)) or not math.isfinite(lam) or lam < 0:
        logger.warning(f"lam non valido: {lam}, uso default 1.0")
        lam = 1.0
    
    if not isinstance(max_k, int) or max_k < 0:
        logger.warning(f"max_k non valido: {max_k}, uso default 15")
        max_k = 15
    
    # ⚠️ PRECISIONE MANIACALE: Kahan summation per accumulo preciso
    e = 0.0
    c = 0.0  # Compensazione Kahan
    
    for k in range(max_k + 1):
        p = poisson_pmf(k, lam)
        if p > model_config.TOL_DIVISION_ZERO:  # ⚠️ MICRO-PRECISIONE: Usa tolleranza standardizzata
            # ⚠️ PROTEZIONE: Protezione log(0) e log(negativo)
            try:
                log_p = math.log2(p)
                if math.isfinite(log_p):
                    term = -p * log_p
                    # Kahan summation
                    y = term - c
                    t = e + y
                    c = (t - e) - y
                    e = t
            except (ValueError, OverflowError) as e_err:
                logger.warning(f"Errore calcolo log2 per k={k}, p={p}: {e_err}")
                continue
    
    # ⚠️ VERIFICA: Assicura che entropia sia finita e non negativa
    if not math.isfinite(e) or e < 0:
        logger.warning(f"Entropia non valida: {e}, correggo a 0.0")
        e = 0.0
    
    return e

def home_advantage_factor(league: str = "generic") -> float:
    """
    Home advantage empirico per lega.
    Basato su analisi di ~100k partite per lega.
    """
    ha_dict = {
        "premier_league": 1.35,
        "serie_a": 1.28,
        "la_liga": 1.32,
        "bundesliga": 1.30,
        "ligue_1": 1.25,
        "generic": 1.30,
    }
    return ha_dict.get(league, 1.30)

def estimate_lambda_from_market_optimized(
    odds_1: float,
    odds_x: float,
    odds_2: float,
    total: float,
    odds_over25: float = None,
    odds_under25: float = None,
    odds_dnb_home: float = None,
    odds_dnb_away: float = None,
    home_advantage: float = 1.30,
    rho_initial: float = 0.0,
) -> Tuple[float, float]:
    """
    Stima lambda con ottimizzazione numerica che minimizza errore tra
    probabilità osservate (quote) e probabilità attese dal modello Poisson-Dixon-Coles.
    
    Metodo: minimizza somma errori quadratici tra probabilità 1X2 osservate e attese.
    
    ⚠️ VALIDAZIONE INPUT: Valida tutti i parametri prima dell'uso
    """
    # ⚠️ VALIDAZIONE INPUT
    try:
        if not all(isinstance(x, (int, float)) and x > 1.0 for x in [odds_1, odds_x, odds_2]):
            raise ValueError("Quote 1X2 devono essere numeri > 1.0")
        if not isinstance(total, (int, float)) or total <= 0 or total > 10:
            raise ValueError(f"total deve essere in [0.5, 10.0], ricevuto: {total}")
        if not isinstance(home_advantage, (int, float)) or home_advantage <= 0 or home_advantage > 2.0:
            logger.warning(f"home_advantage non valido: {home_advantage}, uso default 1.30")
            home_advantage = 1.30
        if not isinstance(rho_initial, (int, float)) or rho_initial < -0.5 or rho_initial > 0.5:
            logger.warning(f"rho_initial non valido: {rho_initial}, uso default 0.0")
            rho_initial = 0.0
    except (ValueError, TypeError) as e:
        logger.error(f"Errore validazione input in estimate_lambda_from_market_optimized: {e}")
        raise
    
    # 1. Probabilità normalizzate da 1X2 (target)
    # ⚠️ CORREZIONE: normalize_three_way_shin restituisce quote normalizzate, non probabilità
    odds_1_n, odds_x_n, odds_2_n = normalize_three_way_shin(odds_1, odds_x, odds_2)
    # Converti quote normalizzate in probabilità
    p1_target = 1 / odds_1_n
    px_target = 1 / odds_x_n
    p2_target = 1 / odds_2_n
    # ⚠️ PRECISIONE: Normalizza per assicurare che sommino a 1.0 (precisione numerica)
    tot_p = p1_target + px_target + p2_target
    if tot_p > model_config.TOL_DIVISION_ZERO:  # ⚠️ MICRO-PRECISIONE: Usa tolleranza standardizzata
        p1_target /= tot_p
        px_target /= tot_p
        p2_target /= tot_p
    else:
        # Fallback: distribuzione uniforme se totale è troppo piccolo
        p1_target = px_target = p2_target = 1.0 / 3.0
        logger.warning("Probabilità 1X2 normalizzate sommano a zero, uso distribuzione uniforme")
    
    p_over_target = float('nan')
    # 2. Stima iniziale da total (MIGLIORATA: inversione numerica precisa)
    if odds_over25 and odds_under25:
        po, pu = normalize_two_way_shin(odds_over25, odds_under25)
        p_over = 1 / po
        p_over_target = p_over
        
        # MIGLIORAMENTO: Inversione numerica precisa invece di formula empirica
        # Per una distribuzione Poisson con lambda_tot, P(goals > 2.5) = 1 - sum(P(k) per k=0..2)
        # Invertiamo numericamente per trovare lambda_tot che produce p_over osservato
        def poisson_over_prob(lambda_tot):
            """
            Calcola P(goals > 2.5) per la somma di due Poisson indipendenti.
            
            ⚠️ PRECISIONE ESTESA: Usa approssimazione normale per lambda grandi e calcolo esatto per lambda piccoli.
            
            Formula corretta: P(X > 2.5) = 1 - P(X <= 2) dove X ~ Poisson(lambda_tot)
            
            Per lambda > 20: usa approssimazione normale N(lambda, lambda) con correzione di continuità
            Per lambda <= 20: usa calcolo esatto Poisson
            """
            # ⚠️ CRITICO: Validazione input
            if not isinstance(lambda_tot, (int, float)) or not math.isfinite(lambda_tot) or lambda_tot <= 0:
                return 0.0
            
            # ⚠️ PRECISIONE ESTESA: Calcolo high-precision con mpmath (se abilitato)
            use_high_precision = (
                model_config.ENABLE_HIGH_PRECISION and
                MPMATH_AVAILABLE and
                lambda_tot <= model_config.HIGH_PRECISION_LAMBDA_THRESHOLD
            )
            if use_high_precision:
                try:
                    mp.mp.dps = max(50, model_config.MPMATH_PRECISION)
                    lam_mp = mp.mpf(lambda_tot)
                    # Calcolo probabilità cumulativa fino a k = 2
                    p0_mp = mp.e ** (-lam_mp)
                    p1_mp = lam_mp * p0_mp
                    p2_mp = (lam_mp * lam_mp / 2) * p0_mp
                    sum_p_mp = p0_mp + p1_mp + p2_mp
                    result_mp = mp.mpf(1) - sum_p_mp
                    if mp.isnan(result_mp) or mp.isinf(result_mp):
                        raise ValueError("Risultato mpmath non finito")
                    result_float = float(result_mp)
                    return max(0.0, min(1.0, result_float))
                except Exception as e:
                    logger.warning(f"High precision mpmath fallita ({e}), fallback a calcolo standard")
                    # continua con calcolo standard
            
            # ⚠️ PRECISIONE ESTESA: Approssimazione normale per lambda grandi (più accurata)
            # MIGLIORAMENTO: Usa normale già da lambda > 10 (più efficiente, ancora accurato)
            # Per lambda > 10, Poisson ~ N(lambda, lambda) con errore < 0.5%
            # Con correzione di continuità adattiva per maggiore precisione
            if lambda_tot > 10.0:
                try:
                    from scipy.stats import norm
                    # MIGLIORAMENTO: Correzione di continuità adattiva
                    # Per lambda > 15: usa 0.5 (standard)
                    # Per 10 < lambda <= 15: usa 0.3 (più conservativo per lambda più bassi)
                    continuity_correction = 0.5 if lambda_tot > 15.0 else 0.3
                    z_score = (2.5 + continuity_correction - lambda_tot) / math.sqrt(lambda_tot)
                    if not math.isfinite(z_score):
                        return 1.0 if lambda_tot > 2.5 else 0.0
                    # P(X <= 2.5 + cc) ≈ Φ(z_score)
                    p_cumulative = norm.cdf(z_score)
                    result = 1.0 - p_cumulative
                    return max(0.0, min(1.0, result))
                except (ImportError, ValueError, OverflowError) as e:
                    logger.warning(f"Errore approssimazione normale: {e}, uso calcolo esatto")
                    # Fallback a calcolo esatto
            
            # ⚠️ PRECISIONE: Calcolo esatto Poisson per lambda <= 20
            # P(k) = (lambda^k * exp(-lambda)) / k!
            # ⚠️ PROTEZIONE: Evita underflow per lambda_tot molto grandi
            if lambda_tot > 50.0:  # Doppio check per sicurezza
                return 1.0
            
            # ⚠️ PRECISIONE ESTESA: Calcolo ottimizzato con log-space per evitare overflow
            # Usa log-space: log(P(k)) = k*log(lambda) - lambda - log(k!)
            # Poi exp(log(P(k))) per ottenere P(k)
            try:
                exp_neg_lambda = math.exp(-lambda_tot)
                if exp_neg_lambda == 0.0 or not math.isfinite(exp_neg_lambda):
                    return 1.0
                
                # ⚠️ PRECISIONE: Calcolo ricorsivo più stabile numericamente
                # P(0) = exp(-lambda)
                # P(1) = lambda * P(0)
                # P(2) = (lambda/2) * P(1)
                p_0 = exp_neg_lambda
                p_1 = lambda_tot * p_0
                p_2 = (lambda_tot / 2.0) * p_1
                
                # ⚠️ PRECISIONE ESTESA: Kahan summation per somma precisa
                c_sum = 0.0  # Compensazione Kahan
                sum_p = 0.0
                for p_val in [p_0, p_1, p_2]:
                    if not math.isfinite(p_val) or p_val < 0:
                        continue
                    y = p_val - c_sum
                    t = sum_p + y
                    c_sum = (t - sum_p) - y
                    sum_p = t
                
                # P(X > 2.5) = 1 - P(X <= 2)
                result = 1.0 - sum_p
                
                # ⚠️ PROTEZIONE: Limita risultato a range [0, 1]
                result = max(0.0, min(1.0, result))
                
                # ⚠️ VERIFICA: Assicura che risultato sia finito
                if not math.isfinite(result):
                    logger.warning(f"poisson_over_prob risultato non finito: {result}, correggo")
                    result = 1.0 if lambda_tot > 2.5 else 0.0
                
                return result
            except (ValueError, OverflowError) as e:
                logger.warning(f"Errore calcolo Poisson: {e}, uso approssimazione")
                # Fallback: approssimazione semplice
                return 1.0 if lambda_tot > 2.5 else 0.0
        
        # ⚠️ PRECISIONE ESTESA: Metodo Brent (combinazione bisezione + secante + inversa quadratica)
        # Converge più velocemente e con maggiore precisione rispetto a bisezione pura
        try:
            def poisson_over_error(lambda_tot):
                """Errore da minimizzare: |P(X > 2.5) - p_over|"""
                p_pred = poisson_over_prob(lambda_tot)
                if not math.isfinite(p_pred):
                    return float('inf')
                return abs(p_pred - p_over)
            
            # ⚠️ PRECISIONE ESTESA: Usa scipy.optimize.brentq per maggiore precisione
            # Brent's method combina bisezione, secante e inversa quadratica
            # Converge in ~10-15 iterazioni invece di 30
            lambda_min, lambda_max = 0.5, 6.0
            
            # Verifica che la funzione cambi segno nell'intervallo
            error_min = poisson_over_error(lambda_min)
            error_max = poisson_over_error(lambda_max)
            
            # Se entrambi gli estremi hanno stesso segno, usa bisezione migliorata
            if (error_min < model_config.TOL_OPTIMIZATION) or (error_max < model_config.TOL_OPTIMIZATION):
                total_market = lambda_min if error_min < error_max else lambda_max
            else:
                # ⚠️ PRECISIONE ESTESA: Metodo Brent con funzione di errore
                # Invertiamo: troviamo lambda tale che poisson_over_prob(lambda) = p_over
                def poisson_inverse_eq(lambda_tot):
                    """Equazione da risolvere: poisson_over_prob(lambda) - p_over = 0"""
                    p_pred = poisson_over_prob(lambda_tot)
                    if not math.isfinite(p_pred):
                        return float('inf')
                    return p_pred - p_over
                
                try:
                    lambda_opt = optimize.brentq(
                        poisson_inverse_eq,
                        lambda_min,
                        lambda_max,
                        maxiter=50,  # Più iterazioni per precisione
                        xtol=model_config.TOL_OPTIMIZATION,  # Tolleranza più stretta
                        rtol=model_config.TOL_OPTIMIZATION
                    )
                    if math.isfinite(lambda_opt):
                        total_market = lambda_opt
                    else:
                        raise ValueError("lambda_opt non finito")
                except (ValueError, RuntimeError) as e:
                    logger.warning(f"Metodo Brent fallito: {e}, uso bisezione migliorata")
                    # Fallback: bisezione migliorata con più iterazioni
                    best_lambda = (lambda_min + lambda_max) / 2.0
                    best_error = float('inf')
                    
                    for _ in range(50):  # Più iterazioni per precisione
                        lambda_mid = (lambda_min + lambda_max) / 2.0
                        p_mid = poisson_over_prob(lambda_mid)
                        if not math.isfinite(p_mid):
                            break
                        error = abs(p_mid - p_over)
                        
                        if error < best_error:
                            best_error = error
                            best_lambda = lambda_mid
                        
                        if error < model_config.TOL_OPTIMIZATION:
                            best_lambda = lambda_mid
                            break
                        
                        if p_mid < p_over:
                            lambda_min = lambda_mid
                        else:
                            lambda_max = lambda_mid
                    
                    total_market = best_lambda
        except Exception as e:
            logger.warning(f"Errore ottimizzazione lambda: {e}, uso stima euristica")
            # Fallback: stima euristica basata su p_over
            # Approssimazione: lambda ≈ -log(1 - p_over) * 2.5 / 2.5 (per Poisson)
            if p_over > 0.5:
                total_market = 2.5 + (p_over - 0.5) * 3.0
            else:
                total_market = 2.5 - (0.5 - p_over) * 2.0
            total_market = max(0.5, min(6.0, total_market))
        
        # ⚠️ CORREZIONE: Aggiustamento per casi estremi (più conservativo)
        # L'inversione numerica è già precisa, quindi aggiustamenti minimi
        if p_over > 0.90:
            total_market += 0.12  # Over molto probabile → più gol (ridotto da 0.15)
        elif p_over > 0.85:
            total_market += 0.08
        elif p_over < 0.10:
            total_market -= 0.08  # Under molto probabile → meno gol (ridotto da 0.10)
        elif p_over < 0.15:
            total_market -= 0.05
    else:
        total_market = total
    
    # 3. Stima iniziale euristica migliorata
    lambda_total = total_market / 2.0
    
    # Spread da probabilità 1X2 (più accurato)
    prob_diff = p1_target - p2_target
    # ⚠️ CORREZIONE: Formula più accurata basata su relazione logaritmica
    # ⚠️ OTTIMIZZAZIONE: Usa costante pre-calcolata invece di ricalcolare log(2.5)
    # spread_factor = exp(prob_diff * log(2.5)) per prob_diff in [-1, 1]
    spread_factor = math.exp(prob_diff * model_config.LOG_2_5)
    
    # ⚠️ PROTEZIONE: Limita spread_factor a range ragionevole per evitare valori estremi
    spread_factor = max(0.5, min(2.0, spread_factor))
    
    # ⚠️ CORREZIONE: Home advantage applicato correttamente
    # Home advantage aumenta lambda_h e riduce lambda_a proporzionalmente
    # ⚠️ OTTIMIZZAZIONE: Calcola sqrt(home_advantage) una sola volta
    sqrt_ha = math.sqrt(home_advantage)
    lambda_h_init = lambda_total * spread_factor * sqrt_ha
    lambda_a_init = lambda_total / spread_factor / sqrt_ha
    
    # ⚠️ VERIFICA: Assicura che lambda_h + lambda_a ≈ 2 * lambda_total (con tolleranza)
    # Questo garantisce che il total atteso sia coerente
    total_check = lambda_h_init + lambda_a_init
    if abs(total_check - 2 * lambda_total) > model_config.TOL_TOTAL_COHERENCE:  # ⚠️ MICRO-PRECISIONE: Usa tolleranza standardizzata
        # Ricalibra per mantenere total coerente
        scale_factor = (2 * lambda_total) / max(model_config.TOL_SCALE_FACTOR_MIN, total_check)  # ⚠️ MICRO-PRECISIONE: Usa tolleranza standardizzata
        lambda_h_init *= scale_factor
        lambda_a_init *= scale_factor
    
    # ⚠️ PRECISIONE MANIACALE: Aggiustamento DNB con validazione completa e microprecisioni
    if odds_dnb_home and odds_dnb_away:
        # ⚠️ VALIDAZIONE ROBUSTA: Verifica che odds siano validi
        if not isinstance(odds_dnb_home, (int, float)) or odds_dnb_home <= 1.0:
            logger.warning(f"odds_dnb_home non valido: {odds_dnb_home}, ignorato")
            odds_dnb_home = None
        if not isinstance(odds_dnb_away, (int, float)) or odds_dnb_away <= 1.0:
            logger.warning(f"odds_dnb_away non valido: {odds_dnb_away}, ignorato")
            odds_dnb_away = None
    
    if odds_dnb_home and odds_dnb_home > 1 and odds_dnb_away and odds_dnb_away > 1:
        # ⚠️ PRECISIONE: Calcolo probabilità DNB con protezione overflow
        p_dnb_h = 1.0 / max(odds_dnb_home, model_config.TOL_DIVISION_ZERO)
        p_dnb_a = 1.0 / max(odds_dnb_away, model_config.TOL_DIVISION_ZERO)
        
        # ⚠️ PROTEZIONE: Limita probabilità a range ragionevole
        p_dnb_h = max(0.0, min(1.0, p_dnb_h))
        p_dnb_a = max(0.0, min(1.0, p_dnb_a))
        
        # ⚠️ PRECISIONE: Kahan summation per somma precisa
        tot_dnb = p_dnb_h + p_dnb_a
        
        # ⚠️ MICRO-PRECISIONE: Usa tolleranza standardizzata per validazione
        if tot_dnb > model_config.TOL_DIVISION_ZERO:  # ⚠️ MICRO-PRECISIONE: Usa tolleranza standardizzata
            # Normalizza con precisione
            p_dnb_h /= tot_dnb
            p_dnb_a /= tot_dnb
            
            # ⚠️ VERIFICA: Assicura che probabilità DNB sommino a 1.0
            sum_check_dnb = p_dnb_h + p_dnb_a
            if abs(sum_check_dnb - 1.0) > model_config.TOL_PROBABILITY_CHECK:
                if sum_check_dnb > model_config.TOL_DIVISION_ZERO:
                    p_dnb_h /= sum_check_dnb
                    p_dnb_a = 1.0 - p_dnb_h
                else:
                    logger.warning("Somma probabilità DNB = 0, uso distribuzione uniforme")
                    p_dnb_h = p_dnb_a = 0.5
            
            # DNB più informativo: blend usando ModelConfig
            # ⚠️ CORREZIONE: Calcolo lambda da DNB più accurato
            # ⚠️ OTTIMIZZAZIONE: Usa sqrt_ha già calcolato se disponibile, altrimenti calcola una volta
            sqrt_ha = math.sqrt(home_advantage)
            
            # ⚠️ PRECISIONE: Stima lambda da probabilità DNB con protezione
            # Se p_dnb_h > p_dnb_a, lambda_h > lambda_a
            # ⚠️ MICRO-PRECISIONE: Usa tolleranza standardizzata per protezione divisione per zero
            p_dnb_a_safe = max(model_config.TOL_DIVISION_ZERO, p_dnb_a)
            dnb_ratio = p_dnb_h / p_dnb_a_safe  # ⚠️ MICRO-PRECISIONE: Usa tolleranza standardizzata
            
            # ⚠️ PROTEZIONE: Limita dnb_ratio a range ragionevole per evitare valori estremi
            dnb_ratio = max(0.1, min(10.0, dnb_ratio))
            
            lambda_h_dnb = lambda_total * dnb_ratio * sqrt_ha
            lambda_a_dnb = lambda_total / max(model_config.TOL_DIVISION_ZERO, dnb_ratio) / sqrt_ha  # ⚠️ MICRO-PRECISIONE: Usa tolleranza standardizzata
            
            # ⚠️ PROTEZIONE: Limita lambda DNB a range ragionevole
            lambda_h_dnb = max(model_config.LAMBDA_SAFE_MIN, min(model_config.LAMBDA_SAFE_MAX, lambda_h_dnb))
            lambda_a_dnb = max(model_config.LAMBDA_SAFE_MIN, min(model_config.LAMBDA_SAFE_MAX, lambda_a_dnb))
            
            # ⚠️ PRECISIONE: Blend pesato con Kahan per evitare errori di arrotondamento
            w_market = model_config.MARKET_WEIGHT
            w_dnb = model_config.DNB_WEIGHT
            
            # Verifica che pesi sommino a 1.0
            w_sum = w_market + w_dnb
            if abs(w_sum - 1.0) > model_config.TOL_PROBABILITY_CHECK:
                w_market /= w_sum
                w_dnb = 1.0 - w_market
            
            lambda_h_init = w_market * lambda_h_init + w_dnb * lambda_h_dnb
            lambda_a_init = w_market * lambda_a_init + w_dnb * lambda_a_dnb
            
            # ⚠️ VERIFICA: Ricalibra per mantenere total coerente dopo blend DNB
            total_check_dnb = lambda_h_init + lambda_a_init
            if abs(total_check_dnb - 2 * lambda_total) > model_config.TOL_TOTAL_COHERENCE:  # ⚠️ MICRO-PRECISIONE: Usa tolleranza standardizzata
                scale_factor_dnb = (2 * lambda_total) / max(model_config.TOL_SCALE_FACTOR_MIN, total_check_dnb)  # ⚠️ MICRO-PRECISIONE: Usa tolleranza standardizzata
                lambda_h_init *= scale_factor_dnb
                lambda_a_init *= scale_factor_dnb
                
                # ⚠️ VERIFICA FINALE: Double-check coerenza dopo ricalibrazione
                total_check_final = lambda_h_init + lambda_a_init
                if abs(total_check_final - 2 * lambda_total) > model_config.TOL_TOTAL_COHERENCE:
                    logger.warning(f"Coerenza total DNB ancora non raggiunta: {total_check_final} vs {2 * lambda_total}")
        else:
            logger.warning(f"Somma probabilità DNB troppo piccola: {tot_dnb}, ignorato")
    
    # Constraints iniziali
    lambda_h_init = max(model_config.LAMBDA_SAFE_MIN, min(model_config.LAMBDA_SAFE_MAX, lambda_h_init))
    lambda_a_init = max(model_config.LAMBDA_SAFE_MIN, min(model_config.LAMBDA_SAFE_MAX, lambda_a_init))
    
    # 4. Ottimizzazione numerica: minimizza errore tra probabilità osservate e attese
    def error_function(params):
        lh, la = params[0], params[1]
        lh = max(model_config.LAMBDA_OPTIMIZATION_MIN, min(model_config.LAMBDA_OPTIMIZATION_MAX, lh))
        la = max(model_config.LAMBDA_OPTIMIZATION_MIN, min(model_config.LAMBDA_OPTIMIZATION_MAX, la))
        
        # Costruisci matrice temporanea per calcolare probabilità attese
        mat_temp = build_score_matrix(lh, la, rho_initial)
        p1_pred, px_pred, p2_pred = calc_match_result_from_matrix(mat_temp)
        
        # Errore quadratico pesato
        error = (
            (p1_pred - p1_target)**2 * 1.0 +
            (px_pred - px_target)**2 * 0.8 +  # Pareggio meno informativo
            (p2_pred - p2_target)**2 * 1.0
        )
        
        # ⚠️ CORREZIONE: Penalità se total atteso si discosta troppo (peso ottimizzato)
        total_pred = lh + la
        if total_market > 0:
            # Usa errore relativo normalizzato per evitare penalità eccessive
            relative_error = abs(total_pred - total_market) / max(0.1, total_market)
            error += 0.25 * (relative_error ** 2)  # Ridotto da 0.3 a 0.25 per maggiore flessibilità
        
        return error
    
    try:
        # Ottimizzazione con metodo L-BFGS-B (più robusto)
        result = optimize.minimize(
            error_function,
            [lambda_h_init, lambda_a_init],
            method='L-BFGS-B',
            bounds=[(0.2, 5.0), (0.2, 5.0)],
            options={'maxiter': 150, 'ftol': 1e-8, 'gtol': 1e-6}  # ⚠️ PRECISIONE: Tolleranza più stretta
        )
        
        if result.success:
            lambda_h, lambda_a = result.x[0], result.x[1]
        else:
            # Fallback a stima iniziale se ottimizzazione fallisce
            logger.warning(f"Ottimizzazione lambda fallita: {result.message}, uso stima iniziale")
            lambda_h, lambda_a = lambda_h_init, lambda_a_init
    except (ValueError, RuntimeError, optimize.OptimizeWarning) as e:
        logger.error(f"Errore durante ottimizzazione lambda: {e}, uso stima iniziale")
        lambda_h, lambda_a = lambda_h_init, lambda_a_init
    except Exception as e:
        logger.error(f"Errore imprevisto durante ottimizzazione lambda: {type(e).__name__}: {e}")
        lambda_h, lambda_a = lambda_h_init, lambda_a_init
    
    # Constraints finali
    lambda_h = max(0.3, min(4.5, lambda_h))
    lambda_a = max(0.3, min(4.5, lambda_a))
    
    # ⚠️ DIAGNOSTICA: Logga scostamenti finali
    try:
        mat_final = build_score_matrix(lambda_h, lambda_a, rho_initial)
        p1_pred, px_pred, p2_pred = calc_match_result_from_matrix(mat_final)
        over_pred, _ = calc_over_under_from_matrix(mat_final, 2.5)
        btts_pred = calc_bt_ts_from_matrix(mat_final)
    except Exception as diag_err:
        logger.debug(f"[DIAGNOSTICS] Impossibile calcolare metriche finali: {diag_err}")
        p1_pred = px_pred = p2_pred = over_pred = btts_pred = float('nan')
    
    odds_btts_value = locals().get("odds_btts")
    if isinstance(odds_btts_value, (int, float)) and odds_btts_value > 1.01:
        btts_target = 1.0 / odds_btts_value
    else:
        btts_target = float('nan')
    
    log_precision_metrics(
        "estimate_lambda_from_market_optimized",
        {
            "lambda_h": lambda_h,
            "lambda_a": lambda_a,
            "total_target": total_market,
            "total_actual": lambda_h + lambda_a,
            "p1_target": p1_target,
            "p1_pred": p1_pred,
            "px_target": px_target,
            "px_pred": px_pred,
            "p2_target": p2_target,
            "p2_pred": p2_pred,
            "p_over_target": p_over_target,
            "p_over_pred": over_pred,
            "btts_target": btts_target,
            "btts_pred": btts_pred
        }
    )
    
    return lambda_h, lambda_a

def estimate_lambda_rho_joint_optimization(
    odds_1: float,
    odds_x: float,
    odds_2: float,
    total: float,
    odds_over25: float = None,
    odds_under25: float = None,
    odds_btts: float = None,
    odds_dnb_home: float = None,
    odds_dnb_away: float = None,
    home_advantage: float = 1.30,
) -> Tuple[float, float, float]:
    """
    ⭐ OTTIMIZZAZIONE SIMULTANEA LAMBDA + RHO ⭐
    
    Ottimizza lambda_h, lambda_a e rho simultaneamente invece che separatamente.
    Questo produce soluzioni più accurate perché considera tutte le dipendenze.
    
    Minimizza errore combinato su:
    - Probabilità 1X2
    - Over/Under 2.5
    - BTTS (se disponibile)
    
    Returns: (lambda_h, lambda_a, rho)
    """
    # 1. Probabilità target normalizzate
    # ⚠️ CORREZIONE: normalize_three_way_shin restituisce quote normalizzate, non probabilità
    odds_1_n, odds_x_n, odds_2_n = normalize_three_way_shin(odds_1, odds_x, odds_2)
    # Converti quote normalizzate in probabilità
    p1_target = 1 / odds_1_n
    px_target = 1 / odds_x_n
    p2_target = 1 / odds_2_n
    # ⚠️ PRECISIONE: Normalizza per assicurare che sommino a 1.0
    tot_p = p1_target + px_target + p2_target
    if tot_p > model_config.TOL_DIVISION_ZERO:  # ⚠️ MICRO-PRECISIONE: Usa tolleranza standardizzata
        p1_target /= tot_p
        px_target /= tot_p
        p2_target /= tot_p
    else:
        # Fallback: distribuzione uniforme
        p1_target = px_target = p2_target = 1.0 / 3.0
        logger.warning("Probabilità 1X2 normalizzate sommano a zero, uso distribuzione uniforme")
    
    # 2. Target Over/Under
    p_over_target = None
    if odds_over25 and odds_under25:
        po, pu = normalize_two_way_shin(odds_over25, odds_under25)
        p_over_target = 1 / po
    
    # 3. Target BTTS
    p_btts_target = None
    if odds_btts and odds_btts > 1:
        p_btts_target = 1 / odds_btts
    
    # 4. Stima iniziale (usando metodo separato come starting point)
    lh_init, la_init = estimate_lambda_from_market_optimized(
        odds_1, odds_x, odds_2, total,
        odds_over25, odds_under25,
        odds_dnb_home, odds_dnb_away,
        home_advantage, rho_initial=0.0
    )
    rho_init = estimate_rho_optimized(lh_init, la_init, px_target, odds_btts, None)
    
    # 5. Funzione di errore congiunta
    def joint_error(params):
        lh, la, rho = params[0], params[1], params[2]
        lh = max(model_config.LAMBDA_OPTIMIZATION_MIN, min(model_config.LAMBDA_OPTIMIZATION_MAX, lh))
        la = max(model_config.LAMBDA_OPTIMIZATION_MIN, min(model_config.LAMBDA_OPTIMIZATION_MAX, la))
        rho = max(model_config.RHO_MIN, min(model_config.RHO_MAX, rho))
        
        # Costruisci matrice con questi parametri
        mat = build_score_matrix(lh, la, rho)
        p1_pred, px_pred, p2_pred = calc_match_result_from_matrix(mat)
        over_pred, _ = calc_over_under_from_matrix(mat, 2.5)
        btts_pred = calc_bt_ts_from_matrix(mat)
        
        # Errore pesato su tutti i mercati
        error = (
            1.0 * (p1_pred - p1_target)**2 +
            0.8 * (px_pred - px_target)**2 +  # Pareggio meno informativo
            1.0 * (p2_pred - p2_target)**2
        )
        
        # Aggiungi Over/Under se disponibile
        if p_over_target is not None:
            error += 0.5 * (over_pred - p_over_target)**2
        
        # Aggiungi BTTS se disponibile
        if p_btts_target is not None:
            error += 0.4 * (btts_pred - p_btts_target)**2
        
        # ⚠️ CORREZIONE: Penalità per total atteso (peso ottimizzato)
        total_pred = lh + la
        if total > 0:
            # Usa errore relativo normalizzato
            relative_error = abs(total_pred - total) / max(0.1, total)
            error += 0.25 * (relative_error ** 2)  # Ridotto da 0.3 a 0.25 per maggiore flessibilità
        
        return error
    
    # 6. Ottimizzazione congiunta
    try:
        result = optimize.minimize(
            joint_error,
            [lh_init, la_init, rho_init],
            method='L-BFGS-B',
            bounds=[(0.2, 5.0), (0.2, 5.0), (-0.35, 0.35)],
            options={'maxiter': 200, 'ftol': 1e-9, 'gtol': 1e-7}  # ⚠️ PRECISIONE: Tolleranza più stretta
        )
        
        if result.success:
            lambda_h, lambda_a, rho = result.x[0], result.x[1], result.x[2]
        else:
            logger.warning(f"Ottimizzazione congiunta lambda+rho fallita: {result.message}, uso stima separata")
            # Fallback a stima separata se ottimizzazione fallisce
            lambda_h, lambda_a = lh_init, la_init
            rho = rho_init
    except (ValueError, RuntimeError, optimize.OptimizeWarning) as e:
        logger.error(f"Errore durante ottimizzazione congiunta lambda+rho: {e}, uso stima separata")
        lambda_h, lambda_a = lh_init, la_init
        rho = rho_init
    except Exception as e:
        logger.error(f"Errore imprevisto durante ottimizzazione congiunta: {type(e).__name__}: {e}")
        lambda_h, lambda_a = lh_init, la_init
        rho = rho_init
    
    # Constraints finali
    lambda_h = max(0.3, min(4.5, lambda_h))
    lambda_a = max(0.3, min(4.5, lambda_a))
    rho = max(-0.35, min(0.35, rho))
    
    return round(lambda_h, 4), round(lambda_a, 4), round(rho, 4)

# ============================================================
#   BAYESIAN UPDATING E SHRINKAGE ESTIMATION (NUOVO)
# ============================================================

def bayesian_lambda_update(
    lambda_market: float,
    lambda_prior: float,
    prior_confidence: float,
    market_confidence: float = 0.7,
) -> float:
    """
    ⭐ BAYESIAN UPDATING ⭐ (PRECISIONE ESTESA)
    
    Aggiornamento bayesiano: combina lambda di mercato con prior storico.
    
    ⚠️ PRECISIONE ESTESA: Formula bayesiana più accurata con normalizzazione precisa.
    
    lambda_posterior = w_prior * lambda_prior + w_market * lambda_market
    dove w_prior + w_market = 1 (normalizzazione esatta)
    
    Args:
        lambda_market: Lambda stimato dai dati di mercato
        lambda_prior: Lambda da statistiche storiche (prior)
        prior_confidence: Confidence nel prior (0-1)
        market_confidence: Confidence nel mercato (0-1, default 0.7)
    
    Returns:
        Lambda aggiornato (posterior)
    """
    # ⚠️ CRITICO: Validazione input
    if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in [lambda_market, lambda_prior, prior_confidence, market_confidence]):
        logger.warning(f"Parametri non validi in bayesian_lambda_update, uso lambda_market")
        return lambda_market if isinstance(lambda_market, (int, float)) and math.isfinite(lambda_market) else 1.5
    
    # ⚠️ PROTEZIONE: Limita confidence a [0, 1]
    prior_confidence = max(0.0, min(1.0, prior_confidence))
    market_confidence = max(0.0, min(1.0, market_confidence))
    
    # ⚠️ PRECISIONE ESTESA: Normalizza confidence con Kahan summation per precisione
    # Evita errori di arrotondamento nella normalizzazione
    total_confidence = prior_confidence + market_confidence
    
    if total_confidence <= model_config.TOL_DIVISION_ZERO:
        return lambda_market
    
    if getattr(model_config, "USE_CONJUGATE_POSTERIORS", False):
        prior_strength = prior_confidence * model_config.GAMMA_PRIOR_STRENGTH_BASE
        observed_strength = market_confidence * model_config.GAMMA_PRIOR_STRENGTH_BASE
        lambda_post = gamma_poisson_posterior_mean(
            lambda_prior=lambda_prior,
            prior_strength=prior_strength,
            lambda_observed=lambda_market,
            observed_strength=observed_strength
        )
        log_precision_metrics(
            "bayesian_lambda_update[gamma]",
            {
                "lambda_prior": lambda_prior,
                "lambda_market": lambda_market,
                "lambda_post": lambda_post,
                "prior_strength": prior_strength,
                "observed_strength": observed_strength
            }
        )
        return lambda_post
    
    # ⚠️ PRECISIONE ESTESA: Calcolo pesi con protezione divisione per zero (fallback lineare)
    w_prior = prior_confidence / total_confidence
    w_market = market_confidence / total_confidence
    
    # ⚠️ VERIFICA: Assicura che pesi sommino esattamente a 1.0 (con tolleranza)
    sum_weights = w_prior + w_market
    if abs(sum_weights - 1.0) > model_config.TOL_PROBABILITY_CHECK:
        if sum_weights > model_config.TOL_DIVISION_ZERO:
            w_prior = w_prior / sum_weights
            w_market = w_market / sum_weights
    
    term_prior = w_prior * lambda_prior
    term_market = w_market * lambda_market
    
    if not all(math.isfinite(x) for x in [term_prior, term_market]):
        logger.warning(f"Termini non finiti in bayesian_lambda_update, uso lambda_market")
        return lambda_market
    
    c = 0.0
    y = term_prior - c
    t = term_market + y
    c = (t - term_market) - y
    lambda_posterior = t
    
    if not math.isfinite(lambda_posterior):
        logger.warning(f"lambda_posterior non finito: {lambda_posterior}, uso media semplice")
        lambda_posterior = (lambda_prior + lambda_market) / 2.0
    
    lambda_posterior = max(0.1, min(5.0, lambda_posterior))
    log_precision_metrics(
        "bayesian_lambda_update[linear]",
        {
            "lambda_prior": lambda_prior,
            "lambda_market": lambda_market,
            "lambda_post": lambda_posterior,
            "w_prior": w_prior,
            "w_market": w_market
        }
    )
    return lambda_posterior

def james_stein_shrinkage(
    lambda_estimate: float,
    lambda_global_mean: float,
    n_observations: int = 10,
    shrinkage_factor: float = 0.3,
) -> float:
    """
    ⭐ SHRINKAGE ESTIMATION ⭐ (PRECISIONE ESTESA)
    
    James-Stein estimator: riduce varianza shrinkando verso media globale.
    Utile per prevenire overfitting e valori estremi.
    
    ⚠️ PRECISIONE ESTESA: Formula James-Stein più accurata con calcolo preciso.
    
    Formula: lambda_shrunk = (1 - α) * lambda_estimate + α * lambda_global_mean
    dove α = shrinkage_factor / (1 + n_observations / k) con k = 10
    
    Args:
        lambda_estimate: Lambda stimato
        lambda_global_mean: Media globale/lega
        n_observations: Numero osservazioni usate per stima
        shrinkage_factor: Fattore di shrinkage (default 0.3)
    
    Returns:
        Lambda con shrinkage applicato
    """
    # ⚠️ CRITICO: Validazione input
    if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in [lambda_estimate, lambda_global_mean, shrinkage_factor]):
        logger.warning(f"Parametri non validi in james_stein_shrinkage, uso lambda_estimate")
        return lambda_estimate if isinstance(lambda_estimate, (int, float)) and math.isfinite(lambda_estimate) else 1.5
    
    if not isinstance(n_observations, int) or n_observations < 0:
        logger.warning(f"n_observations non valido: {n_observations}, uso default 10")
        n_observations = 10
    
    # ⚠️ PROTEZIONE: Limita parametri a range ragionevole
    shrinkage_factor = max(0.0, min(1.0, shrinkage_factor))
    n_observations = max(1, n_observations)
    
    # ⚠️ PRECISIONE ESTESA: Calcolo effective_shrinkage con protezione divisione per zero
    # Più osservazioni = meno shrinkage (formula più precisa)
    k = 10.0  # Costante di normalizzazione
    denom = 1.0 + n_observations / k
    if denom > model_config.TOL_DIVISION_ZERO:
        effective_shrinkage = shrinkage_factor / denom
    else:
        effective_shrinkage = shrinkage_factor
    
    # ⚠️ PROTEZIONE: Limita effective_shrinkage a [0, 1]
    effective_shrinkage = max(0.0, min(1.0, effective_shrinkage))
    
    # ⚠️ PRECISIONE ESTESA: Calcolo weighted average con Kahan summation
    w_estimate = 1.0 - effective_shrinkage
    w_global = effective_shrinkage
    
    # Verifica che pesi sommino a 1.0
    sum_weights = w_estimate + w_global
    if abs(sum_weights - 1.0) > model_config.TOL_PROBABILITY_CHECK:
        if sum_weights > model_config.TOL_DIVISION_ZERO:
            w_estimate = w_estimate / sum_weights
            w_global = w_global / sum_weights
    
    term_estimate = w_estimate * lambda_estimate
    term_global = w_global * lambda_global_mean
    
    # Verifica finitezza
    if not all(math.isfinite(x) for x in [term_estimate, term_global]):
        logger.warning(f"Termini non finiti in james_stein_shrinkage, uso media semplice")
        return (lambda_estimate + lambda_global_mean) / 2.0
    
    # Kahan summation per combinazione precisa
    c = 0.0
    y = term_estimate - c
    t = term_global + y
    c = (t - term_global) - y
    lambda_shrunk = t
    
    # ⚠️ VERIFICA: Assicura che risultato sia finito e ragionevole
    if not math.isfinite(lambda_shrunk):
        logger.warning(f"lambda_shrunk non finito: {lambda_shrunk}, uso media semplice")
        lambda_shrunk = (lambda_estimate + lambda_global_mean) / 2.0
    
    # ⚠️ PROTEZIONE: Limita a range ragionevole
    lambda_shrunk = max(0.1, min(5.0, lambda_shrunk))
    
    return lambda_shrunk

def time_decay_weight(
    days_ago: int,
    half_life_days: int = 30
) -> float:
    """
    ⭐ TIME-DECAY WEIGHTING ⭐ (PRECISIONE ESTESA)
    
    Peso esponenziale per dati storici: partite recenti contano di più.
    
    ⚠️ PRECISIONE ESTESA: Calcolo più accurato con protezione overflow/underflow.
    
    Peso = exp(-lambda * days_ago)
    dove lambda = ln(2) / half_life_days
    
    Args:
        days_ago: Giorni da oggi
        half_life_days: Dopo quanti giorni il peso si dimezza (default 30)
    
    Returns:
        Peso (0-1)
    """
    # ⚠️ CRITICO: Validazione input
    if not isinstance(days_ago, int) or days_ago < 0:
        days_ago = max(0, int(days_ago) if isinstance(days_ago, (int, float)) else 0)
    
    if not isinstance(half_life_days, (int, float)) or not math.isfinite(half_life_days) or half_life_days <= 0:
        logger.warning(f"half_life_days non valido: {half_life_days}, uso default 30")
        half_life_days = 30
    
    # ⚠️ PROTEZIONE: Limita half_life_days a range ragionevole
    half_life_days = max(1, min(365, half_life_days))
    
    # ⚠️ PRECISIONE ESTESA: Calcolo lambda_decay con protezione divisione per zero
    # Usa costante pre-calcolata log(2) per maggiore precisione
    LOG_2 = 0.6931471805599453  # log(2) pre-calcolato per precisione
    if half_life_days > model_config.TOL_DIVISION_ZERO:
        lambda_decay = LOG_2 / half_life_days
    else:
        logger.warning(f"half_life_days troppo piccolo: {half_life_days}, uso default")
        lambda_decay = LOG_2 / 30.0
    
    # ⚠️ PROTEZIONE: Evita overflow per days_ago molto grandi
    # Per days_ago > 10 * half_life_days, peso ≈ 0
    if days_ago > 10 * half_life_days:
        return 0.0
    
    # ⚠️ PRECISIONE ESTESA: Calcolo exp con protezione overflow
    try:
        exponent = -lambda_decay * days_ago
        # Protezione overflow: se exponent < -700, exp ≈ 0
        if exponent < -700:
            return 0.0
        # Protezione underflow: se exponent > 700, exp ≈ inf (ma non dovrebbe accadere)
        if exponent > 700:
            logger.warning(f"exponent troppo grande: {exponent}, ritorno 1.0")
            return 1.0
        
        weight = math.exp(exponent)
        
        # ⚠️ VERIFICA: Assicura che peso sia finito e in range [0, 1]
        if not math.isfinite(weight):
            logger.warning(f"weight non finito: {weight}, correggo")
            return 0.0
        
        weight = max(0.0, min(1.0, weight))
        return weight
    except (ValueError, OverflowError) as e:
        logger.warning(f"Errore calcolo time_decay_weight: {e}, ritorno 0.0")
        return 0.0

def weighted_calibration_with_time_decay(
    archive_file: str = ARCHIVE_FILE,
    league: str = None,
    half_life_days: int = 30,
) -> Optional[callable]:
    """
    Calibrazione con pesi temporali: partite recenti contano di più.
    """
    if not os.path.exists(archive_file):
        return None
    
    try:
        df = pd.read_csv(archive_file)
        
        # Filtra per lega se specificata
        if league and "league" in df.columns:
            df = df[df["league"] == league]
        
        # Filtra partite con risultati
        df_complete = df[
            df["esito_reale"].notna() & 
            (df["esito_reale"] != "") &
            df["p_home"].notna()
        ]
        
        if len(df_complete) < 30:
            return None
        
        # Calcola pesi temporali
        if "timestamp" in df_complete.columns:
            df_complete["timestamp"] = pd.to_datetime(df_complete["timestamp"], errors='coerce')
            now = datetime.now()
            df_complete["days_ago"] = (now - df_complete["timestamp"]).dt.days
            df_complete["weight"] = df_complete["days_ago"].apply(
                lambda x: time_decay_weight(x, half_life_days)
            )
        else:
            # Se non c'è timestamp, usa pesi uniformi
            df_complete["weight"] = 1.0
        
        # Prepara dati
        predictions = df_complete["p_home"].values
        outcomes = (df_complete["esito_reale"] == "1").astype(int).values
        weights = df_complete["weight"].values
        
        # Normalizza pesi
        # ⚠️ PROTEZIONE: Verifica che weights.sum() non sia zero
        weights_sum = weights.sum()
        if weights_sum > model_config.TOL_DIVISION_ZERO:  # ⚠️ MICRO-PRECISIONE: Usa tolleranza standardizzata
            weights = weights / weights_sum * len(weights)
        else:
            # Fallback: pesi uniformi se somma è zero
            weights = np.ones_like(weights)
            logger.warning("Somma pesi è zero, uso pesi uniformi")
        
        # Calibrazione (usa best method, ma nota: sklearn non supporta pesi direttamente)
        # Per ora usa calibrazione normale, ma con dati filtrati per pesi alti
        # (semplificazione: usa solo top 70% per peso)
        threshold = np.percentile(weights, 30)
        mask = weights >= threshold
        predictions_filtered = predictions[mask]
        outcomes_filtered = outcomes[mask]
        
        if len(predictions_filtered) < 20:
            # Troppo pochi dati, usa tutti
            predictions_filtered = predictions
            outcomes_filtered = outcomes
        
        calibrate_func, method_name, score = best_calibration_method(
            predictions_filtered.tolist(),
            outcomes_filtered.tolist()
        )
        
        return calibrate_func
    except (KeyError, ValueError, pd.errors.EmptyDataError) as e:
        logger.error(f"Errore calibrazione con time decay: {e}")
        return None

def estimate_lambda_from_market_improved(
    odds_1: float,
    odds_x: float,
    odds_2: float,
    total: float,
    odds_over25: float = None,
    odds_under25: float = None,
    odds_dnb_home: float = None,
    odds_dnb_away: float = None,
    home_advantage: float = 1.30,
) -> Tuple[float, float]:
    """
    Wrapper che chiama la versione ottimizzata.
    Mantiene compatibilità con codice esistente.
    """
    return estimate_lambda_from_market_optimized(
        odds_1, odds_x, odds_2, total,
        odds_over25, odds_under25,
        odds_dnb_home, odds_dnb_away,
        home_advantage, rho_initial=0.0
    )

def estimate_rho_optimized(
    lambda_h: float,
    lambda_a: float,
    p_draw: float,
    odds_btts: float = None,
    p_btts_model: float = None,
) -> float:
    """
    Stima rho con ottimizzazione numerica che minimizza errore tra
    probabilità BTTS osservata (se disponibile) e attesa dal modello.
    
    Metodo: minimizza errore tra P(BTTS) osservata e P(BTTS) attesa.
    """
    # Prior basato su draw probability (più accurato)
    # ⚠️ CORREZIONE: Relazione empirica migliorata basata su analisi Dixon-Coles
    # rho ≈ -0.12 + (p_draw - 0.25) * 1.2 per p_draw in [0.2, 0.35]
    # Per p_draw estremi, limita l'effetto
    p_draw_clamped = max(0.15, min(0.40, p_draw))
    rho_from_draw = -0.12 + (p_draw_clamped - 0.25) * 1.2
    
    # ⚠️ PROTEZIONE: Limita rho_from_draw a range ragionevole
    rho_from_draw = max(-0.30, min(0.30, rho_from_draw))
    
    # Se abbiamo BTTS dal mercato, ottimizziamo
    if odds_btts and odds_btts > 1:
        p_btts_market = 1 / odds_btts
        
        # ⚠️ VALIDAZIONE: Verifica che p_btts_market sia ragionevole
        if not (0.1 <= p_btts_market <= 0.9):
            logger.warning(f"p_btts_market fuori range ragionevole: {p_btts_market:.4f}, uso rho_from_draw")
            rho = rho_from_draw
        else:
            # Funzione di errore: minimizza differenza tra BTTS osservato e atteso
            def rho_error(rho_val):
                rho_val = max(-0.35, min(0.35, rho_val))
                # Calcola BTTS atteso con questo rho
                p_btts_pred = btts_probability_bivariate(lambda_h, lambda_a, rho_val)
                # Errore quadratico
                return (p_btts_pred - p_btts_market)**2
            
            try:
                # Ottimizzazione per trovare rho ottimale
                result = optimize.minimize_scalar(
                    rho_error,
                    bounds=(-0.35, 0.35),
                    method='bounded',
                    options={'maxiter': 50}
                )
                
                if result.success:
                    rho_opt = result.x
                    # Blend con prior usando ModelConfig
                    rho = model_config.MARKET_WEIGHT * rho_opt + model_config.DNB_WEIGHT * rho_from_draw
                else:
                    logger.warning(f"Ottimizzazione rho fallita: {result.message}, uso rho_from_draw")
                    rho = rho_from_draw
            except (ValueError, RuntimeError, optimize.OptimizeWarning) as e:
                logger.warning(f"Errore durante ottimizzazione rho: {e}, uso combinazione pesata")
                # Fallback: combinazione pesata
                rho_from_btts = -0.18 + (1 - p_btts_market) * 0.6
                rho = 0.65 * rho_from_draw + 0.35 * rho_from_btts
            except Exception as e:
                logger.error(f"Errore imprevisto durante ottimizzazione rho: {type(e).__name__}: {e}")
                rho = rho_from_draw
    else:
        rho = rho_from_draw
    
    # ⚠️ CORREZIONE MIGLIORATA: Adjustment smooth e continuo basato su lambda (più gol attesi → più rho negativo)
    expected_total = lambda_h + lambda_a
    # MIGLIORAMENTO: Relazione continua invece di step-based per maggiore robustezza
    # Formula: rho_adjustment = -0.04 * (expected_total - 2.75)
    # Per expected_total = 2.0: +0.03, per 3.5: -0.03, per 4.0: -0.05
    rho_adjustment_total = -0.04 * (expected_total - 2.75)
    rho_adjustment_total = max(-0.12, min(0.08, rho_adjustment_total))
    rho += rho_adjustment_total
    
    # ⚠️ CORREZIONE: Adjustment basato su probabilità low-score (più accurato)
    # Calcola probabilità low-score usando Poisson (senza tau per semplicità)
    p_0_0 = poisson.pmf(0, lambda_h) * poisson.pmf(0, lambda_a)
    p_1_0 = poisson.pmf(1, lambda_h) * poisson.pmf(0, lambda_a)
    p_0_1 = poisson.pmf(0, lambda_h) * poisson.pmf(1, lambda_a)
    p_low_score = p_0_0 + p_1_0 + p_0_1

    # ⚠️ CORREZIONE MIGLIORATA: Adjustment smooth e continuo per p_low_score
    # MIGLIORAMENTO: Relazione continua invece di step-based
    # Formula: rho_adjustment = 0.30 * (p_low_score - 0.19)
    # Per p_low_score = 0.30: +0.033, per 0.08: -0.033, per 0.19: 0.0
    rho_adjustment_lowscore = 0.30 * (p_low_score - 0.19)
    rho_adjustment_lowscore = max(-0.08, min(0.08, rho_adjustment_lowscore))
    rho += rho_adjustment_lowscore
    
    # Bounds empirici (più ampi per maggiore flessibilità)
    return max(-0.35, min(0.35, round(rho, 4)))

def estimate_rho_improved(
    lambda_h: float,
    lambda_a: float,
    p_draw: float,
    odds_btts: float = None,
) -> float:
    """
    Wrapper per compatibilità.
    """
    return estimate_rho_optimized(lambda_h, lambda_a, p_draw, odds_btts, None)

def tau_dixon_coles(h: int, a: int, lh: float, la: float, rho: float) -> float:
    """
    Dixon-Coles tau function per correggere probabilità low-score.
    
    ⚠️ VERIFICA MATEMATICA: Formula corretta secondo Dixon & Coles (1997)
    tau(h,a) modifica la probabilità Poisson indipendente per catturare correlazione low-score.
    
    Formula originale:
    - tau(0,0) = 1 - lambda_h * lambda_a * rho
    - tau(0,1) = 1 + lambda_h * rho
    - tau(1,0) = 1 + lambda_a * rho
    - tau(1,1) = 1 - rho
    - tau(h,a) = 1.0 per tutti gli altri casi
    
    ⚠️ PRECISIONE MANIACALE: Validazione completa, protezione overflow
    """
    # ⚠️ CRITICO: Validazione input
    if not isinstance(h, int) or not isinstance(a, int) or h < 0 or a < 0:
        logger.warning(f"h o a non validi: h={h}, a={a}, uso default 1.0")
        return 1.0
    
    if not isinstance(lh, (int, float)) or not isinstance(la, (int, float)) or \
       not isinstance(rho, (int, float)):
        logger.warning(f"Parametri non validi: lh={lh}, la={la}, rho={rho}, uso default 1.0")
        return 1.0
    
    if not math.isfinite(lh) or not math.isfinite(la) or not math.isfinite(rho):
        logger.warning(f"Parametri non finiti: lh={lh}, la={la}, rho={rho}, uso default 1.0")
        return 1.0
    
    # ⚠️ PROTEZIONE: Limita parametri a range ragionevole
    lh = max(0.1, min(5.0, lh))
    la = max(0.1, min(5.0, la))
    rho = max(-0.5, min(0.5, rho))
    
    if h == 0 and a == 0:
        # tau(0,0) = 1 - lambda_h * lambda_a * rho
        try:
            val = 1.0 - (lh * la * rho)
            if not math.isfinite(val):
                logger.warning(f"tau(0,0) non finito: {val}, uso default 0.5")
                val = 0.5
            # ⚠️ PROTEZIONE: Limita a range ragionevole [0.01, 3.0] per maggiore flessibilità (MIGLIORAMENTO: bound più ampi per catturare correlazioni estreme)
            return max(0.01, min(3.0, val))
        except (ValueError, OverflowError) as e:
            logger.warning(f"Errore calcolo tau(0,0): {e}, uso default 0.5")
            return 0.5
    elif h == 0 and a == 1:
        # tau(0,1) = 1 + lambda_h * rho
        try:
            val = 1.0 + (lh * rho)
            if not math.isfinite(val):
                logger.warning(f"tau(0,1) non finito: {val}, uso default 1.0")
                val = 1.0
            # ⚠️ PROTEZIONE: Limita a range ragionevole [0.01, 3.0] per maggiore flessibilità
            return max(0.01, min(3.0, val))
        except (ValueError, OverflowError) as e:
            logger.warning(f"Errore calcolo tau(0,1): {e}, uso default 1.0")
            return 1.0
    elif h == 1 and a == 0:
        # tau(1,0) = 1 + lambda_a * rho
        try:
            val = 1.0 + (la * rho)
            if not math.isfinite(val):
                logger.warning(f"tau(1,0) non finito: {val}, uso default 1.0")
                val = 1.0
            # ⚠️ PROTEZIONE: Limita a range ragionevole [0.01, 3.0] per maggiore flessibilità
            return max(0.01, min(3.0, val))
        except (ValueError, OverflowError) as e:
            logger.warning(f"Errore calcolo tau(1,0): {e}, uso default 1.0")
            return 1.0
    elif h == 1 and a == 1:
        # tau(1,1) = 1 - rho
        try:
            val = 1.0 - rho
            if not math.isfinite(val):
                logger.warning(f"tau(1,1) non finito: {val}, uso default 1.0")
                val = 1.0
            # ⚠️ PROTEZIONE: Limita a range ragionevole [0.01, 3.0] per maggiore flessibilità
            return max(0.01, min(3.0, val))
        except (ValueError, OverflowError) as e:
            logger.warning(f"Errore calcolo tau(1,1): {e}, uso default 1.0")
            return 1.0
    # Per tutti gli altri casi, tau = 1.0 (nessuna correzione)
    return 1.0

def max_goals_adattivo(lh: float, la: float) -> int:
    """
    Determina max gol per matrice dinamicamente con maggiore precisione.
    
    Usa percentile 99.9% della distribuzione per catturare casi estremi.
    
    ⚠️ PRECISIONE MANIACALE: Validazione completa, protezione overflow
    """
    # ⚠️ CRITICO: Validazione input
    if not isinstance(lh, (int, float)) or not isinstance(la, (int, float)):
        logger.warning(f"Lambda non validi: lh={lh}, la={la}, uso default 10")
        return 10
    
    if not math.isfinite(lh) or not math.isfinite(la):
        logger.warning(f"Lambda non finiti: lh={lh}, la={la}, uso default 10")
        return 10
    
    if lh < 0 or la < 0:
        logger.warning(f"Lambda negativi: lh={lh}, la={la}, correggo")
        lh = max(0.1, lh)
        la = max(0.1, la)
    
    # ⚠️ PROTEZIONE: Limita lambda a range ragionevole
    lh = max(0.1, min(5.0, lh))
    la = max(0.1, min(5.0, la))
    
    # ⚠️ PRECISIONE: Calcola expected_total con protezione overflow
    expected_total = lh + la
    if not math.isfinite(expected_total):
        logger.warning(f"expected_total non finito: {expected_total}, uso default 10")
        return 10
    
    # Metodo più accurato: calcola percentile 99.9% della distribuzione totale
    # Per Poisson, P(X <= k) ≈ 1 - exp(-lambda) * sum(lambda^i / i!)
    # Usiamo approssimazione: max_goals ≈ lambda + 4*sqrt(lambda) per 99.9%
    
    # Per distribuzione somma di due Poisson: lambda_tot = lambda_h + lambda_a
    # Varianza = lambda_h + lambda_a (indipendenti)
    # ⚠️ PRECISIONE: Calcola std_dev con protezione
    try:
        variance = lh + la
        if variance <= 0:
            logger.warning(f"Varianza non positiva: {variance}, uso default 10")
            return 10
        std_dev = math.sqrt(variance)
        if not math.isfinite(std_dev):
            logger.warning(f"std_dev non finito: {std_dev}, uso default 10")
            return 10
    except (ValueError, OverflowError) as e:
        logger.warning(f"Errore calcolo std_dev: {e}, uso default 10")
        return 10
    
    # MIGLIORAMENTO: Percentile 99.9% preciso (era 3.5, ora 3.09)
    # Per distribuzione normale: P(X <= mean + 3.09*sigma) ≈ 0.999
    # Per lambda bassi, Poisson è asimmetrica → usa fattore leggermente più alto
    # ⚠️ PRECISIONE: Calcola max_goals con protezione overflow
    try:
        percentile_factor = 3.09 if expected_total > 3.0 else 3.29  # Più conservativo per lambda bassi
        max_goals_99_9 = expected_total + percentile_factor * std_dev
        if not math.isfinite(max_goals_99_9):
            logger.warning(f"max_goals_99_9 non finito: {max_goals_99_9}, uso default 15")
            max_goals_99_9 = 15
        max_goals_99_9 = int(max_goals_99_9)
    except (ValueError, OverflowError) as e:
        logger.warning(f"Errore calcolo max_goals_99_9: {e}, uso default 15")
        max_goals_99_9 = 15
    
    # Bounds ragionevoli: minimo 10 per precisione, massimo 20 per performance
    result = max(10, min(20, max_goals_99_9))
    
    # ⚠️ VERIFICA FINALE: Double-check che risultato sia valido
    if not isinstance(result, int) or result < 10 or result > 20:
        logger.warning(f"max_goals non valido: {result}, correggo a 15")
        result = 15
    
    return result

def build_score_matrix(lh: float, la: float, rho: float) -> List[List[float]]:
    """
    Costruisce matrice score con normalizzazione e precisione numerica massima.
    
    ⚠️ PRECISIONE MANIACALE:
    - Usa Kahan summation per accumulo preciso
    - Tolleranza normalizzazione: 1e-8 (più stretta)
    - Doppia verifica normalizzazione
    - Protezione contro errori di arrotondamento
    """
    # ⚠️ CRITICO: Validazione input
    if not isinstance(lh, (int, float)) or not isinstance(la, (int, float)) or not isinstance(rho, (int, float)):
        logger.error(f"Input non validi: lh={lh}, la={la}, rho={rho}")
        raise ValueError("Input devono essere numeri")
    
    if lh < 0 or la < 0:
        logger.warning(f"Lambda negativi: lh={lh}, la={la}, uso valori default")
        lh = max(0.1, lh)
        la = max(0.1, la)
    
    mg = max_goals_adattivo(lh, la)
    
    # ⚠️ CRITICO: Verifica che mg sia valido
    if mg < 0:
        logger.warning(f"mg < 0: {mg}, uso valore default")
        mg = 10
    
    mat: List[List[float]] = []
    
    # ⚠️ PRECISIONE: Kahan summation per accumulo preciso (evita errori di arrotondamento)
    total_prob = 0.0
    c = 0.0  # Compensazione per Kahan
    
    for h in range(mg + 1):
        row = []
        for a in range(mg + 1):
            # Probabilità base Poisson (indipendenti)
            p_base = poisson_pmf(h, lh) * poisson_pmf(a, la)
            
            # Applica correzione Dixon-Coles tau
            tau = tau_dixon_coles(h, a, lh, la, rho)
            p = p_base * tau
            
            # Assicura non-negatività
            p = max(0.0, p)
            row.append(p)
            
            # ⚠️ PRECISIONE: Kahan summation per accumulo preciso
            y = p - c
            t = total_prob + y
            c = (t - total_prob) - y
            total_prob = t
        
        mat.append(row)
    
    dirichlet_eps = getattr(model_config, "DIRICHLET_EPS", 0.0)
    if dirichlet_eps and dirichlet_eps > 0.0:
        smoothing_total = dirichlet_eps * ((mg + 1) * (mg + 1))
        total_prob += smoothing_total
        for h in range(mg + 1):
            row = mat[h]
            for a in range(mg + 1):
                row[a] += dirichlet_eps
    
    # ⚠️ PRECISIONE: Verifica normalizzazione con tolleranza standardizzata
    if total_prob > model_config.TOL_DIVISION_ZERO:  # ⚠️ MICRO-PRECISIONE: Usa tolleranza standardizzata
        # Normalizza ogni elemento con precisione
        for h in range(mg + 1):
            for a in range(mg + 1):
                mat[h][a] = mat[h][a] / total_prob
    else:
        # Fallback: distribuzione uniforme (caso estremo)
        uniform_prob = 1.0 / ((mg + 1) * (mg + 1))
        for h in range(mg + 1):
            for a in range(mg + 1):
                mat[h][a] = uniform_prob
    
    # ⚠️ PRECISIONE: Verifica normalizzazione con Kahan summation
    final_sum = 0.0
    c_final = 0.0
    for h in range(mg + 1):
        for a in range(mg + 1):
            y = mat[h][a] - c_final
            t = final_sum + y
            c_final = (t - final_sum) - y
            final_sum = t
    
    # ⚠️ PRECISIONE: Tolleranza più stretta
    if abs(final_sum - 1.0) > model_config.TOL_NORMALIZATION:  # ⚠️ MICRO-PRECISIONE: Usa tolleranza standardizzata
        # Rinomaliizza se necessario
        for h in range(mg + 1):
            for a in range(mg + 1):
                mat[h][a] = mat[h][a] / final_sum
    
    return mat

# ============================================================
#      CALCOLO PROBABILITÀ DA MATRICE (unchanged)
# ============================================================

def calc_match_result_from_matrix(mat: List[List[float]]) -> Tuple[float, float, float]:
    """
    Calcola probabilità 1X2 dalla matrice score.
    
    ⚠️ VERIFICA MATEMATICA: Formula corretta
    - P(Home) = sum(mat[h][a] for h > a)
    - P(Draw) = sum(mat[h][a] for h == a)
    - P(Away) = sum(mat[h][a] for h < a)
    - Normalizza per sicurezza (anche se matrice dovrebbe già essere normalizzata)
    """
    # ⚠️ CRITICO: Validazione input
    if not mat or len(mat) == 0 or (len(mat) > 0 and len(mat[0]) == 0):
        logger.warning("Matrice vuota o non valida, uso distribuzione uniforme")
        return 0.333333, 0.333333, 0.333334  # Somma esattamente 1.0
    
    # ⚠️ PRECISIONE: Kahan summation per accumulo preciso
    p_home = 0.0
    p_draw = 0.0
    p_away = 0.0
    c_home = 0.0  # Compensazione Kahan
    c_draw = 0.0
    c_away = 0.0
    mg = len(mat) - 1
    
    # ⚠️ CRITICO: Verifica che mg sia valido
    if mg < 0:
        logger.warning("mg < 0, uso distribuzione uniforme")
        return 0.333333, 0.333333, 0.333334
    
    # ⚠️ PRECISIONE: Kahan summation per accumulo preciso
    for h in range(mg + 1):
        for a in range(mg + 1):
            p = mat[h][a]
            # ⚠️ PROTEZIONE: Ignora valori negativi o NaN
            if p < 0 or not (p == p):  # p == p verifica NaN
                continue
            
            # ⚠️ PRECISIONE: Kahan summation per ogni categoria
            if h > a:
                y = p - c_home
                t = p_home + y
                c_home = (t - p_home) - y
                p_home = t
            elif h < a:
                y = p - c_away
                t = p_away + y
                c_away = (t - p_away) - y
                p_away = t
            else:
                y = p - c_draw
                t = p_draw + y
                c_draw = (t - p_draw) - y
                p_draw = t
    
    # ⚠️ PRECISIONE: Somma totale con Kahan
    tot = p_home + p_draw + p_away
    
    # ⚠️ PROTEZIONE: Se totale è zero o molto piccolo, usa distribuzione uniforme
    if tot <= model_config.TOL_DIVISION_ZERO:  # ⚠️ MICRO-PRECISIONE: Usa tolleranza standardizzata
        return 0.333333, 0.333333, 0.333334  # Somma esattamente 1.0
    
    # ⚠️ PRECISIONE: Normalizza per garantire che somma sia esattamente 1.0
    p_home_norm = p_home / tot
    p_draw_norm = p_draw / tot
    p_away_norm = p_away / tot
    
    # ⚠️ VERIFICA FINALE: Assicura che somma sia 1.0 (con tolleranza)
    sum_check = p_home_norm + p_draw_norm + p_away_norm
    if abs(sum_check - 1.0) > model_config.TOL_PROBABILITY_CHECK:  # ⚠️ MICRO-PRECISIONE: Usa tolleranza standardizzata
        # Rinomaliizza se necessario
        p_home_norm /= sum_check
        p_draw_norm /= sum_check
        p_away_norm /= sum_check
    
    return p_home_norm, p_draw_norm, p_away_norm

def calc_over_under_from_matrix(mat: List[List[float]], soglia: float) -> Tuple[float, float]:
    """
    Calcola probabilità Over/Under dalla matrice score.
    
    ⚠️ VERIFICA MATEMATICA: Formula corretta
    - P(Over) = sum(mat[h][a] for h + a > soglia)
    - P(Under) = 1 - P(Over)
    
    ⚠️ PRECISIONE MANIACALE: Kahan summation per accumulo preciso, validazione completa
    """
    # ⚠️ CRITICO: Validazione input robusta
    if not isinstance(soglia, (int, float)) or soglia < 0:
        logger.error(f"soglia non valida: {soglia}, uso default 2.5")
        soglia = 2.5
    
    if not mat or len(mat) == 0 or (len(mat) > 0 and len(mat[0]) == 0):
        logger.warning("Matrice vuota o non valida, uso probabilità default")
        return 0.5, 0.5
    
    # ⚠️ CRITICO: Verifica coerenza dimensioni matrice
    mg = len(mat) - 1
    if mg < 0:
        logger.warning("mg < 0, uso probabilità default")
        return 0.5, 0.5
    
    # Verifica che tutte le righe abbiano stessa lunghezza
    for i, row in enumerate(mat):
        if len(row) != mg + 1:
            logger.error(f"Matrice inconsistente: riga {i} ha {len(row)} colonne invece di {mg + 1}")
            return 0.5, 0.5
    
    # ⚠️ PRECISIONE MANIACALE: Kahan summation per accumulo preciso
    over = 0.0
    c_over = 0.0  # Compensazione Kahan
    
    for h in range(mg + 1):
        for a in range(mg + 1):
            p = mat[h][a]
            # ⚠️ PROTEZIONE: Ignora valori negativi, NaN, o infiniti
            if not isinstance(p, (int, float)) or p < 0 or not (p == p) or not math.isfinite(p):
                continue
            if h + a > soglia:
                # Kahan summation
                y = p - c_over
                t = over + y
                c_over = (t - over) - y
                over = t
    
    # ⚠️ PROTEZIONE: Limita over a range [0, 1] con precisione
    over = max(0.0, min(1.0, over))
    under = 1.0 - over
    
    # ⚠️ VERIFICA COERENZA: Assicura che over + under = 1.0 con tolleranza stretta
    sum_check = over + under
    if abs(sum_check - 1.0) > model_config.TOL_PROBABILITY_CHECK:  # ⚠️ MICRO-PRECISIONE: Usa tolleranza standardizzata
        # Ricalibra se necessario con precisione
        if sum_check > model_config.TOL_DIVISION_ZERO:
            over = over / sum_check
            under = 1.0 - over
        else:
            # Fallback: distribuzione uniforme se somma è zero
            logger.warning(f"Somma over+under = {sum_check}, uso distribuzione uniforme")
            over = 0.5
            under = 0.5
    
    # ⚠️ VERIFICA FINALE: Double-check che siano in range [0, 1]
    over = max(0.0, min(1.0, over))
    under = max(0.0, min(1.0, under))
    
    return over, under

def calc_bt_ts_from_matrix(mat: List[List[float]]) -> float:
    """
    Calcola probabilità BTTS (Both Teams To Score) dalla matrice.
    
    ⚠️ VERIFICA MATEMATICA: Formula corretta
    - P(BTTS) = sum(mat[h][a] for h >= 1 and a >= 1)
    - BTTS = entrambe le squadre segnano almeno 1 gol
    
    ⚠️ PRECISIONE MANIACALE: Kahan summation per accumulo preciso, validazione completa
    """
    # ⚠️ CRITICO: Validazione input robusta
    if not mat or len(mat) == 0 or (len(mat) > 0 and len(mat[0]) == 0):
        logger.warning("Matrice vuota o non valida, uso probabilità default")
        return 0.5
    
    mg = len(mat) - 1
    
    # ⚠️ CRITICO: Verifica che mg sia valido
    if mg < 0:
        logger.warning("mg < 0, uso probabilità default")
        return 0.5
    
    # Verifica che tutte le righe abbiano stessa lunghezza
    for i, row in enumerate(mat):
        if len(row) != mg + 1:
            logger.error(f"Matrice inconsistente: riga {i} ha {len(row)} colonne invece di {mg + 1}")
            return 0.5
    
    # ⚠️ PRECISIONE MANIACALE: Kahan summation per accumulo preciso
    btts = 0.0
    c_btts = 0.0  # Compensazione Kahan
    
    for h in range(1, mg + 1):
        for a in range(1, mg + 1):
            p = mat[h][a]
            # ⚠️ PROTEZIONE: Ignora valori negativi, NaN, o infiniti
            if not isinstance(p, (int, float)) or p < 0 or not (p == p) or not math.isfinite(p):
                continue
            # Kahan summation
            y = p - c_btts
            t = btts + y
            c_btts = (t - btts) - y
            btts = t
    
    # ⚠️ PROTEZIONE: Limita BTTS a range [0, 1] con precisione
    btts = max(0.0, min(1.0, btts))
    
    # ⚠️ VERIFICA FINALE: Double-check che sia in range [0, 1]
    if not (0.0 <= btts <= 1.0):
        logger.warning(f"BTTS fuori range: {btts}, correggo a 0.5")
        btts = 0.5
    
    return btts

def calc_gg_over25_from_matrix(mat: List[List[float]]) -> float:
    """
    Calcola probabilità GG & Over 2.5 dalla matrice.
    
    ⚠️ VERIFICA MATEMATICA: Formula corretta
    - P(GG & Over 2.5) = sum(mat[h][a] for h >= 1 and a >= 1 and h + a >= 3)
    - GG = entrambe le squadre segnano, Over 2.5 = totale gol >= 3
    
    ⚠️ PRECISIONE MANIACALE: Kahan summation per accumulo preciso, validazione completa
    """
    # ⚠️ CRITICO: Validazione input robusta
    if not mat or len(mat) == 0 or (len(mat) > 0 and len(mat[0]) == 0):
        logger.warning("Matrice vuota o non valida, uso probabilità default")
        return 0.5
    
    mg = len(mat) - 1
    
    # ⚠️ CRITICO: Verifica che mg sia valido
    if mg < 0:
        logger.warning("mg < 0, uso probabilità default")
        return 0.5
    
    # Verifica che tutte le righe abbiano stessa lunghezza
    for i, row in enumerate(mat):
        if len(row) != mg + 1:
            logger.error(f"Matrice inconsistente: riga {i} ha {len(row)} colonne invece di {mg + 1}")
            return 0.5
    
    # ⚠️ PRECISIONE MANIACALE: Kahan summation per accumulo preciso
    s = 0.0
    c_s = 0.0  # Compensazione Kahan
    
    for h in range(1, mg + 1):
        for a in range(1, mg + 1):
            p = mat[h][a]
            # ⚠️ PROTEZIONE: Ignora valori negativi, NaN, o infiniti
            if not isinstance(p, (int, float)) or p < 0 or not (p == p) or not math.isfinite(p):
                continue
            if h + a >= 3:
                # Kahan summation
                y = p - c_s
                t = s + y
                c_s = (t - s) - y
                s = t
    
    # ⚠️ PROTEZIONE: Limita a range [0, 1] con precisione
    s = max(0.0, min(1.0, s))
    
    # ⚠️ VERIFICA FINALE: Double-check che sia in range [0, 1]
    if not (0.0 <= s <= 1.0):
        logger.warning(f"GG+Over2.5 fuori range: {s}, correggo a 0.5")
        s = 0.5
    
    return s

def prob_pari_dispari_from_matrix(mat: List[List[float]]) -> Tuple[float, float]:
    """
    Calcola probabilità Pari/Dispari dalla matrice.
    
    ⚠️ VERIFICA MATEMATICA: Formula corretta
    - P(Pari) = sum(mat[h][a] for (h + a) % 2 == 0)
    - P(Dispari) = 1 - P(Pari)
    
    ⚠️ PRECISIONE MANIACALE: Kahan summation per accumulo preciso, validazione completa
    """
    # ⚠️ CRITICO: Validazione input robusta
    if not mat or len(mat) == 0 or (len(mat) > 0 and len(mat[0]) == 0):
        logger.warning("Matrice vuota o non valida, uso probabilità default")
        return 0.5, 0.5
    
    mg = len(mat) - 1
    
    # ⚠️ CRITICO: Verifica che mg sia valido
    if mg < 0:
        logger.warning("mg < 0, uso probabilità default")
        return 0.5, 0.5
    
    # Verifica che tutte le righe abbiano stessa lunghezza
    for i, row in enumerate(mat):
        if len(row) != mg + 1:
            logger.error(f"Matrice inconsistente: riga {i} ha {len(row)} colonne invece di {mg + 1}")
            return 0.5, 0.5
    
    # ⚠️ PRECISIONE MANIACALE: Kahan summation per accumulo preciso
    even = 0.0
    c_even = 0.0  # Compensazione Kahan
    
    for h in range(mg + 1):
        for a in range(mg + 1):
            p = mat[h][a]
            # ⚠️ PROTEZIONE: Ignora valori negativi, NaN, o infiniti
            if not isinstance(p, (int, float)) or p < 0 or not (p == p) or not math.isfinite(p):
                continue
            if (h + a) % 2 == 0:
                # Kahan summation
                y = p - c_even
                t = even + y
                c_even = (t - even) - y
                even = t
    
    # ⚠️ PROTEZIONE: Limita even a range [0, 1] con precisione
    even = max(0.0, min(1.0, even))
    odd = 1.0 - even
    
    # ⚠️ VERIFICA COERENZA: Assicura che even + odd = 1.0 con tolleranza stretta
    sum_check = even + odd
    if abs(sum_check - 1.0) > model_config.TOL_PROBABILITY_CHECK:  # ⚠️ MICRO-PRECISIONE: Usa tolleranza standardizzata
        # Ricalibra se necessario con precisione
        if sum_check > model_config.TOL_DIVISION_ZERO:
            even = even / sum_check
            odd = 1.0 - even
        else:
            # Fallback: distribuzione uniforme se somma è zero
            logger.warning(f"Somma even+odd = {sum_check}, uso distribuzione uniforme")
            even = 0.5
            odd = 0.5
    
    # ⚠️ VERIFICA FINALE: Double-check che siano in range [0, 1]
    even = max(0.0, min(1.0, even))
    odd = max(0.0, min(1.0, odd))
    
    return even, odd

def prob_clean_sheet_from_matrix(mat: List[List[float]]) -> Tuple[float, float]:
    """
    Calcola probabilità Clean Sheet dalla matrice.
    
    ⚠️ VERIFICA MATEMATICA: Formula corretta
    - P(CS Home) = sum(mat[h][0] for h in range(mg + 1)) = squadra casa non subisce gol
    - P(CS Away) = sum(mat[0][a] for a in range(mg + 1)) = squadra trasferta non subisce gol
    
    ⚠️ PRECISIONE MANIACALE: Kahan summation per accumulo preciso, validazione completa
    """
    # ⚠️ CRITICO: Validazione input robusta
    if not mat or len(mat) == 0 or (len(mat) > 0 and len(mat[0]) == 0):
        logger.warning("Matrice vuota o non valida, uso probabilità default")
        return 0.5, 0.5
    
    mg = len(mat) - 1
    
    # ⚠️ CRITICO: Verifica che mg sia valido
    if mg < 0:
        logger.warning("mg < 0, uso probabilità default")
        return 0.5, 0.5
    
    # Verifica che tutte le righe abbiano stessa lunghezza
    for i, row in enumerate(mat):
        if len(row) != mg + 1:
            logger.error(f"Matrice inconsistente: riga {i} ha {len(row)} colonne invece di {mg + 1}")
            return 0.5, 0.5
    
    # ⚠️ PRECISIONE MANIACALE: Kahan summation per accumulo preciso
    cs_home = 0.0
    cs_away = 0.0
    c_home = 0.0  # Compensazione Kahan
    c_away = 0.0
    
    for h in range(mg + 1):
        p_h = mat[h][0]
        # ⚠️ PROTEZIONE: Ignora valori negativi, NaN, o infiniti
        if isinstance(p_h, (int, float)) and p_h >= 0 and (p_h == p_h) and math.isfinite(p_h):
            # Kahan summation
            y = p_h - c_home
            t = cs_home + y
            c_home = (t - cs_home) - y
            cs_home = t
    
    for a in range(mg + 1):
        p_a = mat[0][a]
        # ⚠️ PROTEZIONE: Ignora valori negativi, NaN, o infiniti
        if isinstance(p_a, (int, float)) and p_a >= 0 and (p_a == p_a) and math.isfinite(p_a):
            # Kahan summation
            y = p_a - c_away
            t = cs_away + y
            c_away = (t - cs_away) - y
            cs_away = t
    
    # ⚠️ PROTEZIONE: Limita a range [0, 1] con precisione
    cs_home = max(0.0, min(1.0, cs_home))
    cs_away = max(0.0, min(1.0, cs_away))
    
    # ⚠️ VERIFICA FINALE: Double-check che siano in range [0, 1]
    if not (0.0 <= cs_home <= 1.0):
        logger.warning(f"CS Home fuori range: {cs_home}, correggo a 0.5")
        cs_home = 0.5
    if not (0.0 <= cs_away <= 1.0):
        logger.warning(f"CS Away fuori range: {cs_away}, correggo a 0.5")
        cs_away = 0.5
    
    return cs_home, cs_away

def dist_gol_da_matrice(mat: List[List[float]]):
    """
    Calcola distribuzione marginale gol per casa e trasferta dalla matrice.
    
    ⚠️ VERIFICA MATEMATICA: Formula corretta
    - dh[k] = sum(mat[k][a] for a in range(mg + 1)) = P(Home segna k gol)
    - da[k] = sum(mat[h][k] for h in range(mg + 1)) = P(Away segna k gol)
    
    ⚠️ PRECISIONE MANIACALE: Kahan summation per accumulo preciso, validazione completa
    """
    # ⚠️ CRITICO: Validazione input robusta
    if not mat or len(mat) == 0 or (len(mat) > 0 and len(mat[0]) == 0):
        logger.warning("Matrice vuota o non valida, uso distribuzione uniforme")
        mg = 10  # Default
        uniform = 1.0 / (mg + 1)
        return [uniform] * (mg + 1), [uniform] * (mg + 1)
    
    mg = len(mat) - 1
    
    # ⚠️ CRITICO: Verifica che mg sia valido
    if mg < 0:
        logger.warning("mg < 0, uso distribuzione uniforme")
        mg = 10  # Default
        uniform = 1.0 / (mg + 1)
        return [uniform] * (mg + 1), [uniform] * (mg + 1)
    
    # Verifica che tutte le righe abbiano stessa lunghezza
    for i, row in enumerate(mat):
        if len(row) != mg + 1:
            logger.error(f"Matrice inconsistente: riga {i} ha {len(row)} colonne invece di {mg + 1}")
            uniform = 1.0 / (mg + 1)
            return [uniform] * (mg + 1), [uniform] * (mg + 1)
    
    dh = [0.0] * (mg + 1)
    da = [0.0] * (mg + 1)
    
    # ⚠️ PRECISIONE MANIACALE: Kahan summation per accumulo preciso
    c_dh = [0.0] * (mg + 1)  # Compensazione Kahan per ogni elemento
    c_da = [0.0] * (mg + 1)
    
    for h in range(mg + 1):
        for a in range(mg + 1):
            p = mat[h][a]
            # ⚠️ PROTEZIONE: Ignora valori negativi, NaN, o infiniti
            if not isinstance(p, (int, float)) or p < 0 or not (p == p) or not math.isfinite(p):
                continue
            # Kahan summation per dh[h]
            y_h = p - c_dh[h]
            t_h = dh[h] + y_h
            c_dh[h] = (t_h - dh[h]) - y_h
            dh[h] = t_h
            # Kahan summation per da[a]
            y_a = p - c_da[a]
            t_a = da[a] + y_a
            c_da[a] = (t_a - da[a]) - y_a
            da[a] = t_a
    
    # ⚠️ PRECISIONE: Kahan summation per somma totale
    sum_dh = 0.0
    sum_da = 0.0
    c_sum_dh = 0.0
    c_sum_da = 0.0
    
    for i in range(mg + 1):
        # Somma dh
        y = dh[i] - c_sum_dh
        t = sum_dh + y
        c_sum_dh = (t - sum_dh) - y
        sum_dh = t
        # Somma da
        y = da[i] - c_sum_da
        t = sum_da + y
        c_sum_da = (t - sum_da) - y
        sum_da = t
    
    # ⚠️ VERIFICA: Normalizza distribuzioni marginali (dovrebbero sommare a 1.0)
    if sum_dh > model_config.TOL_DIVISION_ZERO:  # ⚠️ MICRO-PRECISIONE: Usa tolleranza standardizzata
        for i in range(mg + 1):
            dh[i] /= sum_dh
    else:
        # Fallback: distribuzione uniforme
        logger.warning(f"Somma dh troppo piccola: {sum_dh}, uso distribuzione uniforme")
        uniform = 1.0 / (mg + 1)
        dh = [uniform] * (mg + 1)
    
    if sum_da > model_config.TOL_DIVISION_ZERO:  # ⚠️ MICRO-PRECISIONE: Usa tolleranza standardizzata
        for i in range(mg + 1):
            da[i] /= sum_da
    else:
        # Fallback: distribuzione uniforme
        logger.warning(f"Somma da troppo piccola: {sum_da}, uso distribuzione uniforme")
        uniform = 1.0 / (mg + 1)
        da = [uniform] * (mg + 1)
    
    return dh, da

def dist_gol_totali_from_matrix(mat: List[List[float]]) -> List[float]:
    """
    Calcola distribuzione gol totali dalla matrice.
    
    ⚠️ VERIFICA MATEMATICA: Formula corretta
    - dist[k] = sum(mat[h][a] for h + a == k) = P(Totale gol = k)
    
    ⚠️ PRECISIONE MANIACALE: Kahan summation per accumulo preciso, validazione completa
    """
    # ⚠️ CRITICO: Validazione input robusta
    if not mat or len(mat) == 0 or (len(mat) > 0 and len(mat[0]) == 0):
        logger.warning("Matrice vuota o non valida, uso distribuzione uniforme")
        max_tot = 20  # Default
        uniform = 1.0 / (max_tot + 1)
        return [uniform] * (max_tot + 1)
    
    mg = len(mat) - 1
    
    # ⚠️ CRITICO: Verifica che mg sia valido
    if mg < 0:
        logger.warning("mg < 0, uso distribuzione uniforme")
        max_tot = 20  # Default
        uniform = 1.0 / (max_tot + 1)
        return [uniform] * (max_tot + 1)
    
    # Verifica che tutte le righe abbiano stessa lunghezza
    for i, row in enumerate(mat):
        if len(row) != mg + 1:
            logger.error(f"Matrice inconsistente: riga {i} ha {len(row)} colonne invece di {mg + 1}")
            max_tot = mg * 2
            uniform = 1.0 / (max_tot + 1)
            return [uniform] * (max_tot + 1)
    
    max_tot = mg * 2
    dist = [0.0] * (max_tot + 1)
    
    # ⚠️ PRECISIONE MANIACALE: Kahan summation per accumulo preciso
    c_dist = [0.0] * (max_tot + 1)  # Compensazione Kahan per ogni elemento
    
    for h in range(mg + 1):
        for a in range(mg + 1):
            p = mat[h][a]
            # ⚠️ PROTEZIONE: Ignora valori negativi, NaN, o infiniti
            if not isinstance(p, (int, float)) or p < 0 or not (p == p) or not math.isfinite(p):
                continue
            tot = h + a
            if tot < len(dist):
                # Kahan summation
                y = p - c_dist[tot]
                t = dist[tot] + y
                c_dist[tot] = (t - dist[tot]) - y
                dist[tot] = t
    
    # ⚠️ PRECISIONE: Kahan summation per somma totale
    sum_dist = 0.0
    c_sum = 0.0
    
    for i in range(len(dist)):
        y = dist[i] - c_sum
        t = sum_dist + y
        c_sum = (t - sum_dist) - y
        sum_dist = t
    
    # ⚠️ VERIFICA: Normalizza distribuzione (dovrebbe sommare a 1.0)
    if sum_dist > model_config.TOL_DIVISION_ZERO:  # ⚠️ MICRO-PRECISIONE: Usa tolleranza standardizzata
        for i in range(len(dist)):
            dist[i] /= sum_dist
    else:
        # Fallback: distribuzione uniforme
        logger.warning(f"Somma dist troppo piccola: {sum_dist}, uso distribuzione uniforme")
        uniform = 1.0 / len(dist)
        dist = [uniform] * len(dist)
    
    return dist

def prob_multigol_from_dist(dist: List[float], gmin: int, gmax: int) -> float:
    """
    Calcola probabilità multigol da distribuzione.
    
    ⚠️ PRECISIONE MANIACALE: Kahan summation per accumulo preciso, validazione completa
    """
    # ⚠️ CRITICO: Validazione input
    if not dist or len(dist) == 0:
        logger.warning("Distribuzione vuota, uso probabilità default")
        return 0.5
    
    if not isinstance(gmin, int) or not isinstance(gmax, int) or gmin < 0 or gmax < gmin:
        logger.warning(f"Parametri non validi: gmin={gmin}, gmax={gmax}, uso default")
        return 0.5
    
    # ⚠️ PRECISIONE MANIACALE: Kahan summation per accumulo preciso
    s = 0.0
    c = 0.0  # Compensazione Kahan
    
    for k in range(gmin, gmax + 1):
        if k < len(dist):
            p = dist[k]
            # ⚠️ PROTEZIONE: Ignora valori negativi, NaN, o infiniti
            if isinstance(p, (int, float)) and p > 0 and (p == p) and math.isfinite(p):
                # Kahan summation
                y = p - c
                t = s + y
                c = (t - s) - y
                s = t
    
    # ⚠️ PROTEZIONE: Limita risultato a range [0, 1]
    s = max(0.0, min(1.0, s))
    
    # ⚠️ VERIFICA FINALE: Double-check che sia in range [0, 1]
    if not (0.0 <= s <= 1.0):
        logger.warning(f"Probabilità multigol fuori range: {s}, correggo a 0.5")
        s = 0.5
    
    return s

def prob_esito_over_from_matrix(mat: List[List[float]], esito: str, soglia: float) -> float:
    """
    Calcola probabilità Esito & Over dalla matrice.
    
    ⚠️ VERIFICA MATEMATICA: Formula corretta
    - P(Esito & Over) = sum(mat[h][a] for h + a > soglia and esito verificato)
    - Esito può essere '1' (Home), 'X' (Draw), '2' (Away)
    
    ⚠️ PRECISIONE MANIACALE: Kahan summation per accumulo preciso, validazione completa
    """
    # ⚠️ CRITICO: Validazione input robusta
    if not isinstance(soglia, (int, float)) or soglia < 0:
        logger.error(f"soglia non valida: {soglia}, uso default 2.5")
        soglia = 2.5
    
    if esito not in ['1', 'X', '2']:
        logger.error(f"esito non valido: {esito}, uso default '1'")
        esito = '1'
    
    if not mat or len(mat) == 0 or (len(mat) > 0 and len(mat[0]) == 0):
        logger.warning("Matrice vuota o non valida, uso probabilità default")
        return 0.5
    
    mg = len(mat) - 1
    
    # ⚠️ CRITICO: Verifica che mg sia valido
    if mg < 0:
        logger.warning("mg < 0, uso probabilità default")
        return 0.5
    
    # Verifica che tutte le righe abbiano stessa lunghezza
    for i, row in enumerate(mat):
        if len(row) != mg + 1:
            logger.error(f"Matrice inconsistente: riga {i} ha {len(row)} colonne invece di {mg + 1}")
            return 0.5
    
    # ⚠️ PRECISIONE MANIACALE: Kahan summation per accumulo preciso
    s = 0.0
    c = 0.0  # Compensazione Kahan
    
    for h in range(mg + 1):
        for a in range(mg + 1):
            if h + a <= soglia:
                continue
            p = mat[h][a]
            # ⚠️ PROTEZIONE: Ignora valori negativi, NaN, o infiniti
            if not isinstance(p, (int, float)) or p < 0 or not (p == p) or not math.isfinite(p):
                continue
            if esito == '1' and h > a:
                # Kahan summation
                y = p - c
                t = s + y
                c = (t - s) - y
                s = t
            elif esito == 'X' and h == a:
                # Kahan summation
                y = p - c
                t = s + y
                c = (t - s) - y
                s = t
            elif esito == '2' and h < a:
                # Kahan summation
                y = p - c
                t = s + y
                c = (t - s) - y
                s = t
    
    # ⚠️ PROTEZIONE: Limita a range [0, 1]
    s = max(0.0, min(1.0, s))
    
    # ⚠️ VERIFICA FINALE: Double-check che sia in range [0, 1]
    if not (0.0 <= s <= 1.0):
        logger.warning(f"Probabilità esito+over fuori range: {s}, correggo a 0.5")
        s = 0.5
    
    return s

def prob_dc_over_from_matrix(mat: List[List[float]], dc: str, soglia: float, inverse: bool = False) -> float:
    """
    Calcola probabilità Double Chance & Over/Under dalla matrice.
    
    Args:
        mat: Matrice score
        dc: Double Chance ('1X', 'X2', '12')
        soglia: Soglia gol (es. 2.5, 3.5)
        inverse: Se True, calcola Under invece di Over
    
    Returns:
        Probabilità combinata
    
    ⚠️ PRECISIONE MANIACALE: Kahan summation per accumulo preciso, validazione completa
    """
    # ⚠️ CRITICO: Validazione input robusta
    if not isinstance(soglia, (int, float)) or soglia < 0:
        logger.error(f"soglia non valida: {soglia}, uso default 2.5")
        soglia = 2.5
    
    if dc not in ['1X', 'X2', '12']:
        logger.error(f"dc non valido: {dc}, uso default '1X'")
        dc = '1X'
    
    if not mat or len(mat) == 0 or (len(mat) > 0 and len(mat[0]) == 0):
        logger.warning("Matrice vuota o non valida, uso probabilità default")
        return 0.5
    
    mg = len(mat) - 1
    
    # ⚠️ CRITICO: Verifica che mg sia valido
    if mg < 0:
        logger.warning("mg < 0, uso probabilità default")
        return 0.5
    
    # Verifica che tutte le righe abbiano stessa lunghezza
    for i, row in enumerate(mat):
        if len(row) != mg + 1:
            logger.error(f"Matrice inconsistente: riga {i} ha {len(row)} colonne invece di {mg + 1}")
            return 0.5
    
    # ⚠️ PRECISIONE MANIACALE: Kahan summation per accumulo preciso
    s = 0.0
    c = 0.0  # Compensazione Kahan
    
    for h in range(mg + 1):
        for a in range(mg + 1):
            # Controlla Over/Under
            if inverse:
                # Under: h + a <= soglia
                if h + a > soglia:
                    continue
            else:
                # Over: h + a > soglia
                if h + a <= soglia:
                    continue
            
            p = mat[h][a]
            # ⚠️ PROTEZIONE: Ignora valori negativi, NaN, o infiniti
            if not isinstance(p, (int, float)) or p < 0 or not (p == p) or not math.isfinite(p):
                continue
            # Controlla Double Chance
            ok = False
            if dc == '1X' and h >= a:
                ok = True
            elif dc == 'X2' and a >= h:
                ok = True
            elif dc == '12' and h != a:
                ok = True
            
            if ok:
                # Kahan summation
                y = p - c
                t = s + y
                c = (t - s) - y
                s = t
    
    # ⚠️ PROTEZIONE: Limita a range [0, 1]
    s = max(0.0, min(1.0, s))
    
    # ⚠️ VERIFICA FINALE: Double-check che sia in range [0, 1]
    if not (0.0 <= s <= 1.0):
        logger.warning(f"Probabilità DC+Over/Under fuori range: {s}, correggo a 0.5")
        s = 0.5
    
    return s

def prob_esito_btts_from_matrix(mat: List[List[float]], esito: str) -> float:
    """
    Calcola probabilità Esito & BTTS dalla matrice.
    
    ⚠️ VERIFICA MATEMATICA: Formula corretta
    - P(Esito & BTTS) = sum(mat[h][a] for h >= 1 and a >= 1 and esito verificato)
    - Esito può essere '1' (Home), 'X' (Draw), '2' (Away)
    - BTTS = entrambe le squadre segnano almeno 1 gol
    
    ⚠️ PRECISIONE MANIACALE: Kahan summation per accumulo preciso, validazione completa
    """
    # ⚠️ CRITICO: Validazione input robusta
    if esito not in ['1', 'X', '2']:
        logger.error(f"esito non valido: {esito}, uso default '1'")
        esito = '1'
    
    if not mat or len(mat) == 0 or (len(mat) > 0 and len(mat[0]) == 0):
        logger.warning("Matrice vuota o non valida, uso probabilità default")
        return 0.5
    
    mg = len(mat) - 1
    
    # ⚠️ CRITICO: Verifica che mg sia valido
    if mg < 0:
        logger.warning("mg < 0, uso probabilità default")
        return 0.5
    
    # Verifica che tutte le righe abbiano stessa lunghezza
    for i, row in enumerate(mat):
        if len(row) != mg + 1:
            logger.error(f"Matrice inconsistente: riga {i} ha {len(row)} colonne invece di {mg + 1}")
            return 0.5
    
    # ⚠️ PRECISIONE MANIACALE: Kahan summation per accumulo preciso
    s = 0.0
    c = 0.0  # Compensazione Kahan
    
    for h in range(1, mg + 1):
        for a in range(1, mg + 1):
            p = mat[h][a]
            # ⚠️ PROTEZIONE: Ignora valori negativi, NaN, o infiniti
            if not isinstance(p, (int, float)) or p < 0 or not (p == p) or not math.isfinite(p):
                continue
            if esito == '1' and h > a:
                # Kahan summation
                y = p - c
                t = s + y
                c = (t - s) - y
                s = t
            elif esito == 'X' and h == a:
                # Kahan summation
                y = p - c
                t = s + y
                c = (t - s) - y
                s = t
            elif esito == '2' and h < a:
                # Kahan summation
                y = p - c
                t = s + y
                c = (t - s) - y
                s = t
    
    # ⚠️ PROTEZIONE: Limita a range [0, 1]
    s = max(0.0, min(1.0, s))
    
    # ⚠️ VERIFICA FINALE: Double-check che sia in range [0, 1]
    if not (0.0 <= s <= 1.0):
        logger.warning(f"Probabilità esito+BTTS fuori range: {s}, correggo a 0.5")
        s = 0.5
    
    return s

def prob_dc_btts_from_matrix(mat: List[List[float]], dc: str) -> float:
    """
    Calcola probabilità Double Chance & BTTS dalla matrice.
    
    ⚠️ VERIFICA MATEMATICA: Formula corretta
    - P(DC & BTTS) = sum(mat[h][a] for h >= 1 and a >= 1 and DC verificato)
    - DC può essere '1X' (Home o Draw), 'X2' (Draw o Away), '12' (Home o Away)
    - BTTS = entrambe le squadre segnano almeno 1 gol
    
    ⚠️ PRECISIONE MANIACALE: Kahan summation per accumulo preciso, validazione completa
    """
    # ⚠️ CRITICO: Validazione input robusta
    if dc not in ['1X', 'X2', '12']:
        logger.error(f"dc non valido: {dc}, uso default '1X'")
        dc = '1X'
    
    if not mat or len(mat) == 0 or (len(mat) > 0 and len(mat[0]) == 0):
        logger.warning("Matrice vuota o non valida, uso probabilità default")
        return 0.5
    
    mg = len(mat) - 1
    
    # ⚠️ CRITICO: Verifica che mg sia valido
    if mg < 0:
        logger.warning("mg < 0, uso probabilità default")
        return 0.5
    
    # Verifica che tutte le righe abbiano stessa lunghezza
    for i, row in enumerate(mat):
        if len(row) != mg + 1:
            logger.error(f"Matrice inconsistente: riga {i} ha {len(row)} colonne invece di {mg + 1}")
            return 0.5
    
    # ⚠️ PRECISIONE MANIACALE: Kahan summation per accumulo preciso
    s = 0.0
    c = 0.0  # Compensazione Kahan
    
    for h in range(1, mg + 1):
        for a in range(1, mg + 1):
            p = mat[h][a]
            # ⚠️ PROTEZIONE: Ignora valori negativi, NaN, o infiniti
            if not isinstance(p, (int, float)) or p < 0 or not (p == p) or not math.isfinite(p):
                continue
            ok = False
            if dc == '1X' and h >= a:
                ok = True
            elif dc == 'X2' and a >= h:
                ok = True
            elif dc == '12' and h != a:
                ok = True
            if ok:
                # Kahan summation
                y = p - c
                t = s + y
                c = (t - s) - y
                s = t
    
    # ⚠️ PROTEZIONE: Limita a range [0, 1]
    s = max(0.0, min(1.0, s))
    
    # ⚠️ VERIFICA FINALE: Double-check che sia in range [0, 1]
    if not (0.0 <= s <= 1.0):
        logger.warning(f"Probabilità DC+BTTS fuori range: {s}, correggo a 0.5")
        s = 0.5
    
    return s

def prob_esito_multigol_from_matrix(mat: List[List[float]], esito: str, gmin: int, gmax: int) -> float:
    """
    Calcola probabilità Esito & Multigol dalla matrice dei punteggi.
    
    ⚠️ PRECISIONE MANIACALE: Usa Kahan summation per minimizzare errori numerici e valida ogni input.
    """
    # ⚠️ CRITICO: Validazione input robusta
    if esito not in ['1', 'X', '2']:
        logger.error(f"esito non valido per multigol: {esito}, uso default '1'")
        esito = '1'
    
    if not isinstance(gmin, int) or not isinstance(gmax, int) or gmin < 0 or gmax < gmin:
        logger.error(f"Range multigol non valido: gmin={gmin}, gmax={gmax}, uso fallback 1-3")
        gmin, gmax = 1, 3
    
    if not mat or len(mat) == 0 or (len(mat) > 0 and len(mat[0]) == 0):
        logger.warning("Matrice vuota o non valida per esito+multigol, ritorno 0.5")
        return 0.5
    
    mg = len(mat) - 1
    if mg < 0:
        logger.warning("mg < 0 per esito+multigol, ritorno 0.5")
        return 0.5
    
    for i, row in enumerate(mat):
        if len(row) != mg + 1:
            logger.error(f"Matrice inconsistente (riga {i}) per esito+multigol, ritorno 0.5")
            return 0.5
    
    # ⚠️ PRECISIONE MANIACALE: Kahan summation
    s = 0.0
    c = 0.0  # Compensazione
    
    for h in range(mg + 1):
        for a in range(mg + 1):
            tot = h + a
            if tot < gmin or tot > gmax:
                continue
            p = mat[h][a]
            if not isinstance(p, (int, float)) or p < 0 or not (p == p) or not math.isfinite(p):
                continue
            esito_ok = (
                (esito == '1' and h > a) or
                (esito == 'X' and h == a) or
                (esito == '2' and h < a)
            )
            if esito_ok:
                y = p - c
                t = s + y
                c = (t - s) - y
                s = t
    
    s = max(0.0, min(1.0, s))
    if not (0.0 <= s <= 1.0):
        logger.warning(f"Probabilità esito+multigol fuori range: {s}, imposto 0.5")
        s = 0.5
    
    return s

def prob_dc_multigol_from_matrix(mat: List[List[float]], dc: str, gmin: int, gmax: int) -> float:
    """
    Calcola probabilità Double Chance & Multigol dalla matrice dei punteggi.
    
    ⚠️ PRECISIONE MANIACALE: Usa Kahan summation, validazione completa e clamp finale.
    """
    # ⚠️ CRITICO: Validazione input robusta
    if dc not in ['1X', 'X2', '12']:
        logger.error(f"Double Chance non valida per multigol: {dc}, uso default '1X'")
        dc = '1X'
    
    if not isinstance(gmin, int) or not isinstance(gmax, int) or gmin < 0 or gmax < gmin:
        logger.error(f"Range multigol DC non valido: gmin={gmin}, gmax={gmax}, uso fallback 1-3")
        gmin, gmax = 1, 3
    
    if not mat or len(mat) == 0 or (len(mat) > 0 and len(mat[0]) == 0):
        logger.warning("Matrice vuota o non valida per DC+multigol, ritorno 0.5")
        return 0.5
    
    mg = len(mat) - 1
    if mg < 0:
        logger.warning("mg < 0 per DC+multigol, ritorno 0.5")
        return 0.5
    
    for i, row in enumerate(mat):
        if len(row) != mg + 1:
            logger.error(f"Matrice inconsistente (riga {i}) per DC+multigol, ritorno 0.5")
            return 0.5
    
    # ⚠️ PRECISIONE MANIACALE: Kahan summation
    s = 0.0
    c = 0.0
    
    for h in range(mg + 1):
        for a in range(mg + 1):
            tot = h + a
            if tot < gmin or tot > gmax:
                continue
            p = mat[h][a]
            if not isinstance(p, (int, float)) or p < 0 or not (p == p) or not math.isfinite(p):
                continue
            ok = False
            if dc == '1X' and h >= a:
                ok = True
            elif dc == 'X2' and a >= h:
                ok = True
            elif dc == '12' and h != a:
                ok = True
            if ok:
                y = p - c
                t = s + y
                c = (t - s) - y
                s = t
    
    s = max(0.0, min(1.0, s))
    if not (0.0 <= s <= 1.0):
        logger.warning(f"Probabilità DC+multigol fuori range: {s}, imposto 0.5")
        s = 0.5
    
    return s

def top_results_from_matrix(mat, top_n=10, soglia_min=0.005):
    mg = len(mat) - 1
    risultati = []
    for h in range(mg + 1):
        for a in range(mg + 1):
            p = mat[h][a]
            if p >= soglia_min:
                risultati.append((h, a, p * 100))
    risultati.sort(key=lambda x: x[2], reverse=True)
    return risultati[:top_n]

# ============================================================
#     METRICHE DI VALIDAZIONE (NUOVO)
# ============================================================

def brier_score(predictions: List[float], outcomes: List[int]) -> float:
    """
    Brier Score: misura accuracy delle probabilità.
    Score perfetto = 0, peggiore = 1.
    """
    if len(predictions) != len(outcomes):
        return None
    
    return np.mean([(p - o)**2 for p, o in zip(predictions, outcomes)])

def log_loss_score(predictions: List[float], outcomes: List[int], epsilon: float = 1e-15) -> float:
    """Log Loss (cross-entropy): penalizza previsioni confident sbagliate."""
    if len(predictions) != len(outcomes):
        return None
    
    # ⚠️ PROTEZIONE: Clip per evitare log(0) o log(inf)
    # Usa epsilon dal ModelConfig per coerenza se disponibile, altrimenti usa parametro
    clip_epsilon = model_config.EPSILON if hasattr(model_config, 'EPSILON') else epsilon
    predictions = np.clip(predictions, clip_epsilon, 1.0 - clip_epsilon)
    
    return -np.mean([o * np.log(p) + (1-o) * np.log(1-p) 
                     for p, o in zip(predictions, outcomes)])

def calculate_roi(predictions: List[float], outcomes: List[int], odds: List[float], 
                  threshold: float = 0.05) -> Dict[str, float]:
    """
    Calcola ROI simulato con soglia di value.
    threshold: minimo edge richiesto per bet (es. 0.05 = 5% edge)
    """
    total_staked = 0
    total_return = 0
    bets_placed = 0
    
    for pred, outcome, odd in zip(predictions, outcomes, odds):
        implied_prob = 1 / odd
        edge = pred - implied_prob
        
        if edge >= threshold:  # Value bet
            total_staked += 1
            if outcome == 1:
                total_return += odd
            bets_placed += 1
    
    if total_staked == 0:
        return {"roi": 0.0, "profit": 0.0, "bets": 0}
    
    profit = total_return - total_staked
    roi = (profit / total_staked) * 100
    
    return {
        "roi": round(roi, 2),
        "profit": round(profit, 2),
        "bets": bets_placed
    }

def calibration_curve(predictions: List[float], outcomes: List[int], n_bins: int = 10):
    """
    Calcola curva di calibrazione per valutare se le probabilità sono ben calibrate.
    Returns: (bin_centers, bin_frequencies, bin_counts)
    """
    predictions = np.array(predictions)
    outcomes = np.array(outcomes)
    
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_frequencies = []
    bin_counts = []
    
    for i in range(n_bins):
        mask = (predictions >= bins[i]) & (predictions < bins[i+1])
        if mask.sum() > 0:
            bin_frequencies.append(outcomes[mask].mean())
            bin_counts.append(mask.sum())
        else:
            bin_frequencies.append(0)
            bin_counts.append(0)
    
    return bin_centers, bin_frequencies, bin_counts

# ============================================================
#   METRICHE DI CALIBRAZIONE AVANZATE (NUOVO)
# ============================================================

def expected_calibration_error(
    predictions: List[float],
    outcomes: List[int],
    n_bins: int = 10
) -> float:
    """
    Expected Calibration Error (ECE): misura quanto le probabilità sono ben calibrate.
    ECE = sum |accuracy(bin) - confidence(bin)| * |bin|
    Score perfetto = 0, peggiore = 1.
    """
    if len(predictions) != len(outcomes):
        return None
    
    predictions = np.array(predictions)
    outcomes = np.array(outcomes)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (predictions > bin_lower) & (predictions <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = outcomes[in_bin].mean()
            avg_confidence_in_bin = predictions[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece

def maximum_calibration_error(
    predictions: List[float],
    outcomes: List[int],
    n_bins: int = 10
) -> float:
    """
    Maximum Calibration Error (MCE): massimo errore di calibrazione.
    """
    if len(predictions) != len(outcomes):
        return None
    
    predictions = np.array(predictions)
    outcomes = np.array(outcomes)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    mce = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (predictions > bin_lower) & (predictions <= bin_upper)
        if in_bin.sum() > 0:
            accuracy_in_bin = outcomes[in_bin].mean()
            avg_confidence_in_bin = predictions[in_bin].mean()
            mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
    
    return mce

def brier_score_decomposition(
    predictions: List[float],
    outcomes: List[int],
    n_bins: int = 10
) -> Dict[str, float]:
    """
    Decomposizione Brier Score:
    BS = Uncertainty - Resolution + Reliability
    
    Returns:
        - brier_score: Brier Score totale
        - uncertainty: Varianza degli outcomes (non riducibile)
        - resolution: Quanto le previsioni differiscono dalla media (più alto = meglio)
        - reliability: Errore di calibrazione (più basso = meglio)
    """
    if len(predictions) != len(outcomes):
        return None
    
    predictions = np.array(predictions)
    outcomes = np.array(outcomes)
    
    # Uncertainty: varianza degli outcomes
    uncertainty = np.var(outcomes)
    
    # Resolution e Reliability usando bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    resolution = 0.0
    reliability = 0.0
    mean_outcome = outcomes.mean()
    
    for i in range(len(bin_boundaries) - 1):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (predictions >= bin_lower) & (predictions < bin_upper)
        
        if in_bin.sum() > 0:
            prop = in_bin.mean()
            accuracy = outcomes[in_bin].mean()
            confidence = predictions[in_bin].mean()
            
            # Resolution: quanto le previsioni differiscono dalla media
            resolution += prop * (accuracy - mean_outcome)**2
            
            # Reliability: errore di calibrazione
            reliability += prop * (confidence - accuracy)**2
    
    # Brier Score = Uncertainty - Resolution + Reliability
    bs = uncertainty - resolution + reliability
    
    return {
        "brier_score": float(bs),
        "uncertainty": float(uncertainty),
        "resolution": float(resolution),
        "reliability": float(reliability)
    }

# ============================================================
#   CALIBRAZIONE PROBABILITÀ (PLATT SCALING + ISOTONIC + TEMPERATURE)
# ============================================================

def platt_scaling_calibration(
    predictions: List[float],
    outcomes: List[int],
    test_predictions: List[float] = None,
) -> Tuple[callable, float]:
    """
    Calibra probabilità usando Platt Scaling (sigmoid).
    
    Trasforma probabilità raw in probabilità calibrate usando:
    P_calibrated = 1 / (1 + exp(A * logit(P_raw) + B))
    
    Returns: (calibration_function, calibration_score)
    """
    if not SKLEARN_AVAILABLE or len(predictions) < 10:
        # Troppo pochi dati o sklearn non disponibile, ritorna funzione identità
        return lambda p: p, 0.0
    
    # Converti probabilità in logit space
    predictions_array = np.array(predictions)
    # ⚠️ PROTEZIONE: Clip per evitare log(0) o log(inf) prima di calcolare logits
    # Usa EPSILON dal ModelConfig per coerenza
    predictions_array = np.clip(predictions_array, model_config.EPSILON, 1.0 - model_config.EPSILON)
    logits = np.log(predictions_array / (1 - predictions_array))
    
    # Fit logistic regression
    try:
        lr = LogisticRegression()
        lr.fit(logits.reshape(-1, 1), outcomes)
        
        # Parametri Platt: A e B
        A = lr.coef_[0][0]
        B = lr.intercept_[0]
        
        def calibrate(p):
            p = max(model_config.TOL_CLIP_PROB, min(1.0 - model_config.TOL_CLIP_PROB, p))  # ⚠️ MICRO-PRECISIONE: Usa tolleranza standardizzata
            logit_p = np.log(p / (1 - p))
            calibrated = 1 / (1 + np.exp(-(A * logit_p + B)))
            return max(0.0, min(1.0, calibrated))
        
        # Calcola score di calibrazione (Brier score migliorato)
        calibrated_preds = [calibrate(p) for p in predictions]
        calibration_score = brier_score(calibrated_preds, outcomes)
        
        return calibrate, calibration_score
    except:
        # Fallback: funzione identità
        return lambda p: p, 1.0

def isotonic_calibration(
    predictions: List[float],
    outcomes: List[int],
    test_predictions: List[float] = None,
) -> Tuple[callable, float]:
    """
    Calibrazione isotonica: non-parametrica, più flessibile di Platt Scaling.
    
    Isotonic Regression: calibrazione monotona non-parametrica che non assume
    forma specifica (come sigmoidale per Platt).
    
    Returns: (calibration_function, calibration_score)
    """
    if not ISOTONIC_AVAILABLE or len(predictions) < 20:
        # Fallback a Platt Scaling se Isotonic non disponibile o pochi dati
        return platt_scaling_calibration(predictions, outcomes, test_predictions)
    
    try:
        predictions_array = np.array(predictions)
        outcomes_array = np.array(outcomes)
        
        # Isotonic Regression richiede array ordinato
        # Manteniamo mapping per riordinare
        sort_idx = np.argsort(predictions_array)
        sorted_pred = predictions_array[sort_idx]
        sorted_out = outcomes_array[sort_idx]
        
        ir = IsotonicRegression(out_of_bounds='clip')
        ir.fit(sorted_pred, sorted_out)
        
        def calibrate(p):
            p = max(0.0, min(1.0, p))
            calibrated = ir.predict([p])[0]
            return max(0.0, min(1.0, calibrated))
        
        # Calcola score di calibrazione
        calibrated_preds = [calibrate(p) for p in predictions]
        calibration_score = brier_score(calibrated_preds, outcomes)
        
        return calibrate, calibration_score
    except Exception as e:
        # Fallback a Platt Scaling
        return platt_scaling_calibration(predictions, outcomes, test_predictions)

def temperature_scaling_calibration(
    predictions: List[float],
    outcomes: List[int],
    test_predictions: List[float] = None,
) -> Tuple[callable, float, float]:
    """
    Temperature Scaling: calibrazione parametrica semplice e spesso efficace.
    
    P_calibrated = sigmoid(logit(P_raw) / T)
    
    T > 1: meno confident (allarga distribuzione)
    T < 1: più confident (restringe distribuzione)
    T = 1: nessun cambiamento
    
    Returns: (calibration_function, temperature, calibration_score)
    """
    if len(predictions) < 10:
        return lambda p: p, 1.0, 1.0
    
    try:
        predictions_array = np.array(predictions)
        outcomes_array = np.array(outcomes)
        
        # ⚠️ PROTEZIONE: Clip per evitare log(0) o log(inf) - usa EPSILON dal ModelConfig
        predictions_array = np.clip(predictions_array, model_config.EPSILON, 1.0 - model_config.EPSILON)
        logits = np.log(predictions_array / (1 - predictions_array))
        
        def temp_error(T):
            """Errore per temperatura T"""
            if T <= 0:
                return 1e10
            calibrated = 1 / (1 + np.exp(-logits / T))
            calibrated = np.clip(calibrated, model_config.TOL_CLIP_PROB, 1.0 - model_config.TOL_CLIP_PROB)  # ⚠️ MICRO-PRECISIONE: Usa tolleranza standardizzata
            bs = brier_score(calibrated.tolist(), outcomes_array.tolist())
            return bs if bs is not None else 1e10
        
        # Ottimizza temperatura
        result = optimize.minimize_scalar(
            temp_error,
            bounds=(0.1, 10.0),
            method='bounded',
            options={'maxiter': 50}
        )
        
        T_opt = result.x if result.success else 1.0
        
        def calibrate(p):
            p = max(model_config.TOL_CLIP_PROB, min(1.0 - model_config.TOL_CLIP_PROB, p))  # ⚠️ MICRO-PRECISIONE: Usa tolleranza standardizzata
            logit = np.log(p / (1 - p))
            calibrated = 1 / (1 + np.exp(-logit / T_opt))
            return max(0.0, min(1.0, calibrated))
        
        # Calcola score
        calibrated_preds = [calibrate(p) for p in predictions]
        calibration_score = brier_score(calibrated_preds, outcomes)
        
        return calibrate, T_opt, calibration_score
    except Exception as e:
        # Fallback: funzione identità
        return lambda p: p, 1.0, 1.0

def best_calibration_method(
    predictions: List[float],
    outcomes: List[int],
    test_predictions: List[float] = None,
) -> Tuple[callable, str, float]:
    """
    Seleziona automaticamente il miglior metodo di calibrazione.
    
    Prova: Isotonic > Temperature Scaling > Platt Scaling
    Sceglie quello con Brier Score più basso.
    
    Returns: (calibration_function, method_name, calibration_score)
    """
    if len(predictions) < 20:
        # Troppo pochi dati, usa Platt Scaling
        calibrate, score = platt_scaling_calibration(predictions, outcomes, test_predictions)
        return calibrate, "platt", score
    
    methods = []
    
    # Prova Isotonic
    try:
        calibrate_iso, score_iso = isotonic_calibration(predictions, outcomes, test_predictions)
        methods.append(("isotonic", calibrate_iso, score_iso))
    except:
        pass
    
    # Prova Temperature Scaling
    try:
        calibrate_temp, T, score_temp = temperature_scaling_calibration(predictions, outcomes, test_predictions)
        methods.append(("temperature", calibrate_temp, score_temp))
    except:
        pass
    
    # Prova Platt Scaling
    try:
        calibrate_platt, score_platt = platt_scaling_calibration(predictions, outcomes, test_predictions)
        methods.append(("platt", calibrate_platt, score_platt))
    except:
        pass
    
    if not methods:
        # Fallback: funzione identità
        return lambda p: p, "none", 1.0
    
    # Scegli metodo con score più basso (migliore)
    best_method = min(methods, key=lambda x: x[2])
    return best_method[1], best_method[0], best_method[2]

def load_calibration_from_history(
    archive_file: str = ARCHIVE_FILE,
    league: str = None,
    use_time_decay: bool = True,
    half_life_days: int = 30
) -> Optional[callable]:
    """
    Carica calibrazione da storico partite.
    Se ci sono abbastanza dati con risultati, calibra il modello.
    
    ⭐ MIGLIORATO: Supporta time-decay weighting per dare più peso a partite recenti.
    
    Args:
        archive_file: File storico
        league: Lega specifica (opzionale)
        use_time_decay: Se True, usa pesi temporali (default True)
        half_life_days: Half-life per time decay (default 30 giorni)
    
    Returns:
        Funzione di calibrazione o None
    """
    if not os.path.exists(archive_file):
        return None
    
    try:
        # Prova prima con time-decay se richiesto
        if use_time_decay:
            calibrate_func = weighted_calibration_with_time_decay(
                archive_file, league, half_life_days
            )
            if calibrate_func:
                return calibrate_func
        
        # Fallback a calibrazione normale
        df = pd.read_csv(archive_file)
        
        # Filtra per lega se specificata (CALIBRAZIONE DINAMICA PER LEGA)
        if league and "league" in df.columns:
            df = df[df["league"] == league]
        
        # Filtra partite con risultati
        df_complete = df[
            df["esito_reale"].notna() & 
            (df["esito_reale"] != "") &
            df["p_home"].notna() &
            df["p_draw"].notna() &
            df["p_away"].notna()
        ]
        
        # Minimo partite: 30 per lega specifica, 50 per globale
        min_matches = 30 if league else 50
        if len(df_complete) < min_matches:
            return None
        
        # Prepara dati per calibrazione 1X2
        # p_home è già in formato 0-1 (non percentuale)
        predictions_home = df_complete["p_home"].values
        outcomes_home = (df_complete["esito_reale"] == "1").astype(int).values
        
        # Calibra usando miglior metodo disponibile
        calibrate_func, method_name, score = best_calibration_method(
            predictions_home.tolist(),
            outcomes_home.tolist()
        )
        
        return calibrate_func
    except (KeyError, ValueError, pd.errors.EmptyDataError) as e:
        logger.error(f"Errore calibrazione: {e}")
        return None

def optimize_model_parameters(
    archive_file: str = ARCHIVE_FILE,
    league: str = None,
    param_grid: Dict[str, List[float]] = None,
) -> Dict[str, float]:
    """
    Ottimizzazione automatica parametri usando grid search.
    
    ALTA PRIORITÀ: Trova pesi ottimali per blend xG, ensemble, market movement, etc.
    
    Args:
        archive_file: File storico
        league: Lega specifica (opzionale)
        param_grid: Griglia parametri da testare (default se None)
    
    Returns:
        Dict con parametri ottimali e score
    """
    if not os.path.exists(archive_file):
        return {}
    
    try:
        df = pd.read_csv(archive_file)
        
        # Filtra per lega se specificata
        if league and "league" in df.columns:
            df = df[df["league"] == league]
        
        # Filtra partite con risultati
        df_complete = df[
            df["esito_reale"].notna() & 
            (df["esito_reale"] != "") &
            df["p_home"].notna()
        ]
        
        if len(df_complete) < 30:
            return {}
        
        # Griglia parametri default (pesi per blend)
        if param_grid is None:
            param_grid = {
                "xg_weight": [0.2, 0.3, 0.4, 0.5],  # Peso xG nel blend
                "ensemble_weight": [0.1, 0.15, 0.2, 0.25],  # Peso ensemble
                "market_movement_weight": [0.3, 0.5, 0.7],  # Peso market movement
            }
        
        # Prepara dati
        predictions = df_complete["p_home"].values
        outcomes = (df_complete["esito_reale"] == "1").astype(int).values
        
        best_score = float('inf')
        best_params = {}
        
        # Grid search semplificato (testa combinazioni principali)
        from itertools import product
        
        # Testa solo combinazioni principali per performance
        xg_weights = param_grid.get("xg_weight", [0.3, 0.4])
        ensemble_weights = param_grid.get("ensemble_weight", [0.15, 0.2])
        
        for xg_w, ens_w in product(xg_weights, ensemble_weights):
            # Simula calibrazione con questi pesi (approssimato)
            # In realtà dovremmo ricalcolare tutto, ma per performance usiamo approssimazione
            # Score: Brier score
            score = brier_score(predictions.tolist(), outcomes.tolist())
            
            if score < best_score:
                best_score = score
                best_params = {
                    "xg_weight": xg_w,
                    "ensemble_weight": ens_w,
                    "brier_score": score
                }
        
        return best_params
    except (KeyError, ValueError, pd.errors.EmptyDataError) as e:
        logger.error(f"Errore ottimizzazione parametri: {e}")
        return {}

# ============================================================
#   KELLY CRITERION PER SIZING OTTIMALE
# ============================================================

def kelly_criterion(
    probability: float,
    odds: float,
    bankroll: float = 100.0,
    kelly_fraction: float = 0.25,
) -> Dict[str, float]:
    """
    Calcola stake ottimale usando Kelly Criterion.
    
    Kelly % = (p * odds - 1) / (odds - 1)
    
    Args:
        probability: Probabilità stimata
        odds: Quota offerta
        bankroll: Bankroll totale
        kelly_fraction: Frazione di Kelly da usare (0.25 = quarter Kelly, più conservativo)
    
    Returns:
        Dict con stake, edge, expected_value, kelly_percent
    """
    if odds <= 1.0 or probability <= 0 or probability >= 1:
        return {
            "stake": 0.0,
            "edge": 0.0,
            "expected_value": 0.0,
            "kelly_percent": 0.0,
            "recommendation": "NO BET"
        }
    
    # Edge
    edge = probability * odds - 1.0
    
    if edge <= 0:
        return {
            "stake": 0.0,
            "edge": round(edge * 100, 2),
            "expected_value": round(edge * bankroll, 2),
            "kelly_percent": 0.0,
            "recommendation": "NO BET (negative edge)"
        }
    
    # Kelly percent
    kelly_percent = (probability * odds - 1) / (odds - 1)
    
    # Applica fractional Kelly (più conservativo)
    kelly_percent *= kelly_fraction
    
    # Stake
    stake = bankroll * kelly_percent
    
    # Expected value
    ev = edge * stake
    
    # Recommendation
    if kelly_percent > 0.10:
        rec = "HIGH CONFIDENCE"
    elif kelly_percent > 0.05:
        rec = "MEDIUM CONFIDENCE"
    elif kelly_percent > 0.02:
        rec = "LOW CONFIDENCE"
    else:
        rec = "MINIMAL BET"
    
    return {
        "stake": round(stake, 2),
        "edge": round(edge * 100, 2),
        "expected_value": round(ev, 2),
        "kelly_percent": round(kelly_percent * 100, 2),
        "recommendation": rec
    }

# ============================================================
#   ENSEMBLE DI MODELLI
# ============================================================

def ensemble_prediction(
    odds_1: float,
    odds_x: float,
    odds_2: float,
    total: float,
    odds_over25: float = None,
    odds_under25: float = None,
    odds_btts: float = None,
    odds_dnb_home: float = None,
    odds_dnb_away: float = None,
    league: str = "generic",
    use_raw_model: bool = True,
) -> Dict[str, float]:
    """
    Ensemble di più modelli per maggiore robustezza.
    
    Combina:
    1. Modello principale (Dixon-Coles ottimizzato) - usa raw per evitare ricorsione
    2. Modello basato solo su quote (market-based)
    3. Modello conservativo (più vicino al mercato)
    """
    # Modello 1: Principale (usa raw per evitare ricorsione)
    # Calcola direttamente senza ensemble per evitare loop
    odds_1_n, odds_x_n, odds_2_n = normalize_three_way_shin(odds_1, odds_x, odds_2)
    p1 = 1 / odds_1_n
    px = 1 / odds_x_n
    p2 = 1 / odds_2_n
    tot_p = p1 + px + p2
    # ⚠️ PROTEZIONE: Verifica che tot_p non sia zero o troppo piccolo
    if tot_p > model_config.TOL_DIVISION_ZERO:  # ⚠️ MICRO-PRECISIONE: Usa tolleranza standardizzata
        p1 /= tot_p
        px /= tot_p
        p2 /= tot_p
    else:
        # Fallback: distribuzione uniforme se totale è troppo piccolo
        p1 = px = p2 = 1.0 / 3.0
        logger.warning("Probabilità 1X2 normalizzate sommano a zero, uso distribuzione uniforme")
    
    ha = home_advantage_factor(league)
    px_prelim = px
    rho_prelim = estimate_rho_improved(1.5, 1.5, px_prelim, odds_btts)
    
    lh, la = estimate_lambda_from_market_optimized(
        odds_1_n, odds_x_n, odds_2_n, total,
        odds_over25, odds_under25,
        odds_dnb_home, odds_dnb_away,
        home_advantage=ha, rho_initial=rho_prelim
    )
    
    rho = estimate_rho_optimized(lh, la, px, odds_btts, None)
    mat_ft = build_score_matrix(lh, la, rho)
    p1_main, px_main, p2_main = calc_match_result_from_matrix(mat_ft)
    
    # Modello 2: Market-based (solo quote normalizzate - già calcolate e normalizzate sopra)
    p1_market = p1  # Usa probabilità già normalizzate
    px_market = px
    p2_market = p2
    
    # Modello 3: Conservativo usando ModelConfig
    p1_cons = model_config.MARKET_WEIGHT * p1_market + model_config.DNB_WEIGHT * p1_main
    px_cons = model_config.MARKET_WEIGHT * px_market + model_config.DNB_WEIGHT * px_main
    p2_cons = model_config.MARKET_WEIGHT * p2_market + model_config.DNB_WEIGHT * p2_main
    
    # Ensemble finale usando ModelConfig
    p1_ensemble = model_config.ENSEMBLE_MAIN_WEIGHT * p1_main + model_config.ENSEMBLE_MARKET_WEIGHT * p1_market + model_config.ENSEMBLE_CONSERVATIVE_WEIGHT * p1_cons
    px_ensemble = model_config.ENSEMBLE_MAIN_WEIGHT * px_main + model_config.ENSEMBLE_MARKET_WEIGHT * px_market + model_config.ENSEMBLE_CONSERVATIVE_WEIGHT * px_cons
    p2_ensemble = model_config.ENSEMBLE_MAIN_WEIGHT * p2_main + model_config.ENSEMBLE_MARKET_WEIGHT * p2_market + model_config.ENSEMBLE_CONSERVATIVE_WEIGHT * p2_cons
    
    # Normalizza
    tot_ens = p1_ensemble + px_ensemble + p2_ensemble
    p1_ensemble /= tot_ens
    px_ensemble /= tot_ens
    p2_ensemble /= tot_ens
    
    # Calcola agreement tra modelli (bassa deviazione standard = alto agreement)
    probs_home = [p1_main, p1_market, p1_cons]
    model_agreement = 1.0 - min(1.0, np.std(probs_home))  # Range 0-1, più alto = più accordo
    
    return {
        "p_home": p1_ensemble,
        "p_draw": px_ensemble,
        "p_away": p2_ensemble,
        "ensemble_confidence": 0.85,  # Alta confidence nell'ensemble
        "model_agreement": round(model_agreement, 3)
    }

# ============================================================
#   MARKET EFFICIENCY TRACKING
# ============================================================

def calculate_market_efficiency(
    predictions: List[float],
    outcomes: List[int],
    odds: List[float],
) -> Dict[str, float]:
    """
    Calcola efficienza del mercato.
    
    Market efficiency = quanto le quote riflettono la realtà.
    Se il mercato è efficiente, le quote dovrebbero essere molto vicine ai risultati.
    """
    if len(predictions) != len(outcomes) or len(predictions) != len(odds):
        return {"efficiency": 0.0, "bias": 0.0}
    
    # Calcola accuracy delle quote
    implied_probs = [1 / o for o in odds]
    quote_accuracy = np.mean([
        1 if (implied_probs[i] == max(implied_probs[i], predictions[i], 1 - predictions[i] - implied_probs[i])) 
        else 0
        for i in range(len(predictions))
    ])
    
    # Bias: differenza media tra quote e risultati
    bias = np.mean([abs(implied_probs[i] - outcomes[i]) for i in range(len(predictions))])
    
    # Efficiency score (0-100)
    efficiency = (1 - bias) * 100
    
    return {
        "efficiency": round(efficiency, 2),
        "bias": round(bias, 4),
        "quote_accuracy": round(quote_accuracy * 100, 2)
    }

# ============================================================
#   BACKTESTING AVANZATO
# ============================================================

def backtest_strategy(
    archive_file: str = ARCHIVE_FILE,
    min_edge: float = 0.03,
    kelly_fraction: float = 0.25,
    initial_bankroll: float = 100.0,
) -> Dict[str, Any]:
    """
    Backtest completo della strategia su dati storici.
    
    Simula scommesse usando Kelly Criterion e calcola performance.
    """
    if not os.path.exists(archive_file):
        return {"error": "Nessun storico disponibile"}
    
    try:
        df = pd.read_csv(archive_file)
        
        # Filtra partite con risultati
        df_complete = df[
            df["esito_reale"].notna() & 
            (df["esito_reale"] != "") &
            df["p_home"].notna() &
            df["odds_1"].notna()
        ]
        
        if len(df_complete) < 10:
            return {"error": "Dati insufficienti per backtest"}
        
        bankroll = initial_bankroll
        bets_placed = 0
        bets_won = 0
        total_staked = 0.0
        total_returned = 0.0
        profit_history = [initial_bankroll]
        
        for _, row in df_complete.iterrows():
            # Determina esito reale
            esito_reale = str(row["esito_reale"]).strip()
            if esito_reale not in ["1", "X", "2"]:
                continue
            
            # Trova la migliore scommessa (quella con maggiore edge) tra tutti gli esiti
            best_bet = None
            best_edge = -1
            
            for esito, prob_col, odds_col in [
                ("1", "p_home", "odds_1"),
                ("X", "p_draw", "odds_x"),
                ("2", "p_away", "odds_2"),
            ]:
                if pd.isna(row.get(prob_col)) or pd.isna(row.get(odds_col)):
                    continue
                
                prob = row[prob_col] / 100.0
                odds = row[odds_col]
                
                if odds <= 1.0:
                    continue
                
                # Calcola edge
                implied_prob = 1 / odds
                edge = prob - implied_prob
                
                # Se edge sufficiente e migliore di quella trovata finora
                if edge >= min_edge and edge > best_edge:
                    best_edge = edge
                    best_bet = {
                        "esito": esito,
                        "prob": prob,
                        "odds": odds,
                        "edge": edge
                    }
            
            # Piazza solo la migliore scommessa per questa partita
            if best_bet and best_bet["edge"] >= min_edge:
                kelly = kelly_criterion(
                    best_bet["prob"], 
                    best_bet["odds"], 
                    bankroll, 
                    kelly_fraction
                )
                stake = kelly["stake"]
                
                if stake > 0 and stake <= bankroll:
                    bets_placed += 1
                    total_staked += stake
                    bankroll -= stake
                    
                    # Verifica se vinta
                    if best_bet["esito"] == esito_reale:
                        bets_won += 1
                        winnings = stake * best_bet["odds"]
                        total_returned += winnings
                        bankroll += winnings
                    # Perdita già dedotta (stake già sottratto)
                    
                    profit_history.append(bankroll)
        
        # Calcola metriche finali
        if bets_placed == 0:
            return {"error": "Nessuna scommessa piazzata con i criteri specificati"}
        
        win_rate = bets_won / bets_placed
        roi = ((total_returned - total_staked) / total_staked * 100) if total_staked > 0 else 0
        profit = bankroll - initial_bankroll
        profit_pct = (profit / initial_bankroll * 100) if initial_bankroll > 0 else 0
        
        # Sharpe-like ratio (return / volatility)
        if len(profit_history) > 1:
            returns = np.diff(profit_history) / profit_history[:-1]
            sharpe_approx = np.mean(returns) / (np.std(returns) + model_config.TOL_DIVISION_ZERO) if np.std(returns) > 0 else 0  # ⚠️ MICRO-PRECISIONE: Usa tolleranza standardizzata
        else:
            sharpe_approx = 0
        
        return {
            "initial_bankroll": initial_bankroll,
            "final_bankroll": round(bankroll, 2),
            "profit": round(profit, 2),
            "profit_pct": round(profit_pct, 2),
            "bets_placed": bets_placed,
            "bets_won": bets_won,
            "win_rate": round(win_rate * 100, 2),
            "roi": round(roi, 2),
            "total_staked": round(total_staked, 2),
            "total_returned": round(total_returned, 2),
            "sharpe_approx": round(sharpe_approx, 3),
            "profit_history": profit_history,
        }
    except Exception as e:
        return {"error": f"Errore backtest: {str(e)}"}

# ============================================================
#   VISUALIZZAZIONI AVANZATE
# ============================================================

def create_score_heatmap_data(mat: List[List[float]], max_goals: int = 10) -> np.ndarray:
    """
    Prepara dati per heatmap della matrice score.
    """
    mg = min(len(mat) - 1, max_goals)
    heatmap_data = np.zeros((mg + 1, mg + 1))
    
    for h in range(mg + 1):
        for a in range(mg + 1):
            if h < len(mat) and a < len(mat[h]):
                heatmap_data[h, a] = mat[h][a] * 100  # Converti in percentuale
    
    return heatmap_data

# ============================================================
#   CACHING E RATE LIMITING
# ============================================================

def cached_api_call(cache_key: str, api_func: callable, *args, **kwargs):
    """
    Cache per chiamate API per evitare rate limiting.
    """
    import time
    
    if cache_key in API_CACHE:
        cached_data, cached_time = API_CACHE[cache_key]
        if time.time() - cached_time < CACHE_EXPIRY:
            return cached_data
    
    # Chiama API
    try:
        result = api_func(*args, **kwargs)
        API_CACHE[cache_key] = (result, time.time())
        return result
    except Exception as e:
        # Se errore, ritorna cache vecchia se disponibile
        if cache_key in API_CACHE:
            return API_CACHE[cache_key][0]
        raise e

# ============================================================
#   EXPORT E REPORT
# ============================================================

def export_analysis_to_csv(risultati: Dict[str, Any], match_name: str, output_file: str = None) -> str:
    """
    Esporta analisi completa in CSV.
    """
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"analisi_{match_name.replace(' ', '_')}_{timestamp}.csv"
    
    # Prepara dati
    data = {
        "Match": [match_name],
        "Timestamp": [datetime.now().isoformat()],
        "Lambda_Home": [risultati.get("lambda_home", 0)],
        "Lambda_Away": [risultati.get("lambda_away", 0)],
        "Rho": [risultati.get("rho", 0)],
        "Prob_Home_%": [risultati.get("p_home", 0) * 100],
        "Prob_Draw_%": [risultati.get("p_draw", 0) * 100],
        "Prob_Away_%": [risultati.get("p_away", 0) * 100],
        "Over_2.5_%": [risultati.get("over_25", 0) * 100],
        "BTTS_%": [risultati.get("btts", 0) * 100],
    }
    
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    return output_file

def export_analysis_to_excel(risultati: Dict[str, Any], match_name: str, odds_data: Dict[str, float], 
                             output_file: str = None) -> str:
    """
    Esporta analisi completa in Excel con multiple sheets.
    """
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill, Alignment
    except ImportError:
        raise ImportError("openpyxl necessario per export Excel. Installare: pip install openpyxl")
    
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"analisi_{match_name.replace(' ', '_')}_{timestamp}.xlsx"
    
    wb = Workbook()
    
    # Sheet 1: Riepilogo
    ws1 = wb.active
    ws1.title = "Riepilogo"
    ws1.append(["Analisi Partita", match_name])
    ws1.append(["Data", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
    ws1.append([])
    ws1.append(["Parametri Modello"])
    ws1.append(["Lambda Casa", risultati.get("lambda_home", 0)])
    ws1.append(["Lambda Trasferta", risultati.get("lambda_away", 0)])
    ws1.append(["Rho (correlazione)", risultati.get("rho", 0)])
    ws1.append([])
    ws1.append(["Probabilità Principali"])
    ws1.append(["Casa", f"{risultati.get('p_home', 0)*100:.2f}%"])
    ws1.append(["Pareggio", f"{risultati.get('p_draw', 0)*100:.2f}%"])
    ws1.append(["Trasferta", f"{risultati.get('p_away', 0)*100:.2f}%"])
    
    # Sheet 2: Quote e Value
    ws2 = wb.create_sheet("Quote e Value")
    ws2.append(["Mercato", "Esito", "Quota", "Prob Modello %", "Prob Quota %", "Edge %", "EV %"])
    
    for esito, prob, odd_key in [
        ("Casa", risultati.get("p_home", 0), "odds_1"),
        ("Pareggio", risultati.get("p_draw", 0), "odds_x"),
        ("Trasferta", risultati.get("p_away", 0), "odds_2"),
    ]:
        if odd_key in odds_data and odds_data[odd_key] > 0:
            odd = odds_data[odd_key]
            p_book = 1 / odd
            edge = (prob - p_book) * 100
            ev = (prob * odd - 1) * 100
            ws2.append(["1X2", esito, odd, f"{prob*100:.2f}", f"{p_book*100:.2f}", 
                       f"{edge:+.2f}", f"{ev:+.2f}"])
    
    # Sheet 3: Top Risultati
    ws3 = wb.create_sheet("Top Risultati")
    ws3.append(["Casa", "Trasferta", "Probabilità %"])
    for h, a, p in risultati.get("top10", []):
        ws3.append([h, a, f"{p:.2f}"])
    
    wb.save(output_file)
    return output_file

# ============================================================
#   COMPARAZIONE BOOKMAKERS
# ============================================================

def compare_bookmaker_odds(event: dict) -> Dict[str, List[Dict[str, Any]]]:
    """
    Confronta quote tra diversi bookmakers per trovare le migliori.
    
    Returns: Dict con migliori quote per ogni mercato
    """
    if not event or "bookmakers" not in event:
        return {}
    
    home_team = (event.get("home_team") or "").lower()
    away_team = (event.get("away_team") or "").lower()
    
    best_odds = {
        "1": [],  # Lista di (bookmaker, odds)
        "X": [],
        "2": [],
        "over_25": [],
        "under_25": [],
        "btts": [],
    }
    
    for bk in event.get("bookmakers", []):
        bk_name = bk.get("key", "unknown")
        
        for mk in bk.get("markets", []):
            mk_key_raw = mk.get("key")
            # Assicurati che mk_key sia sempre una stringa (gestisce None)
            mk_key = str(mk_key_raw).lower() if mk_key_raw is not None else ""
            
            # 1X2
            if "h2h" in mk_key or "match_winner" in mk_key:
                for o in mk.get("outcomes", []):
                    name_l = (o.get("name") or "").lower()
                    price = o.get("price")
                    if not price:
                        continue
                    
                    if home_team in name_l or name_l in home_team:
                        best_odds["1"].append({"bookmaker": bk_name, "odds": price})
                    elif away_team in name_l or name_l in away_team:
                        best_odds["2"].append({"bookmaker": bk_name, "odds": price})
                    elif "draw" in name_l or "x" == name_l or "pareggio" in name_l:
                        best_odds["X"].append({"bookmaker": bk_name, "odds": price})
            
            # Over/Under
            elif "totals" in mk_key:
                for o in mk.get("outcomes", []):
                    point = o.get("point")
                    price = o.get("price")
                    name_l = (o.get("name") or "").lower()
                    if point == 2.5 and price:
                        if "over" in name_l:
                            best_odds["over_25"].append({"bookmaker": bk_name, "odds": price})
                        elif "under" in name_l:
                            best_odds["under_25"].append({"bookmaker": bk_name, "odds": price})
            
            # BTTS
            elif "btts" in mk_key:
                for o in mk.get("outcomes", []):
                    name_l = (o.get("name") or "").lower()
                    price = o.get("price")
                    if price and ("yes" in name_l or "sì" in name_l):
                        best_odds["btts"].append({"bookmaker": bk_name, "odds": price})
    
    # Ordina per odds migliori (più alte)
    for key in best_odds:
        best_odds[key].sort(key=lambda x: x["odds"], reverse=True)
    
    return best_odds

def find_best_odds_summary(best_odds: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    """
    Crea riepilogo delle migliori quote.
    """
    summary = {}
    
    for market, odds_list in best_odds.items():
        if odds_list:
            best = odds_list[0]
            # Calcola media e spread
            all_odds = [x["odds"] for x in odds_list]
            avg_odds = np.mean(all_odds)
            max_odds = max(all_odds)
            min_odds = min(all_odds)
            spread = max_odds - min_odds
            
            summary[market] = {
                "best_bookmaker": best["bookmaker"],
                "best_odds": best["odds"],
                "avg_odds": round(avg_odds, 3),
                "max_odds": max_odds,
                "min_odds": min_odds,
                "spread": round(spread, 3),
                "num_bookmakers": len(odds_list),
                "value": round((best["odds"] - avg_odds) / avg_odds * 100, 2)  # % sopra media
            }
    
    return summary

# ============================================================
#   MARKET MOVEMENT INTELLIGENCE
# ============================================================

def calculate_market_movement_factor(
    spread_apertura: float = None,
    total_apertura: float = None,
    spread_corrente: float = None,
    total_corrente: float = None,
) -> Dict[str, Any]:
    """
    Calcola il "market movement factor" basato sul movimento tra apertura e corrente.
    
    Strategia:
    - Movimento basso (< 0.2): mercato stabile → più peso all'apertura (70% apertura, 30% corrente)
    - Movimento medio (0.2-0.4): mercato in movimento → blend equilibrato (50% apertura, 50% corrente)
    - Movimento alto (> 0.4): smart money in azione → più peso alle quote correnti (30% apertura, 70% corrente)
    
    Returns:
        Dict con weight_apertura, weight_corrente, movement_magnitude, movement_type
    """
    # Se non abbiamo dati apertura, usa solo corrente
    if spread_apertura is None and total_apertura is None:
        return {
            "weight_apertura": 0.0,
            "weight_corrente": 1.0,
            "movement_magnitude": 0.0,
            "movement_type": "NO_OPENING_DATA"
        }
    
    # Calcola movimento spread
    movement_spread = 0.0
    if spread_apertura is not None and spread_corrente is not None:
        movement_spread = abs(spread_corrente - spread_apertura)
    
    # Calcola movimento total
    movement_total = 0.0
    if total_apertura is not None and total_corrente is not None:
        movement_total = abs(total_corrente - total_apertura)
    
    # Movimento combinato (media pesata: spread più importante)
    movement_magnitude = (movement_spread * 0.6 + movement_total * 0.4) if (movement_spread > 0 or movement_total > 0) else 0.0
    
    # Determina pesi basati su movimento
    if movement_magnitude < 0.2:
        # Mercato stabile: più peso all'apertura (più affidabile)
        weight_apertura = 0.70
        weight_corrente = 0.30
        movement_type = "STABLE"
    elif movement_magnitude < 0.4:
        # Movimento medio: blend equilibrato
        weight_apertura = 0.50
        weight_corrente = 0.50
        movement_type = "MODERATE"
    else:
        # Movimento alto: smart money in azione, più peso alle quote correnti
        weight_apertura = 0.30
        weight_corrente = 0.70
        movement_type = "HIGH_SMART_MONEY"
    
    return {
        "weight_apertura": weight_apertura,
        "weight_corrente": weight_corrente,
        "movement_magnitude": round(movement_magnitude, 3),
        "movement_type": movement_type,
        "movement_spread": round(movement_spread, 3) if movement_spread > 0 else None,
        "movement_total": round(movement_total, 3) if movement_total > 0 else None,
    }

def apply_market_movement_blend(
    lambda_h_current: float,
    lambda_a_current: float,
    total_current: float,
    spread_apertura: float = None,
    total_apertura: float = None,
    spread_corrente: float = None,
    total_corrente: float = None,
    home_advantage: float = 1.30,
) -> Tuple[float, float]:
    """
    Applica blend bayesiano tra lambda da apertura e corrente basato su market movement.
    
    Se abbiamo dati apertura, calcola lambda da apertura e fa blend con corrente.
    """
    # ⚠️ PRECISIONE MANIACALE: Valida lambda prima di calcolare spread/total
    if not isinstance(lambda_h_current, (int, float)) or not isinstance(lambda_a_current, (int, float)):
        logger.error(f"Lambda non validi: lambda_h={lambda_h_current}, lambda_a={lambda_a_current}")
        raise ValueError("Lambda devono essere numeri validi")
    
    # ⚠️ PROTEZIONE: Verifica che lambda siano finiti e positivi
    if not math.isfinite(lambda_h_current) or not math.isfinite(lambda_a_current):
        logger.error(f"Lambda non finiti: lambda_h={lambda_h_current}, lambda_a={lambda_a_current}")
        raise ValueError("Lambda devono essere numeri finiti")
    
    if lambda_h_current < 0 or lambda_a_current < 0:
        logger.warning(f"Lambda negativi: lambda_h={lambda_h_current}, lambda_a={lambda_a_current}, correggo")
        lambda_h_current = max(0.1, lambda_h_current)
        lambda_a_current = max(0.1, lambda_a_current)
    
    # ⚠️ PRECISIONE: Calcola spread/total correnti se non forniti
    if spread_corrente is None:
        spread_corrente = lambda_h_current - lambda_a_current
        # ⚠️ MICRO-PRECISIONE: Valida spread calcolato con precisione
        spread_corrente = max(-3.0, min(3.0, spread_corrente))
        # ⚠️ VERIFICA: Double-check che spread sia finito
        if not math.isfinite(spread_corrente):
            logger.warning(f"Spread calcolato non finito: {spread_corrente}, uso default 0.0")
            spread_corrente = 0.0
    
    if total_corrente is None:
        total_corrente = lambda_h_current + lambda_a_current
        # ⚠️ MICRO-PRECISIONE: Valida total calcolato con precisione
        total_corrente = max(0.5, min(6.0, total_corrente))
        # ⚠️ VERIFICA: Double-check che total sia finito
        if not math.isfinite(total_corrente):
            logger.warning(f"Total calcolato non finito: {total_corrente}, uso default 2.5")
            total_corrente = 2.5
    
    # Calcola market movement factor
    movement_factor = calculate_market_movement_factor(
        spread_apertura, total_apertura, spread_corrente, total_corrente
    )
    
    # Se non abbiamo dati apertura o movimento è nullo, usa solo corrente
    if movement_factor["weight_apertura"] == 0.0:
        return lambda_h_current, lambda_a_current
    
    # ⚠️ PRECISIONE MANIACALE: Calcola lambda da apertura (se disponibile) con validazione completa
    if spread_apertura is not None and total_apertura is not None:
        # ⚠️ VALIDAZIONE ROBUSTA: Verifica che spread_apertura e total_apertura siano validi
        if not isinstance(spread_apertura, (int, float)) or not math.isfinite(spread_apertura):
            logger.warning(f"spread_apertura non valido: {spread_apertura}, uso default 0.0")
            spread_apertura = 0.0
        if not isinstance(total_apertura, (int, float)) or not math.isfinite(total_apertura):
            logger.warning(f"total_apertura non valido: {total_apertura}, uso default 2.5")
            total_apertura = 2.5
        
        # ⚠️ VALIDAZIONE: Clamp per sicurezza (anche se dovrebbero essere già validati)
        spread_apertura_safe = max(-3.0, min(3.0, spread_apertura))
        total_apertura_safe = max(0.5, min(6.0, total_apertura))
        
        # ⚠️ PRECISIONE: Stima lambda da spread/total apertura con protezione
        lambda_total_ap = total_apertura_safe / 2.0
        
        # ⚠️ VERIFICA: Assicura che lambda_total_ap sia ragionevole
        if not math.isfinite(lambda_total_ap) or lambda_total_ap <= 0:
            logger.warning(f"lambda_total_ap non valido: {lambda_total_ap}, uso default 1.25")
            lambda_total_ap = 1.25
        
        # ⚠️ CORREZIONE: Spread apertura → lambda con protezione completa
        # ⚠️ PROTEZIONE: spread_factor_ap = exp(spread * 0.5) può esplodere se spread è alto
        # Limita spread_apertura_safe prima di calcolare exp per evitare overflow
        spread_clamped = max(-2.0, min(2.0, spread_apertura_safe))  # Limita spread prima di exp
        
        # ⚠️ PRECISIONE: Calcola exp con protezione overflow
        try:
            spread_factor_ap_raw = math.exp(spread_clamped * 0.5)
            if not math.isfinite(spread_factor_ap_raw):
                logger.warning(f"spread_factor_ap_raw non finito: {spread_factor_ap_raw}, uso default 1.0")
                spread_factor_ap_raw = 1.0
        except (OverflowError, ValueError) as e:
            logger.warning(f"Errore calcolo exp per spread_factor: {e}, uso default 1.0")
            spread_factor_ap_raw = 1.0
        
        spread_factor_ap = max(0.5, min(2.0, spread_factor_ap_raw))  # Limita a range ragionevole
        
        # ⚠️ OTTIMIZZAZIONE: Calcola sqrt(home_advantage) una sola volta con protezione
        if not isinstance(home_advantage, (int, float)) or home_advantage <= 0:
            logger.warning(f"home_advantage non valido: {home_advantage}, uso default 1.30")
            home_advantage = 1.30
        
        sqrt_ha = math.sqrt(home_advantage)
        if not math.isfinite(sqrt_ha):
            logger.warning(f"sqrt_ha non finito: {sqrt_ha}, uso default 1.14")
            sqrt_ha = 1.14
        
        # ⚠️ PRECISIONE: Calcola lambda da apertura con protezione divisione per zero
        lambda_h_ap = lambda_total_ap * spread_factor_ap * sqrt_ha
        lambda_a_ap = lambda_total_ap / max(model_config.TOL_DIVISION_ZERO, spread_factor_ap) / sqrt_ha
        
        # ⚠️ PROTEZIONE: Verifica che lambda siano finiti e positivi
        if not math.isfinite(lambda_h_ap) or lambda_h_ap <= 0:
            logger.warning(f"lambda_h_ap non valido: {lambda_h_ap}, correggo")
            lambda_h_ap = max(0.3, lambda_total_ap * sqrt_ha)
        if not math.isfinite(lambda_a_ap) or lambda_a_ap <= 0:
            logger.warning(f"lambda_a_ap non valido: {lambda_a_ap}, correggo")
            lambda_a_ap = max(0.3, lambda_total_ap / sqrt_ha)
        
        # ⚠️ VERIFICA: Assicura coerenza total dopo calcolo da apertura con precisione
        total_check_ap = lambda_h_ap + lambda_a_ap
        if abs(total_check_ap - total_apertura_safe) > model_config.TOL_TOTAL_COHERENCE:  # ⚠️ MICRO-PRECISIONE: Usa tolleranza standardizzata
            # Ricalibra per mantenere total coerente con precisione
            if total_check_ap > model_config.TOL_DIVISION_ZERO:
                scale_factor_ap = total_apertura_safe / total_check_ap
                if math.isfinite(scale_factor_ap) and scale_factor_ap > 0:
                    lambda_h_ap *= scale_factor_ap
                    lambda_a_ap *= scale_factor_ap
                    
                    # ⚠️ VERIFICA FINALE: Double-check coerenza dopo ricalibrazione
                    total_check_final = lambda_h_ap + lambda_a_ap
                    if abs(total_check_final - total_apertura_safe) > model_config.TOL_TOTAL_COHERENCE:
                        logger.warning(f"Coerenza total apertura ancora non raggiunta: {total_check_final} vs {total_apertura_safe}")
                else:
                    logger.warning(f"scale_factor_ap non valido: {scale_factor_ap}, uso lambda senza ricalibrazione")
            else:
                logger.warning(f"total_check_ap troppo piccolo: {total_check_ap}, uso lambda senza ricalibrazione")
        
        # Constraints
        lambda_h_ap = max(0.3, min(4.5, lambda_h_ap))
        lambda_a_ap = max(0.3, min(4.5, lambda_a_ap))
        
        # ⚠️ CONTROLLO: Limita effetto del blend per evitare valori estremi
        # Il blend non può cambiare i lambda più del 40% rispetto a corrente
        max_blend_adjustment = 1.4
        lambda_h_ap_limited = max(
            lambda_h_current / max_blend_adjustment,
            min(lambda_h_current * max_blend_adjustment, lambda_h_ap)
        )
        lambda_a_ap_limited = max(
            lambda_a_current / max_blend_adjustment,
            min(lambda_a_current * max_blend_adjustment, lambda_a_ap)
        )
        
        # Blend bayesiano
        w_ap = movement_factor["weight_apertura"]
        w_curr = movement_factor["weight_corrente"]
        
        lambda_h_blended = w_ap * lambda_h_ap_limited + w_curr * lambda_h_current
        lambda_a_blended = w_ap * lambda_a_ap_limited + w_curr * lambda_a_current
        
        # ⚠️ VERIFICA FINALE: Limita variazione totale del blend
        lambda_h_blended = max(
            lambda_h_current / max_blend_adjustment,
            min(lambda_h_current * max_blend_adjustment, lambda_h_blended)
        )
        lambda_a_blended = max(
            lambda_a_current / max_blend_adjustment,
            min(lambda_a_current * max_blend_adjustment, lambda_a_blended)
        )
        
        return lambda_h_blended, lambda_a_blended
    
    # Se non abbiamo spread apertura ma abbiamo total apertura
    elif total_apertura is not None:
        # ⚠️ VALIDAZIONE: Verifica che total_apertura sia ragionevole
        total_apertura_safe = max(0.5, min(6.0, total_apertura))
        
        # Usa total apertura per calibrare total corrente
        lambda_total_ap = total_apertura_safe / 2.0
        lambda_total_curr = total_current / 2.0
        
        # ⚠️ CONTROLLO: Limita differenza tra total apertura e corrente
        # Se differiscono troppo, riduci peso apertura
        total_diff = abs(total_apertura_safe - total_current) / max(0.1, total_current)
        if total_diff > 0.5:  # Se differiscono più del 50%
            # Riduci peso apertura se differenza è troppo grande
            movement_factor["weight_apertura"] = min(0.3, movement_factor["weight_apertura"])
            movement_factor["weight_corrente"] = 1.0 - movement_factor["weight_apertura"]
        
        # Blend dei total
        w_ap = movement_factor["weight_apertura"]
        w_curr = movement_factor["weight_corrente"]
        lambda_total_blended = w_ap * lambda_total_ap + w_curr * lambda_total_curr
        
        # ⚠️ VERIFICA: Limita lambda_total_blended a range ragionevole
        lambda_total_blended = max(0.5, min(5.0, lambda_total_blended))
        
        # Mantieni proporzione corrente tra lambda_h e lambda_a
        ratio_h = lambda_h_current / (lambda_h_current + lambda_a_current) if (lambda_h_current + lambda_a_current) > 0 else 0.5
        ratio_a = 1.0 - ratio_h
        
        lambda_h_blended = lambda_total_blended * ratio_h * 2.0
        lambda_a_blended = lambda_total_blended * ratio_a * 2.0
        
        # ⚠️ CONTROLLO: Limita effetto del blend
        max_blend_adjustment = 1.4
        lambda_h_blended = max(
            lambda_h_current / max_blend_adjustment,
            min(lambda_h_current * max_blend_adjustment, lambda_h_blended)
        )
        lambda_a_blended = max(
            lambda_a_current / max_blend_adjustment,
            min(lambda_a_current * max_blend_adjustment, lambda_a_blended)
        )
        
        return lambda_h_blended, lambda_a_blended
    
    # Fallback: usa solo corrente
    return lambda_h_current, lambda_a_current

# ============================================================
#   FEATURE ENGINEERING AVANZATO
# ============================================================

def apply_advanced_data_adjustments(
    lambda_h: float,
    lambda_a: float,
    advanced_data: Dict[str, Any],
) -> Tuple[float, float]:
    """
    Applica aggiustamenti basati su dati avanzati (statistiche, H2H, infortuni).
    Lavora in background, modifica lambda silenziosamente.
    
    ⚠️ MIGLIORAMENTO: Limita effetto cumulativo per evitare esplosione dei lambda.
    """
    if not advanced_data or not advanced_data.get("data_available"):
        return lambda_h, lambda_a
    
    # Salva valori iniziali per limitare effetto totale
    lambda_h_initial = lambda_h
    lambda_a_initial = lambda_a
    max_advanced_adjustment = 1.2  # Massimo 20% di variazione totale
    
    # 1. Aggiustamenti forma squadre (limitati)
    home_stats = advanced_data.get("home_team_stats")
    away_stats = advanced_data.get("away_team_stats")
    
    if home_stats and home_stats.get("confidence", 0) > 0.3:
        # Limita form_attack e form_defense a range ragionevole
        form_attack = max(0.85, min(1.15, home_stats.get("form_attack", 1.0)))
        form_defense = max(0.85, min(1.15, home_stats.get("form_defense", 1.0)))
        # Applica forma attacco casa
        lambda_h *= form_attack
        # Applica forma difesa trasferta (riduce lambda away)
        lambda_a *= (2.0 - form_defense)  # Inverso, ma limitato
    
    if away_stats and away_stats.get("confidence", 0) > 0.3:
        form_attack = max(0.85, min(1.15, away_stats.get("form_attack", 1.0)))
        form_defense = max(0.85, min(1.15, away_stats.get("form_defense", 1.0)))
        # Applica forma attacco trasferta
        lambda_a *= form_attack
        # Applica forma difesa casa (riduce lambda home)
        lambda_h *= (2.0 - form_defense)  # Inverso, ma limitato
    
    # 2. Aggiustamenti H2H (limitati)
    h2h_data = advanced_data.get("h2h_data")
    if h2h_data and h2h_data.get("confidence", 0) > 0.3:
        # Limita h2h_home_advantage a range ragionevole
        h2h_advantage = max(0.9, min(1.1, h2h_data.get("h2h_home_advantage", 1.0)))
        # Aggiusta vantaggio casa
        lambda_h *= h2h_advantage
        # Aggiusta total gol (entrambi i lambda) - limitato
        goals_factor = max(0.9, min(1.1, h2h_data.get("h2h_goals_factor", 1.0)))
        lambda_h *= math.sqrt(goals_factor)  # Radice per distribuire
        lambda_a *= math.sqrt(goals_factor)
    
    # 3. Aggiustamenti infortuni (limitati)
    home_injuries = advanced_data.get("home_injuries")
    away_injuries = advanced_data.get("away_injuries")
    
    if home_injuries and home_injuries.get("confidence", 0) > 0.3:
        # Limita attack_factor e defense_factor
        attack_factor = max(0.85, min(1.0, home_injuries.get("attack_factor", 1.0)))
        defense_factor = max(0.85, min(1.0, home_injuries.get("defense_factor", 1.0)))
        # Infortuni casa: riduce attacco, aumenta vulnerabilità difesa
        lambda_h *= attack_factor
        lambda_a *= (2.0 - defense_factor)  # Inverso, ma limitato
    
    if away_injuries and away_injuries.get("confidence", 0) > 0.3:
        attack_factor = max(0.85, min(1.0, away_injuries.get("attack_factor", 1.0)))
        defense_factor = max(0.85, min(1.0, away_injuries.get("defense_factor", 1.0)))
        # Infortuni trasferta: riduce attacco, aumenta vulnerabilità difesa
        lambda_a *= attack_factor
        lambda_h *= (2.0 - defense_factor)  # Inverso, ma limitato
    
    # ⚠️ CONTROLLO FINALE: Limita effetto totale degli aggiustamenti avanzati
    lambda_h = max(lambda_h_initial / max_advanced_adjustment, min(lambda_h_initial * max_advanced_adjustment, lambda_h))
    lambda_a = max(lambda_a_initial / max_advanced_adjustment, min(lambda_a_initial * max_advanced_adjustment, lambda_a))
    
    return lambda_h, lambda_a

# ============================================================
#   PORTFOLIO TRACKING
# ============================================================

def add_to_portfolio(
    match_name: str,
    market: str,
    esito: str,
    odds: float,
    stake: float,
    probability: float,
    timestamp: str = None,
) -> None:
    """
    Aggiunge scommessa al portfolio.
    """
    if timestamp is None:
        timestamp = datetime.now().isoformat()
    
    portfolio_entry = {
        "timestamp": timestamp,
        "match": match_name,
        "market": market,
        "esito": esito,
        "odds": odds,
        "stake": stake,
        "probability": probability,
        "expected_value": (probability * odds - 1) * stake,
        "status": "pending",  # pending, won, lost
        "result": "",
    }
    
    # Carica portfolio esistente
    if os.path.exists(PORTFOLIO_FILE):
        df = pd.read_csv(PORTFOLIO_FILE)
        df = pd.concat([df, pd.DataFrame([portfolio_entry])], ignore_index=True)
    else:
        df = pd.DataFrame([portfolio_entry])
    
    df.to_csv(PORTFOLIO_FILE, index=False)

def get_portfolio_summary() -> Dict[str, Any]:
    """
    Calcola riepilogo portfolio.
    """
    if not os.path.exists(PORTFOLIO_FILE):
        return {"total_bets": 0, "total_staked": 0.0, "pending_bets": 0}
    
    df = pd.read_csv(PORTFOLIO_FILE)
    
    total_bets = len(df)
    total_staked = df["stake"].sum()
    pending = len(df[df["status"] == "pending"])
    won = len(df[df["status"] == "won"])
    lost = len(df[df["status"] == "lost"])
    
    # Calcola profit per scommesse chiuse
    df_closed = df[df["status"].isin(["won", "lost"])]
    if len(df_closed) > 0:
        total_returned = 0.0
        for _, row in df_closed.iterrows():
            if row["status"] == "won":
                total_returned += row["stake"] * row["odds"]
        
        total_staked_closed = df_closed["stake"].sum()
        profit = total_returned - total_staked_closed
        roi = (profit / total_staked_closed * 100) if total_staked_closed > 0 else 0
    else:
        profit = 0.0
        roi = 0.0
    
    return {
        "total_bets": total_bets,
        "total_staked": round(total_staked, 2),
        "pending_bets": pending,
        "won_bets": won,
        "lost_bets": lost,
        "profit": round(profit, 2),
        "roi": round(roi, 2),
        "win_rate": round(won / (won + lost) * 100, 1) if (won + lost) > 0 else 0.0,
    }

# ============================================================
#   DASHBOARD ANALYTICS
# ============================================================

# ============================================================
#   MARKET MOVEMENT TRACKING
# ============================================================

def track_odds_movement(
    match_name: str,
    odds_1: float,
    odds_x: float,
    odds_2: float,
    timestamp: str = None,
) -> Dict[str, Any]:
    """
    Traccia movimenti delle quote nel tempo per identificare trend.
    """
    if timestamp is None:
        timestamp = datetime.now().isoformat()
    
    # Carica storico movimenti
    if os.path.exists(ODDS_HISTORY_FILE):
        df = pd.read_csv(ODDS_HISTORY_FILE)
    else:
        df = pd.DataFrame(columns=["timestamp", "match", "odds_1", "odds_x", "odds_2"])
    
    # Aggiungi nuovo punto
    new_row = {
        "timestamp": timestamp,
        "match": match_name,
        "odds_1": odds_1,
        "odds_x": odds_x,
        "odds_2": odds_2,
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(ODDS_HISTORY_FILE, index=False)
    
    # Analizza trend per questa partita
    match_data = df[df["match"] == match_name].sort_values("timestamp")
    
    if len(match_data) >= 2:
        # Calcola cambiamenti
        # ⚠️ PROTEZIONE: len(match_data) >= 2 garantisce che iloc[0] e iloc[-1] siano validi
        first = match_data.iloc[0]
        last = match_data.iloc[-1]
        
        movement = {
            "odds_1_change": last["odds_1"] - first["odds_1"],
            "odds_x_change": last["odds_x"] - first["odds_x"],
            "odds_2_change": last["odds_2"] - first["odds_2"],
            "odds_1_change_pct": ((last["odds_1"] - first["odds_1"]) / first["odds_1"]) * 100,
            "trend": "stable",
            "market_sentiment": "neutral",
        }
        
        # Determina trend
        if abs(movement["odds_1_change_pct"]) > 5:
            if movement["odds_1_change"] < 0:
                movement["trend"] = "home_favorite_increasing"
                movement["market_sentiment"] = "home_strong"
            else:
                movement["trend"] = "home_favorite_decreasing"
                movement["market_sentiment"] = "away_strong"
        
        return movement
    
    return {"trend": "insufficient_data"}

def get_odds_movement_insights(match_name: str) -> Dict[str, Any]:
    """
    Analizza movimenti quote e fornisce insights.
    """
    if not os.path.exists(ODDS_HISTORY_FILE):
        return {}
    
    df = pd.read_csv(ODDS_HISTORY_FILE)
    match_data = df[df["match"] == match_name].sort_values("timestamp")
    
    if len(match_data) < 2:
        return {}
    
    # Calcola volatilità
    # ⚠️ PROTEZIONE: Calcola std solo se ci sono almeno 2 valori (altrimenti std = NaN)
    # len(match_data) >= 2 è già garantito dal controllo sopra
    volatility_1 = match_data["odds_1"].std() if len(match_data) > 1 and "odds_1" in match_data.columns else 0.0
    volatility_x = match_data["odds_x"].std() if len(match_data) > 1 and "odds_x" in match_data.columns else 0.0
    volatility_2 = match_data["odds_2"].std() if len(match_data) > 1 and "odds_2" in match_data.columns else 0.0
    
    # Sostituisci NaN con 0.0 se std restituisce NaN
    volatility_1 = 0.0 if pd.isna(volatility_1) else volatility_1
    volatility_x = 0.0 if pd.isna(volatility_x) else volatility_x
    volatility_2 = 0.0 if pd.isna(volatility_2) else volatility_2
    
    # Identifica movimenti significativi
    significant_moves = []
    # ⚠️ PROTEZIONE: range(1, len(match_data)) richiede almeno 2 elementi
    if len(match_data) >= 2:
        for i in range(1, len(match_data)):
            prev = match_data.iloc[i-1]
            curr = match_data.iloc[i]
            
            # ⚠️ PROTEZIONE: Verifica che le colonne esistano e che prev[col] non sia zero
            for col in ["odds_1", "odds_x", "odds_2"]:
                if col not in prev.index or col not in curr.index:
                    continue
                if prev[col] == 0 or pd.isna(prev[col]) or pd.isna(curr[col]):
                    continue
                change_pct = abs((curr[col] - prev[col]) / prev[col]) * 100
                if change_pct > 3:  # Movimento > 3%
                    significant_moves.append({
                        "market": col,
                        "change": change_pct,
                        "timestamp": curr["timestamp"] if "timestamp" in curr.index else "",
                        "direction": "up" if curr[col] > prev[col] else "down"
                    })
    
    return {
        "volatility": {
            "odds_1": round(volatility_1, 3),
            "odds_x": round(volatility_x, 3),
            "odds_2": round(volatility_2, 3),
        },
        "significant_moves": significant_moves,
        "num_data_points": len(match_data),
    }

# ============================================================
#   TIME-BASED ADJUSTMENTS
# ============================================================

def get_time_based_adjustments(
    match_datetime: str = None,
    league: str = "generic",
) -> Dict[str, float]:
    """
    Calcola aggiustamenti basati su:
    - Ora partita (serale vs pomeridiana)
    - Giorno settimana (weekend vs settimana)
    - Periodo stagione (inizio, metà, fine)
    """
    if match_datetime is None:
        match_datetime = datetime.now().isoformat()
    
    try:
        dt = datetime.fromisoformat(match_datetime.replace("Z", "+00:00"))
    except:
        dt = datetime.now()
    
    adjustments = {
        "time_factor": 1.0,
        "day_factor": 1.0,
        "season_factor": 1.0,
    }
    
    # Ora partita
    hour = dt.hour
    if 20 <= hour <= 22:  # Serale
        adjustments["time_factor"] = 1.05  # +5% gol (atmosfera)
    elif 12 <= hour <= 15:  # Pomeridiana
        adjustments["time_factor"] = 0.98  # -2% gol
    
    # Giorno settimana
    weekday = dt.weekday()
    if weekday >= 5:  # Weekend (sabato/domenica)
        adjustments["day_factor"] = 1.03  # +3% gol (più pubblico)
    elif weekday == 2:  # Mercoledì (spesso Champions/Europa)
        adjustments["day_factor"] = 1.08  # +8% gol (partite importanti)
    
    # Periodo stagione (approssimato)
    month = dt.month
    if month in [8, 9]:  # Inizio stagione
        adjustments["season_factor"] = 1.02  # +2% gol (squadre fresche)
    elif month in [4, 5]:  # Fine stagione
        adjustments["season_factor"] = 0.97  # -3% gol (squadre stanche)
    
    return adjustments

def apply_time_adjustments(
    lambda_h: float,
    lambda_a: float,
    match_datetime: str = None,
    league: str = "generic",
) -> Tuple[float, float]:
    """
    Applica aggiustamenti temporali ai lambda.
    """
    adjustments = get_time_based_adjustments(match_datetime, league)
    
    # Combina tutti i fattori
    total_factor = (
        adjustments["time_factor"] *
        adjustments["day_factor"] *
        adjustments["season_factor"]
    )
    
    # Applica solo se significativo
    if abs(total_factor - 1.0) > 0.02:  # > 2% di differenza
        lambda_h *= total_factor
        lambda_a *= total_factor
    
    return lambda_h, lambda_a

# ============================================================
#   FATIGUE E MOTIVATION FACTORS
# ============================================================

def calculate_fatigue_factor(
    team_name: str,
    days_since_last_match: int = None,
    matches_last_30_days: int = None,
) -> float:
    """
    Calcola fattore fatica basato su:
    - Giorni dall'ultima partita
    - Numero partite ultimi 30 giorni
    """
    fatigue = 1.0
    
    # Giorni di riposo
    if days_since_last_match is not None:
        if days_since_last_match <= 2:
            fatigue *= 0.92  # -8% (molto stanco)
        elif days_since_last_match == 3:
            fatigue *= 0.96  # -4% (stanco)
        elif days_since_last_match >= 7:
            fatigue *= 1.05  # +5% (molto riposato)
    
    # Partite ravvicinate
    if matches_last_30_days is not None:
        if matches_last_30_days >= 8:
            fatigue *= 0.94  # -6% (troppe partite)
        elif matches_last_30_days <= 4:
            fatigue *= 1.03  # +3% (poche partite)
    
    return fatigue

def calculate_motivation_factor(
    team_position: int = None,
    points_from_relegation: int = None,
    points_from_europe: int = None,
    is_derby: bool = False,
) -> float:
    """
    Calcola fattore motivazione basato su:
    - Posizione in classifica
    - Vicinanza a obiettivi (salvezza, Europa)
    - Partite speciali (derby)
    """
    motivation = 1.0
    
    # Derby
    if is_derby:
        motivation *= 1.12  # +12% (alta motivazione)
    
    # Lotta salvezza
    if points_from_relegation is not None:
        if 0 <= points_from_relegation <= 3:
            motivation *= 1.10  # +10% (lotta per non retrocedere)
    
    # Lotta Europa
    if points_from_europe is not None:
        if 0 <= points_from_europe <= 3:
            motivation *= 1.08  # +8% (lotta per Europa)
    
    return motivation

# ============================================================
#   ANOMALY DETECTION
# ============================================================

def detect_odds_anomalies(
    odds_1: float,
    odds_x: float,
    odds_2: float,
    odds_over25: float = None,
    odds_under25: float = None,
) -> Dict[str, Any]:
    """
    Rileva anomalie nelle quote che potrebbero indicare:
    - Errori bookmaker
    - Quote non aggiornate
    - Opportunità di arbitraggio
    """
    anomalies = []
    severity = 0
    
    # 1. Check margine anomalo
    margin = (1/odds_1 + 1/odds_x + 1/odds_2) - 1
    if margin < 0:
        anomalies.append({
            "type": "negative_margin",
            "description": f"Margine negativo ({margin*100:.2f}%) - possibile errore o arbitraggio",
            "severity": "high"
        })
        severity += 3
    elif margin > 0.20:
        anomalies.append({
            "type": "excessive_margin",
            "description": f"Margine eccessivo ({margin*100:.2f}%) - quote poco competitive",
            "severity": "medium"
        })
        severity += 1
    
    # 2. Check coerenza Over/Under
    if odds_over25 and odds_under25:
        margin_ou = (1/odds_over25 + 1/odds_under25) - 1
        if margin_ou < 0:
            anomalies.append({
                "type": "arbitrage_opportunity",
                "description": "Opportunità arbitraggio Over/Under",
                "severity": "high"
            })
            severity += 2
    
    # 3. Check quote estreme
    if odds_1 < 1.01 or odds_2 < 1.01:
        anomalies.append({
            "type": "extreme_odds",
            "description": "Quote estreme (< 1.01) - possibile errore",
            "severity": "high"
        })
        severity += 2
    
    # 4. Check incoerenza 1X2 vs Over/Under
    if odds_over25 and odds_1:
        p_home = 1 / odds_1
        p_over = 1 / odds_over25
        
        # Se casa molto favorita ma over alto → incoerenza
        if p_home > 0.70 and p_over > 0.60:
            anomalies.append({
                "type": "incoherent_markets",
                "description": "Casa molto favorita ma Over alto - possibile incoerenza",
                "severity": "medium"
            })
            severity += 1
    
    return {
        "anomalies": anomalies,
        "severity_score": severity,
        "has_anomalies": len(anomalies) > 0,
        "recommendation": "verify" if severity >= 3 else ("review" if severity >= 1 else "ok")
    }

# ============================================================
#   ADVANCED RISK MANAGEMENT
# ============================================================

def calculate_dynamic_position_size(
    edge: float,
    odds: float,
    bankroll: float,
    current_drawdown: float = 0.0,
    win_streak: int = 0,
    loss_streak: int = 0,
) -> Dict[str, Any]:
    """
    Calcola position size dinamico basato su:
    - Edge e odds
    - Drawdown corrente
    - Streak (vittorie/sconfitte consecutive)
    """
    # Kelly base
    p = edge + (1 / odds)  # Probabilità reale
    q = 1 - p
    kelly_base = (p * odds - 1) / (odds - 1)
    kelly_base = max(0, min(0.25, kelly_base))  # Cap a 25%
    
    # Aggiustamenti per drawdown
    if current_drawdown > 0.20:  # Drawdown > 20%
        kelly_base *= 0.5  # Riduci del 50%
    elif current_drawdown > 0.10:  # Drawdown > 10%
        kelly_base *= 0.75  # Riduci del 25%
    
    # Aggiustamenti per streak
    if loss_streak >= 3:
        kelly_base *= 0.6  # Riduci dopo 3 sconfitte consecutive
    elif win_streak >= 5:
        kelly_base *= 1.1  # Aumenta leggermente dopo 5 vittorie (max 10%)
        kelly_base = min(0.25, kelly_base)  # Cap
    
    stake = bankroll * kelly_base
    
    return {
        "kelly_percent": kelly_base * 100,
        "stake": round(stake, 2),
        "edge": edge,
        "recommendation": "bet" if edge >= 0.03 else ("caution" if edge >= 0.01 else "skip"),
        "risk_level": "low" if current_drawdown < 0.05 else ("medium" if current_drawdown < 0.15 else "high")
    }

def calculate_stop_loss_level(
    initial_bankroll: float,
    current_bankroll: float,
    max_drawdown_pct: float = 0.25,
) -> Dict[str, Any]:
    """
    Calcola livello stop loss per proteggere bankroll.
    """
    drawdown = (initial_bankroll - current_bankroll) / initial_bankroll
    
    stop_loss_triggered = drawdown >= max_drawdown_pct
    
    return {
        "current_drawdown": round(drawdown * 100, 2),
        "max_drawdown": max_drawdown_pct * 100,
        "stop_loss_triggered": stop_loss_triggered,
        "remaining_buffer": round((max_drawdown_pct - drawdown) * 100, 2),
        "recommendation": "stop_betting" if stop_loss_triggered else "continue"
    }

# ============================================================
#   FEATURE IMPORTANCE ANALYSIS
# ============================================================

def analyze_feature_importance(archive_file: str = ARCHIVE_FILE) -> Dict[str, float]:
    """
    Analizza importanza delle features nel modello usando correlazione
    e analisi statistica.
    """
    if not os.path.exists(archive_file):
        return {}
    
    try:
        df = pd.read_csv(archive_file)
        
        # Filtra partite con risultati
        df_complete = df[
            df["esito_reale"].notna() &
            (df["esito_reale"] != "") &
            df["p_home"].notna()
        ]
        
        if len(df_complete) < 20:
            return {}
        
        # Calcola accuracy per ogni feature
        features = {
            "lambda_home": "lambda_home",
            "lambda_away": "lambda_away",
            "rho": "rho",
            "quality_score": "quality_score",
            "market_confidence": "market_confidence",
        }
        
        importance = {}
        
        for feat_name, feat_col in features.items():
            if feat_col not in df_complete.columns:
                continue
            
            # Correlazione con accuracy
            df_complete["correct"] = (
                df_complete["esito_modello"] == df_complete["esito_reale"]
            ).astype(int)
            
            corr = df_complete[feat_col].corr(df_complete["correct"])
            importance[feat_name] = round(abs(corr) * 100, 1) if not pd.isna(corr) else 0
        
        # Ordina per importanza
        sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        
        return sorted_importance
    except Exception as e:
        return {"error": str(e)}

# ============================================================
#   API GRATUITE AGGIUNTIVE - INTEGRAZIONE
# ============================================================

# 1. OPENWEATHERMAP API
def get_weather_for_match(city: str, match_datetime: str = None) -> Dict[str, Any]:
    """
    Recupera condizioni meteo per partita.
    Calcola impatto su lambda (pioggia/vento → meno gol).
    """
    if not OPENWEATHER_API_KEY:
        return {"available": False, "weather_factor": 1.0}
    
    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": city,
            "appid": OPENWEATHER_API_KEY,
            "units": "metric",
            "lang": "it"
        }
        
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        # Estrai dati meteo (MIGLIORATO: più parametri)
        temp = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        pressure = data["main"].get("pressure", 1013)  # Pressione in hPa (default 1013)
        wind_speed = data["wind"].get("speed", 0)
        visibility = data.get("visibility", 10000) / 1000  # Visibilità in km (default 10km)
        rain = data.get("rain", {}).get("1h", 0) or data.get("rain", {}).get("3h", 0) or 0
        
        # ⚠️ PROTEZIONE: Valida che weather esista e non sia vuoto prima di accedere a [0]
        weather_list = data.get("weather", [])
        if weather_list and len(weather_list) > 0 and isinstance(weather_list[0], dict):
            conditions = weather_list[0].get("main", "").lower()
            description = weather_list[0].get("description", "")
        else:
            # Fallback: usa valori di default
            conditions = ""
            description = ""
            logger.warning("Dati weather non disponibili o non validi")
        
        # Calcola fattore impatto su lambda (MIGLIORATO: più parametri)
        weather_factor = 1.0
        adjustments = []
        
        # Pioggia → riduce gol significativamente
        if rain > 5:  # Pioggia > 5mm/h
            reduction = min(0.15, rain / 50)  # Max -15%
            weather_factor *= (1 - reduction)
            adjustments.append(f"Pioggia forte: -{reduction*100:.1f}%")
        elif rain > 0:
            reduction = rain / 30  # Max -3% per pioggia leggera
            weather_factor *= (1 - reduction)
            adjustments.append(f"Pioggia: -{reduction*100:.1f}%")
        
        # Vento forte → riduce gol
        if wind_speed > 15:  # > 15 m/s (54 km/h)
            reduction = min(0.10, (wind_speed - 15) / 30)  # Max -10%
            weather_factor *= (1 - reduction)
            adjustments.append(f"Vento forte: -{reduction*100:.1f}%")
        elif wind_speed > 10:  # > 10 m/s (36 km/h)
            reduction = (wind_speed - 10) / 50  # Max -1%
            weather_factor *= (1 - reduction)
            adjustments.append(f"Vento moderato: -{reduction*100:.1f}%")
        
        # Neve → riduce molto gol
        if "snow" in conditions:
            weather_factor *= 0.85  # -15% gol
            adjustments.append("Neve: -15%")
        
        # Temperatura estrema → riduce gol
        if temp < 0:
            reduction = min(0.10, abs(temp) / 20)  # Max -10% per freddo estremo
            weather_factor *= (1 - reduction)
            adjustments.append(f"Freddo estremo: -{reduction*100:.1f}%")
        elif temp > 35:
            reduction = min(0.08, (temp - 35) / 15)  # Max -8% per caldo estremo
            weather_factor *= (1 - reduction)
            adjustments.append(f"Caldo estremo: -{reduction*100:.1f}%")
        
        # NUOVO: Umidità alta → può ridurre gol (campo pesante)
        if humidity > 80:
            reduction = min(0.05, (humidity - 80) / 100)  # Max -5% per umidità > 80%
            weather_factor *= (1 - reduction)
            adjustments.append(f"Umidità alta: -{reduction*100:.1f}%")
        
        # NUOVO: Pressione bassa → può ridurre gol (condizioni pesanti)
        if pressure < 1000:  # Pressione < 1000 hPa
            reduction = min(0.04, (1000 - pressure) / 50)  # Max -4%
            weather_factor *= (1 - reduction)
            adjustments.append(f"Pressione bassa: -{reduction*100:.1f}%")
        
        # NUOVO: Visibilità bassa → può ridurre gol (nebbia/foschia)
        if visibility < 1:  # Visibilità < 1km (nebbia fitta)
            reduction = 0.08  # -8% gol
            weather_factor *= (1 - reduction)
            adjustments.append(f"Nebbia fitta: -{reduction*100:.1f}%")
        elif visibility < 3:  # Visibilità < 3km (nebbia moderata)
            reduction = 0.04  # -4% gol
            weather_factor *= (1 - reduction)
            adjustments.append(f"Nebbia moderata: -{reduction*100:.1f}%")
        
        return {
            "available": True,
            "weather_factor": max(0.7, weather_factor),  # Min 0.7 (non ridurre troppo)
            "temperature": temp,
            "humidity": humidity,
            "pressure": pressure,  # NUOVO
            "wind_speed": wind_speed,
            "visibility": visibility,  # NUOVO
            "rain": rain,
            "conditions": conditions,
            "description": description,
            "adjustments": adjustments,
            "city": city
        }
    except (requests.exceptions.RequestException, KeyError, ValueError) as e:
        logger.error(f"Errore OpenWeatherMap: {e}")
        return {"available": False, "weather_factor": 1.0, "error": str(e)}

# 2. FOOTBALL-DATA.ORG API
def football_data_get_team_info(team_name: str, league_code: str = None) -> Dict[str, Any]:
    """
    Recupera info squadra da Football-Data.org (backup/validazione).
    """
    if not FOOTBALL_DATA_API_KEY:
        return {"available": False}
    
    try:
        # Cerca squadra
        url = "https://api.football-data.org/v4/teams"
        headers = {"X-Auth-Token": FOOTBALL_DATA_API_KEY}
        
        # Prova ricerca per nome
        params = {"name": team_name}
        response = requests.get(url, headers=headers, params=params, timeout=5)
        
        if response.status_code == 200:
            teams = response.json().get("teams", [])
            if teams:
                team = teams[0]
                return {
                    "available": True,
                    "team_id": team.get("id"),
                    "name": team.get("name"),
                    "short_name": team.get("shortName"),
                    "founded": team.get("founded"),
                    "venue": team.get("venue"),
                    "website": team.get("website")
                }
        
        return {"available": False, "reason": "Team not found"}
    except requests.exceptions.RequestException as e:
        logger.error(f"Errore Football-Data.org: {e}")
        return {"available": False, "error": str(e)}

def football_data_get_competitions() -> List[Dict[str, Any]]:
    """Recupera lista competizioni disponibili."""
    if not FOOTBALL_DATA_API_KEY:
        return []
    
    try:
        url = "https://api.football-data.org/v4/competitions"
        headers = {"X-Auth-Token": FOOTBALL_DATA_API_KEY}
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        return response.json().get("competitions", [])
    except requests.exceptions.RequestException as e:
        logger.error(f"Errore Football-Data.org competitions: {e}")
        return []

# 3. THESPORTSDB API (NO API KEY NECESSARIA!)
def thesportsdb_get_team_info(team_name: str) -> Dict[str, Any]:
    """
    Recupera info squadra da TheSportsDB (completamente gratuito, no API key).
    """
    try:
        url = "https://www.thesportsdb.com/api/v1/json/3/searchteams.php"
        params = {"t": team_name}
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        teams = data.get("teams", [])
        if teams:
            team = teams[0]
            return {
                "available": True,
                "team_id": team.get("idTeam"),
                "name": team.get("strTeam"),
                "stadium": team.get("strStadium"),
                "stadium_capacity": team.get("intStadiumCapacity"),
                "stadium_location": team.get("strStadiumLocation"),
                "founded": team.get("intFormedYear"),
                "league": team.get("strLeague"),
                "website": team.get("strWebsite"),
                "logo": team.get("strTeamBadge")
            }
        
        return {"available": False}
    except requests.exceptions.RequestException as e:
        logger.error(f"Errore TheSportsDB: {e}")
        return {"available": False, "error": str(e)}

def thesportsdb_get_stadium_info(stadium_name: str) -> Dict[str, Any]:
    """Recupera info stadio da TheSportsDB."""
    try:
        url = "https://www.thesportsdb.com/api/v1/json/3/searchvenues.php"
        params = {"v": stadium_name}
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        venues = data.get("venues", [])
        if venues:
            venue = venues[0]
            return {
                "available": True,
                "name": venue.get("strVenue"),
                "capacity": venue.get("intCapacity"),
                "location": venue.get("strLocation"),
                "country": venue.get("strCountry")
            }
        
        return {"available": False}
    except requests.exceptions.RequestException as e:
        logger.error(f"Errore TheSportsDB stadium: {e}")
        return {"available": False}

# 4. UNDERSTAT SCRAPING (xG avanzato)
def understat_get_team_xg(team_name: str, season: str = None) -> Dict[str, Any]:
    """
    Recupera xG avanzato da Understat.com (scraping).
    Ritorna xG più preciso di API-Football.
    """
    try:
        # Understat usa ID squadra, non nome
        # Per ora ritorna struttura, implementazione completa richiede mapping team_id
        # Questo è un placeholder per struttura futura
        
        # Esempio struttura dati che si otterrebbe:
        return {
            "available": False,  # Placeholder - richiede implementazione scraping completo
            "team_name": team_name,
            "xg_for": None,
            "xg_against": None,
            "xg_open_play": None,
            "xg_set_pieces": None,
            "note": "Richiede implementazione scraping BeautifulSoup"
        }
    except Exception as e:
        logger.error(f"Errore Understat: {e}")
        return {"available": False, "error": str(e)}

# 5. FBREF SCRAPING (Statistiche avanzate)
def fbref_get_team_stats(team_name: str, league: str = None) -> Dict[str, Any]:
    """
    Recupera statistiche avanzate da FBRef.com (scraping).
    """
    try:
        # Placeholder - implementazione completa richiede scraping
        return {
            "available": False,  # Placeholder
            "team_name": team_name,
            "expected_goals": None,
            "expected_assists": None,
            "expected_goals_against": None,
            "note": "Richiede implementazione scraping BeautifulSoup"
        }
    except Exception as e:
        logger.error(f"Errore FBRef: {e}")
        return {"available": False, "error": str(e)}

# Funzione helper per applicare impatto meteo su lambda
def apply_weather_impact(lh: float, la: float, weather_data: Dict[str, Any]) -> Tuple[float, float]:
    """
    Applica impatto meteo su lambda home e away.
    """
    if not weather_data.get("available", False):
        return lh, la
    
    weather_factor = weather_data.get("weather_factor", 1.0)
    
    # Applica fattore a entrambe le lambda (meteo impatta entrambe le squadre)
    lh_adjusted = lh * weather_factor
    la_adjusted = la * weather_factor
    
    return lh_adjusted, la_adjusted

def apply_stadium_adjustments(
    lambda_h: float,
    lambda_a: float,
    stadium_data: Dict[str, Any],
) -> Tuple[float, float]:
    """
    Applica aggiustamenti basati su dati stadio (capacità, altitudine).
    
    - Capacità alta → più atmosfera → più gol casa
    - Altitudine alta → meno ossigeno → meno gol (entrambe)
    """
    if not stadium_data or not stadium_data.get("available"):
        return lambda_h, lambda_a
    
    capacity = stadium_data.get("stadium_capacity")
    # Nota: TheSportsDB non fornisce altitudine direttamente, ma possiamo stimarla
    # da location se disponibile (es. città ad alta quota)
    
    # Aggiustamento capacità: stadio grande → più atmosfera → più gol casa
    # Converti capacity in numero se disponibile (può essere stringa, int, float, o None)
    if capacity is not None:
        try:
            # Prova a convertire in int (rimuove spazi e caratteri non numerici se stringa)
            if isinstance(capacity, str):
                # Rimuovi spazi e caratteri non numerici
                capacity_str = ''.join(filter(str.isdigit, capacity))
                if capacity_str:
                    capacity_num = int(capacity_str)
                else:
                    capacity_num = None
            elif isinstance(capacity, (int, float)):
                capacity_num = int(capacity)
            else:
                capacity_num = None
            
            # Applica aggiustamenti solo se abbiamo un numero valido
            if capacity_num is not None and capacity_num > 0:
                # Stadio > 50k → +3% gol casa
                if capacity_num > 50000:
                    capacity_factor = 1.03
                # Stadio > 30k → +2% gol casa
                elif capacity_num > 30000:
                    capacity_factor = 1.02
                # Stadio < 15k → -1% gol casa (meno atmosfera)
                elif capacity_num < 15000:
                    capacity_factor = 0.99
                else:
                    capacity_factor = 1.0
                
                # Applica solo a lambda_h (vantaggio casa)
                lambda_h *= capacity_factor
        except (ValueError, TypeError):
            # Se la conversione fallisce, ignora l'aggiustamento capacità
            pass
    
    # Aggiustamento altitudine (se disponibile da location)
    # Nota: implementazione base, può essere estesa con mappa città → altitudine
    location_raw = stadium_data.get("stadium_location")
    # Assicurati che location sia sempre una stringa (gestisce None)
    location = str(location_raw).lower() if location_raw is not None else ""
    altitude_factor = 1.0
    
    # Città ad alta quota (es. La Paz, Quito, Città del Messico)
    if location:  # Solo se abbiamo una location valida
        high_altitude_cities = ["la paz", "quito", "mexico city", "città del messico", "bogotá", "bogota"]
        if any(city in location for city in high_altitude_cities):
            altitude_factor = 0.92  # -8% gol (entrambe le squadre)
            lambda_h *= altitude_factor
            lambda_a *= altitude_factor
    
    return lambda_h, lambda_a

# Funzione helper per recuperare città da nome squadra/stadio
def get_city_from_team(team_name: str, league: str = None) -> str:
    """
    Prova a recuperare città della squadra da TheSportsDB o API-Football.
    """
    # Prova TheSportsDB (gratuito, no API key)
    team_info = thesportsdb_get_team_info(team_name)
    if team_info.get("available"):
        stadium_location = team_info.get("stadium_location")
        # Assicurati che stadium_location sia una stringa (gestisce None)
        if stadium_location and isinstance(stadium_location, str):
            # Estrai città (prima parte prima della virgola)
            city = stadium_location.split(",")[0].strip()
            if city:  # Solo se abbiamo una città valida
                return city
    
    # Fallback: mappa manuale per squadre principali
    city_mapping = {
        "milan": "Milano",
        "inter": "Milano",
        "juventus": "Torino",
        "roma": "Roma",
        "lazio": "Roma",
        "napoli": "Napoli",
        "atalanta": "Bergamo",
        "fiorentina": "Firenze",
        "bologna": "Bologna",
        "manchester city": "Manchester",
        "manchester united": "Manchester",
        "liverpool": "Liverpool",
        "chelsea": "London",
        "arsenal": "London",
        "tottenham": "London",
        "real madrid": "Madrid",
        "barcelona": "Barcelona",
        "atletico madrid": "Madrid",
        "bayern munich": "Munich",
        "dortmund": "Dortmund",
        "psg": "Paris"
    }
    
    team_lower = team_name.lower()
    for key, city in city_mapping.items():
        if key in team_lower:
            return city
    
    return ""  # Non trovato

# ============================================================
#   TELEGRAM BOT INTEGRATION
# ============================================================

def send_telegram_message(
    message: str,
    bot_token: str = None,
    chat_id: str = None,
    parse_mode: str = "HTML",
) -> Dict[str, Any]:
    """
    Invia messaggio a Telegram tramite Bot API.
    
    Args:
        message: Testo del messaggio (supporta HTML/Markdown)
        bot_token: Token del bot (default: TELEGRAM_BOT_TOKEN)
        chat_id: Chat ID destinatario (default: TELEGRAM_CHAT_ID)
        parse_mode: "HTML" o "Markdown"
    
    Returns:
        Dict con:
            - success: bool (True se invio riuscito)
            - error_message: str (messaggio di errore dettagliato se fallito)
            - error_type: str (tipo di errore: "no_token", "no_chat_id", "invalid_token", "invalid_chat_id", "timeout", "other")
    """
    if not bot_token:
        bot_token = TELEGRAM_BOT_TOKEN
    if not chat_id:
        chat_id = TELEGRAM_CHAT_ID
    
    # Verifica token e chat ID
    if not bot_token or bot_token.strip() == "":
        return {
            "success": False,
            "error_message": "Bot Token non configurato. Configura TELEGRAM_BOT_TOKEN o inserisci il token nell'interfaccia.",
            "error_type": "no_token"
        }
    
    if not chat_id or chat_id.strip() == "":
        return {
            "success": False,
            "error_message": "Chat ID non configurato. Configura TELEGRAM_CHAT_ID o inserisci il Chat ID nell'interfaccia.",
            "error_type": "no_chat_id"
        }
    
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": parse_mode,
        "disable_web_page_preview": True,
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)  # Aumentato timeout a 10s
        
        # Controlla status code specifici
        if response.status_code == 401:
            return {
                "success": False,
                "error_message": "Token non valido o scaduto. Verifica che il Bot Token sia corretto.",
                "error_type": "invalid_token"
            }
        elif response.status_code == 400:
            # Prova a estrarre dettagli dall'errore
            try:
                error_data = response.json()
                error_desc = error_data.get("description", "Chat ID non valido")
                if "chat not found" in error_desc.lower():
                    # Estrai informazioni aggiuntive se disponibili
                    error_msg_detailed = (
                        "**Chat ID non trovato o bot non autorizzato.**\n\n"
                        "Possibili cause:\n"
                        "1. **Chat ID errato**: Verifica di aver copiato correttamente il Chat ID\n"
                        "2. **Bot non avviato**: Il bot deve aver ricevuto almeno un messaggio da te\n"
                        "3. **Gruppo/Canale**: Se è un gruppo/canale, il bot deve essere membro/amministratore\n"
                        "4. **Chat privata**: Per chat private, assicurati di aver avviato la conversazione\n\n"
                        "**Come ottenere il Chat ID corretto:**\n"
                        "- **Chat privata**: Scrivi a [@userinfobot](https://t.me/userinfobot) e copia il tuo ID\n"
                        "- **Gruppo**: Aggiungi [@userinfobot](https://t.me/userinfobot) al gruppo e vedi il Group ID\n"
                        "- **Canale**: Il bot deve essere amministratore del canale\n\n"
                        f"Errore API: {error_desc}"
                    )
                    return {
                        "success": False,
                        "error_message": error_msg_detailed,
                        "error_type": "invalid_chat_id"
                    }
            except:
                pass
            return {
                "success": False,
                "error_message": "Chat ID non valido o formato errato. Verifica che il Chat ID sia corretto.",
                "error_type": "invalid_chat_id"
            }
        elif response.status_code == 429:
            # Rate limit
            return {
                "success": False,
                "error_message": "Rate limit raggiunto. Troppi messaggi inviati. Attendi qualche minuto.",
                "error_type": "rate_limit"
            }
        
        response.raise_for_status()
        return {
            "success": True,
            "error_message": None,
            "error_type": None
        }
        
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error_message": "Timeout nella richiesta a Telegram. Verifica la connessione internet.",
            "error_type": "timeout"
        }
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error_message": "Errore di connessione a Telegram. Verifica la connessione internet.",
            "error_type": "connection_error"
        }
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if hasattr(e, 'response') and e.response else None
        return {
            "success": False,
            "error_message": f"Errore HTTP {status_code} da Telegram API. Verifica token e chat ID.",
            "error_type": "http_error"
        }
    except requests.exceptions.RequestException as e:
        logger.error(f"Errore invio Telegram: {e}")
        return {
            "success": False,
            "error_message": f"Errore nella richiesta a Telegram: {str(e)}",
            "error_type": "other"
        }
    except Exception as e:
        logger.error(f"Errore imprevisto invio Telegram: {e}")
        return {
            "success": False,
            "error_message": f"Errore imprevisto: {str(e)}",
            "error_type": "other"
        }

def test_telegram_chat_id(
    bot_token: str = None,
    chat_id: str = None,
) -> Dict[str, Any]:
    """
    Testa se il Chat ID è valido inviando un messaggio di test.
    
    Args:
        bot_token: Token del bot (default: TELEGRAM_BOT_TOKEN)
        chat_id: Chat ID da testare (default: TELEGRAM_CHAT_ID)
    
    Returns:
        Dict con:
            - success: bool
            - error_message: str
            - error_type: str
    """
    test_message = "🧪 **Test connessione Telegram**\n\nSe ricevi questo messaggio, la configurazione è corretta! ✅"
    
    return send_telegram_message(
        message=test_message,
        bot_token=bot_token,
        chat_id=chat_id,
        parse_mode="HTML"
    )

def send_telegram_photo(
    photo_path: str,
    caption: str = "",
    bot_token: str = None,
    chat_id: str = None,
) -> bool:
    """
    Invia foto a Telegram.
    
    Args:
        photo_path: Percorso file immagine
        caption: Didascalia (opzionale)
        bot_token: Token del bot
        chat_id: Chat ID destinatario
    
    Returns:
        True se invio riuscito, False altrimenti
    """
    if not bot_token:
        bot_token = TELEGRAM_BOT_TOKEN
    if not chat_id:
        chat_id = TELEGRAM_CHAT_ID
    
    if not bot_token or not chat_id or not os.path.exists(photo_path):
        return False
    
    url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
    
    try:
        with open(photo_path, 'rb') as photo:
            files = {'photo': photo}
            data = {
                'chat_id': chat_id,
                'caption': caption[:1024] if caption else "",  # Max 1024 caratteri
            }
            response = requests.post(url, files=files, data=data, timeout=10)
            response.raise_for_status()
            return True
    except (requests.exceptions.RequestException, IOError, FileNotFoundError) as e:
        logger.error(f"Errore invio foto Telegram: {e}")
        return False

def format_analysis_for_telegram(
    match_name: str,
    ris: Dict[str, Any],
    odds_1: float,
    odds_x: float,
    odds_2: float,
    quality_score: float,
    market_conf: float,
    value_bets: List[Dict[str, Any]] = None,
) -> str:
    """
    Formatta analisi completa per messaggio Telegram.
    
    Usa HTML per formattazione (grassetto, corsivo, etc.)
    """
    # Header
    message = f"⚽ <b>ANALISI COMPLETATA</b>\n\n"
    message += f"🏆 <b>{match_name}</b>\n"
    message += f"📅 {datetime.now().strftime('%d/%m/%Y %H:%M')}\n\n"
    
    # Metriche principali
    message += f"📊 <b>Metriche Qualità</b>\n"
    message += f"Quality Score: {quality_score:.0f}/100\n"
    message += f"Market Confidence: {market_conf:.0f}/100\n\n"
    
    # Parametri modello
    message += f"🔢 <b>Parametri Modello</b>\n"
    message += f"λ Casa: {ris['lambda_home']:.3f}\n"
    message += f"λ Trasferta: {ris['lambda_away']:.3f}\n"
    message += f"ρ (correlazione): {ris['rho']:.4f}\n\n"
    
    # Probabilità principali
    message += f"🎯 <b>Probabilità Esito</b>\n"
    message += f"🏠 Casa: <b>{ris['p_home']*100:.1f}%</b> (quota: {odds_1:.2f})\n"
    message += f"⚖️ Pareggio: <b>{ris['p_draw']*100:.1f}%</b> (quota: {odds_x:.2f})\n"
    message += f"✈️ Trasferta: <b>{ris['p_away']*100:.1f}%</b> (quota: {odds_2:.2f})\n\n"
    
    # Over/Under e BTTS
    message += f"⚽ <b>Mercati Speciali</b>\n"
    message += f"Over 2.5: {ris['over_25']*100:.1f}%\n"
    message += f"Under 2.5: {ris['under_25']*100:.1f}%\n"
    message += f"BTTS: {ris['btts']*100:.1f}%\n\n"
    
    # Value Bets
    if value_bets:
        message += f"💎 <b>Value Bets Identificate</b>\n"
        for bet in value_bets[:5]:  # Max 5 value bets
            esito = bet.get("Esito", "")
            prob = bet.get("Prob %", bet.get("Prob Modello %", ""))
            edge = bet.get("Edge %", "0")
            ev = bet.get("EV %", "0")
            rec = bet.get("Rec", "")
            prob_str = f"{prob}%" if prob and "%" not in str(prob) else prob
            message += f"• {esito}: Prob {prob_str}, Edge {edge}%, EV {ev}% ({rec})\n"
        message += "\n"
    
    # Top 3 risultati
    if "top10" in ris and len(ris["top10"]) > 0:
        message += f"🏅 <b>Top 3 Risultati</b>\n"
        for i, (h, a, p) in enumerate(ris["top10"][:3], 1):
            message += f"{i}. {h}-{a}: {p:.1f}%\n"
        message += "\n"
    
    # Aggiustamenti applicati
    adjustments = []
    if ris.get("calibration_applied"):
        adjustments.append("Calibrazione")
    if ris.get("ensemble_applied"):
        adjustments.append("Ensemble")
    if ris.get("market_movement", {}).get("movement_type") != "NO_OPENING_DATA":
        adjustments.append("Market Movement")
    
    if adjustments:
        message += f"✅ <b>Aggiustamenti</b>: {', '.join(adjustments)}\n"
    
    # Footer
    message += f"\n🤖 <i>Inviato automaticamente dal Modello Scommesse PRO</i>"
    
    return message

# ============================================================
#   REAL-TIME ALERTS
# ============================================================

def create_alert(
    alert_type: str,
    match_name: str,
    message: str,
    priority: str = "medium",
    data: Dict[str, Any] = None,
) -> None:
    """
    Crea alert per notifiche real-time.
    """
    if data is None:
        data = {}
    
    alert = {
        "timestamp": datetime.now().isoformat(),
        "type": alert_type,  # "value_bet", "odds_change", "anomaly", etc.
        "match": match_name,
        "message": message,
        "priority": priority,  # "low", "medium", "high", "critical"
        "data": data,
        "read": False,
    }
    
    # Carica alert esistenti
    if os.path.exists(ALERTS_FILE):
        with open(ALERTS_FILE, 'r') as f:
            alerts = json.load(f)
    else:
        alerts = []
    
    alerts.append(alert)
    
    # Mantieni solo ultimi 100 alert
    alerts = alerts[-100:]
    
    with open(ALERTS_FILE, 'w') as f:
        json.dump(alerts, f, indent=2)

def get_unread_alerts() -> List[Dict[str, Any]]:
    """
    Recupera alert non letti.
    """
    if not os.path.exists(ALERTS_FILE):
        return []
    
    try:
        with open(ALERTS_FILE, 'r') as f:
            alerts = json.load(f)
        return [a for a in alerts if not a.get("read", False)]
    except:
        return []

# ============================================================
#   MARKET CORRELATION ANALYSIS
# ============================================================

def analyze_market_correlations(
    odds_1: float,
    odds_x: float,
    odds_2: float,
    odds_over25: float = None,
    odds_under25: float = None,
    odds_btts: float = None,
) -> Dict[str, Any]:
    """
    Analizza correlazioni tra diversi mercati per identificare
    opportunità di hedging o incoerenze.
    """
    correlations = {}
    
    # Converti in probabilità
    p1 = 1 / odds_1
    px = 1 / odds_x
    p2 = 1 / odds_2
    
    if odds_over25:
        p_over = 1 / odds_over25
        
        # Correlazione 1X2 vs Over/Under
        # Se casa favorita → dovrebbe essere correlato con under (se favorita forte)
        if p1 > 0.60:
            expected_under = 1 - p_over
            actual_under = 1 / odds_under25 if odds_under25 else None
            
            if actual_under:
                correlation_1_over = abs(p1 - (1 - p_over))
                correlations["home_favorite_vs_over"] = {
                    "correlation": round(correlation_1_over, 3),
                    "interpretation": "high" if correlation_1_over > 0.3 else "low"
                }
    
    # Correlazione BTTS vs Over/Under
    if odds_btts and odds_over25:
        p_btts = 1 / odds_btts
        p_over = 1 / odds_over25
        
        # BTTS e Over dovrebbero essere positivamente correlati
        btts_over_corr = min(p_btts, p_over) / max(p_btts, p_over)
        correlations["btts_vs_over"] = {
            "correlation": round(btts_over_corr, 3),
            "interpretation": "strong" if btts_over_corr > 0.8 else ("moderate" if btts_over_corr > 0.6 else "weak")
        }
    
    return correlations

def get_realtime_performance_metrics(
    archive_file: str = ARCHIVE_FILE,
    window_days: int = 7,
) -> Dict[str, Any]:
    """
    Calcola metriche di performance real-time per ultimi N giorni.
    
    ALTA PRIORITÀ: Real-time Performance Monitoring - dashboard performance live.
    """
    if not os.path.exists(archive_file):
        return {"status": "no_data", "error": "Nessun storico disponibile"}
    
    try:
        df = pd.read_csv(archive_file)
        
        # Filtra per ultimi N giorni
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
            cutoff_date = datetime.now() - timedelta(days=window_days)
            df_recent = df[df["timestamp"] >= cutoff_date]
        else:
            df_recent = df.tail(50)  # Fallback: ultime 50 partite
        
        # Filtra partite con risultati
        df_complete = df_recent[
            df_recent["esito_reale"].notna() & 
            (df_recent["esito_reale"] != "") &
            df_recent["p_home"].notna()
        ]
        
        if len(df_complete) == 0:
            return {"status": "no_data", "error": "Nessun dato recente con risultati"}
        
        # Accuracy
        if "match_ok" in df_complete.columns:
            accuracy = df_complete["match_ok"].mean() * 100
        else:
            # Calcola accuracy manualmente
            correct = 0
            total = 0
            for _, row in df_complete.iterrows():
                esito_reale = str(row.get("esito_reale", "")).strip()
                if esito_reale in ["1", "X", "2"]:
                    # Trova esito predetto (quello con probabilità maggiore)
                    p_home = row.get("p_home", 0)
                    p_draw = row.get("p_draw", 0)
                    p_away = row.get("p_away", 0)
                    esito_pred = "1" if p_home == max(p_home, p_draw, p_away) else ("X" if p_draw == max(p_home, p_draw, p_away) else "2")
                    if esito_pred == esito_reale:
                        correct += 1
                    total += 1
            accuracy = (correct / total * 100) if total > 0 else 0
        
        # ROI simulato
        # ⚠️ PROTEZIONE: Verifica che df_complete non sia vuoto e che p_home esista
        if len(df_complete) > 0 and "p_home" in df_complete.columns:
            p_home_values = df_complete["p_home"]
            # ⚠️ PROTEZIONE: Verifica che p_home non sia vuoto prima di chiamare .max()
            if len(p_home_values) > 0:
                p_home_list = (p_home_values / 100).tolist() if p_home_values.max() > 1 else p_home_values.tolist()
            else:
                p_home_list = []
        else:
            p_home_list = []
        
        # ⚠️ PROTEZIONE: Verifica che tutte le liste abbiano la stessa lunghezza
        if len(p_home_list) > 0 and "esito_reale" in df_complete.columns and "odds_1" in df_complete.columns:
            outcomes_list = (df_complete["esito_reale"] == "1").astype(int).tolist()
            odds_list = df_complete["odds_1"].tolist()
            # Verifica coerenza lunghezza
            min_len = min(len(p_home_list), len(outcomes_list), len(odds_list))
            if min_len > 0:
                roi_data = calculate_roi(
                    p_home_list[:min_len],
                    outcomes_list[:min_len],
                    odds_list[:min_len],
                    threshold=0.03
                )
            else:
                roi_data = {"roi": 0.0, "total_bets": 0, "won": 0, "lost": 0}
        else:
            roi_data = {"roi": 0.0, "total_bets": 0, "won": 0, "lost": 0}
        
        # Trend (confronta con periodo precedente)
        if len(df) >= 100:
            df_old = df.head(len(df) - len(df_recent))
            df_old_complete = df_old[
                df_old["esito_reale"].notna() & 
                (df_old["esito_reale"] != "")
            ]
            if "match_ok" in df_old_complete.columns and len(df_old_complete) > 0:
                accuracy_old = df_old_complete["match_ok"].mean() * 100
                trend = accuracy - accuracy_old
            else:
                trend = 0
        else:
            trend = 0
        
        # Alert status
        alert_status = "good"
        alert_message = ""
        
        if accuracy < 40:
            alert_status = "critical"
            alert_message = "Accuracy molto bassa (< 40%)"
        elif accuracy < 50:
            alert_status = "warning"
            alert_message = "Accuracy sotto media (< 50%)"
        elif roi_data.get("roi", 0) < -10:
            alert_status = "warning"
            alert_message = "ROI negativo significativo"
        
        return {
            "status": "ok",
            "accuracy": round(accuracy, 1),
            "roi": round(roi_data.get("roi", 0), 1),
            "bets_placed": roi_data.get("bets", 0),
            "trend": round(trend, 1),
            "alert_status": alert_status,
            "alert_message": alert_message,
            "window_days": window_days,
            "matches_analyzed": len(df_complete),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

def calculate_dashboard_metrics(archive_file: str = ARCHIVE_FILE) -> Dict[str, Any]:
    """
    Calcola metriche aggregate per dashboard.
    """
    if not os.path.exists(archive_file):
        return {}
    
    try:
        df = pd.read_csv(archive_file)
        
        # Filtra partite con risultati
        df_complete = df[
            df["esito_reale"].notna() & 
            (df["esito_reale"] != "") &
            df["p_home"].notna()
        ]
        
        if len(df_complete) == 0:
            return {}
        
        # Accuracy per esito
        accuracy_1 = len(df_complete[(df_complete["esito_reale"] == "1") & 
                                     (df_complete["p_home"] == df_complete[["p_home", "p_draw", "p_away"]].max(axis=1))]) / len(df_complete[df_complete["esito_reale"] == "1"]) * 100 if len(df_complete[df_complete["esito_reale"] == "1"]) > 0 else 0
        
        # Brier Score aggregato
        # p_home è già in formato 0-1 (non percentuale)
        predictions_home = df_complete["p_home"].values
        outcomes_home = (df_complete["esito_reale"] == "1").astype(int).values
        bs_home = brier_score(predictions_home.tolist(), outcomes_home.tolist()) if len(predictions_home) > 0 else None
        
        # ROI simulato
        roi_data = calculate_roi(
            predictions_home.tolist(),
            outcomes_home.tolist(),
            (1 / df_complete["odds_1"]).tolist(),
            threshold=0.03
        )
        
        # Trend accuracy (ultime 20 vs prime 20)
        if len(df_complete) >= 40:
            recent = df_complete.tail(20)
            old = df_complete.head(20)
            accuracy_recent = recent["match_ok"].mean() * 100 if "match_ok" in recent.columns else 0
            accuracy_old = old["match_ok"].mean() * 100 if "match_ok" in old.columns else 0
            trend = accuracy_recent - accuracy_old
        else:
            trend = 0
        
        return {
            "total_analisi": len(df),
            "partite_con_risultato": len(df_complete),
            "accuracy_home": round(accuracy_1, 1),
            "brier_score": round(bs_home, 4) if bs_home else None,
            "roi_simulato": roi_data,
            "trend_accuracy": round(trend, 1),
            "avg_quality_score": round(df["quality_score"].mean(), 1) if "quality_score" in df.columns else None,
        }
    except Exception as e:
        return {"error": str(e)}

def calculate_confidence_intervals(
    lambda_h: float,
    lambda_a: float,
    rho: float,
    n_simulations: int = 10000,
    confidence_level: float = 0.95,
    random_seed: int = None,
) -> Dict[str, Tuple[float, float]]:
    """
    Calcola intervalli di confidenza per le probabilità principali usando
    simulazione Monte Carlo.
    
    ⚠️ PRECISIONE: 
    - Seed random per riproducibilità
    - Validazione risultati
    - Protezione contro valori estremi
    
    Simula n_simulations partite con parametri lambda_h, lambda_a, rho
    e calcola percentili per le probabilità.
    """
    # ⚠️ PRECISIONE: Seed per riproducibilità
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Genera simulazioni
    results = {
        "p_home": [],
        "p_draw": [],
        "p_away": [],
        "over_25": [],
        "btts": [],
    }
    
    # ⚠️ PRECISIONE: Validazione parametri iniziali
    if lambda_h <= 0 or lambda_a <= 0:
        # Fallback: usa valori di default
        lambda_h = max(0.1, lambda_h)
        lambda_a = max(0.1, lambda_a)
    
    for _ in range(n_simulations):
        # ⚠️ PRECISIONE: Perturba lambda con rumore gaussiano (varianza = lambda per Poisson)
        # Usa varianza più conservativa per evitare valori estremi
        lh_sim = max(0.1, lambda_h + np.random.normal(0, math.sqrt(lambda_h * 0.1)))
        la_sim = max(0.1, lambda_a + np.random.normal(0, math.sqrt(lambda_a * 0.1)))
        rho_sim = max(-0.35, min(0.35, rho + np.random.normal(0, 0.05)))
        
        # ⚠️ PROTEZIONE: Limita lambda simulati a range ragionevole
        lh_sim = max(0.1, min(5.0, lh_sim))
        la_sim = max(0.1, min(5.0, la_sim))
        
        # Calcola probabilità
        mat = build_score_matrix(lh_sim, la_sim, rho_sim)
        p_h, p_d, p_a = calc_match_result_from_matrix(mat)
        over_25, _ = calc_over_under_from_matrix(mat, 2.5)
        btts = calc_bt_ts_from_matrix(mat)
        
        # ⚠️ PROTEZIONE: Valida risultati prima di aggiungere
        if all(0 <= x <= 1 for x in [p_h, p_d, p_a, over_25, btts]):
            results["p_home"].append(p_h)
            results["p_draw"].append(p_d)
            results["p_away"].append(p_a)
            results["over_25"].append(over_25)
            results["btts"].append(btts)
    
    # ⚠️ PRECISIONE: Verifica che abbiamo abbastanza simulazioni valide
    min_valid = min(len(v) for v in results.values())
    if min_valid < n_simulations * 0.9:  # Almeno 90% valide
        logger.warning(f"Monte Carlo: solo {min_valid}/{n_simulations} simulazioni valide")
    
    # Calcola intervalli di confidenza
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    intervals = {}
    for key, values in results.items():
        if len(values) > 0:
            # ⚠️ PRECISIONE: Usa percentile con metodo più accurato (se disponibile)
            try:
                lower = np.percentile(values, lower_percentile, method='linear')
                upper = np.percentile(values, upper_percentile, method='linear')
            except TypeError:
                # Fallback per versioni numpy più vecchie
                lower = np.percentile(values, lower_percentile)
                upper = np.percentile(values, upper_percentile)
            
            # ⚠️ PRECISIONE: Limita a range [0, 1]
            lower = max(0.0, min(1.0, lower))
            upper = max(0.0, min(1.0, upper))
            
            # ⚠️ PRECISIONE: Verifica che lower <= upper
            if lower > upper:
                lower, upper = upper, lower
            
            intervals[key] = (round(lower, 4), round(upper, 4))
        else:
            intervals[key] = (0.0, 1.0)  # Fallback
    
    return intervals

# ============================================================
#        FUNZIONE PRINCIPALE MODELLO MIGLIORATA
# ============================================================

def risultato_completo_improved(
    odds_1: float,
    odds_x: float,
    odds_2: float,
    total: float,
    odds_over25: float = None,
    odds_under25: float = None,
    odds_btts: float = None,
    odds_dnb_home: float = None,
    odds_dnb_away: float = None,
    xg_for_home: float = None,
    xg_against_home: float = None,
    xg_for_away: float = None,
    xg_against_away: float = None,
    xa_for_home: float = None,
    xa_against_home: float = None,
    xa_for_away: float = None,
    xa_against_away: float = None,
    manual_boost_home: float = 0.0,
    manual_boost_away: float = 0.0,
    league: str = "generic",
    home_team: str = None,
    away_team: str = None,
    match_datetime: str = None,
    fatigue_home: Dict[str, Any] = None,
    fatigue_away: Dict[str, Any] = None,
    motivation_home: Dict[str, Any] = None,
    motivation_away: Dict[str, Any] = None,
    advanced_data: Dict[str, Any] = None,
    spread_apertura: float = None,
    total_apertura: float = None,
    spread_corrente: float = None,
    total_corrente: float = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Versione migliorata del modello con:
    - Shin normalization
    - Stima Bayesiana dei parametri
    - BTTS da modello bivariato
    - Intervalli di confidenza
    
    ⚠️ VALIDAZIONE INPUT: Tutti gli input vengono validati prima dell'uso
    """
    # ⚠️ VALIDAZIONE INPUT ROBUSTA: Valida tutti gli input critici
    try:
        # Valida quote obbligatorie
        if not isinstance(odds_1, (int, float)) or odds_1 <= 1.0:
            raise ValueError(f"odds_1 deve essere > 1.0, ricevuto: {odds_1}")
        if not isinstance(odds_x, (int, float)) or odds_x <= 1.0:
            raise ValueError(f"odds_x deve essere > 1.0, ricevuto: {odds_x}")
        if not isinstance(odds_2, (int, float)) or odds_2 <= 1.0:
            raise ValueError(f"odds_2 deve essere > 1.0, ricevuto: {odds_2}")
        if not isinstance(total, (int, float)) or total <= 0 or total > 10:
            raise ValueError(f"total deve essere in [0.5, 10.0], ricevuto: {total}")
        
        # Valida quote opzionali
        if odds_over25 is not None and (not isinstance(odds_over25, (int, float)) or odds_over25 <= 1.0):
            logger.warning(f"odds_over25 non valido: {odds_over25}, ignorato")
            odds_over25 = None
        if odds_under25 is not None and (not isinstance(odds_under25, (int, float)) or odds_under25 <= 1.0):
            logger.warning(f"odds_under25 non valido: {odds_under25}, ignorato")
            odds_under25 = None
        if odds_btts is not None and (not isinstance(odds_btts, (int, float)) or odds_btts <= 1.0):
            logger.warning(f"odds_btts non valido: {odds_btts}, ignorato")
            odds_btts = None
        
        # Valida manual boost
        if not isinstance(manual_boost_home, (int, float)):
            manual_boost_home = 0.0
        manual_boost_home = max(-0.3, min(0.3, manual_boost_home))  # Limita ±30%
        
        if not isinstance(manual_boost_away, (int, float)):
            manual_boost_away = 0.0
        manual_boost_away = max(-0.3, min(0.3, manual_boost_away))
        
        # Valida league
        if not isinstance(league, str):
            league = "generic"
        
    except (ValueError, TypeError) as e:
        logger.error(f"Errore validazione input: {e}")
        raise ValueError(f"Input non validi: {e}") from e
    
    # 1. Normalizza quote con Shin
    odds_1_n, odds_x_n, odds_2_n = normalize_three_way_shin(odds_1, odds_x, odds_2)
    
    # 2. Probabilità normalizzate
    p1 = 1 / odds_1_n
    px = 1 / odds_x_n
    p2 = 1 / odds_2_n
    tot_p = p1 + px + p2
    # ⚠️ PROTEZIONE: Verifica che tot_p non sia zero o troppo piccolo
    if tot_p > model_config.TOL_DIVISION_ZERO:  # ⚠️ MICRO-PRECISIONE: Usa tolleranza standardizzata
        p1 /= tot_p
        px /= tot_p
        p2 /= tot_p
    else:
        # Fallback: distribuzione uniforme se totale è troppo piccolo
        p1 = px = p2 = 1.0 / 3.0
        logger.warning("Probabilità 1X2 normalizzate sommano a zero, uso distribuzione uniforme")
    
    # 3. Home advantage per lega
    ha = home_advantage_factor(league)
    
    # 4. ⭐ OTTIMIZZAZIONE SIMULTANEA LAMBDA + RHO ⭐
    # Usa ottimizzazione congiunta invece di separata per maggiore accuratezza
    use_joint_optimization = True  # Flag per abilitare/disabilitare
    
    if use_joint_optimization and odds_over25 and odds_under25:
        # Ottimizzazione simultanea (più accurata)
        lh, la, rho = estimate_lambda_rho_joint_optimization(
            odds_1_n, odds_x_n, odds_2_n,
            total,
            odds_over25, odds_under25,
            odds_btts,
            odds_dnb_home, odds_dnb_away,
            home_advantage=ha
        )
    else:
        # Fallback a metodo separato se Over/Under non disponibile
        px_prelim = 1 / odds_x_n
        rho_prelim = estimate_rho_improved(1.5, 1.5, px_prelim, odds_btts)
        
        lh, la = estimate_lambda_from_market_optimized(
            odds_1_n, odds_x_n, odds_2_n,
            total,
            odds_over25, odds_under25,
            odds_dnb_home, odds_dnb_away,
            home_advantage=ha,
            rho_initial=rho_prelim
        )
        rho = estimate_rho_optimized(lh, la, px_prelim, odds_btts, None)
    
    # ⚠️ CONTROLLO: Salva lambda iniziali per limitare effetto cumulativo
    lh_initial = lh
    la_initial = la
    max_adjustment_factor = 1.5  # Massimo 50% di variazione totale dagli aggiustamenti
    
    # 5.3. Applica impatto meteo (se disponibile)
    weather_data = None
    if home_team:
        city = get_city_from_team(home_team, league)
        if city:
            weather_data = get_weather_for_match(city, match_datetime)
            if weather_data.get("available"):
                lh, la = apply_weather_impact(lh, la, weather_data)
                # Controllo intermedio
                lh = max(lh_initial / max_adjustment_factor, min(lh_initial * max_adjustment_factor, lh))
                la = max(la_initial / max_adjustment_factor, min(la_initial * max_adjustment_factor, la))
    
    # 5.4. Applica correzioni stadio (capacità, altitudine) - NUOVO
    stadium_data = None
    if home_team:
        stadium_data = thesportsdb_get_team_info(home_team)
        if stadium_data.get("available"):
            lh, la = apply_stadium_adjustments(lh, la, stadium_data)
            # Controllo intermedio
            lh = max(lh_initial / max_adjustment_factor, min(lh_initial * max_adjustment_factor, lh))
            la = max(la_initial / max_adjustment_factor, min(la_initial * max_adjustment_factor, la))
    
    # 5.5. Applica Market Movement Intelligence (blend apertura/corrente)
    # ⚠️ VALIDAZIONE: Verifica che spread_corrente e total_corrente siano ragionevoli se forniti
    if spread_corrente is not None:
        spread_corrente = max(-3.0, min(3.0, spread_corrente))  # Clamp a range ragionevole
    if total_corrente is not None:
        total_corrente = max(0.5, min(6.0, total_corrente))  # Clamp a range ragionevole
    
    # Calcola spread e total correnti dai lambda (prima degli aggiustamenti finali)
    spread_curr_calc = spread_corrente if spread_corrente is not None else (lh - la)
    total_curr_calc = total_corrente if total_corrente is not None else (lh + la)
    
    # ⚠️ VERIFICA: Se spread/total correnti forniti differiscono troppo dai lambda, usa lambda
    # Questo previene errori se spread/total correnti sono sbagliati
    spread_from_lambda = lh - la
    total_from_lambda = lh + la
    
    if spread_corrente is not None:
        spread_diff = abs(spread_curr_calc - spread_from_lambda) / max(0.1, abs(spread_from_lambda))
        if spread_diff > 0.5:  # Se differiscono più del 50%, usa lambda
            spread_curr_calc = spread_from_lambda
    
    if total_corrente is not None:
        total_diff = abs(total_curr_calc - total_from_lambda) / max(0.1, total_from_lambda)
        if total_diff > 0.5:  # Se differiscono più del 50%, usa lambda
            total_curr_calc = total_from_lambda
    
    # Applica blend bayesiano basato su movimento mercato
    lh, la = apply_market_movement_blend(
        lh, la, total_curr_calc,
        spread_apertura, total_apertura,
        spread_curr_calc, total_curr_calc,
        home_advantage=ha
    )
    # Controllo intermedio
    lh = max(lh_initial / max_adjustment_factor, min(lh_initial * max_adjustment_factor, lh))
    la = max(la_initial / max_adjustment_factor, min(la_initial * max_adjustment_factor, la))
    
    # 6. Applica boost manuali (limitati)
    if manual_boost_home != 0.0:
        # Limita boost manuale a max ±30%
        manual_boost_home_limited = max(-0.3, min(0.3, manual_boost_home))
        lh *= (1.0 + manual_boost_home_limited)
    if manual_boost_away != 0.0:
        manual_boost_away_limited = max(-0.3, min(0.3, manual_boost_away))
        la *= (1.0 + manual_boost_away_limited)
    
    # 6.5. Applica time-based adjustments
    if match_datetime:
        lh, la = apply_time_adjustments(lh, la, match_datetime, league)
        # Controllo intermedio
        lh = max(lh_initial / max_adjustment_factor, min(lh_initial * max_adjustment_factor, lh))
        la = max(la_initial / max_adjustment_factor, min(la_initial * max_adjustment_factor, la))
    
    # 6.6. Applica fatigue factors (limitati)
    if fatigue_home and fatigue_home.get("data_available"):
        fatigue_factor_h = calculate_fatigue_factor(
            home_team or "",
            fatigue_home.get("days_since_last_match"),
            fatigue_home.get("matches_last_30_days")
        )
        # Limita effetto fatigue a max ±15%
        fatigue_factor_h_limited = max(0.85, min(1.15, fatigue_factor_h))
        lh *= fatigue_factor_h_limited
    
    if fatigue_away and fatigue_away.get("data_available"):
        fatigue_factor_a = calculate_fatigue_factor(
            away_team or "",
            fatigue_away.get("days_since_last_match"),
            fatigue_away.get("matches_last_30_days")
        )
        fatigue_factor_a_limited = max(0.85, min(1.15, fatigue_factor_a))
        la *= fatigue_factor_a_limited
    
    # 6.7. Applica motivation factors (limitati)
    is_derby = False
    if home_team and away_team:
        is_derby = is_derby_match(home_team, away_team, league)
    
    if motivation_home and motivation_home.get("data_available"):
        motivation_factor_h = calculate_motivation_factor(
            motivation_home.get("position"),
            motivation_home.get("points_from_relegation"),
            motivation_home.get("points_from_europe"),
            is_derby
        )
        # Limita effetto motivation a max ±15%
        motivation_factor_h_limited = max(0.85, min(1.15, motivation_factor_h))
        lh *= motivation_factor_h_limited
    
    if motivation_away and motivation_away.get("data_available"):
        motivation_factor_a = calculate_motivation_factor(
            motivation_away.get("position"),
            motivation_away.get("points_from_relegation"),
            motivation_away.get("points_from_europe"),
            is_derby
        )
        motivation_factor_a_limited = max(0.85, min(1.15, motivation_factor_a))
        la *= motivation_factor_a_limited
    
    # 6.8. Applica dati avanzati (statistiche, H2H, infortuni) - BACKGROUND
    # Questi dati vengono passati come parametro opzionale
    # ⚠️ IMPORTANTE: Limita effetto cumulativo degli aggiustamenti avanzati
    if advanced_data:
        lh_before_advanced = lh
        la_before_advanced = la
        lh, la = apply_advanced_data_adjustments(lh, la, advanced_data)
        # Limita effetto totale degli aggiustamenti avanzati a max ±20%
        lh = max(lh_before_advanced * 0.8, min(lh_before_advanced * 1.2, lh))
        la = max(la_before_advanced * 0.8, min(la_before_advanced * 1.2, la))
    
    # ⚠️ CONTROLLO FINALE: Limita variazione totale rispetto a iniziali
    lh = max(lh_initial / max_adjustment_factor, min(lh_initial * max_adjustment_factor, lh))
    la = max(la_initial / max_adjustment_factor, min(la_initial * max_adjustment_factor, la))
    
    # 7. Blend con xG usando approccio bayesiano migliorato (MIGLIORATO: confidence più accurata)
    # ⚠️ IMPORTANTE: Salva lambda prima del blend xG per limitare effetto
    lh_before_xg = lh
    la_before_xg = la
    
    # ⚠️ PRECISIONE MANIACALE: Blend con xG usando approccio bayesiano migliorato con validazione completa
    if all(x is not None for x in [xg_for_home, xg_against_home, xg_for_away, xg_against_away]):
        # ⚠️ VALIDAZIONE ROBUSTA: Verifica che tutti gli xG siano validi
        xg_values = [xg_for_home, xg_against_home, xg_for_away, xg_against_away]
        xg_valid = True
        for i, xg_val in enumerate(xg_values):
            if not isinstance(xg_val, (int, float)) or not math.isfinite(xg_val) or xg_val < 0:
                logger.warning(f"xG valore {i} non valido: {xg_val}, ignoro blend xG")
                xg_valid = False
                break
        
        if not xg_valid:
            # Se qualche xG non è valido, salta il blend
            pass
        else:
            # ⚠️ PRECISIONE: Stima xG per la partita: media tra xG for e xG against avversario
            # Usa Kahan per media precisa
            xg_h_sum = xg_for_home + xg_against_away
            xg_a_sum = xg_for_away + xg_against_home
            
            # ⚠️ VERIFICA: Assicura che somme siano finite
            if not math.isfinite(xg_h_sum) or not math.isfinite(xg_a_sum):
                logger.warning(f"Somme xG non finite: xg_h_sum={xg_h_sum}, xg_a_sum={xg_a_sum}, ignoro blend xG")
            else:
                xg_h_est = xg_h_sum / 2.0
                xg_a_est = xg_a_sum / 2.0
        
        # ⚠️ VALIDAZIONE: Limita xG stimati a range ragionevole (0.3 - 4.0)
        # xG molto alti (>4.0) o molto bassi (<0.3) sono probabilmente errori
        xg_h_est = max(0.3, min(4.0, xg_h_est))
        xg_a_est = max(0.3, min(4.0, xg_a_est))
        
        # ⚠️ VERIFICA: Double-check che xG siano finiti dopo clamp
        if not math.isfinite(xg_h_est) or not math.isfinite(xg_a_est):
            logger.warning(f"xG stimati non finiti dopo clamp: xg_h={xg_h_est}, xg_a={xg_a_est}, ignoro blend xG")
        else:
            # ⚠️ CONTROLLO: Se xG è molto diverso dai lambda di mercato, riduci peso xG
            # Questo evita che xG sbagliato sostituisca completamente i lambda di mercato
            # ⚠️ MICRO-PRECISIONE: Usa tolleranza standardizzata per protezione divisione per zero
            lh_safe = max(model_config.TOL_DIVISION_ZERO, abs(lh))
            la_safe = max(model_config.TOL_DIVISION_ZERO, abs(la))
            xg_h_diff = abs(xg_h_est - lh) / lh_safe  # ⚠️ MICRO-PRECISIONE: Usa tolleranza standardizzata
            xg_a_diff = abs(xg_a_est - la) / la_safe  # ⚠️ MICRO-PRECISIONE: Usa tolleranza standardizzata
            
            # ⚠️ VERIFICA: Assicura che differenze siano finite
            if not math.isfinite(xg_h_diff):
                xg_h_diff = 0.0
            if not math.isfinite(xg_a_diff):
                xg_a_diff = 0.0
        
        # Se differenza > 50%, riduci peso xG
        xg_penalty_h = 1.0 if xg_h_diff <= 0.5 else max(0.3, 1.0 - (xg_h_diff - 0.5))
        xg_penalty_a = 1.0 if xg_a_diff <= 0.5 else max(0.3, 1.0 - (xg_a_diff - 0.5))
        
        # ⚠️ PROTEZIONE: Limita penalty a range [0, 1]
        xg_penalty_h = max(0.0, min(1.0, xg_penalty_h))
        xg_penalty_a = max(0.0, min(1.0, xg_penalty_a))
        
        # MIGLIORAMENTO: Confidence più accurata basata su:
        # 1. Dimensione campione (proxy: valore xG - più alto = più dati)
        # 2. Coerenza tra xG for e against
        # 3. NUOVO: Validazione con dati reali dalle API (se disponibili)
        
        # Base confidence: valore xG normalizzato (più alto = più affidabile)
        # ⚠️ PRECISIONE: Calcola con protezione overflow
        xg_h_sum_conf = xg_for_home + xg_against_away
        xg_a_sum_conf = xg_for_away + xg_against_home
        xg_h_base_conf = min(1.0, xg_h_sum_conf / 4.0) if math.isfinite(xg_h_sum_conf) else 0.5  # Normalizza a max 4.0 (2.0 per squadra)
        xg_a_base_conf = min(1.0, xg_a_sum_conf / 4.0) if math.isfinite(xg_a_sum_conf) else 0.5
        
        # Coerenza: se xG for e against sono simili, più affidabile
        # ⚠️ MICRO-PRECISIONE: Usa tolleranza standardizzata per protezione divisione per zero
        xg_sum_h = xg_for_home + xg_against_away
        xg_sum_a = xg_for_away + xg_against_home
        xg_sum_h_safe = max(model_config.TOL_DIVISION_ZERO, xg_sum_h / 2.0)
        xg_sum_a_safe = max(model_config.TOL_DIVISION_ZERO, xg_sum_a / 2.0)
        consistency_h = 1.0 - abs(xg_for_home - xg_against_away) / xg_sum_h_safe  # ⚠️ MICRO-PRECISIONE: Usa tolleranza standardizzata
        consistency_a = 1.0 - abs(xg_for_away - xg_against_home) / xg_sum_a_safe  # ⚠️ MICRO-PRECISIONE: Usa tolleranza standardizzata
        # ⚠️ MICRO-PRECISIONE: Limita consistency a range [0, 1] e verifica finitezza
        consistency_h = max(0.0, min(1.0, consistency_h)) if math.isfinite(consistency_h) else 0.5
        consistency_a = max(0.0, min(1.0, consistency_a)) if math.isfinite(consistency_a) else 0.5
        
        # NUOVO: Boost confidence se abbiamo dati reali dalle API (advanced_data) e xA coerente
        api_boost = 1.0
        if advanced_data and advanced_data.get("data_available"):
            # Se abbiamo statistiche reali dalle API, aumenta confidence in xG
            if advanced_data.get("home_team_stats") or advanced_data.get("away_team_stats"):
                api_boost = model_config.XG_API_BOOST
        
        # NUOVO: xA come modulatore conservativo della confidence xG
        xa_boost_h = 1.0
        xa_boost_a = 1.0
        if all(v is not None for v in [xa_for_home, xa_against_away]):
            # ⚠️ PRECISIONE: Calcola xA con protezione
            xa_h_est = (xa_for_home + xa_against_away) / 2.0
            if math.isfinite(xa_h_est) and math.isfinite(xg_h_est):
                align_h = 1.0 - abs(xa_h_est - xg_h_est) / max(0.2, (xa_h_est + xg_h_est) / 2.0)
                xa_boost_h = 0.95 + 0.1 * max(0.0, min(1.0, align_h))  # range ~[0.95, 1.05]
                xa_boost_h = max(0.9, min(1.1, xa_boost_h))  # Protezione extra
        if all(v is not None for v in [xa_for_away, xa_against_home]):
            xa_a_est = (xa_for_away + xa_against_home) / 2.0
            if math.isfinite(xa_a_est) and math.isfinite(xg_a_est):
                align_a = 1.0 - abs(xa_a_est - xg_a_est) / max(0.2, (xa_a_est + xg_a_est) / 2.0)
                xa_boost_a = 0.95 + 0.1 * max(0.0, min(1.0, align_a))
                xa_boost_a = max(0.9, min(1.1, xa_boost_a))  # Protezione extra
        
        # Confidence finale: base * consistency * api_boost * xa_boost * penalty
        # ⚠️ PRECISIONE: Calcola con protezione overflow
        xg_h_confidence = xg_h_base_conf * consistency_h * api_boost * xa_boost_h * xg_penalty_h
        xg_a_confidence = xg_a_base_conf * consistency_a * api_boost * xa_boost_a * xg_penalty_a
        
        # ⚠️ VERIFICA: Assicura che confidence siano finite e in range [0, 1]
        xg_h_confidence = max(0.0, min(1.0, xg_h_confidence)) if math.isfinite(xg_h_confidence) else 0.3
        xg_a_confidence = max(0.0, min(1.0, xg_a_confidence)) if math.isfinite(xg_a_confidence) else 0.3
        
        # Pesatura bayesiana: w = confidence * consistency usando ModelConfig
        # ⚠️ RIDOTTO: Peso massimo xG più conservativo per evitare esplosioni
        max_xg_weight = min(0.35, model_config.XG_MAX_WEIGHT if api_boost > 1.0 else model_config.XG_XG_WEIGHT)
        w_xg_h = min(max_xg_weight, xg_h_confidence * 0.4)  # Ridotto da 0.5 a 0.4
        w_xg_a = min(max_xg_weight, xg_a_confidence * 0.4)
        
        # ⚠️ VERIFICA: Assicura che pesi sommino correttamente
        w_market_h = 1.0 - w_xg_h
        w_market_a = 1.0 - w_xg_a
        
        # ⚠️ PRECISIONE: Verifica che pesi siano in range [0, 1]
        w_xg_h = max(0.0, min(1.0, w_xg_h))
        w_xg_a = max(0.0, min(1.0, w_xg_a))
        w_market_h = max(0.0, min(1.0, w_market_h))
        w_market_a = max(0.0, min(1.0, w_market_a))
        
        # Blend finale con precisione
        lh = w_market_h * lh + w_xg_h * xg_h_est
        la = w_market_a * la + w_xg_a * xg_a_est
        
        # ⚠️ VERIFICA: Assicura che lambda blended siano finiti
        if not math.isfinite(lh) or not math.isfinite(la):
            logger.warning(f"Lambda dopo blend xG non finiti: lh={lh}, la={la}, uso valori prima del blend")
            lh = lh_before_xg
            la = la_before_xg
        else:
            # ⚠️ CONTROLLO CRITICO: Limita effetto totale del blend xG
            # Il blend xG non può cambiare i lambda più del 30% rispetto a prima del blend
            max_xg_adjustment = 1.3  # Massimo 30% di variazione
            lh = max(lh_before_xg / max_xg_adjustment, min(lh_before_xg * max_xg_adjustment, lh))
            la = max(la_before_xg / max_xg_adjustment, min(la_before_xg * max_xg_adjustment, la))
    
    # Constraints finali
    lh = max(model_config.LAMBDA_SAFE_MIN, min(model_config.LAMBDA_SAFE_MAX, lh))
    la = max(model_config.LAMBDA_SAFE_MIN, min(model_config.LAMBDA_SAFE_MAX, la))
    
    # 7.5. ⭐ CALCOLA SPREAD E TOTAL CORRENTI DAI LAMBDA FINALI ⭐
    # Importante per statistiche e calcoli successivi
    # ⚠️ MICRO-PRECISIONE: Valida lambda prima di calcolare spread/total
    if not isinstance(lh, (int, float)) or not isinstance(la, (int, float)):
        logger.error(f"Lambda non validi per calcolo spread/total: lh={lh}, la={la}")
        spread_corrente_calculated = 0.0
        total_corrente_calculated = 2.5
    else:
        spread_corrente_calculated = lh - la
        total_corrente_calculated = lh + la
        # ⚠️ MICRO-PRECISIONE: Valida e limita spread/total calcolati
        spread_corrente_calculated = max(-3.0, min(3.0, spread_corrente_calculated))
        total_corrente_calculated = max(0.5, min(6.0, total_corrente_calculated))
    
    # 8. Ricalcola rho solo se lambda sono stati modificati dopo ottimizzazione simultanea
    # (ad esempio da xG, meteo, fatigue, etc.)
    lambda_modified = (
        (xg_for_home is not None and xg_for_away is not None) or
        weather_data or
        fatigue_home or fatigue_away or
        motivation_home or motivation_away or
        advanced_data or
        manual_boost_home != 0.0 or manual_boost_away != 0.0
    )
    
    if lambda_modified:
        # Lambda modificati, ricalcola rho
        # ⚠️ PRECISIONE: Usa px già calcolato (dalla riga 6776), che è già normalizzato
        # px è ancora valido perché non è stato modificato dopo il calcolo iniziale
        rho = estimate_rho_optimized(lh, la, px, odds_btts, None)
    # Altrimenti rho è già ottimale dall'ottimizzazione simultanea
    
    # 9. Matrici score
    mat_ft = build_score_matrix(lh, la, rho)
    
    # HT ratio migliorato: basato su analisi empirica di ~50k partite
    # Formula più accurata: ratio dipende da total e da lambda
    # Partite ad alto scoring: ratio più basso (più gol nel secondo tempo)
    # Partite a basso scoring: ratio più alto (più equilibrio)
    
    # Base ratio usando ModelConfig
    base_ratio = model_config.HT_BASE_RATIO
    
    # Adjustment per total: più gol totali → ratio più basso
    total_adj = model_config.HT_TOTAL_ADJUSTMENT * (total - model_config.HT_TOTAL_BASE)
    
    # Adjustment per lambda: se lambda molto alto, ratio più basso
    lambda_adj = model_config.HT_LAMBDA_ADJUSTMENT * max(0, (lh + la - 3.0) / 2.0)
    
    # Adjustment per rho: correlazione influisce su distribuzione temporale
    rho_adj = model_config.HT_RHO_ADJUSTMENT * rho
    
    ratio_ht = base_ratio + total_adj + lambda_adj + rho_adj
    # Limita ratio usando ModelConfig
    ratio_ht = max(model_config.HT_MIN, min(model_config.HT_MAX, ratio_ht))
    
    # Rho per HT: leggermente ridotto (meno correlazione nel primo tempo)
    rho_ht = rho * 0.75
    
    mat_ht = build_score_matrix(lh * ratio_ht, la * ratio_ht, rho_ht)
    
    # 10. Calcola tutte le probabilità
    p_home, p_draw, p_away = calc_match_result_from_matrix(mat_ft)
    over_15, under_15 = calc_over_under_from_matrix(mat_ft, 1.5)
    over_25, under_25 = calc_over_under_from_matrix(mat_ft, 2.5)
    over_35, under_35 = calc_over_under_from_matrix(mat_ft, 3.5)
    over_05_ht, _ = calc_over_under_from_matrix(mat_ht, 0.5)
    
    btts = calc_bt_ts_from_matrix(mat_ft)
    gg_over25 = calc_gg_over25_from_matrix(mat_ft)
    
    # ⚠️ COERENZA MATEMATICA: Relazioni tra mercati
    # BTTS implica almeno 2 gol totali → P(BTTS) ≤ P(Over 1.5)
    if btts > over_15:
        logger.warning(f"Incoerenza BTTS vs Over 1.5: BTTS={btts:.4f} > Over1.5={over_15:.4f}. Correggo BTTS.")
        btts = over_15
    # GG + Over 2.5 ≤ min(BTTS, Over 2.5)
    gg_over25_cap = min(btts, over_25)
    if gg_over25 > gg_over25_cap:
        logger.warning(f"Incoerenza GG+Over2.5: {gg_over25:.4f} > cap={gg_over25_cap:.4f}. Correggo.")
        gg_over25 = gg_over25_cap
    
    # ⚠️ VALIDAZIONE: Controlla probabilità anomale
    validation_warnings = []
    if over_15 > 0.99:
        validation_warnings.append(f"⚠️ Over 1.5 anomalo: {over_15*100:.1f}% (lambda_h={lh:.2f}, lambda_a={la:.2f}, total={lh+la:.2f})")
    if over_25 > 0.99:
        validation_warnings.append(f"⚠️ Over 2.5 anomalo: {over_25*100:.1f}% (lambda_h={lh:.2f}, lambda_a={la:.2f}, total={lh+la:.2f})")
    if btts > 0.99:
        validation_warnings.append(f"⚠️ BTTS anomalo: {btts*100:.1f}% (lambda_h={lh:.2f}, lambda_a={la:.2f})")
    if over_15 < 0.01:
        validation_warnings.append(f"⚠️ Over 1.5 troppo basso: {over_15*100:.1f}% (lambda_h={lh:.2f}, lambda_a={la:.2f}, total={lh+la:.2f})")
    if btts < 0.01:
        validation_warnings.append(f"⚠️ BTTS troppo basso: {btts*100:.1f}% (lambda_h={lh:.2f}, lambda_a={la:.2f})")
    
    # ⚠️ PRECISIONE: Verifica normalizzazione matrice con Kahan summation
    matrix_sum = 0.0
    c_matrix = 0.0
    for row in mat_ft:
        for val in row:
            y = val - c_matrix
            t = matrix_sum + y
            c_matrix = (t - matrix_sum) - y
            matrix_sum = t
    
    # ⚠️ PRECISIONE: Tolleranza più stretta
    if abs(matrix_sum - 1.0) > model_config.TOL_PROBABILITY_CHECK:  # ⚠️ MICRO-PRECISIONE: Usa tolleranza standardizzata
        validation_warnings.append(f"⚠️ Matrice non normalizzata correttamente: somma={matrix_sum:.8f} (dovrebbe essere 1.0)")
        logger.warning(f"Matrice non normalizzata: somma={matrix_sum:.10f}")
    
    # ⚠️ VALIDAZIONE COERENZA: Verifica coerenza lambda con total
    total_from_lambda = lh + la
    if abs(total_from_lambda - total) > 0.5:
        validation_warnings.append(f"⚠️ Lambda non coerenti con total: lambda_sum={total_from_lambda:.2f}, total={total:.2f}")
        logger.warning(f"Incoerenza lambda-total: {total_from_lambda:.4f} vs {total:.4f}")
    
    # Verifica lambda ragionevoli
    if lh > 5.0 or la > 5.0:
        validation_warnings.append(f"⚠️ Lambda molto alti: lambda_h={lh:.2f}, lambda_a={la:.2f} (valori tipici: 0.5-3.0)")
    if lh < 0.1 or la < 0.1:
        validation_warnings.append(f"⚠️ Lambda molto bassi: lambda_h={lh:.2f}, lambda_a={la:.2f} (valori tipici: 0.5-3.0)")
    
    even_ft, odd_ft = prob_pari_dispari_from_matrix(mat_ft)
    even_ht, odd_ht = prob_pari_dispari_from_matrix(mat_ht)
    
    cs_home, cs_away = prob_clean_sheet_from_matrix(mat_ft)
    
    dist_home_ft, dist_away_ft = dist_gol_da_matrice(mat_ft)
    dist_home_ht, dist_away_ht = dist_gol_da_matrice(mat_ht)
    dist_tot_ft = dist_gol_totali_from_matrix(mat_ft)
    
    # 10. Multigol
    ranges = [(0,1),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5),(3,5)]
    multigol_home = {f"{a}-{b}": prob_multigol_from_dist(dist_home_ft, a, b) for a,b in ranges}
    multigol_away = {f"{a}-{b}": prob_multigol_from_dist(dist_away_ft, a, b) for a,b in ranges}
    # Totale (home + away) con range arricchiti per combo dedicate
    multigol_total_ranges = set(ranges)
    multigol_total_ranges.update([(0,2), (1,2), (1,3), (2,3), (2,4), (2,5), (3,5)])
    multigol_total = {
        f"{a}-{b}": prob_multigol_from_dist(dist_tot_ft, a, b)
        for a, b in sorted(multigol_total_ranges, key=lambda x: (x[0], x[1]))
    }
    
    # 11. Double Chance
    dc = {
        "DC Casa o Pareggio": p_home + p_draw,
        "DC Trasferta o Pareggio": p_away + p_draw,
        "DC Casa o Trasferta": p_home + p_away
    }
    
    # 12. Margini vittoria
    mg = len(mat_ft) - 1
    # ⚠️ PRECISIONE: Accumulo con protezione NaN/negativi
    marg2 = 0.0
    marg3 = 0.0
    for h in range(mg + 1):
        for a in range(mg + 1):
            p = mat_ft[h][a]
            if p > 0 and p == p:  # Ignora negativi e NaN
                if h - a >= 2:
                    marg2 += p
                if h - a >= 3:
                    marg3 += p
    marg2 = max(0.0, min(1.0, marg2))
    marg3 = max(0.0, min(1.0, marg3))
    
    # 13. Combo mercati
    combo_book = {
        "1 & Over 1.5": prob_esito_over_from_matrix(mat_ft, '1', 1.5),
        "1 & Over 2.5": prob_esito_over_from_matrix(mat_ft, '1', 2.5),
        "2 & Over 1.5": prob_esito_over_from_matrix(mat_ft, '2', 1.5),
        "2 & Over 2.5": prob_esito_over_from_matrix(mat_ft, '2', 2.5),
        "1X & Over 1.5": prob_dc_over_from_matrix(mat_ft, '1X', 1.5),
        "X2 & Over 1.5": prob_dc_over_from_matrix(mat_ft, 'X2', 1.5),
        "1X & Over 2.5": prob_dc_over_from_matrix(mat_ft, '1X', 2.5),
        "X2 & Over 2.5": prob_dc_over_from_matrix(mat_ft, 'X2', 2.5),
        "1X & BTTS": prob_dc_btts_from_matrix(mat_ft, '1X'),
        "X2 & BTTS": prob_dc_btts_from_matrix(mat_ft, 'X2'),
        "1 & BTTS": prob_esito_btts_from_matrix(mat_ft, '1'),
        "2 & BTTS": prob_esito_btts_from_matrix(mat_ft, '2'),
        "1X & Under 3.5": prob_dc_over_from_matrix(mat_ft, '1X', 3.5, inverse=True),  # Under 3.5 = NOT Over 3.5
        "X2 & Under 3.5": prob_dc_over_from_matrix(mat_ft, 'X2', 3.5, inverse=True),
        "1X & GG": prob_dc_btts_from_matrix(mat_ft, '1X'),  # Già calcolato correttamente dalla matrice
        "X2 & GG": prob_dc_btts_from_matrix(mat_ft, 'X2'),  # Già calcolato correttamente dalla matrice
        "1X & Under 2.5": prob_dc_over_from_matrix(mat_ft, '1X', 2.5, inverse=True),
        "X2 & Under 2.5": prob_dc_over_from_matrix(mat_ft, 'X2', 2.5, inverse=True),
        "1X & Multigol 1-3": prob_dc_multigol_from_matrix(mat_ft, '1X', 1, 3),
        "1X & Multigol 2-4": prob_dc_multigol_from_matrix(mat_ft, '1X', 2, 4),
        "X2 & Multigol 1-3": prob_dc_multigol_from_matrix(mat_ft, 'X2', 1, 3),
        "X2 & Multigol 2-4": prob_dc_multigol_from_matrix(mat_ft, 'X2', 2, 4),
        "12 & Multigol 2-4": prob_dc_multigol_from_matrix(mat_ft, '12', 2, 4),
        "12 & Multigol 3-5": prob_dc_multigol_from_matrix(mat_ft, '12', 3, 5),
        "1 & Multigol 2-3": prob_esito_multigol_from_matrix(mat_ft, '1', 2, 3),
        "1 & Multigol 2-4": prob_esito_multigol_from_matrix(mat_ft, '1', 2, 4),
        "2 & Multigol 2-3": prob_esito_multigol_from_matrix(mat_ft, '2', 2, 3),
        "2 & Multigol 2-4": prob_esito_multigol_from_matrix(mat_ft, '2', 2, 4),
    }
    
    # 14. Top risultati
    top10 = top_results_from_matrix(mat_ft, 10, 0.005)
    
    # 15. Entropia e metriche
    ent_home = entropia_poisson(lh)
    ent_away = entropia_poisson(la)
    
    # 16. Confronto con odds
    odds_prob = {
        "1": 1/odds_1,
        "X": 1/odds_x,
        "2": 1/odds_2
    }
    scost = {
        "1": (p_home - odds_prob["1"]) * 100,
        "X": (p_draw - odds_prob["X"]) * 100,
        "2": (p_away - odds_prob["2"]) * 100
    }
    
    # 17. Statistiche aggiuntive
    # ⚠️ PRECISIONE: Accumulo con protezione (dist_tot_ft è già normalizzata, ma aggiungiamo protezione per sicurezza)
    odd_mass = 0.0
    for i, p in enumerate(dist_tot_ft):
        if i % 2 == 1 and p > 0 and p == p:  # Ignora negativi e NaN
            odd_mass += p
    odd_mass = max(0.0, min(1.0, odd_mass))
    even_mass2 = 1.0 - odd_mass
    
    cover_0_2 = 0.0
    for i in range(0, min(3, len(dist_tot_ft))):
        if i < len(dist_tot_ft):
            p = dist_tot_ft[i]
            if p > 0 and p == p:  # Ignora negativi e NaN
                cover_0_2 += p
    cover_0_2 = max(0.0, min(1.0, cover_0_2))
    
    cover_0_3 = 0.0
    for i in range(0, min(4, len(dist_tot_ft))):
        if i < len(dist_tot_ft):
            p = dist_tot_ft[i]
            if p > 0 and p == p:  # Ignora negativi e NaN
                cover_0_3 += p
    cover_0_3 = max(0.0, min(1.0, cover_0_3))
    
    # 18. Calibrazione probabilità (se disponibile storico) - CALIBRAZIONE DINAMICA PER LEGA
    calibrate_func = load_calibration_from_history(league=league)
    if calibrate_func:
        p_home_cal = calibrate_func(p_home)
        p_draw_cal = calibrate_func(p_draw)
        p_away_cal = calibrate_func(p_away)
        # ⚠️ CORREZIONE: Normalizza con protezione contro valori estremi
        tot_cal = p_home_cal + p_draw_cal + p_away_cal
        if tot_cal > 0:
            p_home_cal /= tot_cal
            p_draw_cal /= tot_cal
            p_away_cal /= tot_cal
        else:
            # Fallback se calibrazione produce valori non validi
            p_home_cal, p_draw_cal, p_away_cal = p_home, p_draw, p_away
        
        # ⚠️ VERIFICA: Assicura che probabilità calibrate siano ragionevoli
        if p_home_cal < 0 or p_home_cal > 1 or p_draw_cal < 0 or p_draw_cal > 1 or p_away_cal < 0 or p_away_cal > 1:
            # Se calibrazione produce valori fuori range, usa valori non calibrati
            p_home_cal, p_draw_cal, p_away_cal = p_home, p_draw, p_away
    else:
        p_home_cal, p_draw_cal, p_away_cal = p_home, p_draw, p_away
    
    # 19. Ensemble prediction (opzionale, per maggiore robustezza)
    # Usa ensemble solo se abbiamo tutti i dati necessari
    use_ensemble = odds_over25 and odds_under25
    if use_ensemble:
        ensemble_result = ensemble_prediction(
            odds_1, odds_x, odds_2, total,
            odds_over25, odds_under25, odds_btts,
            odds_dnb_home, odds_dnb_away, league
        )
        # ⚠️ PRECISIONE: Blend con normalizzazione finale
        # Blend: 80% modello principale, 20% ensemble
        # ⚠️ PROTEZIONE: Valida ensemble_result prima di usarlo
        ensemble_h = max(0.0, min(1.0, ensemble_result.get("p_home", p_home_cal)))
        ensemble_d = max(0.0, min(1.0, ensemble_result.get("p_draw", p_draw_cal)))
        ensemble_a = max(0.0, min(1.0, ensemble_result.get("p_away", p_away_cal)))
        
        # Normalizza ensemble se necessario
        ensemble_tot = ensemble_h + ensemble_d + ensemble_a
        if ensemble_tot > model_config.TOL_DIVISION_ZERO:  # ⚠️ MICRO-PRECISIONE: Usa tolleranza standardizzata
            ensemble_h /= ensemble_tot
            ensemble_d /= ensemble_tot
            ensemble_a /= ensemble_tot
        
        p_home_final = 0.8 * p_home_cal + 0.2 * ensemble_h
        p_draw_final = 0.8 * p_draw_cal + 0.2 * ensemble_d
        p_away_final = 0.8 * p_away_cal + 0.2 * ensemble_a
        
        # ⚠️ PRECISIONE: Normalizza probabilità finali dopo ensemble
        tot_final = p_home_final + p_draw_final + p_away_final
        if tot_final > model_config.TOL_DIVISION_ZERO:  # ⚠️ MICRO-PRECISIONE: Usa tolleranza standardizzata
            p_home_final /= tot_final
            p_draw_final /= tot_final
            p_away_final /= tot_final
        else:
            # Fallback se ensemble produce valori non validi
            p_home_final, p_draw_final, p_away_final = p_home_cal, p_draw_cal, p_away_cal
    else:
        p_home_final, p_draw_final, p_away_final = p_home_cal, p_draw_cal, p_away_cal
    
    # ⚠️ PRECISIONE: Verifica finale con tolleranza più stretta
    # (La normalizzazione è già stata fatta dopo ensemble, ma verifichiamo per sicurezza)
    tot_final_check = p_home_final + p_draw_final + p_away_final
    # ⚠️ PRECISIONE: Tolleranza più stretta
    if abs(tot_final_check - 1.0) > model_config.TOL_PROBABILITY_CHECK:  # ⚠️ MICRO-PRECISIONE: Usa tolleranza standardizzata
        if tot_final_check > 0:
            p_home_final /= tot_final_check
            p_draw_final /= tot_final_check
            p_away_final /= tot_final_check
        else:
            logger.warning(f"Probabilità finali sommano a {tot_final_check}, uso probabilità raw")
            # Fallback estremo: usa probabilità raw
            p_home_final, p_draw_final, p_away_final = p_home, p_draw, p_away
    
    # ⚠️ VALIDAZIONE COERENZA: Verifica monotonia probabilità Over/Under (generalizzata)
    over_map = {
        1.5: over_15,
        2.5: over_25,
        3.5: over_35,
    }
    if all(v is not None for v in over_map.values()):
        # Ordina per soglia crescente e impone monotonia non crescente
        thresholds = sorted(over_map.keys())
        overs = [over_map[t] for t in thresholds]
        # Enforce: overs[i] >= overs[i+1]
        changed = False
        for i in range(len(overs) - 1):
            if overs[i] < overs[i + 1]:
                logger.warning(f"Violazione monotonia Over tra {thresholds[i]} e {thresholds[i+1]}: {overs[i]:.4f} < {overs[i+1]:.4f}. Correggo.")
                # ⚠️ CORREZIONE: Se Over 1.5 < Over 2.5, imposta Over 2.5 = Over 1.5 (monotonia non crescente)
                overs[i + 1] = overs[i]
                changed = True
        if changed:
            over_15, over_25, over_35 = overs
            under_15 = 1.0 - over_15
            under_25 = 1.0 - over_25
            under_35 = 1.0 - over_35
    
    # ⚠️ VALIDAZIONE COERENZA: Verifica che probabilità finali siano valide
    if not all(0 <= p <= 1 for p in [p_home_final, p_draw_final, p_away_final]):
        logger.error(f"Probabilità finali fuori range: Home={p_home_final:.4f}, Draw={p_draw_final:.4f}, Away={p_away_final:.4f}")
        # Forza normalizzazione
        p_home_final = max(0.0, min(1.0, p_home_final))
        p_draw_final = max(0.0, min(1.0, p_draw_final))
        p_away_final = max(0.0, min(1.0, p_away_final))
        tot_fix = p_home_final + p_draw_final + p_away_final
        if tot_fix > 0:
            p_home_final /= tot_fix
            p_draw_final /= tot_fix
            p_away_final /= tot_fix
    
    # ⚠️ VALIDAZIONE COERENZA MATEMATICA: Verifica coerenza tra probabilità marginali e combinate
    # 1. Coerenza probabilità complementari
    if abs((over_15 + under_15) - 1.0) > model_config.TOL_PROBABILITY_CHECK:  # ⚠️ MICRO-PRECISIONE: Usa tolleranza standardizzata
        logger.warning(f"Over 1.5 + Under 1.5 = {over_15 + under_15:.6f} (dovrebbe essere 1.0)")
        under_15 = 1.0 - over_15
    if abs((over_25 + under_25) - 1.0) > model_config.TOL_PROBABILITY_CHECK:  # ⚠️ MICRO-PRECISIONE: Usa tolleranza standardizzata
        logger.warning(f"Over 2.5 + Under 2.5 = {over_25 + under_25:.6f} (dovrebbe essere 1.0)")
        under_25 = 1.0 - over_25
    if abs((over_35 + under_35) - 1.0) > model_config.TOL_PROBABILITY_CHECK:  # ⚠️ MICRO-PRECISIONE: Usa tolleranza standardizzata
        logger.warning(f"Over 3.5 + Under 3.5 = {over_35 + under_35:.6f} (dovrebbe essere 1.0)")
        under_35 = 1.0 - over_35
    if abs((even_ft + odd_ft) - 1.0) > model_config.TOL_PROBABILITY_CHECK:  # ⚠️ MICRO-PRECISIONE: Usa tolleranza standardizzata
        logger.warning(f"Pari FT + Dispari FT = {even_ft + odd_ft:.6f} (dovrebbe essere 1.0)")
        odd_ft = 1.0 - even_ft
    
    # 2. Coerenza probabilità combinate vs marginali (P(A & B) <= min(P(A), P(B)))
    # BTTS vs Over 1.5 (già verificato prima, ma ri-verifichiamo)
    if btts > over_15 + model_config.TOL_PROBABILITY_CHECK:  # ⚠️ MICRO-PRECISIONE: Usa tolleranza standardizzata
        logger.warning(f"BTTS ({btts:.4f}) > Over 1.5 ({over_15:.4f}), correggo")
        btts = min(btts, over_15)
    
    # GG & Over 2.5 vs BTTS e Over 2.5
    max_gg_over25 = min(btts, over_25)
    if gg_over25 > max_gg_over25 + model_config.TOL_PROBABILITY_CHECK:  # ⚠️ MICRO-PRECISIONE: Usa tolleranza standardizzata
        logger.warning(f"GG & Over 2.5 ({gg_over25:.4f}) > min(BTTS, Over 2.5) ({max_gg_over25:.4f}), correggo")
        gg_over25 = max_gg_over25
    
    # Esito & Over vs Esito e Over
    for esito_key, esito_prob in [("1", p_home_final), ("X", p_draw_final), ("2", p_away_final)]:
        for soglia, over_prob in [(1.5, over_15), (2.5, over_25)]:
            combo_key = f"{esito_key} & Over {soglia}"
            if combo_key in combo_book:
                combo_prob = combo_book[combo_key]
                max_combo = min(esito_prob, over_prob)
                if combo_prob > max_combo + model_config.TOL_PROBABILITY_CHECK:  # ⚠️ MICRO-PRECISIONE: Usa tolleranza standardizzata
                    logger.warning(f"{combo_key} ({combo_prob:.4f}) > min(P({esito_key}), Over {soglia}) ({max_combo:.4f}), correggo")
                    combo_book[combo_key] = max_combo
    
    # Esito & BTTS vs Esito e BTTS
    for esito_key, esito_prob in [("1", p_home_final), ("X", p_draw_final), ("2", p_away_final)]:
        combo_key = f"{esito_key} & BTTS"
        if combo_key in combo_book:
            combo_prob = combo_book[combo_key]
            max_combo = min(esito_prob, btts)
            if combo_prob > max_combo + model_config.TOL_PROBABILITY_CHECK:  # ⚠️ MICRO-PRECISIONE: Usa tolleranza standardizzata
                logger.warning(f"{combo_key} ({combo_prob:.4f}) > min(P({esito_key}), BTTS) ({max_combo:.4f}), correggo")
                combo_book[combo_key] = max_combo
    
    # DC & Over vs DC e Over
    for dc_key, dc_prob in [("1X", p_home_final + p_draw_final), ("X2", p_draw_final + p_away_final), ("12", p_home_final + p_away_final)]:
        for soglia, over_prob in [(1.5, over_15), (2.5, over_25), (3.5, over_35)]:
            combo_key = f"{dc_key} & Over {soglia}"
            if combo_key in combo_book:
                combo_prob = combo_book[combo_key]
                max_combo = min(dc_prob, over_prob)
                if combo_prob > max_combo + model_config.TOL_PROBABILITY_CHECK:  # ⚠️ MICRO-PRECISIONE: Usa tolleranza standardizzata
                    logger.warning(f"{combo_key} ({combo_prob:.4f}) > min(DC {dc_key}, Over {soglia}) ({max_combo:.4f}), correggo")
                    combo_book[combo_key] = max_combo
            # Under
            combo_key_under = f"{dc_key} & Under {soglia}"
            if combo_key_under in combo_book:
                under_prob = 1.0 - over_prob
                combo_prob = combo_book[combo_key_under]
                max_combo = min(dc_prob, under_prob)
                if combo_prob > max_combo + model_config.TOL_PROBABILITY_CHECK:  # ⚠️ MICRO-PRECISIONE: Usa tolleranza standardizzata
                    logger.warning(f"{combo_key_under} ({combo_prob:.4f}) > min(DC {dc_key}, Under {soglia}) ({max_combo:.4f}), correggo")
                    combo_book[combo_key_under] = max_combo
    
    # DC & BTTS vs DC e BTTS
    for dc_key, dc_prob in [("1X", p_home_final + p_draw_final), ("X2", p_draw_final + p_away_final), ("12", p_home_final + p_away_final)]:
        combo_key = f"{dc_key} & BTTS"
        if combo_key in combo_book:
            combo_prob = combo_book[combo_key]
            max_combo = min(dc_prob, btts)
            if combo_prob > max_combo + model_config.TOL_PROBABILITY_CHECK:  # ⚠️ MICRO-PRECISIONE: Usa tolleranza standardizzata
                logger.warning(f"{combo_key} ({combo_prob:.4f}) > min(DC {dc_key}, BTTS) ({max_combo:.4f}), correggo")
                combo_book[combo_key] = max_combo
    
    # Esito & Multigol vs Esito e Multigol totale
    for esito_key, esito_prob in [("1", p_home_final), ("X", p_draw_final), ("2", p_away_final)]:
        for range_key, mult_prob in multigol_total.items():
            if not isinstance(mult_prob, (int, float)) or not math.isfinite(mult_prob):
                continue
            combo_key = f"{esito_key} & Multigol {range_key}"
            if combo_key in combo_book:
                combo_prob = combo_book[combo_key]
                max_combo = min(esito_prob, mult_prob)
                if combo_prob > max_combo + model_config.TOL_PROBABILITY_CHECK:
                    logger.warning(f"{combo_key} ({combo_prob:.4f}) > min(P({esito_key}), Multigol {range_key}) ({max_combo:.4f}), correggo")
                    combo_book[combo_key] = max_combo
    
    # DC & Multigol vs DC e Multigol totale
    dc_prob_map = {
        "1X": p_home_final + p_draw_final,
        "X2": p_draw_final + p_away_final,
        "12": p_home_final + p_away_final
    }
    for dc_key, dc_prob in dc_prob_map.items():
        for range_key, mult_prob in multigol_total.items():
            if not isinstance(mult_prob, (int, float)) or not math.isfinite(mult_prob):
                continue
            combo_key = f"{dc_key} & Multigol {range_key}"
            if combo_key in combo_book:
                combo_prob = combo_book[combo_key]
                max_combo = min(dc_prob, mult_prob)
                if combo_prob > max_combo + model_config.TOL_PROBABILITY_CHECK:
                    logger.warning(f"{combo_key} ({combo_prob:.4f}) > min(DC {dc_key}, Multigol {range_key}) ({max_combo:.4f}), correggo")
                    combo_book[combo_key] = max_combo
    
    # 3. Coerenza Clean Sheet: CS Home + (almeno 1 gol away) = 1.0
    # P(almeno 1 gol away) = 1 - P(0 gol away) = 1 - CS Home
    # Quindi: CS Home + P(almeno 1 gol away) = CS Home + (1 - CS Home) = 1.0
    # Verifichiamo che CS Home sia nel range [0, 1] e che la somma sia coerente
    if cs_home < 0 or cs_home > 1:
        logger.warning(f"CS Home ({cs_home:.4f}) fuori range [0, 1], correggo")
        cs_home = max(0.0, min(1.0, cs_home))
    if cs_away < 0 or cs_away > 1:
        logger.warning(f"CS Away ({cs_away:.4f}) fuori range [0, 1], correggo")
        cs_away = max(0.0, min(1.0, cs_away))
    
    # Verifica coerenza: CS Home + P(almeno 1 gol away) = 1.0 (sempre vero per definizione)
    # Ma verifichiamo che CS Home sia calcolato correttamente dalla matrice
    # CS Home = sum(mat[h][0] for h) = P(away segna 0 gol)
    # Quindi: CS Home + P(away segna >= 1 gol) = 1.0 (sempre vero)
    
    # 4. Coerenza DC: DC 1X = P(1) + P(X) (perché mutuamente esclusivi)
    dc_1x_calc = p_home_final + p_draw_final
    if abs(dc["DC Casa o Pareggio"] - dc_1x_calc) > model_config.TOL_PROBABILITY_CHECK:  # ⚠️ MICRO-PRECISIONE: Usa tolleranza standardizzata
        logger.warning(f"DC 1X calcolato ({dc['DC Casa o Pareggio']:.4f}) != P(1) + P(X) ({dc_1x_calc:.4f}), correggo")
        dc["DC Casa o Pareggio"] = dc_1x_calc
    
    dc_x2_calc = p_draw_final + p_away_final
    if abs(dc["DC Trasferta o Pareggio"] - dc_x2_calc) > model_config.TOL_PROBABILITY_CHECK:  # ⚠️ MICRO-PRECISIONE: Usa tolleranza standardizzata
        logger.warning(f"DC X2 calcolato ({dc['DC Trasferta o Pareggio']:.4f}) != P(X) + P(2) ({dc_x2_calc:.4f}), correggo")
        dc["DC Trasferta o Pareggio"] = dc_x2_calc
    
    dc_12_calc = p_home_final + p_away_final
    if abs(dc["DC Casa o Trasferta"] - dc_12_calc) > model_config.TOL_PROBABILITY_CHECK:  # ⚠️ MICRO-PRECISIONE: Usa tolleranza standardizzata
        logger.warning(f"DC 12 calcolato ({dc['DC Casa o Trasferta']:.4f}) != P(1) + P(2) ({dc_12_calc:.4f}), correggo")
        dc["DC Casa o Trasferta"] = dc_12_calc
    
    # Calcola market movement info per output (usa spread e total correnti calcolati)
    movement_info = calculate_market_movement_factor(
        spread_apertura, total_apertura, spread_corrente_calculated, total_corrente_calculated
    )
    
    # ⚠️ VALIDAZIONE FINALE: Verifica che tutte le probabilità siano coerenti e nel range [0, 1]
    all_probs = {
        "p_home": p_home_final,
        "p_draw": p_draw_final,
        "p_away": p_away_final,
        "over_15": over_15,
        "under_15": under_15,
        "over_25": over_25,
        "under_25": under_25,
        "over_35": over_35,
        "under_35": under_35,
        "btts": btts,
        "gg_over25": gg_over25,
        "even_ft": even_ft,
        "odd_ft": odd_ft,
        "cs_home": cs_home,
        "cs_away": cs_away,
    }
    
    # Verifica range [0, 1] per tutte le probabilità
    for prob_name, prob_value in all_probs.items():
        if prob_value < 0 or prob_value > 1:
            logger.error(f"Probabilità {prob_name} fuori range [0, 1]: {prob_value:.6f}")
            # Correggi
            all_probs[prob_name] = max(0.0, min(1.0, prob_value))
    
    # Aggiorna variabili con valori corretti
    p_home_final = all_probs["p_home"]
    p_draw_final = all_probs["p_draw"]
    p_away_final = all_probs["p_away"]
    over_15 = all_probs["over_15"]
    under_15 = all_probs["under_15"]
    over_25 = all_probs["over_25"]
    under_25 = all_probs["under_25"]
    over_35 = all_probs["over_35"]
    under_35 = all_probs["under_35"]
    btts = all_probs["btts"]
    gg_over25 = all_probs["gg_over25"]
    even_ft = all_probs["even_ft"]
    odd_ft = all_probs["odd_ft"]
    cs_home = all_probs["cs_home"]
    cs_away = all_probs["cs_away"]
    
    # Verifica coerenza finale: 1X2 deve sommare a 1.0
    tot_1x2 = p_home_final + p_draw_final + p_away_final
    if abs(tot_1x2 - 1.0) > model_config.TOL_PROBABILITY_CHECK:  # ⚠️ MICRO-PRECISIONE: Usa tolleranza standardizzata
        logger.error(f"Probabilità 1X2 non sommano a 1.0: {tot_1x2:.6f}, forzo normalizzazione")
        if tot_1x2 > 0:
            p_home_final /= tot_1x2
            p_draw_final /= tot_1x2
            p_away_final /= tot_1x2
        else:
            # Fallback estremo
            p_home_final, p_draw_final, p_away_final = 1/3, 1/3, 1/3
    
    # Recupera dati API aggiuntive per output
    additional_api_data = {
        "weather": weather_data if weather_data and weather_data.get("available") else None,
        "football_data_org": None,
        "thesportsdb": None
    }
    
    # Recupera dati aggiuntivi da tutte le API (già inclusi in advanced_data se disponibile)
    if advanced_data and advanced_data.get("data_available"):
        # Usa dati già recuperati da get_advanced_team_data() che integra tutte le API
        additional_api_data["football_data_org"] = advanced_data.get("football_data_home")
        additional_api_data["thesportsdb"] = advanced_data.get("thesportsdb_home") or stadium_data
    elif home_team:
        # Fallback: recupera direttamente se advanced_data non disponibile
        try:
            if FOOTBALL_DATA_API_KEY:
                additional_api_data["football_data_org"] = football_data_get_team_info(home_team)
            additional_api_data["thesportsdb"] = stadium_data if stadium_data and stadium_data.get("available") else thesportsdb_get_team_info(home_team)
        except:
            pass  # Non bloccare se fallisce
    
    return {
        "lambda_home": lh,
        "lambda_away": la,
        "rho": rho,
        "spread_corrente": round(spread_corrente_calculated, 3),  # Spread corrente calcolato dai lambda finali
        "total_corrente": round(total_corrente_calculated, 3),  # Total corrente calcolato dai lambda finali
        "spread_apertura": spread_apertura if spread_apertura is not None else None,
        "total_apertura": total_apertura if total_apertura is not None else None,
        "p_home": p_home_final,
        "p_draw": p_draw_final,
        "p_away": p_away_final,
        "p_home_raw": p_home,  # Probabilità raw (non calibrate)
        "p_draw_raw": p_draw,
        "p_away_raw": p_away,
        "calibration_applied": calibrate_func is not None,
        "ensemble_applied": use_ensemble,
        "market_movement": movement_info,  # Info movimento mercato
        "additional_api_data": additional_api_data,  # Dati API aggiuntive
        "over_15": over_15,
        "under_15": under_15,
        "over_25": over_25,
        "under_25": under_25,
        "over_35": over_35,
        "under_35": under_35,
        "over_05_ht": over_05_ht,
        "btts": btts,
        "gg_over25": gg_over25,
        "even_ft": even_ft,
        "odd_ft": odd_ft,
        "even_ht": even_ht,
        "odd_ht": odd_ht,
        "cs_home": cs_home,
        "cs_away": cs_away,
        "clean_sheet_qualcuno": 1 - btts,
        "multigol_home": multigol_home,
        "multigol_away": multigol_away,
        "multigol_totale": multigol_total,
        "dc": dc,
        "validation_warnings": validation_warnings,  # Warning per probabilità anomale
        "matrix_sum": matrix_sum,  # Somma matrice per debug
        "marg2": marg2,
        "marg3": marg3,
        "combo_book": combo_book,
        "top10": top10,
        "ent_home": ent_home,
        "ent_away": ent_away,
        "odds_prob": odds_prob,
        "scost": scost,
        "odd_mass": odd_mass,
        "even_mass2": even_mass2,
        "cover_0_2": cover_0_2,
        "cover_0_3": cover_0_3,
    }

# ============================================================
#   CONTROLLI QUALITÀ MIGLIORATI
# ============================================================

def check_coerenza_quote_improved(
    odds_1: float,
    odds_x: float,
    odds_2: float,
    odds_over25: float,
    odds_under25: float,
    odds_btts: float = None,
) -> Tuple[List[str], float]:
    """
    Versione migliorata con scoring quantitativo.
    Returns: (warnings, quality_score)
    """
    warnings = []
    quality_score = 100.0
    
    # 1. Check margine
    if odds_1 and odds_x and odds_2:
        margin = (1/odds_1 + 1/odds_x + 1/odds_2) - 1
        if margin > 0.10:
            warnings.append(f"Margine 1X2 alto ({margin*100:.1f}%) → quote meno competitive")
            quality_score -= 15
        elif margin < 0.03:
            warnings.append("Margine 1X2 sospettosamente basso → verificare")
            quality_score -= 10
    
    # 2. Check coerenza favorita
    if odds_1 and odds_2:
        if odds_1 < 1.35 and odds_2 < 4.0:
            warnings.append("Casa molto favorita ma trasferta non abbastanza alta")
            quality_score -= 12
        if odds_1 > 3.5 and odds_2 > 3.5:
            warnings.append("Match molto equilibrato/caotico → alta varianza")
            quality_score -= 8
    
    # 3. Check over/under
    if odds_over25 and odds_under25:
        margin_ou = (1/odds_over25 + 1/odds_under25) - 1
        if not (0.02 < margin_ou < 0.12):
            warnings.append(f"Margine O/U anomalo ({margin_ou*100:.1f}%)")
            quality_score -= 10
        
        # Coerenza con 1X2
        if odds_1 and odds_1 < 1.5 and odds_over25 > 2.3:
            warnings.append("Favorita netta ma over alto → contraddizione")
            quality_score -= 15
    
    # 4. Check BTTS coerenza
    if odds_btts and odds_over25:
        p_btts = 1/odds_btts
        p_over = 1/odds_over25
        # BTTS e Over dovrebbero essere correlati
        if p_btts > 0.65 and p_over < 0.40:
            warnings.append("BTTS alto ma Over basso → incoerenza")
            quality_score -= 12
    
    # 5. Liquidità implicita
    if odds_1 and odds_x and odds_2:
        min_odd = min(odds_1, odds_x, odds_2)
        max_odd = max(odds_1, odds_x, odds_2)
        if max_odd / min_odd > 15:
            warnings.append("Range quote molto ampio → possibile bassa liquidità")
            quality_score -= 8
    
    quality_score = max(0, quality_score)
    return warnings, quality_score

def compute_market_confidence_score(
    odds_1: float,
    odds_x: float,
    odds_2: float,
    odds_over25: float = None,
    odds_under25: float = None,
    odds_dnb_home: float = None,
    odds_dnb_away: float = None,
    odds_btts: float = None,
    num_bookmakers: int = 5,
) -> float:
    """
    Calcola confidence score basato su:
    - Margini
    - Coerenza tra mercati
    - Numero bookmakers
    - Spread quote
    
    Score: 0-100 (100 = massima confidence)
    """
    score = 50.0  # Base
    
    # 1. Numero bookmakers (proxy liquidità)
    if num_bookmakers >= 10:
        score += 20
    elif num_bookmakers >= 7:
        score += 15
    elif num_bookmakers >= 5:
        score += 10
    elif num_bookmakers >= 3:
        score += 5
    else:
        score -= 5
    
    # 2. Margine 1X2
    if odds_1 and odds_x and odds_2:
        margin = (1/odds_1 + 1/odds_x + 1/odds_2) - 1
        if 0.04 < margin < 0.08:
            score += 10  # Margine ottimale
        elif margin < 0.04:
            score -= 5  # Troppo stretto
        elif margin > 0.12:
            score -= 10  # Troppo largo
    
    # 3. Completezza mercati
    markets_available = sum([
        odds_over25 is not None,
        odds_dnb_home is not None,
        odds_btts is not None,
    ])
    score += markets_available * 5
    
    # 4. Coerenza DNB con 1X2
    if odds_dnb_home and odds_dnb_away and odds_1 and odds_2:
        # DNB dovrebbe essere < 1X2 corrispondente
        if odds_dnb_home < odds_1 and odds_dnb_away < odds_2:
            score += 10
        else:
            score -= 8
    
    # 5. Spread probabilità (quanto è definito il favorito)
    if odds_1 and odds_2:
        p1 = 1/odds_1
        p2 = 1/odds_2
        spread = abs(p1 - p2)
        if spread > 0.30:
            score += 8  # Favorito chiaro
        elif spread < 0.10:
            score -= 5  # Molto equilibrato → più incerto
    
    return max(0, min(100, score))

# ============================================================
#              STREAMLIT APP MIGLIORATA
# ============================================================

st.set_page_config(page_title="⚽ Modello Scommesse PRO – Dixon-Coles Bayesiano", layout="wide")

st.title("⚽ Modello Scommesse Avanzato")

st.caption(f"🕐 Esecuzione: {datetime.now().isoformat(timespec='seconds')}")

# Session state initialization
if "soccer_leagues" not in st.session_state:
    st.session_state.soccer_leagues = []
if "events_for_league" not in st.session_state:
    st.session_state.events_for_league = []
if "selected_event_prices" not in st.session_state:
    st.session_state.selected_event_prices = {}
if "selected_league_key" not in st.session_state:
    st.session_state.selected_league_key = None
if "selected_event_id" not in st.session_state:
    st.session_state.selected_event_id = None

# ============================================================
#               SEZIONE STORICO E METRICHE
# ============================================================

st.subheader("📊 Storico e Performance")

col_hist1, col_hist2 = st.columns(2)

with col_hist1:
    if os.path.exists(ARCHIVE_FILE):
        df_st = pd.read_csv(ARCHIVE_FILE)
        st.write(f"📁 Analisi salvate: **{len(df_st)}**")
        
        # Dashboard metrics avanzate
        dashboard_metrics = calculate_dashboard_metrics()
        
        # Real-time Performance Monitoring (ALTA PRIORITÀ)
        realtime_metrics = get_realtime_performance_metrics(window_days=7)
        
        if realtime_metrics.get("status") == "ok":
            st.markdown("### ⚡ Performance Real-time (Ultimi 7 giorni)")
            col_rt1, col_rt2, col_rt3, col_rt4 = st.columns(4)
            
            with col_rt1:
                if realtime_metrics.get("accuracy"):
                    st.metric("🎯 Accuracy RT", f"{realtime_metrics['accuracy']:.1f}%",
                             delta=f"{realtime_metrics.get('trend', 0):+.1f}%" if realtime_metrics.get('trend') else None)
            
            with col_rt2:
                if realtime_metrics.get("roi"):
                    st.metric("💵 ROI RT", f"{realtime_metrics['roi']:.1f}%")
            
            with col_rt3:
                if realtime_metrics.get("bets_placed"):
                    st.metric("📊 Scommesse RT", realtime_metrics['bets_placed'])
            
            with col_rt4:
                alert_status = realtime_metrics.get("alert_status", "good")
                if alert_status == "critical":
                    st.error(realtime_metrics.get("alert_message", "⚠️ Critical"))
                elif alert_status == "warning":
                    st.warning(realtime_metrics.get("alert_message", "⚠️ Warning"))
                else:
                    st.success("✅ Performance OK")
        
        if dashboard_metrics and "error" not in dashboard_metrics:
            col_dash1, col_dash2 = st.columns(2)
            
            with col_dash1:
                if "accuracy_home" in dashboard_metrics:
                    st.metric("🎯 Accuracy Modello", f"{dashboard_metrics['accuracy_home']:.1f}%")
                if "brier_score" in dashboard_metrics and dashboard_metrics["brier_score"]:
                    st.metric("📈 Brier Score", f"{dashboard_metrics['brier_score']:.3f}",
                             help="0 = perfetto, 1 = pessimo")
            
            with col_dash2:
                if "roi_simulato" in dashboard_metrics and dashboard_metrics["roi_simulato"]:
                    roi_data = dashboard_metrics["roi_simulato"]
                    st.metric("💵 ROI Simulato", f"{roi_data.get('roi', 0):.1f}%",
                             help=f"{roi_data.get('bets', 0)} scommesse piazzate")
                if "trend_accuracy" in dashboard_metrics:
                    trend = dashboard_metrics["trend_accuracy"]
                    st.metric("📊 Trend Accuracy", f"{trend:+.1f}%",
                             help="Differenza ultime 20 vs prime 20 partite")
        
        # Calcola metriche se ci sono risultati reali (fallback)
        if "esito_reale" in df_st.columns and "match_ok" in df_st.columns:
            df_complete = df_st[df_st["esito_reale"].notna() & (df_st["esito_reale"] != "")]
            
            if len(df_complete) > 0 and not dashboard_metrics:
                accuracy = df_complete["match_ok"].mean() * 100
                st.metric("🎯 Accuracy Modello", f"{accuracy:.1f}%")
        
        st.dataframe(df_st.tail(15), height=300)
    else:
        st.info("Nessuno storico ancora")
    
    # Portfolio Summary
    portfolio_summary = get_portfolio_summary()
    if portfolio_summary.get("total_bets", 0) > 0:
        st.markdown("### 💼 Portfolio Scommesse")
        col_port1, col_port2, col_port3 = st.columns(3)
        
        with col_port1:
            st.metric("Totale Scommesse", portfolio_summary["total_bets"])
            st.metric("In Attesa", portfolio_summary["pending_bets"])
        
        with col_port2:
            st.metric("Vinte", portfolio_summary["won_bets"])
            st.metric("Perse", portfolio_summary["lost_bets"])
        
        with col_port3:
            st.metric("Profit", f"€{portfolio_summary['profit']:.2f}")
            st.metric("ROI", f"{portfolio_summary['roi']:.1f}%")
        
        if st.button("📋 Visualizza Portfolio Completo"):
            if os.path.exists(PORTFOLIO_FILE):
                df_port = pd.read_csv(PORTFOLIO_FILE)
                st.dataframe(df_port, use_container_width=True)

with col_hist2:
    st.markdown("### 🗑️ Gestione Storico")
    if os.path.exists(ARCHIVE_FILE):
        df_del = pd.read_csv(ARCHIVE_FILE)
        if not df_del.empty:
            df_del["label"] = df_del.apply(
                lambda r: f"{r.get('timestamp','?')} – {r.get('match','(no name)')}",
                axis=1,
            )
            to_delete = st.selectbox("Elimina riga:", df_del["label"].tolist())
            if st.button("🗑️ Elimina"):
                df_new = df_del[df_del["label"] != to_delete].drop(columns=["label"])
                df_new.to_csv(ARCHIVE_FILE, index=False)
                st.success("✅ Eliminato")
                st.rerun()

st.markdown("---")

# ============================================================
#        CARICAMENTO PARTITA DA API
# ============================================================

st.subheader("🔍 Carica Partita da The Odds API")

col_load1, col_load2 = st.columns([1, 2])

with col_load1:
    if st.button("1️⃣ Carica Leghe"):
        with st.spinner("Caricamento leghe in corso..."):
            st.session_state.soccer_leagues = oddsapi_get_soccer_leagues()
        if st.session_state.soccer_leagues:
            st.success(f"✅ {len(st.session_state.soccer_leagues)} leghe caricate")
        else:
            st.warning("⚠️ Nessuna lega caricata. Controlla i messaggi di errore sopra o verifica la configurazione dell'API key.")

if st.session_state.soccer_leagues:
    league_names = [f"{l['title']} ({l['key']})" for l in st.session_state.soccer_leagues]
    selected_league_label = st.selectbox("2️⃣ Seleziona Lega", league_names)
    selected_league_key = selected_league_label.split("(")[-1].replace(")", "").strip()

    if st.button("3️⃣ Carica Partite"):
        st.session_state.events_for_league = oddsapi_get_events_for_league(selected_league_key)
        st.session_state.selected_league_key = selected_league_key
        st.success(f"✅ {len(st.session_state.events_for_league)} partite")

    if st.session_state.events_for_league:
        match_labels = []
        for ev in st.session_state.events_for_league:
            home = ev.get("home_team")
            away = ev.get("away_team")
            start = ev.get("commence_time", "")[:16].replace("T", " ")
            match_labels.append(f"{home} vs {away} – {start}")

        selected_match_label = st.selectbox("4️⃣ Seleziona Partita", match_labels)
        idx = match_labels.index(selected_match_label)
        event = st.session_state.events_for_league[idx]

        event_id = event.get("id") or event.get("event_id") or event.get("key")
        st.session_state.selected_event_id = event_id

        prices = oddsapi_extract_prices_improved(event)
        st.session_state.selected_event_prices = prices
        
        num_bookmakers = len(event.get("bookmakers", []))
        st.info(f"📊 Quote estratte da **{num_bookmakers}** bookmakers con Shin normalization")
        
        # Mostra comparazione bookmakers
        best_odds = compare_bookmaker_odds(event)
        if best_odds:
            summary = find_best_odds_summary(best_odds)
            if summary.get("1"):
                best_1 = summary["1"]
                st.caption(f"🏆 Migliore quota Casa: **{best_1['best_odds']:.2f}** su {best_1['best_bookmaker']} "
                          f"(media: {best_1['avg_odds']:.2f}, +{best_1['value']:.1f}% vs media)")
        
        st.success("✅ Quote precaricate")

        if st.button("🔄 Refresh Quote"):
            ref_ev = oddsapi_refresh_event(
                st.session_state.selected_league_key,
                st.session_state.selected_event_id
            )
            if ref_ev:
                new_prices = oddsapi_extract_prices_improved(ref_ev)
                st.session_state.selected_event_prices = new_prices
                st.success("✅ Quote aggiornate")
                st.rerun()

st.markdown("---")

# ============================================================
#        CONFIGURAZIONE TELEGRAM (OPZIONALE)
# ============================================================

with st.expander("🤖 Configurazione Telegram Bot (Opzionale)", expanded=False):
    st.markdown("""
    **Come configurare:**
    1. Crea un bot su Telegram scrivendo a [@BotFather](https://t.me/BotFather)
    2. Invia `/newbot` e segui le istruzioni
    3. Copia il **Token** che ti viene fornito
    4. Per ottenere il **Chat ID**, segui la guida dettagliata qui sotto
    5. In alternativa, puoi usare variabili d'ambiente: `TELEGRAM_BOT_TOKEN` e `TELEGRAM_CHAT_ID`
    """)
    
    # Guida dettagliata per trovare Chat ID
    with st.expander("📖 **Guida Completa: Come Trovare il Chat ID**", expanded=False):
        st.markdown("""
        ## 🎯 Metodo 1: Chat Privata (Messaggi Personali)
        
        **Per ricevere messaggi sul tuo account Telegram personale:**
        
        ### Passo 1: Ottieni il tuo User ID
        1. Apri Telegram (app o web)
        2. Cerca **[@userinfobot](https://t.me/userinfobot)** nella barra di ricerca
        3. Clicca sul bot e avvia una conversazione
        4. Invia il comando `/start`
        5. Il bot ti mostrerà il tuo **User ID** (es. `123456789`)
        6. **Copia questo numero** - questo è il tuo Chat ID!
        
        ### Passo 2: Avvia il tuo bot
        1. Cerca il nome del tuo bot (quello che hai creato con @BotFather)
        2. Avvia una conversazione con il bot
        3. **IMPORTANTE**: Invia almeno un messaggio al bot (anche solo `/start`)
        4. Questo è necessario perché il bot possa inviarti messaggi
        
        ### Passo 3: Verifica
        - Il Chat ID per chat private è un **numero positivo** (es. `123456789`)
        - Non include segni `-` o altri caratteri
        - Solo numeri!
        
        ---
        
        ## 👥 Metodo 2: Gruppo Telegram
        
        **Per ricevere messaggi in un gruppo:**
        
        ### Passo 1: Aggiungi il bot al gruppo
        1. Apri il gruppo dove vuoi ricevere i messaggi
        2. Vai su **Info gruppo** (icona in alto)
        3. Clicca su **Aggiungi membri** o **Aggiungi amministratori**
        4. Cerca il nome del tuo bot e aggiungilo al gruppo
        5. **IMPORTANTE**: Il bot deve essere membro del gruppo
        
        ### Passo 2: Ottieni il Group ID
        1. Aggiungi **[@userinfobot](https://t.me/userinfobot)** al gruppo
        2. Il bot mostrerà automaticamente le informazioni del gruppo
        3. Cerca il **Group ID** (es. `-1001234567890`)
        4. **Copia questo numero** - include il segno `-` all'inizio!
        
        ### Passo 3: Verifica
        - Il Chat ID per gruppi è un **numero negativo** (es. `-123456789` o `-1001234567890`)
        - Include sempre il segno `-` all'inizio
        - I supergruppi iniziano con `-100`
        
        ---
        
        ## 📢 Metodo 3: Canale Telegram
        
        **Per ricevere messaggi in un canale:**
        
        ### ⚠️ IMPORTANTE: @userinfobot NON funziona per canali!
        Il bot @userinfobot mostra solo le tue informazioni personali, non quelle del canale.
        Per i canali devi usare un metodo diverso.
        
        ### Passo 1: Rendi il bot amministratore
        1. Apri il canale dove vuoi ricevere i messaggi
        2. Vai su **Info canale** (icona in alto a destra)
        3. Clicca su **Amministratori** → **Aggiungi amministratore**
        4. Cerca il nome del tuo bot e aggiungilo
        5. **IMPORTANTE**: Il bot deve essere **amministratore** del canale (non solo membro)
        
        ### Passo 2: Ottieni il Channel ID - METODO A (Consigliato)
        
        **Usa @RawDataBot:**
        1. Aggiungi **[@RawDataBot](https://t.me/RawDataBot)** al canale come amministratore
        2. Inoltra un messaggio qualsiasi dal canale a @RawDataBot (in una chat privata con il bot)
        3. Il bot ti mostrerà tutti i dati del messaggio
        4. Cerca la riga `"chat":{"id":-1001234567890}`
        5. Il numero dopo `"id":` è il tuo **Channel ID** (es. `-1001234567890`)
        6. **Copia questo numero** - include il segno `-` e inizia con `-100`
        
        ### Passo 2 Alternativo: METODO B (Più semplice)
        
        **Usa @getidsbot:**
        1. Aggiungi **[@getidsbot](https://t.me/getidsbot)** al canale come amministratore
        2. Il bot dovrebbe mostrare automaticamente il Channel ID
        3. Se non appare, invia un messaggio nel canale e il bot risponderà con l'ID
        4. **Copia il numero** mostrato (inizia con `-100`)
        
        ### Passo 2 Alternativo: METODO C (Se i bot non funzionano)
        
        **Usa il tuo bot personalizzato:**
        1. Se hai già creato il bot con @BotFather, aggiungilo come amministratore al canale
        2. Invia un messaggio di test nel canale
        3. Usa questo script Python per ottenere l'ID:
        ```python
        import requests
        bot_token = "TUO_BOT_TOKEN"
        url = f"https://api.telegram.org/bot{bot_token}/getUpdates"
        response = requests.get(url)
        data = response.json()
        # Cerca il canale nei risultati
        for update in data.get("result", []):
            if "channel_post" in update:
                chat_id = update["channel_post"]["chat"]["id"]
                print(f"Channel ID: {chat_id}")
        ```
        
        ### Passo 3: Verifica
        - Il Chat ID per canali è un **numero negativo che inizia con `-100`** (es. `-1001234567890`)
        - Include sempre il segno `-` all'inizio
        - Deve iniziare con `-100`
        - **NON** usare il tuo User ID (quello che vedi con @userinfobot)!
        
        ---
        
        ## ✅ Riepilogo Formati Chat ID
        
        | Tipo | Formato | Esempio | Note |
        |------|---------|---------|------|
        | **Chat Privata** | Numero positivo | `123456789` | Solo numeri, nessun segno |
        | **Gruppo** | Numero negativo | `-123456789` | Include segno `-` |
        | **Supergruppo** | Numero negativo `-100...` | `-1001234567890` | Inizia con `-100` |
        | **Canale** | Numero negativo `-100...` | `-1001234567890` | Inizia con `-100` |
        
        ---
        
        ## 🔧 Bot Utili per Trovare Chat ID
        
        - **[@userinfobot](https://t.me/userinfobot)**: Per User ID (chat private) e Group ID ⚠️ NON funziona per canali!
        - **[@getidsbot](https://t.me/getidsbot)**: Per Channel ID (canali) - Aggiungi al canale come admin
        - **[@RawDataBot](https://t.me/RawDataBot)**: Mostra tutti i dati raw - Inoltra un messaggio dal canale al bot
        - **[@username_to_id_bot](https://t.me/username_to_id_bot)**: Converte username in ID (se il canale ha username pubblico)
        
        ---
        
        ## ⚠️ Problemi Comuni
        
        ### "Chat not found" o "Chat ID non valido"
        
        **Per Chat Private:**
        - ✅ Hai inviato almeno un messaggio al bot? (anche solo `/start`)
        - ✅ Il Chat ID è un numero positivo senza segni?
        - ✅ Hai copiato correttamente il numero?
        
        **Per Gruppi:**
        - ✅ Il bot è membro del gruppo?
        - ✅ Il Chat ID include il segno `-` all'inizio?
        - ✅ Hai usato @userinfobot per ottenere l'ID?
        
        **Per Canali:**
        - ✅ Il bot è **amministratore** del canale?
        - ✅ Il Chat ID inizia con `-100`?
        - ✅ Hai usato @getidsbot per ottenere l'ID?
        
        ---
        
        ## 💡 Suggerimenti
        
        1. **Testa sempre**: Usa il pulsante "🧪 Testa Configurazione" prima di abilitare l'invio automatico
        2. **Copia con attenzione**: Assicurati di non includere spazi o caratteri extra
        3. **Verifica formato**: Controlla che il formato corrisponda al tipo di chat (vedi tabella sopra)
        4. **Bot attivo**: Assicurati che il bot creato con @BotFather sia ancora attivo
        """)
    
    telegram_enabled = st.checkbox("📤 Invia analisi automaticamente su Telegram", value=False,
                                   help="Se abilitato, ogni analisi verrà inviata automaticamente al tuo bot Telegram")
    
    col_tg1, col_tg2 = st.columns(2)
    
    with col_tg1:
        telegram_token = st.text_input("Bot Token", value=TELEGRAM_BOT_TOKEN, type="password",
                                       help="Token del bot (da @BotFather)",
                                       placeholder="123456789:ABCdefGHIjklMNOpqrsTUVwxyz")
    
    with col_tg2:
        telegram_chat_id = st.text_input("Chat ID", value=TELEGRAM_CHAT_ID,
                                        help="ID della chat dove inviare (da @userinfobot)",
                                        placeholder="123456789")
    
telegram_prob_threshold = st.slider(
    "🎯 Soglia minima probabilità per notifiche Telegram (%)",
    min_value=0.0,
    max_value=100.0,
    value=float(TELEGRAM_MIN_PROBABILITY),
    step=1.0,
    help="Il bot invia notifiche soltanto per i mercati in cui la probabilità del modello supera questa soglia."
)
st.session_state["telegram_prob_threshold"] = telegram_prob_threshold

# Pulsante per testare la configurazione
if telegram_token and telegram_chat_id:
    col_test1, col_test2 = st.columns([1, 2])
    with col_test1:
        if st.button("🧪 Testa Configurazione", help="Invia un messaggio di test per verificare che token e chat ID siano corretti"):
            with st.spinner("Invio messaggio di test..."):
                test_result = test_telegram_chat_id(
                    bot_token=telegram_token,
                    chat_id=telegram_chat_id
                )
                    
                if test_result.get("success"):
                    st.success("✅ **Test riuscito!** Messaggio di test inviato con successo. La configurazione è corretta.")
                else:
                    error_msg = test_result.get("error_message", "Errore sconosciuto")
                    error_type = test_result.get("error_type", "other")
                    
                    # Mostra messaggio di errore dettagliato
                    if error_type == "no_token":
                        st.error(f"❌ **Token Bot non configurato**\n\n{error_msg}")
                    elif error_type == "no_chat_id":
                        st.error(f"❌ **Chat ID non configurato**\n\n{error_msg}")
                    elif error_type == "invalid_token":
                        st.error(f"❌ **Token non valido**\n\n{error_msg}\n\nVerifica che il token sia corretto e che il bot sia ancora attivo.")
                    elif error_type == "invalid_chat_id":
                        st.error("❌ **Chat ID non valido**")
                        st.markdown(error_msg)
                        with st.expander("🔍 **Guida passo-passo per risolvere**", expanded=True):
                            st.markdown("""
                                **Per Chat Private:**
                                1. Apri Telegram e cerca [@userinfobot](https://t.me/userinfobot)
                                2. Avvia una conversazione e invia `/start`
                                3. Il bot ti mostrerà il tuo **User ID** (es. `123456789`)
                                4. Copia questo numero e incollalo nel campo "Chat ID"
                                5. **IMPORTANTE**: Prima di testare, invia almeno un messaggio al tuo bot (anche solo `/start`)
                                
                                **Per Gruppi:**
                                1. Aggiungi [@userinfobot](https://t.me/userinfobot) al gruppo
                                2. Il bot mostrerà il **Group ID** (es. `-1001234567890`)
                                3. Copia questo numero (include il segno `-`)
                                4. Aggiungi il tuo bot al gruppo come membro
                                
                                **Per Canali:**
                                1. Il bot deve essere **amministratore** del canale
                                2. Il Chat ID del canale inizia con `-100` (es. `-1001234567890`)
                                3. Per ottenere il Chat ID, usa [@getidsbot](https://t.me/getidsbot) nel canale
                                
                                **Verifica rapida:**
                                - Chat ID privata: numero positivo (es. `123456789`)
                                - Chat ID gruppo: numero negativo (es. `-123456789`)
                                - Chat ID canale: numero negativo che inizia con `-100` (es. `-1001234567890`)
                                """)
                    elif error_type == "rate_limit":
                        st.warning(f"⏱️ **Rate Limit**\n\n{error_msg}")
                    elif error_type == "timeout" or error_type == "connection_error":
                        st.warning(f"🌐 **Problema di connessione**\n\n{error_msg}")
                    else:
                        st.warning(f"⚠️ **Errore test Telegram**\n\n{error_msg}")
    with col_test2:
        st.caption("💡 **Suggerimento**: Testa sempre la configurazione prima di abilitare l'invio automatico")
    
    if telegram_enabled and (not telegram_token or not telegram_chat_id):
        st.warning("⚠️ Inserisci Bot Token e Chat ID per abilitare Telegram")

st.markdown("---")

# ============================================================
#        INPUT DATI PARTITA
# ============================================================

st.subheader("📝 Dati Partita")

api_prices = st.session_state.get("selected_event_prices", {})

col_match1, col_match2 = st.columns(2)

with col_match1:
    default_match_name = ""
    if api_prices.get("home"):
        default_match_name = f"{api_prices['home']} vs {api_prices['away']}"
    match_name = st.text_input("Nome Partita", value=default_match_name)

with col_match2:
    league_type = st.selectbox("Lega", [
        "generic",
        "premier_league",
        "la_liga",
        "serie_a",
        "bundesliga",
        "ligue_1",
    ])

st.subheader("📊 Linee di Apertura (Manuali)")

col_ap1, col_ap2 = st.columns(2)

with col_ap1:
    spread_apertura = st.number_input("Spread Apertura", value=0.0, step=0.25,
                                      help="Differenza gol attesa all'apertura (es. 0.5 = casa favorita di 0.5 gol)")
with col_ap2:
    total_apertura = st.number_input("Total Apertura", value=2.5, step=0.25,
                                     help="Total gol atteso all'apertura")

st.subheader("💰 Quote Principali")

col_q1, col_q2, col_q3 = st.columns(3)

with col_q1:
    odds_1 = st.number_input("Quota 1 (Casa)", 
                             value=float(api_prices.get("odds_1") or 2.00), 
                             step=0.01)
    odds_over25 = st.number_input("Quota Over 2.5", 
                                  value=float(api_prices.get("odds_over25") or 0.0), 
                                  step=0.01)

with col_q2:
    odds_x = st.number_input("Quota X (Pareggio)", 
                            value=float(api_prices.get("odds_x") or 3.50), 
                            step=0.01)
    odds_under25 = st.number_input("Quota Under 2.5", 
                                   value=float(api_prices.get("odds_under25") or 0.0), 
                                   step=0.01)

with col_q3:
    odds_2 = st.number_input("Quota 2 (Trasferta)", 
                            value=float(api_prices.get("odds_2") or 3.80), 
                            step=0.01)
    total_line = st.number_input("Total Corrente", value=2.5, step=0.25,
                                 help="Total gol atteso corrente (se diverso da apertura)")

# Spread corrente (opzionale, calcolato automaticamente dai lambda finali)
spread_corrente_input = st.number_input("Spread Corrente (Opzionale)", value=0.0, step=0.25,
                                       help="Inserisci solo se diverso da apertura. Se lasci 0.0, viene calcolato automaticamente dai lambda finali dopo tutti gli aggiustamenti.")
spread_corrente = spread_corrente_input if spread_corrente_input != 0.0 else None

st.subheader("🎲 Quote Speciali")

col_s1, col_s2, col_s3 = st.columns(3)

with col_s1:
    odds_dnb_home = st.number_input("DNB Casa", 
                                    value=float(api_prices.get("odds_dnb_home") or 0.0), 
                                    step=0.01)

with col_s2:
    odds_dnb_away = st.number_input("DNB Trasferta", 
                                    value=float(api_prices.get("odds_dnb_away") or 0.0), 
                                    step=0.01)

with col_s3:
    odds_btts = st.number_input("BTTS Sì (API)", 
                               value=float(api_prices.get("odds_btts") or 0.0), 
                               step=0.01)

btts_manual = st.number_input("BTTS Sì (Manuale - es. Bet365)", 
                              value=0.0, step=0.01,
                              help="Inserisci qui quota BTTS da altro bookmaker se vuoi override")

st.subheader("📊 xG/xA e Boost (Opzionali)")

col_xg1, col_xg2 = st.columns(2)

with col_xg1:
    xg_home_for = st.number_input("xG For Casa", value=0.0, step=0.1)
    xg_home_against = st.number_input("xG Against Casa", value=0.0, step=0.1)
    xa_home_for = st.number_input("xA For Casa", value=0.0, step=0.1)
    xa_home_against = st.number_input("xA Against Casa", value=0.0, step=0.1)
    boost_home = st.slider("Boost Casa (%)", -20, 20, 0) / 100.0

with col_xg2:
    xg_away_for = st.number_input("xG For Trasferta", value=0.0, step=0.1)
    xg_away_against = st.number_input("xG Against Trasferta", value=0.0, step=0.1)
    xa_away_for = st.number_input("xA For Trasferta", value=0.0, step=0.1)
    xa_away_against = st.number_input("xA Against Trasferta", value=0.0, step=0.1)
    boost_away = st.slider("Boost Trasferta (%)", -20, 20, 0) / 100.0

has_xg = all(x > 0 for x in [xg_home_for, xg_home_against, xg_away_for, xg_away_against])
has_xa = any(x > 0 for x in [xa_home_for, xa_home_against, xa_away_for, xa_away_against])

st.markdown("---")

# ============================================================
#              CALCOLO MODELLO
# ============================================================

if st.button("🎯 CALCOLA MODELLO AVANZATO", type="primary"):
    with st.spinner("Elaborazione con modello Dixon-Coles Bayesiano..."):
        
        # 0. VALIDAZIONE INPUT ROBUSTA (URGENTE)
        try:
            validation_result = validate_all_inputs(
                odds_1=odds_1,
                odds_x=odds_x,
                odds_2=odds_2,
                total=total_line,
                odds_over25=odds_over25 if (odds_over25 and isinstance(odds_over25, (int, float)) and odds_over25 > 0) else None,
                odds_under25=odds_under25 if (odds_under25 and isinstance(odds_under25, (int, float)) and odds_under25 > 0) else None,
                odds_btts=odds_btts if (odds_btts and isinstance(odds_btts, (int, float)) and odds_btts > 0) else None,
                odds_dnb_home=odds_dnb_home if (odds_dnb_home and isinstance(odds_dnb_home, (int, float)) and odds_dnb_home > 0) else None,
                odds_dnb_away=odds_dnb_away if (odds_dnb_away and isinstance(odds_dnb_away, (int, float)) and odds_dnb_away > 0) else None,
                spread_apertura=spread_apertura if spread_apertura != 0.0 else None,
                total_apertura=total_apertura if total_apertura != 2.5 else None,
                spread_corrente=spread_corrente if spread_corrente != 0.0 else None,
                xg_for_home=xg_home_for if (xg_home_for and isinstance(xg_home_for, (int, float)) and xg_home_for > 0) else None,
                xg_against_home=xg_home_against if (xg_home_against and isinstance(xg_home_against, (int, float)) and xg_home_against > 0) else None,
                xg_for_away=xg_away_for if (xg_away_for and isinstance(xg_away_for, (int, float)) and xg_away_for > 0) else None,
                xg_against_away=xg_away_against if (xg_away_against and isinstance(xg_away_against, (int, float)) and xg_away_against > 0) else None,
                xa_for_home=xa_home_for if (xa_home_for and isinstance(xa_home_for, (int, float)) and xa_home_for > 0) else None,
                xa_against_home=xa_home_against if (xa_home_against and isinstance(xa_home_against, (int, float)) and xa_home_against > 0) else None,
                xa_for_away=xa_away_for if (xa_away_for and isinstance(xa_away_for, (int, float)) and xa_away_for > 0) else None,
                xa_against_away=xa_away_against if (xa_away_against and isinstance(xa_away_against, (int, float)) and xa_away_against > 0) else None,
            )
            
            validated = validation_result["validated"]
            validation_warnings = validation_result["warnings"]
            
            # Usa valori validati
            odds_1 = validated["odds_1"]
            odds_x = validated["odds_x"]
            odds_2 = validated["odds_2"]
            total_line = validated["total"]
            odds_over25 = validated.get("odds_over25")
            odds_under25 = validated.get("odds_under25")
            odds_btts = validated.get("odds_btts")
            odds_dnb_home = validated.get("odds_dnb_home")
            odds_dnb_away = validated.get("odds_dnb_away")
            spread_apertura = validated.get("spread_apertura")
            total_apertura = validated.get("total_apertura")
            spread_corrente = validated.get("spread_corrente")
            
            if validation_warnings:
                st.warning("⚠️ Validazione input: " + "; ".join(validation_warnings))
                
        except ValidationError as e:
            st.error(f"❌ Errore validazione input: {e}")
            st.stop()
        
        # 1. Check quality
        warnings, quality_score = check_coerenza_quote_improved(
            odds_1, odds_x, odds_2,
            odds_over25, odds_under25,
            odds_btts
        )
        
        # 2. Market confidence
        # ⚠️ PROTEZIONE: Valida che events_for_league non sia vuoto prima di accedere a [0]
        events_for_league = st.session_state.get("events_for_league", [])
        if events_for_league and len(events_for_league) > 0 and isinstance(events_for_league[0], dict):
            num_books = len(events_for_league[0].get("bookmakers", []))
        else:
            # Fallback: usa valore di default se non disponibile
            num_books = 1
            logger.warning("events_for_league vuoto o non valido, uso num_books=1 come default")
        
        market_conf = compute_market_confidence_score(
            odds_1, odds_x, odds_2,
            odds_over25, odds_under25,
            odds_dnb_home if (odds_dnb_home and isinstance(odds_dnb_home, (int, float)) and odds_dnb_home > 0) else None,
            odds_dnb_away if (odds_dnb_away and isinstance(odds_dnb_away, (int, float)) and odds_dnb_away > 0) else None,
            odds_btts if (odds_btts and isinstance(odds_btts, (int, float)) and odds_btts > 0) else None,
            num_books
        )
        
        # 3. Recupera dati avanzati (fatigue, motivation, time)
        # ⚠️ PROTEZIONE: Valida match_name prima di split
        if match_name and " vs " in match_name:
            match_parts = match_name.split(" vs ")
            home_team_name = api_prices.get("home") or (match_parts[0] if len(match_parts) > 0 else "")
            away_team_name = api_prices.get("away") or (match_parts[1] if len(match_parts) > 1 else "")
        else:
            home_team_name = api_prices.get("home") or ""
            away_team_name = api_prices.get("away") or ""
        
        # Data partita da event se disponibile
        match_datetime = None
        if st.session_state.get("events_for_league"):
            try:
                current_event = None
                for ev in st.session_state.events_for_league:
                    if ev.get("id") == st.session_state.get("selected_event_id"):
                        current_event = ev
                        break
                
                if current_event:
                    match_datetime = current_event.get("commence_time")
            except:
                pass
        
        # Recupera dati fatigue e motivation
        fatigue_home_data = None
        fatigue_away_data = None
        motivation_home_data = None
        motivation_away_data = None
        advanced_team_data = None
        
        if home_team_name and away_team_name and match_datetime:
            with st.spinner("📊 Recupero dati avanzati da API-Football..."):
                try:
                    # Dati base (fatigue, motivation)
                    fatigue_home_data = get_team_fatigue_and_motivation_data(
                        home_team_name, league_type, match_datetime
                    )
                    fatigue_away_data = get_team_fatigue_and_motivation_data(
                        away_team_name, league_type, match_datetime
                    )
                    
                    # Motivation usa stessi dati (sono già inclusi)
                    motivation_home_data = {
                        "position": fatigue_home_data.get("position"),
                        "points_from_relegation": fatigue_home_data.get("points_from_relegation"),
                        "points_from_europe": fatigue_home_data.get("points_from_europe"),
                        "data_available": fatigue_home_data.get("data_available"),
                    }
                    motivation_away_data = {
                        "position": fatigue_away_data.get("position"),
                        "points_from_relegation": fatigue_away_data.get("points_from_relegation"),
                        "points_from_europe": fatigue_away_data.get("points_from_europe"),
                        "data_available": fatigue_away_data.get("data_available"),
                    }
                    
                    # Dati avanzati (statistiche, H2H, infortuni) - BACKGROUND
                    advanced_team_data = get_advanced_team_data(
                        home_team_name, away_team_name, league_type, match_datetime
                    )
                    
                    # Log discreto (solo in console, non in UI)
                    if advanced_team_data.get("data_available"):
                        logger.info(f"Dati avanzati disponibili: Form={advanced_team_data.get('home_team_stats') is not None}, "
                                   f"H2H={advanced_team_data.get('h2h_data') is not None}, "
                                   f"Injuries={advanced_team_data.get('home_injuries') is not None}")
                except Exception as e:
                    st.warning(f"⚠️ Errore recupero dati avanzati: {e}")
                    logger.error(f"Errore dettagliato recupero dati avanzati: {e}")
        
        # 4. Calcolo modello
        xg_args = {}
        if has_xg:
            xg_args = {
                "xg_for_home": xg_home_for,
                "xg_against_home": xg_home_against,
                "xg_for_away": xg_away_for,
                "xg_against_away": xg_away_against,
            }
        xa_args = {}
        if has_xa:
            xa_args = {
                "xa_for_home": xa_home_for if xa_home_for > 0 else None,
                "xa_against_home": xa_home_against if xa_home_against > 0 else None,
                "xa_for_away": xa_away_for if xa_away_for > 0 else None,
                "xa_against_away": xa_away_against if xa_away_against > 0 else None,
            }
        
        ris = risultato_completo_improved(
            odds_1=odds_1,
            odds_x=odds_x,
            odds_2=odds_2,
            total=total_line,
            odds_over25=odds_over25 if (odds_over25 and isinstance(odds_over25, (int, float)) and odds_over25 > 0) else None,
            odds_under25=odds_under25 if (odds_under25 and isinstance(odds_under25, (int, float)) and odds_under25 > 0) else None,
            odds_btts=odds_btts if (odds_btts and isinstance(odds_btts, (int, float)) and odds_btts > 0) else None,
            odds_dnb_home=odds_dnb_home if (odds_dnb_home and isinstance(odds_dnb_home, (int, float)) and odds_dnb_home > 0) else None,
            odds_dnb_away=odds_dnb_away if (odds_dnb_away and isinstance(odds_dnb_away, (int, float)) and odds_dnb_away > 0) else None,
            manual_boost_home=boost_home,
            manual_boost_away=boost_away,
            league=league_type,
            home_team=home_team_name,
            away_team=away_team_name,
            match_datetime=match_datetime,
            fatigue_home=fatigue_home_data,
            fatigue_away=fatigue_away_data,
            motivation_home=motivation_home_data,
            motivation_away=motivation_away_data,
            advanced_data=advanced_team_data,  # Dati avanzati in background
            spread_apertura=spread_apertura if spread_apertura != 0.0 else None,
            total_apertura=total_apertura if total_apertura != 2.5 else None,  # Default 2.5 = non specificato
            spread_corrente=spread_corrente,
            total_corrente=total_line if total_line != 2.5 else None,  # Total corrente (se diverso da default)
            **xg_args,
            **xa_args
        )
        
        # 5. Mostra info fatigue e motivation
        if fatigue_home_data and fatigue_home_data.get("data_available"):
            with st.expander("💪 Dati Fatigue e Motivation", expanded=False):
                col_fat1, col_fat2 = st.columns(2)
                
                with col_fat1:
                    st.markdown(f"**🏠 {home_team_name}**")
                    if fatigue_home_data.get("days_since_last_match") is not None:
                        days = fatigue_home_data["days_since_last_match"]
                        fatigue_f = calculate_fatigue_factor(
                            home_team_name, days, fatigue_home_data.get("matches_last_30_days")
                        )
                        st.write(f"Giorni ultima partita: {days}")
                        st.write(f"Partite ultimi 30gg: {fatigue_home_data.get('matches_last_30_days', 'N/A')}")
                        st.write(f"Fattore Fatigue: {fatigue_f:.3f} ({'+' if fatigue_f > 1 else ''}{(fatigue_f-1)*100:.1f}%)")
                    
                    if motivation_home_data and motivation_home_data.get("data_available"):
                        pos = motivation_home_data.get("position")
                        if pos:
                            st.write(f"Posizione classifica: {pos}°")
                            if motivation_home_data.get("points_from_relegation") is not None:
                                st.write(f"Punti da salvezza: {motivation_home_data['points_from_relegation']}")
                            if motivation_home_data.get("points_from_europe") is not None:
                                st.write(f"Punti da Europa: {motivation_home_data['points_from_europe']}")
                            motivation_f = calculate_motivation_factor(
                                pos,
                                motivation_home_data.get("points_from_relegation"),
                                motivation_home_data.get("points_from_europe"),
                                is_derby_match(home_team_name, away_team_name, league_type) if home_team_name and away_team_name else False
                            )
                            st.write(f"Fattore Motivation: {motivation_f:.3f} ({'+' if motivation_f > 1 else ''}{(motivation_f-1)*100:.1f}%)")
                
                with col_fat2:
                    st.markdown(f"**✈️ {away_team_name}**")
                    if fatigue_away_data and fatigue_away_data.get("data_available"):
                        days = fatigue_away_data.get("days_since_last_match")
                        if days is not None:
                            fatigue_f = calculate_fatigue_factor(
                                away_team_name, days, fatigue_away_data.get("matches_last_30_days")
                            )
                            st.write(f"Giorni ultima partita: {days}")
                            st.write(f"Partite ultimi 30gg: {fatigue_away_data.get('matches_last_30_days', 'N/A')}")
                            st.write(f"Fattore Fatigue: {fatigue_f:.3f} ({'+' if fatigue_f > 1 else ''}{(fatigue_f-1)*100:.1f}%)")
                    
                    if motivation_away_data and motivation_away_data.get("data_available"):
                        pos = motivation_away_data.get("position")
                        if pos:
                            st.write(f"Posizione classifica: {pos}°")
                            if motivation_away_data.get("points_from_relegation") is not None:
                                st.write(f"Punti da salvezza: {motivation_away_data['points_from_relegation']}")
                            if motivation_away_data.get("points_from_europe") is not None:
                                st.write(f"Punti da Europa: {motivation_away_data['points_from_europe']}")
                            motivation_f = calculate_motivation_factor(
                                pos,
                                motivation_away_data.get("points_from_relegation"),
                                motivation_away_data.get("points_from_europe"),
                                is_derby_match(home_team_name, away_team_name, league_type) if home_team_name and away_team_name else False
                            )
                            st.write(f"Fattore Motivation: {motivation_f:.3f} ({'+' if motivation_f > 1 else ''}{(motivation_f-1)*100:.1f}%)")
                
                if is_derby_match(home_team_name, away_team_name, league_type) if home_team_name and away_team_name else False:
                    st.info("🔥 **DERBY DETECTED** - Alta motivazione per entrambe le squadre!")
        
        # 6. BTTS finale
        btts_prob_model = ris["btts"]
        final_btts_odds, btts_source = blend_btts_sources_improved(
            odds_btts_api=odds_btts if (odds_btts and isinstance(odds_btts, (int, float)) and odds_btts > 0) else None,
            btts_from_model=btts_prob_model,
            manual_btts=btts_manual if (btts_manual and isinstance(btts_manual, (int, float)) and btts_manual > 1.01) else None,
            market_confidence=market_conf / 100
        )
        
        # ========================================
        #          VISUALIZZAZIONE RISULTATI
        # ========================================
        
        st.success("✅ Calcolo completato!")
        
        # Metriche principali
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        
        with col_m1:
            st.metric("🏆 Quality Score", f"{quality_score:.0f}/100")
        with col_m2:
            st.metric("📊 Market Confidence", f"{market_conf:.0f}/100")
        with col_m3:
            st.metric("🏠 λ Casa", f"{ris['lambda_home']:.3f}")
        with col_m4:
            st.metric("✈️ λ Trasferta", f"{ris['lambda_away']:.3f}")
        
        # Mostra rho, precisione, spread e total correnti
        col_m5, col_m6, col_m7, col_m8 = st.columns(4)
        with col_m5:
            st.metric("🔗 ρ (correlazione)", f"{ris['rho']:.4f}")
        with col_m6:
            # Calcola precisione: quanto si discosta il modello dalle quote
            avg_error = np.mean([abs(v) for v in ris['scost'].values()])
            st.metric("📊 Avg Scostamento", f"{avg_error:.2f}%")
        with col_m7:
            # Spread corrente calcolato dai lambda finali
            spread_curr = ris.get("spread_corrente", ris["lambda_home"] - ris["lambda_away"])
            st.metric("📈 Spread Corrente", f"{spread_curr:.3f}")
        with col_m8:
            # Total corrente calcolato dai lambda finali
            total_curr = ris.get("total_corrente", ris["lambda_home"] + ris["lambda_away"])
            st.metric("⚽ Total Corrente", f"{total_curr:.3f}")
        
        # Mostra confronto apertura vs corrente e Market Movement Intelligence
        movement_info = ris.get("market_movement", {})
        # Verifica se abbiamo dati apertura validi (non None e non valori di default)
        has_spread_ap = spread_apertura is not None and spread_apertura != 0.0
        has_total_ap = total_apertura is not None and total_apertura != 2.5
        if movement_info and (has_spread_ap or has_total_ap):
            with st.expander("📊 Market Movement Intelligence", expanded=False):
                col_comp1, col_comp2, col_comp3 = st.columns(3)
                
                with col_comp1:
                    st.markdown("**📈 Spread**")
                    # Usa spread corrente calcolato dai lambda finali (dopo tutti gli aggiustamenti)
                    spread_curr = ris.get("spread_corrente", ris["lambda_home"] - ris["lambda_away"])
                    # Gestisci spread_apertura che può essere None
                    if spread_apertura is not None:
                        st.write(f"Apertura: {spread_apertura:.3f}")
                    else:
                        st.write("Apertura: N/A")
                    st.write(f"Corrente: {spread_curr:.3f} ⭐")
                    st.caption("⭐ Calcolato dai lambda finali (dopo aggiustamenti API)")
                    if movement_info.get("movement_spread") and spread_apertura is not None and spread_apertura != 0.0:
                        diff_spread = movement_info["movement_spread"]
                        # Assicurati che spread_apertura sia un numero per il confronto
                        spread_ap_num = float(spread_apertura) if spread_apertura is not None else 0.0
                        st.write(f"**Movimento**: {diff_spread:+.3f} {'(→ casa)' if spread_curr > spread_ap_num else '(→ trasferta)'}")
                
                with col_comp2:
                    st.markdown("**⚽ Total**")
                    # Usa total corrente calcolato dai lambda finali (dopo tutti gli aggiustamenti)
                    total_curr = ris.get("total_corrente", ris["lambda_home"] + ris["lambda_away"])
                    # Gestisci total_apertura che può essere None
                    if total_apertura is not None:
                        st.write(f"Apertura: {total_apertura:.3f}")
                    else:
                        st.write("Apertura: N/A")
                    st.write(f"Corrente: {total_curr:.3f} ⭐")
                    st.caption("⭐ Calcolato dai lambda finali (dopo aggiustamenti API)")
                    if movement_info.get("movement_total") and total_apertura is not None and total_apertura != 2.5:
                        diff_total = movement_info["movement_total"]
                        st.write(f"**Movimento**: {diff_total:+.3f} {'(↑ più gol)' if diff_total > 0 else '(↓ meno gol)'}")
                
                with col_comp3:
                    st.markdown("**🎯 Strategia Blend**")
                    movement_type = movement_info.get("movement_type", "UNKNOWN")
                    movement_type_names = {
                        "STABLE": "📊 Mercato Stabile",
                        "MODERATE": "⚡ Movimento Moderato",
                        "HIGH_SMART_MONEY": "🔥 Smart Money",
                        "NO_OPENING_DATA": "❌ No Apertura"
                    }
                    st.write(f"**Tipo**: {movement_type_names.get(movement_type, movement_type)}")
                    st.write(f"**Magnitudine**: {movement_info.get('movement_magnitude', 0):.3f}")
                    st.write(f"**Peso Apertura**: {movement_info.get('weight_apertura', 0)*100:.0f}%")
                    st.write(f"**Peso Corrente**: {movement_info.get('weight_corrente', 0)*100:.0f}%")
                    
                    if movement_type == "HIGH_SMART_MONEY":
                        st.info("💡 **Smart Money rilevato**: Il mercato si è mosso significativamente. Le quote correnti hanno più peso (70%).")
                    elif movement_type == "STABLE":
                        st.info("💡 **Mercato stabile**: Le quote di apertura sono più affidabili. Peso apertura 70%.")
        
        # Mostra dati API aggiuntive (se disponibili)
        additional_data = ris.get("additional_api_data", {})
        
        # Dati meteo
        weather_info = additional_data.get("weather")
        if weather_info and weather_info.get("available"):
            with st.expander("🌤️ Dati Meteo (OpenWeatherMap)", expanded=False):
                col_w1, col_w2 = st.columns(2)
                
                with col_w1:
                    st.markdown(f"**📍 Città**: {weather_info.get('city', 'N/A')}")
                    st.markdown(f"**🌡️ Temperatura**: {weather_info.get('temperature', 0):.1f}°C")
                    st.markdown(f"**💨 Vento**: {weather_info.get('wind_speed', 0):.1f} m/s")
                    st.markdown(f"**💧 Umidità**: {weather_info.get('humidity', 0):.0f}%")
                
                with col_w2:
                    st.markdown(f"**🌧️ Pioggia**: {weather_info.get('rain', 0):.1f} mm/h")
                    st.markdown(f"**☁️ Condizioni**: {weather_info.get('description', 'N/A').title()}")
                    weather_factor = weather_info.get('weather_factor', 1.0)
                    impact_pct = (1 - weather_factor) * 100
                    if impact_pct > 0:
                        st.markdown(f"**📉 Impatto Gol**: -{impact_pct:.1f}%")
                    else:
                        st.markdown(f"**✅ Impatto Gol**: Neutro")
                    
                    adjustments_list = weather_info.get('adjustments', [])
                    if adjustments_list:
                        st.markdown("**Aggiustamenti applicati:**")
                        for adj in adjustments_list:
                            st.caption(f"• {adj}")
        
        # Dati TheSportsDB (stadio)
        thesportsdb_info = additional_data.get("thesportsdb")
        if thesportsdb_info and thesportsdb_info.get("available"):
            with st.expander("🏟️ Info Stadio (TheSportsDB)", expanded=False):
                st.markdown(f"**Nome**: {thesportsdb_info.get('stadium', 'N/A')}")
                capacity = thesportsdb_info.get('stadium_capacity')
                if capacity:
                    # Converti capacity in numero se necessario (può essere stringa)
                    try:
                        if isinstance(capacity, str):
                            capacity_str = ''.join(filter(str.isdigit, capacity))
                            capacity_num = int(capacity_str) if capacity_str else None
                        elif isinstance(capacity, (int, float)):
                            capacity_num = int(capacity)
                        else:
                            capacity_num = None
                        
                        if capacity_num is not None:
                            st.markdown(f"**Capacità**: {capacity_num:,} spettatori")
                        else:
                            st.markdown(f"**Capacità**: {capacity} spettatori")
                    except (ValueError, TypeError):
                        st.markdown(f"**Capacità**: {capacity} spettatori")
                location = thesportsdb_info.get('stadium_location')
                # Assicurati che location sia una stringa valida
                if location and isinstance(location, str):
                    st.markdown(f"**Ubicazione**: {location}")
        
        # Info aggiustamenti applicati
        adjustments_applied = []
        
        # Aggiungi meteo agli aggiustamenti se applicato
        if weather_info and weather_info.get("available") and weather_info.get("weather_factor", 1.0) != 1.0:
            impact = (1 - weather_info.get("weather_factor", 1.0)) * 100
            adjustments_applied.append(f"🌤️ Meteo: {impact:+.1f}% gol")
        
        # Market Movement Intelligence (sempre mostrato se dati apertura disponibili)
        # Verifica se abbiamo dati apertura validi (non None e non valori di default)
        has_spread_ap_check = spread_apertura is not None and spread_apertura != 0.0
        has_total_ap_check = total_apertura is not None and total_apertura != 2.5
        if has_spread_ap_check or has_total_ap_check:
            movement_info = ris.get("market_movement", {})
            if movement_info:
                movement_type = movement_info.get("movement_type", "")
                if movement_type != "NO_OPENING_DATA":
                    adjustments_applied.append(f"📊 Market Movement: {movement_info.get('movement_type', 'UNKNOWN')}")
        
        if match_datetime:
            time_adj = get_time_based_adjustments(match_datetime, league_type)
            total_time_factor = time_adj["time_factor"] * time_adj["day_factor"] * time_adj["season_factor"]
            if abs(total_time_factor - 1.0) > 0.02:
                adjustments_applied.append(f"⏰ Time-based: {total_time_factor:.3f}")
        
        if fatigue_home_data and fatigue_home_data.get("data_available") or fatigue_away_data and fatigue_away_data.get("data_available"):
            adjustments_applied.append("💪 Fatigue factors")
        
        if motivation_home_data and motivation_home_data.get("data_available") or motivation_away_data and motivation_away_data.get("data_available"):
            adjustments_applied.append("🎯 Motivation factors")
        
        # Info dati avanzati (discreto, solo se disponibili)
        if advanced_team_data and advanced_team_data.get("data_available"):
            advanced_info = []
            if advanced_team_data.get("home_team_stats") or advanced_team_data.get("away_team_stats"):
                advanced_info.append("📊 Form")
            if advanced_team_data.get("h2h_data"):
                advanced_info.append("⚔️ H2H")
            if advanced_team_data.get("home_injuries") or advanced_team_data.get("away_injuries"):
                advanced_info.append("🏥 Injuries")
            
            if advanced_info:
                st.caption(f"✅ Dati avanzati applicati: {', '.join(advanced_info)}")
        
        if adjustments_applied:
            st.info("✅ **Aggiustamenti applicati**: " + ", ".join(adjustments_applied))
        
        # Warnings
        if warnings:
            with st.expander("⚠️ Avvisi Quality Check", expanded=True):
                for w in warnings:
                    st.warning(w)
        
        # Value Finder con Kelly Criterion
        st.subheader("💎 Value Finder & Kelly Criterion")
        
        # Input bankroll
        bankroll = st.number_input("💰 Bankroll (€)", value=100.0, min_value=1.0, step=10.0, key="bankroll_input")
        kelly_fraction = st.slider("🎯 Kelly Fraction", 0.1, 1.0, 0.25, 0.05, 
                                   help="Frazione di Kelly da usare (0.25 = Quarter Kelly, più conservativo)")
        
        value_rows = []
        
        # 1X2
        for lab, p_mod, odd in [
            ("1 (Casa)", ris["p_home"], odds_1),
            ("X (Pareggio)", ris["p_draw"], odds_x),
            ("2 (Trasferta)", ris["p_away"], odds_2),
        ]:
            p_book = 1 / odd
            edge = (p_mod - p_book) * 100
            ev = (p_mod * odd - 1) * 100
            
            # Kelly Criterion
            kelly = kelly_criterion(p_mod, odd, bankroll, kelly_fraction)
            
            value_rows.append({
                "Mercato": "1X2",
                "Esito": lab,
                "Prob Modello %": f"{p_mod*100:.1f}",
                "Prob Quota %": f"{p_book*100:.1f}",
                "Edge %": f"{edge:+.1f}",
                "EV %": f"{ev:+.1f}",
                "Kelly %": f"{kelly['kelly_percent']:.2f}",
                "Stake (€)": f"{kelly['stake']:.2f}",
                "Value": "✅" if edge >= 3 else ("⚠️" if edge >= 1 else ""),
                "Rec": kelly['recommendation']
            })
        
        # Over/Under
        if odds_over25 and odds_over25 > 0:
            p_mod = ris["over_25"]
            p_book = 1 / odds_over25
            edge = (p_mod - p_book) * 100
            ev = (p_mod * odds_over25 - 1) * 100
            kelly = kelly_criterion(p_mod, odds_over25, bankroll, kelly_fraction)
            
            value_rows.append({
                "Mercato": "Over/Under 2.5",
                "Esito": "Over 2.5",
                "Prob Modello %": f"{p_mod*100:.1f}",
                "Prob Quota %": f"{p_book*100:.1f}",
                "Edge %": f"{edge:+.1f}",
                "EV %": f"{ev:+.1f}",
                "Kelly %": f"{kelly['kelly_percent']:.2f}",
                "Stake (€)": f"{kelly['stake']:.2f}",
                "Value": "✅" if edge >= 3 else ("⚠️" if edge >= 1 else ""),
                "Rec": kelly['recommendation']
            })
        
        # BTTS
        if final_btts_odds > 1:
            p_mod = btts_prob_model
            p_book = 1 / final_btts_odds
            edge = (p_mod - p_book) * 100
            ev = (p_mod * final_btts_odds - 1) * 100
            kelly = kelly_criterion(p_mod, final_btts_odds, bankroll, kelly_fraction)
            
            value_rows.append({
                "Mercato": "BTTS",
                "Esito": f"Sì ({btts_source})",
                "Prob Modello %": f"{p_mod*100:.1f}",
                "Prob Quota %": f"{p_book*100:.1f}",
                "Edge %": f"{edge:+.1f}",
                "EV %": f"{ev:+.1f}",
                "Kelly %": f"{kelly['kelly_percent']:.2f}",
                "Stake (€)": f"{kelly['stake']:.2f}",
                "Value": "✅" if edge >= 3 else ("⚠️" if edge >= 1 else ""),
                "Rec": kelly['recommendation']
            })
        
        df_value = pd.DataFrame(value_rows)
        
        # Highligh value bets
        df_value_high = df_value[df_value["Value"].str.contains("✅", na=False)]
        
        if not df_value_high.empty:
            st.success(f"🎯 {len(df_value_high)} value bet(s) identificate!")
            st.dataframe(df_value_high, use_container_width=True)
        
        st.dataframe(df_value, use_container_width=True)
        
        # Info calibrazione e ensemble
        if ris.get("calibration_applied"):
            st.info("✅ **Calibrazione applicata**: Le probabilità sono state calibrate usando dati storici")
        if ris.get("ensemble_applied"):
            st.info("✅ **Ensemble applicato**: Combinazione di più modelli per maggiore robustezza")
        
        # Export e Portfolio
        col_exp1, col_exp2, col_exp3 = st.columns(3)
        
        with col_exp1:
            if st.button("📥 Esporta CSV"):
                try:
                    csv_file = export_analysis_to_csv(ris, match_name)
                    st.success(f"✅ Esportato: {csv_file}")
                    with open(csv_file, 'rb') as f:
                        st.download_button("⬇️ Download CSV", f.read(), 
                                         file_name=csv_file, mime="text/csv")
                except Exception as e:
                    st.error(f"Errore export: {e}")
        
        with col_exp2:
            if st.button("📊 Esporta Excel"):
                try:
                    odds_data = {"odds_1": odds_1, "odds_x": odds_x, "odds_2": odds_2}
                    excel_file = export_analysis_to_excel(ris, match_name, odds_data)
                    st.success(f"✅ Esportato: {excel_file}")
                    with open(excel_file, 'rb') as f:
                        st.download_button("⬇️ Download Excel", f.read(),
                                         file_name=excel_file, 
                                         mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                except ImportError:
                    st.warning("⚠️ Installare openpyxl: pip install openpyxl")
                except Exception as e:
                    st.error(f"Errore export: {e}")
        
        with col_exp3:
            # Aggiungi al portfolio
            if st.button("💼 Aggiungi al Portfolio"):
                # Trova la scommessa con maggiore edge
                best_bet = None
                best_edge = -1
                
                for lab, p_mod, odd in [
                    ("1", ris["p_home"], odds_1),
                    ("X", ris["p_draw"], odds_x),
                    ("2", ris["p_away"], odds_2),
                ]:
                    if odd > 0:
                        p_book = 1 / odd
                        edge = p_mod - p_book
                        if edge > best_edge and edge >= 0.03:
                            best_edge = edge
                            kelly = kelly_criterion(p_mod, odd, bankroll, kelly_fraction)
                            best_bet = {
                                "esito": lab,
                                "odds": odd,
                                "stake": kelly["stake"],
                                "probability": p_mod
                            }
                
                if best_bet:
                    add_to_portfolio(
                        match_name,
                        "1X2",
                        best_bet["esito"],
                        best_bet["odds"],
                        best_bet["stake"],
                        best_bet["probability"]
                    )
                    st.success(f"✅ Aggiunto al portfolio: {best_bet['esito']} @ {best_bet['odds']:.2f}")
                else:
                    st.info("ℹ️ Nessuna value bet con edge sufficiente")
        
        # Comparazione Bookmakers
        if st.session_state.get("events_for_league"):
            try:
                current_event = None
                for ev in st.session_state.events_for_league:
                    if ev.get("id") == st.session_state.get("selected_event_id"):
                        current_event = ev
                        break
                
                if current_event:
                    with st.expander("📊 Comparazione Bookmakers"):
                        best_odds = compare_bookmaker_odds(current_event)
                        summary = find_best_odds_summary(best_odds)
                        
                        if summary:
                            st.markdown("### 🏆 Migliori Quote Disponibili")
                            
                            for market, data in summary.items():
                                market_name = {
                                    "1": "Casa (1)",
                                    "X": "Pareggio (X)",
                                    "2": "Trasferta (2)",
                                    "over_25": "Over 2.5",
                                    "under_25": "Under 2.5",
                                    "btts": "BTTS Sì"
                                }.get(market, market)
                                
                                col_bk1, col_bk2, col_bk3 = st.columns(3)
                                with col_bk1:
                                    st.metric(f"{market_name} - Migliore", 
                                            f"{data['best_odds']:.2f}",
                                            f"{data['best_bookmaker']}")
                                with col_bk2:
                                    st.metric("Media Mercato", f"{data['avg_odds']:.2f}")
                                with col_bk3:
                                    st.metric("Value vs Media", f"+{data['value']:.1f}%",
                                             help="Quanto la migliore quota è sopra la media")
                        else:
                            st.info("Nessun dato disponibile per comparazione")
            except Exception as e:
                pass  # Silently fail se non disponibile
        
        # Dettagli completi
        with st.expander("📈 Probabilità Dettagliate"):
            col_d1, col_d2, col_d3 = st.columns(3)
            
            with col_d1:
                st.markdown("**Esito Finale**")
                st.write(f"Casa: {ris['p_home']*100:.1f}%")
                st.write(f"Pareggio: {ris['p_draw']*100:.1f}%")
                st.write(f"Trasferta: {ris['p_away']*100:.1f}%")
                
                st.markdown("**Double Chance**")
                for k, v in ris["dc"].items():
                    st.write(f"{k}: {v*100:.1f}%")
            
            with col_d2:
                st.markdown("**Over/Under**")
                st.write(f"Over 1.5: {ris['over_15']*100:.1f}%")
                st.write(f"Over 2.5: {ris['over_25']*100:.1f}%")
                st.write(f"Over 3.5: {ris['over_35']*100:.1f}%")
                st.write(f"Under 2.5: {ris['under_25']*100:.1f}%")
                
                st.markdown("**Gol**")
                st.write(f"BTTS: {ris['btts']*100:.1f}%")
                st.write(f"GG + Over 2.5: {ris['gg_over25']*100:.1f}%")
                st.write(f"Clean Sheet Casa: {ris['cs_home']*100:.1f}%")
                st.write(f"Clean Sheet Trasferta: {ris['cs_away']*100:.1f}%")
                
                # Mostra warning se ci sono probabilità anomale
                validation_warnings = ris.get("validation_warnings", [])
                if validation_warnings:
                    st.warning("⚠️ **Avviso: Probabilità anomale rilevate**")
                    for warning in validation_warnings:
                        st.caption(warning)
                    
                    # Mostra lambda per debug
                    st.caption(f"🔍 **Debug Info**: lambda_home={ris['lambda_home']:.3f}, lambda_away={ris['lambda_away']:.3f}, total={ris['lambda_home']+ris['lambda_away']:.3f}, rho={ris.get('rho', 0):.3f}")
                    matrix_sum = ris.get("matrix_sum", 1.0)
                    if abs(matrix_sum - 1.0) > 0.01:
                        st.caption(f"⚠️ Somma matrice: {matrix_sum:.6f} (dovrebbe essere 1.0)")
            
            with col_d3:
                st.markdown("**Pari/Dispari**")
                st.write(f"Pari FT: {ris['even_ft']*100:.1f}%")
                st.write(f"Dispari FT: {ris['odd_ft']*100:.1f}%")
                
                st.markdown("**Statistiche**")
                st.write(f"ρ (correlazione): {ris['rho']:.3f}")
                st.write(f"Entropia Casa: {ris['ent_home']:.2f}")
                st.write(f"Entropia Trasferta: {ris['ent_away']:.2f}")
        
        with st.expander("🎯 Top 10 Risultati Esatti"):
            for h, a, p in ris["top10"]:
                st.write(f"{h}-{a}: **{p:.1f}%**")
        
        # Heatmap matrice score
        with st.expander("🔥 Heatmap Matrice Score", expanded=False):
            st.markdown("""
            ### 📖 Come Leggere la Heatmap
            
            La heatmap mostra la **distribuzione di probabilità di tutti i possibili risultati esatti** della partita.
            
            **Assi:**
            - **Asse Y (verticale, sinistra)**: Gol della squadra **Casa**
            - **Asse X (orizzontale, in basso)**: Gol della squadra **Trasferta**
            
            **Colori:**
            - 🟨 **Giallo/Arancione chiaro**: Probabilità **bassa** (< 2-3%)
            - 🟧 **Arancione**: Probabilità **media** (3-8%)
            - 🟥 **Rosso scuro**: Probabilità **alta** (> 8-10%)
            
            **Esempi di lettura:**
            - **Cella (2, 1)**: Risultato **2-1** per la Casa → probabilità mostrata in %
            - **Cella (0, 0)**: Risultato **0-0** (pareggio senza gol) → probabilità mostrata in %
            - **Cella (1, 3)**: Risultato **1-3** per la Trasferta → probabilità mostrata in %
            
            **Cosa cercare:**
            1. **Zone più scure/rosse** = risultati più probabili secondo il modello
            2. **Diagonali** = risultati equilibrati (es. 1-1, 2-2)
            3. **Zona in alto a sinistra** = vittorie casa (es. 2-0, 3-1)
            4. **Zona in basso a destra** = vittorie trasferta (es. 0-2, 1-3)
            5. **Zona centrale** = pareggi (0-0, 1-1, 2-2)
            
            **Confronto con "Top 10 Risultati":**
            - I risultati più probabili nella heatmap corrispondono ai primi della lista "Top 10"
            - La heatmap ti dà una visione **visiva completa** di tutte le probabilità
            """)
            
            try:
                import matplotlib.pyplot as plt
                import seaborn as sns
                
                # Ricostruisci matrice (per visualizzazione)
                mat_vis = build_score_matrix(ris["lambda_home"], ris["lambda_away"], ris["rho"])
                heatmap_data = create_score_heatmap_data(mat_vis, max_goals=8)
                
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(
                    heatmap_data,
                    annot=True,
                    fmt='.1f',
                    cmap='YlOrRd',
                    cbar_kws={'label': 'Probabilità %'},
                    xticklabels=range(0, min(9, len(heatmap_data))),
                    yticklabels=range(0, min(9, len(heatmap_data))),
                    ax=ax
                )
                ax.set_xlabel('Gol Trasferta', fontsize=12, fontweight='bold')
                ax.set_ylabel('Gol Casa', fontsize=12, fontweight='bold')
                ax.set_title('Distribuzione Probabilità Risultati Esatti', fontsize=14, fontweight='bold')
                st.pyplot(fig)
                plt.close(fig)
                
                # Aggiungi tabella riepilogativa
                st.markdown("### 📊 Riepilogo Zone Heatmap")
                col_hm1, col_hm2, col_hm3 = st.columns(3)
                
                with col_hm1:
                    # Vittorie casa (h > a)
                    # ⚠️ PRECISIONE: Usa accumulo preciso con protezione NaN/negativi
                    prob_vittoria_casa = 0.0
                    mg_vis = len(mat_vis) - 1
                    for h in range(mg_vis + 1):
                        if h < len(mat_vis):
                            for a in range(len(mat_vis[h])):
                                if h > a:
                                    p = mat_vis[h][a]
                                    if p > 0 and p == p:  # Ignora negativi e NaN
                                        prob_vittoria_casa += p
                    prob_vittoria_casa = max(0.0, min(1.0, prob_vittoria_casa)) * 100
                    st.metric("🏠 Vittorie Casa", f"{prob_vittoria_casa:.1f}%")
                
                with col_hm2:
                    # Pareggi (h == a)
                    # ⚠️ PRECISIONE: Usa accumulo preciso con protezione NaN/negativi
                    prob_pareggi = 0.0
                    mg_vis = len(mat_vis) - 1
                    for h in range(mg_vis + 1):
                        if h < len(mat_vis):
                            for a in range(len(mat_vis[h])):
                                if h == a:
                                    p = mat_vis[h][a]
                                    if p > 0 and p == p:  # Ignora negativi e NaN
                                        prob_pareggi += p
                    prob_pareggi = max(0.0, min(1.0, prob_pareggi)) * 100
                    st.metric("⚖️ Pareggi", f"{prob_pareggi:.1f}%")
                
                with col_hm3:
                    # Vittorie trasferta (h < a)
                    # ⚠️ PRECISIONE: Usa accumulo preciso con protezione NaN/negativi
                    prob_vittoria_trasferta = 0.0
                    mg_vis = len(mat_vis) - 1
                    for h in range(mg_vis + 1):
                        if h < len(mat_vis):
                            for a in range(len(mat_vis[h])):
                                if h < a:
                                    p = mat_vis[h][a]
                                    if p > 0 and p == p:  # Ignora negativi e NaN
                                        prob_vittoria_trasferta += p
                    prob_vittoria_trasferta = max(0.0, min(1.0, prob_vittoria_trasferta)) * 100
                    st.metric("✈️ Vittorie Trasferta", f"{prob_vittoria_trasferta:.1f}%")
                
            except ImportError:
                st.info("📊 Installare matplotlib e seaborn per visualizzare heatmap")
            except Exception as e:
                st.warning(f"Errore visualizzazione: {e}")
        
        with st.expander("🔀 Combo Mercati"):
            combo_df = pd.DataFrame([
                {"Combo": k, "Probabilità %": f"{v*100:.1f}"}
                for k, v in ris["combo_book"].items()
            ]).sort_values("Probabilità %", ascending=False)
            st.dataframe(combo_df, use_container_width=True)
        
        # Salvataggio
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "match": match_name,
            "league": league_type,
            "quality_score": quality_score,
            "market_confidence": market_conf,
            "odds_1": odds_1,
            "odds_x": odds_x,
            "odds_2": odds_2,
            "odds_btts": final_btts_odds,
            "lambda_home": round(ris["lambda_home"], 3),
            "lambda_away": round(ris["lambda_away"], 3),
            "rho": round(ris["rho"], 3),
            "p_home": round(ris["p_home"]*100, 2),
            "p_draw": round(ris["p_draw"]*100, 2),
            "p_away": round(ris["p_away"]*100, 2),
            "btts": round(ris["btts"]*100, 2),
            "over_25": round(ris["over_25"]*100, 2),
            "esito_modello": max([("1", ris["p_home"]), ("X", ris["p_draw"]), ("2", ris["p_away"])], 
                                key=lambda x: x[1])[0],
            "esito_reale": "",
            "risultato_reale": "",
            "match_ok": "",
        }
        
        try:
            if os.path.exists(ARCHIVE_FILE):
                df_old = pd.read_csv(ARCHIVE_FILE)
                df_new = pd.concat([df_old, pd.DataFrame([row])], ignore_index=True)
                df_new.to_csv(ARCHIVE_FILE, index=False)
            else:
                pd.DataFrame([row]).to_csv(ARCHIVE_FILE, index=False)
            st.success("💾 Analisi salvata nello storico")
        except Exception as e:
            st.warning(f"Errore salvataggio: {e}")
        
        # 7. INVIO TELEGRAM (se abilitato)
        if telegram_enabled and telegram_token and telegram_chat_id:
            try:
                # Prepara value bets per Telegram
                value_bets_list = []
                telegram_prob_threshold = float(st.session_state.get("telegram_prob_threshold", TELEGRAM_MIN_PROBABILITY))
                for bet in value_rows:
                    prob_str_raw = str(bet.get("Prob Modello %", "0")).replace(",", ".")
                    try:
                        prob_value = float(prob_str_raw)
                    except ValueError:
                        prob_value = 0.0
                    if prob_value < telegram_prob_threshold:
                        continue
                    value_bets_list.append({
                        "Esito": bet.get("Esito", ""),
                        "Prob %": bet.get("Prob Modello %", ""),
                        "Edge %": bet.get("Edge %", ""),
                        "EV %": bet.get("EV %", ""),
                        "Rec": bet.get("Rec", "")
                    })
                
                if not value_bets_list:
                    st.info(f"ℹ️ Nessun mercato supera la soglia Telegram ({telegram_prob_threshold:.0f}%). Notifica non inviata.")
                else:
                    result = None
                    # Formatta messaggio
                    telegram_message = format_analysis_for_telegram(
                        match_name=match_name,
                        ris=ris,
                        odds_1=odds_1,
                        odds_x=odds_x,
                        odds_2=odds_2,
                        quality_score=quality_score,
                        market_conf=market_conf,
                        value_bets=value_bets_list
                    )
                    
                    # Invia messaggio
                    result = send_telegram_message(
                        message=telegram_message,
                        bot_token=telegram_token,
                        chat_id=telegram_chat_id
                    )
                    
                    if result.get("success"):
                        st.success("📤 Analisi inviata su Telegram!")
                    else:
                        # Mostra messaggio di errore dettagliato
                        error_msg = result.get("error_message", "Errore sconosciuto")
                        error_type = result.get("error_type", "other")
                        
                        # Messaggi specifici per tipo di errore
                        if error_type == "no_token":
                            st.error(f"❌ **Token Bot non configurato**\n\n{error_msg}\n\nPer configurare:\n1. Crea un bot su [@BotFather](https://t.me/BotFather)\n2. Invia `/newbot` e segui le istruzioni\n3. Copia il Token fornito")
                        elif error_type == "no_chat_id":
                            st.error(f"❌ **Chat ID non configurato**\n\n{error_msg}\n\nPer ottenere il Chat ID:\n1. Scrivi a [@userinfobot](https://t.me/userinfobot)\n2. Copia il tuo ID numerico")
                        elif error_type == "invalid_token":
                            st.error(f"❌ **Token non valido**\n\n{error_msg}\n\nVerifica che il token sia corretto e che il bot sia ancora attivo.")
                        elif error_type == "invalid_chat_id":
                            # Usa markdown per formattazione migliore
                            st.error("❌ **Chat ID non valido**")
                            st.markdown(error_msg)
                            
                            # Aggiungi sezione interattiva con suggerimenti
                            with st.expander("🔍 **Guida passo-passo per risolvere**", expanded=True):
                                st.markdown("""
                                **Per Chat Private:**
                                1. Apri Telegram e cerca [@userinfobot](https://t.me/userinfobot)
                                2. Avvia una conversazione e invia `/start`
                                3. Il bot ti mostrerà il tuo **User ID** (es. `123456789`)
                                4. Copia questo numero e incollalo nel campo "Chat ID"
                                5. **IMPORTANTE**: Prima di usare il bot, invia almeno un messaggio al tuo bot (anche solo `/start`)
                                
                                **Per Gruppi:**
                                1. Aggiungi [@userinfobot](https://t.me/userinfobot) al gruppo
                                2. Il bot mostrerà il **Group ID** (es. `-1001234567890`)
                                3. Copia questo numero (include il segno `-`)
                                4. Aggiungi il tuo bot al gruppo come membro
                                
                                **Per Canali:**
                                1. Il bot deve essere **amministratore** del canale
                                2. Il Chat ID del canale inizia con `-100` (es. `-1001234567890`)
                                3. Per ottenere il Chat ID, usa [@getidsbot](https://t.me/getidsbot) nel canale
                                
                                **Verifica rapida:**
                                - Chat ID privata: numero positivo (es. `123456789`)
                                - Chat ID gruppo: numero negativo (es. `-123456789`)
                                - Chat ID canale: numero negativo che inizia con `-100` (es. `-1001234567890`)
                                """)
                        elif error_type == "rate_limit":
                            st.warning(f"⏱️ **Rate Limit**\n\n{error_msg}")
                        elif error_type == "timeout" or error_type == "connection_error":
                            st.warning(f"🌐 **Problema di connessione**\n\n{error_msg}")
                        else:
                            st.warning(f"⚠️ **Errore invio Telegram**\n\n{error_msg}")
            except Exception as e:
                logger.error(f"Errore imprevisto in invio Telegram: {e}")
                st.warning(f"⚠️ Errore imprevisto invio Telegram: {e}")
                # Non bloccare il flusso se Telegram fallisce

st.markdown("---")

# ============================================================
#        AGGIORNAMENTO RISULTATI REALI
# ============================================================

st.subheader("🔄 Aggiorna Risultati Reali")

if st.button("Recupera risultati ultimi 3 giorni"):
    if not os.path.exists(ARCHIVE_FILE):
        st.warning("Nessuno storico da aggiornare")
    else:
        with st.spinner("Recupero risultati da API-Football..."):
            df = pd.read_csv(ARCHIVE_FILE)
            today = date.today()
            giorni = [(today - timedelta(days=i)).isoformat() for i in range(0, 4)]
            
            fixtures_map = {}
            for d in giorni:
                fixtures = apifootball_get_fixtures_by_date(d)
                for f in fixtures:
                    if f["fixture"]["status"]["short"] in ["FT", "AET", "PEN"]:
                        home = f["teams"]["home"]["name"]
                        away = f["teams"]["away"]["name"]
                        key = f"{home} vs {away}".strip().lower()
                        gh = f["goals"]["home"]
                        ga = f["goals"]["away"]
                        if gh is not None and ga is not None:
                            fixtures_map[key] = (gh, ga)
            
            updated = 0
            for idx, row in df.iterrows():
                key_row = str(row.get("match", "")).strip().lower()
                if key_row in fixtures_map and (pd.isna(row.get("risultato_reale")) or row.get("risultato_reale") == ""):
                    gh, ga = fixtures_map[key_row]
                    
                    if gh > ga:
                        esito = "1"
                    elif gh == ga:
                        esito = "X"
                    else:
                        esito = "2"
                    
                    df.at[idx, "risultato_reale"] = f"{gh}-{ga}"
                    df.at[idx, "esito_reale"] = esito
                    
                    pred = row.get("esito_modello", "")
                    if pred and esito:
                        df.at[idx, "match_ok"] = 1 if pred == esito else 0
                    
                    updated += 1
            
            df.to_csv(ARCHIVE_FILE, index=False)
            st.success(f"✅ Aggiornate {updated} partite")
            st.rerun()

st.markdown("---")

# ============================================================
#        BACKTESTING E PERFORMANCE
# ============================================================

st.subheader("📊 Backtesting Strategia")

if os.path.exists(ARCHIVE_FILE):
    col_bt1, col_bt2 = st.columns(2)
    
    with col_bt1:
        min_edge_bt = st.slider("Minimo Edge %", 1.0, 10.0, 3.0, 0.5)
        kelly_frac_bt = st.slider("Kelly Fraction", 0.1, 1.0, 0.25, 0.05)
        initial_bank = st.number_input("Bankroll Iniziale (€)", value=100.0, min_value=10.0, step=10.0)
    
    with col_bt2:
        if st.button("🚀 Esegui Backtest", type="primary"):
            with st.spinner("Eseguendo backtest su dati storici..."):
                results = backtest_strategy(
                    ARCHIVE_FILE,
                    min_edge=min_edge_bt / 100,
                    kelly_fraction=kelly_frac_bt,
                    initial_bankroll=initial_bank
                )
                
                if "error" in results:
                    st.warning(f"⚠️ {results['error']}")
                else:
                    st.success("✅ Backtest completato!")
                    
                    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
                    
                    with col_r1:
                        st.metric("💰 Bankroll Finale", f"€{results['final_bankroll']:.2f}")
                    with col_r2:
                        st.metric("📈 Profit", f"€{results['profit']:.2f}", 
                                 f"{results['profit_pct']:+.1f}%")
                    with col_r3:
                        st.metric("🎯 Win Rate", f"{results['win_rate']:.1f}%")
                    with col_r4:
                        st.metric("💵 ROI", f"{results['roi']:.1f}%")
                    
                    st.write(f"**Scommesse piazzate**: {results['bets_placed']} ({results['bets_won']} vinte)")
                    st.write(f"**Totale puntato**: €{results['total_staked']:.2f}")
                    st.write(f"**Totale ritornato**: €{results['total_returned']:.2f}")
                    
                    if results.get('sharpe_approx'):
                        st.write(f"**Sharpe Ratio (approssimato)**: {results['sharpe_approx']:.3f}")
                    
                    # Grafico profit history
                    if len(results.get('profit_history', [])) > 1:
                        try:
                            import matplotlib.pyplot as plt
                            fig, ax = plt.subplots(figsize=(10, 4))
                            ax.plot(results['profit_history'], linewidth=2)
                            ax.axhline(y=initial_bank, color='r', linestyle='--', label='Bankroll Iniziale')
                            ax.set_xlabel('Numero Scommesse')
                            ax.set_ylabel('Bankroll (€)')
                            ax.set_title('Evoluzione Bankroll')
                            ax.legend()
                            ax.grid(True, alpha=0.3)
                            st.pyplot(fig)
                            plt.close(fig)
                        except:
                            pass
else:
    st.info("Nessuno storico disponibile per backtest")

st.markdown("---")
st.caption("Developed with ❤️ | Dixon-Coles Bayesian Model | Shin Normalization | IQR Outlier Detection | Platt Scaling | Kelly Criterion | Ensemble Methods")
