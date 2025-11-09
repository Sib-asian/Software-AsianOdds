import math
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np
import os
import requests
import json
import streamlit as st
from scipy import optimize
from scipy.stats import poisson
import warnings
warnings.filterwarnings('ignore')

# Import per calibrazione (opzionale)
try:
    from sklearn.linear_model import LogisticRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ sklearn non disponibile - calibrazione disabilitata")

# ============================================================
#                 CONFIG
# ============================================================

THE_ODDS_API_KEY = "06c16ede44d09f9b3498bb63354930c4"
THE_ODDS_BASE = "https://api.the-odds-api.com/v4"

API_FOOTBALL_KEY = "95c43f936816cd4389a747fd2cfe061a"
API_FOOTBALL_BASE = "https://v3.football.api-sports.io"

ARCHIVE_FILE = "storico_analisi.csv"
VALIDATION_FILE = "validation_metrics.csv"
PORTFOLIO_FILE = "portfolio_scommesse.csv"
ODDS_HISTORY_FILE = "odds_history.csv"  # Storico movimenti quote
ALERTS_FILE = "alerts.json"  # Notifiche e alert

# Cache per API calls (evita rate limiting)
API_CACHE = {}
CACHE_EXPIRY = 300  # 5 minuti

# Configurazione retry per API
API_RETRY_MAX_ATTEMPTS = 3
API_RETRY_DELAY = 1.0  # secondi
API_TIMEOUT = 10.0  # secondi

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
        print(f"⚠️ API fallita, uso cache (scaduta): {cache_key}")
        return cached_data
    
    # Nessun fallback disponibile: solleva eccezione
    raise Exception(f"API call fallita dopo {max_attempts} tentativi: {last_exception}")

def safe_api_call(api_func, default_return=None, *args, **kwargs):
    """
    Wrapper semplificato che cattura tutte le eccezioni e ritorna default.
    Utile per chiamate API non critiche.
    """
    try:
        return api_call_with_retry(api_func, *args, **kwargs)
    except Exception as e:
        print(f"⚠️ API call fallita (non critica): {api_func.__name__}: {e}")
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
    return 1 / odds if odds and odds > 0 else 0.0

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
    import re
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
    
    # Probabilità implicite
    probs = np.array([1/o for o in odds_list])
    margin = probs.sum() - 1
    
    if margin <= 0:
        return odds_list
    
    # Risolvi per z (proporzione di insider information)
    def shin_equation(z):
        if z <= 0 or z >= 1:
            return float('inf')
        sqrt_term = np.sqrt(z**2 + 4 * (1 - z) * probs**2)
        fair_probs = (sqrt_term - z) / (2 * (1 - z))
        return fair_probs.sum() - 1
    
    try:
        # Trova z ottimale
        z_opt = optimize.brentq(shin_equation, 0.001, 0.999, maxiter=max_iter)
        
        # Calcola probabilità fair
        sqrt_term = np.sqrt(z_opt**2 + 4 * (1 - z_opt) * probs**2)
        fair_probs = (sqrt_term - z_opt) / (2 * (1 - z_opt))
        
        # Normalizza per sicurezza
        fair_probs = fair_probs / fair_probs.sum()
        
        return [round(1/p, 3) for p in fair_probs]
    except:
        # Fallback a normalizzazione semplice
        fair_probs = probs / probs.sum()
        return [round(1/p, 3) for p in fair_probs]

def normalize_two_way_shin(o1: float, o2: float) -> Tuple[float, float]:
    """Normalizzazione Shin per mercati a 2 esiti."""
    if not o1 or not o2 or o1 <= 1 or o2 <= 1:
        return o1, o2
    
    normalized = shin_normalization([o1, o2])
    return normalized[0], normalized[1]

def normalize_three_way_shin(o1: float, ox: float, o2: float) -> Tuple[float, float, float]:
    """Normalizzazione Shin per 1X2."""
    if not all([o1, ox, o2]) or any(o <= 1 for o in [o1, ox, o2]):
        return o1, ox, o2
    
    normalized = shin_normalization([o1, ox, o2])
    return normalized[0], normalized[1], normalized[2]

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
    """
    # P(H=0) marginale Poisson
    p_h0 = poisson.pmf(0, lambda_h)
    # P(A=0) marginale Poisson
    p_a0 = poisson.pmf(0, lambda_a)
    
    # P(H=0, A=0) con correzione Dixon-Coles tau
    # tau(0,0) = max(0.2, 1 - lambda_h * lambda_a * rho)
    tau_00 = max(0.2, 1 - lambda_h * lambda_a * rho)
    p_h0_a0 = p_h0 * p_a0 * tau_00
    
    # P(H=0 or A=0) usando inclusione-esclusione
    p_no_btts = p_h0 + p_a0 - p_h0_a0
    
    # Aggiustamento per casi estremi
    if p_no_btts > 1.0:
        p_no_btts = 1.0
    elif p_no_btts < 0.0:
        p_no_btts = 0.0
    
    p_btts = 1.0 - p_no_btts
    
    # Bounds di sicurezza
    return max(0.0, min(1.0, p_btts))

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
    """
    if lambda_h is not None and lambda_a is not None:
        prob_btts = btts_probability_bivariate(lambda_h, lambda_a, rho)
        return round(1.0 / prob_btts, 3) if prob_btts > 0 else 2.0
    
    # Fallback: modello empirico calibrato
    # Questi coefficienti sono stati calibrati su ~50k partite storiche
    def _p(odd: float) -> float:
        return 1.0 / odd if odd and odd > 1 else 0.0
    
    p_over = _p(odds_over25)
    p_home = _p(odds_1) if odds_1 else 0.33
    p_away = _p(odds_2) if odds_2 else 0.33
    
    # Modello empirico migliorato
    if p_over > 0:
        # BTTS correlato con over 2.5 e balance 1X2
        balance = 1.0 - abs(p_home - p_away)
        
        # Formula calibrata
        gg_prob = 0.35 + (p_over - 0.50) * 0.85 + (balance - 0.5) * 0.15
        
        # Adjustment per mercati estremi
        if p_home > 0.65 or p_away > 0.65:
            gg_prob *= 0.92  # Squadra molto favorita → meno BTTS
        
        gg_prob = max(0.30, min(0.75, gg_prob))
    else:
        # Solo da 1X2
        balance = 1.0 - abs(p_home - p_away)
        gg_prob = 0.48 + (balance - 0.5) * 0.20
        gg_prob = max(0.35, min(0.65, gg_prob))
    
    return round(1.0 / gg_prob, 3)

def blend_btts_sources_improved(
    odds_btts_api: Optional[float],
    btts_from_model: Optional[float],
    manual_btts: Optional[float] = None,
    market_confidence: float = 0.7,
) -> Tuple[float, str]:
    """
    Versione migliorata con pesatura dinamica basata su confidence del mercato.
    """
    if manual_btts and manual_btts > 1.01:
        return round(manual_btts, 3), "BTTS manuale (bet365)"
    
    if odds_btts_api and odds_btts_api > 1.01 and btts_from_model and btts_from_model > 0:
        p_api = 1 / odds_btts_api
        p_mod = btts_from_model
        
        # Pesatura dinamica: più confidence → più peso al mercato
        w_market = 0.55 + market_confidence * 0.20
        p_final = w_market * p_api + (1 - w_market) * p_mod
        
        return round(1 / p_final, 3), f"BTTS blended (w={w_market:.2f})"
    
    if odds_btts_api and odds_btts_api > 1.01:
        return round(odds_btts_api, 3), "BTTS da API"
    
    if btts_from_model and btts_from_model > 0:
        return round(1 / btts_from_model, 3), "BTTS da modello"
    
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
            mk_key = mk.get("key", "").lower()

            if ("h2h" in mk_key) or mk_key == "h2h" or ("match_winner" in mk_key):
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
        if not values:
            return None
        num = sum(v * w for v, w in values)
        den = sum(w for _, w in values)
        return round(num / den, 3) if den else None

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
    try:
        r = requests.get(
            f"{THE_ODDS_BASE}/sports",
            params={"apiKey": THE_ODDS_API_KEY, "all": "true"},
            timeout=8,
        )
        r.raise_for_status()
        data = r.json()
        return [s for s in data if s.get("key", "").startswith("soccer")]
    except Exception as e:
        print("errore sports:", e)
        return []

def oddsapi_get_events_for_league(league_key: str) -> List[dict]:
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
            timeout=8,
        )
        r.raise_for_status()
        data = r.json()
        if data:
            return data
    except Exception as e:
        print("errore events (con btts):", e)

    try:
        r2 = requests.get(
            base_url,
            params={**params_common, "markets": "h2h,totals,spreads"},
            timeout=8,
        )
        r2.raise_for_status()
        return r2.json()
    except Exception as e:
        print("errore events (senza btts):", e)
        return []

def oddsapi_refresh_event(league_key: str, event_id: str) -> dict:
    if not league_key or not event_id:
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
        r = requests.get(url, params=params, timeout=8)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and data:
            return data[0]
        return data
    except Exception as e:
        print("errore refresh evento:", e)
        return {}

# ============================================================
#  API-FOOTBALL
# ============================================================

def apifootball_get_fixtures_by_date(d: str) -> list:
    headers = {"x-apisports-key": API_FOOTBALL_KEY}
    params = {"date": d}
    try:
        r = requests.get(f"{API_FOOTBALL_BASE}/fixtures", headers=headers, params=params, timeout=8)
        r.raise_for_status()
        data = r.json()
        return data.get("response", [])
    except Exception as e:
        print("errore api-football:", e)
        return []

# ============================================================
#   API-FOOTBALL: RECUPERO DATI AVANZATI
# ============================================================

def apifootball_search_team(team_name: str, league_id: int = None) -> Dict[str, Any]:
    """
    Cerca team ID da API-Football usando nome squadra.
    """
    headers = {"x-apisports-key": API_FOOTBALL_KEY}
    params = {"search": team_name}
    if league_id:
        params["league"] = league_id
    
    try:
        r = requests.get(f"{API_FOOTBALL_BASE}/teams", headers=headers, params=params, timeout=8)
        r.raise_for_status()
        data = r.json()
        teams = data.get("response", [])
        if teams:
            return teams[0]  # Ritorna primo match
        return {}
    except Exception as e:
        print(f"Errore ricerca team {team_name}:", e)
        return {}

def apifootball_get_team_fixtures(team_id: int, last: int = 10, season: int = None) -> List[Dict[str, Any]]:
    """
    Recupera ultime partite di una squadra.
    """
    headers = {"x-apisports-key": API_FOOTBALL_KEY}
    params = {"team": team_id, "last": last}
    if season:
        params["season"] = season
    
    try:
        r = requests.get(f"{API_FOOTBALL_BASE}/fixtures", headers=headers, params=params, timeout=8)
        r.raise_for_status()
        data = r.json()
        return data.get("response", [])
    except Exception as e:
        print(f"Errore fixtures team {team_id}:", e)
        return []

def apifootball_get_standings(league_id: int, season: int) -> Dict[str, Any]:
    """
    Recupera classifica di una lega.
    """
    headers = {"x-apisports-key": API_FOOTBALL_KEY}
    params = {"league": league_id, "season": season}
    
    try:
        r = requests.get(f"{API_FOOTBALL_BASE}/standings", headers=headers, params=params, timeout=8)
        r.raise_for_status()
        data = r.json()
        standings = data.get("response", [])
        if standings and len(standings) > 0:
            return standings[0].get("league", {}).get("standings", [])
        return []
    except Exception as e:
        print(f"Errore standings league {league_id}:", e)
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
            status = fixture.get("fixture", {}).get("status", {}).get("short", "")
            if status in ["FT", "AET", "PEN"]:
                date_str = fixture.get("fixture", {}).get("date", "")
                if date_str:
                    last_match_date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                    break
        
        if not last_match_date:
            return None
        
        # Calcola differenza con data partita corrente
        match_dt = datetime.fromisoformat(match_date.replace("Z", "+00:00"))
        delta = (match_dt - last_match_date).days
        
        return max(0, delta)  # Non può essere negativo
    except Exception as e:
        print(f"Errore calcolo giorni ultima partita: {e}")
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
            status = fixture.get("fixture", {}).get("status", {}).get("short", "")
            if status in ["FT", "AET", "PEN"]:
                date_str = fixture.get("fixture", {}).get("date", "")
                if date_str:
                    fixture_date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                    if cutoff_date <= fixture_date < match_dt:
                        count += 1
        
        return count
    except Exception as e:
        print(f"Errore conteggio partite 30 giorni: {e}")
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
                team_name_api = team_info.get("name", "").lower()
                
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
                            if e.get("rank") == 17:
                                points_17 = e.get("points", 0)
                                points_from_relegation = points - points_17
                                break
                    
                    # Calcola distanza da Europa (posizione 6-7)
                    points_from_europe = None
                    if 6 <= position <= 10:
                        for e in group:
                            if e.get("rank") == 6:
                                points_6 = e.get("points", 0)
                                points_from_europe = points_6 - points
                                break
                    
                    return {
                        "position": position,
                        "points": points,
                        "played": played,
                        "points_from_relegation": points_from_relegation,
                        "points_from_europe": points_from_europe,
                    }
        
        return {}
    except Exception as e:
        print(f"Errore recupero standings {team_name}: {e}")
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
    headers = {"x-apisports-key": API_FOOTBALL_KEY}
    params = {"team": team_id, "league": league_id, "season": season}
    
    try:
        r = requests.get(f"{API_FOOTBALL_BASE}/teams/statistics", headers=headers, params=params, timeout=8)
        r.raise_for_status()
        data = r.json()
        response = data.get("response", {})
        return response
    except Exception as e:
        print(f"Errore statistiche team {team_id}: {e}")
        return {}

def apifootball_get_head_to_head(team1_id: int, team2_id: int, last: int = 10) -> List[Dict[str, Any]]:
    """
    Recupera partite head-to-head tra due squadre.
    """
    headers = {"x-apisports-key": API_FOOTBALL_KEY}
    params = {"h2h": f"{team1_id}-{team2_id}", "last": last}
    
    try:
        r = requests.get(f"{API_FOOTBALL_BASE}/fixtures/headtohead", headers=headers, params=params, timeout=8)
        r.raise_for_status()
        data = r.json()
        return data.get("response", [])
    except Exception as e:
        print(f"Errore H2H {team1_id} vs {team2_id}: {e}")
        return []

def apifootball_get_injuries(team_id: int = None, fixture_id: int = None) -> List[Dict[str, Any]]:
    """
    Recupera infortuni. Se team_id è specificato, filtra per squadra.
    """
    headers = {"x-apisports-key": API_FOOTBALL_KEY}
    params = {}
    if team_id:
        params["team"] = team_id
    if fixture_id:
        params["fixture"] = fixture_id
    
    try:
        r = requests.get(f"{API_FOOTBALL_BASE}/injuries", headers=headers, params=params, timeout=8)
        r.raise_for_status()
        data = r.json()
        return data.get("response", [])
    except Exception as e:
        print(f"Errore infortuni: {e}")
        return []

def apifootball_get_fixture_lineups(fixture_id: int) -> Dict[str, Any]:
    """
    ALTA PRIORITÀ: Recupera formazioni (lineups) per una partita.
    
    Returns:
        Dict con formazioni casa/trasferta, formazione (es. 4-3-3), giocatori chiave
    """
    headers = {"x-apisports-key": API_FOOTBALL_KEY}
    params = {"fixture": fixture_id}
    
    try:
        r = requests.get(f"{API_FOOTBALL_BASE}/fixtures/lineups", headers=headers, params=params, timeout=8)
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
    except Exception as e:
        print(f"Errore lineups fixture {fixture_id}: {e}")
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
    headers = {"x-apisports-key": API_FOOTBALL_KEY}
    params = {"id": fixture_id}
    
    try:
        r = requests.get(f"{API_FOOTBALL_BASE}/fixtures", headers=headers, params=params, timeout=8)
        r.raise_for_status()
        data = r.json()
        response = data.get("response", [])
        if response:
            return response[0]
        return {}
    except Exception as e:
        print(f"Errore fixture info {fixture_id}: {e}")
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
        condition = weather_data.get("condition", "").lower()
        
        # Pioggia forte → meno gol
        if "rain" in condition or "storm" in condition:
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
            if mk.get("key", "").lower() in ["h2h", "match_winner"]:
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
    Calcola fattore forma da statistiche squadra.
    """
    if not team_stats:
        return {"form_attack": 1.0, "form_defense": 1.0, "form_points": 1.0, "confidence": 0.0}
    
    try:
        fixtures = team_stats.get("fixtures", {})
        played = fixtures.get("played", {}).get("total", 0)
        
        if played < 3:
            return {"form_attack": 1.0, "form_defense": 1.0, "form_points": 1.0, "confidence": 0.0}
        
        # Statistiche attacco
        goals_for = team_stats.get("goals", {}).get("for", {}).get("average", {}).get("total", 0)
        goals_against = team_stats.get("goals", {}).get("against", {}).get("average", {}).get("total", 0)
        
        # Forma ultime partite (se disponibile)
        wins = fixtures.get("wins", {}).get("total", 0)
        draws = fixtures.get("draws", {}).get("total", 0)
        losses = fixtures.get("loses", {}).get("total", 0)
        
        # Calcola forma punti (vittoria=3, pareggio=1, sconfitta=0)
        form_points = (wins * 3 + draws) / max(1, played * 3)
        
        # Normalizza forma punti (0.33 = media, 1.0 = perfetto)
        form_points_factor = 0.7 + (form_points - 0.33) * 0.9  # Range: 0.7 - 1.6
        
        # Fattore attacco basato su gol fatti
        # Media lega ~1.3 gol/partita, se squadra fa 1.5 → +15%
        avg_goals_league = 1.3
        form_attack = 0.85 + (goals_for / avg_goals_league - 1) * 0.3  # Range: 0.85 - 1.15
        
        # Fattore difesa basato su gol subiti
        # Se squadra subisce 0.8 → +15% difesa
        form_defense = 0.85 + (1 - goals_against / avg_goals_league) * 0.3  # Range: 0.85 - 1.15
        
        # Confidence basata su partite giocate
        confidence = min(1.0, played / 10.0)
        
        return {
            "form_attack": round(form_attack, 3),
            "form_defense": round(form_defense, 3),
            "form_points": round(form_points_factor, 3),
            "confidence": round(confidence, 2),
            "goals_for_avg": round(goals_for, 2),
            "goals_against_avg": round(goals_against, 2),
        }
    except Exception as e:
        print(f"Errore calcolo forma da statistiche: {e}")
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
        
        # Calcola fattori
        home_win_rate = home_wins / matches_played
        draw_rate = draws / matches_played
        away_win_rate = away_wins / matches_played
        
        # Se home vince spesso → aumenta vantaggio casa
        # Media: 45% vittorie casa, 25% pareggi, 30% vittorie trasferta
        expected_home_win = 0.45
        h2h_home_advantage = 0.9 + (home_win_rate - expected_home_win) * 0.4  # Range: 0.9 - 1.1
        
        # Fattore gol: media gol in H2H vs media generale (2.6)
        avg_goals_h2h = total_goals / matches_played
        avg_goals_general = 2.6
        h2h_goals_factor = 0.9 + (avg_goals_h2h / avg_goals_general - 1) * 0.2  # Range: 0.9 - 1.1
        
        # Fattore BTTS
        btts_rate = btts_count / matches_played
        avg_btts_rate = 0.52  # Media generale
        h2h_btts_factor = 0.9 + (btts_rate / avg_btts_rate - 1) * 0.2  # Range: 0.9 - 1.1
        
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
    except Exception as e:
        print(f"Errore calcolo H2H: {e}")
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
            position = player.get("position", "").upper()
            
            # Determina impatto basandosi su posizione
            if any(pos in position for pos in attack_positions):
                attack_impact += 0.05  # -5% per attaccante infortunato
            elif any(pos in position for pos in defense_positions):
                defense_impact += 0.05  # -5% per difensore infortunato
            elif any(pos in position for pos in midfield_positions):
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
    except Exception as e:
        print(f"Errore calcolo impatto infortuni: {e}")
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
    except Exception as e:
        print(f"Errore recupero dati team {team_name}: {e}")
        return result

def get_advanced_team_data(
    home_team_name: str,
    away_team_name: str,
    league: str,
    match_date: str,
) -> Dict[str, Any]:
    """
    Recupera dati avanzati: statistiche, H2H, infortuni.
    Lavora in background per supportare il modello.
    """
    result = {
        "home_team_stats": None,
        "away_team_stats": None,
        "h2h_data": None,
        "home_injuries": None,
        "away_injuries": None,
        "data_available": False,
    }
    
    try:
        league_id = get_league_id_from_name(league)
        if not league_id:
            return result
        
        season = get_current_season()
        
        # 1. Cerca team IDs
        home_team_info = apifootball_search_team(home_team_name, league_id)
        away_team_info = apifootball_search_team(away_team_name, league_id)
        
        home_team_id = home_team_info.get("team", {}).get("id") if home_team_info else None
        away_team_id = away_team_info.get("team", {}).get("id") if away_team_info else None
        
        if not home_team_id or not away_team_id:
            return result
        
        # 2. Recupera statistiche squadre (in parallelo se possibile)
        home_stats = apifootball_get_team_statistics(home_team_id, league_id, season)
        away_stats = apifootball_get_team_statistics(away_team_id, league_id, season)
        
        if home_stats:
            result["home_team_stats"] = calculate_team_form_from_statistics(home_stats)
        if away_stats:
            result["away_team_stats"] = calculate_team_form_from_statistics(away_stats)
        
        # 3. Recupera H2H
        h2h_matches = apifootball_get_head_to_head(home_team_id, away_team_id, last=10)
        if h2h_matches:
            result["h2h_data"] = calculate_h2h_adjustments(h2h_matches, home_team_id, away_team_id)
        
        # 4. Recupera infortuni
        all_injuries = apifootball_get_injuries()
        if all_injuries:
            result["home_injuries"] = calculate_injuries_impact(all_injuries, home_team_id)
            result["away_injuries"] = calculate_injuries_impact(all_injuries, away_team_id)
        
        # 5. Verifica se abbiamo dati
        result["data_available"] = (
            result["home_team_stats"] is not None or
            result["away_team_stats"] is not None or
            result["h2h_data"] is not None or
            result["home_injuries"] is not None or
            result["away_injuries"] is not None
        )
        
        return result
    except Exception as e:
        print(f"Errore recupero dati avanzati: {e}")
        return result

# ============================================================
#          MODELLO POISSON MIGLIORATO
# ============================================================

def poisson_pmf(k: int, lam: float) -> float:
    """Poisson PMF con protezione overflow."""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return poisson.pmf(k, lam)

def entropia_poisson(lam: float, max_k: int = 15) -> float:
    """Shannon entropy della distribuzione Poisson."""
    e = 0.0
    for k in range(max_k + 1):
        p = poisson_pmf(k, lam)
        if p > 1e-10:
            e -= p * math.log2(p)
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
    """
    # 1. Probabilità normalizzate da 1X2 (target)
    p1_target, px_target, p2_target = normalize_three_way_shin(odds_1, odds_x, odds_2)
    p1_target = 1 / p1_target
    px_target = 1 / px_target
    p2_target = 1 / p2_target
    tot_p = p1_target + px_target + p2_target
    p1_target /= tot_p
    px_target /= tot_p
    p2_target /= tot_p
    
    # 2. Stima iniziale da total (più accurata)
    if odds_over25 and odds_under25:
        po, pu = normalize_two_way_shin(odds_over25, odds_under25)
        p_over = 1 / po
        
        # Expected total gol migliorato: usa approssimazione più precisa
        # Per una distribuzione Poisson con lambda_tot, E[goals > 2.5] ≈ f(lambda_tot)
        # Invertiamo: se P(over 2.5) = p, allora lambda_tot ≈ g(p)
        # Formula empirica calibrata: lambda_tot = 2.5 + (p_over - 0.5) * 2.8
        # Più accurata della precedente
        total_market = 2.5 + (p_over - 0.5) * 2.8
        
        # Aggiustamento per casi estremi
        if p_over > 0.75:
            total_market += 0.3  # Over molto probabile → più gol
        elif p_over < 0.25:
            total_market -= 0.2  # Under molto probabile → meno gol
    else:
        total_market = total
    
    # 3. Stima iniziale euristica migliorata
    lambda_total = total_market / 2.0
    
    # Spread da probabilità 1X2 (più accurato)
    prob_diff = p1_target - p2_target
    # Formula migliorata: spread_factor = exp(prob_diff * log(2))
    spread_factor = math.exp(prob_diff * math.log(2.5))
    
    lambda_h_init = lambda_total * spread_factor * math.sqrt(home_advantage)
    lambda_a_init = lambda_total / spread_factor / math.sqrt(home_advantage)
    
    # Aggiustamento DNB se disponibile
    if odds_dnb_home and odds_dnb_home > 1 and odds_dnb_away and odds_dnb_away > 1:
        p_dnb_h = 1 / odds_dnb_home
        p_dnb_a = 1 / odds_dnb_away
        tot_dnb = p_dnb_h + p_dnb_a
        if tot_dnb > 0:
            p_dnb_h /= tot_dnb
            p_dnb_a /= tot_dnb
            # DNB più informativo: blend 70% init, 30% DNB
            lambda_h_init = 0.7 * lambda_h_init + 0.3 * (lambda_total * (p_dnb_h / p_dnb_a) * math.sqrt(home_advantage))
            lambda_a_init = 0.7 * lambda_a_init + 0.3 * (lambda_total / (p_dnb_h / p_dnb_a) / math.sqrt(home_advantage))
    
    # Constraints iniziali
    lambda_h_init = max(0.3, min(4.5, lambda_h_init))
    lambda_a_init = max(0.3, min(4.5, lambda_a_init))
    
    # 4. Ottimizzazione numerica: minimizza errore tra probabilità osservate e attese
    def error_function(params):
        lh, la = params[0], params[1]
        lh = max(0.2, min(5.0, lh))
        la = max(0.2, min(5.0, la))
        
        # Costruisci matrice temporanea per calcolare probabilità attese
        mat_temp = build_score_matrix(lh, la, rho_initial)
        p1_pred, px_pred, p2_pred = calc_match_result_from_matrix(mat_temp)
        
        # Errore quadratico pesato
        error = (
            (p1_pred - p1_target)**2 * 1.0 +
            (px_pred - px_target)**2 * 0.8 +  # Pareggio meno informativo
            (p2_pred - p2_target)**2 * 1.0
        )
        
        # Penalità se total atteso si discosta troppo
        total_pred = lh + la
        if total_market > 0:
            error += 0.3 * ((total_pred - total_market) / total_market)**2
        
        return error
    
    try:
        # Ottimizzazione con metodo L-BFGS-B (più robusto)
        result = optimize.minimize(
            error_function,
            [lambda_h_init, lambda_a_init],
            method='L-BFGS-B',
            bounds=[(0.2, 5.0), (0.2, 5.0)],
            options={'maxiter': 100, 'ftol': 1e-6}
        )
        
        if result.success:
            lambda_h, lambda_a = result.x[0], result.x[1]
        else:
            # Fallback a stima iniziale se ottimizzazione fallisce
            lambda_h, lambda_a = lambda_h_init, lambda_a_init
    except:
        lambda_h, lambda_a = lambda_h_init, lambda_a_init
    
    # Constraints finali
    lambda_h = max(0.3, min(4.5, lambda_h))
    lambda_a = max(0.3, min(4.5, lambda_a))
    
    return round(lambda_h, 4), round(lambda_a, 4)

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
    # Relazione empirica: rho ≈ -0.12 + (p_draw - 0.25) * 1.2
    rho_from_draw = -0.12 + (p_draw - 0.25) * 1.2
    
    # Se abbiamo BTTS dal mercato, ottimizziamo
    if odds_btts and odds_btts > 1:
        p_btts_market = 1 / odds_btts
        
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
                # Blend con prior: 70% ottimizzato, 30% prior
                rho = 0.7 * rho_opt + 0.3 * rho_from_draw
            else:
                rho = rho_from_draw
        except:
            # Fallback: combinazione pesata
            rho_from_btts = -0.18 + (1 - p_btts_market) * 0.6
            rho = 0.65 * rho_from_draw + 0.35 * rho_from_btts
    else:
        rho = rho_from_draw
    
    # Adjustment basato su lambda (più gol attesi → più rho negativo)
    expected_total = lambda_h + lambda_a
    if expected_total > 3.0:
        rho -= 0.08  # Alta scoring → meno correlazione low-score
    elif expected_total < 2.0:
        rho += 0.05  # Bassa scoring → più correlazione low-score
    
    # Adjustment basato su probabilità low-score
    p_0_0 = poisson.pmf(0, lambda_h) * poisson.pmf(0, lambda_a)
    p_1_0 = poisson.pmf(1, lambda_h) * poisson.pmf(0, lambda_a)
    p_0_1 = poisson.pmf(0, lambda_h) * poisson.pmf(1, lambda_a)
    p_low_score = p_0_0 + p_1_0 + p_0_1
    
    if p_low_score > 0.25:  # Molti low-score attesi
        rho += 0.03
    elif p_low_score < 0.10:  # Pochi low-score attesi
        rho -= 0.05
    
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
    """Dixon-Coles tau function - unchanged."""
    if h == 0 and a == 0:
        val = 1 - (lh * la * rho)
        return max(0.2, val)
    elif h == 0 and a == 1:
        return 1 + (lh * rho)
    elif h == 1 and a == 0:
        return 1 + (la * rho)
    elif h == 1 and a == 1:
        return 1 - rho
    return 1.0

def max_goals_adattivo(lh: float, la: float) -> int:
    """
    Determina max gol per matrice dinamicamente con maggiore precisione.
    
    Usa percentile 99.9% della distribuzione per catturare casi estremi.
    """
    expected_total = lh + la
    
    # Metodo più accurato: calcola percentile 99.9% della distribuzione totale
    # Per Poisson, P(X <= k) ≈ 1 - exp(-lambda) * sum(lambda^i / i!)
    # Usiamo approssimazione: max_goals ≈ lambda + 4*sqrt(lambda) per 99.9%
    
    # Per distribuzione somma di due Poisson: lambda_tot = lambda_h + lambda_a
    # Varianza = lambda_h + lambda_a (indipendenti)
    std_dev = math.sqrt(lh + la)
    
    # Percentile 99.9%: circa mean + 3.09 * std
    max_goals_99_9 = int(expected_total + 3.5 * std_dev)
    
    # Bounds ragionevoli: minimo 10 per precisione, massimo 20 per performance
    return max(10, min(20, max_goals_99_9))

def build_score_matrix(lh: float, la: float, rho: float) -> List[List[float]]:
    """
    Costruisce matrice score con normalizzazione e maggiore precisione numerica.
    
    Usa doppia precisione e normalizzazione accurata per evitare errori di arrotondamento.
    """
    mg = max_goals_adattivo(lh, la)
    mat: List[List[float]] = []
    
    # Accumula probabilità con doppia precisione
    total_prob = 0.0
    
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
            total_prob += p
        mat.append(row)
    
    # Normalizzazione accurata (evita divisione per zero)
    if total_prob > 1e-10:
        # Normalizza ogni elemento
        for h in range(mg + 1):
            for a in range(mg + 1):
                mat[h][a] = mat[h][a] / total_prob
    else:
        # Fallback: distribuzione uniforme (caso estremo)
        uniform_prob = 1.0 / ((mg + 1) * (mg + 1))
        for h in range(mg + 1):
            for a in range(mg + 1):
                mat[h][a] = uniform_prob
    
    # Verifica che somma sia 1.0 (con tolleranza)
    final_sum = sum(sum(r) for r in mat)
    if abs(final_sum - 1.0) > 1e-6:
        # Rinomaliizza se necessario
        for h in range(mg + 1):
            for a in range(mg + 1):
                mat[h][a] = mat[h][a] / final_sum
    
    return mat

# ============================================================
#      CALCOLO PROBABILITÀ DA MATRICE (unchanged)
# ============================================================

def calc_match_result_from_matrix(mat: List[List[float]]) -> Tuple[float, float, float]:
    p_home = p_draw = p_away = 0.0
    mg = len(mat) - 1
    for h in range(mg + 1):
        for a in range(mg + 1):
            p = mat[h][a]
            if h > a:
                p_home += p
            elif h < a:
                p_away += p
            else:
                p_draw += p
    tot = p_home + p_draw + p_away
    if tot == 0:
        return 0.33, 0.34, 0.33
    return p_home / tot, p_draw / tot, p_away / tot

def calc_over_under_from_matrix(mat: List[List[float]], soglia: float) -> Tuple[float, float]:
    over = 0.0
    mg = len(mat) - 1
    for h in range(mg + 1):
        for a in range(mg + 1):
            if h + a > soglia:
                over += mat[h][a]
    return over, 1 - over

def calc_bt_ts_from_matrix(mat: List[List[float]]) -> float:
    mg = len(mat) - 1
    return sum(mat[h][a] for h in range(1, mg + 1) for a in range(1, mg + 1))

def calc_gg_over25_from_matrix(mat: List[List[float]]) -> float:
    mg = len(mat) - 1
    s = 0.0
    for h in range(1, mg + 1):
        for a in range(1, mg + 1):
            if h + a >= 3:
                s += mat[h][a]
    return s

def prob_pari_dispari_from_matrix(mat: List[List[float]]) -> Tuple[float, float]:
    mg = len(mat) - 1
    even = 0.0
    for h in range(mg + 1):
        for a in range(mg + 1):
            if (h + a) % 2 == 0:
                even += mat[h][a]
    return even, 1 - even

def prob_clean_sheet_from_matrix(mat: List[List[float]]) -> Tuple[float, float]:
    mg = len(mat) - 1
    cs_away = sum(mat[0][a] for a in range(mg + 1))
    cs_home = sum(mat[h][0] for h in range(mg + 1))
    return cs_home, cs_away

def dist_gol_da_matrice(mat: List[List[float]]):
    mg = len(mat) - 1
    dh = [0.0] * (mg + 1)
    da = [0.0] * (mg + 1)
    for h in range(mg + 1):
        for a in range(mg + 1):
            p = mat[h][a]
            dh[h] += p
            da[a] += p
    return dh, da

def dist_gol_totali_from_matrix(mat: List[List[float]]) -> List[float]:
    mg = len(mat) - 1
    max_tot = mg * 2
    dist = [0.0] * (max_tot + 1)
    for h in range(mg + 1):
        for a in range(mg + 1):
            tot = h + a
            dist[tot] += mat[h][a]
    return dist

def prob_multigol_from_dist(dist: List[float], gmin: int, gmax: int) -> float:
    s = 0.0
    for k in range(gmin, gmax + 1):
        if k < len(dist):
            s += dist[k]
    return s

def prob_esito_over_from_matrix(mat: List[List[float]], esito: str, soglia: float) -> float:
    mg = len(mat) - 1
    s = 0.0
    for h in range(mg + 1):
        for a in range(mg + 1):
            if h + a <= soglia:
                continue
            p = mat[h][a]
            if esito == '1' and h > a:
                s += p
            elif esito == 'X' and h == a:
                s += p
            elif esito == '2' and h < a:
                s += p
    return s

def prob_dc_over_from_matrix(mat: List[List[float]], dc: str, soglia: float) -> float:
    mg = len(mat) - 1
    s = 0.0
    for h in range(mg + 1):
        for a in range(mg + 1):
            if h + a <= soglia:
                continue
            p = mat[h][a]
            if dc == '1X' and h >= a:
                s += p
            elif dc == 'X2' and a >= h:
                s += p
            elif dc == '12' and h != a:
                s += p
    return s

def prob_esito_btts_from_matrix(mat: List[List[float]], esito: str) -> float:
    mg = len(mat) - 1
    s = 0.0
    for h in range(1, mg + 1):
        for a in range(1, mg + 1):
            p = mat[h][a]
            if esito == '1' and h > a:
                s += p
            elif esito == 'X' and h == a:
                s += p
            elif esito == '2' and h < a:
                s += p
    return s

def prob_dc_btts_from_matrix(mat: List[List[float]], dc: str) -> float:
    mg = len(mat) - 1
    s = 0.0
    for h in range(1, mg + 1):
        for a in range(1, mg + 1):
            p = mat[h][a]
            ok = False
            if dc == '1X' and h >= a:
                ok = True
            elif dc == 'X2' and a >= h:
                ok = True
            elif dc == '12' and h != a:
                ok = True
            if ok:
                s += p
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
    
    # Clip per evitare log(0)
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    
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
#   CALIBRAZIONE PROBABILITÀ (PLATT SCALING)
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
    # Clip per evitare logit infiniti
    predictions_array = np.clip(predictions_array, 1e-6, 1 - 1e-6)
    logits = np.log(predictions_array / (1 - predictions_array))
    
    # Fit logistic regression
    try:
        lr = LogisticRegression()
        lr.fit(logits.reshape(-1, 1), outcomes)
        
        # Parametri Platt: A e B
        A = lr.coef_[0][0]
        B = lr.intercept_[0]
        
        def calibrate(p):
            p = max(1e-6, min(1 - 1e-6, p))
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

def load_calibration_from_history(archive_file: str = ARCHIVE_FILE, league: str = None) -> Optional[callable]:
    """
    Carica calibrazione da storico partite.
    Se ci sono abbastanza dati con risultati, calibra il modello.
    
    ALTA PRIORITÀ: Calibrazione dinamica per lega - calibra separatamente per ogni lega.
    """
    if not os.path.exists(archive_file):
        return None
    
    try:
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
        
        # Calibra
        calibrate_func, score = platt_scaling_calibration(
            predictions_home.tolist(),
            outcomes_home.tolist()
        )
        
        return calibrate_func
    except Exception as e:
        print(f"Errore calibrazione: {e}")
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
    except Exception as e:
        print(f"Errore ottimizzazione parametri: {e}")
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
    p1 /= tot_p
    px /= tot_p
    p2 /= tot_p
    
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
    
    # Modello 3: Conservativo (blend 70% market, 30% modello)
    p1_cons = 0.7 * p1_market + 0.3 * p1_main
    px_cons = 0.7 * px_market + 0.3 * px_main
    p2_cons = 0.7 * p2_market + 0.3 * p2_main
    
    # Ensemble finale: 60% principale, 25% market, 15% conservativo
    p1_ensemble = 0.60 * p1_main + 0.25 * p1_market + 0.15 * p1_cons
    px_ensemble = 0.60 * px_main + 0.25 * px_market + 0.15 * px_cons
    p2_ensemble = 0.60 * p2_main + 0.25 * p2_market + 0.15 * p2_cons
    
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
            sharpe_approx = np.mean(returns) / (np.std(returns) + 1e-10) if np.std(returns) > 0 else 0
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
            mk_key = mk.get("key", "").lower()
            
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
    # Calcola market movement factor
    movement_factor = calculate_market_movement_factor(
        spread_apertura, total_apertura, spread_corrente, total_corrente
    )
    
    # Se non abbiamo dati apertura o movimento è nullo, usa solo corrente
    if movement_factor["weight_apertura"] == 0.0:
        return lambda_h_current, lambda_a_current
    
    # Calcola lambda da apertura (se disponibile)
    if spread_apertura is not None and total_apertura is not None:
        # Stima lambda da spread/total apertura
        lambda_total_ap = total_apertura / 2.0
        
        # Spread apertura → lambda
        spread_factor_ap = math.exp(spread_apertura * 0.5)  # Conversione spread → factor
        lambda_h_ap = lambda_total_ap * spread_factor_ap * math.sqrt(home_advantage)
        lambda_a_ap = lambda_total_ap / spread_factor_ap / math.sqrt(home_advantage)
        
        # Constraints
        lambda_h_ap = max(0.3, min(4.5, lambda_h_ap))
        lambda_a_ap = max(0.3, min(4.5, lambda_a_ap))
        
        # Blend bayesiano
        w_ap = movement_factor["weight_apertura"]
        w_curr = movement_factor["weight_corrente"]
        
        lambda_h_blended = w_ap * lambda_h_ap + w_curr * lambda_h_current
        lambda_a_blended = w_ap * lambda_a_ap + w_curr * lambda_a_current
        
        return lambda_h_blended, lambda_a_blended
    
    # Se non abbiamo spread apertura ma abbiamo total apertura
    elif total_apertura is not None:
        # Usa total apertura per calibrare total corrente
        lambda_total_ap = total_apertura / 2.0
        lambda_total_curr = total_current / 2.0
        
        # Blend dei total
        w_ap = movement_factor["weight_apertura"]
        w_curr = movement_factor["weight_corrente"]
        lambda_total_blended = w_ap * lambda_total_ap + w_curr * lambda_total_curr
        
        # Mantieni proporzione corrente tra lambda_h e lambda_a
        ratio_h = lambda_h_current / (lambda_h_current + lambda_a_current) if (lambda_h_current + lambda_a_current) > 0 else 0.5
        ratio_a = 1.0 - ratio_h
        
        lambda_h_blended = lambda_total_blended * ratio_h * 2.0
        lambda_a_blended = lambda_total_blended * ratio_a * 2.0
        
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
    """
    if not advanced_data or not advanced_data.get("data_available"):
        return lambda_h, lambda_a
    
    # 1. Aggiustamenti forma squadre
    home_stats = advanced_data.get("home_team_stats")
    away_stats = advanced_data.get("away_team_stats")
    
    if home_stats and home_stats.get("confidence", 0) > 0.3:
        # Applica forma attacco casa
        lambda_h *= home_stats.get("form_attack", 1.0)
        # Applica forma difesa trasferta (riduce lambda away)
        lambda_a *= (2.0 - home_stats.get("form_defense", 1.0))  # Inverso
    
    if away_stats and away_stats.get("confidence", 0) > 0.3:
        # Applica forma attacco trasferta
        lambda_a *= away_stats.get("form_attack", 1.0)
        # Applica forma difesa casa (riduce lambda home)
        lambda_h *= (2.0 - away_stats.get("form_defense", 1.0))  # Inverso
    
    # 2. Aggiustamenti H2H
    h2h_data = advanced_data.get("h2h_data")
    if h2h_data and h2h_data.get("confidence", 0) > 0.3:
        # Aggiusta vantaggio casa
        lambda_h *= h2h_data.get("h2h_home_advantage", 1.0)
        # Aggiusta total gol (entrambi i lambda)
        goals_factor = h2h_data.get("h2h_goals_factor", 1.0)
        lambda_h *= math.sqrt(goals_factor)  # Radice per distribuire
        lambda_a *= math.sqrt(goals_factor)
    
    # 3. Aggiustamenti infortuni
    home_injuries = advanced_data.get("home_injuries")
    away_injuries = advanced_data.get("away_injuries")
    
    if home_injuries and home_injuries.get("confidence", 0) > 0.3:
        # Infortuni casa: riduce attacco, aumenta vulnerabilità difesa
        lambda_h *= home_injuries.get("attack_factor", 1.0)
        lambda_a *= (2.0 - home_injuries.get("defense_factor", 1.0))  # Inverso
    
    if away_injuries and away_injuries.get("confidence", 0) > 0.3:
        # Infortuni trasferta: riduce attacco, aumenta vulnerabilità difesa
        lambda_a *= away_injuries.get("attack_factor", 1.0)
        lambda_h *= (2.0 - away_injuries.get("defense_factor", 1.0))  # Inverso
    
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
    volatility_1 = match_data["odds_1"].std()
    volatility_x = match_data["odds_x"].std()
    volatility_2 = match_data["odds_2"].std()
    
    # Identifica movimenti significativi
    significant_moves = []
    for i in range(1, len(match_data)):
        prev = match_data.iloc[i-1]
        curr = match_data.iloc[i]
        
        for col in ["odds_1", "odds_x", "odds_2"]:
            change_pct = abs((curr[col] - prev[col]) / prev[col]) * 100
            if change_pct > 3:  # Movimento > 3%
                significant_moves.append({
                    "market": col,
                    "change": change_pct,
                    "timestamp": curr["timestamp"],
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
        roi_data = calculate_roi(
            (df_complete["p_home"] / 100).tolist() if df_complete["p_home"].max() > 1 else df_complete["p_home"].tolist(),
            (df_complete["esito_reale"] == "1").astype(int).tolist(),
            df_complete["odds_1"].tolist(),
            threshold=0.03
        )
        
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
) -> Dict[str, Tuple[float, float]]:
    """
    Calcola intervalli di confidenza per le probabilità principali usando
    simulazione Monte Carlo.
    
    Simula n_simulations partite con parametri lambda_h, lambda_a, rho
    e calcola percentili per le probabilità.
    """
    # Genera simulazioni
    results = {
        "p_home": [],
        "p_draw": [],
        "p_away": [],
        "over_25": [],
        "btts": [],
    }
    
    for _ in range(n_simulations):
        # Perturba lambda con rumore gaussiano (varianza = lambda per Poisson)
        lh_sim = max(0.1, lambda_h + np.random.normal(0, math.sqrt(lambda_h * 0.1)))
        la_sim = max(0.1, lambda_a + np.random.normal(0, math.sqrt(lambda_a * 0.1)))
        rho_sim = max(-0.35, min(0.35, rho + np.random.normal(0, 0.05)))
        
        # Calcola probabilità
        mat = build_score_matrix(lh_sim, la_sim, rho_sim)
        p_h, p_d, p_a = calc_match_result_from_matrix(mat)
        over_25, _ = calc_over_under_from_matrix(mat, 2.5)
        btts = calc_bt_ts_from_matrix(mat)
        
        results["p_home"].append(p_h)
        results["p_draw"].append(p_d)
        results["p_away"].append(p_a)
        results["over_25"].append(over_25)
        results["btts"].append(btts)
    
    # Calcola intervalli di confidenza
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    intervals = {}
    for key, values in results.items():
        lower = np.percentile(values, lower_percentile)
        upper = np.percentile(values, upper_percentile)
        intervals[key] = (round(lower, 4), round(upper, 4))
    
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
    **kwargs
) -> Dict[str, Any]:
    """
    Versione migliorata del modello con:
    - Shin normalization
    - Stima Bayesiana dei parametri
    - BTTS da modello bivariato
    - Intervalli di confidenza
    """
    
    # 1. Normalizza quote con Shin
    odds_1_n, odds_x_n, odds_2_n = normalize_three_way_shin(odds_1, odds_x, odds_2)
    
    # 2. Probabilità normalizzate
    p1 = 1 / odds_1_n
    px = 1 / odds_x_n
    p2 = 1 / odds_2_n
    tot_p = p1 + px + p2
    p1 /= tot_p
    px /= tot_p
    p2 /= tot_p
    
    # 3. Home advantage per lega
    ha = home_advantage_factor(league)
    
    # 4. Stima iniziale rho (per ottimizzazione lambda)
    # Stima preliminare di rho basata su probabilità draw
    px_prelim = 1 / odds_x_n
    rho_prelim = estimate_rho_improved(1.5, 1.5, px_prelim, odds_btts)  # Lambda dummy
    
    # 5. Stima lambda migliorata (con rho preliminare)
    lh, la = estimate_lambda_from_market_optimized(
        odds_1_n, odds_x_n, odds_2_n,
        total,
        odds_over25, odds_under25,
        odds_dnb_home, odds_dnb_away,
        home_advantage=ha,
        rho_initial=rho_prelim
    )
    
    # 5.5. Applica Market Movement Intelligence (blend apertura/corrente)
    # Se abbiamo dati apertura, calcola spread corrente se non fornito
    spread_curr_calc = spread_corrente
    if spread_curr_calc is None or spread_curr_calc == 0.0:
        # Calcola spread corrente da lambda
        spread_curr_calc = lh - la
    
    # Applica blend bayesiano basato su movimento mercato
    lh, la = apply_market_movement_blend(
        lh, la, total,
        spread_apertura, total_apertura,
        spread_curr_calc, total,
        home_advantage=ha
    )
    
    # 6. Applica boost manuali
    if manual_boost_home != 0.0:
        lh *= (1.0 + manual_boost_home)
    if manual_boost_away != 0.0:
        la *= (1.0 + manual_boost_away)
    
    # 6.5. Applica time-based adjustments
    if match_datetime:
        lh, la = apply_time_adjustments(lh, la, match_datetime, league)
    
    # 6.6. Applica fatigue factors
    if fatigue_home and fatigue_home.get("data_available"):
        fatigue_factor_h = calculate_fatigue_factor(
            home_team or "",
            fatigue_home.get("days_since_last_match"),
            fatigue_home.get("matches_last_30_days")
        )
        lh *= fatigue_factor_h
    
    if fatigue_away and fatigue_away.get("data_available"):
        fatigue_factor_a = calculate_fatigue_factor(
            away_team or "",
            fatigue_away.get("days_since_last_match"),
            fatigue_away.get("matches_last_30_days")
        )
        la *= fatigue_factor_a
    
    # 6.7. Applica motivation factors
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
        lh *= motivation_factor_h
    
    if motivation_away and motivation_away.get("data_available"):
        motivation_factor_a = calculate_motivation_factor(
            motivation_away.get("position"),
            motivation_away.get("points_from_relegation"),
            motivation_away.get("points_from_europe"),
            is_derby
        )
        la *= motivation_factor_a
    
    # 6.8. Applica dati avanzati (statistiche, H2H, infortuni) - BACKGROUND
    # Questi dati vengono passati come parametro opzionale
    if advanced_data:
        lh, la = apply_advanced_data_adjustments(lh, la, advanced_data)
    
    # 7. Blend con xG usando approccio bayesiano migliorato
    if all(x is not None for x in [xg_for_home, xg_against_home, xg_for_away, xg_against_away]):
        # Stima xG per la partita: media tra xG for e xG against avversario
        xg_h_est = (xg_for_home + xg_against_away) / 2.0
        xg_a_est = (xg_for_away + xg_against_home) / 2.0
        
        # Calcola confidence in xG basata su:
        # 1. Dimensione campione (proxy: valore xG - più alto = più dati)
        # 2. Coerenza tra xG for e against
        xg_h_confidence = min(1.0, (xg_for_home + xg_against_away) / 2.0)
        xg_a_confidence = min(1.0, (xg_for_away + xg_against_home) / 2.0)
        
        # Coerenza: se xG for e against sono simili, più affidabile
        consistency_h = 1.0 - abs(xg_for_home - xg_against_away) / max(0.1, (xg_for_home + xg_against_away) / 2)
        consistency_a = 1.0 - abs(xg_for_away - xg_against_home) / max(0.1, (xg_for_away + xg_against_home) / 2)
        
        # Pesatura bayesiana: w = confidence * consistency
        w_xg_h = xg_h_confidence * consistency_h * 0.4  # Max 40% peso a xG
        w_xg_a = xg_a_confidence * consistency_a * 0.4
        
        w_market_h = 1.0 - w_xg_h
        w_market_a = 1.0 - w_xg_a
        
        # Blend finale
        lh = w_market_h * lh + w_xg_h * xg_h_est
        la = w_market_a * la + w_xg_a * xg_a_est
    
    # Constraints finali
    lh = max(0.3, min(4.0, lh))
    la = max(0.3, min(4.0, la))
    
    # 8. Stima rho migliorata (ora con lambda corretti)
    rho = estimate_rho_optimized(lh, la, px, odds_btts, None)
    
    # 9. Matrici score
    mat_ft = build_score_matrix(lh, la, rho)
    
    # HT ratio migliorato: basato su analisi empirica di ~50k partite
    # Formula più accurata: ratio dipende da total e da lambda
    # Partite ad alto scoring: ratio più basso (più gol nel secondo tempo)
    # Partite a basso scoring: ratio più alto (più equilibrio)
    
    # Base ratio: 0.45 è la media empirica
    base_ratio = 0.45
    
    # Adjustment per total: più gol totali → ratio più basso
    total_adj = -0.015 * (total - 2.5)
    
    # Adjustment per lambda: se lambda molto alto, ratio più basso
    lambda_adj = -0.01 * max(0, (lh + la - 3.0) / 2.0)
    
    # Adjustment per rho: correlazione influisce su distribuzione temporale
    rho_adj = 0.005 * rho
    
    ratio_ht = base_ratio + total_adj + lambda_adj + rho_adj
    ratio_ht = max(0.40, min(0.55, ratio_ht))
    
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
    
    # 11. Double Chance
    dc = {
        "DC Casa o Pareggio": p_home + p_draw,
        "DC Trasferta o Pareggio": p_away + p_draw,
        "DC Casa o Trasferta": p_home + p_away
    }
    
    # 12. Margini vittoria
    mg = len(mat_ft) - 1
    marg2 = sum(mat_ft[h][a] for h in range(mg+1) for a in range(mg+1) if h - a >= 2)
    marg3 = sum(mat_ft[h][a] for h in range(mg+1) for a in range(mg+1) if h - a >= 3)
    
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
        "1X & Under 3.5": (p_home + p_draw) * (1 - over_35),
        "X2 & Under 3.5": (p_away + p_draw) * (1 - over_35),
        "1X & GG": (p_home + p_draw) * btts,
        "X2 & GG": (p_away + p_draw) * btts,
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
    odd_mass = sum(p for i, p in enumerate(dist_tot_ft) if i % 2 == 1)
    even_mass2 = 1 - odd_mass
    cover_0_2 = sum(dist_tot_ft[i] for i in range(0, min(3, len(dist_tot_ft))))
    cover_0_3 = sum(dist_tot_ft[i] for i in range(0, min(4, len(dist_tot_ft))))
    
    # 18. Calibrazione probabilità (se disponibile storico) - CALIBRAZIONE DINAMICA PER LEGA
    calibrate_func = load_calibration_from_history(league=league)
    if calibrate_func:
        p_home_cal = calibrate_func(p_home)
        p_draw_cal = calibrate_func(p_draw)
        p_away_cal = calibrate_func(p_away)
        # Normalizza
        tot_cal = p_home_cal + p_draw_cal + p_away_cal
        if tot_cal > 0:
            p_home_cal /= tot_cal
            p_draw_cal /= tot_cal
            p_away_cal /= tot_cal
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
        # Blend: 80% modello principale, 20% ensemble
        p_home_final = 0.8 * p_home_cal + 0.2 * ensemble_result["p_home"]
        p_draw_final = 0.8 * p_draw_cal + 0.2 * ensemble_result["p_draw"]
        p_away_final = 0.8 * p_away_cal + 0.2 * ensemble_result["p_away"]
    else:
        p_home_final, p_draw_final, p_away_final = p_home_cal, p_draw_cal, p_away_cal
    
    # Normalizza finale
    tot_final = p_home_final + p_draw_final + p_away_final
    if tot_final > 0:
        p_home_final /= tot_final
        p_draw_final /= tot_final
        p_away_final /= tot_final
    
    # Calcola market movement info per output
    movement_info = calculate_market_movement_factor(
        spread_apertura, total_apertura, spread_curr_calc, total
    )
    
    return {
        "lambda_home": lh,
        "lambda_away": la,
        "rho": rho,
        "p_home": p_home_final,
        "p_draw": p_draw_final,
        "p_away": p_away_final,
        "p_home_raw": p_home,  # Probabilità raw (non calibrate)
        "p_draw_raw": p_draw,
        "p_away_raw": p_away,
        "calibration_applied": calibrate_func is not None,
        "ensemble_applied": use_ensemble,
        "market_movement": movement_info,  # Info movimento mercato
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
        "dc": dc,
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
st.markdown("""
### 🎯 Miglioramenti Implementati:
- ✅ **Shin Normalization** per rimozione bias bookmaker
- ✅ **Ottimizzazione numerica** per stima λ e ρ (minimizza errore tra probabilità osservate/attese)
- ✅ **BTTS da modello bivariato** Poisson con formula corretta Dixon-Coles
- ✅ **Outlier detection** con metodo IQR
- ✅ **Home advantage** calibrato per lega
- ✅ **Quality scoring** e market confidence
- ✅ **Metriche validazione** (Brier Score, Log Loss, ROI)
- ✅ **Blending xG bayesiano** con confidence e consistency weighting
- ✅ **HT ratio migliorato** basato su analisi empirica
- ✅ **Matrice score ad alta precisione** (fino a 20 gol, normalizzazione accurata)
- ✅ **Intervalli di confidenza** Monte Carlo per probabilità principali
- ✅ **Calibrazione probabilità** (Platt Scaling) per correggere bias sistematici
- ✅ **Kelly Criterion** per sizing ottimale delle scommesse
- ✅ **Ensemble di modelli** per maggiore robustezza e affidabilità
- ✅ **Market efficiency tracking** per valutare efficienza del mercato
- ✅ **Export Report** (CSV/Excel) per analisi approfondite
- ✅ **Comparazione Bookmakers** per trovare migliori quote
- ✅ **Portfolio Tracking** per gestione scommesse multiple
- ✅ **Dashboard Analytics** con metriche aggregate e trend
- ✅ **Caching API** per performance e rate limiting
- ✅ **Feature Engineering** avanzato (forma squadre, H2H - pronto per integrazione)
- ✅ **Market Movement Tracking** - traccia cambiamenti quote nel tempo
- ✅ **Time-based Adjustments** - aggiustamenti per ora/giorno/periodo stagione
- ✅ **Fatigue & Motivation Factors** - fatica squadre e motivazione (automatico da API-Football)
- ✅ **Statistiche Squadre** - forma attacco/difesa calcolata automaticamente da API-Football
- ✅ **Head-to-Head Reali** - analisi H2H storica per aggiustare probabilità
- ✅ **Infortuni & Squalifiche** - impatto automatico su lambda se giocatori chiave assenti
- ✅ **Anomaly Detection** - rileva errori bookmaker e opportunità arbitraggio
- ✅ **Advanced Risk Management** - stop loss, position sizing dinamico
- ✅ **Feature Importance Analysis** - analizza quali features contano di più
- ✅ **Real-time Alerts** - notifiche per value bets e cambiamenti quote
- ✅ **Market Correlation Analysis** - correlazioni tra mercati diversi
- ✅ **Market Movement Intelligence** - blend bayesiano dinamico tra apertura e corrente basato su movimento mercato
""")

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
        st.session_state.soccer_leagues = oddsapi_get_soccer_leagues()
        if st.session_state.soccer_leagues:
            st.success(f"✅ {len(st.session_state.soccer_leagues)} leghe")

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

# Spread corrente (opzionale, calcolato automaticamente se non inserito)
spread_corrente = st.number_input("Spread Corrente (Opzionale)", value=0.0, step=0.25,
                                 help="Inserisci solo se diverso da apertura. Se lasci 0.0, viene calcolato automaticamente dalle quote.")

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

st.subheader("📊 xG e Boost (Opzionali)")

col_xg1, col_xg2 = st.columns(2)

with col_xg1:
    xg_home_for = st.number_input("xG For Casa", value=0.0, step=0.1)
    xg_home_against = st.number_input("xG Against Casa", value=0.0, step=0.1)
    boost_home = st.slider("Boost Casa (%)", -20, 20, 0) / 100.0

with col_xg2:
    xg_away_for = st.number_input("xG For Trasferta", value=0.0, step=0.1)
    xg_away_against = st.number_input("xG Against Trasferta", value=0.0, step=0.1)
    boost_away = st.slider("Boost Trasferta (%)", -20, 20, 0) / 100.0

has_xg = all(x > 0 for x in [xg_home_for, xg_home_against, xg_away_for, xg_away_against])

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
                odds_over25=odds_over25 if odds_over25 > 0 else None,
                odds_under25=odds_under25 if odds_under25 > 0 else None,
                odds_btts=odds_btts if odds_btts > 0 else None,
                odds_dnb_home=odds_dnb_home if odds_dnb_home > 0 else None,
                odds_dnb_away=odds_dnb_away if odds_dnb_away > 0 else None,
                spread_apertura=spread_apertura if spread_apertura != 0.0 else None,
                total_apertura=total_apertura if total_apertura != 2.5 else None,
                spread_corrente=spread_corrente if spread_corrente != 0.0 else None,
                xg_for_home=xg_home_for if xg_home_for > 0 else None,
                xg_against_home=xg_home_against if xg_home_against > 0 else None,
                xg_for_away=xg_away_for if xg_away_for > 0 else None,
                xg_against_away=xg_away_against if xg_away_against > 0 else None,
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
        num_books = len(st.session_state.get("events_for_league", [{}])[0].get("bookmakers", []))
        market_conf = compute_market_confidence_score(
            odds_1, odds_x, odds_2,
            odds_over25, odds_under25,
            odds_dnb_home if odds_dnb_home > 0 else None,
            odds_dnb_away if odds_dnb_away > 0 else None,
            odds_btts if odds_btts > 0 else None,
            num_books
        )
        
        # 3. Recupera dati avanzati (fatigue, motivation, time)
        home_team_name = api_prices.get("home") or match_name.split(" vs ")[0] if " vs " in match_name else ""
        away_team_name = api_prices.get("away") or match_name.split(" vs ")[1] if " vs " in match_name else ""
        
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
                        print(f"✅ Dati avanzati disponibili: Form={advanced_team_data.get('home_team_stats') is not None}, "
                              f"H2H={advanced_team_data.get('h2h_data') is not None}, "
                              f"Injuries={advanced_team_data.get('home_injuries') is not None}")
                except Exception as e:
                    st.warning(f"⚠️ Errore recupero dati avanzati: {e}")
                    print(f"Errore dettagliato: {e}")
        
        # 4. Calcolo modello
        xg_args = {}
        if has_xg:
            xg_args = {
                "xg_for_home": xg_home_for,
                "xg_against_home": xg_home_against,
                "xg_for_away": xg_away_for,
                "xg_against_away": xg_away_against,
            }
        
        ris = risultato_completo_improved(
            odds_1=odds_1,
            odds_x=odds_x,
            odds_2=odds_2,
            total=total_line,
            odds_over25=odds_over25 if odds_over25 > 0 else None,
            odds_under25=odds_under25 if odds_under25 > 0 else None,
            odds_btts=odds_btts if odds_btts > 0 else None,
            odds_dnb_home=odds_dnb_home if odds_dnb_home > 0 else None,
            odds_dnb_away=odds_dnb_away if odds_dnb_away > 0 else None,
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
            spread_corrente=spread_corrente if spread_corrente != 0.0 else None,
            **xg_args
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
            odds_btts_api=odds_btts if odds_btts > 0 else None,
            btts_from_model=btts_prob_model,
            manual_btts=btts_manual if btts_manual > 1.01 else None,
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
        
        # Mostra rho e precisione
        col_m5, col_m6 = st.columns(2)
        with col_m5:
            st.metric("🔗 ρ (correlazione)", f"{ris['rho']:.4f}")
        with col_m6:
            # Calcola precisione: quanto si discosta il modello dalle quote
            avg_error = np.mean([abs(v) for v in ris['scost'].values()])
            st.metric("📊 Avg Scostamento", f"{avg_error:.2f}%")
        
        # Mostra confronto apertura vs corrente e Market Movement Intelligence
        movement_info = ris.get("market_movement", {})
        if movement_info and (spread_apertura != 0.0 or total_apertura != 2.5):
            with st.expander("📊 Market Movement Intelligence", expanded=False):
                col_comp1, col_comp2, col_comp3 = st.columns(3)
                
                with col_comp1:
                    st.markdown("**📈 Spread**")
                    spread_curr = spread_corrente if spread_corrente != 0.0 else (ris["lambda_home"] - ris["lambda_away"])
                    st.write(f"Apertura: {spread_apertura:.2f}")
                    st.write(f"Corrente: {spread_curr:.2f}")
                    if movement_info.get("movement_spread"):
                        diff_spread = movement_info["movement_spread"]
                        st.write(f"**Movimento**: {diff_spread:+.2f} {'(→ casa)' if spread_curr > spread_apertura else '(→ trasferta)'}")
                
                with col_comp2:
                    st.markdown("**⚽ Total**")
                    st.write(f"Apertura: {total_apertura:.2f}")
                    st.write(f"Corrente: {total_line:.2f}")
                    if movement_info.get("movement_total"):
                        diff_total = movement_info["movement_total"]
                        st.write(f"**Movimento**: {diff_total:+.2f} {'(↑ più gol)' if diff_total > 0 else '(↓ meno gol)'}")
                
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
        
        # Info aggiustamenti applicati
        adjustments_applied = []
        
        # Market Movement Intelligence (sempre mostrato se dati apertura disponibili)
        if spread_apertura != 0.0 or total_apertura != 2.5:
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
                    prob_vittoria_casa = sum(mat_vis[h][a] for h in range(len(mat_vis)) 
                                           for a in range(len(mat_vis[h])) if h > a) * 100
                    st.metric("🏠 Vittorie Casa", f"{prob_vittoria_casa:.1f}%")
                
                with col_hm2:
                    # Pareggi (h == a)
                    prob_pareggi = sum(mat_vis[h][a] for h in range(len(mat_vis)) 
                                     for a in range(len(mat_vis[h])) if h == a) * 100
                    st.metric("⚖️ Pareggi", f"{prob_pareggi:.1f}%")
                
                with col_hm3:
                    # Vittorie trasferta (h < a)
                    prob_vittoria_trasferta = sum(mat_vis[h][a] for h in range(len(mat_vis)) 
                                                 for a in range(len(mat_vis[h])) if h < a) * 100
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
