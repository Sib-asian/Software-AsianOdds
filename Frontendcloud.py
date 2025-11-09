import math
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np
import os
import requests
import streamlit as st
from scipy import optimize
from scipy.stats import poisson
import warnings
warnings.filterwarnings('ignore')

# ============================================================
#                 CONFIG
# ============================================================

# PINNACLE API (GRATIS E SHARP!)
PINNACLE_BASE = "https://guest.api.arcadia.pinnacle.com/0.1"

# API-FOOTBALL per risultati reali (GRATIS 100 req/day)
API_FOOTBALL_KEY = "95c43f936816cd4389a747fd2cfe061a"
API_FOOTBALL_BASE = "https://v3.football.api-sports.io"

ARCHIVE_FILE = "storico_analisi.csv"
VALIDATION_FILE = "validation_metrics.csv"

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
#   NORMALIZZAZIONE AVANZATA (SHIN METHOD)
# ============================================================

def shin_normalization(odds_list: List[float], max_iter: int = 100, tol: float = 1e-6) -> List[float]:
    """
    Shin method per rimuovere il margine considerando insider trading.
    Più robusto della normalizzazione proporzionale.
    """
    if not odds_list or any(o <= 1 for o in odds_list):
        return odds_list
    
    probs = np.array([1/o for o in odds_list])
    margin = probs.sum() - 1
    
    if margin <= 0:
        return odds_list
    
    def shin_equation(z):
        if z <= 0 or z >= 1:
            return float('inf')
        sqrt_term = np.sqrt(z**2 + 4 * (1 - z) * probs**2)
        fair_probs = (sqrt_term - z) / (2 * (1 - z))
        return fair_probs.sum() - 1
    
    try:
        z_opt = optimize.brentq(shin_equation, 0.001, 0.999, maxiter=max_iter)
        sqrt_term = np.sqrt(z_opt**2 + 4 * (1 - z_opt) * probs**2)
        fair_probs = (sqrt_term - z_opt) / (2 * (1 - z_opt))
        fair_probs = fair_probs / fair_probs.sum()
        return [round(1/p, 3) for p in fair_probs]
    except:
        fair_probs = probs / probs.sum()
        return [round(1/p, 3) for p in fair_probs]

def normalize_two_way_shin(o1: float, o2: float) -> Tuple[float, float]:
    if not o1 or not o2 or o1 <= 1 or o2 <= 1:
        return o1, o2
    normalized = shin_normalization([o1, o2])
    return normalized[0], normalized[1]

def normalize_three_way_shin(o1: float, ox: float, o2: float) -> Tuple[float, float, float]:
    if not all([o1, ox, o2]) or any(o <= 1 for o in [o1, ox, o2]):
        return o1, ox, o2
    normalized = shin_normalization([o1, ox, o2])
    return normalized[0], normalized[1], normalized[2]

# ============================================================
#         PINNACLE API (GRATIS!)
# ============================================================

def pinnacle_get_sports() -> List[dict]:
    """Ottieni lista sport disponibili."""
    try:
        r = requests.get(f"{PINNACLE_BASE}/sports", timeout=10)
        r.raise_for_status()
        data = r.json()
        return [s for s in data if s.get('name', '').lower() == 'soccer']
    except Exception as e:
        print(f"Errore Pinnacle sports: {e}")
        return []

def pinnacle_get_leagues(sport_id: int = 29) -> List[dict]:
    """
    Ottieni leghe di calcio.
    Sport ID 29 = Soccer
    """
    try:
        r = requests.get(
            f"{PINNACLE_BASE}/leagues",
            params={"sport_id": sport_id},
            timeout=10
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"Errore Pinnacle leagues: {e}")
        return []

def pinnacle_get_matchups(league_ids: List[int]) -> List[dict]:
    """Ottieni partite per le leghe specificate."""
    try:
        league_ids_str = ",".join(str(lid) for lid in league_ids)
        r = requests.get(
            f"{PINNACLE_BASE}/matchups",
            params={
                "league_ids": league_ids_str,
                "is_have_odds": "true"
            },
            timeout=10
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"Errore Pinnacle matchups: {e}")
        return []

def pinnacle_get_odds(matchup_id: int) -> dict:
    """Ottieni quote per una specifica partita."""
    try:
        r = requests.get(
            f"{PINNACLE_BASE}/odds",
            params={"matchup_id": matchup_id},
            timeout=10
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"Errore Pinnacle odds: {e}")
        return {}

def pinnacle_extract_prices(matchup: dict, odds_data: dict) -> dict:
    """
    Estrae e normalizza le quote da Pinnacle.
    
    Pinnacle fornisce:
    - Moneyline (1X2)
    - Totals (Over/Under)
    - Spread (Handicap)
    
    NON fornisce sempre:
    - BTTS (lo calcoliamo dal modello)
    - DNB (lo calcoliamo da 1X2)
    """
    
    out = {
        "home": matchup.get("participants", [{}])[0].get("name", "Home"),
        "away": matchup.get("participants", [{}])[1].get("name", "Away") if len(matchup.get("participants", [])) > 1 else "Away",
        "odds_1": None,
        "odds_x": None,
        "odds_2": None,
        "odds_over25": None,
        "odds_under25": None,
        "odds_dnb_home": None,
        "odds_dnb_away": None,
        "spread_line": 0.0,
        "total_line": 2.5,
    }
    
    if not odds_data:
        return out
    
    # Estrai Moneyline (1X2)
    for market in odds_data:
        market_type = market.get("type", "")
        
        # MONEYLINE (1X2)
        if market_type == "moneyline":
            prices = market.get("prices", [])
            if len(prices) >= 3:
                # Pinnacle: [Home, Draw, Away]
                out["odds_1"] = prices[0].get("price")
                out["odds_x"] = prices[1].get("price")
                out["odds_2"] = prices[2].get("price")
        
        # TOTALS (Over/Under)
        elif market_type == "total":
            points = market.get("points")
            if points == 2.5:
                prices = market.get("prices", [])
                if len(prices) >= 2:
                    out["odds_over25"] = prices[0].get("price")  # Over
                    out["odds_under25"] = prices[1].get("price")  # Under
                out["total_line"] = points
        
        # SPREAD (Handicap)
        elif market_type == "spread":
            points = market.get("points")
            if points == 0.0:  # Spread 0 = Asian Handicap 0
                prices = market.get("prices", [])
                if len(prices) >= 2:
                    # Possiamo usare questo per DNB estimation
                    out["odds_dnb_home"] = prices[0].get("price")
                    out["odds_dnb_away"] = prices[1].get("price")
            out["spread_line"] = points if points else 0.0
    
    # Calcola DNB da 1X2 se non disponibile
    if not out["odds_dnb_home"] and out["odds_1"] and out["odds_x"]:
        # Formula: DNB = (1 * X) / (1 + X)
        try:
            out["odds_dnb_home"] = round(
                (out["odds_1"] * out["odds_x"]) / (out["odds_1"] + out["odds_x"]) * 0.995,
                3
            )
        except:
            pass
    
    if not out["odds_dnb_away"] and out["odds_2"] and out["odds_x"]:
        try:
            out["odds_dnb_away"] = round(
                (out["odds_2"] * out["odds_x"]) / (out["odds_2"] + out["odds_x"]) * 0.995,
                3
            )
        except:
            pass
    
    # Normalizzazione Shin per 1X2
    if out["odds_1"] and out["odds_x"] and out["odds_2"]:
        n1, nx, n2 = normalize_three_way_shin(out["odds_1"], out["odds_x"], out["odds_2"])
        out["odds_1"], out["odds_x"], out["odds_2"] = n1, nx, n2
    
    # Normalizzazione Shin per Over/Under
    if out["odds_over25"] and out["odds_under25"]:
        no, nu = normalize_two_way_shin(out["odds_over25"], out["odds_under25"])
        out["odds_over25"], out["odds_under25"] = no, nu
    
    return out

# ============================================================
#  API-FOOTBALL SOLO PER RISULTATI REALI
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
        print(f"Errore API-Football: {e}")
        return []

# ============================================================
#  STIMA BTTS DA MODELLO BIVARIATO (NO API NEEDED!)
# ============================================================

def btts_probability_bivariate(lambda_h: float, lambda_a: float, rho: float) -> float:
    """
    Calcola P(BTTS) usando distribuzione Poisson bivariata.
    NON serve API esterna - calcoliamo dal modello!
    """
    p_h0 = poisson.pmf(0, lambda_h)
    p_a0 = poisson.pmf(0, lambda_a)
    
    # P(H=0, A=0) con correlazione Dixon-Coles
    p_h0_a0 = p_h0 * p_a0 * (1 - lambda_h * lambda_a * rho)
    
    # P(no BTTS) = P(H=0 or A=0)
    p_no_btts = p_h0 + p_a0 - p_h0_a0
    
    return max(0.0, min(1.0, 1 - p_no_btts))

# ============================================================
#          MODELLO POISSON MIGLIORATO
# ============================================================

def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return poisson.pmf(k, lam)

def entropia_poisson(lam: float, max_k: int = 15) -> float:
    e = 0.0
    for k in range(max_k + 1):
        p = poisson_pmf(k, lam)
        if p > 1e-10:
            e -= p * math.log2(p)
    return e

def home_advantage_factor(league: str = "generic") -> float:
    """Home advantage calibrato per lega."""
    ha_dict = {
        "premier_league": 1.35,
        "serie_a": 1.28,
        "la_liga": 1.32,
        "bundesliga": 1.30,
        "ligue_1": 1.25,
        "champions_league": 1.20,  # Meno home advantage in Europa
        "europa_league": 1.22,
        "generic": 1.30,
    }
    return ha_dict.get(league.lower().replace(" ", "_"), 1.30)

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
    """Stima lambda con approccio Bayesiano multi-sorgente."""
    
    # Probabilità normalizzate
    p1, px, p2 = normalize_three_way_shin(odds_1, odds_x, odds_2)
    p1_n = 1 / p1
    px_n = 1 / px
    p2_n = 1 / p2
    tot_p = p1_n + px_n + p2_n
    p1_n /= tot_p
    p2_n /= tot_p
    
    # Expected total da over/under
    if odds_over25 and odds_under25:
        po, pu = normalize_two_way_shin(odds_over25, odds_under25)
        p_over = 1 / po
        total_market = 2.5 + (p_over - 0.5) * 2.0
    else:
        total_market = total
    
    # Lambda base
    lambda_total = total_market / 2.0
    
    # Spread implicito
    prob_diff = p1_n - p2_n
    spread_factor = 1.0 + prob_diff * 0.8
    
    lambda_h_base = lambda_total * spread_factor
    lambda_a_base = lambda_total / spread_factor
    
    # Aggiustamento con DNB
    if odds_dnb_home and odds_dnb_home > 1 and odds_dnb_away and odds_dnb_away > 1:
        p_dnb_h = 1 / odds_dnb_home
        p_dnb_a = 1 / odds_dnb_away
        dnb_ratio = p_dnb_h / p_dnb_a
        spread_factor_dnb = math.sqrt(dnb_ratio)
        
        lambda_h_base *= (0.7 + 0.3 * spread_factor_dnb)
        lambda_a_base *= (0.7 + 0.3 / spread_factor_dnb)
    
    # Home advantage
    lambda_h = lambda_h_base * math.sqrt(home_advantage)
    lambda_a = lambda_a_base / math.sqrt(home_advantage)
    
    # Constraints
    lambda_h = max(0.3, min(4.0, lambda_h))
    lambda_a = max(0.3, min(4.0, lambda_a))
    
    return lambda_h, lambda_a

def estimate_rho_improved(
    lambda_h: float,
    lambda_a: float,
    p_draw: float,
) -> float:
    """Stima rho (correlazione gol)."""
    
    # Prior da draw probability
    rho_from_draw = -0.15 + (p_draw - 0.25) * 0.8
    
    # Expected low-scoring
    exp_low_score = math.exp(-lambda_h) + math.exp(-lambda_a)
    if exp_low_score > 0.6:
        rho_from_draw -= 0.05
    
    return max(-0.30, min(0.30, rho_from_draw))

def tau_dixon_coles(h: int, a: int, lh: float, la: float, rho: float) -> float:
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
    expected_total = lh + la
    return max(8, min(15, int(expected_total * 3.0)))

def build_score_matrix(lh: float, la: float, rho: float) -> List[List[float]]:
    mg = max_goals_adattivo(lh, la)
    mat: List[List[float]] = []
    
    for h in range(mg + 1):
        row = []
        for a in range(mg + 1):
            p = poisson_pmf(h, lh) * poisson_pmf(a, la)
            p *= tau_dixon_coles(h, a, lh, la, rho)
            row.append(max(0, p))
        mat.append(row)
    
    # Normalizzazione
    tot = sum(sum(r) for r in mat)
    if tot > 0:
        mat = [[p / tot for p in r] for r in mat]
    
    return mat

# ============================================================
#      CALCOLO PROBABILITÀ DA MATRICE
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
#        FUNZIONE PRINCIPALE MODELLO
# ============================================================

def risultato_completo_improved(
    odds_1: float,
    odds_x: float,
    odds_2: float,
    total: float,
    odds_over25: float = None,
    odds_under25: float = None,
    odds_dnb_home: float = None,
    odds_dnb_away: float = None,
    xg_for_home: float = None,
    xg_against_home: float = None,
    xg_for_away: float = None,
    xg_against_away: float = None,
    manual_boost_home: float = 0.0,
    manual_boost_away: float = 0.0,
    league: str = "generic",
) -> Dict[str, Any]:
    """Modello Dixon-Coles completo con Pinnacle odds."""
    
    # Normalizza quote Shin
    odds_1_n, odds_x_n, odds_2_n = normalize_three_way_shin(odds_1, odds_x, odds_2)
    
    # Probabilità normalizzate
    p1 = 1 / odds_1_n
    px = 1 / odds_x_n
    p2 = 1 / odds_2_n
    tot_p = p1 + px + p2
    p1 /= tot_p
    px /= tot_p
    p2 /= tot_p
    
    # Home advantage
    ha = home_advantage_factor(league)
    
    # Stima lambda
    lh, la = estimate_lambda_from_market_improved(
        odds_1_n, odds_x_n, odds_2_n,
        total,
        odds_over25, odds_under25,
        odds_dnb_home, odds_dnb_away,
        home_advantage=ha
    )
    
    # Boost manuali
    if manual_boost_home != 0.0:
        lh *= (1.0 + manual_boost_home)
    if manual_boost_away != 0.0:
        la *= (1.0 + manual_boost_away)
    
    # Blend con xG
    if all(x is not None for x in [xg_for_home, xg_against_home, xg_for_away, xg_against_away]):
        xg_h_est = (xg_for_home + xg_against_away) / 2
        xg_a_est = (xg_for_away + xg_against_home) / 2
        
        w_market = 0.65
        if xg_for_home > 0.5 and xg_for_away > 0.5:
            w_market = 0.60
        
        lh = w_market * lh + (1 - w_market) * xg_h_est
        la = w_market * la + (1 - w_market) * xg_a_est
    
    # Constraints
    lh = max(0.3, min(4.0, lh))
    la = max(0.3, min(4.0, la))
    
    # Stima rho
    rho = estimate_rho_improved(lh, la, px)
    
    # Matrici
    mat_ft = build_score_matrix(lh, la, rho)
    
    ratio_ht = 0.44 + 0.03 * (total - 2.5)
    ratio_ht = max(0.38, min(0.52, ratio_ht))
    mat_ht = build_score_matrix(lh * ratio_ht, la * ratio_ht, rho * 0.8)
    
    # Calcola probabilità
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
    
    # Multigol
    ranges = [(0,1),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5),(3,5)]
    multigol_home = {f"{a}-{b}": prob_multigol_from_dist(dist_home_ft, a, b) for a,b in ranges}
    multigol_away = {f"{a}-{b}": prob_multigol_from_dist(dist_away_ft, a, b) for a,b in ranges}
    
    # Double Chance
    dc = {
        "DC Casa o Pareggio": p_home + p_draw,
        "DC Trasferta o Pareggio": p_away + p_draw,
        "DC Casa o Trasferta": p_home + p_away
    }
    
    # Margini
    mg = len(mat_ft) - 1
    marg2 = sum(mat_ft[h][a] for h in range(mg+1) for a in range(mg+1) if h - a >= 2)
    marg3 = sum(mat_ft[h][a] for h in range(mg+1) for a in range(mg+1) if h - a >= 3)
    
    # Combo
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
    
    top10 = top_results_from_matrix(mat_ft, 10, 0.005)
    
    ent_home = entropia_poisson(lh)
    ent_away = entropia_poisson(la)
    
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
    
    odd_mass = sum(p for i, p in enumerate(dist_tot_ft) if i % 2 == 1)
    even_mass2 = 1 - odd_mass
    cover_0_2 = sum(dist_tot_ft[i] for i in range(0, min(3, len(dist_tot_ft))))
    cover_0_3 = sum(dist_tot_ft[i] for i in range(0, min(4, len(dist_tot_ft))))
    
    return {
        "lambda_home": lh,
        "lambda_away": la,
        "rho": rho,
        "p_home": p_home,
        "p_draw": p_draw,
        "p_away": p_away,
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
#   CONTROLLI QUALITÀ
# ============================================================

def check_coerenza_quote_improved(
    odds_1: float,
    odds_x: float,
    odds_2: float,
    odds_over25: float,
    odds_under25: float,
) -> Tuple[List[str], float]:
    
    warnings = []
    quality_score = 100.0
    
    if odds_1 and odds_x and odds_2:
        margin = (1/odds_1 + 1/odds_x + 1/odds_2) - 1
        if margin > 0.08:
            warnings.append(f"Margine 1X2 alto per Pinnacle ({margin*100:.1f}%) → verifica")
            quality_score -= 15
        elif margin < 0.02:
            warnings.append("Margine 1X2 molto basso → ottimo!")
            quality_score += 5
    
    if odds_1 and odds_2:
        if odds_1 < 1.35 and odds_2 < 4.0:
            warnings.append("Casa molto favorita ma trasferta non alta")
            quality_score -= 12
        if odds_1 > 3.5 and odds_2 > 3.5:
            warnings.append("Match molto equilibrato → alta varianza")
            quality_score -= 8
    
    if odds_over25 and odds_under25:
        margin_ou = (1/odds_over25 + 1/odds_under25) - 1
        if not (0.02 < margin_ou < 0.08):
            warnings.append(f"Margine O/U anomalo ({margin_ou*100:.1f}%)")
            quality_score -= 10
        
        if odds_1 and odds_1 < 1.5 and odds_over25 > 2.3:
            warnings.append("Favorita netta ma over alto → contraddizione")
            quality_score -= 15
    
    quality_score = max(0, quality_score)
    return warnings, quality_score

def compute_market_confidence_pinnacle(
    odds_1: float,
    odds_x: float,
    odds_2: float,
    odds_over25: float = None,
    odds_under25: float = None,
) -> float:
    """
    Confidence per Pinnacle: già è sharp, quindi score alto base.
    """
    score = 75.0  # Base alto (Pinnacle è sharp!)
    
    # Margine
    if odds_1 and odds_x and odds_2:
        margin = (1/odds_1 + 1/odds_x + 1/odds_2) - 1
        if margin < 0.04:
            score += 15  # Pinnacle ha margini bassissimi
        elif margin > 0.06:
            score -= 10
    
    # Completezza
    if odds_over25:
        score += 10
    
    return max(0, min(100, score))

# ============================================================
#              STREAMLIT APP
# ============================================================

st.set_page_config(page_title="⚽ Modello Scommesse PRO – Pinnacle Sharp", layout="wide")

st.title("⚽ Modello Scommesse con Pinnacle API")
st.markdown("""
### 🎯 Pinnacle Sharp Odds (GRATIS!)
- ✅ **Quote più accurate** del mercato (sharp book)
- ✅ **Margini bassissimi** (2-3%)
- ✅ **GRATIS illimitato** (no API key!)
- ✅ **BTTS calcolato dal modello** Dixon-Coles bivariato
- ✅ **Shin Normalization** automatica
""")

st.caption(f"🕐 Esecuzione: {datetime.now().isoformat(timespec='seconds')}")

# Session state
if "pinnacle_leagues" not in st.session_state:
    st.session_state.pinnacle_leagues = []
if "pinnacle_matchups" not in st.session_state:
    st.session_state.pinnacle_matchups = []
if "selected_matchup" not in st.session_state:
    st.session_state.selected_matchup = None
if "selected_odds" not in st.session_state:
    st.session_state.selected_odds = {}

# ============================================================
#               STORICO
# ============================================================

st.subheader("📊 Storico Analisi")

col_hist1, col_hist2 = st.columns(2)

with col_hist1:
    if os.path.exists(ARCHIVE_FILE):
        df_st = pd.read_csv(ARCHIVE_FILE)
        st.write(f"📁 Analisi salvate: **{len(df_st)}**")
        
        if "esito_reale" in df_st.columns and "match_ok" in df_st.columns:
            df_complete = df_st[df_st["esito_reale"].notna() & (df_st["esito_reale"] != "")]
            
            if len(df_complete) > 0:
                accuracy = df_complete["match_ok"].mean() * 100
                st.metric("🎯 Accuracy", f"{accuracy:.1f}%")
        
        st.dataframe(df_st.tail(15), height=300)
    else:
        st.info("Nessuno storico")

with col_hist2:
    st.markdown("### 🗑️ Gestione")
    if os.path.exists(ARCHIVE_FILE):
        df_del = pd.read_csv(ARCHIVE_FILE)
        if not df_del.empty:
            df_del["label"] = df_del.apply(
                lambda r: f"{r.get('timestamp','?')} – {r.get('match','?')}",
                axis=1,
            )
            to_delete = st.selectbox("Elimina:", df_del["label"].tolist())
            if st.button("🗑️ Elimina"):
                df_new = df_del[df_del["label"] != to_delete].drop(columns=["label"])
                df_new.to_csv(ARCHIVE_FILE, index=False)
                st.success("✅ Eliminato")
                st.rerun()

st.markdown("---")

# ============================================================
#        CARICAMENTO DA PINNACLE
# ============================================================

st.subheader("🔍 Carica Partita da Pinnacle (Sharp & Gratis)")

if st.button("1️⃣ Carica Leghe Pinnacle"):
    with st.spinner("Caricamento leghe..."):
        leagues = pinnacle_get_leagues(sport_id=29)
        st.session_state.pinnacle_leagues = leagues
        if leagues:
            st.success(f"✅ {len(leagues)} leghe caricate")
        else:
            st.error("Errore caricamento leghe")

if st.session_state.pinnacle_leagues:
    league_options = {f"{l['name']} (ID: {l['id']})": l['id'] for l in st.session_state.pinnacle_leagues}
    selected_league_name = st.selectbox("2️⃣ Seleziona Lega", list(league_options.keys()))
    selected_league_id = league_options[selected_league_name]
    
    if st.button("3️⃣ Carica Partite"):
        with st.spinner("Caricamento partite..."):
            matchups = pinnacle_get_matchups([selected_league_id])
            st.session_state.pinnacle_matchups = matchups
            if matchups:
                st.success(f"✅ {len(matchups)} partite trovate")
            else:
                st.warning("Nessuna partita con quote disponibile")
    
    if st.session_state.pinnacle_matchups:
        matchup_labels = []
        for m in st.session_state.pinnacle_matchups:
            participants = m.get("participants", [])
            if len(participants) >= 2:
                home = participants[0].get("name", "Home")
                away = participants[1].get("name", "Away")
                start = m.get("starts", "")[:16].replace("T", " ")
                matchup_labels.append(f"{home} vs {away} – {start}")
        
        if matchup_labels:
            selected_match_label = st.selectbox("4️⃣ Seleziona Partita", matchup_labels)
            idx = matchup_labels.index(selected_match_label)
            matchup = st.session_state.pinnacle_matchups[idx]
            
            st.session_state.selected_matchup = matchup
            
            matchup_id = matchup.get("id")
            if matchup_id:
                odds_data = pinnacle_get_odds(matchup_id)
                prices = pinnacle_extract_prices(matchup, odds_data)
                st.session_state.selected_odds = prices
                
                st.success("✅ Quote Pinnacle caricate (Sharp & Shin normalized)")
                
                # Mostra info
                margin_1x2 = 0
                if prices.get("odds_1") and prices.get("odds_x") and prices.get("odds_2"):
                    margin_1x2 = (1/prices["odds_1"] + 1/prices["odds_x"] + 1/prices["odds_2"] - 1) * 100
                
                col_info1, col_info2 = st.columns(2)
                with col_info1:
                    st.metric("📊 Margine 1X2", f"{margin_1x2:.2f}%")
                with col_info2:
                    st.metric("🏆 Bookmaker", "Pinnacle (Sharp)")

st.markdown("---")

# ============================================================
#        INPUT DATI
# ============================================================

st.subheader("📝 Dati Partita")

prices = st.session_state.get("selected_odds", {})

col_m1, col_m2 = st.columns(2)

with col_m1:
    default_name = ""
    if prices.get("home"):
        default_name = f"{prices['home']} vs {prices['away']}"
    match_name = st.text_input("Nome Partita", value=default_name)

with col_m2:
    league_type = st.selectbox("Lega", [
        "generic",
        "premier_league",
        "la_liga",
        "serie_a",
        "bundesliga",
        "ligue_1",
        "champions_league",
        "europa_league",
    ])

st.subheader("💰 Quote Principali (Pinnacle)")

col_q1, col_q2, col_q3 = st.columns(3)

with col_q1:
    odds_1 = st.number_input("Quota 1 (Casa)", 
                             value=float(prices.get("odds_1") or 2.00), 
                             step=0.01)
    odds_over25 = st.number_input("Quota Over 2.5", 
                                  value=float(prices.get("odds_over25") or 0.0), 
                                  step=0.01)

with col_q2:
    odds_x = st.number_input("Quota X (Pareggio)", 
                            value=float(prices.get("odds_x") or 3.50), 
                            step=0.01)
    odds_under25 = st.number_input("Quota Under 2.5", 
                                   value=float(prices.get("odds_under25") or 0.0), 
                                   step=0.01)

with col_q3:
    odds_2 = st.number_input("Quota 2 (Trasferta)", 
                            value=float(prices.get("odds_2") or 3.80), 
                            step=0.01)
    total_line = st.number_input("Linea Total", 
                                value=float(prices.get("total_line", 2.5)), 
                                step=0.25)

st.info("ℹ️ BTTS viene calcolato automaticamente dal modello Dixon-Coles bivariato (più accurato delle quote!)")

st.subheader("🎲 Quote Speciali")

col_s1, col_s2 = st.columns(2)

with col_s1:
    odds_dnb_home = st.number_input("DNB Casa", 
                                    value=float(prices.get("odds_dnb_home") or 0.0), 
                                    step=0.01)

with col_s2:
    odds_dnb_away = st.number_input("DNB Trasferta", 
                                    value=float(prices.get("odds_dnb_away") or 0.0), 
                                    step=0.01)

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

if st.button("🎯 CALCOLA MODELLO (Pinnacle Sharp)", type="primary"):
    with st.spinner("Elaborazione con modello Dixon-Coles Bayesiano + Pinnacle Sharp odds..."):
        
        # Check quality
        warnings, quality_score = check_coerenza_quote_improved(
            odds_1, odds_x, odds_2,
            odds_over25, odds_under25
        )
        
        # Market confidence
        market_conf = compute_market_confidence_pinnacle(
            odds_1, odds_x, odds_2,
            odds_over25, odds_under25
        )
        
        # Calcolo modello
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
            odds_dnb_home=odds_dnb_home if odds_dnb_home > 0 else None,
            odds_dnb_away=odds_dnb_away if odds_dnb_away > 0 else None,
            manual_boost_home=boost_home,
            manual_boost_away=boost_away,
            league=league_type,
            **xg_args
        )
        
        # BTTS dal modello
        btts_prob_model = ris["btts"]
        btts_odds_model = 1 / btts_prob_model if btts_prob_model > 0 else 2.0
        
        st.success("✅ Calcolo completato con Pinnacle Sharp Odds!")
        
        # Metriche
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        
        with col_m1:
            st.metric("🏆 Quality Score", f"{quality_score:.0f}/100")
        with col_m2:
            st.metric("📊 Confidence", f"{market_conf:.0f}/100", 
                     help="Pinnacle è già sharp → confidence alta")
        with col_m3:
            st.metric("🏠 λ Casa", f"{ris['lambda_home']:.2f}")
        with col_m4:
            st.metric("✈️ λ Trasferta", f"{ris['lambda_away']:.2f}")
        
        # Warnings
        if warnings:
            with st.expander("⚠️ Avvisi", expanded=True):
                for w in warnings:
                    st.warning(w)
        
        # Value Finder
        st.subheader("💎 Value Finder vs Pinnacle Sharp")
        
        value_rows = []
        
        # 1X2
        for lab, p_mod, odd in [
            ("1 (Casa)", ris["p_home"], odds_1),
            ("X (Pareggio)", ris["p_draw"], odds_x),
            ("2 (Trasferta)", ris["p_away"], odds_2),
        ]:
            p_pinnacle = 1 / odd
            edge = (p_mod - p_pinnacle) * 100
            ev = (p_mod * odd - 1) * 100
            
            value_rows.append({
                "Mercato": "1X2",
                "Esito": lab,
                "Prob Modello %": f"{p_mod*100:.1f}",
                "Prob Pinnacle %": f"{p_pinnacle*100:.1f}",
                "Edge %": f"{edge:+.1f}",
                "EV %": f"{ev:+.1f}",
                "Value": "✅" if edge >= 2 else ("⚠️" if edge >= 0.5 else "")
            })
        
        # Over/Under
        if odds_over25 and odds_over25 > 0:
            p_mod = ris["over_25"]
            p_pinnacle = 1 / odds_over25
            edge = (p_mod - p_pinnacle) * 100
            ev = (p_mod * odds_over25 - 1) * 100
            
            value_rows.append({
                "Mercato": "Over/Under 2.5",
                "Esito": "Over 2.5",
                "Prob Modello %": f"{p_mod*100:.1f}",
                "Prob Pinnacle %": f"{p_pinnacle*100:.1f}",
                "Edge %": f"{edge:+.1f}",
                "EV %": f"{ev:+.1f}",
                "Value": "✅" if edge >= 2 else ("⚠️" if edge >= 0.5 else "")
            })
        
        # BTTS (dal modello)
        st.info(f"📊 BTTS calcolato dal modello: {btts_prob_model*100:.1f}% (quota teorica: {btts_odds_model:.2f})")
        
        df_value = pd.DataFrame(value_rows)
        
        df_value_high = df_value[df_value["Value"].str.contains("✅", na=False)]
        
        if not df_value_high.empty:
            st.success(f"🎯 {len(df_value_high)} value bet(s) vs Pinnacle Sharp!")
            st.dataframe(df_value_high, use_container_width=True)
        else:
            st.info("ℹ️ Nessun strong value trovato (normale con Pinnacle - sono sharp!)")
        
        st.dataframe(df_value, use_container_width=True)
        
        # Dettagli
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
            
            with col_d3:
                st.markdown("**Pari/Dispari**")
                st.write(f"Pari FT: {ris['even_ft']*100:.1f}%")
                st.write(f"Dispari FT: {ris['odd_ft']*100:.1f}%")
                
                st.markdown("**Modello**")
                st.write(f"ρ: {ris['rho']:.3f}")
                st.write(f"Entropia H: {ris['ent_home']:.2f}")
                st.write(f"Entropia A: {ris['ent_away']:.2f}")
        
        with st.expander("🎯 Top 10 Risultati Esatti"):
            for h, a, p in ris["top10"]:
                st.write(f"{h}-{a}: **{p:.1f}%**")
        
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
            "bookmaker": "Pinnacle",
            "quality_score": quality_score,
            "market_confidence": market_conf,
            "odds_1": odds_1,
            "odds_x": odds_x,
            "odds_2": odds_2,
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
            st.success("💾 Analisi salvata")
        except Exception as e:
            st.warning(f"Errore salvataggio: {e}")

st.markdown("---")

# ============================================================
#        AGGIORNAMENTO RISULTATI
# ============================================================

st.subheader("🔄 Aggiorna Risultati Reali (API-Football)")

if st.button("Recupera risultati ultimi 3 giorni"):
    if not os.path.exists(ARCHIVE_FILE):
        st.warning("Nessuno storico")
    else:
        with st.spinner("Recupero risultati..."):
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
                    
                    esito = "1" if gh > ga else ("X" if gh == ga else "2")
                    
                    df.at[idx, "risultato_reale"] = f"{gh}-{ga}"
                    df.at[idx, "esito_reale"] = esito
                    
                    pred = row.get("esito_modello", "")
                    if pred and esito:
                        df.at[idx, "match_ok"] = 1 if pred == esito else 0
                    
                    updated += 1
            
            df.to_csv(ARCHIVE_FILE, index=False)
            st.success(f"✅ Aggiornate {updated} partite")
            if updated > 0:
                st.rerun()

st.markdown("---")
st.caption("🏆 Powered by Pinnacle Sharp Odds (FREE) | Dixon-Coles Bayesian Model | Shin Normalization")
