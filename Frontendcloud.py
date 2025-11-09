import math
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, date, timedelta
import pandas as pd
import os
import requests
import streamlit as st

# ============================================================
#                 CONFIG
# ============================================================

# The Odds API (PRO)
THE_ODDS_API_KEY = "06c16ede44d09f9b3498bb63354930c4"
THE_ODDS_BASE = "https://api.the-odds-api.com/v4"

# API-FOOTBALL per risultati reali
API_FOOTBALL_KEY = "95c43f936816cd4389a747fd2cfe061a"
API_FOOTBALL_BASE = "https://v3.football.api-sports.io"

ARCHIVE_FILE = "storico_analisi.csv"

# autorefresh facoltativo
AUTOREFRESH_DEFAULT_SEC = 0  # metti 60/120 se vuoi refresh automatico

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

# ============================================================
#         FUNZIONI THE ODDS API
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
        soccer = [s for s in data if s.get("key", "").startswith("soccer")]
        return soccer
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

    # 1) con btts
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

    # 2) senza btts
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
#  FUNZIONI DI NORMALIZZAZIONE QUOTE
# ============================================================

def normalize_two_way(o1: float, o2: float) -> Tuple[Optional[float], Optional[float]]:
    if not o1 or not o2 or o1 <= 1 or o2 <= 1:
        return o1, o2
    p1 = 1 / o1
    p2 = 1 / o2
    overround = p1 + p2
    if overround <= 0:
        return o1, o2
    p1_f = p1 / overround
    p2_f = p2 / overround
    return round(1 / p1_f, 3), round(1 / p2_f, 3)

def normalize_three_way(o1: float, ox: float, o2: float) -> Tuple[float, float, float]:
    p1 = 1 / o1 if o1 and o1 > 1 else 0.0
    px = 1 / ox if ox and ox > 1 else 0.0
    p2 = 1 / o2 if o2 and o2 > 1 else 0.0
    tot = p1 + px + p2
    if tot == 0:
        return o1, ox, o2
    p1_f = p1 / tot
    px_f = px / tot
    p2_f = p2 / tot
    return round(1 / p1_f, 3), round(1 / px_f, 3), round(1 / p2_f, 3)

# ============================================================
#  STIMA GG QUANDO NON ARRIVA DALL’API
# ============================================================

def estimate_btts_from_basic_odds(
    odds_1: float = None,
    odds_x: float = None,
    odds_2: float = None,
    odds_over25: float = None,
    odds_under25: float = None,
) -> float:
    def _p(odd: float) -> float:
        return 1.0 / odd if odd and odd > 1 else 0.0

    p_over = _p(odds_over25)
    p_home = _p(odds_1)
    p_away = _p(odds_2)

    if p_over == 0:
        balance = 1.0 - abs(p_home - p_away)
        gg_prob = 0.50 + (balance - 0.5) * 0.25
        gg_prob = max(0.38, min(0.70, gg_prob))
        return round(1.0 / gg_prob, 3)

    gg_prob = 0.50 + (p_over - 0.50) * 0.9
    balance = 1.0 - abs(p_home - p_away)
    gg_prob += (balance - 0.5) * 0.20
    gg_prob = max(0.35, min(0.75, gg_prob))
    return round(1.0 / gg_prob, 3)

# ============================================================
#   ESTRATTORE QUOTE DA EVENTO (pesi + trimming + normalizzazione)
# ============================================================

def oddsapi_extract_prices(event: dict) -> dict:
    WEIGHTS = {
        "pinnacle": 1.7,
        "bet365": 1.5,
        "unibet_eu": 1.2,
        "marathonbet": 1.2,
        "williamhill": 1.1,
        "bwin": 1.0,
        "betonlineag": 1.0,
        "10bet": 1.0,
        "bovada": 0.8,
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
        out["odds_btts"] = estimate_btts_from_basic_odds()
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

            # 1X2
            if ("h2h" in mk_key) or (mk_key == "h2h") or ("match_winner" in mk_key):
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

            # TOTALS → 2.5
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

            # DNB
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

            # SPREADS 0 → DNB
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

            # BTTS
            elif "btts" in mk_key or "both_teams_to_score" in mk_key:
                for o in mk.get("outcomes", []):
                    name_l = (o.get("name") or "").lower()
                    price = o.get("price")
                    if not price:
                        continue
                    if "yes" in name_l or "sì" in name_l or "si" in name_l:
                        btts_list.append((price, w))

    # pulizia outlier
    def _trim_outliers(values: List[Tuple[float, float]], tol: float = 0.10):
        if len(values) <= 2:
            return values
        avg = sum(v for v, _ in values) / len(values)
        low, high = avg * (1 - tol), avg * (1 + tol)
        return [(v, w) for (v, w) in values if low <= v <= high]

    h2h_home = _trim_outliers(h2h_home)
    h2h_draw = _trim_outliers(h2h_draw)
    h2h_away = _trim_outliers(h2h_away)
    over25_list = _trim_outliers(over25_list)
    under25_list = _trim_outliers(under25_list)
    dnb_home_list = _trim_outliers(dnb_home_list)
    dnb_away_list = _trim_outliers(dnb_away_list)
    btts_list = _trim_outliers(btts_list)

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

    # stima btts se manca
    if out["odds_btts"] is None:
        out["odds_btts"] = estimate_btts_from_basic_odds(
            odds_1=out["odds_1"],
            odds_x=out["odds_x"],
            odds_2=out["odds_2"],
            odds_over25=out["odds_over25"],
            odds_under25=out["odds_under25"],
        )

    # normalizzazione leggera del 1X2
    if out["odds_1"] and out["odds_x"] and out["odds_2"]:
        n1, nx, n2 = normalize_three_way(out["odds_1"], out["odds_x"], out["odds_2"])
        out["odds_1"], out["odds_x"], out["odds_2"] = n1, nx, n2

    # normalizzazione OU se ci sono entrambi
    if out["odds_over25"] and out["odds_under25"]:
        no, nu = normalize_two_way(out["odds_over25"], out["odds_under25"])
        out["odds_over25"], out["odds_under25"] = no, nu

    return out

# ============================================================
#            API-FOOTBALL SOLO PER RISULTATI REALI
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
#                  FUNZIONI MODELLO + NUOVI MERCATI
# ============================================================

def poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * (lam ** k) / math.factorial(k)

def entropia_poisson(lam: float, max_k: int = 15) -> float:
    e = 0.0
    for k in range(max_k + 1):
        p = poisson_pmf(k, lam)
        if p > 0:
            e -= p * math.log2(p)
    return e

def decimali_a_prob(odds: float) -> float:
    return 1 / odds if odds and odds > 0 else 0.0

def normalize_1x2_from_odds(o1: float, ox: float, o2: float) -> Tuple[float, float, float]:
    p1 = 1 / o1 if o1 and o1 > 0 else 0.0
    px = 1 / ox if ox and ox > 0 else 0.0
    p2 = 1 / o2 if o2 and o2 > 0 else 0.0
    tot = p1 + px + p2
    if tot == 0:
        return 0.33, 0.34, 0.33
    return p1 / tot, px / tot, p2 / tot

def gol_attesi_migliorati(spread: float, total: float,
                          p1: float, p2: float) -> Tuple[float, float]:
    if total < 2.25:
        total_eff = total * 1.03
    elif total > 3.0:
        total_eff = total * 0.97
    else:
        total_eff = total
    base = total_eff / 2.0
    diff = spread / 2.0
    fatt_int = 1 + (total_eff - 2.5) * 0.15
    lh = (base - diff) * fatt_int
    la = (base + diff) * fatt_int
    fatt_dir = ((p1 - p2) * 0.2) + 1.0
    lh *= fatt_dir
    la /= fatt_dir
    return max(lh, 0.05), max(la, 0.05)

def blend_lambda_market_xg(lambda_market_home: float,
                           lambda_market_away: float,
                           xg_for_home: float,
                           xg_against_home: float,
                           xg_for_away: float,
                           xg_against_away: float,
                           w_market: float = 0.6) -> Tuple[float, float]:
    xg_home_est = (xg_for_home + xg_against_away) / 2
    xg_away_est = (xg_for_away + xg_against_home) / 2
    lh = w_market * lambda_market_home + (1 - w_market) * xg_home_est
    la = w_market * lambda_market_away + (1 - w_market) * xg_away_est
    return max(lh, 0.05), max(la, 0.05)

def max_goals_adattivo(lh: float, la: float) -> int:
    return max(8, int((lh + la) * 2.5))

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

def build_score_matrix(lh: float, la: float, rho: float) -> List[List[float]]:
    mg = max_goals_adattivo(lh, la)
    mat: List[List[float]] = []
    for h in range(mg + 1):
        row = []
        for a in range(mg + 1):
            p = poisson_pmf(h, lh) * poisson_pmf(a, la)
            p *= tau_dixon_coles(h, a, lh, la, rho)
            row.append(p)
        mat.append(row)
    tot = sum(sum(r) for r in mat)
    mat = [[p / tot for p in r] for r in mat]
    return mat

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
    
    # ============================================================
# 1. DATI PARTITA
# ============================================================

st.subheader("1. Dati partita")

default_match_name = ""
if st.session_state.get("selected_event_prices", {}).get("home"):
    default_match_name = (
        f"{st.session_state['selected_event_prices']['home']} vs "
        f"{st.session_state['selected_event_prices']['away']}"
    )

match_name = st.text_input(
    "Nome partita (es. Milan vs Inter)",
    value=default_match_name
)

# ============================================================
# 2. LINEE DI APERTURA
# ============================================================

st.subheader("2. Linee di apertura (manuali)")
col_ap1, col_ap2 = st.columns(2)
with col_ap1:
    spread_ap = st.number_input("Spread apertura", value=0.0, step=0.25)
with col_ap2:
    total_ap = st.number_input("Total apertura", value=2.5, step=0.25)

# ============================================================
# 3. LINEE CORRENTI E QUOTE
# ============================================================

st.subheader("3. Linee correnti e quote (precompilate)")

api_prices = st.session_state.get("selected_event_prices", {})

# se manca BTTS lo stimo
if not api_prices.get("odds_btts") or api_prices.get("odds_btts") <= 1.01:
    api_prices["odds_btts"] = estimate_btts_from_basic_odds(
        odds_1=api_prices.get("odds_1"),
        odds_x=api_prices.get("odds_x"),
        odds_2=api_prices.get("odds_2"),
        odds_over25=api_prices.get("odds_over25"),
        odds_under25=api_prices.get("odds_under25"),
    )

odds1_tmp = api_prices.get("odds_1")
oddsx_tmp = api_prices.get("odds_x")
odds2_tmp = api_prices.get("odds_2")

def _safe_div(a, b):
    try:
        return a / b
    except Exception:
        return None

# stima DNB se non arriva dall’API
if (not api_prices.get("odds_dnb_home")) and odds1_tmp and oddsx_tmp:
    dnb_home_calc = _safe_div(odds1_tmp * oddsx_tmp, (odds1_tmp + oddsx_tmp))
    if dnb_home_calc:
        # leggero haircut per non essere troppo ottimisti
        api_prices["odds_dnb_home"] = round(dnb_home_calc * 0.995, 3)

if (not api_prices.get("odds_dnb_away")) and odds2_tmp and oddsx_tmp:
    dnb_away_calc = _safe_div(odds2_tmp * oddsx_tmp, (odds2_tmp + oddsx_tmp))
    if dnb_away_calc:
        api_prices["odds_dnb_away"] = round(dnb_away_calc * 0.995, 3)

# chiave stabile legata alla partita selezionata
key_suffix = st.session_state.get("selected_event_key", "match")

col_co1, col_co2, col_co3 = st.columns(3)
with col_co1:
    spread_co = st.number_input(
        "Spread corrente",
        value=0.0,
        step=0.25,
        key=f"spread_co_{key_suffix}"
    )
    odds_1 = st.number_input(
        "Quota 1",
        value=float(api_prices.get("odds_1") or 1.80),
        step=0.01,
        key=f"odds1_{key_suffix}"
    )

with col_co2:
    total_co = st.number_input(
        "Total corrente",
        value=2.5,
        step=0.25,
        key=f"total_co_{key_suffix}"
    )
    odds_x = st.number_input(
        "Quota X",
        value=float(api_prices.get("odds_x") or 3.50),
        step=0.01,
        key=f"oddsx_{key_suffix}"
    )

with col_co3:
    odds_2 = st.number_input(
        "Quota 2",
        value=float(api_prices.get("odds_2") or 4.50),
        step=0.01,
        key=f"odds2_{key_suffix}"
    )
    odds_btts = st.number_input(
        "Quota GG (BTTS sì)",
        value=float(api_prices.get("odds_btts") or 1.95),
        step=0.01,
        key=f"odds_btts_{key_suffix}"
    )

st.subheader("3.b DNB (Draw No Bet)")
col_dnb1, col_dnb2 = st.columns(2)
with col_dnb1:
    odds_dnb_home = st.number_input(
        "Quota DNB Casa",
        value=float(api_prices.get("odds_dnb_home") or 0.0),
        step=0.01,
        key=f"dnb_home_{key_suffix}"
    )
with col_dnb2:
    odds_dnb_away = st.number_input(
        "Quota DNB Trasferta",
        value=float(api_prices.get("odds_dnb_away") or 0.0),
        step=0.01,
        key=f"dnb_away_{key_suffix}"
    )

st.subheader("3.c Quote Over/Under 2.5")
col_ou1, col_ou2 = st.columns(2)
with col_ou1:
    odds_over25 = st.number_input(
        "Quota Over 2.5",
        value=float(api_prices.get("odds_over25") or 0.0),
        step=0.01,
        key=f"over25_{key_suffix}"
    )
with col_ou2:
    odds_under25 = st.number_input(
        "Quota Under 2.5",
        value=float(api_prices.get("odds_under25") or 0.0),
        step=0.01,
        key=f"under25_{key_suffix}"
    )

# ============================================================
# 4. XG (manuali)
# ============================================================

st.subheader("4. xG avanzati (opzionali)")
col_xg1, col_xg2 = st.columns(2)
with col_xg1:
    xg_tot_home = st.text_input("xG totali CASA", "")
    xga_tot_home = st.text_input("xGA totali CASA", "")
    partite_home = st.text_input("Partite giocate CASA (es. 10 o 5-3-2)", "")
with col_xg2:
    xg_tot_away = st.text_input("xG totali OSPITE", "")
    xga_tot_away = st.text_input("xGA totali OSPITE", "")
    partite_away = st.text_input("Partite giocate OSPITE (es. 10 o 5-3-2)", "")

def parse_xg_block(xg_tot_s: str, xga_tot_s: str, record_s: str):
    """
    Converte i tuoi xG totali e le partite giocate in xG medi.
    Se non compili bene, restituisce (None, None) e il modello resta in modalità base.
    """
    if xg_tot_s.strip() == "" or xga_tot_s.strip() == "" or record_s.strip() == "":
        return None, None
    try:
        xg_tot = float(xg_tot_s.replace(",", "."))
        xga_tot = float(xga_tot_s.replace(",", "."))
        if "-" in record_s:
            parts = record_s.split("-")
            matches = sum(int(p) for p in parts if p.strip() != "")
        else:
            matches = int(record_s.strip())
        if matches <= 0:
            return None, None
        return xg_tot / matches, xga_tot / matches
    except Exception:
        return None, None

xg_home_for, xg_home_against = parse_xg_block(xg_tot_home, xga_tot_home, partite_home)
xg_away_for, xg_away_against = parse_xg_block(xg_tot_away, xga_tot_away, partite_away)

has_xg = not (
    xg_home_for is None or xg_home_against is None or
    xg_away_for is None or xg_away_against is None
)

if not has_xg:
    st.info("Modalità: BASE (spread/total/quote). Se inserisci xG passo in modalità avanzata.")
else:
    st.success("Modalità: AVANZATA (spread/total + quote + xG/xGA).")

# ============================================================
# 5. CALCOLO MODELLO
# ============================================================

if st.button("CALCOLA MODELLO"):
    # 1) modello su linee di apertura (per confronto)
    ris_ap = risultato_completo(
        spread_ap, total_ap,
        odds_1, odds_x, odds_2,
        0.0,
        xg_home_for, xg_home_against,
        xg_away_for, xg_away_against,
        odds_dnb_home=odds_dnb_home if odds_dnb_home > 0 else None,
        odds_dnb_away=odds_dnb_away if odds_dnb_away > 0 else None,
    )

    # 2) modello su linee correnti (quello "vero")
    ris_co = risultato_completo(
        spread_co, total_co,
        odds_1, odds_x, odds_2,
        odds_btts,
        xg_home_for, xg_home_against,
        xg_away_for, xg_away_against,
        odds_dnb_home=odds_dnb_home if odds_dnb_home > 0 else None,
        odds_dnb_away=odds_dnb_away if odds_dnb_away > 0 else None,
    )

    # se il BTTS era mancante lo ricalibro dal modello
    if not odds_btts or odds_btts <= 1.01:
        p_gg_modello = ris_co["btts"]
        if p_gg_modello and p_gg_modello > 0:
            quota_gg_modello = 1 / p_gg_modello
            odds_btts = round(quota_gg_modello * 0.99, 3)

    ent_media = (ris_co["ent_home"] + ris_co["ent_away"]) / 2

    warnings = check_coerenza_quote(
        odds_1, odds_x, odds_2,
        odds_over25, odds_under25
    )

    mpi = compute_market_pressure_index(
        odds_1, odds_x, odds_2,
        odds_over25, odds_under25,
        odds_dnb_home, odds_dnb_away
    )

    aff = compute_structure_affidability(
        spread_ap, spread_co,
        total_ap, total_co,
        ent_media,
        has_xg,
        odds_1, odds_x, odds_2
    )

    global_conf = compute_global_confidence(
        base_aff=aff,
        n_warnings=len(warnings),
        mpi=mpi,
        has_xg=has_xg
    )

    st.success("Calcolo completato ✅")
    st.subheader("⭐ Sintesi Match")
    st.write(f"Affidabilità del match (struttura): **{aff}/100**")
    st.write(f"Confidence globale: **{global_conf}/100**")
    st.write(f"Market Pressure Index: **{mpi}/100**")

    # --------------------------------------------------------
    # blocco warning
    # --------------------------------------------------------
    if warnings:
        st.subheader("⚠️ Check coerenza quote")
        for w in warnings:
            st.write(f"- {w}")
    else:
        st.subheader("✅ Check coerenza quote")
        st.write("Quote coerenti con il modello minimo.")

    # --------------------------------------------------------
    # movimento di mercato
    # --------------------------------------------------------
    delta_spread = spread_co - spread_ap
    delta_total = total_co - total_ap

    st.subheader("🔁 Movimento di mercato")
    if abs(delta_spread) < 0.01 and abs(delta_total) < 0.01:
        st.write("Linee stabili.")
    else:
        if abs(delta_spread) >= 0.01:
            if delta_spread < 0:
                st.write(f"- Spread sceso di {abs(delta_spread):.2f} → mercato più pro CASA")
            else:
                st.write(f"- Spread salito di {abs(delta_spread):.2f} → mercato più pro TRASFERTA")
        if abs(delta_total) >= 0.01:
            if delta_total > 0:
                st.write(f"- Total salito di {delta_total:.2f} → mercato si aspetta più gol")
            else:
                st.write(f"- Total sceso di {abs(delta_total):.2f} → mercato si aspetta meno gol")

    # --------------------------------------------------------
    # Value Finder
    # --------------------------------------------------------
    st.subheader("💰 Value Finder (modello vs quote)")

    rows = []
    anomalo_ou = any("over/under 2.5" in w.lower() for w in warnings)

    # 1X2
    for lab, p_mod, odd in [
        ("1", ris_co["p_home"], odds_1),
        ("X", ris_co["p_draw"], odds_x),
        ("2", ris_co["p_away"], odds_2),
    ]:
        p_book = decimali_a_prob(odd)
        diff = (p_mod - p_book) * 100
        rows.append({
            "Mercato": "1X2",
            "Esito": lab,
            "Prob modello %": round(p_mod*100, 2),
            "Prob quota %": round(p_book*100, 2),
            "Δ pp": round(diff, 2),
        })

    # Over/Under
    if not anomalo_ou:
        if odds_over25 and odds_over25 > 1:
            p_mod = ris_co["over_25"]
            p_book = decimali_a_prob(odds_over25)
            diff = (p_mod - p_book) * 100
            rows.append({
                "Mercato": "Over/Under 2.5",
                "Esito": "Over 2.5",
                "Prob modello %": round(p_mod*100, 2),
                "Prob quota %": round(p_book*100, 2),
                "Δ pp": round(diff, 2),
            })

        if odds_under25 and odds_under25 > 1:
            p_mod = ris_co["under_25"]
            p_book = decimali_a_prob(odds_under25)
            diff = (p_mod - p_book) * 100
            rows.append({
                "Mercato": "Over/Under 2.5",
                "Esito": "Under 2.5",
                "Prob modello %": round(p_mod*100, 2),
                "Prob quota %": round(p_book*100, 2),
                "Δ pp": round(diff, 2),
            })

    # BTTS
    if odds_btts and odds_btts > 1:
        p_mod = ris_co["btts"]
        p_book = decimali_a_prob(odds_btts)
        diff = (p_mod - p_book) * 100
        rows.append({
            "Mercato": "Entrambe segnano",
            "Esito": "Sì",
            "Prob modello %": round(p_mod*100, 2),
            "Prob quota %": round(p_book*100, 2),
            "Δ pp": round(diff, 2),
        })

    df_vf = pd.DataFrame(rows)
    df_vf_pos = df_vf[df_vf["Δ pp"] >= 2]
    st.dataframe(df_vf_pos if not df_vf_pos.empty else df_vf)

    # --------------------------------------------------------
    # Espansioni
    # --------------------------------------------------------
    with st.expander("① Probabilità principali"):
        st.write(f"BTTS: {ris_co['btts']*100:.1f}%")
        st.write(f"No Goal: {(1 - ris_co['btts'])*100:.1f}%")
        st.write(f"GG + Over 2.5: {ris_co['gg_over25']*100:.1f}%")

    with st.expander("② Esito finale e parziale"):
        st.write(f"Vittoria Casa: {ris_co['p_home']*100:.1f}% (apertura {ris_ap['p_home']*100:.1f}%)")
        st.write(f"Pareggio: {ris_co['p_draw']*100:.1f}% (apertura {ris_ap['p_draw']*100:.1f}%)")
        st.write(f"Vittoria Trasferta: {ris_co['p_away']*100:.1f}% (apertura {ris_ap['p_away']*100:.1f}%)")
        st.write("Double Chance:")
        for k, v in ris_co["dc"].items():
            st.write(f"- {k}: {v*100:.1f}%")

    with st.expander("③ Over / Under"):
        st.write(f"Over 1.5: {ris_co['over_15']*100:.1f}%")
        st.write(f"Under 1.5: {ris_co['under_15']*100:.1f}%")
        st.write(f"Over 2.5: {ris_co['over_25']*100:.1f}%")
        st.write(f"Under 2.5: {ris_co['under_25']*100:.1f}%")
        st.write(f"Over 3.5: {ris_co['over_35']*100:.1f}%")
        st.write(f"Under 3.5: {ris_co['under_35']*100:.1f}%")
        st.write(f"Over 0.5 HT: {ris_co['over_05_ht']*100:.1f}%")

    with st.expander("④ Gol pari/dispari"):
        st.write(f"Gol pari FT (classico): {ris_co['even_ft']*100:.1f}%")
        st.write(f"Gol dispari FT (classico): {ris_co['odd_ft']*100:.1f}%")
        st.write(f"Gol pari HT: {ris_co['even_ht']*100:.1f}%")
        st.write(f"Gol dispari HT: {ris_co['odd_ht']*100:.1f}%")

    with st.expander("⑤ Clean sheet e info modello"):
        st.write(f"Clean Sheet Casa: {ris_co['cs_home']*100:.1f}%")
        st.write(f"Clean Sheet Trasferta: {ris_co['cs_away']*100:.1f}%")
        st.write(f"Clean Sheet qualcuno (No Goal): {ris_co['clean_sheet_qualcuno']*100:.1f}%")
        st.write(f"λ Casa: {ris_co['lambda_home']:.3f}")
        st.write(f"λ Trasferta: {ris_co['lambda_away']:.3f}")
        st.write(f"Entropia Casa: {ris_co['ent_home']:.3f}")
        st.write(f"Entropia Trasferta: {ris_co['ent_away']:.3f}")

    with st.expander("⑥ Multigol Casa"):
        st.write({k: f"{v*100:.1f}%" for k, v in ris_co["multigol_home"].items()})

    with st.expander("⑦ Multigol Trasferta"):
        st.write({k: f"{v*100:.1f}%" for k, v in ris_co["multigol_away"].items()})

    with st.expander("⑧ Vittoria con margine"):
        st.write(f"Vittoria casa almeno 2 gol scarto: {ris_co['marg2']*100:.1f}%")
        st.write(f"Vittoria casa almeno 3 gol scarto: {ris_co['marg3']*100:.1f}%")

    with st.expander("⑨ Combo mercati (1&Over, DC+GG, ecc.)"):
        for k, v in ris_co["combo_book"].items():
            st.write(f"{k}: {v*100:.1f}%")

    with st.expander("⑩ Top 10 risultati esatti"):
        for h, a, p in ris_co["top10"]:
            st.write(f"{h}-{a}: {p:.1f}%")

    with st.expander("⑪ Combo Multigol Filtrate (>=50%)"):
        for c in ris_co["combo_ft_filtrate"]:
            st.write(f"{c['combo']}: {c['prob']*100:.1f}%")

    with st.expander("⑫ Combo Over HT + Over FT"):
        for k, v in ris_co["combo_ht_ft"].items():
            st.write(f"{k}: {v*100:.1f}%")

    with st.expander("⑬ Statistiche globali (pari/dispari & coperture)"):
        st.write(f"Somma gol DISPARI (robusta): {ris_co['odd_mass']*100:.1f}%")
        st.write(f"Somma gol PARI (robusta): {ris_co['even_mass2']*100:.1f}%")
        st.write(f"Copertura 0–2 gol (FT): {ris_co['cover_0_2']*100:.1f}%")
        st.write(f"Copertura 0–3 gol (FT): {ris_co['cover_0_3']*100:.1f}%")
        st.caption("Queste usano la distribuzione dei gol totali, quindi sono più stabili.")

    # --------------------------------------------------------
    # salvataggio nel CSV
    # --------------------------------------------------------
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "match": match_name,
        "match_date": date.today().isoformat(),
        "spread_ap": spread_ap,
        "total_ap": total_ap,
        "spread_co": spread_co,
        "total_co": total_co,
        "odds_1": odds_1,
        "odds_x": odds_x,
        "odds_2": odds_2,
        "odds_over25": odds_over25,
        "odds_under25": odds_under25,
        "odds_dnb_home": odds_dnb_home,
        "odds_dnb_away": odds_dnb_away,
        "odds_btts": odds_btts,
        "p_home": round(ris_co["p_home"]*100, 2),
        "p_draw": round(ris_co["p_draw"]*100, 2),
        "p_away": round(ris_co["p_away"]*100, 2),
        "btts": round(ris_co["btts"]*100, 2),
        "over_25": round(ris_co["over_25"]*100, 2),
        "affidabilita": aff,
        "confidence_globale": global_conf,
        "market_pressure_index": mpi,
        "esito_modello": max(
            [("1", ris_co["p_home"]), ("X", ris_co["p_draw"]), ("2", ris_co["p_away"])],
            key=lambda x: x[1]
        )[0],
        "esito_reale": "",
        "risultato_reale": "",
        "match_ok": "",
        "odd_mass": round(ris_co["odd_mass"]*100, 2),
        "even_mass2": round(ris_co["even_mass2"]*100, 2),
        "cover_0_2": round(ris_co["cover_0_2"]*100, 2),
        "cover_0_3": round(ris_co["cover_0_3"]*100, 2),
    }

    try:
        if os.path.exists(ARCHIVE_FILE):
            df_old = pd.read_csv(ARCHIVE_FILE)
            df_new = pd.concat([df_old, pd.DataFrame([row])], ignore_index=True)
            df_new.to_csv(ARCHIVE_FILE, index=False)
        else:
            pd.DataFrame([row]).to_csv(ARCHIVE_FILE, index=False)
        st.success("📁 Analisi salvata in storico_analisi.csv")
    except Exception as e:
        st.warning(f"Non sono riuscito a salvare l'analisi: {e}")

# ============================================================
#       🔢 SEZIONE FACOLTATIVA: COMBINATORE DI PROBABILITÀ
# ============================================================

st.markdown("---")
st.subheader("🧮 Calcola combinazioni di mercati (facoltativo)")

def calcola_combinata(prob_a, prob_b, correlazione=0.1):
    """
    Calcola la probabilità combinata di due eventi, tenendo conto di una
    correlazione positiva (default 10%).
    Esempio: 1X + Under 3.5
    """
    try:
        prob_a = float(prob_a) / 100.0
        prob_b = float(prob_b) / 100.0
    except Exception:
        st.warning("Inserisci due percentuali valide.")
        return None

    prob_b_cond = min(prob_b + correlazione, 1.0)
    prob_comb = prob_a * prob_b_cond
    return round(prob_comb * 100, 2)

col_ca, col_cb, col_cc = st.columns([1, 1, 1])
with col_ca:
    prob_a = st.number_input("Probabilità evento A (%)", value=81.5, step=0.1)
with col_cb:
    prob_b = st.number_input("Probabilità evento B (%)", value=72.0, step=0.1)
with col_cc:
    correl = st.slider("Correlazione (%)", min_value=0, max_value=20, value=10, step=1) / 100.0

if st.button("Calcola combinazione"):
    res = calcola_combinata(prob_a, prob_b, correl)
    if res is not None:
        st.success(f"Probabilità combinata ≈ **{res}%**")
        st.caption(
            "Esempio: se A = 1X e B = Under 3.5 e sai che sono mercati abbastanza legati, "
            "metti 10–15% di correlazione per non sottostimare."
        )

# ============================================================
#           AGGIORNA RISULTATI REALI (API-FOOTBALL)
# ============================================================

st.subheader("🔄 Aggiorna risultati reali nello storico (API-Football)")

if st.button("Recupera risultati degli ultimi 3 giorni"):
    if not os.path.exists(ARCHIVE_FILE):
        st.warning("Non c'è ancora uno storico da aggiornare.")
    else:
        df = pd.read_csv(ARCHIVE_FILE)
        today = date.today()
        giorni_da_controllare = [
            (today - timedelta(days=i)).isoformat() for i in range(0, 4)
        ]
        fixtures_by_day = {}
        for d in giorni_da_controllare:
            fixtures_by_day[d] = apifootball_get_fixtures_by_date(d)

        results_map = {}
        for d, fixtures in fixtures_by_day.items():
            for f in fixtures:
                if f["fixture"]["status"]["short"] in ["FT", "AET", "PEN"]:
                    home = f["teams"]["home"]["name"]
                    away = f["teams"]["away"]["name"]
                    key = f"{home} vs {away}".strip().lower()
                    goals_home = f["goals"]["home"]
                    goals_away = f["goals"]["away"]
                    results_map[key] = (goals_home, goals_away)

        updated = 0
        for idx, row in df.iterrows():
            key_row = str(row.get("match", "")).strip().lower()
            if (
                key_row in results_map
                and (pd.isna(row.get("risultato_reale")) or row.get("risultato_reale") == "")
            ):
                gh, ga = results_map[key_row]
                if gh is None or ga is None:
                    continue
                if gh > ga:
                    esito_real = "1"
                elif gh == ga:
                    esito_real = "X"
                else:
                    esito_real = "2"
                df.at[idx, "risultato_reale"] = f"{gh}-{ga}"
                df.at[idx, "esito_reale"] = esito_real
                pred = row.get("esito_modello", "")
                if pred != "" and esito_real != "":
                    df.at[idx, "match_ok"] = 1 if pred == esito_real else 0
                updated += 1

        df.to_csv(ARCHIVE_FILE, index=False)
        st.success(f"Aggiornamento completato. Partite aggiornate: {updated}")
        st.dataframe(df.tail(30))