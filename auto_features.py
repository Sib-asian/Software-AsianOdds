"""
AUTO-DETECTION per Advanced Features

Questo modulo gestisce la modalità automatica per le advanced features:
- Auto-detection stile tattico da database squadre
- Auto-detection motivazione da posizione/contesto
- Auto-calcolo fixture congestion da date match
- LEVEL 2: API integration per squadre non in database

Utilizzo:
    from auto_features import auto_detect_all_features

    # LEVEL 1 (Solo Database)
    features = auto_detect_all_features(
        home_team="Inter",
        away_team="Milan",
        league="Serie A",
        match_datetime="2025-01-15T20:45:00",
        position_home=1,
        position_away=2
    )

    # LEVEL 2 (Database + API)
    features = auto_detect_all_features(
        home_team="Midtjylland",
        away_team="Nordsjælland",
        league="Superliga",
        use_api=True  # Enable API fallback
    )
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, Any

logger = logging.getLogger(__name__)

# ============================================================
# API MANAGER INTEGRATION (LEVEL 2)
# ============================================================
try:
    from api_manager import APIManager
    API_MANAGER = APIManager()
    API_AVAILABLE = True
    logger.info("✅ API Manager disponibile (LEVEL 2 attivo)")
except ImportError:
    API_MANAGER = None
    API_AVAILABLE = False
    logger.info("ℹ️ API Manager non disponibile (solo LEVEL 1)")

# ============================================================
# LOAD TEAM PROFILES DATABASE
# ============================================================

def load_team_profiles(json_path: str = "team_profiles.json") -> Dict:
    """
    Carica database squadre da JSON

    Returns:
        Dict con teams, motivation_rules, default_settings e '_file_loaded' (bool)
    """
    try:
        json_file = Path(json_path)
        if not json_file.exists():
            logger.warning(f"⚠️ File {json_path} non trovato, uso defaults")
            logger.warning(f"⚠️ Auto-detection tornerà sempre valori di default senza questo file!")
            logger.warning(f"⚠️ Percorso atteso: {json_file.absolute()}")
            return {
                "teams": {},
                "motivation_rules": {},
                "default_settings": {},
                "_file_loaded": False,
                "_error": f"File non trovato: {json_path}"
            }

        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Add metadata to indicate successful load
        data["_file_loaded"] = True
        data["_error"] = None

        logger.info(f"✅ Team profiles caricati da {json_path}")
        logger.info(f"✅ Caricate {len(data.get('teams', {}))} leghe nel database")
        return data

    except Exception as e:
        logger.error(f"❌ Errore caricamento team profiles: {e}")
        return {
            "teams": {},
            "motivation_rules": {},
            "default_settings": {},
            "_file_loaded": False,
            "_error": str(e)
        }


# Carica al startup (cache globale)
# NOTA: Cache statico, non thread-safe. Se modifichi team_profiles.json durante
# l'esecuzione, usa reload_team_profiles() per ricaricare.
TEAM_PROFILES = load_team_profiles()


def reload_team_profiles(json_path: str = "team_profiles.json") -> Dict:
    """
    Ricarica team profiles da file (invalida cache).

    Utile se team_profiles.json viene modificato durante l'esecuzione.

    Args:
        json_path: Path al file JSON

    Returns:
        Dict aggiornato con teams, motivation_rules, default_settings
    """
    global TEAM_PROFILES
    TEAM_PROFILES = load_team_profiles(json_path)
    logger.info(f"🔄 Team profiles ricaricati da {json_path}")
    return TEAM_PROFILES

# ============================================================
# LEAGUE CODE MAPPING
# ============================================================

LEAGUE_TO_CODE = {
    "Serie A": "ITA_A",
    "Italia Serie A": "ITA_A",
    "Italian Serie A": "ITA_A",
    "Premier League": "ENG_1",
    "English Premier League": "ENG_1",
    "EPL": "ENG_1",
    "La Liga": "ESP_1",
    "Spanish La Liga": "ESP_1",
    "Spain La Liga": "ESP_1",
    "Bundesliga": "GER_1",
    "German Bundesliga": "GER_1",
    "Ligue 1": "FRA_1",
    "French Ligue 1": "FRA_1",
    "France Ligue 1": "FRA_1",
}


def get_league_code(league_name: str) -> str:
    """
    Converte nome lega in codice

    Args:
        league_name: Nome lega (es. "Serie A", "Premier League")

    Returns:
        Codice lega (es. "ITA_A", "ENG_1") o "UNKNOWN"
    """
    if not league_name:
        return "UNKNOWN"

    # Prova match esatto
    if league_name in LEAGUE_TO_CODE:
        return LEAGUE_TO_CODE[league_name]

    # Prova match case-insensitive
    league_lower = league_name.lower()
    for name, code in LEAGUE_TO_CODE.items():
        if name.lower() == league_lower:
            return code

    # Prova match parziale
    for name, code in LEAGUE_TO_CODE.items():
        if name.lower() in league_lower or league_lower in name.lower():
            return code

    logger.warning(f"⚠️ Lega '{league_name}' non trovata, uso UNKNOWN")
    return "UNKNOWN"


# ============================================================
# AUTO-DETECT TACTICAL STYLE
# ============================================================

def auto_detect_tactical_style(team_name: str, league: str, use_api: bool = False) -> str:
    """
    Auto-detect stile tattico da database squadre (LEVEL 1) o API (LEVEL 2)

    Args:
        team_name: Nome squadra (es. "Inter", "Manchester City")
        league: Nome lega (es. "Serie A", "Premier League")
        use_api: Se True, prova API se non trovato in database

    Returns:
        Stile tattico: "Possesso", "Contropiede", "Pressing Alto", "Difensiva"
        Default: "Possesso" se non trovato
    """
    if not team_name:
        return "Possesso"

    league_code = get_league_code(league)
    teams_db = TEAM_PROFILES.get("teams", {})

    if league_code not in teams_db:
        logger.info(f"ℹ️ Lega {league} ({league_code}) non in database")

        # Try API if enabled
        if use_api and API_AVAILABLE:
            return _get_style_from_api(team_name, league)

        logger.info(f"  → Uso Possesso (fallback)")
        return "Possesso"

    league_teams = teams_db[league_code]

    # Prova match esatto
    if team_name in league_teams:
        style = league_teams[team_name].get("style", "Possesso")
        logger.info(f"✅ {team_name}: {style} (exact match)")
        return style

    # Prova match case-insensitive
    team_lower = team_name.lower()
    for team_key, team_data in league_teams.items():
        if team_key.lower() == team_lower:
            style = team_data.get("style", "Possesso")
            logger.info(f"✅ {team_name}: {style} (case-insensitive)")
            return style

    # Prova aliases
    for team_key, team_data in league_teams.items():
        aliases = team_data.get("aliases", [])
        for alias in aliases:
            if alias.lower() == team_lower or alias.lower() in team_lower:
                style = team_data.get("style", "Possesso")
                logger.info(f"✅ {team_name}: {style} (alias '{alias}')")
                return style

    # Fallback: Try API or use default
    if use_api and API_AVAILABLE:
        return _get_style_from_api(team_name, league)

    logger.info(f"ℹ️ {team_name} non trovata in {league}, uso Possesso")
    return "Possesso"


# ============================================================
# API HELPER FUNCTIONS (LEVEL 2)
# ============================================================

def _get_style_from_api(team_name: str, league: str) -> str:
    """
    Get tactical style from API (LEVEL 2)

    Args:
        team_name: Team name
        league: League name

    Returns:
        Tactical style inferred from API data
    """
    try:
        logger.info(f"📡 Trying API for {team_name}...")

        result = API_MANAGER.get_team_context(team_name, league)

        if result["source"] == "api":
            style = result["data"].get("style", "Possesso")
            logger.info(f"✅ API success: {team_name} → {style}")
            return style
        elif result["source"] == "cache":
            style = result["data"].get("style", "Possesso")
            logger.info(f"✅ Cache hit: {team_name} → {style}")
            return style
        else:
            # Fallback within API module
            logger.info(f"⚠️ API fallback for {team_name}")
            return result["data"].get("style", "Possesso")

    except Exception as e:
        logger.error(f"❌ API error for {team_name}: {e}")
        return "Possesso"


# ============================================================
# AUTO-DETECT MOTIVATION
# ============================================================

def auto_detect_motivation(
    team_name: str,
    position: Optional[int] = None,
    points_from_relegation: Optional[int] = None,
    points_from_europe: Optional[int] = None,
    is_derby: bool = False,
    is_cup: bool = False,
    is_end_season: bool = False
) -> str:
    """
    Auto-detect motivazione da contesto

    Args:
        team_name: Nome squadra
        position: Posizione in classifica (1-20)
        points_from_relegation: Punti sopra zona retrocessione
        points_from_europe: Punti sotto zona Europa
        is_derby: True se è un derby
        is_cup: True se è finale di coppa
        is_end_season: True se fine stagione senza obiettivi

    Returns:
        Motivazione: una delle 6 opzioni MOTIVATION_FACTORS
    """
    # Context overrides
    if is_derby:
        logger.info(f"🔥 {team_name}: Derby rilevato → Motivazione Alta")
        return "Derby / Rivalità storica"

    if is_cup:
        logger.info(f"🏆 {team_name}: Finale coppa → Motivazione Massima")
        return "Finale di coppa / Match decisivo"

    if is_end_season:
        logger.info(f"😴 {team_name}: Fine stagione senza obiettivi → Motivazione Bassa")
        return "Fine stagione (nulla in palio)"

    # Position-based detection
    if position is not None:
        rules = TEAM_PROFILES.get("motivation_rules", {}).get("position_based", {})

        # Trova range corretto
        for pos_range, data in rules.items():
            if "-" in pos_range:
                start, end = map(int, pos_range.split("-"))
                if start <= position <= end:
                    motivation = data["motivation"]
                    reason = data["reason"]
                    logger.info(f"📊 {team_name}: Pos {position} → {motivation} ({reason})")
                    return motivation

    # Points-based detection (più accurato se disponibile)
    if points_from_relegation is not None and points_from_relegation <= 5:
        logger.info(f"🚨 {team_name}: A {points_from_relegation}pts da retrocessione → Lotta Salvezza")
        return "Lotta Salvezza (retrocessione)"

    if points_from_europe is not None and points_from_europe <= 3:
        logger.info(f"⚡ {team_name}: A {points_from_europe}pts da Europa → Lotta Champions")
        return "Lotta Champions (4° posto)"

    # Default: Normale
    logger.info(f"ℹ️ {team_name}: Nessun contesto speciale → Normale")
    return "Normale"


# ============================================================
# AUTO-CALCULATE FIXTURE CONGESTION
# ============================================================

def auto_calculate_fixture_congestion(
    match_datetime: Optional[str] = None,
    last_match_datetime: Optional[str] = None,
    next_important_match_datetime: Optional[str] = None
) -> Tuple[int, int]:
    """
    Auto-calcola fixture congestion da date match

    Args:
        match_datetime: Data match corrente (ISO format)
        last_match_datetime: Data ultimo match (ISO format)
        next_important_match_datetime: Data prossimo match importante (ISO format)

    Returns:
        (days_since_last, days_until_next)
        Default: (7, 7) se date non disponibili
    """
    try:
        if not match_datetime:
            logger.info("ℹ️ match_datetime non fornito, uso defaults (7, 7)")
            return 7, 7

        # Parse match date
        match_dt = datetime.fromisoformat(match_datetime.replace("Z", "+00:00"))

        # Calculate days since last match
        days_since = 7  # default
        if last_match_datetime:
            last_dt = datetime.fromisoformat(last_match_datetime.replace("Z", "+00:00"))
            days_raw = (match_dt - last_dt).days
            if days_raw < 0:
                logger.warning(f"⚠️ Last match è nel futuro (date invertite), uso default (7)")
                days_since = 7
            else:
                days_since = max(2, min(21, days_raw))
            logger.info(f"📅 Days since last match: {days_since}")

        # Calculate days until next important match
        days_until = 7  # default
        if next_important_match_datetime:
            next_dt = datetime.fromisoformat(next_important_match_datetime.replace("Z", "+00:00"))
            days_raw = (next_dt - match_dt).days
            if days_raw < 0:
                logger.warning(f"⚠️ Next match è nel passato (date invertite), uso default (7)")
                days_until = 7
            else:
                days_until = max(2, min(14, days_raw))
            logger.info(f"📅 Days until next important match: {days_until}")

        return days_since, days_until

    except Exception as e:
        logger.warning(f"⚠️ Errore calcolo fixture congestion: {e}, uso defaults")
        return 7, 7


# ============================================================
# AUTO-DETECT ALL FEATURES (Main Function)
# ============================================================

def auto_detect_all_features(
    home_team: str,
    away_team: str,
    league: str,
    match_datetime: Optional[str] = None,
    position_home: Optional[int] = None,
    position_away: Optional[int] = None,
    points_from_relegation_home: Optional[int] = None,
    points_from_relegation_away: Optional[int] = None,
    points_from_europe_home: Optional[int] = None,
    points_from_europe_away: Optional[int] = None,
    is_derby: bool = False,
    is_cup: bool = False,
    is_end_season: bool = False,
    last_match_datetime_home: Optional[str] = None,
    last_match_datetime_away: Optional[str] = None,
    next_important_match_datetime_home: Optional[str] = None,
    next_important_match_datetime_away: Optional[str] = None,
    use_api: bool = False
) -> Dict[str, Any]:
    """
    Auto-detect TUTTE le advanced features in un colpo solo

    Args:
        home_team: Nome squadra casa
        away_team: Nome squadra trasferta
        league: Nome lega
        match_datetime: Data match corrente (ISO format)
        position_home/away: Posizione in classifica (1-20)
        points_from_relegation_home/away: Punti sopra retrocessione
        points_from_europe_home/away: Punti sotto Europa
        is_derby: True se derby
        is_cup: True se coppa
        is_end_season: True se fine stagione
        last_match_datetime_home/away: Data ultimo match
        next_important_match_datetime_home/away: Data prossimo match importante
        use_api: Se True, usa API per squadre non in database (LEVEL 2)

    Returns:
        Dict con tutti i parametri per advanced features:
        {
            'style_home': 'Possesso',
            'style_away': 'Contropiede',
            'motivation_home': 'Lotta Champions (4° posto)',
            'motivation_away': 'Normale',
            'days_since_home': 3,
            'days_since_away': 7,
            'days_until_home': 7,
            'days_until_away': 4,
            'apply_constraints': True,
            'apply_calibration_enabled': True,
            'use_precision_math': True,
            'api_used': False,  # True if API was called
            'api_calls_count': 0  # Number of API calls made
        }
    """
    mode_str = "LEVEL 2 (DB + API)" if use_api else "LEVEL 1 (DB only)"
    logger.info(f"🤖 AUTO-DETECTION [{mode_str}]: {home_team} vs {away_team} ({league})")

    # Check if team_profiles.json was loaded successfully
    if not TEAM_PROFILES.get("_file_loaded", False):
        error_msg = TEAM_PROFILES.get("_error", "Unknown error")
        logger.error(f"❌ team_profiles.json non caricato: {error_msg}")
        logger.error("❌ AUTO-DETECTION userà solo valori di default!")
        logger.error("❌ Per funzionalità complete, fornisci team_profiles.json")

    api_calls_count = 0

    # 1. Tactical Styles
    style_home = auto_detect_tactical_style(home_team, league, use_api=use_api)
    style_away = auto_detect_tactical_style(away_team, league, use_api=use_api)

    # Track if API was actually used
    if use_api and API_AVAILABLE and API_MANAGER:
        # Get actual usage count (simplified - would need proper tracking)
        api_calls_count = 2 if (style_home != "Possesso" or style_away != "Possesso") else 0

    # 2. Motivations
    motivation_home = auto_detect_motivation(
        home_team,
        position=position_home,
        points_from_relegation=points_from_relegation_home,
        points_from_europe=points_from_europe_home,
        is_derby=is_derby,
        is_cup=is_cup,
        is_end_season=is_end_season
    )

    motivation_away = auto_detect_motivation(
        away_team,
        position=position_away,
        points_from_relegation=points_from_relegation_away,
        points_from_europe=points_from_europe_away,
        is_derby=is_derby,
        is_cup=is_cup,
        is_end_season=is_end_season
    )

    # 3. Fixture Congestion
    days_since_home, days_until_home = auto_calculate_fixture_congestion(
        match_datetime,
        last_match_datetime_home,
        next_important_match_datetime_home
    )

    days_since_away, days_until_away = auto_calculate_fixture_congestion(
        match_datetime,
        last_match_datetime_away,
        next_important_match_datetime_away
    )

    # 4. Default Options
    defaults = TEAM_PROFILES.get("default_settings", {})

    result = {
        'style_home': style_home,
        'style_away': style_away,
        'motivation_home': motivation_home,
        'motivation_away': motivation_away,
        'days_since_home': days_since_home,
        'days_since_away': days_since_away,
        'days_until_home': days_until_home,
        'days_until_away': days_until_away,
        'apply_constraints': defaults.get('apply_constraints', True),
        'apply_calibration_enabled': defaults.get('apply_calibration_enabled', True),
        'use_precision_math': defaults.get('use_precision_math', True),
        'api_used': use_api and API_AVAILABLE,
        'api_calls_count': api_calls_count
    }

    logger.info(f"✅ AUTO-DETECTION completata [{mode_str}]: {result}")
    return result


# ============================================================
# UTILITY: Detect Derby
# ============================================================

def is_derby_match(home_team: str, away_team: str, league: str) -> bool:
    """
    Rileva se è un derby basato su città/rivalità note

    Args:
        home_team: Squadra casa
        away_team: Squadra trasferta
        league: Lega

    Returns:
        True se è un derby
    """
    # Derby keywords nel nome
    derby_keywords = ["derby", "rivalità", "classico", "stracittadina"]
    combined = f"{home_team} {away_team}".lower()

    for keyword in derby_keywords:
        if keyword in combined:
            return True

    # Derby noti per città
    known_derbies = {
        "ITA_A": [
            ("Inter", "Milan"),  # Derby di Milano
            ("Roma", "Lazio"),   # Derby di Roma
            ("Juventus", "Torino"),  # Derby di Torino
        ],
        "ENG_1": [
            ("Manchester United", "Manchester City"),  # Manchester Derby
            ("Arsenal", "Tottenham"),  # North London Derby
            ("Liverpool", "Everton"),  # Merseyside Derby
        ],
        "ESP_1": [
            ("Barcelona", "Real Madrid"),  # El Clasico
            ("Real Madrid", "Atletico Madrid"),  # Madrid Derby
            ("Sevilla", "Real Betis"),  # Seville Derby
        ],
        "GER_1": [
            ("Bayern Munich", "Borussia Dortmund"),  # Der Klassiker
            ("Borussia Dortmund", "Schalke"),  # Revierderby
        ],
        "FRA_1": [
            ("Paris Saint-Germain", "Marseille"),  # Le Classique
            ("Lyon", "Saint-Etienne"),  # Derby Rhone-Alpes
        ]
    }

    league_code = get_league_code(league)
    if league_code in known_derbies:
        for team1, team2 in known_derbies[league_code]:
            if (team1.lower() in home_team.lower() and team2.lower() in away_team.lower()) or \
               (team2.lower() in home_team.lower() and team1.lower() in away_team.lower()):
                logger.info(f"🔥 Derby rilevato: {home_team} vs {away_team}")
                return True

    return False


# ============================================================
# TEST (se eseguito direttamente)
# ============================================================

if __name__ == "__main__":
    # Test 1: Derby di Milano
    print("=" * 70)
    print("TEST 1: Derby di Milano (Inter vs Milan)")
    print("=" * 70)

    features = auto_detect_all_features(
        home_team="Inter",
        away_team="Milan",
        league="Serie A",
        match_datetime="2025-01-15T20:45:00",
        position_home=1,
        position_away=3,
        is_derby=True
    )

    print(f"\nRisultato:")
    for key, value in features.items():
        print(f"  {key}: {value}")

    # Test 2: Match normale con fixture congestion
    print("\n" + "=" * 70)
    print("TEST 2: Liverpool vs Brighton (fixture congestion)")
    print("=" * 70)

    features = auto_detect_all_features(
        home_team="Liverpool",
        away_team="Brighton",
        league="Premier League",
        match_datetime="2025-01-20T15:00:00",
        position_home=2,
        position_away=8,
        last_match_datetime_home="2025-01-17T20:00:00",  # 3 giorni fa
        last_match_datetime_away="2025-01-13T15:00:00",  # 7 giorni fa
        next_important_match_datetime_home="2025-01-24T20:45:00"  # Champions fra 4gg
    )

    print(f"\nRisultato:")
    for key, value in features.items():
        print(f"  {key}: {value}")

    # Test 3: Lotta salvezza
    print("\n" + "=" * 70)
    print("TEST 3: Lotta salvezza (Cagliari vs Empoli)")
    print("=" * 70)

    features = auto_detect_all_features(
        home_team="Cagliari",
        away_team="Empoli",
        league="Serie A",
        match_datetime="2025-02-01T18:00:00",
        position_home=18,
        position_away=16,
        points_from_relegation_home=2,
        points_from_relegation_away=4
    )

    print(f"\nRisultato:")
    for key, value in features.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 70)
    print("✅ TUTTI I TEST COMPLETATI")
    print("=" * 70)
