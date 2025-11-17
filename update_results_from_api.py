#!/usr/bin/env python3
"""
Script per aggiornare risultati partite da API
==============================================

Aggiorna risultati partite finite nel database.
"""

import os
import requests
import logging
from datetime import datetime, timedelta
from betting_results_tracker import BettingResultsTracker
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def update_finished_matches(tracker: BettingResultsTracker):
    """Aggiorna risultati partite finite"""
    
    # Ottieni opportunità pending
    conn = tracker.db_path
    import sqlite3
    conn_db = sqlite3.connect(conn)
    conn_db.row_factory = sqlite3.Row
    cursor = conn_db.cursor()
    
    # Partite con match_date passata e risultato non ancora aggiornato
    yesterday = datetime.now() - timedelta(days=1)
    cursor.execute("""
        SELECT DISTINCT match_id, home_team, away_team, match_date, market
        FROM opportunities
        WHERE (result IS NULL OR result = 'P')
        AND match_date < ?
    """, (datetime.now(),))
    
    pending_matches = cursor.fetchall()
    conn_db.close()
    
    logger.info(f"Trovate {len(pending_matches)} partite da aggiornare")
    
    # Prova a ottenere risultati da API-Football
    api_key = os.getenv("API_FOOTBALL_KEY", "")
    if not api_key:
        logger.warning("API_FOOTBALL_KEY non configurata, skip aggiornamento risultati")
        return
    
    updated = 0
    for match_row in pending_matches:
        match_id = match_row['match_id']
        home = match_row['home_team']
        away = match_row['away_team']
        market = match_row['market']
        
        try:
            # Cerca risultato partita (semplificato - in produzione usare API reale)
            # Per ora, salta (richiede integrazione con API-Football fixtures)
            logger.debug(f"Skip aggiornamento {home} vs {away} (richiede integrazione API)")
            continue
        except Exception as e:
            logger.warning(f"Errore aggiornamento {match_id}: {e}")
            continue
    
    logger.info(f"✅ Aggiornate {updated} partite")


if __name__ == '__main__':
    tracker = BettingResultsTracker()
    update_finished_matches(tracker)

