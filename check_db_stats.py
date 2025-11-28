#!/usr/bin/env python3
"""
Script per consultare statistiche database su Render
Uso: python3 check_db_stats.py
"""

import sqlite3
import os
from pathlib import Path
from datetime import datetime

def check_database(db_name, queries):
    """Controlla un database e esegue query"""
    # Cerca il database in /data (Render) o directory corrente (locale)
    if os.path.exists('/data') and os.path.isdir('/data'):
        db_path = f'/data/{db_name}'
    else:
        db_path = db_name

    if not os.path.exists(db_path):
        print(f"❌ Database non trovato: {db_path}")
        return

    file_size = os.path.getsize(db_path)
    file_size_mb = file_size / (1024 * 1024)

    print(f"\n{'='*70}")
    print(f"📊 DATABASE: {db_name}")
    print(f"📁 Path: {db_path}")
    print(f"💾 Size: {file_size_mb:.2f} MB ({file_size:,} bytes)")
    print('='*70)

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        for query_name, query_sql in queries.items():
            print(f"\n🔍 {query_name}")
            print("-"*70)
            cursor.execute(query_sql)
            rows = cursor.fetchall()

            if rows:
                for row in rows:
                    print(f"   {row}")
            else:
                print("   (nessun dato)")

        conn.close()

    except Exception as e:
        print(f"❌ Errore: {e}")


def main():
    print("\n" + "="*70)
    print("🔍 STATISTICHE DATABASE - RENDER")
    print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    # 1. Signal Quality Learning
    check_database('signal_quality_learning.db', {
        'Total segnali tracciati': 'SELECT COUNT(*) as total FROM signal_records',
        'Segnali approvati vs bloccati': '''
            SELECT
                SUM(CASE WHEN was_approved = 1 THEN 1 ELSE 0 END) as approvati,
                SUM(CASE WHEN was_blocked = 1 THEN 1 ELSE 0 END) as bloccati
            FROM signal_records
        ''',
        'Segnali con risultato noto': '''
            SELECT
                COUNT(*) as totale,
                SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as corretti,
                SUM(CASE WHEN was_correct = 0 THEN 1 ELSE 0 END) as sbagliati,
                ROUND(SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as accuracy_pct
            FROM signal_records
            WHERE was_correct IS NOT NULL
        ''',
        'Ultimi 5 segnali': '''
            SELECT
                match_id,
                market,
                minute,
                ROUND(quality_score, 1) as quality,
                was_approved,
                was_correct,
                timestamp
            FROM signal_records
            ORDER BY id DESC
            LIMIT 5
        ''',
        'Parametri appresi': 'SELECT parameter_name, ROUND(parameter_value, 4) FROM learned_parameters',
        'Performance metrics': '''
            SELECT
                date,
                total_signals,
                approved_signals,
                ROUND(precision, 3) as prec,
                ROUND(recall, 3) as rec,
                ROUND(accuracy, 3) as acc
            FROM performance_metrics
            ORDER BY date DESC
            LIMIT 5
        '''
    })

    # 2. Betting Results
    check_database('betting_results.db', {
        'Total opportunità': 'SELECT COUNT(*) FROM opportunities',
        'Risultati': '''
            SELECT
                COUNT(*) as totale,
                SUM(CASE WHEN is_winner = 1 THEN 1 ELSE 0 END) as vinte,
                SUM(CASE WHEN is_winner = 0 THEN 1 ELSE 0 END) as perse,
                SUM(CASE WHEN result IS NULL OR result = "P" THEN 1 ELSE 0 END) as pending
            FROM opportunities
        ''',
        'Profit/Loss totale': '''
            SELECT
                ROUND(SUM(COALESCE(profit_loss, 0)), 2) as total_pl,
                ROUND(SUM(recommended_stake), 2) as total_stake,
                ROUND(SUM(COALESCE(profit_loss, 0)) * 100.0 / NULLIF(SUM(recommended_stake), 0), 2) as roi_pct
            FROM opportunities
        ''',
        'Ultime 5 opportunità': '''
            SELECT
                home_team,
                away_team,
                market,
                ROUND(odds, 2) as odds,
                result,
                is_winner,
                ROUND(COALESCE(profit_loss, 0), 2) as pl
            FROM opportunities
            ORDER BY id DESC
            LIMIT 5
        ''',
        'Performance per market': '''
            SELECT
                market,
                COUNT(*) as count,
                SUM(CASE WHEN is_winner = 1 THEN 1 ELSE 0 END) as wins,
                ROUND(SUM(CASE WHEN is_winner = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as win_rate,
                ROUND(SUM(COALESCE(profit_loss, 0)), 2) as pl
            FROM opportunities
            WHERE result IS NOT NULL AND result != "P"
            GROUP BY market
            ORDER BY count DESC
            LIMIT 10
        '''
    })

    # 3. Bankroll
    check_database('bankroll.db', {
        'Bankroll attuale': '''
            SELECT
                ROUND(amount, 2) as bankroll,
                change_reason,
                date
            FROM bankroll_history
            ORDER BY id DESC
            LIMIT 1
        ''',
        'Ultimi 10 movimenti': '''
            SELECT
                ROUND(amount, 2) as bankroll,
                ROUND(COALESCE(change_amount, 0), 2) as change,
                change_reason,
                note,
                date
            FROM bankroll_history
            ORDER BY id DESC
            LIMIT 10
        ''',
        'ROI totale': '''
            SELECT
                ROUND((SELECT amount FROM bankroll_history ORDER BY id DESC LIMIT 1), 2) as current,
                ROUND((SELECT amount FROM bankroll_history ORDER BY id ASC LIMIT 1), 2) as initial,
                ROUND(
                    ((SELECT amount FROM bankroll_history ORDER BY id DESC LIMIT 1) -
                     (SELECT amount FROM bankroll_history ORDER BY id ASC LIMIT 1)) * 100.0 /
                    (SELECT amount FROM bankroll_history ORDER BY id ASC LIMIT 1), 2
                ) as roi_pct
            FROM bankroll_history
            LIMIT 1
        '''
    })

    # 4. API Cache
    check_database('api_cache.db', {
        'Cache entries': '''
            SELECT
                (SELECT COUNT(*) FROM team_cache) as team_cache,
                (SELECT COUNT(*) FROM predictions_cache) as predictions,
                (SELECT COUNT(*) FROM over_markets_cache) as over_markets
        ''',
        'API usage oggi': '''
            SELECT
                provider,
                calls
            FROM api_usage
            WHERE date = DATE('now')
        ''',
        'Cache hits vs misses': '''
            SELECT
                date,
                hits,
                misses,
                ROUND(hits * 100.0 / (hits + misses), 2) as hit_rate_pct
            FROM cache_stats
            ORDER BY date DESC
            LIMIT 5
        '''
    })

    print("\n" + "="*70)
    print("✅ CONTROLLO COMPLETATO")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
