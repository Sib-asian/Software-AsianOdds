#!/usr/bin/env python3
"""
Script per visualizzare le partite attualmente monitorate dal sistema
"""

import os
import sys
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

# Aggiungi path per import
sys.path.insert(0, str(Path(__file__).parent))

def extract_matches_from_logs():
    """Estrae partite monitorate dai log recenti"""
    log_dir = Path(__file__).parent / "logs"
    if not log_dir.exists():
        return []
    
    # Trova log più recente
    log_files = sorted(log_dir.glob("automation_service_*.log"), key=lambda x: x.stat().st_mtime, reverse=True)
    if not log_files:
        return []
    
    log_file = log_files[0]
    matches_found = {}
    
    # Pattern per trovare partite nei log
    patterns = [
        r'ANALYZING:\s+(.+?)\s+vs\s+(.+?)(?:\s|$)',
        r'Found\s+(\d+)\s+matches\s+to\s+monitor',
        r'Score:\s+(\d+)-(\d+)',
        r'(\d+:\d+)\s+(.+?)\s+vs\s+(.+?)(?:\s|$)',
    ]
    
    # Leggi ultime 500 righe del log
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            recent_lines = lines[-500:] if len(lines) > 500 else lines
            
            for line in recent_lines:
                # Cerca partite analizzate - pattern più robusto
                if 'ANALYZING:' in line or 'analyzing:' in line.lower():
                    # Pattern: ANALYZING: Team1 vs Team2
                    match = re.search(r'ANALYZING:\s+(.+?)\s+vs\s+(.+?)(?:\s|$|\(|\[|-)', line, re.IGNORECASE)
                    if match:
                        home = match.group(1).strip()
                        away = match.group(2).strip()
                        # Pulisci nomi (rimuovi timestamp, log level, etc.)
                        home = re.sub(r'^\d{4}-\d{2}-\d{2}.*?INFO\s*-\s*', '', home).strip()
                        away = re.sub(r'\s*-\s*.*$', '', away).strip()
                        key = f"{home} vs {away}"
                        if key not in matches_found and len(home) > 2 and len(away) > 2:
                            matches_found[key] = {
                                'home': home,
                                'away': away,
                                'last_seen': None,
                                'status': 'Unknown',
                                'score': None
                            }
                
                # Cerca score
                score_match = re.search(r'Score:\s+(\d+)-(\d+)', line)
                if score_match:
                    score = f"{score_match.group(1)}-{score_match.group(2)}"
                    # Associa score all'ultima partita trovata
                    if matches_found:
                        last_key = list(matches_found.keys())[-1]
                        matches_found[last_key]['score'] = score
                        matches_found[last_key]['status'] = 'LIVE'
                
                # Cerca timestamp
                time_match = re.search(r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})', line)
                if time_match and matches_found:
                    timestamp = time_match.group(1)
                    last_key = list(matches_found.keys())[-1]
                    matches_found[last_key]['last_seen'] = timestamp
                    
    except Exception as e:
        print(f"⚠️  Errore lettura log: {e}")
    
    return list(matches_found.values())

def check_api_cache():
    """Controlla cache API per partite"""
    cache_file = Path(__file__).parent / "api_cache.db"
    if cache_file.exists():
        try:
            import sqlite3
            conn = sqlite3.connect(cache_file)
            cursor = conn.cursor()
            
            # Prova a leggere dalla cache
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            
            matches = []
            for table in tables:
                try:
                    cursor.execute(f"SELECT * FROM {table[0]} LIMIT 10")
                    rows = cursor.fetchall()
                    if rows:
                        matches.append(f"Table: {table[0]} - {len(rows)} entries")
                except:
                    pass
            
            conn.close()
            return matches
        except Exception as e:
            return [f"Errore cache: {e}"]
    return []

def main():
    print("=" * 70)
    print("📋 CHECKLIST PARTITE MONITORATE")
    print("=" * 70)
    print()
    
    # 1. Estrai partite dai log
    print("🔍 Analizzando log recenti...")
    matches = extract_matches_from_logs()
    
    if matches:
        print(f"\n✅ Trovate {len(matches)} partite monitorate:\n")
        for i, match in enumerate(matches, 1):
            home = match.get('home', 'N/A')
            away = match.get('away', 'N/A')
            score = match.get('score', 'N/A')
            status = match.get('status', 'Unknown')
            last_seen = match.get('last_seen', 'N/A')
            
            print(f"{i}. {home} vs {away}")
            if score != 'N/A':
                print(f"   📊 Score: {score} ({status})")
            if last_seen != 'N/A':
                print(f"   🕐 Ultimo aggiornamento: {last_seen}")
            print()
    else:
        print("⚠️  Nessuna partita trovata nei log recenti")
        print("   (Il sistema potrebbe essere appena partito o non ci sono partite live)")
    
    # 2. Controlla log per statistiche
    print("\n" + "=" * 70)
    print("📊 STATISTICHE SISTEMA")
    print("=" * 70)
    
    log_dir = Path(__file__).parent / "logs"
    if log_dir.exists():
        log_files = sorted(log_dir.glob("automation_service_*.log"), key=lambda x: x.stat().st_mtime, reverse=True)
        if log_files:
            log_file = log_files[0]
            try:
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    recent_lines = lines[-200:] if len(lines) > 200 else lines
                    
                    # Conta occorrenze
                    matches_count = sum(1 for line in recent_lines if 'matches to monitor' in line.lower())
                    opportunities_count = sum(1 for line in recent_lines if 'opportunit' in line.lower() and 'trovat' in line.lower())
                    errors_count = sum(1 for line in recent_lines if 'error' in line.lower() or 'exception' in line.lower())
                    
                    print(f"\n📈 Ultime 200 righe di log:")
                    print(f"   • Cicli di monitoraggio: {matches_count}")
                    print(f"   • Opportunità trovate: {opportunities_count}")
                    print(f"   • Errori: {errors_count}")
                    
                    # Ultima attività
                    if recent_lines:
                        last_line = recent_lines[-1].strip()
                        if last_line:
                            print(f"\n🕐 Ultima attività:")
                            print(f"   {last_line[:100]}...")
                            
            except Exception as e:
                print(f"⚠️  Errore lettura statistiche: {e}")
    
    # 3. Verifica processi
    print("\n" + "=" * 70)
    print("🖥️  STATO PROCESSI")
    print("=" * 70)
    
    try:
        import subprocess
        result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe', '/FO', 'CSV'], 
                              capture_output=True, text=True)
        if 'python.exe' in result.stdout:
            lines = result.stdout.strip().split('\n')
            python_processes = len([l for l in lines if 'python.exe' in l])
            print(f"\n✅ Processi Python attivi: {python_processes}")
        else:
            print("\n⚠️  Nessun processo Python trovato")
    except:
        print("\n⚠️  Impossibile verificare processi")
    
    print("\n" + "=" * 70)
    print("💡 SUGGERIMENTI")
    print("=" * 70)
    print("""
• Per vedere i log in tempo reale: monitor_logs.bat
• Per verificare lo stato: STATO_24H.bat
• I log sono in: logs\\automation_service_*.log
• Il sistema monitora partite ogni 5 minuti (default)
    """)
    
    print("=" * 70)

if __name__ == '__main__':
    main()

