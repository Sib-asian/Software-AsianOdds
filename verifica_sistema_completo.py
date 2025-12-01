#!/usr/bin/env python3
"""
Script per verificare lo stato completo del sistema
"""
import os
import sys
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

def verifica_partite_live():
    """Verifica 1: Partite live in questo momento"""
    print("=" * 70)
    print("🔍 VERIFICA 1: PARTITE LIVE IN QUESTO MOMENTO")
    print("=" * 70)
    
    api_key = os.getenv('RAPIDAPI_KEY') or '95c43f936816cd4389a747fd2cfe061a'
    headers = {
        'x-rapidapi-key': api_key,
        'x-rapidapi-host': 'api-football-v1.p.rapidapi.com'
    }
    url = 'https://api-football-v1.p.rapidapi.com/v3/fixtures'
    params = {'live': 'all'}
    
    try:
        r = requests.get(url, headers=headers, params=params, timeout=10)
        data = r.json()
        
        if 'response' in data:
            live_matches = data['response']
            print(f"\n✅ Trovate {len(live_matches)} partite LIVE\n")
            
            if live_matches:
                print("📋 Prime 10 partite live:")
                for i, match in enumerate(live_matches[:10], 1):
                    home = match['teams']['home']['name']
                    away = match['teams']['away']['name']
                    score_h = match['goals']['home']
                    score_a = match['goals']['away']
                    minute = match['fixture']['status'].get('elapsed', '?')
                    league = match['league']['name']
                    print(f"  {i}. {home} vs {away} - {score_h}-{score_a} al {minute}'")
                    print(f"     Lega: {league}")
                
                # Verifica Champions League femminile
                cl_women = [m for m in live_matches if 'women' in m['league']['name'].lower() or 'champions' in m['league']['name'].lower()]
                if cl_women:
                    print(f"\n✅ Trovate {len(cl_women)} partite Champions League femminile!")
            else:
                print("⚠️  Nessuna partita live al momento")
        else:
            print("❌ Errore nella risposta API")
            print(f"   Risposta: {data}")
    except Exception as e:
        print(f"❌ Errore: {e}")
        import traceback
        traceback.print_exc()

def verifica_analisi_sistema():
    """Verifica 2: Se il sistema sta analizzando partite"""
    print("\n" + "=" * 70)
    print("🔍 VERIFICA 2: SE IL SISTEMA STA ANALIZZANDO PARTITE")
    print("=" * 70)
    
    log_file = "logs/automation_24h.log"
    if not os.path.exists(log_file):
        print(f"❌ File log non trovato: {log_file}")
        return
    
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    # Cerca informazioni su partite analizzate
    matches_found = []
    cycles = []
    
    for line in lines[-500:]:  # Ultime 500 righe
        if 'Found' in line and 'matches to monitor' in line:
            matches_found.append(line.strip())
        if 'Running analysis cycle' in line:
            cycles.append(line.strip())
        if 'partite' in line.lower() and ('trovate' in line.lower() or 'found' in line.lower()):
            matches_found.append(line.strip())
    
    print(f"\n📊 Ultimi cicli di analisi: {len(cycles)}")
    if cycles:
        print("   Ultimi 3 cicli:")
        for cycle in cycles[-3:]:
            print(f"   • {cycle}")
    
    print(f"\n📊 Partite trovate: {len(matches_found)}")
    if matches_found:
        print("   Ultime 5 occorrenze:")
        for match in matches_found[-5:]:
            print(f"   • {match}")
    
    # Verifica ultime opportunità
    opportunities = [l for l in lines[-200:] if 'opportunity' in l.lower() and ('found' in l.lower() or 'trovate' in l.lower())]
    if opportunities:
        print(f"\n📊 Ultime opportunità trovate:")
        for opp in opportunities[-3:]:
            print(f"   • {opp.strip()}")

def verifica_soglie():
    """Verifica 3: Analisi soglie e possibili modifiche"""
    print("\n" + "=" * 70)
    print("🔍 VERIFICA 3: ANALISI SOGLIE E POSSIBILI MODIFICHE")
    print("=" * 70)
    
    log_file = "logs/automation_24h.log"
    if not os.path.exists(log_file):
        print(f"❌ File log non trovato: {log_file}")
        return
    
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    # Analizza confidence e EV trovati
    confidences = []
    evs = []
    
    for line in lines[-1000:]:  # Ultime 1000 righe
        if 'Confidence:' in line and '/100' in line:
            try:
                conf = int(line.split('Confidence:')[1].split('/')[0].strip())
                confidences.append(conf)
            except:
                pass
        if 'EV:' in line and '%' in line:
            try:
                ev_str = line.split('EV:')[1].split('%')[0].strip()
                ev = float(ev_str)
                evs.append(ev)
            except:
                pass
    
    print("\n📊 SOGLIE ATTUALI:")
    print("   • min_confidence: 75.0%")
    print("   • min_ev: 10.0%")
    
    print("\n📊 SOGLIE PRECEDENTI:")
    print("   • min_confidence: 72.0%")
    print("   • min_ev: 8.0%")
    
    if confidences:
        print(f"\n📊 ANALISI CONFIDENCE TROVATE ({len(confidences)} valori):")
        print(f"   • Min: {min(confidences)}%")
        print(f"   • Max: {max(confidences)}%")
        print(f"   • Media: {sum(confidences)/len(confidences):.1f}%")
        above_75 = [c for c in confidences if c >= 75]
        print(f"   • >= 75%: {len(above_75)}/{len(confidences)} ({len(above_75)/len(confidences)*100:.1f}%)")
        between_72_75 = [c for c in confidences if 72 <= c < 75]
        print(f"   • 72-75%: {len(between_72_75)}/{len(confidences)} ({len(between_72_75)/len(confidences)*100:.1f}%)")
    
    if evs:
        print(f"\n📊 ANALISI EV TROVATE ({len(evs)} valori):")
        positive_evs = [e for e in evs if e > 0]
        print(f"   • EV positivi: {len(positive_evs)}/{len(evs)} ({len(positive_evs)/len(evs)*100:.1f}%)")
        if positive_evs:
            print(f"   • Min EV positivo: {min(positive_evs):.1f}%")
            print(f"   • Max EV positivo: {max(positive_evs):.1f}%")
            print(f"   • Media EV positivo: {sum(positive_evs)/len(positive_evs):.1f}%")
        above_10 = [e for e in positive_evs if e >= 10]
        print(f"   • >= 10%: {len(above_10)}/{len(positive_evs)} ({len(above_10)/len(positive_evs)*100:.1f}% se > 0)")
        between_8_10 = [e for e in positive_evs if 8 <= e < 10]
        print(f"   • 8-10%: {len(between_8_10)}/{len(positive_evs)} ({len(between_8_10)/len(positive_evs)*100:.1f}% se > 0)")
    
    print("\n💡 RACCOMANDAZIONI:")
    if confidences and evs:
        above_75_count = len([c for c in confidences if c >= 75])
        between_72_75_count = len([c for c in confidences if 72 <= c < 75])
        above_10_ev = len([e for e in evs if e > 0 and e >= 10])
        between_8_10_ev = len([e for e in evs if e > 0 and 8 <= e < 10])
        
        if between_72_75_count > 0 or between_8_10_ev > 0:
            print("   ✅ Ci sono opportunità tra 72-75% conf o 8-10% EV")
            print("   💡 Considera di abbassare leggermente le soglie:")
            print("      • 74% conf, 9.5% EV (compromesso)")
            print("      • 73% conf, 9% EV (più segnali)")
        else:
            print("   ✅ Le soglie attuali sono appropriate")
            print("   💡 Mantieni 75% conf, 10% EV per qualità massima")
    else:
        print("   ⚠️  Dati insufficienti per raccomandazioni")
        print("   💡 Prova ad abbassare a 73% conf, 9% EV per vedere più segnali")

if __name__ == '__main__':
    verifica_partite_live()
    verifica_analisi_sistema()
    verifica_soglie()
    print("\n" + "=" * 70)
    print("✅ VERIFICA COMPLETA")
    print("=" * 70)

