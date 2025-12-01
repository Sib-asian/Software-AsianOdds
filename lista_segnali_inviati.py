#!/usr/bin/env python3
"""
Script per visualizzare i segnali effettivamente inviati su Telegram
con le relative confidence
"""

import os
import sys
import re
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

# Aggiungi path per import
sys.path.insert(0, str(Path(__file__).parent))

def extract_sent_signals():
    """Estrae segnali inviati dai log"""
    log_dir = Path(__file__).parent / "logs"
    if not log_dir.exists():
        return []
    
    # Trova tutti i log
    log_files = sorted(log_dir.glob("automation_service_*.log"), key=lambda x: x.stat().st_mtime, reverse=True)
    if not log_files:
        return []
    
    signals = []
    
    # Pattern per trovare segnali inviati (pattern specifici del sistema)
    patterns = [
        r'notified opportunity',
        r'✅.*notified',
        r'send_betting_opportunity',
        r'opportunity.*ready.*send',
        r'segnal.*inviato',
        r'signal.*sent',
    ]
    
    # Leggi tutti i log (ultimi 3 file)
    for log_file in log_files[:3]:
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                
                for i, line in enumerate(lines):
                    # Cerca invii di segnali - pattern principale: "Notified opportunity"
                    if 'notified opportunity' in line.lower() or '✅' in line and 'notified' in line.lower():
                        # Cerca informazioni sul segnale nelle righe vicine (più contesto)
                        context_lines = lines[max(0, i-30):min(len(lines), i+5)]
                        signal_info = extract_signal_info(context_lines, i)
                        if signal_info:
                            signals.append(signal_info)
                    
                    # Cerca anche pattern alternativi
                    if any(p in line.lower() for p in patterns):
                        context_lines = lines[max(0, i-30):min(len(lines), i+5)]
                        signal_info = extract_signal_info(context_lines, i)
                        if signal_info:
                            signals.append(signal_info)
                    
                    # Cerca pattern "Trovate X opportunità" che indica segnali trovati
                    if 'trovate' in line.lower() and 'opportunit' in line.lower():
                        context_lines = lines[max(0, i-20):min(len(lines), i+20)]
                        signal_info = extract_signal_info(context_lines, i)
                        if signal_info:
                            signals.append(signal_info)
                            
        except Exception as e:
            print(f"⚠️  Errore lettura {log_file.name}: {e}")
    
    return signals

def extract_signal_info(lines, center_index):
    """Estrae informazioni sul segnale dalle righe di contesto"""
    signal = {
        'timestamp': None,
        'match': None,
        'market': None,
        'confidence': None,
        'odds': None,
        'recommendation': None,
        'score': None,
        'minute': None
    }
    
    # Cerca timestamp
    for line in lines:
        time_match = re.search(r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})', line)
        if time_match:
            signal['timestamp'] = time_match.group(1)
            break
    
    # Cerca match
    for line in lines:
        match_match = re.search(r'(.+?)\s+vs\s+(.+?)(?:\s|$|\(|\[|-)', line, re.IGNORECASE)
        if match_match:
            home = match_match.group(1).strip()
            away = match_match.group(2).strip()
            # Pulisci
            home = re.sub(r'^\d{4}-\d{2}-\d{2}.*?INFO\s*-\s*', '', home).strip()
            home = re.sub(r'🔍\s*ANALYZING:\s*', '', home, flags=re.IGNORECASE).strip()
            away = re.sub(r'\s*-\s*.*$', '', away).strip()
            if len(home) > 2 and len(away) > 2:
                signal['match'] = f"{home} vs {away}"
                break
    
    # Cerca confidence - pattern più robusti
    for line in lines:
        # Pattern: "confidence: 75.5%" o "confidence 75.5"
        conf_match = re.search(r'confidence[:\s]+(\d+\.?\d*)%?', line, re.IGNORECASE)
        if conf_match:
            try:
                signal['confidence'] = float(conf_match.group(1))
            except:
                pass
        
        # Pattern alternativo: "75.5% confidence"
        if not signal['confidence']:
            conf_match2 = re.search(r'(\d+\.?\d*)%?\s*confidence', line, re.IGNORECASE)
            if conf_match2:
                try:
                    signal['confidence'] = float(conf_match2.group(1))
                except:
                    pass
        
        # Pattern: "confidence: 75" o "conf: 75"
        if not signal['confidence']:
            conf_match3 = re.search(r'conf[:\s]+(\d+\.?\d*)', line, re.IGNORECASE)
            if conf_match3:
                try:
                    signal['confidence'] = float(conf_match3.group(1))
                except:
                    pass
        
        # Pattern nei log: "confidence: 75.0%" o "(confidence: 75%)"
        if not signal['confidence']:
            conf_match4 = re.search(r'\(?confidence[:\s]+(\d+\.?\d*)%?\)?', line, re.IGNORECASE)
            if conf_match4:
                try:
                    signal['confidence'] = float(conf_match4.group(1))
                except:
                    pass
    
    # Cerca market
    markets = ['over_0.5', 'over_1.5', 'over_2.5', 'over_3.5', 
               'under_0.5', 'under_1.5', 'under_2.5', 'under_3.5',
               'home_win', 'away_win', 'match_winner', '1x', 'x2',
               'btts', 'clean_sheet', 'exact_score', 'dnb']
    for line in lines:
        for market in markets:
            if market in line.lower():
                signal['market'] = market
                break
        if signal['market']:
            break
    
    # Cerca odds
    for line in lines:
        odds_match = re.search(r'odds[:\s]+(\d+\.?\d*)', line, re.IGNORECASE)
        if odds_match:
            try:
                signal['odds'] = float(odds_match.group(1))
            except:
                pass
    
    # Cerca score
    for line in lines:
        score_match = re.search(r'score[:\s]+(\d+)-(\d+)', line, re.IGNORECASE)
        if score_match:
            signal['score'] = f"{score_match.group(1)}-{score_match.group(2)}"
            break
    
    # Cerca minute
    for line in lines:
        minute_match = re.search(r"(\d+)'", line)
        if minute_match:
            try:
                signal['minute'] = int(minute_match.group(1))
            except:
                pass
    
    # Cerca recommendation
    for line in lines:
        if 'recommendation' in line.lower() or 'punta' in line.lower() or 'sugger' in line.lower():
            rec_match = re.search(r'(?:recommendation|punta|sugger)[:\s]+(.+?)(?:\n|$)', line, re.IGNORECASE)
            if rec_match:
                signal['recommendation'] = rec_match.group(1).strip()[:100]
                break
    
    # Ritorna solo se ha almeno match o market
    if signal['match'] or signal['market']:
        return signal
    return None

def extract_from_telegram_logs():
    """Cerca anche nei log di Telegram se esistono"""
    signals = []
    
    # Cerca pattern nei log che indicano invio
    log_dir = Path(__file__).parent / "logs"
    if not log_dir.exists():
        return signals
    
    log_files = sorted(log_dir.glob("*.log"), key=lambda x: x.stat().st_mtime, reverse=True)
    
    for log_file in log_files[:5]:  # Ultimi 5 log
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
                # Cerca pattern di opportunità trovate
                opp_pattern = r'opportunit[àa].*?trovat[ae]|found.*?opportunit'
                if re.search(opp_pattern, content, re.IGNORECASE):
                    # Estrai informazioni
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if 'opportunit' in line.lower() and ('trovat' in line.lower() or 'found' in line.lower()):
                            # Cerca nelle righe successive
                            context = lines[i:i+20]
                            signal = extract_signal_info(context, i)
                            if signal:
                                signals.append(signal)
        except:
            pass
    
    return signals

def main():
    print("=" * 80)
    print("📊 LISTA SEGNALI EFFETTIVAMENTE INVIATI SU TELEGRAM")
    print("=" * 80)
    print()
    
    # Estrai segnali
    print("🔍 Analizzando log per trovare segnali inviati...")
    signals = extract_sent_signals()
    
    # Se non trovati, cerca in modo alternativo
    if not signals:
        print("   Cercando in modo alternativo...")
        signals = extract_from_telegram_logs()
    
    if signals:
        # Raggruppa per match
        by_match = defaultdict(list)
        for sig in signals:
            key = sig.get('match', 'Unknown')
            by_match[key].append(sig)
        
        print(f"\n✅ Trovati {len(signals)} segnali inviati:\n")
        print("=" * 80)
        
        for match, match_signals in by_match.items():
            print(f"\n🏆 {match}")
            print("-" * 80)
            
            for i, sig in enumerate(match_signals, 1):
                print(f"\n  Segnale #{i}:")
                
                if sig.get('timestamp'):
                    print(f"    📅 Data/Ora: {sig['timestamp']}")
                
                if sig.get('score') and sig.get('minute'):
                    print(f"    ⚽ Score: {sig['score']} al {sig['minute']}'")
                elif sig.get('score'):
                    print(f"    ⚽ Score: {sig['score']}")
                elif sig.get('minute'):
                    print(f"    ⏱️  Minuto: {sig['minute']}'")
                
                if sig.get('market'):
                    print(f"    📊 Mercato: {sig['market']}")
                
                if sig.get('confidence'):
                    conf = sig['confidence']
                    status = "✅" if conf >= 72 else "⚠️" if conf >= 60 else "❌"
                    print(f"    {status} Confidence: {conf:.1f}%")
                
                if sig.get('odds'):
                    print(f"    💰 Odds: {sig['odds']:.2f}")
                
                if sig.get('recommendation'):
                    print(f"    💡 Consiglio: {sig['recommendation']}")
        
        # Statistiche
        print("\n" + "=" * 80)
        print("📈 STATISTICHE")
        print("=" * 80)
        
        confidences = [s.get('confidence') for s in signals if s.get('confidence')]
        if confidences:
            avg_conf = sum(confidences) / len(confidences)
            min_conf = min(confidences)
            max_conf = max(confidences)
            print(f"\n  Confidence media: {avg_conf:.1f}%")
            print(f"  Confidence minima: {min_conf:.1f}%")
            print(f"  Confidence massima: {max_conf:.1f}%")
            
            # Conta per range
            high_conf = sum(1 for c in confidences if c >= 80)
            med_conf = sum(1 for c in confidences if 70 <= c < 80)
            low_conf = sum(1 for c in confidences if c < 70)
            
            print(f"\n  Distribuzione:")
            print(f"    Alta (≥80%): {high_conf}")
            print(f"    Media (70-79%): {med_conf}")
            print(f"    Bassa (<70%): {low_conf}")
        
        markets_count = defaultdict(int)
        for sig in signals:
            if sig.get('market'):
                markets_count[sig['market']] += 1
        
        if markets_count:
            print(f"\n  Mercati più inviati:")
            for market, count in sorted(markets_count.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"    {market}: {count}")
        
    else:
        print("\n⚠️  Nessun segnale inviato trovato nei log recenti")
        print("\nPossibili motivi:")
        print("  • Il sistema è appena partito")
        print("  • Non ci sono state opportunità valide")
        print("  • I log non contengono informazioni sui segnali inviati")
        print("\n💡 Suggerimento: Controlla i log in tempo reale con monitor_logs.bat")
    
    print("\n" + "=" * 80)
    print("💡 NOTA: Questa lista mostra i segnali trovati nei log.")
    print("   Per vedere i segnali in tempo reale, controlla Telegram o i log live.")
    print("=" * 80)

if __name__ == '__main__':
    main()

