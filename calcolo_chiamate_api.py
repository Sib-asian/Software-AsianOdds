#!/usr/bin/env python3
"""
Calcolo chiamate API-Football
"""

update_interval = 900  # 15 minuti (ottimizzato per 100 chiamate/giorno)
cycles_per_hour = 3600 / update_interval
cycles_per_day = cycles_per_hour * 24

print("="*60)
print("📊 CALCOLO CHIAMATE API-FOOTBALL")
print("="*60)

print(f"\n⚙️  CONFIGURAZIONE ATTUALE:")
print(f"   Update interval: {update_interval}s ({update_interval/60:.0f} minuti)")
print(f"   Cicli all'ora: {cycles_per_hour:.1f}")
print(f"   Cicli al giorno: {cycles_per_day:.0f}")

print(f"\n⚠️  PROBLEMA PRIMA DELL'OTTIMIZZAZIONE:")
print(f"   Se ci sono 5 partite live, fa 5 chiamate per ciclo")
print(f"   Totale: {cycles_per_day:.0f} * 5 = {cycles_per_day * 5:.0f} chiamate/giorno")
print(f"   Limite: 100 chiamate/giorno")
print(f"   ❌ SUPERATO DI {cycles_per_day * 5 - 100:.0f} chiamate!")

print(f"\n✅ DOPO OTTIMIZZAZIONE:")
calls_per_cycle = 1  # UNA sola chiamata per tutte le partite
total_calls = cycles_per_day * calls_per_cycle
print(f"   Chiamate per ciclo: {calls_per_cycle} (UNA sola per tutte le partite)")
print(f"   Totale chiamate/giorno: {total_calls:.0f}")
print(f"   Limite: 100 chiamate/giorno")
if total_calls <= 100:
    print(f"   ✅ OK! Rientra nel limite")
    print(f"   Margine: {100 - total_calls:.0f} chiamate disponibili")
else:
    print(f"   ❌ SUPERATO DI {total_calls - 100:.0f} chiamate!")

print(f"\n💡 SUGGERIMENTO:")
optimal_interval = (24 * 3600) / 100
print(f"   Per stare esattamente sotto 100 chiamate/giorno:")
print(f"   Intervallo minimo: {optimal_interval:.0f}s ({optimal_interval/60:.1f} minuti)")
print(f"   Intervallo consigliato: {int(optimal_interval) + 100}s ({int(optimal_interval + 100)/60:.1f} minuti) per margine sicurezza")

print(f"\n📈 CHIAMATE ALL'ORA:")
calls_per_hour = cycles_per_hour * calls_per_cycle
print(f"   {calls_per_hour:.1f} chiamate/ora")
print(f"   {calls_per_hour * 24:.0f} chiamate/giorno")

print("\n" + "="*60)

