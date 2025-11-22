#!/usr/bin/env python3
"""
Calcolo Consumo API Reale con Cicli 20 Minuti
==============================================
"""

print("=" * 80)
print("📊 CALCOLO CONSUMO API REALE - CICLI 20 MINUTI H24")
print("=" * 80)
print()

# Parametri
CICLO_MINUTI = 20
CACHE_TTL_MINUTI = 5
PARTITE_LIVE_MEDIA = 50

print(f"Parametri:")
print(f"  - Intervallo cicli: {CICLO_MINUTI} minuti")
print(f"  - Cache TTL: {CACHE_TTL_MINUTI} minuti")
print(f"  - Partite live medie: {PARTITE_LIVE_MEDIA}")
print()

# Calcolo cicli in 24 ore
cicli_al_giorno = (24 * 60) // CICLO_MINUTI
print(f"📅 Cicli in 24 ore: {cicli_al_giorno}")
print()

# Chiamate per ciclo
print("=" * 80)
print("🔍 CHIAMATE PER SINGOLO CICLO")
print("=" * 80)
print()

chiamate_fixtures = 1  # /fixtures?date=...
chiamate_live = 1      # /fixtures?live=all

print(f"1. Fixtures del giorno: {chiamate_fixtures} chiamata")
print(f"2. Partite live: {chiamate_live} chiamata")
print()

print(f"3. Per ogni partita live ({PARTITE_LIVE_MEDIA} partite):")
chiamate_stats_per_partita = 1    # /fixtures/statistics
chiamate_odds_per_partita = 1     # /odds/live

print(f"   - Statistiche: {chiamate_stats_per_partita} chiamata/partita")
print(f"   - Quote live: {chiamate_odds_per_partita} chiamata/partita")
print(f"   = {chiamate_stats_per_partita + chiamate_odds_per_partita} chiamate/partita")
print()

chiamate_per_partite = PARTITE_LIVE_MEDIA * (chiamate_stats_per_partita + chiamate_odds_per_partita)
print(f"   Totale per {PARTITE_LIVE_MEDIA} partite: {chiamate_per_partite} chiamate")
print()

chiamate_per_ciclo = chiamate_fixtures + chiamate_live + chiamate_per_partite

print(f"📊 TOTALE CHIAMATE PER CICLO: {chiamate_per_ciclo}")
print()

# Effetto cache
print("=" * 80)
print("🔍 EFFETTO CACHE TRA CICLI")
print("=" * 80)
print()

print(f"Cache TTL: {CACHE_TTL_MINUTI} minuti")
print(f"Intervallo tra cicli: {CICLO_MINUTI} minuti")
print()

if CICLO_MINUTI > CACHE_TTL_MINUTI:
    print(f"⚠️  CACHE SCADE TRA CICLI!")
    print(f"   {CICLO_MINUTI} minuti (intervallo) > {CACHE_TTL_MINUTI} minuti (TTL)")
    print()
    print("📉 Effetto cache tra cicli: NULLO")
    print("   Ogni ciclo richiede tutte le chiamate perché cache è scaduta")
    print()

    risparmio_tra_cicli = 0
    chiamate_con_cache_per_ciclo = chiamate_per_ciclo
else:
    print(f"✅ CACHE VALIDA TRA CICLI!")
    print(f"   {CICLO_MINUTI} minuti (intervallo) <= {CACHE_TTL_MINUTI} minuti (TTL)")
    print()
    print("📈 Effetto cache tra cicli: ALTO")
    print(f"   Solo {chiamate_fixtures + chiamate_live} chiamate per ciclo (fixtures + live)")
    print(f"   Statistiche e quote prese da cache")
    print()

    risparmio_tra_cicli = chiamate_per_partite
    chiamate_con_cache_per_ciclo = chiamate_fixtures + chiamate_live

# Consumo giornaliero
print("=" * 80)
print("📊 CONSUMO GIORNALIERO (24 ORE)")
print("=" * 80)
print()

consumo_giornaliero = cicli_al_giorno * chiamate_con_cache_per_ciclo

print(f"SENZA ottimizzazioni:")
print(f"  {cicli_al_giorno} cicli × {chiamate_per_ciclo} chiamate = {cicli_al_giorno * chiamate_per_ciclo} chiamate/giorno")
print()

if risparmio_tra_cicli > 0:
    print(f"CON cache (TTL {CACHE_TTL_MINUTI} min, cicli ogni {CICLO_MINUTI} min):")
    print(f"  {cicli_al_giorno} cicli × {chiamate_con_cache_per_ciclo} chiamate = {consumo_giornaliero} chiamate/giorno")
    print(f"  Risparmio: {cicli_al_giorno * chiamate_per_ciclo - consumo_giornaliero} chiamate/giorno")
    print()
else:
    print(f"CON cache (TTL {CACHE_TTL_MINUTI} min, cicli ogni {CICLO_MINUTI} min):")
    print(f"  {cicli_al_giorno} cicli × {chiamate_per_ciclo} chiamate = {consumo_giornaliero} chiamate/giorno")
    print(f"  ⚠️  NESSUN risparmio (cache scade tra cicli)")
    print()

# Limite API-SPORTS
print("=" * 80)
print("🎯 VERIFICA LIMITE API-SPORTS")
print("=" * 80)
print()

limite_api_sports = 7500
print(f"Limite giornaliero API-SPORTS: {limite_api_sports} chiamate")
print(f"Consumo previsto: {consumo_giornaliero} chiamate")
print()

utilizzo_pct = (consumo_giornaliero / limite_api_sports) * 100
print(f"Utilizzo: {utilizzo_pct:.1f}%")
print(f"Margine disponibile: {limite_api_sports - consumo_giornaliero} chiamate")
print()

if consumo_giornaliero < limite_api_sports:
    print(f"✅ ENTRO IL LIMITE (margine: {((limite_api_sports - consumo_giornaliero) / limite_api_sports * 100):.1f}%)")
elif consumo_giornaliero < limite_api_sports * 1.1:
    print(f"⚠️  VICINO AL LIMITE (solo {limite_api_sports - consumo_giornaliero} chiamate di margine)")
else:
    print(f"❌ OLTRE IL LIMITE di {consumo_giornaliero - limite_api_sports} chiamate!")
print()

# Soluzioni per ridurre consumo
if CICLO_MINUTI > CACHE_TTL_MINUTI:
    print("=" * 80)
    print("💡 SOLUZIONI PER RIDURRE CONSUMO")
    print("=" * 80)
    print()

    print("Opzione 1: Aumentare TTL cache")
    ttl_ottimale = CICLO_MINUTI + 5
    print(f"  - Portare TTL cache da {CACHE_TTL_MINUTI} a {ttl_ottimale} minuti")
    print(f"  - Risparmio stimato: {risparmio_tra_cicli * cicli_al_giorno} chiamate/giorno")
    consumo_con_ttl_alto = cicli_al_giorno * (chiamate_fixtures + chiamate_live)
    print(f"  - Nuovo consumo: ~{consumo_con_ttl_alto} chiamate/giorno")
    print(f"  - Utilizzo: {(consumo_con_ttl_alto / limite_api_sports * 100):.1f}%")
    print()

    print("Opzione 2: Ridurre intervallo cicli")
    ciclo_ottimale = CACHE_TTL_MINUTI - 1
    cicli_con_intervallo_ridotto = (24 * 60) // ciclo_ottimale
    print(f"  - Portare intervallo da {CICLO_MINUTI} a {ciclo_ottimale} minuti")
    print(f"  - Numero cicli: {cicli_con_intervallo_ridotto} al giorno")
    consumo_con_cicli_ridotti = chiamate_per_ciclo + (cicli_con_intervallo_ridotto - 1) * (chiamate_fixtures + chiamate_live)
    print(f"  - Consumo stimato: ~{consumo_con_cicli_ridotti} chiamate/giorno")
    print(f"  - Utilizzo: {(consumo_con_cicli_ridotti / limite_api_sports * 100):.1f}%")
    print()

    print("Opzione 3: Richiesta selettiva")
    print(f"  - Richiedere stats/quote solo per partite con segnali")
    partite_con_segnali = int(PARTITE_LIVE_MEDIA * 0.3)  # 30% delle partite
    print(f"  - Assumendo {partite_con_segnali} partite con segnali su {PARTITE_LIVE_MEDIA}")
    consumo_selettivo = cicli_al_giorno * (chiamate_fixtures + chiamate_live + partite_con_segnali * 2)
    print(f"  - Consumo stimato: ~{consumo_selettivo} chiamate/giorno")
    print(f"  - Risparmio: {consumo_giornaliero - consumo_selettivo} chiamate/giorno ({((consumo_giornaliero - consumo_selettivo) / consumo_giornaliero * 100):.1f}%)")
    print(f"  - Utilizzo: {(consumo_selettivo / limite_api_sports * 100):.1f}%")
    print()

print("=" * 80)
print("📝 CONCLUSIONE")
print("=" * 80)
print()

print(f"Con cicli ogni {CICLO_MINUTI} minuti e cache TTL {CACHE_TTL_MINUTI} minuti:")
print(f"  📊 Consumo: {consumo_giornaliero} chiamate/giorno")
print(f"  📈 Utilizzo: {utilizzo_pct:.1f}% del limite API-SPORTS")

if CICLO_MINUTI > CACHE_TTL_MINUTI:
    print(f"  ⚠️  Cache non efficace tra cicli (scade dopo {CACHE_TTL_MINUTI} min)")
    print()
    print("💡 Raccomandazione: Aumentare TTL cache o ridurre intervallo cicli")
else:
    print(f"  ✅ Cache efficace tra cicli")
    print(f"  💾 Risparmio: {risparmio_tra_cicli * cicli_al_giorno} chiamate/giorno")

print()
