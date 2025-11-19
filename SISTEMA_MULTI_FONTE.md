# 🌐 Sistema Multi-Fonte per Trovare Partite

## 📋 Panoramica

Sistema implementato per trovare partite da **multiple fonti** per massima copertura, incluse **leghe minori** e **partite di altre nazionalità**.

## 🎯 Fonti Integrate

### 1. **TheOddsAPI** ✅
- **Cosa fornisce**: Partite con quote da bookmaker
- **Copertura**: Partite principali con quote disponibili
- **Vantaggi**: Sempre ha quote, buona qualità dati
- **Limitazioni**: Non copre tutte le leghe minori

### 2. **API-SPORTS** ✅ (CONFIGURATA)
- **Cosa fornisce**: Oltre **2000 competizioni** incluse leghe minori
- **Copertura**: Massima - leghe minori, nazionali, internazionali
- **Vantaggi**: 
  - Copertura enorme (2000+ competizioni)
  - Include leghe minori e nazionali
  - Dati live aggiornati ogni 15 secondi
- **Limitazioni**: Non sempre ha quote (ma ha dati partita)

### 3. **Football-Data.org** ⚠️ (Opzionale)
- **Cosa fornisce**: Leghe europee principali
- **Copertura**: Leghe top europee
- **Vantaggi**: Dati ufficiali, alta qualità
- **Limitazioni**: Solo leghe principali, 10 chiamate/minuto

## 📊 Risultati Test

### Test Eseguito
- **Totale partite trovate**: **137 partite**
- **TheOddsAPI**: 8 partite
- **API-SPORTS**: 127 partite (leghe minori incluse!)
- **Football-Data.org**: 2 partite

### Leghe Disponibili
- **API-SPORTS**: 100+ competizioni (incluse leghe minori)
- **Football-Data.org**: 13 competizioni principali
- **TheOddsAPI**: Varie (basate su quote disponibili)

## 🔧 Come Funziona

### Flusso di Ricerca

1. **Sistema Multi-Fonte** cerca partite da tutte le fonti disponibili
2. **Deduplicazione** rimuove partite duplicate (stesse squadre, stessa data)
3. **Priorità**: Mantiene partite con più informazioni (quote, ecc.)
4. **Ordinamento** per data

### Configurazione

Il sistema è **automaticamente integrato** in `automation_24h.py`:

```python
# Cerca partite da tutte le fonti
matches = multi_source_finder.find_all_matches(
    days_ahead=1,              # Cerca partite per oggi
    include_minor_leagues=True, # Include leghe minori
    countries=None              # Tutti i paesi (None = tutti)
)
```

## 🎛️ Opzioni Disponibili

### Filtri per Paese
```python
# Solo partite italiane
matches = finder.find_all_matches(countries=["Italy"])

# Solo partite europee
matches = finder.find_all_matches(countries=["Italy", "Spain", "England", "Germany"])
```

### Escludere Leghe Minori
```python
# Solo leghe principali
matches = finder.find_all_matches(include_minor_leagues=False)
```

### Cercare Più Giorni
```python
# Cerca partite per i prossimi 3 giorni
matches = finder.find_all_matches(days_ahead=3)
```

## 📈 Vantaggi

### ✅ Più Partite
- **137 partite** vs **8 partite** (solo TheOddsAPI)
- **17x più partite** trovate!

### ✅ Leghe Minori
- API-SPORTS copre leghe minori e nazionali
- Partite di altre nazionalità incluse

### ✅ Ridondanza
- Se una fonte fallisce, altre continuano a funzionare
- Maggiore affidabilità

### ✅ Dati Completi
- Combina quote (TheOddsAPI) con dati partita (API-SPORTS)
- Informazioni più complete

## 🔍 Verifica Leghe Disponibili

Per vedere quali leghe sono coperte:

```python
from multi_source_match_finder import MultiSourceMatchFinder

finder = MultiSourceMatchFinder()
leagues = finder.get_leagues_available()

# Mostra leghe per fonte
for source, league_list in leagues.items():
    print(f"{source}: {len(league_list)} competizioni")
```

## ⚙️ Configurazione Richiesta

### Minimo Richiesto
- ✅ **TheOddsAPI** (già configurata)
- ✅ **API-SPORTS** (già configurata: `94d5ec5f491217af0874f8a2874dfbd8`)

### Opzionale (per più partite)
- ⚠️ **Football-Data.org** (opzionale, aggiungi `FOOTBALL_DATA_KEY` al `.env`)

## 🚀 Integrazione Automatica

Il sistema è **già integrato** in `automation_24h.py`:

1. **Automaticamente** usa sistema multi-fonte quando disponibile
2. **Fallback** a TheOddsAPI se multi-fonte non disponibile
3. **Logging** dettagliato di quale fonte usa

## 📝 Log Esempio

```
🔍 Usando sistema multi-fonte per trovare partite (TheOddsAPI + API-SPORTS + Football-Data.org)...
📡 Cercando partite da TheOddsAPI...
   ✅ Trovate 8 partite da TheOddsAPI
📡 Cercando partite da API-SPORTS...
   ✅ Trovate 127 partite da API-SPORTS
📡 Cercando partite da Football-Data.org...
   ✅ Trovate 2 partite da Football-Data.org
📊 Totale partite uniche trovate: 137
✅ Sistema multi-fonte ha trovato 137 partite
```

## ✅ Sistema Pronto!

Il sistema multi-fonte è **completamente implementato e funzionante**. 

Ora troverai:
- ✅ **Molto più partite** (137 vs 8)
- ✅ **Leghe minori** incluse
- ✅ **Partite di altre nazionalità**
- ✅ **Maggiore copertura** complessiva

**Riavvia il servizio** per applicare le modifiche!



