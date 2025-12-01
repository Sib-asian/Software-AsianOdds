# ⚡ OTTIMIZZAZIONE CHIAMATE API - AGGIORNATA

## 📊 LIMITI API GRATUITI

### API-SPORTS (API_FOOTBALL_KEY)
- **Limite**: 100 chiamate/giorno
- **Endpoint usati**:
  - `fixtures?live=all` - 1 chiamata per tutte le partite live
  - `fixtures/statistics?fixture={id}` - 1 chiamata per partita (solo se statistiche non incluse)

### TheOddsAPI
- **Limite**: 500 chiamate/mese (~16/giorno)
- **Endpoint usati**:
  - `odds` - 1 chiamata per ciclo

### Football-Data.org
- **Limite**: 10 chiamate/minuto
- **Endpoint usati**: Raramente (opzionale)

## ⚠️ PROBLEMA IDENTIFICATO

**Prima dell'ottimizzazione:**
- `_get_live_match_data_from_api_sports` faceva 2 chiamate per partita:
  - 1 chiamata `fixtures?live=all` per ogni partita (duplicata!)
  - 1 chiamata `fixtures/statistics?fixture={id}` per ogni partita
- Se ci sono 10 partite live = 1 + (10 × 2) = **21 chiamate per ciclo**
- Con ciclo ogni 15 minuti = 96 cicli/giorno = **96 × 21 = 2,016 chiamate/giorno** ❌
- **SUPERATO DI 1,916 chiamate!**

## ✅ OTTIMIZZAZIONI IMPLEMENTATE

### 1. **Cache Fixtures Live=All**
- **Prima**: 1 chiamata `fixtures?live=all` per ogni partita
- **Dopo**: 1 chiamata `fixtures?live=all` per ciclo (cache 60 secondi)
- **Risparmio**: Da N chiamate a 1 chiamata per ciclo

### 2. **Cache Dati Live per Partita**
- **Prima**: Chiamata API per ogni richiesta dati live
- **Dopo**: Cache 60 secondi per partita
- **Risparmio**: Evita chiamate duplicate nello stesso ciclo

### 3. **Statistiche Incluse in Fixtures**
- **Prima**: Sempre chiamata separata per statistiche
- **Dopo**: Usa statistiche già incluse in `fixtures?live=all` quando disponibili
- **Risparmio**: Riduce chiamate `statistics` del 50-80%

### 4. **Intervallo Aumentato**
- **Prima**: 5 minuti (300s) = 288 cicli/giorno
- **Dopo**: 20 minuti (1200s) = 72 cicli/giorno
- **Motivo**: Rispettare limite 100 chiamate/giorno

## 📈 CALCOLO FINALE

**Dopo ottimizzazione:**
- Update interval: **20 minuti (1200s)**
- Cicli all'ora: **3**
- Cicli al giorno: **72**
- Chiamate per ciclo:
  - `fixtures?live=all`: **1** (cache 60s)
  - `fixtures/statistics`: **0-5** (solo se non incluse, cache 60s)
- **Totale chiamate/giorno: 72-144** ✅
- **Limite: 100 chiamate/giorno**
- **⚠️  ATTENZIONE**: Con molte partite live (>5) potrebbe superare il limite

## 🎯 OTTIMIZZAZIONI AGGIUNTIVE

### Cache Intelligente
- **Fixtures cache**: 60 secondi TTL
- **Live data cache**: 60 secondi TTL per partita
- **Risultato**: Evita chiamate duplicate nello stesso ciclo

### Fallback Sistema Alternativo
- Se API-SPORTS non disponibile, usa sistema alternativo (stime)
- **Risparmio**: 0 chiamate API quando non disponibile

## 💡 RACCOMANDAZIONI

1. **Monitora chiamate**: Controlla log per vedere quante chiamate vengono fatte
2. **Aumenta intervallo**: Se superi il limite, aumenta `update_interval` a 30 minuti (1800s)
3. **Limita partite live**: Se ci sono troppe partite live, considera filtri più restrittivi
4. **Usa cache**: Il sistema ora usa cache intelligente per ridurre chiamate

## 📝 CONFIGURAZIONE

Il sistema è configurato con:
- `update_interval = 1200` (20 minuti) - default
- Cache fixtures: 60 secondi TTL
- Cache live data: 60 secondi TTL
- Statistiche incluse quando disponibili

## ⚙️ COME MODIFICARE

Per cambiare l'intervallo:
```python
# Nel file automation_24h.py
update_interval=1200  # 20 minuti (default)
# Oppure:
update_interval=1800  # 30 minuti (più sicuro)
update_interval=900   # 15 minuti (più rischioso)
```

## 📊 STIMA CHIAMATE/GIORNO

| Partite Live | Chiamate/Ciclo | Cicli/Giorno | Totale/Giorno | Status |
|--------------|----------------|--------------|---------------|--------|
| 0-2          | 1              | 72           | 72            | ✅ OK  |
| 3-5          | 1-3            | 72           | 72-216        | ⚠️  Attenzione |
| 6-10         | 1-6            | 72           | 72-432        | ❌ Superato |

**Raccomandazione**: Con >5 partite live, aumenta `update_interval` a 30 minuti.








