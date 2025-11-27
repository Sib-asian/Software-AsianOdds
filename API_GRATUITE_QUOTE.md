# 🆓 API Gratuite per Quote Reali

## 📊 Panoramica

Questo documento elenca le API **GRATUITE** disponibili per ottenere quote reali di scommesse sportive.

## ✅ API Gratuite Disponibili

### 1. **Oddspedia API** ⭐ RACCOMANDATO
- **URL**: https://widgets.oddspedia.com/
- **Chiave API**: ❓ Da verificare (potrebbe non essere richiesta)
- **Costo**: ✅ **GRATUITO**
- **Quote**: ✅ Quote in tempo reale
- **Bookmaker**: Multiple bookmaker
- **Copertura**: Varie competizioni sportive
- **Limiti**: Non specificati (da verificare)
- **Status**: ✅ Attivo (2024)

**Endpoint** (da verificare):
```
https://widgets.oddspedia.com/api/v1/...
```

**Vantaggi**:
- ✅ Gratuito
- ✅ Quote in tempo reale
- ✅ Multiple bookmaker
- ✅ Supporto clienti

**Svantaggi**:
- ⚠️ Documentazione da verificare
- ⚠️ Formato dati da verificare
- ⚠️ Limiti di rate da verificare

---

### 2. **TheOddsAPI Free Tier**
- **URL**: https://the-odds-api.com/
- **Chiave API**: ✅ Richiesta (gratuita)
- **Costo**: ✅ **GRATUITO** (500 chiamate/mese)
- **Quote**: ✅ Quote reali da multiple bookmaker
- **Bookmaker**: 10+ bookmaker
- **Copertura**: Varie competizioni
- **Limiti**: 500 chiamate/mese = ~20/giorno
- **Status**: ✅ Attivo (già implementato)

**Endpoint**:
```
https://api.the-odds-api.com/v4/sports/soccer/odds
```

**Vantaggi**:
- ✅ Già implementato nel sistema
- ✅ Quote reali garantite
- ✅ Best odds da multiple bookmaker
- ✅ Documentazione completa

**Svantaggi**:
- ⚠️ Budget limitato (500/mese)
- ⚠️ Non copre tutte le partite se usato per tutto

---

### 3. **API-SPORTS Odds Endpoint**
- **URL**: https://www.api-sports.io/
- **Chiave API**: ✅ Richiesta (gratuita con tier free)
- **Costo**: ✅ **GRATUITO** (tier free disponibile)
- **Quote**: ✅ Quote reali (endpoint `/odds`)
- **Bookmaker**: Multiple bookmaker
- **Copertura**: Oltre 2000 competizioni
- **Limiti**: Tier free limitato, tier Pro: 7500 chiamate/giorno
- **Status**: ✅ Attivo (già usato per partite)

**Endpoint**:
```
https://v3.football.api-sports.io/odds
```

**Vantaggi**:
- ✅ Già usato per trovare partite
- ✅ Budget generoso (se tier Pro)
- ✅ Copertura mondiale
- ✅ Quote reali

**Svantaggi**:
- ⚠️ Tier free limitato
- ⚠️ Richiede chiamata API aggiuntiva
- ⚠️ Quote potrebbero non essere sempre disponibili nel response base

---

### 4. **ODDSCORP**
- **URL**: https://oddscorp.com/
- **Chiave API**: ❓ Da verificare
- **Costo**: ❓ Non chiaro (potrebbe essere a pagamento)
- **Quote**: ✅ Quote da 49 bookmaker
- **Status**: ⚠️ Da verificare

**Nota**: Non è chiaro se sia gratuito. Richiede verifica diretta.

---

### 5. **OpticOdds**
- **URL**: https://opticodds.com/
- **Chiave API**: ✅ Richiesta
- **Costo**: ❌ **A PAGAMENTO**
- **Quote**: ✅ Quote in tempo reale da 100+ bookmaker
- **Status**: ❌ Non gratuito

---

## 📊 Confronto API Gratuite

| API | Gratuito? | Quote Reali? | Bookmaker | Limiti | Status | Implementato? |
|-----|-----------|--------------|-----------|--------|--------|---------------|
| **Oddspedia** | ✅ Sì | ✅ Sì | Multiple | ❓ Da verificare | ✅ Attivo | ❌ No |
| **TheOddsAPI** | ✅ Sì | ✅ Sì | 10+ | 500/mese | ✅ Attivo | ✅ Sì |
| **API-SPORTS** | ✅ Sì (tier free) | ✅ Sì | Multiple | Tier free limitato | ✅ Attivo | ⚠️ Parziale |
| **ODDSCORP** | ❓ ? | ✅ Sì | 49 | ❓ ? | ⚠️ Da verificare | ❌ No |
| **OpticOdds** | ❌ No | ✅ Sì | 100+ | - | ❌ A pagamento | ❌ No |

---

## 🎯 Raccomandazione

### Opzione 1: **Oddspedia API** (Nuova)
**Pro**:
- ✅ Gratuito
- ✅ Quote in tempo reale
- ✅ Multiple bookmaker
- ✅ Nessun limite noto (da verificare)

**Contro**:
- ⚠️ Non ancora implementato
- ⚠️ Documentazione da verificare
- ⚠️ Formato dati da testare

**Implementazione**: Richiede integrazione da zero

---

### Opzione 2: **Estrarre Quote da API-SPORTS** (Già usato)
**Pro**:
- ✅ Già usato per trovare partite
- ✅ Budget generoso (7500/giorno se tier Pro)
- ✅ Copertura mondiale
- ✅ Quote disponibili (endpoint `/odds`)

**Contro**:
- ⚠️ Richiede chiamata API aggiuntiva per quote
- ⚠️ Quote potrebbero non essere sempre nel response base

**Implementazione**: Relativamente semplice (già integrato)

---

### Opzione 3: **TheOddsAPI** (Già implementato)
**Pro**:
- ✅ Già implementato
- ✅ Quote reali garantite
- ✅ Best odds automatiche

**Contro**:
- ⚠️ Budget limitato (500/mese)
- ⚠️ Non copre tutte le partite

**Implementazione**: Già fatto, ma limitato

---

## 💡 Strategia Consigliata

### Approccio Ibrido Ottimale:

1. **Prima**: Estrarre quote da API-SPORTS response (se disponibili)
   - Costo: 0 chiamate aggiuntive
   - Copertura: 100% delle partite trovate

2. **Seconda**: Usare endpoint `/odds` di API-SPORTS per partite senza quote
   - Costo: Chiamate aggiuntive (ma budget generoso)
   - Copertura: 100% delle partite

3. **Terza**: Integrare Oddspedia API come supplemento
   - Costo: Gratuito
   - Copertura: Aggiuntiva per partite non coperte

4. **Ultima**: TheOddsAPI come fallback o per partite selezionate
   - Costo: Budget limitato (500/mese)
   - Copertura: Limitata ma quote garantite

---

## 🔧 Prossimi Passi

### Per Implementare Oddspedia:

1. **Verificare documentazione**:
   - Endpoint disponibili
   - Formato dati
   - Autenticazione richiesta
   - Limiti di rate

2. **Test API**:
   - Chiamata di prova
   - Verifica formato response
   - Verifica qualità quote

3. **Implementazione**:
   - Aggiungere in `multi_source_match_finder.py`
   - Integrare con sistema esistente
   - Gestire errori e fallback

### Per Estrarre Quote da API-SPORTS:

1. **Verificare response**:
   - Controllare se `odds_data` contiene quote reali
   - Verificare formato dati

2. **Implementare estrazione**:
   - Modificare `_fetch_from_api_sports()` in `multi_source_match_finder.py`
   - Estrarre quote da `odds_data`
   - Usare endpoint `/odds` se non disponibili

---

## 📝 Note

- **TheOddsAPI**: Già implementato ma budget limitato
- **API-SPORTS**: Già usato per partite, quote da estrarre
- **Oddspedia**: Nuova opzione gratuita da verificare e implementare
- **Priorità**: Massimizzare copertura con budget minimo

---

## ✅ Conclusione

**La migliore opzione attuale è**:
1. ✅ **Estrarre quote da API-SPORTS** (già usato, budget generoso)
2. ✅ **Integrare Oddspedia** come supplemento gratuito
3. ✅ **Usare TheOddsAPI** solo per partite selezionate o come fallback

Questo approccio massimizza:
- ✅ Copertura (100% partite)
- ✅ Quote reali (da multiple fonti)
- ✅ Budget (principalmente API-SPORTS già pagato)
- ✅ Affidabilità (multiple fonti = fallback)

---

**Ultimo aggiornamento**: 2024








