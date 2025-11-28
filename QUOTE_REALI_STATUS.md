# Status Quote Reali vs Quote Stimate

## 📊 Panoramica

Il sistema utilizza **quote reali** da API-SPORTS (preferenza Bet365) per i mercati principali e **quote stimate** per mercati secondari non supportati dall'API.

---

## ✅ MERCATI CON QUOTE REALI (Da API-SPORTS)

### 1X2 e Varianti
- ✅ **1X2** (Vittoria Casa/Pareggio/Ospite)
- ✅ **Double Chance** (1X, X2, 12)
- ✅ **Draw No Bet** (DNB Home/Away)

### Over/Under Gol
- ✅ **Over 0.5** Full Time
- ✅ **Over 1.5** Full Time
- ✅ **Over 2.5** Full Time
- ✅ **Over 3.5** Full Time
- ✅ **Under 0.5** Full Time
- ✅ **Under 1.5** Full Time
- ✅ **Under 2.5** Full Time
- ✅ **Under 3.5** Full Time

### BTTS (Both Teams To Score)
- ✅ **BTTS Yes**
- ✅ **BTTS No**

### Tempo Parziale
- ✅ **Half Time Result** (1X2 Primo Tempo)
- ✅ **Second Half Result** (1X2 Secondo Tempo)

### Altri Mercati
- ✅ **Odd/Even Goals**

---

## ⚠️ MERCATI CON QUOTE STIMATE (Non Supportati da API)

### Over/Under Primo/Secondo Tempo
- ⚠️ **Over 0.5 HT** (Primo Tempo)
- ⚠️ **Over 1.5 HT** (Primo Tempo)
- ⚠️ **Over 0.5 ST** (Secondo Tempo)
- ⚠️ **Under 0.5 HT**
- ⚠️ **Under 1.5 HT**

### BTTS Primo Tempo
- ⚠️ **BTTS Yes First Half**

### Mercati Speciali
- ⚠️ **Over 8.5 Corners**
- ⚠️ **Over 5.5 Cards**
- ⚠️ **Next Goal** (Home/Away)
- ⚠️ **Clean Sheet** (Home/Away)
- ⚠️ **Win To Nil**
- ⚠️ **First/Last Goal**
- ⚠️ **Goal Range** (0-1, 2-3, 4+)
- ⚠️ **Highest Scoring Half**
- ⚠️ **Win Either Half**
- ⚠️ **Team To Score** (First/Last/Next)
- ⚠️ **Home/Away Handicap**

---

## 🎯 BOOKMAKER UTILIZZATI (In Ordine di Preferenza)

1. **Bet365** (ID: 8) ⭐ **PREFERITO**
2. **Pinnacle** (ID: 12) - Sharp bookmaker
3. **William Hill** (ID: 3)
4. **1xBet** (ID: 5)
5. **Betfair** (ID: 11) - Exchange

---

## 🔧 COMPORTAMENTO DEL SISTEMA

### Mercati con Quote Reali
```
Se quota REALE disponibile:
  → Usa quota reale da Bet365/Pinnacle
  → EV calculation CORRETTO al 100%
  → Genera segnale

Se quota REALE NON disponibile:
  → SKIP segnale
  → Log warning
  → NON genera segnale (protezione utente)
```

### Mercati con Quote Stimate
```
Usa quote STIMATE (hardcoded):
  → EV calculation APPROSSIMATIVO
  → Quote potrebbero non riflettere mercato reale
  → ⚠️ VERIFICA MANUALMENTE quota prima di puntare
```

---

## 📈 STATISTICHE QUOTE

| Tipo Mercato | Quote Reali | Quote Stimate | % Reali |
|--------------|-------------|---------------|---------|
| Over/Under FT | 8/8 | 0/8 | 100% |
| BTTS FT | 2/2 | 0/2 | 100% |
| 1X2 Varianti | 9/9 | 0/9 | 100% |
| HT/ST Specifici | 3/15 | 12/15 | 20% |
| Speciali | 0/30 | 30/30 | 0% |
| **TOTALE** | **22/64** | **42/64** | **34%** |

---

## ⚡ PRIORITÀ COPERTURA

**Alta Priorità** (Quote Reali Implementate):
- ✅ Over/Under 0.5, 1.5, 2.5, 3.5 FT
- ✅ BTTS Yes/No FT
- ✅ 1X2 e varianti

**Media Priorità** (Da Implementare):
- ⏳ Over/Under HT/ST specifici (se API supporta)
- ⏳ BTTS HT (se API supporta)

**Bassa Priorità** (Non Supportato da API):
- ❌ Corner markets
- ❌ Cards markets
- ❌ Next Goal markets
- ❌ Clean Sheet specifici
- ❌ Altri mercati speciali

---

## 🚨 IMPORTANTE

**SEMPRE VERIFICA la quota sul bookmaker prima di puntare**, specialmente per mercati con quote stimate (⚠️).

Il sistema mostra se la quota è reale o stimata nei log:
```
✅ Quote recuperate per fixture 12345: ['odds_1', 'odds_x', 'odds_2', 'odds_over_2_5', ...]
⚠️  Over 0.5 HT saltato: quota reale non disponibile
```

---

## 📝 NOTE TECNICHE

### Perché alcuni mercati usano quote stimate?

API-SPORTS fornisce quote per mercati principali supportati dai bookmaker partner. Mercati molto specifici (es. "Over 0.5 Solo Secondo Tempo", "Corner esatti", ecc.) non sono sempre disponibili perché:

1. **Non tutti i bookmaker li offrono**
2. **Quote cambiano troppo rapidamente** per essere tracciabili
3. **Mercati troppo specifici** non hanno liquidità sufficiente

### Cosa fare?

Per mercati con quote stimate:
1. Usa il segnale come **indicazione generale**
2. **Verifica manualmente** la quota sul bookmaker
3. Se EV sembra troppo alto (>20%), **sospetta** di quota stimata errata
4. Considera il segnale come "opportunità da verificare" non come "scommessa garantita"

---

## 🔄 AGGIORNAMENTI FUTURI

- [ ] Testare se API-SPORTS supporta Over/Under HT/ST
- [ ] Integrare bookmaker aggiuntivi per mercati speciali
- [ ] Documentare quote stimate con maggiore trasparenza nelle notifiche
- [ ] Aggiungere flag nelle notifiche Telegram: `[REAL ODDS]` vs `[ESTIMATED]`

---

**Ultima modifica**: 2025-11-22
**Versione**: 2.0
