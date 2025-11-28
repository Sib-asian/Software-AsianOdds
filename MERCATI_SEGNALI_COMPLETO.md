# 📊 LISTA COMPLETA MERCATI/SEGNALI

## 🎯 PANORAMICA

Il sistema invia segnali per **PRE-MATCH** e **LIVE** con diversi mercati. Ogni segnale viene inviato solo se:
- **EV ≥ 8%** (configurabile)
- **Confidence ≥ 70%** (configurabile, alcuni mercati hanno soglie più alte)
- **Nessun duplicato** (stesso match_id + stesso market)

---

## ⚽ PRE-MATCH (Partite non ancora iniziate)

### 1. **1X2 (Esito Finale)**
- `1X2_HOME` - Vittoria squadra casa
- `1X2_DRAW` - Pareggio
- `1X2_AWAY` - Vittoria squadra trasferta

**Sistema:** AI Pipeline (analisi pre-partita con modelli ML)

---

## 🔴 LIVE (Partite in corso)

### **Sistema 1: Live Betting Advisor** (sistema principale live)

#### **A. RISULTATO FINALE (1X2)**
- `1x2_home` - Vittoria squadra casa (ribaltone/comeback)
- `1x2_away` - Vittoria squadra trasferta (ribaltone/comeback)
- `1x` - Squadra casa pareggio o vittoria (Double Chance)
- `x2` - Squadra trasferta pareggio o vittoria (Double Chance)

#### **B. OVER/UNDER GOL TOTALI**
- `over_0.5` - Over 0.5 gol
- `over_1.5` - Over 1.5 gol
- `over_2.5` - Over 2.5 gol
- `over_3.5` - Over 3.5 gol
- `under_0.5` - Under 0.5 gol
- `under_1.5` - Under 1.5 gol
- `under_2.5` - Under 2.5 gol
- `under_3.5` - Under 3.5 gol

#### **C. OVER/UNDER PRIMO TEMPO (HT)**
- `over_0.5_ht` - Over 0.5 gol primo tempo
- `over_1.5_ht` - Over 1.5 gol primo tempo
- `under_0.5_ht` - Under 0.5 gol primo tempo
- `under_1.5_ht` - Under 1.5 gol primo tempo

#### **D. OVER/UNDER SECONDO TEMPO (2H)**
- `over_0.5_second_half` - Over 0.5 gol secondo tempo

#### **E. PROSSIMO GOL**
- `next_goal_home` - Prossimo gol segnato da squadra casa
- `next_goal_away` - Prossimo gol segnato da squadra trasferta

#### **F. DRAW NO BET (DNB)**
- `dnb_home` - Draw No Bet squadra casa
- `dnb_away` - Draw No Bet squadra trasferta

#### **G. HANDICAP**
- `away_handicap_+1.5` - Handicap +1.5 squadra trasferta
- `home_handicap_+1.5` - Handicap +1.5 squadra casa

#### **H. BTTS (Both Teams To Score)**
- `btts_yes` - Entrambe le squadre segnano
- `btts_no` - Almeno una squadra non segna

#### **I. WIN TO NIL**
- `home_win_to_nil` - Squadra casa vince senza subire gol
- `away_win_to_nil` - Squadra trasferta vince senza subire gol

#### **J. CORNER**
- `over_8.5_corners` - Over 8.5 corner totali

#### **K. CARTELLINI**
- `over_5.5_cards` - Over 5.5 cartellini totali

#### **L. TOTAL GOALS PARI/DISPARI**
- `total_goals_odd` - Totale gol dispari
- `total_goals_even` - Totale gol pari

---

### **Sistema 2: Live Match AI** (nuova IA dedicata ai match live)

#### **A. RISULTATO FINALE (1X2)**
- `1X2_HOME` - Vittoria squadra casa
- `1X2_DRAW` - Pareggio
- `1X2_AWAY` - Vittoria squadra trasferta

#### **B. OVER/UNDER GOL TOTALI**
- `OVER_2_5` - Over 2.5 gol
- `UNDER_2_5` - Under 2.5 gol

**Nota:** La Live Match AI analizza in tempo reale e calcola probabilità aggiornate basate su:
- Score corrente
- Minuto di gioco
- Statistiche (possesso, tiri, ecc.)
- Momentum della partita
- Pattern rilevati

---

## 📋 RIEPILOGO PER CATEGORIA

### **Totale Mercati Disponibili: ~40+**

#### **Risultato Partita:**
- 1X2 (3 varianti)
- Double Chance (2 varianti: 1X, X2)
- Draw No Bet (2 varianti)

#### **Gol Totali:**
- Over/Under 0.5, 1.5, 2.5, 3.5 (8 varianti)
- Over/Under HT 0.5, 1.5 (4 varianti)
- Over 0.5 Secondo Tempo (1 variante)

#### **Gol Specifici:**
- Prossimo Gol (2 varianti)
- Total Goals Pari/Dispari (2 varianti)

#### **Squadre:**
- BTTS (2 varianti)
- Win To Nil (2 varianti)
- Handicap (2+ varianti)

#### **Statistiche:**
- Corner (1+ varianti)
- Cartellini (1+ varianti)

---

## ⚙️ SOGLIE CONFIDENCE PER MERCATO

Alcuni mercati hanno soglie di confidence più alte per maggiore sicurezza:

| Mercato | Confidence Minima |
|---------|-------------------|
| Over 0.5 | 72% |
| Over 0.5 HT | 75% |
| Over 1.5 | 75% |
| Over 2.5 | 78% |
| Over 3.5 | 80% |
| Under 0.5 | 75% |
| Under 0.5 HT | 78% |
| Under 1.5 | 78% |
| Under 2.5 | 80% |
| Under 3.5 | 82% |
| 1X2 (ribaltone) | 75% |
| Double Chance | 75% |
| DNB | 78% |
| BTTS | 75% |
| Win To Nil | 80% |
| Prossimo Gol | 78% |
| Corner | 75% |
| Cartellini | 75% |

---

## 🔄 COME FUNZIONA LA DEDUPLICAZIONE

Il sistema traccia ogni segnale con chiave: `match_id|market`

**Esempio:**
- ✅ `match_123|1X2_HOME` - OK (primo segnale)
- ❌ `match_123|1X2_HOME` - BLOCCATO (duplicato)
- ✅ `match_123|OVER_2_5` - OK (mercato diverso, stessa partita)

**Risultato:** Sulla stessa partita puoi ricevere segnali per mercati diversi, ma non duplicati dello stesso mercato.

---

## 📱 FORMATO NOTIFICHE

Ogni segnale include:
- 📅 Match (squadre, lega, data/ora)
- 💰 Recommendation (mercato, odds, stake)
- 📊 Analysis (EV, probabilità, confidence)
- 🤖 AI Ensemble (predizioni modelli)
- 🧠 AI Playbook (ragionamento AI)

---

## 🎯 PRIORITÀ SEGNALI

1. **Live Match AI** (nuova) - Priorità alta per match live
2. **Live Betting Advisor** - Sistema principale live
3. **AI Pipeline** - Pre-match

I segnali vengono combinati evitando duplicati (stesso match_id + stesso market).

---

## 📊 STATISTICHE

- **Pre-match:** ~3 mercati principali (1X2)
- **Live:** ~35+ mercati diversi
- **Totale:** ~40+ mercati disponibili

---

**Ultimo aggiornamento:** Dicembre 2024
**Versione sistema:** Con Live Match AI integrata








