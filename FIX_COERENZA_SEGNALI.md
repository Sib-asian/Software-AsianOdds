# 🔧 FIX COERENZA SEGNALI - ELIMINAZIONE CONTRADDIZIONI

## 🎯 PROBLEMA IDENTIFICATO

Segnali contrastanti sulla stessa partita:
- **Under 1.5** su 1-0 (partita chiusa)
- **Ribaltone segno 2** (partita aperta, può ribaltare)
- **Contraddizione logica**: Se Under 1.5, la partita è chiusa → no ribaltone!

## ✅ FIX IMPLEMENTATO

### 1. **Filtro Coerenza Segnali** (`_filter_contradictory_signals`)
- Raggruppa opportunità per `match_id`
- Verifica contraddizioni logiche tra segnali della stessa partita
- Mantiene solo segnali coerenti (priorità alla confidence più alta)

### 2. **Contraddizioni Rilevate**

#### CONTRADDIZIONE 1: Under + Ribaltone
- **Logica**: Se c'è Under, la partita è chiusa → no ribaltone
- **Esempio**: Under 1.5 + Ribaltone → **BLOCCATO**

#### CONTRADDIZIONE 2: Under + Over (stesso goal line)
- **Logica**: Under e Over con goal line simile sono contraddittori
- **Esempio**: Under 1.5 + Over 2.5 → **BLOCCATO** (se differenza <= 1.0)

#### CONTRADDIZIONE 3: Under HT + Over HT
- **Logica**: Under e Over primo tempo sono contraddittori
- **Esempio**: Under 0.5 HT + Over 1.5 HT → **BLOCCATO**

#### CONTRADDIZIONE 4: Clean Sheet + BTTS
- **Logica**: Clean Sheet e BTTS sono mutuamente esclusivi
- **Esempio**: Clean Sheet + BTTS → **BLOCCATO**

#### CONTRADDIZIONE 5: Under + Ribaltone (partita chiusa)
- **Logica**: Se Under 1.5 o meno e gol vicini al limite, partita chiusa → no ribaltone
- **Esempio**: Under 1.5 su 1-0 + Ribaltone → **BLOCCATO**

### 3. **Priorità**
- Segnali ordinati per confidence (migliore prima)
- Se contraddittori, mantiene quello con confidence più alta
- Rimuove quello con confidence più bassa

## 📊 RISULTATO

Ora i segnali sono **COERENTI**:
- ✅ Under 1.5 su 1-0 → **NON** può esserci Ribaltone
- ✅ Under 1.5 → **NON** può esserci Over 2.5
- ✅ Under HT → **NON** può esserci Over HT
- ✅ Clean Sheet → **NON** può esserci BTTS

**Il sistema ora invia solo segnali coerenti per partita!** 🚀



