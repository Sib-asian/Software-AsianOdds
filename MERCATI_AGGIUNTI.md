# ✅ Mercati Aggiunti e Migliorati - Con IA

## 🎯 Mercati Primo Tempo (HT) - MIGLIORATI

### ✅ Over 0.5 HT
- **Quando**: Nessun gol, partita aperta (15'-40')
- **Analisi IA**: Tiri/minuto, tiri in porta
- **Confidence**: 60-85% (base + IA boost)
- **Criteri**: 
  - Tiri/minuto > 0.3
  - Tiri in porta/minuto > 0.1
  - Confidence aumenta con minuto e tiri

### ✅ Over 1.5 HT
- **Quando**: Già 1 gol, partita ancora aperta (20'-40')
- **Analisi IA**: Tiri totali, tiri in porta
- **Confidence**: 65-88% (base + IA boost)
- **Criteri**: 
  - Tiri totali >= 8
  - Tiri in porta >= 3

### 🆕 Under 0.5 HT
- **Quando**: Nessun gol, partita chiusa (30'-44')
- **Analisi IA**: Tiri/minuto bassi, pochi tiri in porta
- **Confidence**: 70-90% (base + IA boost)
- **Criteri**: 
  - Tiri/minuto < 0.2
  - Tiri in porta/minuto < 0.05

### 🆕 Under 1.5 HT
- **Quando**: Max 1 gol, partita chiusa (35'-44')
- **Analisi IA**: Tiri/minuto bassi
- **Confidence**: 75-92% (base + IA boost)
- **Criteri**: 
  - Tiri/minuto < 0.25

## 🎯 Mercati Over/Under - MIGLIORATI

### ✅ Over 0.5
- **Quando**: Nessun gol, partita aperta (20'-70')
- **Analisi IA**: Tiri, tiri in porta, tasso gol
- **Confidence**: 70-88% (base + IA boost)

### ✅ Over 1.5
- **Quando**: Già 1 gol, partita aperta (25'-75')
- **Analisi IA**: Tiri totali, tiri in porta
- **Confidence**: 72-90% (base + IA boost)

### ✅ Over 2.5 - MIGLIORATO
- **Quando**: 
  - Già 2 gol (30'-75')
  - Solo 1 gol ma partita molto aperta (40'-70')
- **Analisi IA**: Tiri, tasso gol, partita aperta
- **Confidence**: 68-92% (base + IA boost)
- **Criteri**: 
  - Già 2 gol: Tiri >= 15
  - 1 gol ma aperta: Tiri/minuto > 0.4, Tiri >= 20

### 🆕 Over 3.5
- **Quando**: Già 3 gol (40'-80')
- **Analisi IA**: Partita estremamente aperta
- **Confidence**: 70-90% (base + IA boost)

### 🆕 Under 1.5
- **Quando**: Max 1 gol, partita chiusa (50'-80')
- **Analisi IA**: Tiri/minuto bassi, tasso gol basso
- **Confidence**: 75-93% (base + IA boost)
- **Criteri**: 
  - Tiri/minuto < 0.2
  - Tiri totali < 15

### ✅ Under 2.5 - MIGLIORATO
- **Quando**: Max 2 gol, partita chiusa (60'-85')
- **Analisi IA**: Tiri/minuto bassi, tasso gol basso
- **Confidence**: 72-91% (base + IA boost)
- **Criteri**: 
  - Tiri/minuto < 0.25
  - Tiri totali < 20

### 🆕 Under 3.5
- **Quando**: Max 3 gol, partita chiusa (70'-85')
- **Analisi IA**: Tiri/minuto bassi
- **Confidence**: 80-95% (base + IA boost)
- **Criteri**: 
  - Tiri/minuto < 0.3

## 🤖 Miglioramenti IA

### Analisi Over Markets
L'IA ora analizza:
- ✅ Tiri totali (>10, >15, >20 = boost crescente)
- ✅ Tiri in porta (>5, >8 = boost crescente)
- ✅ Tasso gol (gol/minuto > 0.03, > 0.04 = boost)
- ✅ Partita equilibrata (differenza gol <= 1 = boost)

### Analisi Under Markets
L'IA ora analizza:
- ✅ Tiri totali (<8, <5, <3 = boost crescente)
- ✅ Tiri in porta (<3, <1 = boost crescente)
- ✅ Tasso gol (gol/minuto < 0.02, < 0.015 = boost)
- ✅ Possesso equilibrato (40-60% = partita chiusa)

## 📊 Esempi di Notifiche

### Over 2.5 (Partita Aperta)
```
🎯 OVER 2.5!

• Score: 1-1 al 45'
• Partita MOLTO APERTA:
  - Tiri: 22 (media: 0.49/min)
  - Tiri in porta: 9
• Alta probabilità altri gol → Over 2.5
• IA boost: +12%
• Confidence: 82%
```

### Under 1.5 HT
```
🎯 UNDER 1.5 HT!

• Score: 0-0 al 40'
• Partita CHIUSA:
  - Tiri: 6 (media: 0.15/min)
• Alta probabilità max 1 gol al primo tempo
• IA boost: +8%
• Confidence: 88%
```

## ✅ Risultato

### Mercati Disponibili
- **HT**: Over 0.5, 1.5 | Under 0.5, 1.5
- **Over/Under**: 0.5, 1.5, 2.5, 3.5
- **Totale**: **10 mercati** principali

### Miglioramenti
- ✅ Analisi IA avanzata per ogni mercato
- ✅ Confidence dinamica basata su statistiche
- ✅ Filtri intelligenti (no banali)
- ✅ Criteri più severi (confidence >= 70%)

**Sistema pronto con tutti i mercati richiesti!** 🎉








