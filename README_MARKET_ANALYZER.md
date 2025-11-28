# 📊 Market Movement Analyzer

Tool interattivo per analizzare i movimenti di **Spread** e **Total** nei mercati delle scommesse e generare interpretazioni e giocate consigliate.

## 🎯 Caratteristiche

- ✅ **Analisi automatica** dei movimenti Spread e Total
- ✅ **Matrice 4 combinazioni** per interpretazioni precise
- ✅ **Calcolo mercati HT** derivati da FT
- ✅ **Sistema di confidenza** (Alta/Media/Bassa)
- ✅ **Interfaccia web moderna** con Streamlit
- ✅ **Deploy su Streamlit Cloud** - disponibile online 24/7

## 🚀 Uso Locale

### Installazione

```bash
# Clona il repository (se non l'hai già)
git clone https://github.com/Sib-asian/Software-AsianOdds.git
cd Software-AsianOdds

# Installa le dipendenze (se non già installate)
pip install -r requirements.txt
```

### Avvio

```bash
# Metodo 1: Comando diretto
streamlit run market_movement_analyzer_app.py

# Metodo 2: File batch (Windows)
AVVIA_MARKET_ANALYZER.bat
```

L'app si aprirà automaticamente su `http://localhost:8501`

## 🌐 Uso Online (Streamlit Cloud)

L'app è disponibile online su Streamlit Cloud:

👉 **https://market-movement-analyzer.streamlit.app**

(URL aggiornato dopo il deploy - vedi `MARKET_ANALYZER_DEPLOY.md`)

## 📖 Come Usare

1. **Inserisci i valori**:
   - Spread Apertura (es: -1.5)
   - Spread Chiusura (es: -1.0)
   - Total Apertura (es: 2.5)
   - Total Chiusura (es: 2.75)

2. **Clicca "Analizza Movimenti"**

3. **Visualizza i risultati**:
   - Analisi dei movimenti
   - Interpretazione combinata
   - Mercati FT consigliati
   - Mercati HT consigliati
   - Livello di confidenza

## 🎓 Esempi

Nella sidebar dell'app trovi esempi pre-compilati:
- **Esempio 1**: Favorito forte + Partita viva
- **Esempio 2**: Favorito cala + Partita chiusa

## 📊 Matrice Combinazioni

L'app analizza 4 combinazioni principali:

| Spread | Total | Interpretazione |
|--------|-------|-----------------|
| 🔽 Si indurisce | 🔼 Sale | Favorito più forte e partita viva → GOAL/Over |
| 🔽 Si indurisce | 🔽 Scende | Favorito solido ma tattico → 1/Under/NOGOAL |
| 🔼 Si ammorbidisce | 🔼 Sale | Match equilibrato e aperto → GOAL/Over/X2 |
| 🔼 Si ammorbidisce | 🔽 Scende | Fiducia calante + ritmo basso → Under/X/NOGOAL |

## 🔧 Tecnologie

- **Python 3.8+**
- **Streamlit** per l'interfaccia web
- **Librerie standard** (enum, dataclasses)

## 📝 File Correlati

- `market_movement_analyzer_app.py` - App Streamlit principale
- `market_movement_analyzer.py` - Versione console (opzionale)
- `MARKET_ANALYZER_DEPLOY.md` - Guida deploy su Streamlit Cloud
- `AVVIA_MARKET_ANALYZER.bat` - Script di avvio rapido (Windows)

## 🤝 Contributi

Sentiti libero di migliorare l'app! Aggiungi:
- Nuove interpretazioni di movimenti
- Mercati aggiuntivi
- Miglioramenti UI

## 📄 Licenza

Parte del progetto Software-AsianOdds.

