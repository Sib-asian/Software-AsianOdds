# 🔧 Fix Dimensione Docker - Guida

## Problema
Immagine Docker 8.5 GB > limite 4.0 GB Railway

## Soluzione Applicata

1. ✅ Creato `requirements.automation.txt` - SOLO dipendenze essenziali
   - NO torch (2-3 GB)
   - NO transformers (1-2 GB)  
   - NO xgboost (500 MB)
   - NO streamlit, plotly, seaborn

2. ✅ Dockerfile ottimizzato:
   - Multi-stage build
   - Copia selettiva solo file necessari
   - Rimozione modelli, training, data
   - Pulizia cache

3. ✅ .dockerignore aggiornato:
   - Esclude documentazione, test, modelli, dati

## Dimensione Attesa
- Prima: 8.5 GB
- Dopo: ~500 MB - 1 GB

## Prossimi Passi

1. Commit e push modifiche
2. Railway rileverà automaticamente e rifarà deploy
3. Verifica che build completi

## Se Ancora Troppo Grande

Opzioni:
1. Usare GitHub Actions invece (non ha limite dimensione)
2. Usare Render.com (limite 10 GB)
3. Ottimizzare ulteriormente rimuovendo più dipendenze

