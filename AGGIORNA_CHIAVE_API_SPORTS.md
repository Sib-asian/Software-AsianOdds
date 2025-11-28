# 🔑 Configurazione API-SPORTS

## Chiave API fornita
```
94d5ec5f491217af0874f8a2874dfbd8
```

## Istruzioni

### Opzione 1: Aggiungere al file .env (CONSIGLIATO)
Aggiungi questa riga al file `.env`:
```
API_FOOTBALL_KEY=94d5ec5f491217af0874f8a2874dfbd8
```

### Opzione 2: Variabile d'ambiente Windows
```powershell
[System.Environment]::SetEnvironmentVariable("API_FOOTBALL_KEY", "94d5ec5f491217af0874f8a2874dfbd8", "User")
```

## Verifica
Dopo aver aggiunto la chiave, riavvia il servizio e verifica nei log:
```
✅ API-Football provider disponibile
```

## Note
- API-SPORTS usa lo stesso endpoint di API-Football (v3.football.api-sports.io)
- La chiave funziona con entrambi i servizi
- Piano gratuito: 100 chiamate/giorno
- Dati live aggiornati ogni 15 secondi








