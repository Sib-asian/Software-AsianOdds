# 🔍 Problemi Critici Identificati

## ❌ PROBLEMA 1: Identificazione Home/Away negli Eventi
**Gravità: CRITICA**

La logica attuale per determinare se un gol è home/away è **ERRATA**:
```python
team_id = str(team_info.get("id", ""))
is_home_goal = "home" in team_id.lower() or team_id == ""
```

**Problema**: API-SPORTS negli eventi usa `team.id` che è un **numero** (es. 1234), non "home" o "away". Dobbiamo confrontare con i team IDs della fixture.

**Soluzione**: Confrontare `team.id` dell'evento con `teams.home.id` e `teams.away.id` della fixture.

---

## ❌ PROBLEMA 2: Gestione Errori API Silenziosa
**Gravità: ALTA**

Se eventi/statistiche falliscono, il sistema continua ma i mercati complessi restano "unknown" senza notificare.

**Soluzione**: 
- Ritentare con backoff esponenziale
- Loggare errori critici
- Notificare se fallisce per partite importanti

---

## ❌ PROBLEMA 3: Parsing Statistiche Fragile
**Gravità: MEDIA**

La logica di parsing statistiche potrebbe fallire per:
- Valori "None" o "N/A" come stringhe
- Formati diversi tra leghe
- Statistiche mancanti

**Soluzione**: Validazione robusta e fallback.

---

## ❌ PROBLEMA 4: Edge Cases Next Goal
**Gravità: MEDIA**

- Se segnale è dopo l'ultimo gol → dovrebbe essere "loss" ✅ (già gestito)
- Se segnale è dopo la fine partita → edge case non gestito
- Se ci sono gol ai supplementari → potrebbero essere inclusi erroneamente

**Soluzione**: Filtrare eventi per tempo regolamentare.

---

## ❌ PROBLEMA 5: Rate Limit API Non Gestito
**Gravità: ALTA**

3 chiamate API per partita (fixture + events + statistics) possono esaurire quota rapidamente.

**Soluzione**: 
- Cache risultati
- Batch processing
- Priorità partite

---

## ❌ PROBLEMA 6: Eventi/Statistiche Non Salvati
**Gravità: MEDIA**

Eventi/statistiche non vengono salvati nel database. Se partita viene riprocessata, dobbiamo rifare chiamate API.

**Soluzione**: Salvare eventi/statistiche in tabella separata.

---

## ❌ PROBLEMA 7: Validazione Dati Mancante
**Gravità: MEDIA**

Non c'è validazione che:
- Eventi siano coerenti con risultato finale
- Statistiche siano ragionevoli
- Dati non siano corrotti

**Soluzione**: Validazione incrociata.

---

## ❌ PROBLEMA 8: Corner/Cards Senza Threshold
**Gravità: BASSA**

Se mercato non ha numero (es. "corner"), regex fallisce silenziosamente.

**Soluzione**: Validazione e log errore.

---

## ❌ PROBLEMA 9: Team Goal Anytime - Gol Multipli
**Gravità: BASSA**

Se ci sono gol multipli, la logica potrebbe non essere ottimale.

**Soluzione**: Verificare che almeno un gol sia stato segnato.

---

## ❌ PROBLEMA 10: Gestione Partite Senza Eventi
**Gravità: MEDIA**

Se eventi non disponibili, tutti i mercati next_goal restano "unknown" invece di essere gestiti meglio.

**Soluzione**: Fallback intelligente o ritentare più tardi.


