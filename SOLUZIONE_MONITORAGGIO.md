# Soluzione: Come Monitorare i Log

## ✅ Soluzione 1: Usa i File .BAT (PIÙ SEMPLICE)

Ho creato due file `.bat` che funzionano senza problemi di permessi:

### Monitoraggio Completo
1. Vai nella cartella: `C:\Users\aless\OneDrive\Desktop\Software-AsianOdds-main`
2. Fai doppio clic su: `monitor_logs.bat`
3. Vedrai i log in tempo reale

### Solo Notifiche
1. Vai nella cartella: `C:\Users\aless\OneDrive\Desktop\Software-AsianOdds-main`
2. Fai doppio clic su: `monitor_notifiche.bat`
3. Vedrai solo le notifiche

---

## ✅ Soluzione 2: Comandi PowerShell Diretti

Apri PowerShell e incolla questi comandi:

### Monitoraggio Completo
```powershell
cd "C:\Users\aless\OneDrive\Desktop\Software-AsianOdds-main"
Get-Content automation_24h.log -Wait -Tail 50
```

### Solo Notifiche
```powershell
cd "C:\Users\aless\OneDrive\Desktop\Software-AsianOdds-main"
Get-Content automation_24h.log -Wait | Select-String -Pattern "opportunità|notifica|Live betting opportunity"
```

---

## ✅ Soluzione 3: Abilita Script PowerShell (OPZIONALE)

Se vuoi usare gli script `.ps1`, esegui questo comando in PowerShell **come Amministratore**:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Poi puoi usare:
```powershell
.\monitor_logs.ps1
```

**Nota**: Questa modifica è permanente per il tuo utente.

---

## 🎯 RACCOMANDAZIONE

**Usa i file `.bat`** - sono più semplici e non richiedono permessi speciali!

1. Vai nella cartella del progetto
2. Fai doppio clic su `monitor_logs.bat`
3. Vedrai i log in tempo reale








