#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Versione semplificata per creare PR velocemente
    
.DESCRIPTION
    Wrapper semplificato che chiama create_pr.ps1 con parametri di default intelligenti
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$Message,
    
    [string]$Description = ""
)

# Se description è vuota, usa il message
if ([string]::IsNullOrWhiteSpace($Description)) {
    $Description = $Message
}

# Chiama lo script principale
& "$PSScriptRoot\create_pr.ps1" -CommitMessage $Message -PRDescription $Description

