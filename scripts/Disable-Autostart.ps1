$ErrorActionPreference = 'Stop'
$appRoot = Split-Path -Parent $PSScriptRoot
$taskName = 'HelmetCCTV-' + (Split-Path -Leaf $appRoot)
Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
Write-Host ('Disabled logon startup: ' + $taskName)
