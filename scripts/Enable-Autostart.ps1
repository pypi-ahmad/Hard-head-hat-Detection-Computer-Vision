$ErrorActionPreference = 'Stop'
$appRoot = Split-Path -Parent $PSScriptRoot
$script = Join-Path $PSScriptRoot 'Start.ps1'
$identity = [Security.Principal.WindowsIdentity]::GetCurrent().Name
$taskName = 'HelmetCCTV-' + (Split-Path -Leaf $appRoot)
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -WorkingDirectory $appRoot `
    -Argument ('-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File "' + $script + '" -NoBrowser')
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $identity
$principal = New-ScheduledTaskPrincipal -UserId $identity -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew `
    -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger `
    -Principal $principal -Settings $settings -Force | Out-Null
Write-Host ('Enabled logon startup: ' + $taskName)
