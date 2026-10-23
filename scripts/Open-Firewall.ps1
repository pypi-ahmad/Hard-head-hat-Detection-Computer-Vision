# Run in an administrator PowerShell on the CCTV PC.
$ErrorActionPreference = 'Stop'
$appRoot = Split-Path -Parent $PSScriptRoot
$program = Join-Path $appRoot 'vendor\caddy\caddy.exe'
if (-not (Test-Path -LiteralPath $program)) { throw 'Install the LAN gateway first.' }
New-NetFirewallRule -DisplayName 'Helmet CCTV HTTPS LAN' -Direction Inbound -Action Allow `
    -Protocol TCP -LocalPort 8443 -RemoteAddress LocalSubnet -Profile Private -Program $program
