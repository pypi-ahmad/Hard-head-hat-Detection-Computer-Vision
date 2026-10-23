param([switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
$appRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $appRoot
$pythonCommand = Join-Path $appRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonCommand)) {
    throw 'Install dependencies first using Install.cmd.'
}
$gatewayConfig = Join-Path $appRoot 'config\Caddyfile.local'
$runnerArguments = @('cctv.py')
if (Test-Path -LiteralPath $gatewayConfig) {
    $runnerArguments += '--lan'
}
if (-not $NoBrowser) { $runnerArguments += '--open-browser' }
& $pythonCommand @runnerArguments
exit $LASTEXITCODE
