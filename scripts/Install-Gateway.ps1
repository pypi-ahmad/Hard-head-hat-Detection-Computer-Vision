$ErrorActionPreference = 'Stop'
$appRoot = Split-Path -Parent $PSScriptRoot
$destination = Join-Path $appRoot 'vendor\caddy'
if (Test-Path -LiteralPath (Join-Path $destination 'caddy.exe')) { return }
$downloadDir = Join-Path $appRoot 'artifacts\downloads'
New-Item -ItemType Directory -Path $downloadDir -Force | Out-Null
$archive = Join-Path $downloadDir 'caddy_2.11.7_windows_amd64.zip'
Invoke-WebRequest -Uri 'https://github.com/caddyserver/caddy/releases/download/v2.11.7/caddy_2.11.7_windows_amd64.zip' `
    -OutFile $archive -UseBasicParsing
$expected = '0a1edc0b799512051c57071ce0e798d3f2cf67dc98171366f1d2b326072e3b06'
if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected) {
    throw 'Gateway download checksum mismatch.'
}
Expand-Archive -LiteralPath $archive -DestinationPath $destination
