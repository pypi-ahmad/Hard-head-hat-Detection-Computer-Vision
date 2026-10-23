param([switch]$AutoStart, [switch]$LocalOnly, [switch]$SkipSync)
$ErrorActionPreference = 'Stop'
$appRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $appRoot

if ($env:OS -ne 'Windows_NT' -or -not [Environment]::Is64BitOperatingSystem) {
    throw 'The installer requires 64-bit Windows.'
}
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    $existingUv = Join-Path $env:USERPROFILE '.local\bin\uv.exe'
    if (Test-Path -LiteralPath $existingUv) {
        $uvCommand = $existingUv
    } else {
        Write-Host 'Install uv with: winget install --id=astral-sh.uv -e'
        throw 'uv is required. Install it, then run Install.cmd again.'
    }
} else {
    $uvCommand = (Get-Command uv).Source
}
if (-not $SkipSync) {
    & $uvCommand sync --locked --no-dev
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
}
$localConfig = Join-Path $appRoot 'config\cameras.local.yaml'
if (-not (Test-Path -LiteralPath $localConfig)) {
    Copy-Item -LiteralPath (Join-Path $appRoot 'config\cameras.example.yaml') -Destination $localConfig
}
if (Test-Path -LiteralPath (Join-Path $appRoot 'release-manifest.json')) {
    $manifest = Get-Content -LiteralPath (Join-Path $appRoot 'release-manifest.json') -Raw | ConvertFrom-Json
    foreach ($entry in $manifest.files.PSObject.Properties) {
        $target = [IO.Path]::GetFullPath((Join-Path $appRoot $entry.Name))
        if (-not $target.StartsWith($appRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
            throw 'Invalid package manifest path.'
        }
        if ((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.Value) {
            throw ('Package checksum mismatch: ' + $entry.Name)
        }
    }
}
& $uvCommand run --no-sync python cctv.py --check
if ($LASTEXITCODE -ne 0) { throw 'Installation check failed. See docs\CCTV_INSTALL.md.' }
if (-not $LocalOnly) {
    & (Join-Path $PSScriptRoot 'Install-Gateway.ps1')
    if (-not (Test-Path -LiteralPath (Join-Path $appRoot 'config\Caddyfile.local'))) {
        & $uvCommand run --no-sync python scripts/configure_lan.py
        if ($LASTEXITCODE -ne 0) { throw 'LAN setup failed.' }
    }
}

$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut((Join-Path ([Environment]::GetFolderPath('Desktop')) 'Helmet CCTV.lnk'))
$shortcut.TargetPath = Join-Path $appRoot 'Start.cmd'
$shortcut.WorkingDirectory = $appRoot
$shortcut.Description = 'Continuous helmet and bare-head monitoring'
$shortcut.Save()
if ($AutoStart) {
    & (Join-Path $PSScriptRoot 'Enable-Autostart.ps1')
}
Write-Host 'Installed. Configure cameras in config\cameras.local.yaml, then open Helmet CCTV on the desktop.'
Write-Host 'See config\lan.local.json for the HTTPS LAN address, or http://127.0.0.1:8503 locally.'
