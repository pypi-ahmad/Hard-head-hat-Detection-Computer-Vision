# Build a Windows app release

Use the supported Python 3.13 environment on Windows. Run commands from the
repository root. This builds an app ZIP; it does not publish a GitHub release.

## Prerequisites

1. Run `uv sync --locked`.
2. Prepare and train SHWD using the [README workflow](../README.md). A valid
   `artifacts/shwd/selected_model.json` must reference an existing SHWD checkpoint.
3. Download the official
   [Caddy 2.11.7 Windows amd64 archive](https://github.com/caddyserver/caddy/releases/tag/v2.11.7)
   to `artifacts/vendor/caddy/caddy_2.11.7_windows_amd64.zip`. The builder requires
   the archive itself; an installed `caddy.exe` is insufficient. It verifies this
   pinned SHA256 before bundling:

```text
0a1edc0b799512051c57071ce0e798d3f2cf67dc98171366f1d2b326072e3b06
```

## Build

```powershell
uv run python -m scripts.build_release --output dist --caddy artifacts/vendor/caddy/caddy_2.11.7_windows_amd64.zip
```

The version comes from `pyproject.toml`. Outputs are
`dist/HelmetCCTV-<version>-windows-x64.zip` and `dist/SHA256SUMS.txt`.
The builder refuses to overwrite an existing archive. Preserve published assets;
use a separate output directory for a local rebuild, which may have a different hash.

The ZIP includes selected SHWD weights, Caddy, locked dependency metadata, install
scripts, source, tests, documentation, and a file-hash manifest. Python and Python
packages download during installation. Training data, local camera configuration,
credentials, PKI, event data, and other runtime artifacts are excluded.

## Before publication

Verify archive contents and checksums. Extract into a fresh writable folder and
follow [CCTV_INSTALL.md](CCTV_INSTALL.md), including model/configuration checks and
a recorded-video demonstration. Record observed results in
[TEST_REPORT.md](../TEST_REPORT.md). Keep physical-camera, LAN-client, logon, and
site-accuracy checks explicit when they have not been performed. Publish only with
authorized repository access, including the ZIP, checksums, and release notes.
