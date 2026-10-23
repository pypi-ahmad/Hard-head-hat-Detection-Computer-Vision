# Install Helmet CCTV on Windows

Version 0.2.0 provides continuous helmet and bare-head monitoring from RTSP and
HTTP(S) video streams. The Windows app package includes the selected SHWD model,
a dark dashboard, and Caddy for authenticated HTTPS access across your LAN.
It records confirmed events and snapshots; it does not replace an NVR or record
continuous footage.

## Install

1. Download `HelmetCCTV-0.2.0-windows-x64.zip` and `SHA256SUMS.txt` from the
   [GitHub release](https://github.com/pypi-ahmad/Hard-head-hat-Detection-Computer-Vision/releases/tag/v0.2.0).
   Compare `Get-FileHash .\HelmetCCTV-0.2.0-windows-x64.zip -Algorithm SHA256`
   with the published checksum, then extract into a permanent writable folder,
   such as `C:\Users\YOUR_NAME\Apps\HelmetCCTV`.
2. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if needed:
   `winget install --id=astral-sh.uv -e`. Open a new terminal afterward.
3. Run `Install.cmd`. It installs Python 3.13 and locked dependencies, verifies
   the package files, checks the model, and creates a desktop shortcut. The first
   installation needs internet access and several GB of disk space. NVIDIA
   acceleration uses the locked CUDA 13.2 wheels and a compatible driver; CPU
   inference is supported, with throughput dependent on the PC.
4. Enter the PC's LAN name or fixed IPv4 address and choose a dashboard username
   and password. Only the password hash is stored in `config/Caddyfile.local`.
   Use `Install.cmd -LocalOnly` to skip the LAN gateway.

Camera streams need configuration before live monitoring. An empty camera list
is valid and displays setup guidance. The app ZIP includes its trained model;
GitHub's automatically generated source ZIP does not.

## Add cameras

Edit `config/cameras.local.yaml`, created by the installer:

```yaml
confidence: 0.12
inference_fps: 10
retention_days: 7
storage_mb: 2048
cameras:
  - id: entrance
    name: Construction entrance
    url_env: CCTV_ENTRANCE_URL
    helmet_required_zone: [0.0, 0.0, 1.0, 1.0]
  - id: workshop
    name: Workshop
    url_env: CCTV_WORKSHOP_URL
```

Camera IDs must be unique lowercase names. Each `url_env` names an environment
variable available to the account running CCTV. Supply the complete camera URL
through your existing environment or credential provisioning system; the app reads
it with `os.environ`. It does not create or overwrite environment variables.
RTSP paths vary by camera/NVR; obtain the correct stream URL from its configuration.
Percent-encode special characters in URL credentials. Relaunch the launcher after
changing its environment. Never put actual URLs/passwords in Git or issue reports.

`helmet_required_zone` contains normalized left, top, right, bottom coordinates.
Omit it for detection without violation recording. One second of observed bare-head
evidence confirms an event. Counts refer to detections, not unique workers.

Verify setup and camera decoding:

```powershell
uv run python cctv.py --check
uv run python cctv.py --probe
```

`--check` validates configuration and loads the model without opening a camera.
`--probe` connects and decodes one frame from each camera. Neither prints camera URLs.

## LAN access

Open `Start.cmd` or the **Helmet CCTV** desktop shortcut. Read your HTTPS address
from `config/lan.local.json`, for example `https://cctv-pc:8443`.
The gateway authenticates all requests and proxies to Streamlit on `127.0.0.1:8503`.
Monitoring continues after closing the browser. Closing the launcher stops it.

Caddy creates a local certificate authority under
`artifacts/cctv/pki/pki/authorities/local/`. After first startup, copy **only**
`root.crt` to trusted LAN clients and import it into their trusted root certificate
store. Windows supports `certutil -user -addstore Root .\root.crt`.
Confirm its SHA256 fingerprint with the administrator before trusting it.
Keep all `.key` files on the CCTV PC. Caddy's documentation explains
[local HTTPS trust](https://caddyserver.com/docs/automatic-https#local-https).

On the CCTV PC, allow TCP 8443 from the private LAN by running this once in an
administrator PowerShell:

```powershell
.\scripts\Open-Firewall.ps1
```

The rule is limited to the Caddy program, Private network profile, and local subnet.
Use the configured PC name/IP in client browsers. Keep port 8503 bound to localhost.
The gateway uses [hashed-password authentication](https://caddyserver.com/docs/caddyfile/directives/basic_auth).
To change the login or hostname, run `uv run python scripts/configure_lan.py`, then
restart CCTV. Configure a fixed IP or stable DNS name to avoid address changes.

## Automatic startup and operation

```powershell
.\scripts\Enable-Autostart.ps1
.\scripts\Disable-Autostart.ps1
```

Startup uses Windows Task Scheduler at **user logon**, with three restart attempts
after a process failure. This is not a service that starts before login. Configure
the task under the same Windows account that has access to the camera environment.
Keep the PC awake while monitoring. The task name is `HelmetCCTV-<folder-name>`.

Each camera reconnects with a delay of 1–30 seconds after an unavailable stream.
Decoder open/read operations time out after five seconds. Capture and inference
run separately and retain only the latest frame. The dashboard marks stale capture
or inference and detects a missing worker heartbeat. Model/storage errors stop the
runner with an error so the scheduled task can restart it.

Events live in `artifacts/cctv/events/`, with a JSON journal and confirmation
snapshots. Closed sessions are pruned every minute after seven days or above the
configured storage budget. Active sessions are preserved, so the budget is a soft
limit. Sessions rotate hourly or after 1,000 events. Interrupted journals retain
unknown event end times. Review and download events through **Recorded events**.
This release sends no email, SMS, or external alarm notifications.

For a local recorded-video demonstration:

```powershell
uv run python cctv.py --demo "C:\path\to\video.mp4"
```

## Validation and updates

Two real RTSP connections were tested against a local MediaMTX server using a
supplied recorded clip. Disconnect/reconnect and the unaffected second feed passed.
HTTPS authentication and browser rendering passed locally. An actual IP camera,
a second physical LAN client, long-duration uptime, and camera-specific accuracy
still need checks at the installation site. See [TEST_REPORT](../TEST_REPORT.md).
Start with one camera; add feeds while watching actual inference FPS and latency.

The included model has SHWD public-test bare-head precision 81.5% and recall 95.5%.
These are image metrics, not a guarantee for a site's CCTV view. Camera adaptation
is a separate training run and has not supplied a promoted checkpoint for this
release. Reviewed event intervals are still required for event accuracy metrics.

Before updating, stop monitoring and back up `config/*.local*` and `artifacts/cctv/`.
Keep credential material and PKI private. Install a new release in a separate folder,
restore your local configuration/runtime, and rerun the checks. Disable the old
startup task before enabling the new one. Remove the shortcut, startup task, and
the named firewall rule when uninstalling; delete recordings only when intended.
