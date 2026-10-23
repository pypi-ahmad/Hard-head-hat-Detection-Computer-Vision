# Helmet CCTV 0.2.0

Continuous helmet and bare-head monitoring for a Windows PC, with a dark dashboard
available across the LAN through authenticated HTTPS.

## Install the app

Download `HelmetCCTV-0.2.0-windows-x64.zip`, verify it using `SHA256SUMS.txt`, and
extract it into a permanent writable folder. Install uv if needed, then run
`Install.cmd`. It downloads Python and locked dependencies, checks the bundled
SHWD checkpoint, and prompts for the LAN login. Configure cameras as described
in [CCTV_INSTALL.md](https://github.com/pypi-ahmad/Hard-head-hat-Detection-Computer-Vision/blob/v0.2.0/docs/CCTV_INSTALL.md), then launch `Start.cmd`.

The app ZIP includes the trained model and Caddy gateway. GitHub's automatic
source archives contain source code and do not include the trained checkpoint.

## Included

- Multiple RTSP/HTTP(S) feeds, independent capture, latest-frame inference,
  automatic reconnection, and stale/offline indicators.
- Live camera status, inference FPS/latency, helmet/bare-head counts, and event review.
- Configurable helmet-required regions, event snapshots/journals, hourly rotation,
  crash recovery, and closed-session age/storage retention.
- Windows desktop launch, optional user-logon startup, and a restricted LAN firewall helper.
- The preceding SHWD overhaul, annotation review/export workflow, training/evaluation
  tools, documentation updates, and removal of active Pictor support/artifacts.

## Verification and limits

106 tests passed, Ruff passed, and the PowerShell scripts parsed. Two real RTSP
connections to a local test server passed disconnect/reconnect isolation checks.
Local HTTPS authentication and Chromium rendering passed. The physical USB webcam
and WebRTC path also passed earlier verification.

This is an installation release for site evaluation. A physical IP camera/NVR,
another LAN client, long-duration monitoring, and real logon startup still need
checks at the target site. Public SHWD image scores do not qualify a site's camera
accuracy. Camera adaptation is still running and reviewed event intervals remain
outstanding. The included model is the existing selected SHWD YOLO26 Small
checkpoint (public-test bare-head precision 81.5%, recall 95.5%).

The first installation needs internet access and several GB of disk space.
Event snapshots are retained locally; continuous NVR recording and external alarm
notifications are not included. Component licenses are listed in
[THIRD_PARTY_NOTICES.md](https://github.com/pypi-ahmad/Hard-head-hat-Detection-Computer-Vision/blob/v0.2.0/THIRD_PARTY_NOTICES.md).
