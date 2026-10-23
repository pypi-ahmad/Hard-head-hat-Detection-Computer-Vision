# Verification report

## CCTV installation release 0.2.0 — 23 October 2026

- Full suite: 106 tests passed after adding CCTV configuration, worker lifecycle,
  credential redaction, reconnect, stale inference, retention, crash recovery,
  gateway configuration, and dashboard checks. Ruff passed.
- Two actual RTSP/TCP connections were opened to MediaMTX 1.21.2 on loopback,
  publishing the supplied recorded clip. Both processed frames using the SHWD
  detector on CPU. The first publisher was stopped and restarted; its worker
  recovered after two reconnect attempts while the second continued processing.
  The recovery snapshot had 8 and 27 processed frames. Evidence:
  `artifacts/cctv_rtsp_test/verification.json`. This was not a physical IP camera test.
- Caddy 2.11.7 returned 401 for absent/wrong credentials and 200 for the correct
  credentials. TLS was validated against the generated local root CA. Chromium
  authenticated and rendered the dark dashboard without a Streamlit exception.
  The test found and fixed Windows CRLF translation during password hashing.
  Evidence: `artifacts/cctv_gateway_test/verification.json`.
- Windows PowerShell scripts parsed successfully. Caddy validated the generated
  HTTPS/authentication configuration. Its downloaded archive matched the upstream
  SHA512 checksum; the build and download script pin its SHA256 hash.
- A fresh app extraction installed 98 locked runtime packages. The actual Windows
  installer passed its package checksums, model load, local configuration creation,
  and desktop shortcut creation; the previous desktop state was restored afterward.
  The extracted runner then started its worker, Streamlit, and Caddy together,
  processed a demo frame on CPU, exited normally, and released ports 8503/8443.
  This short run included a 14.3-second cold CPU inference during concurrent GPU
  training; it is not a throughput measurement. Preview freshness checks include
  the processed frame's age so slow inference cannot label old content as live.

Physical IP-camera/NVR compatibility, a second LAN client, long-duration uptime,
real Windows logon restart, and per-site detection/event accuracy remain unverified.
The app package uses the existing selected SHWD checkpoint; the separate camera
adaptation run is still training. No new camera accuracy result is claimed.

The published app ZIP was downloaded back from GitHub and its SHA256 matched the
local release build: `163251c91f41e5f399b0164040a9ec4f27eab0b8ed125a1d85a760e751429426`.
Its size is 36,965,746 bytes. The archive passed CRC and file-manifest checks;
76 payload files were hashed. See [release notes](docs/RELEASE_0.2.0.md).

The sections below retain earlier check results in execution order. The current
release suite result is 106 passed; lower counts describe earlier stages.

## SHWD overhaul — 23 October 2026

- Pytest: 87 passed after adding review/export, event-timing, and training-gate checks.
- Ruff: passed. Dependency compatibility: 103 installed packages compatible.
- All four supplied videos processed with SHWD and independent ByteTrack state:
  921 custom-clip frames, 1,826 hardhat frames, 648 Indian-worker frames, and 4,784
  JapanPPE frames. All 8,179 output frames decoded.
- First tracked full-video pass measured 23.93, 52.22, 26.56, and 44.00 FPS respectively.
  These include local processing/encoding, not browser/network latency; they are
  not a guarantee of live-camera throughput.
- Chromium rendered the annotation editor. Drawing, resizing, relabeling, deleting the final box, saving, and reload persistence
  passed. The browser test exposed a stale frame-picker object; selecting by stable
  frame ID fixed it. The automated temporary edit was restored, and no frames were approved.
- `review_data.py extract` decoded all four videos and prepared 154 unreviewed drafts.
- Pictor is absent from active Python source, profile choices, and converter paths.

## Remaining verification limits

Physical webcam capture and local WebRTC transport passed in headed Chromium:
the real USB camera supplied 640×480 at 30 FPS, the peer connection reached connected,
and returned decoded frames advanced from 12 to 306 over about 10 seconds (29.3 FPS).
The app displayed inference metrics. No synthetic media was used, no camera imagery
was saved, and the browser/camera was closed after the check. Remote HTTPS/TURN
deployment remains unverified. Evidence: `artifacts/physical_webrtc_verification.json`.

The user confirmed personal review and correction of all 154 draft annotations.
The database was backed up, approval recorded, and `camera-v1` exported with
112 train, 31 validation, and 11 test frames. Fine-tuning is in progress. Independent
camera accuracy, event recall, false-alert rate, delay, and subgroup metrics remain
unmeasured. Event intervals still lack recorded review.

The bulk deletion command was rejected by automatic approval review. Narrower,
verified exact-path deletions succeeded: Pictor data, weights, evaluation/video
artifacts, and obsolete orchestration files are removed. Shared runtime scripts
and reports no longer include its profile.

Existing SHWD and experimental PPE metrics remain documented in
[model results](docs/MODEL_RESULTS.md). They predate this software overhaul and are
not measurements of newly trained camera models.

A real PyAV camera callback passed using SHWD inference. A 120-frame cold-start
check measured 11.61 FPS and 78.8 ms p95; startup was included. This short check
did not meet the 15 FPS target. Full-clip averages were higher; long-duration browser
throughput and queue behavior remain unverified. Local Markdown links: 22 checked,
none broken.

A subsequent 120-frame run measured 17.62 FPS after 20 warm-up frames, with
78.2 ms p95 latency; its cold-start average was 11.19 FPS. Saved review edits survive
a browser reload; all 154 frames remained unreviewed at that point, before the
subsequent user approval recorded above.

The normalized rectangular-area drawing control passed in Chromium.

## Camera training preparation follow-up

The first attempt stopped before training because SHWD image hashes had changed.
All 7,564 source and label records were checked. Each of the 64 changed JPEGs matched
Ultralytics' repair output byte-for-byte; original downloads and labels were unchanged.
Preparation now repairs JPEG end markers before hashing and records the original
hash separately. The previous manifest is backed up. Integrity validation now runs
before creating training output; the failed partial directory was moved to
`bin/camera-v1-failed-integrity`.

Focused regression checks passed: 16 dataset tests and 8 review/event tests.
The complete suite then passed: 89 tests. Ruff also passed.
