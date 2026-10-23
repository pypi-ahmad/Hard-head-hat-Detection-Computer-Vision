# Changelog

## [0.2.0] — 2026-10-23

- Added continuous RTSP/HTTP(S) camera monitoring with independent capture and inference.
- Added automatic reconnect, stale-worker indicators, per-camera status, and a dark CCTV dashboard.
- Added Windows installation/launch scripts, optional logon startup, and an HTTPS LAN gateway.
- Added event-session rotation, crash recovery, age/storage retention, and UTC confirmation times.
- Added a Windows app ZIP containing the selected SHWD model, Caddy, notices, and file checksums.
- Verified two RTSP feeds, reconnect isolation, HTTPS authentication, and browser rendering locally.
- Retained the qualified public-SHWD checkpoint while camera adaptation remains in progress.

Actual IP-camera deployment and event accuracy still require site qualification.

### SHWD application overhaul included in 0.2.0

- Made SHWD the app and CLI default; retained PPE as an explicit experiment.
- Removed Pictor data, weights, evaluation/video artifacts, and active application and conversion support.
- Added per-stream ByteTrack, helmet-required areas, timed events, and local exports.
- Added durable annotation review, approved-only dataset export, and controlled fine-tuning.
- Prepared 154 unreviewed frames from the four supplied videos.
- Recorded the user's review confirmation and exported 154 approved frames as camera-v1.
- Verified physical webcam capture and local WebRTC video return (29.3 FPS).
- Fixed SHWD provenance drift caused by automatic JPEG repairs; original hashes are retained.
- Validate SHWD integrity before creating a camera training run directory.


This changelog records notable project changes.

This project follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and aims to follow [Semantic Versioning](https://semver.org/).

## Earlier development log

These entries preserve the earlier implementation stages. Later removal of Pictor
and camera verification are recorded above.

### Added

- SHWD helmet/bare-head and Pictor helmet/vest/worker detection modes.
- Resumable official dataset downloads, source hashes, format conversion, and explicit
  exclusion reports for unlabelled files, unknown classes, and cross-split duplicates.
- Profile-specific Small training, evaluation, checkpoint selection, and full-video checks.
- Python 3.13 environment managed by uv, with a lockfile and official CUDA 13.2 PyTorch wheels.
- Dataset audit with image decoding, label checks, source hashes, overlap detection,
  and annotated contact sheets.
- CUDA smoke training and sequential YOLO26 Small, Medium, and Large experiments.
- Validation threshold selection that maximizes macro violation recall with at least
  80% macro precision, plus test evaluation and video benchmarks.
- Browser camera input and cancellable downloads of public prerecorded videos.
- Benchmarks for the supplied custom clip and all three source MP4 videos,
  with per-video and aggregate latency measurements.
- Regression checks for confidence forwarding, checkpoint paths, concurrency,
  URL restrictions, model caching, and media cleanup.
- [Contributor guidelines](AGENTS.md), [dataset review](docs/DATASET_REVIEW.md),
  and [verification report](TEST_REPORT.md).

### Changed

- Extracted shared inference, annotation, media, and path modules.
- Moved the historical notebook into `notebooks/`.
- Counts now describe PPE items and violation detections.
- Exposed all five YOLO26 presets: Nano, Small, Medium, Large, and Extra Large.
- Updated the [README](README.md) with setup, usage, training progress commands,
  and evaluation outputs.
- Replaced the stress harness with pytest assertions and nonzero exit codes on failure.

### Fixed

- Failed PPE test evaluation preserves the previous selected model; successful selections
  are published atomically.
- The app reports invalid selected-model metadata without crashing;
  confidence values must be finite and between zero and one.
- Dataset paths resolve independently of the current working directory.
- Training reports the actual saved checkpoint and propagates failures.
- Confidence reaches prediction; cached models serialize concurrent inference.
- Video processing releases decoders and temporary uploads.
- Download redirects are checked for private network destinations; text playlists
  are rejected before video decoding.
- Cancelled downloads finish temporary-file cleanup before interpreter shutdown on Windows.

### Verification at that stage

- Automated run on 23 October 2026: 87 tests passed; Ruff passed.
- Real CUDA training smoke, historical YOLOv8 inference, direct video processing,
  and YouTube transfer cancellation were verified.
- All five full training runs completed 50 epochs. PPE selected YOLO26 Medium;
  SHWD and Pictor selected YOLO26 Small under the validation rule.
- SHWD test bare-head precision/recall: 81.5%/95.5%. Pictor test macro
  precision/recall: 45.6%/58.1%, with no successful vest detections.
- All three selected modes completed four full videos: 8,179 verified output frames
  per mode. Original PPE source hashes passed and derived caches were restored.
- [Completed model results](docs/MODEL_RESULTS.md) records metrics and limitations.
- Physical browser camera access and WebRTC transport were verified in the later follow-up above.

## [2026-06-13]

### Added

- OSS companion documentation initialized (license, contributing, security, conduct, changelog).
