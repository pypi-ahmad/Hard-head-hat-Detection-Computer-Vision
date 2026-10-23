# Helmet and bare-head detection

Detect helmets and bare heads in images, uploaded video, browser camera frames,
and public prerecorded video links. SHWD YOLO26 Small is the default. The original
ten-class construction PPE workflow remains an explicit experimental mode.

## Install as a CCTV monitor

The [v0.2.0 Windows app package](https://github.com/pypi-ahmad/Hard-head-hat-Detection-Computer-Vision/releases/tag/v0.2.0)
includes the trained SHWD model, `Install.cmd`, `Start.cmd`, and an authenticated
HTTPS gateway for LAN access. Extract it into a permanent folder, run `Install.cmd`,
then configure cameras using the [installation guide](docs/CCTV_INSTALL.md).
The first install downloads Python and locked dependencies. GitHub's source ZIP
does not include the trained model.

The CCTV worker accepts multiple RTSP/HTTP(S) camera streams, reconnects automatically,
and runs while the browser is closed. The dark dashboard shows per-camera status,
latest detections, FPS, latency, and recorded events. Optional Windows logon startup
uses Task Scheduler. Closed event sessions have age and storage retention limits.
Camera URLs stay in the server account's environment; they are not sent to clients.

```powershell
uv run python cctv.py --check
uv run python cctv.py --probe
uv run python cctv.py
```

The local dashboard uses `http://127.0.0.1:8503`; configured LAN access uses HTTPS
on port 8443. The original app also has a **CCTV monitor** workspace. Two RTSP feeds
and local HTTPS login have passed runtime checks. Actual IP-camera compatibility,
access from a second LAN PC, and site accuracy still need installation checks.

### Published package

- [Download Windows x64 app](https://github.com/pypi-ahmad/Hard-head-hat-Detection-Computer-Vision/releases/download/v0.2.0/HelmetCCTV-0.2.0-windows-x64.zip)
- [Download SHA256SUMS.txt](https://github.com/pypi-ahmad/Hard-head-hat-Detection-Computer-Vision/releases/download/v0.2.0/SHA256SUMS.txt)
- ZIP size: 36,965,746 bytes. SHA256:

```text
163251c91f41e5f399b0164040a9ec4f27eab0b8ed125a1d85a760e751429426
```

This package contains the v0.2.0 source snapshot. Documentation on `main` can be
newer; updating these files does not update an installed app.

## Run locally

Use Python 3.13 and uv. The lockfile uses official CUDA 13.2 PyTorch wheels;
GPU execution needs a compatible NVIDIA driver. CPU inference is supported.

```powershell
uv sync --locked
uv run streamlit run app.py
```

Select **Helmets and bare heads**, then **Selected checkpoint**. Local trained
weights are ignored by Git; a fresh source checkout must train them first or use
the app release, which includes the selected SHWD model.

```powershell
uv run data_setup.py
uv run run_experiments.py
```

The second command audits SHWD, runs a training smoke check, trains Small when no
matching completed run exists, evaluates it, and processes the four supplied videos.
Use `--profile ppe` explicitly for historical experiments. No equipment/worker
dataset is included in the supported workflow.

## Detection and events

- **Image:** upload JPG/PNG and select Analyze image.
- **Video:** upload MP4/AVI/MOV; Start, pause/resume, stop, and adjust playback speed.
- **Browser camera:** allow camera access. Remote use requires HTTPS and may need TURN.
- **Video link:** download a public YouTube or direct video, then Start.

Confidence defaults to **0.12**, favoring bare-head recall. Detection counts are
per-frame observations, not unique workers. Enable **Helmet-required area** to
record events for the entire frame or a normalized rectangular area. Without an
area, the app displays detections without declaring violations.

Each stream owns its ByteTrack instance. One second of observed bare-head evidence
confirms an event; short gaps do not add evidence. One second without qualifying
evidence closes the episode. Recorded videos use source time, so playback speed
and pausing do not change event timing. Track IDs apply only to one stream.

Local event images and JSON live under `artifacts/events/`. Download JSON/CSV from
the app. Changing source, model, threshold, or area starts new stream state.

Downloads are capped at 1,000 MB. YouTube uses up to 720p and may require Node.js;
upstream bot/sign-in restrictions can prevent downloads. Private network URLs,
playlists, live streams, embedded credentials, and unsupported ports are rejected.

## Review the camera dataset

```powershell
uv run review_data.py extract
```

This samples one frame every two seconds from the custom clip and the three videos
under `datasets/safety/source_files/source_files/`. The prepared pack contains
154 frames. New predictions are **unreviewed drafts**, never training ground truth.
The user confirmed review and correction of the current 154 frames; these were
approved and exported as `camera-v1` (112 train, 31 validation, 11 test).

In the app select **Workspace → Annotation review**:

1. Assign related clips to one source group and one train, validation, or test split.
2. Correct boxes on the canvas. Check missed heads; approve, exclude, or leave drafts.
3. Export a new dataset version. Approved empty frames become background negatives.

Review is stored in SQLite and survives reloads. Export rejects unassigned sources,
changed images, conflicting group splits, and exact duplicates across splits.
Additional frames can be extracted at specific timestamps. The video interval
panel records reviewed bare-head intervals and normalized head locations.
Evaluate only fully reviewed clips using `evaluate_events.py --source-hash HASH
--whole-frame-helmet-required`; this explicitly asserts that the whole frame requires helmets.
The report includes event recall, false alerts per hour, delay, FPS, and p95 latency.

```powershell
uv run finetune_camera.py --data artifacts/camera_review/versions/camera-v1/data.yaml --name camera-v1
```

Fine-tuning requires an approved export with all three splits. It combines reviewed
camera training images with eligible SHWD training images, keeping camera holdouts
frozen. It tries Small at 640, then 960 and Medium at 960 if validation recall remains
below target. Runs use up to 50 epochs, patience 10, seed 42, and batch 4 with memory
fallback. A replacement requires successful test evaluation, at least 80% bare-head
precision and 95% recall, and no recall regression against the prior model.

Camera fine-tuning is in progress. No camera model has been promoted yet. The test
split contains only 10 bare-head boxes, so results will have limited statistical
support. Event intervals still need recorded review before event metrics can run.
Existing SHWD results do not establish deployment-camera accuracy.

## Measured model results

| Checkpoint | Target | Test precision | Test recall | Confidence |
|---|---|---:|---:|---:|
| SHWD YOLO26 Small | bare head | 81.5% | 95.5% | 0.12 |
| Experimental PPE Medium | macro explicit violations | 81.6% | 79.7% | 0.17 |

SHWD uses `helmet` and `bare_head`; its original `person` annotation means a head,
not a full body. Source overlap limits claims about independent cameras. Higher
thresholds trade recall for precision: SHWD validation at 0.29 gave 90.05% precision
and 93.06% recall; 0.47 gave 95.01% precision and 88.44% recall.

## Development

```powershell
uv run pytest -q
uv run ruff check .
uv pip check
uv run verify_videos.py --profile shwd
uv export --no-dev --format requirements-txt --no-hashes --output-file requirements.txt
```

`app.py` owns the interface; `safety/` contains inference, tracking/events, media,
dataset conversion, and annotation review. Training and evaluation are standalone
CLIs. Tests live under `tests/`; reports, weights, and review data live under ignored
`artifacts/`. See [AGENTS.md](AGENTS.md), [TEST_REPORT.md](TEST_REPORT.md), and
[model results](docs/MODEL_RESULTS.md).

## Documentation

- [Windows installation and operation](docs/CCTV_INSTALL.md)
- [Release building](docs/RELEASE_BUILD.md) and [v0.2.0 release notes](docs/RELEASE_0.2.0.md)
- [Dataset choice](docs/DATASET_SELECTION.md), [preparation](docs/DATASET_PREPARATION.md),
  and [review findings](docs/DATASET_REVIEW.md)
- [Security](SECURITY.md), [component licenses](THIRD_PARTY_NOTICES.md), and [changelog](CHANGELOG.md)
- [Documentation synchronization record](docs/DOC_SYNC_2026-10-23.md)
