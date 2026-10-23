# Repository Guidelines

## Project structure

`app.py` contains the Streamlit interface. Shared modules live in `safety/`: inference
and locking, annotations, media workers, paths, dataset profiles, and conversion.

Use `data_setup.py` to prepare data, `audit_dataset.py` to inspect it, `train.py` for
individual runs, `run_experiments.py` for comparisons, `evaluate.py` for metrics,
and `verify_videos.py` for full-video checks. Tests are grouped under `tests/unit/`,
`tests/integration/`, `tests/ml/`, and `tests/stress/`. Historical notebooks live in
`notebooks/`. Generated checkpoints, reports, and videos belong in ignored `artifacts/`.

## Development commands

Use Python 3.13 and uv:

```powershell
uv sync --locked
uv run streamlit run app.py
uv run pytest -q
uv run ruff check .
uv pip check
```

These install locked dependencies, launch the app, run tests, lint source, and check
installed dependency compatibility. The lockfile selects CUDA 13.2 PyTorch wheels.
GPU execution needs a compatible NVIDIA driver; inference and individual training
also support CPU.

GPU experiments take substantial time. Run them sequentially when training is part
of the task. Check existing jobs before starting another run.

## Coding conventions

Use four-space indentation, `snake_case` functions/modules, and `PascalCase` classes.
Follow configured Ruff rules and target 100-character lines. Resolve paths through
`safety.paths`. Preserve prediction locking, confidence forwarding, and media cleanup.
Keep changes focused and preserve unrelated work.

## Dataset contracts

SHWD is the application and CLI default; it uses `helmet` and `bare_head`. Keep the
experimental ten-class `ppe` profile separate. SHWD's source `person`
means a bare head. Missing equipment detections do not establish violations.

Preserve original data and splits unless a dataset revision is requested. Keep
credentials outside project files. Update `uv.lock` with dependencies and regenerate
`requirements.txt` using the README command.

## Tests and documentation

Name pytest files `test_*.py` and tests `test_*`. Add meaningful regression coverage
for changed behavior. No numerical coverage threshold is configured. Mocked checks
establish software behavior; accuracy claims need real evaluation. Record results
and remaining limits in [TEST_REPORT.md](TEST_REPORT.md). Consult
[model results](docs/MODEL_RESULTS.md) before describing model quality.

## Commits and pull requests

History mixes prefixes such as `docs:`, `chore:`, and `restore:` with plain summaries.
Use concise imperative subjects. PRs should explain behavior, link relevant issues,
report checks, and include screenshots for interface changes. Model changes need
dataset provenance and measured metrics.

## Camera review and events

`review_data.py extract` prepares unreviewed drafts; `safety/review.py` owns review
and export. Never train on unreviewed labels. Keep source groups within one split.
`finetune_camera.py` evaluates reviewed holdouts before checkpoint publication.
Each stream owns ByteTrack and event state; the cached detector owns inference only.

## CCTV distribution

`cctv.py` owns continuous workers and supervises the local dashboard and optional
HTTPS gateway. `safety/cctv.py` owns capture, latest-frame inference, status, and
retention. `cctv_app.py` reads runtime artifacts; browsers do not own camera workers.
Use `uv run python cctv.py --check` and `--probe` for installation checks. Private
camera URLs are environment references in ignored `config/*.local.yaml`. Never
publish camera credentials, recordings, PKI keys, or local gateway configuration.
Follow [release building](docs/RELEASE_BUILD.md): the builder requires a selected
SHWD checkpoint and the pinned Caddy ZIP. Build with
`uv run python -m scripts.build_release`; include the
selected model, dependency notices, and SHA256 checksums. Keep actual-camera
qualification separate from software installation and public-dataset metrics.
