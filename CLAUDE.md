# Claude repository instructions

Read [AGENTS.md](AGENTS.md) for shared contributor guidance, commands, coding
conventions, and dataset contracts. Apply it when working in this repository.

## Before making changes

Inspect the relevant source and existing edits. Use Python 3.13 through uv and keep
changes within the requested task. Check active training jobs before starting GPU
work; run experiments sequentially.

## Inference and data

Preserve model locks, media cleanup, and confidence forwarding. Use SHWD by default
and keep experimental PPE checkpoints separate. Never export unreviewed camera
annotations for training. Preserve original annotations and splits unless the task
requests a dataset revision. Keep credentials out of files.

CCTV workers run in `cctv.py`, independently of Streamlit browser sessions. Keep
Streamlit bound to localhost; LAN access goes through the authenticated Caddy TLS
gateway. Preserve bounded latest-frame capture, reconnection, stale indicators, and
event retention. Local camera configs, gateway hashes, PKI, and runtime footage
must stay outside Git and release packages. See [CCTV_INSTALL.md](docs/CCTV_INSTALL.md).

## Verification and reporting

Run checks that cover the changed behavior. Distinguish mocked tests, real runtime
checks, and accuracy evaluation. Report observed results and unresolved limits.
Consult [README.md](README.md) for usage,
[TEST_REPORT.md](TEST_REPORT.md) for verification, and
[model results](docs/MODEL_RESULTS.md) for measured performance. Camera accuracy requires reviewed camera holdouts.
Use [release building](docs/RELEASE_BUILD.md) for distribution changes; preserve
published archives and keep release-snapshot claims separate from branch updates.
