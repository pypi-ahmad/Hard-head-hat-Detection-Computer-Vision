# Model results

## Release checkpoint

The v0.2.0 Windows app bundles the selected SHWD YOLO26 Small model at 640-pixel
inference resolution and confidence 0.12. The package places it at
`artifacts/shwd/model.pt`; the development selection points to
`artifacts/shwd/runs/yolo26s/weights/best.pt`. These are the existing public-dataset
results below, not completed camera-adaptation results.

## Existing checkpoints

| Profile | Checkpoint | Target | Test precision | Test recall | Confidence |
|---|---|---|---:|---:|---:|
| SHWD | YOLO26 Small, 50 epochs | bare head | 81.4783% | 95.5360% | 0.12 |
| Experimental PPE | YOLO26 Medium, 50 epochs | macro violations | 81.6103% | 79.6973% | 0.17 |

SHWD test bare heads: 22,558 targets, 26,450 predictions, 21,551 true positives.
Helmet precision was 82.4553% and recall 93.3440%. These are image detection metrics,
not independent camera or event metrics. Source overlap limits generalization claims.

SHWD validation operating points:

| Confidence | Bare-head precision | Bare-head recall |
|---|---:|---:|
| 0.12 | 80.23% | 95.51% |
| 0.29 | 90.05% | 93.06% |
| 0.47 | 95.01% | 88.44% |

The app retains 0.12 to minimize missed bare heads. The higher thresholds have not
been promoted or tested as deployment defaults.

## Camera adaptation

`finetune_camera.py` requires approved annotations and frozen camera holdouts.
Initial acceptance is 95% bare-head recall at 80% precision. Report event recall,
false alerts per camera-hour, delay, hardware, resolution, and latency separately.
The user-approved `camera-v1` export contains 112 training, 31 validation, and
11 test frames, split by source video. Fine-tuning is running; no completed camera
accuracy result or replacement checkpoint is available yet. The test set has only
10 bare-head boxes. Event intervals have not been marked reviewed in the database.
