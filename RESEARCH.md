# Helmet detection research

The chosen baseline is SHWD with `helmet` and `bare_head` classes. The original
`person` label represents a bare head. YOLO26 offers n, s, m, l, and x scales;
the current local SHWD checkpoint uses Small.

Relevant primary sources:

- [SHWD repository](https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset)
- [YOLO26 documentation](https://docs.ultralytics.com/models/yolo26/)
- [Tracking documentation](https://docs.ultralytics.com/modes/track/)

The local implementation uses the installed Ultralytics ByteTrack API with an
independent tracker per stream. Public dataset metrics do not establish accuracy
on deployment cameras. Review local video annotations, group related sources,
and evaluate frozen holdouts before replacing the checkpoint.

See [model results](docs/MODEL_RESULTS.md) for observed values and
[dataset review](docs/DATASET_REVIEW.md) for overlap limitations.
