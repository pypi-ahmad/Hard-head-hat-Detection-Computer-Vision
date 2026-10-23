# SHWD overhaul: implemented design

The application defaults to SHWD helmet/bare-head detection. The historical PPE
workflow is explicitly experimental. The implementation adds per-stream ByteTrack,
helmet-required areas, timestamp-based event confirmation, local event exports,
and human annotation review.

Review drafts must not become training ground truth automatically. Related sources
must share a split. Fine-tuning compares candidates against the existing checkpoint
on frozen camera holdouts before publishing a replacement.

Windows CCTV distribution is implemented in v0.2.0: continuous workers, a separate
read-only dashboard, and authenticated HTTPS LAN access. Follow the
[installation guide](CCTV_INSTALL.md) for operation. Camera fine-tuning and reviewed
event evaluation remain separate qualification work; implementation does not
establish accuracy on a new site.

Primary references used for the design:

- [SHWD authors](https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset)
- [Ultralytics tracking](https://docs.ultralytics.com/modes/track/)
- [YOLO26](https://docs.ultralytics.com/models/yolo26/)
- [YOLO detection labels](https://docs.ultralytics.com/datasets/detect/)
- [Grouped validation](https://scikit-learn.org/stable/modules/cross_validation.html)

See [README](../README.md) for the implemented workflow and [TEST_REPORT](../TEST_REPORT.md)
for observed checks and remaining limits. Human review and camera fine-tuning remain
separate from software implementation.
