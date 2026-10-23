# Dataset choice

Use SHWD for helmet and bare-head detection. Its annotation contract matches the
application directly. Use the supplied video frames to adapt to local blur, head
size, viewpoints, and lighting after manual annotation.

The original ten-class PPE dataset remains an experiment. Do not merge its class
IDs or use absence of equipment detections as evidence of a violation.

Additional public data should be added only when reviewed camera errors expose a
missing condition. Verify reuse terms, provenance, and annotation completeness;
stage additions as unreviewed and keep related sources within a single split.

The current Windows release bundles the selected SHWD YOLO26 Small checkpoint.
Camera-v1 contains 154 user-approved frames split by whole video; adaptation is
still running and has not replaced that checkpoint. Its 11-frame test split has
only 10 bare-head boxes. More independent site footage is needed before making
strong deployment claims. See [model results](MODEL_RESULTS.md).

See [preparation](DATASET_PREPARATION.md) and [review findings](DATASET_REVIEW.md).
