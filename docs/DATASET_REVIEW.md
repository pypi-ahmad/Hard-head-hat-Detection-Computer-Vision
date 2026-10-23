# Dataset review

SHWD preparation excluded one image containing unknown `dog` labels and 16 exact
cross-split copies. The prepared dataset has 7,564 images. The previous audit found
no malformed labels, orphan labels, or exact cross-split duplicates after exclusions.

Perceptual matching produced 31,412 cross-split candidates. Six inspected pairs were
repeated photographs with watermark or border differences. These are evidence of
source overlap; the full candidate set has not been manually adjudicated.

The supplied four videos contain 8,179 frames. Sampling every two seconds prepared
154 images. The user confirmed personal review and correction of all drafts;
the stored annotations were approved and exported as `camera-v1`. The original
database was backed up and the confirmation recorded alongside it.

Whole-video splits: custom clip and JapanPPE train (112 frames), hardhat validation
(31), and indianworkers test (11). The test split has 10 bare-head boxes. These
four clips do not establish independence across deployment sites.

All 7,564 SHWD image/label records were verified during camera preparation.
Ultralytics had repaired 64 JPEG files across the splits; each was reproduced
byte-for-byte from the unchanged original. The manifest now records original and
prepared image hashes separately. Preparation repairs JPEG end markers before
hashing, preventing training from silently invalidating these records.

Frame approval covers the stored image annotations only. It does not approve
video event intervals or establish independence between recording sites.

Camera test accuracy, event recall, false alerts per hour, and alert delay remain
unmeasured while fine-tuning runs and event intervals await recorded review.
