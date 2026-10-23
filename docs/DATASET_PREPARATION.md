# Dataset preparation

SHWD is the supported helmet dataset. Download and convert it with:

```powershell
uv run data_setup.py
uv run audit_dataset.py --profile shwd
```

The archive linked by the [authors](https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset)
contains 7,581 images. Conversion maps `hat` to `helmet` and source `person` to
`bare_head`; those annotations describe heads, not full bodies. Pascal VOC
one-based inclusive coordinates become continuous pixel edges before normalization.

The prepared set has 7,564 images: 5,441 train, 606 validation, and 1,517 test.
One image with unexpected classes is excluded completely; 16 lower-priority exact
cross-split duplicates are excluded. Original data stays intact. Download and
conversion manifests record source hashes, label hashes, exclusions, and clipping.

For supplied camera videos, run `uv run review_data.py extract`. Review the draft
annotations in the app before exporting a version. Assign related sources to one
group and split; exact duplicate images cannot cross splits. Unreviewed and excluded
frames are omitted. Approved empty frames are explicit background negatives.

The original ten-class dataset remains available with `--profile ppe`.
