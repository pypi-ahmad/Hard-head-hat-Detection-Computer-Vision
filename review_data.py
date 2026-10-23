"""Extract prediction drafts or export human-approved camera annotations."""

import argparse

from safety.inference import Detector, selected_checkpoint
from safety.paths import DATASET, ROOT
from safety.review import ReviewStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["extract", "export"])
    parser.add_argument("--destination")
    args = parser.parse_args()
    store = ReviewStore()
    if args.action == "export":
        if not args.destination:
            parser.error("export requires --destination (new version directory)")
        print(store.export(args.destination))
        return
    checkpoint, _ = selected_checkpoint("shwd")
    if checkpoint is None:
        raise ValueError("A trained SHWD checkpoint is required")
    detector = Detector(checkpoint)
    sources = [ROOT / "custom Youtube video file by Mujtaba1_Part1Trim.mp4"]
    sources += sorted((DATASET / "source_files/source_files").glob("*.mp4"))
    for source in sources:
        store.extract(source, detector)
        print(f"Prepared {source.name}", flush=True)
    print(f"{len(store.frames())} frames available for human review")


if __name__ == "__main__":
    main()
