"""Audit the existing dataset without changing images, labels, or splits."""

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageOps

from safety.annotations import CLASSES, VIOLATION_IDS
from safety.profiles import PROFILES, artifact_root, dataset_root


def inspect_label(path, class_count=len(CLASSES)):
    boxes, errors = [], []
    if not path.exists():
        return boxes, ["missing label"]
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            values = np.array([float(value) for value in line.split()])
            if len(values) != 5 or not np.isfinite(values).all():
                raise ValueError("expected five finite numbers")
            cls, x, y, width, height = values
            if cls != int(cls) or not 0 <= cls < class_count:
                raise ValueError("invalid class ID")
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
                raise ValueError("invalid normalized coordinates or dimensions")
            boxes.append([int(cls), x, y, width, height])
            if min(x - width / 2, y - height / 2) < -0.001 or max(x + width / 2, y + height / 2) > 1.001:
                errors.append(f"line {line_no}: box extends beyond image")
        except ValueError as exc:
            errors.append(f"line {line_no}: {exc}")
    return boxes, errors


def image_hash(path):
    with Image.open(path) as source:
        source.load()
        image = source.convert("RGB")
        gray = np.asarray(image.resize((9, 8)).convert("L"))
        bits = gray[:, 1:] > gray[:, :-1]
        fingerprint = int.from_bytes(np.packbits(bits).tobytes(), "big")
        return image.size, fingerprint


def contact_sheets(records, output, title, names=CLASSES, violations=VIOLATION_IDS):
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for start in range(0, len(records), 12):
        sheet = Image.new("RGB", (1280, 960), "white")
        for index, record in enumerate(records[start:start + 12]):
            with Image.open(record["absolute"]) as source:
                picture = source.convert("RGB")
            draw = ImageDraw.Draw(picture)
            w, h = picture.size
            for cls, cx, cy, bw, bh in record["boxes"]:
                rect = ((cx - bw / 2) * w, (cy - bh / 2) * h,
                        (cx + bw / 2) * w, (cy + bh / 2) * h)
                color = "red" if cls in violations else "lime"
                draw.rectangle(rect, outline=color, width=2)
                draw.text((max(0, rect[0]), max(0, rect[1])), names[cls], fill=color)
            picture = ImageOps.contain(picture, (320, 280))
            x, y = (index % 4) * 320, (index // 4) * 320
            sheet.paste(picture, (x, y))
            ImageDraw.Draw(sheet).text((x + 3, y + 282),
                                      record["path"][:48] + "\n" + str(start + index), fill="black")
        path = output / f"{title}_{start // 12 + 1}.jpg"
        sheet.save(path)
        paths.append(str(path))
    return paths


def audit(root=None, output=None, profile="ppe"):
    root = Path(root or dataset_root(profile)).resolve()
    output = Path(output or artifact_root(profile) / "audit").resolve()
    spec = PROFILES[profile]
    names = spec["names"]
    config = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))
    data_root = Path(config.get("path", root))
    output.mkdir(parents=True, exist_ok=True)
    records, issues, split_counts = [], [], {}
    for split in ("train", "valid", "test"):
        split_dir = (data_root / config["val" if split == "valid" else split]).parent
        images = sorted((split_dir / "images").glob("*"))
        counts = Counter()
        for path in images:
            label = split_dir / "labels" / (path.stem + ".txt")
            boxes, errors = inspect_label(label, len(names))
            try:
                size, fingerprint = image_hash(path)
            except (OSError, ValueError) as exc:
                issues.append({"path": str(path.relative_to(root)), "errors": [str(exc)]})
                continue
            if errors:
                issues.append({"path": str(path.relative_to(root)), "errors": errors})
            counts.update(names[box[0]] for box in boxes)
            records.append({"path": path.relative_to(root).as_posix(), "absolute": str(path),
                            "split": split, "size": size, "boxes": boxes,
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "label_sha256": hashlib.sha256(label.read_bytes()).hexdigest()
                            if label.exists() else None,
                            "dhash": fingerprint, "group": re.sub(r"\.rf\..*$", "", path.stem)})
        orphaned = [p.name for p in (split_dir / "labels").glob("*.txt")
                    if not any((split_dir / "images" / (p.stem + suffix)).exists()
                               for suffix in (".jpg", ".jpeg", ".png"))]
        split_counts[split] = {"images": len(images), "classes": dict(counts),
                               "orphaned_labels": orphaned}
    shared, exact = [], []
    for field, destination in (("group", shared), ("sha256", exact)):
        groups = defaultdict(list)
        for record in records:
            groups[record[field]].append(record)
        destination.extend([r["path"] for r in group] for group in groups.values()
                           if len({r["split"] for r in group}) > 1)
    # Supported profiles have fewer than 8,000 images; use indexing for larger corpora.
    similar, similar_count = [], 0
    for i, a in enumerate(records):
        for b in records[i + 1:]:
            if a["split"] != b["split"] and (a["dhash"] ^ b["dhash"]).bit_count() <= 5:
                similar_count += 1
                if len(similar) < 2000:
                    similar.append([a["path"], b["path"]])
    manifest = [{k: v for k, v in record.items() if k not in ("absolute", "boxes", "dhash")}
                for record in records]
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    empty = [r for r in records if not r["boxes"]]
    paths = contact_sheets(empty, output / "review", "empty", names, spec["violations"])
    overlap_paths = {p for group in shared for p in group}
    paths += contact_sheets([r for r in records if r["path"] in overlap_paths],
                            output / "review", "overlap", names, spec["violations"])
    rng = random.Random(42)
    for cls in spec["targets"]:
        selected = []
        for split, count in (("train", 8), ("valid", 6), ("test", 6)):
            pool = [r for r in records if r["split"] == split and any(b[0] == cls for b in r["boxes"])]
            selected.extend(rng.sample(pool, min(count, len(pool))))
        paths += contact_sheets(selected, output / "review", f"class_{cls}", names, spec["violations"])
    report = {"splits": split_counts, "decoded_images": len(records), "issues": issues,
              "empty_labels": len(empty), "shared_prefix_groups": shared,
              "exact_cross_split_duplicates": exact, "perceptual_candidates": similar,
              "perceptual_candidate_count": similar_count, "contact_sheets": paths,
              "limitations": "Filename and perceptual matches are review candidates, not proven provenance. Original splits and files unchanged."}
    (output / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = (f"# Dataset audit\n\nDecoded images: {len(records)}. Empty labels: {len(empty)}.\n\n"
               f"Images with issues: {len(issues)}. Shared filename groups: {len(shared)}. "
               f"Exact cross-split duplicate groups: {len(exact)}. "
               f"Perceptual candidates: {similar_count}.\n\n"
               "See audit.json for paths, class counts, and findings; review/ contains annotated contact sheets. "
               "The manifest hashes original images and labels. Similarity candidates require visual review.\n")
    (output / "AUDIT.md").write_text(summary, encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profile", choices=list(PROFILES), default="shwd")
    report = audit(**vars(parser.parse_args()))
    print(json.dumps({key: report[key] for key in ("decoded_images", "empty_labels", "perceptual_candidate_count")}))
