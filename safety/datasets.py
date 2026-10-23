"""Download the official public datasets and convert their distinct label formats."""

import hashlib
import json
import math
import random
import shutil
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gdown
import yaml
from PIL import Image, ImageOps

from safety.profiles import PROFILES, dataset_root

SHWD_ID = "1qWm7rrwvjAWs1slymbrLaCf7Q-wnGLEX"


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def copy_training_image(source, target):
    """Repair missing JPEG end markers before recording the training-file hash."""
    with Image.open(source) as image:
        if image.format == "JPEG":
            with Path(source).open("rb") as handle:
                handle.seek(-2, 2)
                missing_end = handle.read() != b"\xff\xd9"
            if missing_end:
                ImageOps.exif_transpose(image).save(target, "JPEG", subsampling=0, quality=100)
                return
    shutil.copy2(source, target)


def download(profile):
    root = dataset_root(profile) / "raw"
    root.mkdir(parents=True, exist_ok=True)
    previous_path = root / "downloads.json"
    previous = json.loads(previous_path.read_text(encoding="utf-8")) if previous_path.exists() else []
    cached = {row["path"]: row for row in previous if "sha256" in row}
    entries = []
    if profile == "shwd":
        entries = [(SHWD_ID, root / "VOC2028.zip")]
    else:
        raise ValueError(f"Unsupported download profile: {profile}")

    def fetch(entry):
        file_id, path = entry
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            prior = cached.get(path.relative_to(root).as_posix())
            if prior and path.is_file():
                if sha256(path) != prior["sha256"]:
                    raise ValueError("Cached source changed since download; inspect it before replacing")
                return prior
            gdown.download(id=file_id, output=str(path), quiet=True, resume=True,
                           use_cookies=False, timeout=30, retries=3)
            if not path.is_file() or not path.stat().st_size:
                raise RuntimeError("Incomplete download")
        except Exception as exc:
            return {"id": file_id, "path": path.relative_to(root).as_posix(), "error": str(exc)}
        return {"id": file_id, "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size, "sha256": sha256(path),
                "url": f"https://drive.google.com/file/d/{file_id}/view"}

    manifest = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, item in enumerate(pool.map(fetch, entries), 1):
            manifest.append(item)
            if index % 25 == 0 or index == len(entries):
                failed = sum("error" in entry for entry in manifest)
                print(f"{profile}: checked {index}/{len(entries)} downloads; {failed} failed", flush=True)
    (root / "downloads.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    failures = [item["path"] for item in manifest if "error" in item]
    if failures:
        raise RuntimeError(f"Failed downloads: {len(failures)}; examples: {failures[:3]}. "
                           "Re-run to resume; inspect raw/downloads.json for the full list.")
    if profile == "shwd":
        extract_zip(root / "VOC2028.zip", root / "extracted")
    return root


def extract_zip(archive, destination):
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination) or (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"Unsafe ZIP member: {member.filename}")
        bad = source.testzip()
        if bad:
            raise ValueError(f"Corrupt ZIP member: {bad}")
        source.extractall(destination)


def normalized_box(cls, coordinates, size, corrections):
    """Convert pixel edges to normalized YOLO xywh; record any boundary clipping."""
    x1, y1, x2, y2 = map(float, coordinates)
    width, height = size
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid pixel box: {coordinates}")
    clipped = (max(0, x1), max(0, y1), min(width, x2), min(height, y2))
    if clipped != (x1, y1, x2, y2):
        corrections.append({"original": [x1, y1, x2, y2], "clipped": clipped})
    x1, y1, x2, y2 = clipped
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Box outside image: {coordinates}")
    return [cls, (x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
            (x2 - x1) / width, (y2 - y1) / height]


def voc_boxes(path, size, corrections):
    tree = ET.parse(path).getroot()
    mapping = {"hat": 0, "person": 1}
    boxes = []
    for obj in tree.findall("object"):
        name = obj.findtext("name")
        if name not in mapping:
            raise ValueError(f"Unknown SHWD class: {name}")
        box = obj.find("bndbox")
        coords = [float(box.findtext(key)) for key in ("xmin", "ymin", "xmax", "ymax")]
        # VOC uses one-based inclusive bounds; YOLO uses continuous image edges.
        coords[0] -= 1
        coords[1] -= 1
        boxes.append(normalized_box(mapping[name], coords, size, corrections))
    return boxes


def shwd_splits(voc):
    ids = sorted(path.stem for path in (voc / "Annotations").glob("*.xml"))
    split_dir = voc / "ImageSets" / "Main"

    def read(name):
        path = split_dir / f"{name}.txt"
        return path.read_text().split() if path.is_file() else None

    train, val, test, trainval = (read(name) for name in ("train", "val", "test", "trainval"))
    if train is not None and val is not None and test is not None:
        return {"train": train, "val": val, "test": test}, "official train/val/test"
    rng = random.Random(42)
    if trainval is not None and test is not None:
        pool = sorted(trainval)
        rng.shuffle(pool)
        count = max(1, round(len(pool) * .2))
        return {"train": pool[count:], "val": pool[:count], "test": test}, "official test; seed-42 trainval 80/20"
    rng.shuffle(ids)
    a, b = round(len(ids) * .8), round(len(ids) * .9)
    return {"train": ids[:a], "val": ids[a:b], "test": ids[b:]}, "seed-42 80/10/10"


def prepare(profile, raw=None):
    root = dataset_root(profile)
    raw = Path(raw or root / "raw")
    records, unmatched, policy = [], [], "SHWD splits"
    if profile == "shwd":
        candidates = list(raw.rglob("Annotations"))
        if len(candidates) != 1:
            raise ValueError("Expected exactly one extracted SHWD Annotations directory")
        voc = candidates[0].parent
        splits, policy = shwd_splits(voc)
        image_index = {path.stem: path for path in (voc / "JPEGImages").iterdir() if path.is_file()}
        for split, ids in splits.items():
            for image_id in ids:
                records.append((split, image_index[image_id], voc / "Annotations" / f"{image_id}.xml"))
    else:
        raise ValueError(f"Unsupported conversion profile: {profile}")

    records, excluded = filter_records(records, profile)
    target = root / "yolo"
    # Remove only outputs from this converter for explicitly excluded source records.
    for item in excluded:
        for kind, name in (("images", item["image"]), ("labels", Path(item["image"]).stem + ".txt")):
            path = (target / item["split"] / kind / name).resolve()
            if not path.is_relative_to(target.resolve()):
                raise ValueError("Generated output path escapes dataset")
            path.unlink(missing_ok=True)
    manifest, seen, hashes, clipped = [], set(), {}, []
    for split, image, annotation in records:
        if image.stem in seen:
            raise ValueError(f"Repeated image or cross-split filename: {image.name}")
        seen.add(image.stem)
        with Image.open(image) as source:
            source.load()
            size = source.size
        digest = sha256(image)
        if digest in hashes and hashes[digest] != split:
            raise ValueError(f"Exact image duplicate across splits: {image.name}")
        hashes[digest] = split
        corrections = []
        boxes = voc_boxes(annotation, size, corrections)
        for correction in corrections:
            clipped.append({"image": image.name, **correction})
        image_out = target / split / "images" / image.name
        label_out = target / split / "labels" / f"{image.stem}.txt"
        image_out.parent.mkdir(parents=True, exist_ok=True)
        label_out.parent.mkdir(parents=True, exist_ok=True)
        copy_training_image(image, image_out)
        label_out.write_text("".join(" ".join(map(str, box)) + "\n" for box in boxes), encoding="utf-8")
        manifest.append({"image": image_out.relative_to(root).as_posix(), "split": split,
                         "source": image.relative_to(raw).as_posix(), "source_sha256": digest,
                         "sha256": sha256(image_out),
                         "label_sha256": sha256(label_out), "objects": len(boxes)})
    for split in ("train", "val", "test"):
        expected = {row["image"] for row in manifest if row["split"] == split}
        actual = {path.relative_to(root).as_posix() for path in (target / split / "images").glob("*")}
        if actual != expected or not expected:
            raise ValueError(f"Empty or stale generated {split} split; inspect {target}")
    config = {"path": str(root.resolve()), "train": "yolo/train/images", "val": "yolo/val/images",
              "test": "yolo/test/images", "names": PROFILES[profile]["names"]}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (root / "conversion.json").write_text(json.dumps({"profile": profile, "seed": 42,
        "split_policy": policy, "images": len(records), "unmatched_images": unmatched,
        "boundary_clipping": clipped, "excluded_images": excluded}, indent=2), encoding="utf-8")
    config_path = root / "data.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def filter_records(records, profile):
    """Keep holdout membership; exclude lower-priority exact copies and unknown labels."""
    kept, excluded, hashes = [], [], {}
    for split, image, annotation in sorted(records, key=lambda row: {"test": 0, "val": 1, "train": 2}[row[0]]):
        if profile == "shwd":
            unknown = {obj.findtext("name") for obj in ET.parse(annotation).getroot().findall("object")} - {"hat", "person"}
            if unknown:
                excluded.append({"split": split, "image": image.name,
                                 "reason": "Unknown annotation classes; requires manual review", "classes": sorted(unknown)})
                continue
        digest = sha256(image)
        if digest in hashes and hashes[digest][0] != split:
            excluded.append({"split": split, "image": image.name,
                             "reason": "Exact duplicate in higher-priority holdout split",
                             "retained": hashes[digest][1], "sha256": digest})
            continue
        hashes[digest] = (split, image.name)
        kept.append((split, image, annotation))
    return kept, excluded
