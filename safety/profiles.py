"""The supported annotation contracts and their artifact locations."""

from safety.paths import ARTIFACTS, DATASET, ROOT

PPE_CLASSES = [
    "Hardhat", "Mask", "NO-Hardhat", "NO-Mask", "NO-Safety Vest", "Person",
    "Safety Cone", "Safety Vest", "machinery", "vehicle",
]
PROFILES = {
    "ppe": {"title": "Experimental construction PPE", "names": PPE_CLASSES, "targets": (2, 3, 4),
            "violations": (2, 3, 4), "equipment": (0, 1, 7)},
    "shwd": {"title": "Helmets and bare heads", "names": ["helmet", "bare_head"],
             "targets": (1,), "violations": (1,), "equipment": (0,)},
}


def dataset_root(profile):
    return DATASET if profile == "ppe" else ROOT / "datasets" / profile


def artifact_root(profile):
    return ARTIFACTS if profile == "ppe" else ARTIFACTS / profile


def checkpoint_profile(names):
    values = list(names.values()) if isinstance(names, dict) else list(names)
    return next((key for key, spec in PROFILES.items()
                 if len(values) == len(spec["names"]) and set(values) == set(spec["names"])), None)
