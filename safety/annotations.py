from collections import Counter

import cv2
import numpy as np

from safety.profiles import PPE_CLASSES, PROFILES, checkpoint_profile

CLASSES = PPE_CLASSES
CLASSES_CONFIG = dict.fromkeys(CLASSES, "neutral")
CLASSES_CONFIG.update(dict.fromkeys(["Hardhat", "Mask", "Safety Vest"], "safe"))
CLASSES_CONFIG.update(dict.fromkeys(["NO-Hardhat", "NO-Mask", "NO-Safety Vest"], "danger"))
COLOR_MAP = {"safe": (0, 255, 0), "danger": (0, 0, 255), "neutral": (255, 0, 0)}
VIOLATION_IDS = (2, 3, 4)


def format_reasons(reasons):
    return ", ".join(f"{name} ({count})" for name, count in Counter(reasons).items())


def draw_safety_box(image, result, conf_threshold, is_custom):
    if image is None:
        raise ValueError("Input image cannot be None")
    if result is None:
        raise ValueError("Inference result cannot be None")
    canvas = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    violations, equipment = [], []
    profile = checkpoint_profile(result.names)
    statuses = CLASSES_CONFIG
    if profile:
        spec = PROFILES[profile]
        statuses = dict.fromkeys(spec["names"], "neutral")
        statuses.update({spec["names"][i]: "safe" for i in spec["equipment"]})
        statuses.update({spec["names"][i]: "danger" for i in spec["violations"]})
    if result.boxes is not None:
        for box in result.boxes:
            confidence = float(box.conf[0])
            if confidence < conf_threshold:
                continue
            class_id = int(box.cls[0])
            name = result.names.get(class_id, str(class_id))
            if not is_custom:
                if name.casefold() != "person":
                    continue
                name, status = "Person", "neutral"
            else:
                status = statuses.get(name, "neutral")
            if status == "danger":
                violations.append(name)
            elif status == "safe":
                equipment.append(name)
            color = COLOR_MAP[status]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            text = f"{name}: {confidence:.0%}"
            (width, height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            top = max(height + 5, y1)
            cv2.rectangle(canvas, (x1, top - height - 5), (x1 + width, top), color, -1)
            cv2.putText(canvas, text, (x1, top - 3), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), violations, equipment
