import io
import json
import time
import traceback
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

import app
import data_setup
import train


class FakeBox:
    def __init__(self, conf, cls_id, xyxy):
        self.conf = np.array([conf], dtype=float)
        self.cls = np.array([cls_id], dtype=float)
        self.xyxy = np.array([xyxy], dtype=float)


class FakeResult:
    def __init__(self, boxes, names):
        self.boxes = boxes
        self.names = names


class FakeYOLO:
    def __init__(self, model_path):
        self.model_path = model_path

    def __call__(self, image):
        box = FakeBox(0.95, 0, [1, 1, 20, 20])
        return [FakeResult([box], {0: "Hardhat"})]


class BrokenYOLO:
    def __init__(self, _):
        raise RuntimeError("Corrupted weights file")


class FakeSidebar:
    def __init__(self):
        self.errors = []

    def error(self, message):
        self.errors.append(message)


class FakeStreamlit:
    def __init__(self):
        self.sidebar = FakeSidebar()


class FakeCuda:
    def __init__(self, available):
        self.available = available

    def is_available(self):
        return self.available

    def get_device_name(self, _):
        return "Fake GPU"


class FakeTorch:
    def __init__(self, available):
        self.cuda = FakeCuda(available)


class RecordingYOLO:
    def __init__(self, model_name):
        self.model_name = model_name
        self.calls = []

    def train(self, **kwargs):
        self.calls.append(kwargs)
        return {"ok": True}


def run_case(results, category, scenario, fn):
    start = time.perf_counter()
    try:
        payload = fn()
        duration_ms = int((time.perf_counter() - start) * 1000)
        results.append(
            {
                "category": category,
                "scenario": scenario,
                "status": "pass",
                "duration_ms": duration_ms,
                "details": payload,
                "stack_trace": None,
            }
        )
    except Exception as exc:
        duration_ms = int((time.perf_counter() - start) * 1000)
        results.append(
            {
                "category": category,
                "scenario": scenario,
                "status": "fail",
                "duration_ms": duration_ms,
                "details": f"{type(exc).__name__}: {exc}",
                "stack_trace": traceback.format_exc(),
            }
        )


def main():
    results = []

    original_yolo = app.YOLO
    original_st = app.st
    original_train_yolo = train.YOLO
    original_torch = train.torch
    original_dataset_dir = data_setup.DATASET_DIR

    try:
        # SYSTEM STRESS: Large images
        run_case(
            results,
            "system",
            "large_image_single_pass_8k",
            lambda: _large_image_case(),
        )

        # SYSTEM STRESS: Repeated inference calls
        run_case(
            results,
            "system",
            "repeated_inference_2000_calls",
            lambda: _repeated_inference_case(),
        )

        # SYSTEM STRESS: Rapid UI interactions (simulated reruns)
        run_case(
            results,
            "system",
            "rapid_ui_interactions_1000_model_loads",
            lambda: _rapid_ui_case(),
        )

        # SYSTEM STRESS: Large video handling path from app logic
        run_case(
            results,
            "system",
            "large_video_tempfile_lock_windows_path",
            lambda: _video_tempfile_lock_case(),
        )

        run_case(
            results,
            "system",
            "video_tempfile_lifecycle_create_use_cleanup_50",
            lambda: _video_tempfile_leak_case(),
        )

        # SYSTEM STRESS: CSV uploaded to image path (invalid schema/type)
        run_case(
            results,
            "system",
            "csv_as_image_upload_decode",
            lambda: _csv_as_image_case(),
        )

        # ML STRESS: Batch processing
        run_case(
            results,
            "ml",
            "batch_processing_128_frames",
            lambda: _ml_batch_case(),
        )

        # ML STRESS: GPU/CPU switching
        run_case(
            results,
            "ml",
            "gpu_cpu_switch_train_device_selection",
            lambda: _gpu_cpu_switch_case(),
        )

        # ML STRESS: Missing model file
        run_case(
            results,
            "ml",
            "missing_custom_model_file",
            lambda: _missing_model_case(),
        )

        # ML STRESS: Corrupted model weights
        run_case(
            results,
            "ml",
            "corrupted_weights_model_load",
            lambda: _corrupted_weights_case(),
        )

        # DATA STRESS: Null values
        run_case(
            results,
            "data",
            "null_input_to_draw_safety_box",
            lambda: _null_value_case(),
        )

        # DATA STRESS: Wrong schema (missing train/images)
        run_case(
            results,
            "data",
            "wrong_schema_missing_train_images",
            lambda: _wrong_schema_case(),
        )

        # DATA STRESS: Large datasets (many folders)
        run_case(
            results,
            "data",
            "large_dataset_many_dirs_find_folder",
            lambda: _large_dataset_case(),
        )

    finally:
        app.YOLO = original_yolo
        app.st = original_st
        train.YOLO = original_train_yolo
        train.torch = original_torch
        data_setup.DATASET_DIR = original_dataset_dir

    out_path = Path("tests/stress/phase4_stress_results.json")
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


def _large_image_case():
    image = Image.fromarray(np.zeros((8000, 8000, 3), dtype=np.uint8))
    box = FakeBox(0.99, 0, [100, 100, 1200, 1200])
    result = FakeResult([box], {0: "Hardhat"})
    annotated, violations, safe = app.draw_safety_box(image, result, 0.4, True)
    return {
        "annotated_shape": list(annotated.shape),
        "violations": len(violations),
        "safe": len(safe),
    }


def _repeated_inference_case():
    image = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))
    box = FakeBox(0.95, 1, [10, 10, 100, 100])
    result = FakeResult([box], {1: "NO-Mask"})
    total_v = 0
    total_s = 0
    for _ in range(2000):
        _, v, s = app.draw_safety_box(image, result, 0.4, True)
        total_v += len(v)
        total_s += len(s)
    return {"total_violations": total_v, "total_safe": total_s}


def _rapid_ui_case():
    app.YOLO = FakeYOLO
    fake_st = FakeStreamlit()
    app.st = fake_st

    choices = ["YOLO26 Nano", "YOLO26 Large", "Custom Trained Model"]
    custom_path = "missing-model.pt"
    loaded = 0
    for i in range(1000):
        model_name = choices[i % len(choices)]
        try:
            app.load_model(model_name, custom_path)
        except Exception:
            pass
        loaded += 1
    return {"loads_attempted": loaded, "sidebar_errors": len(fake_st.sidebar.errors)}


def _video_tempfile_lock_case():
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    try:
        temp_file.write(b"0" * (20 * 1024 * 1024))
        video = cv2.VideoCapture(temp_file.name)
        opened = bool(video.isOpened())
        video.release()
        return {
            "tempfile_opened_by_cv2": opened,
            "temp_path": temp_file.name,
            "file_closed_before_videocapture": False,
        }
    finally:
        temp_file.close()


def _video_tempfile_leak_case():
    created = []
    try:
        for _ in range(50):
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.write(b"0" * (1024 * 1024))
            path = temp_file.name
            video = cv2.VideoCapture(path)
            _ = bool(video.isOpened())
            video.release()
            temp_file.close()
            created.append(path)
        existing_before_cleanup = sum(1 for p in created if Path(p).exists())
        return {
            "created_files": len(created),
            "existing_before_cleanup": existing_before_cleanup,
        }
    finally:
        for p in created:
            try:
                Path(p).unlink(missing_ok=True)
            except Exception:
                pass


def _csv_as_image_case():
    csv_bytes = b"col1,col2\n1,2\n3,4\n"
    bio = io.BytesIO(csv_bytes)
    try:
        Image.open(bio)
        return {"decoded": True}
    except UnidentifiedImageError as exc:
        return {"handled_invalid_image": True, "error": str(exc)}


def _ml_batch_case():
    app.YOLO = FakeYOLO
    model = app.YOLO("yolov8n.pt")
    frames = [Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8)) for _ in range(128)]
    outputs = []
    for frame in frames:
        result = model(frame)[0]
        annotated, violations, safe = app.draw_safety_box(frame, result, 0.4, True)
        outputs.append((annotated.shape, len(violations), len(safe)))
    return {"batch_size": len(outputs), "sample_output": outputs[0]}


def _gpu_cpu_switch_case():
    records = []

    def yolo_factory(model_name):
        obj = RecordingYOLO(model_name)
        records.append(obj)
        return obj

    train.YOLO = yolo_factory

    train.torch = FakeTorch(available=False)
    train.train_safety_model()
    cpu_device = records[-1].calls[-1]["device"]

    train.torch = FakeTorch(available=True)
    train.train_safety_model()
    gpu_device = records[-1].calls[-1]["device"]

    return {"cpu_run_device": cpu_device, "gpu_run_device": gpu_device}


def _missing_model_case():
    app.YOLO = FakeYOLO
    fake_st = FakeStreamlit()
    app.st = fake_st
    model, is_custom = app.load_model("Custom Trained Model", "no_such_model.pt")
    return {
        "is_custom": is_custom,
        "fallback_model": model.model_path,
        "sidebar_errors": len(fake_st.sidebar.errors),
    }


def _corrupted_weights_case():
    app.YOLO = BrokenYOLO
    app.st = FakeStreamlit()
    model, is_custom = app.load_model("YOLO26 Nano", None)
    return {
        "model_is_none": model is None,
        "is_custom": is_custom,
        "sidebar_errors": len(app.st.sidebar.errors),
    }


def _null_value_case():
    result = FakeResult([], {})
    try:
        app.draw_safety_box(None, result, 0.4, True)
        return {"handled": False}
    except ValueError as exc:
        return {"handled": True, "error": str(exc)}


def _wrong_schema_case():
    with tempfile.TemporaryDirectory() as td:
        dataset_dir = Path(td) / "datasets" / "safety"
        (dataset_dir / "css-data" / "train").mkdir(parents=True)
        (dataset_dir / "css-data" / "valid" / "images").mkdir(parents=True)
        data_setup.DATASET_DIR = dataset_dir
        data_setup.setup_data()
        yaml_path = dataset_dir / "data.yaml"
        return {"yaml_created": yaml_path.exists(), "dataset_dir": str(dataset_dir)}


def _large_dataset_case():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "datasets" / "safety"
        root.mkdir(parents=True)
        for i in range(300):
            (root / f"sub_{i}" / "nested" / "x" / "y").mkdir(parents=True)
        target = root / "bulk" / "train"
        target.mkdir(parents=True)
        found = data_setup.find_folder(root, "train")
        return {"found": str(found) if found else None, "expected_suffix": "bulk\\train"}


if __name__ == "__main__":
    main()
