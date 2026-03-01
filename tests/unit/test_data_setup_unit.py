from pathlib import Path

import yaml

import data_setup


def test_find_folder_returns_none_when_root_missing(tmp_path):
    missing_root = tmp_path / "does-not-exist"
    assert data_setup.find_folder(missing_root, "train") is None


def test_find_folder_returns_matching_directory(tmp_path):
    target = tmp_path / "a" / "b" / "train"
    target.mkdir(parents=True)

    found = data_setup.find_folder(tmp_path, "train")

    assert found == target


def test_setup_data_writes_yaml_for_existing_dataset(tmp_path, monkeypatch):
    dataset_dir = tmp_path / "datasets" / "safety"
    (dataset_dir / "css-data" / "train" / "images").mkdir(parents=True)
    (dataset_dir / "css-data" / "train" / "labels").mkdir(parents=True)
    (dataset_dir / "css-data" / "valid" / "images").mkdir(parents=True)
    (dataset_dir / "css-data" / "valid" / "labels").mkdir(parents=True)
    (dataset_dir / "css-data" / "test" / "images").mkdir(parents=True)
    (dataset_dir / "css-data" / "test" / "labels").mkdir(parents=True)

    monkeypatch.setattr(data_setup, "DATASET_DIR", dataset_dir)

    data_setup.setup_data()

    yaml_path = dataset_dir / "data.yaml"
    assert yaml_path.exists()

    data = yaml.safe_load(yaml_path.read_text())
    assert data["path"] == "."
    assert data["train"] == "css-data/train/images"
    assert data["val"] == "css-data/valid/images"
    assert data["test"] == "css-data/test/images"
    assert data["nc"] == len(data_setup.CLASSES)
    assert data["names"] == data_setup.CLASSES


def test_setup_data_fails_when_val_missing(tmp_path, monkeypatch):
    dataset_dir = tmp_path / "datasets" / "safety"
    (dataset_dir / "css-data" / "train" / "images").mkdir(parents=True)
    (dataset_dir / "css-data" / "train" / "labels").mkdir(parents=True)
    (dataset_dir / "css-data" / "test" / "images").mkdir(parents=True)
    (dataset_dir / "css-data" / "test" / "labels").mkdir(parents=True)

    monkeypatch.setattr(data_setup, "DATASET_DIR", dataset_dir)

    data_setup.setup_data()

    assert not (dataset_dir / "data.yaml").exists()


def test_setup_data_fails_when_test_missing(tmp_path, monkeypatch):
    dataset_dir = tmp_path / "datasets" / "safety"
    (dataset_dir / "css-data" / "train" / "images").mkdir(parents=True)
    (dataset_dir / "css-data" / "train" / "labels").mkdir(parents=True)
    (dataset_dir / "css-data" / "valid" / "images").mkdir(parents=True)
    (dataset_dir / "css-data" / "valid" / "labels").mkdir(parents=True)

    monkeypatch.setattr(data_setup, "DATASET_DIR", dataset_dir)

    data_setup.setup_data()

    assert not (dataset_dir / "data.yaml").exists()
