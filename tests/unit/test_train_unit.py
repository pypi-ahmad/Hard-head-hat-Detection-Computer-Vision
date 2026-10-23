from types import SimpleNamespace

import pytest
import yaml

import train


@pytest.fixture
def dataset(tmp_path):
    for split in ('train', 'val', 'test'):
        folder = tmp_path / split
        folder.mkdir()
        (folder / 'example.jpg').write_bytes(b'image')
    path = tmp_path / 'data.yaml'
    path.write_text(yaml.safe_dump({**{split: split for split in ('train', 'val', 'test')},
                                   'names': train.PROFILES['ppe']['names']}))
    return path


def test_dataset_paths_are_independent_of_cwd(dataset, tmp_path, monkeypatch):
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert train.validate_dataset(dataset) == dataset


def test_ambiguous_root_rejected(dataset):
    data = yaml.safe_load(dataset.read_text())
    data['path'] = '.'
    dataset.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match='ambiguous'):
        train.validate_dataset(dataset)


def test_actual_incremented_checkpoint_is_reported(dataset, tmp_path, monkeypatch):
    checkpoint = tmp_path / 'run2/weights/best.pt'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'checkpoint')
    recorded = {}

    class Model:
        trainer = SimpleNamespace(best=checkpoint)

        def train(self, **kwargs):
            recorded.update(kwargs)

    monkeypatch.setattr(train, 'YOLO', lambda _: Model())
    monkeypatch.setattr(train.torch.cuda, 'is_available', lambda: False)
    result = train.train_safety_model(data=dataset, project=tmp_path, name='run')
    assert result['ok'] is True
    assert result['best_model'] == str(checkpoint)
    assert recorded['device'] == 'cpu'
    assert recorded['data'] == str(dataset)
    assert recorded['seed'] == 42
    assert (checkpoint.parent.parent / 'run_metadata.json').is_file()


@pytest.mark.parametrize('failure', ['load', 'train', 'checkpoint'])
def test_failures_are_not_success(dataset, tmp_path, monkeypatch, failure):
    class Model:
        trainer = SimpleNamespace(best=tmp_path / 'missing.pt')

        def __init__(self, _):
            if failure == 'load':
                raise RuntimeError('corrupted')

        def train(self, **kwargs):
            if failure == 'train':
                raise RuntimeError('out of memory')

    monkeypatch.setattr(train, 'YOLO', Model)
    result = train.train_safety_model(data=dataset, device='cpu')
    assert result['ok'] is False
    assert result['error'] == ('model_load_failed' if failure == 'load' else 'train_failed')


def test_missing_dataset(tmp_path):
    result = train.train_safety_model(data=tmp_path / 'missing.yaml')
    assert result['ok'] is False
    assert result['error'] == 'missing_data_yaml'


@pytest.mark.parametrize('contents', ['', '[unclosed'])
def test_malformed_yaml_returns_failure(tmp_path, contents):
    path = tmp_path / 'data.yaml'
    path.write_text(contents)
    result = train.train_safety_model(data=path)
    assert result['ok'] is False
    assert result['error'] == 'invalid_dataset'
