import importlib.util
from pathlib import Path

import numpy as np
import pytest


def api():
    path = Path(__file__).resolve().parents[1] / 'prepare_dataset.py'
    assert path.exists(), 'Dataset converter has not been implemented'
    spec = importlib.util.spec_from_file_location('dataset', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mask_boxes_preserve_separate_regions_and_edge_pixels():
    mask = np.zeros((100, 200), dtype=np.uint8)
    mask[10:30, 20:60] = 255
    mask[99, 199] = 255
    boxes = api().mask_to_boxes(mask)
    assert len(boxes) == 2
    assert boxes[0] == pytest.approx((0.2, 0.2, 0.2, 0.2))
    assert boxes[1] == pytest.approx((0.9975, 0.995, 0.005, 0.01))


def test_empty_mask_is_rejected_instead_of_labeling_defect_normal():
    with pytest.raises(ValueError, match='empty'):
        api().mask_to_boxes(np.zeros((10, 10), dtype=np.uint8))


def test_split_is_reproducible_disjoint_and_keeps_each_split():
    module = api()
    records = list(range(23))
    splits = module.split_records(records, seed=42)
    assert splits == module.split_records(records, seed=42)
    assert [len(splits[k]) for k in ('train', 'val', 'test')] == [13, 4, 6]
    assert sorted(sum(splits.values(), [])) == records
    assert not set(splits['train']) & set(splits['test'])


def test_split_rejects_too_small_stratum():
    with pytest.raises(ValueError):
        api().split_records([1, 2], seed=42)


def test_prepare_writes_negatives_boxes_and_manifest_without_leakage(tmp_path):
    import cv2
    module = api()
    source = tmp_path / '원본' / 'screw'
    for folder in ('train/good', 'test/good', 'test/scratch', 'ground_truth/scratch'):
        (source / folder).mkdir(parents=True)
    index = 0
    for folder in ('train/good', 'test/good', 'test/scratch'):
        for i in range(5):
            index += 1
            image = np.full((20, 20, 3), index, dtype=np.uint8)
            cv2.imencode('.png', image)[1].tofile(str(source / folder / f'{i:03}.png'))
            if folder == 'test/scratch':
                mask = np.zeros((20, 20), dtype=np.uint8)
                mask[2:6, 4:8] = 255
                cv2.imencode('.png', mask)[1].tofile(str(source / 'ground_truth/scratch' / f'{i:03}_mask.png'))
    output = tmp_path / '학습'
    manifest = module.prepare({'screw': source.parent}, output)
    assert len(manifest['records']) == 15
    assert len({r['sha256'] for r in manifest['records']}) == 15
    for record in manifest['records']:
        image = output / record['image']
        label = output / 'labels' / record['split'] / (image.stem + '.txt')
        assert image.is_file()
        values = label.read_text().split()
        if record['defect'] == 'good':
            assert values == []
        else:
            assert values[0] == '0'
            assert list(map(float, values[1:])) == pytest.approx([.3, .2, .2, .2])
    with pytest.raises(FileExistsError):
        module.prepare({'screw': source}, output)


def test_prepare_missing_mask_leaves_no_output(tmp_path):
    import cv2
    module = api()
    source = tmp_path / 'source'
    for folder in ('train', 'test/scratch', 'ground_truth'):
        (source / folder).mkdir(parents=True)
    for i in range(3):
        image = np.full((10, 10, 3), i, dtype=np.uint8)
        cv2.imencode('.png', image)[1].tofile(str(source / 'test/scratch' / f'{i}.png'))
    output = tmp_path / 'result'
    with pytest.raises(FileNotFoundError, match='Missing mask'):
        module.prepare({'screw': source}, output)
    assert not output.exists()
