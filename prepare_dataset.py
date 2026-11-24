"""MVTec AD masks -> supervised single-class YOLO detection dataset."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent


def read_image(path, flags=cv2.IMREAD_COLOR):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flags)
    if image is None:
        raise ValueError(f'Cannot decode image: {path}')
    return image


def mask_to_boxes(mask):
    if mask.ndim != 2 or not np.any(mask):
        raise ValueError('Invalid or empty defect mask')
    height, width = mask.shape
    count, _, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    return [((x + w / 2) / width, (y + h / 2) / height, w / width, h / height)
            for x, y, w, h, area in stats[1:count]]


def split_records(records, seed=42):
    if len(records) < 3:
        raise ValueError('Each source stratum needs at least 3 images')
    records = list(records)
    random.Random(seed).shuffle(records)
    ntrain = max(1, int(len(records) * .6))
    nval = max(1, int(len(records) * .2))
    return {'train': records[:ntrain], 'val': records[ntrain:ntrain+nval],
            'test': records[ntrain+nval:]}


def resolve_source(path, category):
    path = Path(path).expanduser().resolve()
    for candidate in (path, path / category):
        if all((candidate / name).is_dir() for name in ('train', 'test', 'ground_truth')):
            return candidate
    raise FileNotFoundError(f'MVTec folders not found under {path}')


def prepare(sources, output, seed=42):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Output exists; use a new --output directory: {output}')
    records = []
    seen = {}
    # Validate everything before creating output. Keep original source split in manifest.
    for category, source in sources.items():
        source = resolve_source(source, category)
        for source_split in ('train', 'test'):
            for defect_dir in sorted((source / source_split).iterdir()):
                if not defect_dir.is_dir():
                    continue
                paths = sorted(defect_dir.glob('*.png'))
                if not paths:
                    raise ValueError(f'No PNG images in {defect_dir}')
                for split, selected in split_records(paths, seed).items():
                    for path in selected:
                        digest = hashlib.sha256(path.read_bytes()).hexdigest()
                        if digest in seen:
                            raise ValueError(f'Duplicate image: {path} and {seen[digest]}')
                        seen[digest] = str(path)
                        image = read_image(path)
                        boxes = []
                        if defect_dir.name != 'good':
                            mask_path = source / 'ground_truth' / defect_dir.name / f'{path.stem}_mask.png'
                            if not mask_path.is_file():
                                raise FileNotFoundError(f'Missing mask: {mask_path}')
                            mask = read_image(mask_path, cv2.IMREAD_GRAYSCALE)
                            if mask.shape != image.shape[:2]:
                                raise ValueError(f'Mask size mismatch: {mask_path}')
                            boxes = mask_to_boxes(mask)
                        records.append({'source': str(path), 'source_split': source_split,
                                        'category': category, 'defect': defect_dir.name,
                                        'split': split, 'sha256': digest, 'boxes': boxes})
    for split in ('train', 'val', 'test'):
        (output / 'images' / split).mkdir(parents=True)
        (output / 'labels' / split).mkdir(parents=True)
    for record in records:
        source = Path(record['source'])
        name = f"{record['category']}__{record['source_split']}__{record['defect']}__{source.stem}"
        image_path = output / 'images' / record['split'] / f'{name}.png'
        shutil.copy2(source, image_path)
        label = '\n'.join('0 ' + ' '.join(f'{v:.8f}' for v in box) for box in record.pop('boxes'))
        (output / 'labels' / record['split'] / f'{name}.txt').write_text(label, encoding='utf-8')
        record['image'] = image_path.relative_to(output).as_posix()
    config = {'path': output.as_posix(), 'train': 'images/train', 'val': 'images/val',
              'test': 'images/test', 'names': {0: 'defect'}}
    (output / 'data.yaml').write_text(yaml.safe_dump(config, allow_unicode=True), encoding='utf-8')
    manifest = {'seed': seed, 'note': 'Supervised repartition; NOT official MVTec AD benchmark.', 'records': records}
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    counts = {split: sum(r['split'] == split for r in records) for split in ('train', 'val', 'test')}
    print(json.dumps(counts))
    print(f'Dataset ready: {output / "data.yaml"}')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--screw', type=Path, default=Path.home() / 'Downloads' / 'screw')
    parser.add_argument('--metal-nut', type=Path, default=Path.home() / 'Downloads' / 'metal_nut')
    parser.add_argument('--output', type=Path, default=ROOT / 'data' / 'yolo')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    prepare({'screw': args.screw, 'metal_nut': args.metal_nut}, args.output, args.seed)


if __name__ == '__main__':
    main()
