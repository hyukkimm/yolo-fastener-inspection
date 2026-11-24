"""Localization metrics and per-product image-level defect confusion counts."""
import argparse
import json
from pathlib import Path

import yaml
from common import ROOT, DATA, add_inference_args, validate_inference_args, device_name, load_detector
from prepare_dataset import read_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--split', choices=('val', 'test'), default='test')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs' / 'evaluation.json')
    add_inference_args(parser)
    args = parser.parse_args()
    validate_inference_args(args)
    model = load_detector(args.weights)
    device = device_name(args.device)
    # Low threshold for PR/mAP; selected operating threshold is used separately below.
    metrics = model.val(data=str(args.data.resolve()), split=args.split, conf=.001,
                        imgsz=args.imgsz, device=device, workers=0,
                        project=str(ROOT / 'runs' / 'evaluate'), name=args.split, plots=True)
    config = yaml.safe_load(args.data.read_text(encoding='utf-8'))
    root = Path(config['path'])
    if not root.is_absolute():
        root = (args.data.parent / root).resolve()
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    counts = {key: dict(TP=0, TN=0, FP=0, FN=0) for key in ('screw', 'metal_nut', 'all')}
    for record in manifest['records']:
        if record['split'] != args.split:
            continue
        result = model.predict(read_image(root / record['image']), conf=args.conf,
                               imgsz=args.imgsz, device=device, verbose=False)[0]
        actual = record['defect'] != 'good'
        predicted = len(result.boxes) > 0
        key = ('TP' if predicted else 'FN') if actual else ('FP' if predicted else 'TN')
        counts[record['category']][key] += 1
        counts['all'][key] += 1
    report = {'split': args.split, 'confidence': args.conf,
              'map50': float(metrics.box.map50), 'map50_95': float(metrics.box.map),
              'image_confusion': counts,
              'note': 'Custom supervised split. Image counts do not measure box localization.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    print(f'Saved: {args.output}')


if __name__ == '__main__':
    main()
