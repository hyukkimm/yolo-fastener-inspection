"""Train a small YOLO detector; --smoke performs a bounded pipeline check."""
import argparse
from pathlib import Path

from common import ROOT, DATA, device_name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--model', default='yolo11n.pt')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--name', default=None)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error('Dataset missing. Run prepare_dataset.py first.')
    if args.epochs < 1 or args.batch < 1 or args.imgsz < 32 or args.imgsz % 32:
        parser.error('epochs/batch must be positive; imgsz must be a multiple of 32.')
    name = args.name or ('smoke' if args.smoke else 'fastener')
    if Path(name).name != name or name in ('.', '..'):
        parser.error('--name must be a simple folder name')
    target = ROOT / 'runs' / 'detect' / name
    if target.exists():
        parser.error(f'Run already exists: {target}. Choose another --name.')
    from ultralytics import YOLO
    model = YOLO(args.model)
    model.train(data=str(args.data.resolve()), epochs=1 if args.smoke else args.epochs,
                imgsz=320 if args.smoke else args.imgsz, batch=args.batch,
                device=device_name(args.device), workers=0, seed=42, deterministic=True,
                project=str(ROOT / 'runs' / 'detect'), name=name, exist_ok=False,
                fraction=.08 if args.smoke else 1.0, patience=20,
                optimizer='AdamW', lr0=.001, mosaic=0.0, mixup=0.0,
                fliplr=0.0, flipud=0.0, degrees=10.0, translate=.05, scale=.15,
                plots=True, cache=False)
    print(f'Weights: {model.trainer.save_dir / "weights" / "best.pt"}')
    if args.smoke:
        print('SMOKE ONLY: these weights do not establish defect detection accuracy.')


if __name__ == '__main__':
    main()
