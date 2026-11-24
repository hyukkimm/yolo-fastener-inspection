"""Shared inference configuration. Paths are relative to the project, not the shell."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'yolo' / 'data.yaml'
WEIGHTS = ROOT / 'runs' / 'detect' / 'fastener' / 'weights' / 'best.pt'


def device_name(value):
    if value != 'auto':
        return value
    import torch
    return '0' if torch.cuda.is_available() else 'cpu'


def load_detector(weights):
    path = Path(weights)
    if not path.is_file():
        raise FileNotFoundError(f'Train first or set --weights to a trained best.pt: {path}')
    from ultralytics import YOLO
    model = YOLO(str(path))
    if model.task != 'detect' or set(model.names.values()) != {'defect'}:
        raise ValueError('Expected the trained single-class defect detector, not a COCO model.')
    return model


def add_inference_args(parser):
    parser.add_argument('--weights', type=Path, default=WEIGHTS)
    parser.add_argument('--conf', type=float, default=.25)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--device', default='auto')


def validate_inference_args(args):
    if not 0 < args.conf <= 1:
        raise ValueError('--conf must be greater than 0 and at most 1')
    if args.imgsz < 32 or args.imgsz % 32:
        raise ValueError('--imgsz must be a positive multiple of 32')
