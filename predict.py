"""Annotate one image without opening a camera."""
import argparse
from pathlib import Path

import cv2
from common import ROOT, add_inference_args, validate_inference_args, device_name, load_detector
from prepare_dataset import read_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs' / 'prediction.jpg')
    add_inference_args(parser)
    args = parser.parse_args()
    validate_inference_args(args)
    model = load_detector(args.weights)
    frame = read_image(args.source)
    result = model.predict(frame, conf=args.conf, imgsz=args.imgsz,
                           device=device_name(args.device), verbose=False)[0]
    annotated = result.plot()
    status = 'NG: DEFECT DETECTED' if len(result.boxes) else 'NO DEFECT DETECTED (not verified OK)'
    cv2.putText(annotated, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 255), 2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(args.output.suffix or '.jpg', annotated)
    if not ok:
        raise RuntimeError('Could not encode output image')
    encoded.tofile(str(args.output))
    print(status)
    print(f'Saved: {args.output}')


if __name__ == '__main__':
    main()
