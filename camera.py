"""Live webcam inspection. Q/Esc exits; S saves the annotated inspection ROI."""
import argparse
from datetime import datetime
import time

import cv2
from common import ROOT, add_inference_args, validate_inference_args, device_name, load_detector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--roi', nargs=4, type=int, metavar=('X', 'Y', 'W', 'H'),
                        help='Crop inspection area in actual camera pixels')
    add_inference_args(parser)
    args = parser.parse_args()
    validate_inference_args(args)
    model = load_detector(args.weights)
    device = device_name(args.device)
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(args.camera)
    try:
        if not cap.isOpened():
            raise RuntimeError('Cannot open webcam. Check Windows permissions, other apps, or --camera 1.')
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        print('Q/Esc: quit | S: save. No automatic OK: product presence is not verified.')
        while True:
            start = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError('Webcam stopped delivering frames')
            if args.roi:
                x, y, w, h = args.roi
                if x < 0 or y < 0 or w <= 0 or h <= 0 or x+w > frame.shape[1] or y+h > frame.shape[0]:
                    raise ValueError(f'ROI outside actual frame {frame.shape[1]}x{frame.shape[0]}')
                frame = frame[y:y+h, x:x+w]
            result = model.predict(frame, conf=args.conf, imgsz=args.imgsz, device=device, verbose=False)[0]
            display = result.plot()
            status = 'NG: DEFECT DETECTED' if len(result.boxes) else 'NO DEFECT DETECTED'
            color = (0, 0, 255) if len(result.boxes) else (0, 200, 255)
            fps = 1 / max(time.perf_counter() - start, 1e-6)
            cv2.putText(display, f'{status} | {fps:.1f} FPS', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2)
            cv2.putText(display, 'Presence not checked | Q: quit S: save', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
            cv2.imshow('Fastener inspection', display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            if key == ord('s'):
                folder = ROOT / 'outputs' / 'captures'
                folder.mkdir(parents=True, exist_ok=True)
                path = folder / (datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.jpg')
                encoded_ok, encoded = cv2.imencode('.jpg', display)
                if encoded_ok:
                    encoded.tofile(str(path))
                    print(f'Saved: {path}')
            if cv2.getWindowProperty('Fastener inspection', cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
