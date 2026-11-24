# YOLO 기반 나사·너트 결함 검출

Python + Ultralytics YOLO11n + OpenCV로 MVTec AD의 나사/금속 너트 결함을 학습하고 웹캠 영상에 표시합니다.
첫 버전은 모든 불량을 `defect` 한 종류로 학습합니다. 나사/너트 자체를 찾거나 결함 유형을 구분하는 모델은 아닙니다.

## 1. 설치 (최초 한 번)

PowerShell에서 프로젝트 폴더로 이동합니다. 현재 PC에는 `.venv`를 준비해 두었습니다.
새 PC/다시 설치할 때만 다음 설치 명령을 실행하세요.

```powershell
cd "C:\Users\tkdgu\Desktop\욜로 탐지\yolo-fastener-inspection"
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

가상환경 활성화 없이 `.venv`의 Python을 직접 사용하므로 PowerShell 실행 정책을 변경할 필요가 없습니다.
PyCharm 인터프리터도 프로젝트의 `.venv\Scripts\python.exe`로 지정하세요.

## 2. 데이터 준비

```powershell
.\.venv\Scripts\python.exe prepare_dataset.py
```

기본으로 `Downloads\screw`, `Downloads\metal_nut`을 읽습니다. 동일 이름의 중첩 폴더도 자동으로 찾습니다.
원본은 그대로 두고 `data\yolo`에 이미지 사본, YOLO 라벨, `data.yaml`, `manifest.json`을 생성합니다.
이미 준비되어 있으면 다시 실행하지 마세요. 기존 출력은 덮어쓰지 않습니다.

다른 경로는 다음처럼 지정할 수 있습니다.

```powershell
.\.venv\Scripts\python.exe prepare_dataset.py --screw "D:\datasets\screw" --metal-nut "D:\datasets\metal_nut" --output "D:\datasets\fastener-yolo"
```

- 각 제품/원본 split/불량 종류별로 약 60/20/20%를 train/val/test로 고정 분할합니다(seed=42).
- 원본 test의 결함 일부를 학습에 사용하므로 **MVTec 공식 이상 탐지 벤치마크 결과와 직접 비교할 수 없습니다.**
- 마스크의 연결된 영역마다 박스를 생성합니다. 정상 이미지는 빈 라벨입니다.
- 읽을 수 없는 이미지, 누락/빈 마스크, 크기 불일치, 이미지 바이트 중복을 검사합니다.
- manifest에 원본 경로, 원본 분할, 새 분할, SHA256을 기록합니다. SHA256은 시각적 유사 이미지까지 찾아내지는 않습니다.
- 최종 test는 모델/임계값 선택에 사용하지 마세요. 같은 실물의 연속 촬영 사진을 추가할 때는 실물 단위로 분리해야 합니다.

## 3. 학습

실행 확인용 짧은 학습(1 epoch, 학습 이미지 일부, 320px):

```powershell
.\.venv\Scripts\python.exe train.py --smoke --device cpu
```

이 결과는 `runs\detect\smoke`에 저장됩니다. **성능을 검증한 실사용 모델이 아닙니다.**
이미 시험 학습을 실행한 환경에서 다시 확인하려면 `--smoke --name smoke_v2`처럼 새 이름을 지정하세요.

본 학습:

```powershell
.\.venv\Scripts\python.exe train.py --epochs 100 --imgsz 640 --batch 4 --device cpu
```

첫 실행 시 사전 학습 YOLO11n 가중치를 다운로드하므로 인터넷 연결이 필요합니다.
학습 중 손실과 검증 지표가 출력됩니다. 검증 성능이 20 epoch 동안 개선되지 않으면 조기 종료됩니다.
`runs\detect\fastener\weights\best.pt`가 검증 성능 기준 최적 모델이고 `last.pt`는 마지막 모델입니다.
`results.csv`, `results.png`, `args.yaml`에 학습 기록이 저장됩니다.
동일 이름의 실행 폴더가 있으면 `--name fastener_v2`처럼 새 이름을 쓰세요. 이때 검사 명령의 `--weights`도 새 경로로 지정하세요.

현재 PC는 Intel 내장 그래픽 환경이라 CPU 학습이 느릴 수 있습니다. CUDA 지원 NVIDIA GPU와 해당 PyTorch 환경이 있는 다른 PC에서는 `--device 0`을 사용할 수 있습니다.
영상 추론 FPS도 하드웨어에 따라 달라집니다. 작은 결함은 해상도를 낮추면 놓칠 수 있습니다.

## 4. 검증 및 최종 평가

검증 세트로 임계값(`--conf`)을 결정한 뒤, 고정한 설정으로 test를 평가합니다.

```powershell
.\.venv\Scripts\python.exe evaluate.py --split val --conf 0.25 --output outputs/validation.json
.\.venv\Scripts\python.exe evaluate.py --split test --conf 0.25
```

박스 mAP와 제품별 TP/TN/FP/FN을 출력합니다. TP=불량 검출, FN=불량 놓침, FP=정상 오검출, TN=정상 무검출입니다.
이미지 단위 TP는 결함 위치가 정확하다는 뜻이 아닙니다. 위치 성능은 mAP와 결과 이미지를 함께 보세요.
`outputs\evaluation.json`에 평가 결과를 저장합니다. mAP 계산은 낮은 confidence를 사용하며, 이미지 판정에는 사용자가 준 `--conf`를 사용합니다.

## 5. 사진 한 장 검사

```powershell
.\.venv\Scripts\python.exe predict.py --source "C:\Users\tkdgu\Downloads\screw\screw\test\scratch_head\000.png"
```

결과는 `outputs\prediction.jpg`입니다. 위 명령은 사용 예시이며 최종 성능 평가는 새로 나눈 test로만 하세요.

## 6. 웹캠 검사

본 학습이 끝난 후:

```powershell
.\.venv\Scripts\python.exe camera.py --camera 0 --conf 0.25
```

- 카메라를 못 찾으면 `--camera 1`로 바꿔보세요. Windows 카메라 권한과 다른 앱의 점유 여부도 확인하세요.
- `Q` 또는 `Esc`: 종료. `S`: 현재 결과 저장.
- 검사 영역을 자르려면 `--roi X Y W H`를 사용합니다. 예: `--roi 320 40 640 640` (실제 카메라 해상도 범위 안이어야 합니다).
- 화면에 결함 박스와 confidence, FPS를 표시합니다.
- **결함 검출 시 NG, 미검출 시 NO DEFECT DETECTED**입니다. 빈 화면/초점 불량/검사 대상 부재를 판단하지 않으므로 미검출을 OK로 확정하지 않습니다.

너트는 상부 수직 촬영, 나사는 데이터와 유사한 측면 촬영부터 시작하세요. 카메라·거리·배경·조명을 고정하고 부품이 검사 영역을 충분히 채우도록 배치하세요.
윗면만 촬영하면 옆면/밑면 결함은 검사할 수 없습니다. 공개 데이터와 실제 웹캠 환경 차이 때문에 직접 촬영한 정상/불량 데이터로 추가 검증·학습이 필요합니다.
너트의 `flip` 등은 스크래치가 아닌 상태 이상도 포함합니다. `crack` 전용 클래스나 균열 검출 성능은 주장하지 않습니다.

## 코드

| 파일 | 역할 |
|---|---|
| prepare_dataset.py | 원본 확인, 데이터 분할, 마스크→박스 변환 |
| train.py | 사전 학습 모델 추가 학습 |
| evaluate.py | 검증/테스트 성능 평가 |
| predict.py | 이미지 파일 검사 |
| camera.py | 웹캠 실시간 검사 |
| common.py | 경로, 장치, 학습 모델 확인 |

테스트: `.\.venv\Scripts\python.exe -m pytest -q`

데이터, 모델, 실행 결과, 가상환경은 `.gitignore`로 제외합니다. 코드만 GitHub에 올리면 됩니다.
MVTec AD는 CC BY-NC-SA 4.0 비상업 조건입니다. 데이터 재배포/상용 사용은 공식 조건을 확인하세요.
Ultralytics의 AGPL-3.0/Enterprise 라이선스도 배포 방식에 맞게 확인하세요.

공식 참고:
- https://www.mvtec.com/research-teaching/datasets/mvtec-ad
- https://docs.ultralytics.com/modes/train/
- https://docs.ultralytics.com/datasets/detect/
- https://www.ultralytics.com/license
