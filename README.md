# ML Training Manager (ML 학습 관리자)

ML 학습 과정을 자동화하고 모니터링하는 관리 도구입니다. 이 도구는 CSV 파일에 나열된 모델들을 자동으로 학습시키고, 학습 상태를 모니터링하며, 학습 완료 시 알림을 제공합니다.

## 주요 기능

- CSV 파일에서 학습할 모델 목록 관리
- 자동으로 학습 프로세스 실행 및 관리
- 터미널 UI로 학습 상태 실시간 모니터링
- WandB를 통한 학습 진행 상황 모니터링
- 학습 완료/중단 시 알림 (데스크톱, 소리, 이메일)
- 학습 완료 후 결과 및 가중치 파일 정보 저장
- INI 파일을 통한 설정 관리 (CSV 경로, WandB, 이메일, GPU 할당 등)
- 학습 전 환경 설정 스크립트 지원 (conda 환경 활성화 등)
- 프로세스 순서 기반 GPU 할당 기능
- 프로세스별 다중 GPU 할당 지원
- 사전 학습 모델 가중치 파일 자동 검색 및 적용
- 자동 연속 학습 지원 (학습 완료 후 다음 모델 자동 시작)

## 설치 방법

### 요구 사항

- Python 3.7 이상
- pandas, wandb, psutil 등의 패키지

### 설치

```bash
git clone <repository-url>
cd ml_training_manager
pip install -e .
```

## 사용 방법

### CSV 파일 구조

학습 관리를 위해 `ML_Experiment_Table.csv` 파일이 필요합니다. 파일 형식은 다음과 같습니다:

| ID | Name | TrainingCommand | TrainingCheck | WandbRunID | WeightFile | PretrainedModelId | ... |
|----|------|-----------------|--------------|------------|------------|------------------|-----|
| model1 | 모델1 | python train.py --param=value | | | | | ... |
| model2 | 모델2 | python train.py --other=value | Training | run-abcdef | | model1 | ... |
| model3 | 모델3 | python train.py --test=value | Done | run-123456 | model.pth | | ... |

- ID: 모델 고유 식별자
- Name: 모델 이름
- TrainingCommand: 학습 실행을 위한 명령어
- TrainingCheck: 학습 상태 (공백: 학습 전, Training: 학습 중, Done: 완료, Crash: 중단)
- WandbRunID: WandB Run ID
- WeightFile: 학습된 모델 가중치 파일 경로
- PretrainedModelId: 사전 학습 모델의 ID (해당 모델의 가중치 파일 자동 검색)

### 설정 파일 (INI)

설정 파일 예제:
```ini
[general]
# CSV 파일 경로
csv_file = training_status/ML_Experiment_Table.csv
# 학습 상태 확인 간격 (초)
check_interval = 30
# 동시에 실행할 최대 학습 수
max_training_process = 2
# 프로세스별 GPU 할당
process_gpu_mapping = process0=0+1,process1=2,process2=3
# 자동으로 다음 모델 학습 진행 여부
auto_continue = true

[wandb]
# WandB 사용자 또는 팀 이름
entity = your_entity
# WandB 프로젝트 이름
project = Controller-Imitator-Multi-Final

[environment]
# 학습 전 실행할 설정 스크립트
setup_script = setup.sh
# conda 환경 사용 여부
use_conda = true
# conda 환경 이름
conda_env = ml_env
# 추가 환경 변수
env_vars = PYTHONPATH=/path/to/add

[gpu]
# GPU 할당 활성화 여부
enable_gpu_assignment = true
# 기본 GPU ID
default_gpu = 0
# 사용 가능한 GPU 목록
gpu_list = 0,1,2,3
# 프로세스 순서로 GPU 할당
use_process_order = true
# 멀티 GPU 할당 허용 여부
allow_multi_gpu = true
```

#### 환경 설정 스크립트

학습 전에 환경 설정을 위한 스크립트를 지정할 수 있습니다. 예를 들어 `setup.sh`:

```bash
#!/bin/bash
# conda 환경 설정 등 필요한 초기화 작업

# 예: CUDA 설정
export CUDA_HOME=/usr/local/cuda-11.7
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 기타 환경 설정
```

### 실행 방법

설정 파일을 사용하는 경우:
```bash
ml-training-manager --config /path/to/config.ini
```

명령행 인자를 직접 지정하는 경우:
```bash
ml-training-manager --csv /path/to/ML_Experiment_Table.csv --wandb_entity 사용자명 --max_training_process 2 --auto_continue
```

### 명령행 인자

- `--csv`: ML_Experiment_Table.csv 파일 경로 (설정 파일보다 우선)
- `--config`: 설정 파일 경로 (INI 형식)
- `--create_config`: 지정된 경로에 기본 설정 파일 생성
- `--check_interval`: 학습 상태 확인 간격 (초)
- `--max_training_process`: 동시에 실행할 최대 학습 수
- `--wandb_entity`: WandB 사용자 또는 팀 이름
- `--wandb_project`: WandB 프로젝트 이름
- `--no_ui`: 터미널 UI 비활성화
- `--auto_continue`: 자동으로 다음 모델 학습 진행

## 다중 GPU 할당

프로세스별로 여러 개의 GPU를 할당할 수 있습니다. 이를 위해 두 가지 방법이 제공됩니다:

### 1. 설정 파일에서 다중 GPU 할당

`process_gpu_mapping` 설정에서 `+` 기호를 사용하여 여러 GPU를 할당할 수 있습니다:

```ini
process_gpu_mapping = process0=0+1+2,process1=3
```

위 설정은 첫 번째 프로세스(process0)에 GPU 0, 1, 2를 할당하고, 두 번째 프로세스(process1)에 GPU 3을 할당합니다.

### 2. 자동 다중 GPU 할당

`allow_multi_gpu = true`로 설정하면 프로세스 0에 모든 사용 가능한 GPU가 자동으로 할당되고, 나머지 프로세스는 개별 GPU를 할당받습니다.

## 사전 학습 모델 사용

CSV 파일의 `PretrainedModelId` 열을 사용하여 다른 모델의 가중치 파일을 자동으로 가져와 학습에 사용할 수 있습니다:

1. `PretrainedModelId` 열에 사용할 모델의 ID를 입력합니다.
2. 프로그램은 해당 ID를 가진 모델의 `WeightFile` 값을 찾습니다.
3. 해당 가중치 파일을 새 모델의 학습에 `--pretrained_path` 인자로 자동 추가합니다.
4. 가중치 파일을 찾을 수 없는 경우, `PretrainedModelId` 값을 음수로 변경하고 경고 메시지를 표시합니다.

## GPU 할당 우선순위

GPU 할당은 크게 두 가지 방식으로 이루어집니다:

### 1. 프로세스 순서 기반 할당 (기본 방식)

이 방식은 프로세스가 시작된 순서대로 GPU를 할당합니다. 설정 파일의 `[gpu]` 섹션에서 `use_process_order = true`로 설정하면 활성화됩니다.

- 프로세스별 GPU 매핑: `[general]` 섹션의 `process_gpu_mapping` 설정
  - 예: `process0=0+1,process1=2,process2=3`
  - 특정 매핑이 없는 경우, 라운드 로빈 방식으로 GPU 목록에서 할당

### 2. 모델 ID 기반 할당

이 방식은 CSV 파일의 GpuID 열을 사용하여 직접 GPU를 할당합니다. 설정 파일에서 `use_process_order = false`로 설정하면 활성화됩니다.

## 자동 연속 학습

`auto_continue = true`로 설정하면 학습 완료 후 자동으로 다음 모델의 학습을 시작합니다. 이 기능은 하나의 모델이 학습을 완료하면 자동으로 대기 중인 다음 모델을 학습 큐에 넣습니다.

## 터미널 UI 사용법

터미널 UI가 활성화된 상태에서는 다음 키를 사용하여 조작할 수 있습니다:

- `↑`/`↓`: 모델 선택 이동
- `r`: 화면 새로고침
- `s`: 선택한 모델의 학습 중지
- `q`: 프로그램 종료

## 알림 설정

이메일 알림을 활성화하려면 설정 파일의 `[email]` 섹션에서 설정하거나 다음 환경 변수를 설정하세요:

```bash
export NOTIFY_SMTP_SERVER="smtp.gmail.com"
export NOTIFY_SMTP_PORT="587"
export NOTIFY_EMAIL_USERNAME="your_email@gmail.com"
export NOTIFY_EMAIL_PASSWORD="your_password_or_app_password"
export NOTIFY_FROM_ADDR="your_email@gmail.com"
export NOTIFY_TO_ADDR="recipient_email@example.com"
```

## 프로젝트 구조

```
ml_training_manager/
├── training_manager/
│   ├── __init__.py
│   ├── main_training_manager.py      # 메인 관리자 프로그램
│   ├── csv_handler.py                # CSV 파일 처리
│   ├── process_manager.py            # 학습 프로세스 관리
│   ├── wandb_monitor.py              # WandB 모니터링
│   ├── terminal_ui.py                # 터미널 UI
│   ├── notification.py               # 알림 시스템
│   └── config_handler.py             # 설정 파일 핸들러
├── example_config.ini                # 예제 설정 파일
├── setup.py
└── README.md
``` 