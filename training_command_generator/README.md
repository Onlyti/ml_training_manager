# 머신러닝 학습 명령어 생성기

이 프로그램은 머신러닝 모델 학습을 위한 실험 조합을 쉽게 구성하고, 학습 명령어를 생성하는 GUI 도구입니다.

## 기능

- INI 설정 파일을 통한 실험 매개변수 구성
- 설정된 매개변수를 체크박스로 쉽게 선택/해제
- 사용자 지정 학습 스크립트 파일 지원 (기본값: train.py)
- 기본 설정 파일 자동 로드 기능
  - 프로그램 실행 시 같은 경로의 default_config.ini 파일 자동 로드
  - 사용자 지정 기본 INI 파일 설정 및 자동 로드 기능
- 선택된 매개변수로 학습 명령어 자동 생성
- 생성된 명령어를 별도의 터미널 창에서 실행 기능
  - Windows: 명령 프롬프트 창
  - Linux: gnome-terminal, konsole, xterm 등
  - macOS: Terminal.app

## 필요 라이브러리

```
PyQt5
```

## 설치 방법

1. 필요한 라이브러리 설치:

```bash
pip install PyQt5
```

2. 프로그램 다운로드:

```bash
git clone [리포지토리 URL]
cd training_command_generator
```

## 사용 방법

1. 프로그램 실행:

```bash
python training_command_generator.py
```

2. 프로그램이 실행되면 자동으로 같은 경로의 default_config.ini 파일을 로드
3. 다른 INI 파일을 사용하려면 "찾아보기" 버튼을 클릭하여 선택
4. 자주 사용하는 INI 파일은 "기본 경로로 설정" 버튼을 클릭하여 기본 경로로 설정 (다음 실행 시 자동 로드)
5. 학습 스크립트 파일 이름 입력 또는 "찾아보기" 버튼으로 선택 (기본값: train.py)
6. "로드" 버튼을 클릭하여 설정 파일 로드
7. 체크박스를 통해 원하는 매개변수 선택/해제
8. "명령어 생성" 버튼을 클릭하여 학습 명령어 생성
9. "명령어 실행" 버튼을 클릭하면 별도의 터미널 창에서 명령어 실행

## 설정 파일 우선순위

프로그램은 다음 순서로 설정 파일을 로드합니다:

1. 프로그램 경로의 default_config.ini 파일
2. 사용자가 지정한 기본 INI 파일 (settings.json에 저장된 경로)
3. 수동으로 선택한 INI 파일

## 설정 저장

프로그램은 다음 정보를 자동으로 저장하고 다음 실행 시 복원합니다:

- 기본 INI 파일 경로 (자동 로드)
- 마지막으로 사용한 학습 스크립트 이름

## INI 파일 형식

INI 파일은 다음과 같은 새로운 형식을 따릅니다:

```ini
[parameter_name]
display_name1 = command_value1
display_name2 = command_value2

[system_parameter_name]
display_name1 = command_value1
display_name2 = command_value2
```

- 각 섹션 이름(`parameter_name`)은 명령줄 파라미터 이름으로 사용됩니다
- 각 변수 이름(`display_name`)은 UI에 표시되는 간단한 이름입니다
- 각 변수 값(`command_value`)은 실제 명령어에 사용되는 값입니다
- 섹션 이름이 "system_"으로 시작하면 "시스템 설정" 탭에 표시됨
- 그 외의 섹션은 "학습 설정" 탭에 표시됨

### 예시

```ini
[batch_size]
small = 16
medium = 32
large = 64

[system_gpu]
enable = true
disable = false
```

위 설정에서 "batch_size" 섹션의 "small"을 선택하면 명령어에 `-batch_size 16`이 추가됩니다.
만약 "medium"과 "large"도 함께 선택하면 `-batch_size 32 64`가 추가됩니다.

## 예시

`default_config.ini` 파일을 확인하여 구성 예시를 확인할 수 있습니다.

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 