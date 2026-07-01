# Jetson Orin NX 16GB 이식 가이드

PC(x86_64, Ubuntu 24.04, 클라우드 LLM)에서 만든 VICA 음성 파이프라인을
Jetson(ARM64, Ubuntu 22.04, 로컬 LLM)으로 옮겨 실행하는 절차.

> 핵심: 코드는 거의 그대로 간다. **환경을 다시 세우는 것**이 이식이다.
> - `.venv` 재생성 (ARM64 휠)
> - `vica_interfaces` 다시 `colcon build`
> - LLM 을 로컬 Ollama(gemma4 e2b)로 (`.env` 만 바꾸면 됨)

## 대상 환경 (확인됨)
- Ubuntu 22.04 / Python 3.10 / ARM64(aarch64)
- JetPack 6.2.1, CUDA 12.6, cuDNN 9.3, TensorRT 10.3

---

## 1. 코드 가져오기
```bash
cd ~
git clone <레포주소> langchain    # 또는 PC 에서 scp 로 복사
cd langchain
```

## 2. 시스템 의존성 (apt)
```bash
sudo apt update
sudo apt install -y python3-venv python3-dev \
    portaudio19-dev libportaudio2 \
    espeak-ng ffmpeg
```
- `portaudio` : 마이크/스피커(sounddevice)
- `espeak-ng` : (TTS 음소화 백엔드에서 쓰일 수 있음)

## 3. 파이썬 가상환경 재생성
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
⚠️ `requirements.txt` 는 PC(Python 3.12) 기준으로 핀이 박혀 있어, ARM64/Python 3.10
에서 **일부 버전이 안 맞을 수 있다.** 실패하면:
- 해당 줄의 `==버전` 을 지우고 (핀 완화) 다시 설치, 또는
- 핵심 패키지만 우선 설치:
  `pip install langchain langchain-ollama faster-whisper supertonic sounddevice numpy python-dotenv PyYAML pydantic`

우리 코드는 `torch` 를 직접 쓰지 않는다(faster-whisper=ctranslate2, supertonic=onnxruntime).

## 4. 로컬 LLM (Ollama + gemma4 e2b)
```bash
curl -fsSL https://ollama.com/install.sh | sh   # ARM64 지원, CUDA 자동 감지
ollama pull gemma4:e2b        # 정확한 태그는 https://ollama.com/library 에서 확인
ollama serve &                # 백그라운드 서버 (보통 자동 실행됨)
```
빠른 확인: `ollama run gemma4:e2b "안녕"`

## 5. `.env` 설정 (로컬 LLM 으로)
`.env.example` 을 참고해 `.env` 를 만든다:
```bash
OLLAMA_HOST=http://localhost:11434
VICA_LLM_MODEL=gemma4:e2b
# OLLAMA_API_KEY 는 로컬에선 필요 없음 (비워두거나 삭제)
```
→ 코드 수정 없이 로컬 LLM 으로 전환된다.

## 6. (CLI 부터) 먼저 검증
```bash
source .venv/bin/activate
python -m src.main            # 키보드 + TTS
# 잘 되면: VICA_STT=1 python -m src.main   (마이크)
```
로봇/ROS2 없이 파이프라인부터 확인하는 게 안전하다.

## 7. ROS2 (Humble) 설치 + 메시지 빌드
Jetson 에 ROS2 Humble 이 없으면 설치:
```bash
# https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html 참고
sudo apt install -y ros-humble-ros-base ros-dev-tools
```
커스텀 메시지 재빌드:
```bash
source /opt/ros/humble/setup.bash
cd ~/langchain/ros2_ws
colcon build --packages-select vica_interfaces
source install/setup.bash
```

## 8. ROS2 실행
```bash
source /opt/ros/humble/setup.bash
source ~/langchain/ros2_ws/install/setup.bash
ros2 launch launch/vica_voice.launch.py     # LLM + TTS
# 별도 터미널(같은 source 2줄): python -m src.ros_stt_node   (마이크)
```

---

## 성능 메모 (나중에)
- Jetson 은 CUDA/cuDNN/TensorRT 가 있으므로:
  - faster-whisper 를 `device="cuda", compute_type="float16"` 로 바꾸면 STT 가속 (src/stt.py)
  - Ollama 는 CUDA 를 자동으로 쓴다
- gemma4 e2b 가 16GB 에 맞는지, 응답 속도가 실시간 대화에 충분한지 실기에서 확인.
- 안 맞으면 더 작은 모델(예: llama3.2:3b, qwen2.5:3b)로 교체 — `.env` 의 `VICA_LLM_MODEL` 만 바꾸면 됨.
