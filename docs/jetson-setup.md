# Jetson Orin NX 16GB 이식 가이드

PC(x86_64, Ubuntu 24.04)에서 만든 VICA 음성 파이프라인을 Jetson(ARM64, Ubuntu 22.04)
으로 옮겨 실행하는 절차. STT/TTS 는 Jetson 온디바이스(GPU), LLM 은 Ollama Cloud(주 모델).

> 핵심: 코드는 거의 그대로 간다. **환경을 다시 세우는 것**이 이식이다.
> - `.venv` 재생성 (ARM64 휠)
> - `vica_interfaces` 다시 `colcon build`
> - LLM 은 클라우드(Ollama Cloud) 유지 — `.env` 의 클라우드 3줄만 채우면 된다.
>   (오프라인이 필요하면 로컬 Ollama 로 전환 가능)

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

## 4. (선택) 오프라인/폴백용 로컬 LLM (Ollama + gemma4 e2b)

기본 LLM 은 Ollama Cloud 다. 아래는 오프라인이 필요할 때만 설정하는 선택 단계다.
```bash
curl -fsSL https://ollama.com/install.sh | sh   # ARM64 지원, CUDA 자동 감지
ollama pull gemma4:e2b        # 정확한 태그는 https://ollama.com/library 에서 확인
ollama serve &                # 백그라운드 서버 (보통 자동 실행됨)
```
빠른 확인: `ollama run gemma4:e2b "안녕"`

## 5. `.env` 설정
`.env.example` 을 참고해 `.env` 를 만든다. 기본은 클라우드 LLM:
```bash
OLLAMA_HOST=https://ollama.com
VICA_LLM_MODEL=gemma4:cloud
OLLAMA_API_KEY=<ollama cloud 키>
```
오프라인/폴백으로 로컬 Ollama 를 쓰려면 (4번 설치 후) 위 3줄 대신:
```bash
OLLAMA_HOST=http://localhost:11434
VICA_LLM_MODEL=gemma4:e2b
# OLLAMA_API_KEY 는 로컬에선 필요 없음 (비워두거나 삭제)
```
→ 코드 수정 없이 `.env` 만으로 전환된다.

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
- Ollama 는 CUDA 를 자동으로 쓴다.
- gemma4 e2b 가 16GB 에 맞는지, 응답 속도가 실시간 대화에 충분한지 실기에서 확인.
- 안 맞으면 더 작은 모델(예: llama3.2:3b, qwen2.5:3b)로 교체 — `.env` 의 `VICA_LLM_MODEL` 만 바꾸면 됨.

## STT CUDA 가속 (2026-07-07 완료)

pip 의 `ctranslate2` ARM64 휠은 **CPU 전용**이라 CUDA 소스 빌드가 필요하다.
(Jetson AI Lab 인덱스의 cu128 휠도 `libctranslate2.so.4` 미포함이라 단독으로 못 씀)

결과: STT(10초 오디오) small/CPU 5.2초 → **medium/CUDA 1.4초** (더 정확한 모델인데 3.7배 빠름).
small 은 "식당"→"직당" 오인식이 있어 medium 채택. `.env`:
`VICA_STT_MODEL=medium`, `VICA_STT_DEVICE=cuda`, `VICA_STT_COMPUTE=float16`

빌드 산출물은 `~/jetson-builds/` 에 보존 (`ct2-install`=라이브러리+헤더, `ct2-python-src`=파이썬 패키지 소스).

### `.venv` 재생성 시 재설치 방법 (재빌드 불필요)
```bash
cd ~/jetson-builds/ct2-python-src
CTRANSLATE2_ROOT=~/jetson-builds/ct2-install LDFLAGS='-Wl,-rpath,$ORIGIN' \
  ~/dev/vica-voice-llm/.venv/bin/pip install .
cp ~/jetson-builds/ct2-install/lib/libctranslate2.so.4* \
   ~/dev/vica-voice-llm/.venv/lib/python3.10/site-packages/ctranslate2/
```
(`LDFLAGS` 의 rpath `$ORIGIN` + lib 복사로 sudo/LD_LIBRARY_PATH 없이 동작)

### 처음부터 다시 빌드하는 방법 (~10분, 8코어)
```bash
git clone --depth 1 --recursive https://github.com/OpenNMT/CTranslate2.git
cd CTranslate2 && mkdir build && cd build
PATH=/usr/local/cuda/bin:$PATH cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DWITH_CUDA=ON -DWITH_CUDNN=ON -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DWITH_MKL=OFF -DWITH_OPENBLAS=ON -DOPENMP_RUNTIME=COMP -DBUILD_CLI=OFF \
  -DCMAKE_INSTALL_PREFIX=$HOME/jetson-builds/ct2-install
make -j4 && make install
# 이후 위 '재설치 방법' 수행
```
필요 패키지: cmake, gcc/g++, nvcc(JetPack 포함), libcudnn9-dev-cuda-12, libopenblas-dev

## TTS CUDA 가속 (2026-07-07 완료)

supertonic(onnxruntime)은 Jetson AI Lab 의 GPU 휠만 설치하면 된다 (빌드 불필요):

```bash
.venv/bin/pip uninstall -y onnxruntime
.venv/bin/pip install onnxruntime-gpu==1.24.0 --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
```

결과: 합성 3.0초 → **0.3초** (워밍업 후). provider 전환은 `src/tts.py` 에 이미 반영되어 있고,
CUDA 가 없는 환경(PC)에서는 자동으로 CPU 폴백된다.
