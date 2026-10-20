# vica-voice-llm

시각장애인 동행 로봇 **VICA**의 음성 인터페이스입니다.
호출어·긴급어 감지, 음성 인식(STT), LLM 의도 해석, 음성 합성(TTS)을 ROS 2 노드로 제공하며, 모든 처리가 로봇(Jetson) 안에서 이루어집니다.

> 프로젝트 소개는 [26_HC021](https://github.com/VICA-smarthandle/26_HC021) 저장소를 참고해 주세요.

## 주요 기능

- **호출어·긴급어 상시 감지** : "비카야"와 "멈춰·정지"를 직접 학습한 openWakeWord 모델로 항상 듣습니다. 긴급어는 STT로 한 번 더 확인한 뒤 LLM을 거치지 않고 바로 비상정지 이벤트를 보냅니다.
- **온디바이스 음성 인식** : Qwen3-ASR 1.7B(transcribe.cpp, CUDA). 말이 없을 때 없는 말을 지어내지 않도록 2단계로 판정합니다.
- **로컬 LLM 의도 해석** : KT Mi:dm 2.0 Mini(Ollama)로 의도 12종과 목적지 후보를 정해진 JSON 형식으로 받습니다. "그래", "아니" 같은 짧은 대답은 LLM 없이 규칙으로 바로 처리합니다.
- **음성 합성** : Supertonic-3로 실시간 합성하고, 자주 쓰는 안내 문장은 CosyVoice 3로 미리 만들어 바로 재생합니다.
- **안전한 구조** : LLM은 목적지를 "제안"만 하고, 실제 출발 여부는 미션 매니저(`vica_ros2_ws`)가 확인한 뒤 결정합니다.

## 구조

```
마이크(reSpeaker) ─▶ ros_wakeword_node ──(/vica/user_text)──▶ ros_node (LLM) ──(/vica/intent)──▶ 미션 매니저
                        │  호출어·긴급어 감지                      │  규칙 + Mi:dm 2.0 Mini
                        │  Qwen3-ASR 받아쓰기                      │
                        └──(/vica/emergency)──▶ 비상정지            └──(/vica/tts_request)──▶ ros_tts_node ─▶ 스피커
```

| 노드 | 하는 일 |
|---|---|
| `src/ros_wakeword_node.py` | 마이크 입력, 호출어·긴급어 감지, 음성 입력창 관리, STT |
| `src/ros_node.py` | 받아쓴 문장 → 의도 해석(규칙·LLM) → 미션 매니저로 전달 |
| `src/ros_tts_node.py` | 안내 문장 재생(미리 만든 음성 → 실시간 합성 순) |

## 주요 토픽

| 토픽 | 방향 | 내용 |
|---|---|---|
| `/vica/user_text` | 발행 | 받아쓴 사용자 발화 |
| `/vica/intent` | 발행 | 의도 해석 결과 (`vica_interfaces/VicaIntent`) |
| `/vica/emergency` | 발행 | 긴급어 감지 (`vica_interfaces/EmergencyEvent`) |
| `/vica/tts_request` | 발행·구독 | 재생할 안내 문장 (`우선순위:문장`) |
| `/vica/robot_state` | 구독 | 로봇 상태 (`vica_interfaces/RobotState`) |
| `/vica/listen_request` | 발행·구독 | 질문 뒤 대답 받을 준비 |
| `/vica/wake`, `/vica/wake_reply` | 발행·구독 | 호출어 감지와 응답 여부 |

전체 목록은 [docs/ros2-interface.md](docs/ros2-interface.md)에 있습니다.

## 실행 환경

- Jetson Orin NX 16GB · JetPack 6.2 · Ubuntu 22.04
- ROS 2 Humble, Python 3.10
- reSpeaker Mic Array v3.0
- Ollama, transcribe.cpp(CUDA 빌드)
- `vica_interfaces` 메시지 패키지 ([vica_ros2_ws](https://github.com/VICA-smarthandle/vica_ros2_ws))

## 설치

```bash
git clone https://github.com/VICA-smarthandle/vica-voice-llm.git
cd vica-voice-llm
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Jetson에서는 일부 패키지(onnxruntime-gpu 등)를 따로 설치해야 합니다. 자세한 순서는 [docs/jetson-setup.md](docs/jetson-setup.md)를 따라 주세요.

**로컬 LLM 준비**

```bash
# config/ollama/Modelfile.midm2-mini 의 FROM 경로를 받은 GGUF 파일 위치로 바꾼 뒤
ollama create midm2-mini -f config/ollama/Modelfile.midm2-mini
```

**STT 모델** : Qwen3-ASR 1.7B GGUF(Q8_0)를 받고, transcribe.cpp를 CUDA로 빌드해 둡니다.

## 설정 (.env)

`.env.example`을 참고해 `.env`를 만듭니다. `.env`는 저장소에 올리지 않습니다.

| 변수 | 값 (실제 운영) | 설명 |
|---|---|---|
| `VICA_STT_ENGINE` | `qwen` | 대화 받아쓰기 엔진 (기본값 `whisper`는 이전 엔진) |
| `VICA_VERIFY_STT_ENGINE` | `qwen` | 긴급어 확인 엔진 |
| `VICA_QWEN_MODEL` | GGUF 파일 경로 | Qwen3-ASR 1.7B Q8_0 |
| `VICA_TRANSCRIBE_LIB`, `VICA_TRANSCRIBE_PY` | 빌드한 transcribe.cpp 경로 | 라이브러리와 파이썬 바인딩 |
| `VICA_LLM_PROVIDER` | `ollama` | LLM 백엔드 |
| `OLLAMA_HOST` | `http://localhost:11434` | 로봇 안 Ollama |
| `VICA_LLM_MODEL` | `midm2-mini` | 위에서 만든 Mi:dm 모델 이름 |
| `VICA_LOCAL_RULES` | `1` | 짧은 대답 규칙·대화 단계 규칙 켜기 |

## 실행

```bash
source /opt/ros/humble/setup.bash
source <vica_ros2_ws>/install/setup.bash     # vica_interfaces
ros2 launch launch/vica_voice.launch.py
```

launch 파일이 Ollama 서버와 세 노드를 함께 띄웁니다.

## 테스트

```bash
.venv/bin/python -m pytest -q tests
```

## 폴더 구조

```
vica-voice-llm/
├── src/          ROS 2 노드와 음성·LLM 처리 코드
├── launch/       ROS 2 launch 파일
├── models/       호출어·긴급어 감지 모델 (자체 학습)
├── assets/       미리 만든 안내 음성
├── config/       목적지 예시, Ollama 모델 파일
├── scripts/      음성 굽기, 지연 측정 등 도구
├── tools/        마이크 점검, STT 비교 같은 현장 시험 도구
├── docs/         설계·설치·실험 기록
└── tests/        자동 테스트
```

## 사용 모델

| 용도 | 모델 | 라이선스 |
|---|---|---|
| STT | Qwen3-ASR 1.7B | Apache 2.0 |
| LLM | KT Mi:dm 2.0 Mini Instruct 2.3B | MIT |
| TTS | Supertonic-3, CosyVoice 3 | 각 모델 라이선스를 따름 |
| 호출어·긴급어 | openWakeWord (자체 학습 모델) | |

## 관련 저장소

- [26_HC021](https://github.com/VICA-smarthandle/26_HC021) : 프로젝트 소개·문서
- [vica_ros2_ws](https://github.com/VICA-smarthandle/vica_ros2_ws) : 자율주행·미션 매니저·안전 제어
- [VICA_Supervisor](https://github.com/VICA-smarthandle/VICA_Supervisor) : 관리자 앱
