# 소프트웨어 에코 제거 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** reSpeaker ch0 에서 로봇 말소리를 speexdsp 로 약 9 dB 줄이는 단계를 `process_frame` 앞에 넣고(`.env` 스위치 `off|log|on`, 기본 `off`), 켰을 때와 껐을 때를 같은 녹음으로 견주는 도구를 만든다.

**Architecture:** 새 모듈 `src/echo_cancel.py` 가 `libspeexdsp.so.1` 을 ctypes 로 부른다(기준 ch5 를 100 ms 늦춤, 필터 256 ms). `WakewordMonitor.run()` 의 콜백은 ch0·ch5 를 함께 큐에 넣고, 감시 루프가 `process_frame` 직전에 `EchoStage.feed` 를 거친다. 실패하면 원래 ch0 을 그대로 넘긴다. 검증은 6채널 녹음을 `off`·`on` 두 길로 운영 감시기에 다시 흘려 사건 수·시각을 견준다.

**Tech Stack:** Python 3.10 · numpy · ctypes + `libspeexdsp.so.1`(설치돼 있음, 새 의존성 없음) · pytest · soundfile/sounddevice(도구만)

**Spec:** `docs/superpowers/specs/2026-11-18-software-echo-cancel-design.md`

## Global Constraints

- 작업 자리는 `~/wt-aec-voice`(브랜치 `feat/sw-aec`)뿐이다. 운영 폴더 `~/VICA-smarthandle/vica-voice-llm` 은 건드리지 않는다. 푸시하지 않는다.
- systemd 유닛(`vica-*`)을 켜거나 끄지 않는다. 시험은 소리 장치를 열지 않는다.
- 시험 명령: `cd ~/wt-aec-voice && .venv/bin/python -m pytest -q -p no:cacheprovider <파일>`. 기준 상태 = `1700 passed, 16 skipped`.
- `git add` 는 파일을 지정한다(`.venv` 심볼릭 링크가 미추적으로 보인다 — `git add -A` 금지).
- 커밋 메시지는 한국어 `feat(voice): …` 꼴, 끝에 `Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>`.
- 값(설계 4.1): 블록 1280 표본(80 ms) · speex 프레임 160 · 필터 4096 표본(256 ms) · 기준 늦춤 1600 표본(100 ms) · 16000 Hz · int16.
- 스위치(설계 4.2): `VICA_SW_AEC=off|log|on` 기본 `off` · `VICA_SW_AEC_DELAY_MS` 기본 100 · `VICA_SW_AEC_TAIL_MS` 기본 256.
- **`off` 는 지금과 한 비트도 다르지 않다**: 모듈을 만들지 않고, 큐에는 1차원 ch0 만 들어가고, `clips.jsonl` 에 칸을 더하지 않는다.
- **죽어도 소리는 흐른다**(설계 4.3): 에코 제거의 어떤 실패도 `process_frame` 에 원래 ch0 이 가는 것을 막지 못한다.
- 건드리지 않는 것(설계 2절): 칩 설정 코드(`src/dsp_state.py`), 점검기(`tools/mic_venue_check.py`·`src/mic_check_*.py` — 가져다 쓰기만), 문턱값, 자기 목소리 거름·마이크 닫기·메아리 기각.
- 주석·로그·경고 문장은 한국어, 한 줄 한 뜻.

## Review Focus

1. 받아쓰기로 감시 루프가 몇 초 막혔다가 쌓인 블록이 한꺼번에 풀릴 때 — 한 블록도 건너뛰지 않고 순서대로 에코 제거를 거친다(Task 4).
2. 로봇이 오래 말하지 않을 때(ch5 가 계속 0) — 출력이 입력과 1 dB 안에서 같고 값이 넘치지 않는다(Task 1).
3. 최대 크기 입력(±32767)과 메모리가 이어지지 않은 배열(2열 배열의 한 열) — 오류 없이 int16 1280 개를 돌려준다(Task 1).
4. `.env` 값이 `ON`, ` on `, `1`, 빈 값, 숫자 아닌 지연값일 때 — 대소문자·공백은 받아 주고, 그 밖은 `off`·기본값으로 돌며 경고 한 줄(Task 3).
5. 로봇이 말하는 한가운데서 처리 예외가 날 때 — 그 블록부터 원래 소리, 경고 한 번, 그 실행 동안 다시 켜지지 않는다(Task 3).

---

## File Structure

| 파일 | 책임 |
|---|---|
| `src/echo_cancel.py` (새) | `EchoCanceller`(speexdsp 감싸개 + 기준 지연 버퍼) · `EchoConfig`·`echo_config_from_env` · `EchoStage`(모드·실패 처리) · `build_echo_stage` · `clip_info_with_aec` |
| `src/echo_meter.py` (새) | `EchoMeter` — 로봇 말 한 덩어리의 줄임 dB 기록(`~/vica_data/aec_log/YYYYMMDD.jsonl`)과 건강 경고 |
| `src/wakeword_monitor.py` (고침) | `echo_stage` 속성 · `_queue_item` · `_frame_from_queue` · `run()` 연결 |
| `src/ros_wakeword_node.py` (고침) | 스위치 읽어 붙이기 · 소리 저장 줄에 `aec` 칸 |
| `.env.example` · `tests/conftest.py` (고침) | 키 안내 · 시험 기본값 `VICA_SW_AEC=off` |
| `tools/aec_record.py` (새) | 검증용 6채널 녹음 묶음(로봇만·사람만·겹침)과 시각표 json |
| `src/aec_replay.py` (새) · `tools/aec_replay.py` (새) | 녹음을 `off`·`on` 으로 운영 감시기에 다시 흘려 판정 |
| `tests/test_sw_aec_canceller.py` · `test_sw_aec_meter.py` · `test_sw_aec_stage.py` · `test_sw_aec_wiring.py` · `test_sw_aec_node_wiring.py` · `test_aec_record_tool.py` · `test_aec_replay.py` (새) | 위 각각의 시험 |

---

### Task 1: `EchoCanceller` — speexdsp 감싸개와 기준 지연

**Files:**
- Create: `src/echo_cancel.py`
- Test: `tests/test_sw_aec_canceller.py`

**Interfaces:**
- Produces:
  ```python
  BLOCK = 1280          # wakeword_monitor.FRAME 과 같다
  SPEEX_FRAME = 160
  class EchoCancelUnavailable(RuntimeError): ...
  class EchoCanceller:
      def __init__(self, delay_ms: int = 100, tail_ms: int = 256, rate: int = 16000,
                   lib_name: str = "libspeexdsp.so.1") -> None      # 못 열거나 초기화 실패면 EchoCancelUnavailable
      def process(self, mic: np.ndarray, ref: np.ndarray) -> np.ndarray   # int16 1차원, 같은 길이. 길이가 160 의 배수가 아니거나 서로 다르면 ValueError
      def close(self) -> None
  ```

- [ ] **Step 1: 실패하는 시험을 쓴다** — `tests/test_sw_aec_canceller.py`

```python
import ctypes
import numpy as np
import pytest
from src.echo_cancel import BLOCK, EchoCancelUnavailable, EchoCanceller

SR = 16000

def _has_lib() -> bool:
    try:
        ctypes.CDLL("libspeexdsp.so.1")
        return True
    except OSError:
        return False

needs_lib = pytest.mark.skipif(not _has_lib(), reason="libspeexdsp.so.1 없음")

def _noise(sec, seed, amp=6000):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(sec * SR)) * amp).clip(-30000, 30000).astype(np.int16)

def _echo_of(ref, delay_ms=118, gain=0.3):          # 118 ms 늦은 짧은 방 응답(32 ms)
    h = np.zeros(512); h[0], h[40], h[200] = gain, gain * 0.5, gain * 0.25
    y = np.convolve(np.concatenate([np.zeros(delay_ms * 16), ref.astype(np.float64)]), h)[:len(ref)]
    return y.astype(np.int16)

def _run(ec, mic, ref, block=BLOCK):
    out = np.zeros_like(mic)
    for i in range(0, len(mic) - block + 1, block):
        out[i:i + block] = ec.process(mic[i:i + block], ref[i:i + block])
    return out

def _db(a, b):
    r = lambda v: np.sqrt(np.mean(v.astype(np.float64) ** 2)) + 1e-9
    return 20 * np.log10(r(a) / r(b))

@needs_lib
def test_removes_echo_delayed_118ms():
    ref = _noise(8, 1); mic = _echo_of(ref)
    out = _run(EchoCanceller(), mic, ref)
    assert _db(out[5 * SR:], mic[5 * SR:]) <= -6.0           # 설계 5.1 — 5 s 뒤 6 dB 이상

@needs_lib
def test_quiet_reference_keeps_input_within_1db_for_20s():   # Review Focus 2
    mic = _noise(20, 2, amp=3000)
    out = _run(EchoCanceller(), mic, np.zeros_like(mic))
    assert abs(_db(out, mic)) <= 1.0
    assert abs(_db(out[-SR:], mic[-SR:])) <= 1.0

@needs_lib
def test_block_size_does_not_change_result():               # 지연 버퍼가 블록 경계를 넘어 이어진다
    ref = _noise(2, 3); mic = _echo_of(ref)
    assert np.array_equal(_run(EchoCanceller(), mic, ref, 1280), _run(EchoCanceller(), mic, ref, 160))

@needs_lib
def test_full_scale_and_non_contiguous_input():             # Review Focus 3
    pair = np.stack([np.full(BLOCK, 32767, np.int16), np.full(BLOCK, -32768, np.int16)], axis=1)
    out = EchoCanceller().process(pair[:, 0], pair[:, 1])
    assert out.dtype == np.int16 and out.shape == (BLOCK,)

@needs_lib
def test_rejects_bad_lengths():
    ec = EchoCanceller()
    with pytest.raises(ValueError):
        ec.process(np.zeros(1000, np.int16), np.zeros(1000, np.int16))
    with pytest.raises(ValueError):
        ec.process(np.zeros(1280, np.int16), np.zeros(160, np.int16))

def test_missing_library_raises_unavailable():
    with pytest.raises(EchoCancelUnavailable):
        EchoCanceller(lib_name="libnope-speexdsp.so.9")
```

- [ ] **Step 2: 실패 확인** — `.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_sw_aec_canceller.py` → `ModuleNotFoundError: src.echo_cancel`

- [ ] **Step 3: `src/echo_cancel.py` 에 `EchoCanceller` 를 만든다.** 이 감싸개가 알아야 할 C 쪽 사실:
  - `speex_echo_state_init(frame_size: int, filter_length: int) -> void*` — `restype = ctypes.c_void_p` 를 꼭 준다(안 주면 64비트 포인터가 잘린다). 값: `(160, tail_ms * rate // 1000)`.
  - `speex_echo_ctl(st, 24, byref(c_int(rate)))` — 24 = `SPEEX_ECHO_SET_SAMPLING_RATE`.
  - `speex_echo_cancellation(st, rec: int16*, play: int16*, out: int16*)` — 160 표본씩. `rec` = 마이크, `play` = **늦춘** 기준.
  - `speex_echo_state_destroy(st)` — `close()` 와 `__del__` 에서 한 번만.
  - 모든 함수에 `argtypes` 를 준다. 입력은 `np.ascontiguousarray(…, dtype=np.int16)` 로 받는다.
  - 기준 지연: 길이 `delay_ms * rate // 1000` 의 버퍼를 0 으로 시작해, 블록마다 `concatenate([버퍼, ref])` 의 앞 `len(ref)` 를 쓰고 나머지를 새 버퍼로 둔다.

- [ ] **Step 4: 통과 확인** — 같은 명령 → `6 passed`

- [ ] **Step 5: 커밋** — `git add src/echo_cancel.py tests/test_sw_aec_canceller.py` · `feat(voice): speexdsp 에코 제거 감싸개 — 기준 ch5 를 100 ms 늦춰 256 ms 필터로(설계 4.1)`

---

### Task 2: `EchoMeter` — 줄임량 기록과 건강 경고

**Files:**
- Create: `src/echo_meter.py`
- Test: `tests/test_sw_aec_meter.py`

**Interfaces:**
- Produces:
  ```python
  AEC_LOG_DIR = "~/vica_data/aec_log"
  REF_ACTIVE_RMS = 50.0      # int16 눈금 — ch5 가 이보다 크면 로봇이 말하는 블록
  HANG_SEC = 0.5             # 이만큼 조용하면 한 덩어리 끝
  MIN_SEC = 1.0              # 이보다 짧은 덩어리는 기록하지 않는다
  WEAK_DB, WEAK_MIN_SEC, WEAK_RUN = -3.0, 3.0, 5
  WEAK_WARNING = "에코 제거가 듣지 않음 — 최근 5번 연속 줄임 3 dB 미만(스피커 지연이 바뀌었을 수 있음)"
  class EchoMeter:
      def __init__(self, mode: str, log_dir: Optional[Path], warn: Callable[[str], None],
                   block_sec: float = 0.08, wall: Callable[[], float] = time.time) -> None
      def update(self, mic: np.ndarray, out: np.ndarray, ref: np.ndarray) -> Optional[dict]
          # 덩어리가 방금 끝났으면 그 줄(dict)을 돌려주고 파일에도 한 줄 쓴다. 아니면 None.
  ```
- 줄의 칸: `time`(ISO, 밀리초) · `sec`(말한 블록 수 × 0.08) · `rms_in` · `rms_out`(말한 블록들의 rms, 0~1 눈금) · `reduce_db`(소수 2자리) · `mode`.

- [ ] **Step 1: 실패하는 시험을 쓴다** — `tests/test_sw_aec_meter.py`

```python
import json
import numpy as np
from src.echo_meter import EchoMeter, WEAK_WARNING

B = 1280
def blk(v): return np.full(B, v, np.int16)

def feed(m, n, mic, out, ref):
    return [r for _ in range(n) if (r := m.update(blk(mic), blk(out), blk(ref))) is not None]

def test_one_row_when_robot_chunk_ends(tmp_path):
    m = EchoMeter("log", tmp_path, lambda s: None, wall=lambda: 1795000000.0)
    assert feed(m, 25, 4000, 1000, 5000) == []            # 2.0 s 말하는 중
    (row,) = feed(m, 8, 10, 10, 0)                        # 0.64 s 조용 → 끝
    assert row["mode"] == "log" and abs(row["sec"] - 2.0) < 0.01
    assert abs(row["reduce_db"] - (-12.04)) < 0.05        # 20·log10(1000/4000)
    assert abs(row["rms_in"] - 4000 / 32768) < 1e-4
    (f,) = tmp_path.glob("*.jsonl")
    assert len(f.stem) == 8 and json.loads(f.read_text().splitlines()[0]) == row

def test_short_chunk_is_not_logged(tmp_path):
    m = EchoMeter("on", tmp_path, lambda s: None)
    assert feed(m, 10, 4000, 1000, 5000) + feed(m, 8, 10, 10, 0) == []   # 0.8 s
    assert list(tmp_path.glob("*.jsonl")) == []

def test_gap_shorter_than_hang_joins_chunks(tmp_path):
    m = EchoMeter("on", None, lambda s: None)
    rows = feed(m, 15, 4000, 1000, 5000) + feed(m, 5, 10, 10, 0) + feed(m, 15, 4000, 1000, 5000) + feed(m, 8, 10, 10, 0)
    assert len(rows) == 1 and abs(rows[0]["sec"] - 2.4) < 0.01           # 조용한 0.4 s 는 길이에 안 든다

def test_weak_warning_once_then_rearms_after_recovery():
    warns = []
    m = EchoMeter("on", None, warns.append)
    def chunk(out): feed(m, 40, 4000, out, 5000); feed(m, 8, 10, 10, 0)  # 3.2 s
    for _ in range(4): chunk(3900)
    assert warns == []
    chunk(3900); chunk(3900)
    assert warns == [WEAK_WARNING]                        # 5번째에 한 번, 6번째엔 다시 내지 않는다
    chunk(1000)                                           # 회복
    for _ in range(5): chunk(3900)
    assert warns == [WEAK_WARNING, WEAK_WARNING]

def test_chunks_under_3s_do_not_count_as_weak():
    warns = []
    m = EchoMeter("on", None, warns.append)
    for _ in range(8): feed(m, 25, 4000, 3900, 5000); feed(m, 8, 10, 10, 0)   # 2.0 s 짜리
    assert warns == []

def test_unwritable_log_dir_does_not_raise(tmp_path):
    notdir = tmp_path / "file"; notdir.write_text("x")
    warns = []
    m = EchoMeter("on", notdir, warns.append)
    for _ in range(2): feed(m, 25, 4000, 1000, 5000); rows = feed(m, 8, 10, 10, 0)
    assert len(rows) == 1 and len(warns) == 1             # 줄은 돌려주고, 저장 실패 경고는 한 번
```

- [ ] **Step 2: 실패 확인** — `…pytest… tests/test_sw_aec_meter.py` → `ModuleNotFoundError: src.echo_meter`
- [ ] **Step 3: `src/echo_meter.py` 에 위 Interfaces 대로 만든다.** 파일 이름은 `wall()` 의 날짜 `%Y%m%d.jsonl`, 폴더는 처음 쓸 때 만든다. 저장 실패 경고 문장: `f"에코 제거 기록 저장 실패(이후 경고 생략): {exc}"`.
- [ ] **Step 4: 통과 확인** → `6 passed`
- [ ] **Step 5: 커밋** — `git add src/echo_meter.py tests/test_sw_aec_meter.py` · `feat(voice): 에코 제거 줄임량 기록·건강 경고(설계 4.4)`

---

### Task 3: 스위치와 `EchoStage` — 모드·실패 처리

**Files:**
- Modify: `src/echo_cancel.py`
- Test: `tests/test_sw_aec_stage.py`

**Interfaces:**
- Consumes: `EchoCanceller`, `EchoCancelUnavailable`, `BLOCK`(Task 1) · `EchoMeter`, `AEC_LOG_DIR`(Task 2)
- Produces:
  ```python
  @dataclass(frozen=True)
  class EchoConfig:
      mode: str = "off"; delay_ms: int = 100; tail_ms: int = 256; warnings: tuple = ()
  def echo_config_from_env(env: Mapping[str, str]) -> EchoConfig
  class EchoStage:
      def __init__(self, config: EchoConfig, warn: Callable[[str], None], meter: Optional[EchoMeter] = None,
                   factory: Callable[..., EchoCanceller] = EchoCanceller) -> None
      @property
      def mode(self) -> str        # 지금 실제로 도는 모드. 만들기 실패·처리 예외 뒤에는 "off"
      def feed(self, ch0: np.ndarray, ch5: np.ndarray) -> np.ndarray   # process_frame 에 넘길 조각
  def build_echo_stage(env: Mapping[str, str], info: Callable[[str], None], warn: Callable[[str], None],
                       log_dir: Optional[Path] = None, factory=EchoCanceller) -> Optional[EchoStage]
  def clip_info_with_aec(info: dict, mode: str) -> dict
  ```
- 규칙:
  - `VICA_SW_AEC`: 앞뒤 공백을 떼고 소문자로. 없거나 빈 값 = `off`(경고 없음). `off|log|on` 이 아니면 `off` + 경고 `f"VICA_SW_AEC={raw!r} 는 off|log|on 이 아님 — off 로 돈다"`.
  - `VICA_SW_AEC_DELAY_MS`: 정수 0~400, 아니면 100 + 경고. `VICA_SW_AEC_TAIL_MS`: 정수 32~512, 아니면 256 + 경고. 경고 꼴 `f"{key}={raw!r} 를 못 읽음 — 기본값 {default} 로 돈다"`.
  - 감싸개는 `factory(delay_ms=…, tail_ms=…)` 처럼 이름 붙인 인자로 만든다.
  - `feed`: `on` 은 처리한 소리, `log` 는 **받은 `ch0` 그 객체**를 돌려준다(처리와 기록은 돈다). 죽은 뒤에는 `ch0`.
  - 길이가 `BLOCK` 이 아니거나 두 입력 길이가 다르면: `ch0` 을 돌려주고 `factory` 로 상태를 새로 만든다(새로 만들기도 실패하면 죽음).
  - `factory`·`process` 의 예외: 경고 `f"소프트웨어 에코 제거 중단 — 원래 소리로 계속 감시: {exc}"` 한 번, 그 뒤 `mode == "off"`, 다시 켜지지 않는다.
  - `build_echo_stage`: 설정 경고를 `warn` 으로 흘린다. `off` 면 `factory` 를 부르지 않고 `None`. 만들었는데 `mode == "off"` 면 `None`. 켜졌으면 `info(f"소프트웨어 에코 제거: {mode} (기준 늦춤 {delay_ms} ms · 필터 {tail_ms} ms)")`. `log_dir` 기본은 `Path(AEC_LOG_DIR).expanduser()`.
  - `clip_info_with_aec`: `mode == "off"` 면 받은 `info` 그대로(같은 객체), 아니면 `{**info, "aec": mode}`.

- [ ] **Step 1: 실패하는 시험을 쓴다** — `tests/test_sw_aec_stage.py`

```python
import numpy as np
import pytest
from src.echo_cancel import (BLOCK, EchoCancelUnavailable, EchoConfig, EchoStage, build_echo_stage,
                             clip_info_with_aec, echo_config_from_env)

class FakeEc:
    made = 0
    def __init__(self, delay_ms=100, tail_ms=256, **_):
        FakeEc.made += 1; self.calls = 0; self.boom = False; self.args = (delay_ms, tail_ms)
    def process(self, mic, ref):
        self.calls += 1
        if self.boom: raise RuntimeError("x")
        return (mic // 2).astype(np.int16)
    def close(self): pass

@pytest.fixture(autouse=True)
def _reset(): FakeEc.made = 0

CH0 = np.full(BLOCK, 1000, np.int16); CH5 = np.full(BLOCK, 3000, np.int16)

@pytest.mark.parametrize("raw,mode,warned", [(None, "off", False), ("", "off", False), ("on", "on", False),
    ("ON", "on", False), (" log ", "log", False), ("1", "off", True), ("true", "off", True)])
def test_mode_from_env(raw, mode, warned):                 # Review Focus 4
    cfg = echo_config_from_env({} if raw is None else {"VICA_SW_AEC": raw})
    assert cfg.mode == mode and bool(cfg.warnings) == warned

def test_numbers_from_env():
    cfg = echo_config_from_env({"VICA_SW_AEC": "on", "VICA_SW_AEC_DELAY_MS": "80", "VICA_SW_AEC_TAIL_MS": "128"})
    assert (cfg.delay_ms, cfg.tail_ms, cfg.warnings) == (80, 128, ())
    bad = echo_config_from_env({"VICA_SW_AEC": "on", "VICA_SW_AEC_DELAY_MS": "abc", "VICA_SW_AEC_TAIL_MS": "9999"})
    assert (bad.delay_ms, bad.tail_ms) == (100, 256) and len(bad.warnings) == 2

def test_on_returns_processed_and_log_returns_same_object():
    on = EchoStage(EchoConfig("on"), lambda s: None, factory=FakeEc)
    assert np.array_equal(on.feed(CH0, CH5), CH0 // 2)
    log = EchoStage(EchoConfig("log"), lambda s: None, factory=FakeEc)
    assert log.feed(CH0, CH5) is CH0 and log.mode == "log"

def test_exception_mid_speech_falls_back_for_good():      # Review Focus 5
    warns = []
    st = EchoStage(EchoConfig("on"), warns.append, factory=FakeEc)
    st.feed(CH0, CH5); ec = st._ec; ec.boom = True
    assert st.feed(CH0, CH5) is CH0 and st.mode == "off" and len(warns) == 1
    ec.boom = False
    assert st.feed(CH0, CH5) is CH0 and ec.calls == 2 and len(warns) == 1   # 다시 부르지 않는다

def test_short_block_passes_through_and_rebuilds_state():
    st = EchoStage(EchoConfig("on"), lambda s: None, factory=FakeEc)
    short = np.zeros(640, np.int16)
    assert st.feed(short, short) is short and FakeEc.made == 2 and st.mode == "on"
    assert np.array_equal(st.feed(CH0, CH5), CH0 // 2)

def test_unavailable_library_means_no_stage():
    def nope(**_): raise EchoCancelUnavailable("없음")
    infos, warns = [], []
    assert build_echo_stage({"VICA_SW_AEC": "on"}, infos.append, warns.append, log_dir=None, factory=nope) is None
    assert infos == [] and len(warns) == 1

def test_build_off_never_touches_the_library(tmp_path):
    assert build_echo_stage({}, lambda s: None, lambda s: None, log_dir=tmp_path, factory=FakeEc) is None
    assert FakeEc.made == 0

def test_build_log_announces_and_passes_numbers(tmp_path):
    infos = []
    st = build_echo_stage({"VICA_SW_AEC": "log", "VICA_SW_AEC_DELAY_MS": "90"}, infos.append, lambda s: None,
                          log_dir=tmp_path, factory=FakeEc)
    assert st.mode == "log" and st._ec.args == (90, 256)
    assert infos == ["소프트웨어 에코 제거: log (기준 늦춤 90 ms · 필터 256 ms)"]

def test_clip_info():
    info = {"result": "wake"}
    assert clip_info_with_aec(info, "off") is info
    assert clip_info_with_aec(info, "on") == {"result": "wake", "aec": "on"} and info == {"result": "wake"}
```

(시험이 `st._ec` 를 본다 — 살아 있는 감싸개를 담는 속성 이름은 `_ec` 로 한다.)

- [ ] **Step 2: 실패 확인** → `ImportError: cannot import name 'EchoConfig'`
- [ ] **Step 3: `src/echo_cancel.py` 에 위 Interfaces·규칙대로 더한다.** `EchoStage.feed` 는 `meter` 가 있으면 처리 성공 때마다 `meter.update(ch0, out, ch5)` 를 부른다(`log`·`on` 모두).
- [ ] **Step 4: 통과 확인** → `tests/test_sw_aec_stage.py` 전부 통과
- [ ] **Step 5: 커밋** — `git add src/echo_cancel.py tests/test_sw_aec_stage.py` · `feat(voice): 에코 제거 스위치(off|log|on)와 실패해도 원래 소리가 흐르는 단계(설계 4.2·4.3)`

---

### Task 4: 감시기 연결 — `process_frame` 앞에 넣기

**Files:**
- Modify: `src/wakeword_monitor.py` — `self.patrol_tap = None` 줄(약 650행) 아래, `run()`(약 1612~1694행)
- Test: `tests/test_sw_aec_wiring.py`

**Interfaces:**
- Consumes: `EchoStage.feed(ch0, ch5) -> np.ndarray`(Task 3)
- Produces:
  ```python
  def _queue_item(block: np.ndarray, with_ref: bool) -> np.ndarray      # 모듈 함수. block = (N, 6) int16
      # with_ref False → block[:, 0].copy() (지금과 같음) / True → block[:, (0, 5)].copy()
  class WakewordMonitor:
      echo_stage = None                                                  # __init__ 에서 None. 노드가 붙인다
      def _frame_from_queue(self, item: np.ndarray) -> np.ndarray
          # 1차원이면 그대로(같은 객체). 2열이면 echo_stage.feed(item[:, 0], item[:, 1]) — echo_stage 가 None 이면 item[:, 0]
  ```

- [ ] **Step 1: 실패하는 시험을 쓴다** — `tests/test_sw_aec_wiring.py`

```python
import ast
from pathlib import Path
import numpy as np
from src.wakeword_monitor import FRAME, WakewordMonitor, _queue_item

SRC = Path(__file__).resolve().parents[1] / "src" / "wakeword_monitor.py"

class Stage:
    def __init__(self): self.seen = []
    def feed(self, ch0, ch5):
        self.seen.append((int(ch0[0]), int(ch5[0]))); return np.full(len(ch0), 7, np.int16)

def monitor():
    return WakewordMonitor(on_emergency=lambda e: None, on_user_text=lambda t: None,
                           predict=lambda f: {"a": 0.0, "b": 0.0}, transcribe=lambda a: "")

def block6(v0, v5):
    b = np.zeros((FRAME, 6), np.int16); b[:, 0] = v0; b[:, 5] = v5; return b

def test_queue_item_off_is_plain_ch0():
    b = block6(11, 55)
    item = _queue_item(b, False)
    assert item.ndim == 1 and np.array_equal(item, b[:, 0]) and item.base is None

def test_queue_item_with_ref_carries_ch0_and_ch5():
    item = _queue_item(block6(11, 55), True)
    assert item.shape == (FRAME, 2) and item[0, 0] == 11 and item[0, 1] == 55

def test_plain_frame_passes_untouched():
    m = monitor(); f = np.zeros(FRAME, np.int16)
    assert m.echo_stage is None and m._frame_from_queue(f) is f

def test_pair_goes_through_stage_even_while_muted_and_in_order():   # 설계 4.1 · Review Focus 1
    m = monitor(); m.echo_stage = Stage(); m.set_muted(True, now=0.0)
    for i in range(40):                                              # 밀린 블록 40개가 한꺼번에 풀린다
        f = m._frame_from_queue(_queue_item(block6(i, 100 + i), True))
        assert f[0] == 7
        m.process_frame(f, now=i * 0.08)
    assert m.echo_stage.seen == [(i, 100 + i) for i in range(40)]

def test_pair_without_stage_falls_back_to_ch0():
    m = monitor()
    assert np.array_equal(m._frame_from_queue(_queue_item(block6(11, 55), True)), np.full(FRAME, 11, np.int16))

def test_run_uses_both_helpers_before_process_frame():
    tree = ast.parse(SRC.read_text())
    run = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "run")
    calls = lambda name: [n for n in ast.walk(run) if isinstance(n, ast.Call) and ast.unparse(n.func) == name]
    assert len(calls("_queue_item")) == 1 and len(calls("self._frame_from_queue")) == 1
    assert calls("self._frame_from_queue")[0].lineno < calls("self.process_frame")[0].lineno
```

- [ ] **Step 2: 실패 확인** → `ImportError: cannot import name '_queue_item'`
- [ ] **Step 3: `src/wakeword_monitor.py` 를 고친다.**
  - `self.patrol_tap = None` 아래: `self.echo_stage = None` (주석: 소프트웨어 에코 제거(2026-11-18, echo_cancel.EchoStage). 노드가 붙인다 — None 이면 지금 그대로).
  - 모듈 함수 `_queue_item`, 메서드 `_frame_from_queue` 를 위 Interfaces 대로.
  - `run()`: 스트림을 열기 전에 `with_ref = self.echo_stage is not None` 를 한 번 정한다. 콜백의 `q.put(block[:, 0].copy())` → `q.put(_queue_item(block, with_ref))`. 루프의 `frame = q.get(timeout=0.5)` → `frame = self._frame_from_queue(q.get(timeout=0.5))`. `tap.push(block)`·`FrameWatch`·칩 VAD 조회는 그대로 둔다.
- [ ] **Step 4: 통과 확인** — `tests/test_sw_aec_wiring.py` 통과, 그리고 `…pytest… tests/test_wakeword_monitor.py tests/test_wake_clip.py` 가 기준과 같이 통과
- [ ] **Step 5: 커밋** — `git add src/wakeword_monitor.py tests/test_sw_aec_wiring.py` · `feat(voice): 감시 루프가 process_frame 앞에서 에코 제거를 거친다 — 꺼져 있으면 큐·조각이 지금과 같다(설계 4.1)`

---

### Task 5: 노드 연결 — 스위치 읽기와 저장 줄의 `aec` 칸

**Files:**
- Modify: `src/ros_wakeword_node.py` — 칩 에코 억제 설정 블록 뒤·`PatrolSoundTap` 블록 앞(약 318행), `_save_estop_clip`·`_save_listen_clip`·`_save_wake_clip`(약 537~565행)
- Modify: `.env.example`, `tests/conftest.py`
- Test: `tests/test_sw_aec_node_wiring.py`

**Interfaces:**
- Consumes: `build_echo_stage(env, info, warn) -> Optional[EchoStage]`, `clip_info_with_aec(info, mode) -> dict`(Task 3) · `WakewordMonitor.echo_stage`(Task 4)

- [ ] **Step 1: 실패하는 시험을 쓴다** — `tests/test_sw_aec_node_wiring.py` (노드는 rclpy 없이 못 띄우므로 `tests/test_one_voice_wiring.py` 처럼 소스 구조로 본다)

```python
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TREE = ast.parse((ROOT / "src" / "ros_wakeword_node.py").read_text())

def _method(name):
    return next(n for n in ast.walk(TREE) if isinstance(n, ast.FunctionDef) and n.name == name)

def _calls(node, dotted):
    return [n for n in ast.walk(node) if isinstance(n, ast.Call) and ast.unparse(n.func) == dotted]

def test_stage_is_built_from_env_and_attached_before_the_mic_thread():
    init = _method("__init__")
    (build,) = _calls(init, "build_echo_stage")
    assert ast.unparse(build.args[0]) == "os.environ"
    attach = [n for n in ast.walk(init) if isinstance(n, ast.Assign)
              and ast.unparse(n.targets[0]) == "self._monitor.echo_stage"]
    assert len(attach) == 1
    (thread,) = _calls(init, "threading.Thread")
    assert build.lineno < attach[0].lineno < thread.lineno

def test_every_clip_row_goes_through_clip_info_with_aec():
    for name in ("_save_estop_clip", "_save_listen_clip", "_save_wake_clip"):
        assert len(_calls(_method(name), "clip_info_with_aec")) == 1, name

def test_env_example_and_test_defaults_name_the_switch():
    example = (ROOT / ".env.example").read_text()
    for key in ("VICA_SW_AEC=", "VICA_SW_AEC_DELAY_MS=", "VICA_SW_AEC_TAIL_MS="):
        assert key in example, key
    assert '"VICA_SW_AEC": "off"' in (ROOT / "tests" / "conftest.py").read_text()
```

- [ ] **Step 2: 실패 확인** → 세 시험 실패
- [ ] **Step 3: 고친다.**
  - `ros_wakeword_node.py`: `from .echo_cancel import build_echo_stage, clip_info_with_aec`. 칩 에코 억제 블록 뒤에
    `self._echo_stage = build_echo_stage(os.environ, self.get_logger().info, self.get_logger().warning)` 와
    `self._monitor.echo_stage = self._echo_stage   # 감시 스레드를 띄우기 전에 붙인다`.
  - 세 저장 함수: `save_emergency_clip` 에 넘기는 `info` 를 `clip_info_with_aec(<지금 넘기던 dict>, self._echo_stage.mode if self._echo_stage is not None else "off")` 로 감싼다(모드 식은 작은 메서드 `_aec_mode()` 로 빼도 된다).
  - `.env.example` 의 마이크 칩 키(`VICA_MIC_NLATTEN`) 근처에 주석 한 줄씩과 함께 세 키를 더한다(값: `off`, `100`, `256`, 주석 처리된 꼴은 그 파일의 기존 관례를 따른다).
  - `tests/conftest.py` 의 `_DEFAULTS` 에 `"VICA_SW_AEC": "off",` (운영 `.env` 가 `on` 이어도 시험은 끈 채로 돈다).
- [ ] **Step 4: 통과 확인** — `tests/test_sw_aec_node_wiring.py` 통과 · `.venv/bin/python -c "import ast,sys; ast.parse(open('src/ros_wakeword_node.py').read())"` 오류 없음
- [ ] **Step 5: 커밋** — `git add src/ros_wakeword_node.py .env.example tests/conftest.py tests/test_sw_aec_node_wiring.py` · `feat(voice): 음성 노드가 VICA_SW_AEC 로 에코 제거를 붙이고 저장 줄에 aec 칸을 남긴다(설계 4.2·4.4)`

---

### Task 6: `tools/aec_record.py` — 검증용 녹음 묶음

**Files:**
- Create: `tools/aec_record.py`
- Test: `tests/test_aec_record_tool.py`

**Interfaces:**
- Consumes: `ROBOT_CLIPS`, `load_clips`, `voice_stack_running`(`tools/mic_venue_check.py` — 가져다 쓰기만) · `src.audio_out.play` · `src.dsp_state` 의 `agc_desired_from_env`·`apply_agc_desired_level`·`agc_max_gain_from_env`·`apply_agc_max_gain`·`echo_tuning_from_env`·`apply_echo_tuning`
- Produces:
  ```python
  OVERLAP_CLIPS = ("mission_msg_approach_question.wav", "mission_msg_estopped.wav",
                   "mission_msg_paused.wav", "mission_msg_estop_reject.wav")
  LEAD_SEC, GAP_SEC, USER_WINDOW_SEC, OVERLAP_TAIL_SEC = 2.0, 0.3, 4.0, 2.0
  @dataclass(frozen=True)
  class Step:
      kind: str                 # "robot_only" | "user_only" | "overlap" | "cue"
      clip: Optional[str]; say: Optional[str]; expect: Optional[str]; item_id: Optional[str]
  def build_script() -> list[Step]
  def session_dict(started: str, wav_name: str, chip: dict, ments: list[dict], items: list[dict]) -> dict
  def main(argv: Optional[Sequence[str]] = None) -> int     # --out(기본 ~/vica_data/aec_sets/<YYYYmmdd-HHMMSS>) · --countdown(기본 10)
  ```
- 시각표 json(Task 7 이 읽는다, **녹음은 한 번이라 이 꼴을 바꾸지 않는다**):
  ```json
  {"version": 1, "rate": 16000, "channels": 6, "started": "<ISO>", "wav": "set.wav", "chip": {},
   "ments": [{"name": "mission_msg_paused.wav", "text": "잠시 멈추겠습니다. …", "start": 2.01, "end": 5.45}],
   "items": [{"id": "over_stop_1", "kind": "overlap", "say": "멈춰", "expect": "emergency", "start": 70.1, "end": 75.6}]}
  ```
  시각은 녹음 첫 표본부터의 초(콜백이 받은 표본 수 ÷ 16000). `ments` 는 `play` 를 부르기 직전·돌아온 직후.
- 대본(설계 5.3):
  1. `LEAD_SEC` 조용 → **로봇만**: `ROBOT_CLIPS` 8개를 두 바퀴(`GAP_SEC` 간격), 16 걸음 모두 `item_id="robot_only"`(항목 하나: 첫 시작 ~ 마지막 끝 + 0.5).
  2. `cue`(삐 두 번) → **사람만**: 삐 한 번 뒤 `USER_WINDOW_SEC` — `user_wake_1..5`(`say="비카야 화장실 가자"`, `expect="wake"`), `user_stop_1..5`(`say="멈춰"`, `expect="emergency"`).
  3. `cue` → **겹침**: 멘트 하나를 틀고 끝난 뒤 `OVERLAP_TAIL_SEC` — `over_stop_1..10`(`"멈춰"`, `"emergency"`), `over_wake_1..10`(`"비카야"`, `"wake"`). 멘트는 `OVERLAP_CLIPS` 를 차례로 돌려 쓴다. 항목 구간 = 멘트 시작 ~ 멘트 끝 + `OVERLAP_TAIL_SEC`.
  - `OVERLAP_CLIPS` 를 이 넷으로 고른 이유: 3.4 s 이상, 문장에 '비카야'가 없음(있으면 감시기가 그 멘트 동안 호출을 확정하지 않는다), "정지합니다"(말하는 동안 마이크 닫기)가 아님.
- 녹음 전에: 음성 노드가 켜져 있으면 끝낸다(`voice_stack_running`). 칩 값은 노드와 같은 순서로 `.env` 값을 쓴다(목표 레벨 → 최대 증폭 → 에코 억제) — 쓴 값을 `chip` 에 남긴다. 화면에 다음에 할 말을 한 줄씩 찍는다. 삐는 880 Hz 0.15 s, `peak_dbfs=-20.0`.

- [ ] **Step 1: 실패하는 시험을 쓴다** — `tests/test_aec_record_tool.py`

```python
import json
from pathlib import Path
from tools.aec_record import OVERLAP_CLIPS, build_script, session_dict

MANIFEST = json.loads((Path(__file__).resolve().parents[1] / "assets/baked/manifest.json").read_text())

def test_script_matches_the_design_counts():
    steps = build_script()
    by = lambda kind, expect: [s for s in steps if s.kind == kind and s.expect == expect]
    assert len([s for s in steps if s.kind == "robot_only"]) == 16
    assert {s.item_id for s in steps if s.kind == "robot_only"} == {"robot_only"}
    assert [s.item_id for s in by("user_only", "wake")] == [f"user_wake_{i}" for i in range(1, 6)]
    assert [s.item_id for s in by("user_only", "emergency")] == [f"user_stop_{i}" for i in range(1, 6)]
    assert [s.item_id for s in by("overlap", "emergency")] == [f"over_stop_{i}" for i in range(1, 11)]
    assert [s.item_id for s in by("overlap", "wake")] == [f"over_wake_{i}" for i in range(1, 11)]
    assert {s.say for s in by("overlap", "wake")} == {"비카야"} and {s.say for s in by("overlap", "emergency")} == {"멈춰"}

def test_overlap_ments_never_block_the_listener():
    assert "mission_msg_obstacle_stop.wav" not in OVERLAP_CLIPS
    for s in build_script():
        if s.kind == "overlap":
            assert s.clip in OVERLAP_CLIPS and "비카야" not in MANIFEST[s.clip]

def test_session_dict_shape():
    d = session_dict("2026-11-19T10:00:00", "set.wav", {"agc": 0.007}, [{"name": "a.wav", "text": "가", "start": 2.0, "end": 5.0}],
                     [{"id": "over_stop_1", "kind": "overlap", "say": "멈춰", "expect": "emergency", "start": 2.0, "end": 7.0}])
    assert (d["version"], d["rate"], d["channels"], d["wav"]) == (1, 16000, 6, "set.wav")
    assert d["ments"][0]["text"] == "가" and d["items"][0]["expect"] == "emergency"
    json.dumps(d, ensure_ascii=False)
```

- [ ] **Step 2: 실패 확인** → `ModuleNotFoundError: tools.aec_record`
- [ ] **Step 3: `tools/aec_record.py` 를 만든다.** 녹음·재생 부분은 `~/vica_data/mic_checks/20261118-aec/aec_probe.py` 의 `record()` 와 같은 방식이다(6채널 `sd.InputStream` 콜백으로 모으고 `audio_out.play(..., blocking=True)` 로 튼다). 출력: `<out>/set.wav`(6ch PCM_16) · `<out>/set.json`.
- [ ] **Step 4: 통과 확인** → `3 passed` · `.venv/bin/python -m tools.aec_record --help` 가 도움말을 찍고 0 으로 끝난다(장치를 열지 않는다)
- [ ] **Step 5: 커밋** — `git add tools/aec_record.py tests/test_aec_record_tool.py` · `feat(voice): 에코 제거 검증용 6채널 녹음 도구 — 로봇만·사람만·겹침과 멘트 시각표(설계 5.3)`

---

### Task 7: 다시 틀기 — `off`·`on` 을 운영 감시기로 견주기

**Files:**
- Create: `src/aec_replay.py`, `tools/aec_replay.py`
- Test: `tests/test_aec_replay.py`

**Interfaces:**
- Consumes: `EchoStage`, `EchoConfig`, `EchoCanceller`, `BLOCK`(Task 1·3) · `_Run`, `Models`, `DT`, `FLUSH_FRAMES`, `PAD_FRAMES`(`src/mic_check_replay.py` — 가져다 쓰기만) · `MicValues`, `values_from`, `read_env`(`src/env_file.py`) · 시각표 json(Task 6)
- Produces (`src/aec_replay.py`):
  ```python
  REDUCE_MIN_DB, LATE_MAX_SEC, LEVEL_TOL_DB, LEARN_SKIP_SEC, PRE_ROLL_SEC, TTS_TAIL_SEC = 6.0, 0.08, 1.0, 4.0, 1.0, 0.4
  @dataclass(frozen=True)
  class Ment: name: str; text: str; start: float; end: float
  @dataclass(frozen=True)
  class Item: id: str; kind: str; say: Optional[str]; expect: Optional[str]; start: float; end: float
  @dataclass(frozen=True)
  class Session: wav: Path; ments: tuple; items: tuple
  @dataclass(frozen=True)
  class ItemResult:
      id: str; wakes: int; emergencies: int; first_wake: Optional[float]; first_emergency: Optional[float]
      peak_a: float; peak_b: float; texts: tuple; check_text: str; rms: float
  def load_session(json_path: Path) -> Session                     # wav 는 json 과 같은 폴더 기준
  def split_blocks(x6: np.ndarray) -> list                         # (N, 6) int16 → (1280, 6) 블록들, 남는 꼬리는 버림
  def frames_for(blocks: Sequence[np.ndarray], mode: str, config: EchoConfig, factory=EchoCanceller) -> list
      # "off": 블록마다 ch0 / "on": EchoStage 하나로 처음부터 끝까지(운영처럼 한 번 학습을 이어 간다)
  def run_item(models: Models, values: MicValues, frames: Sequence[np.ndarray], item: Item, ments: Sequence[Ment]) -> ItemResult
  def ment_reductions(session: Session, off: Sequence[np.ndarray], on: Sequence[np.ndarray]) -> list
  def judge(session: Session, off: Mapping[str, ItemResult], on: Mapping[str, ItemResult], reductions: Sequence[float]) -> dict
  def format_report(report: dict) -> str
  ```
- `run_item`(설계 5.2 — 항목마다 새 감시기, 점검기의 `_play` 와 같은 틀):
  - `_Run(models, values)` 를 새로 만든다. 조용한 조각 `FLUSH_FRAMES` 개 → `frames[a:b]`(`a = max(0, int((item.start - PRE_ROLL_SEC) / DT))`, `b = min(len(frames), ceil(item.end / DT))`) → 조용한 조각 `PAD_FRAMES` 개. 시각 `t` 는 파일 시각(블록 번호 × `DT`)이고, 앞의 조용한 조각은 `a·DT - FLUSH_FRAMES·DT` 에서 시작한다.
  - 구간 조각을 넣는 동안만 `run.counting = True`. 칩 VAD 는 넣지 않는다(`vads` 는 `None`).
  - 구간과 겹치는 멘트마다: 그 멘트 `start` 를 처음 지난 조각 **앞에서** `run.monitor.note_robot_speech(text, now=t)` → `run.monitor.set_speaking(True, now=t)`. `end + TTS_TAIL_SEC` 를 처음 지난 조각 앞에서 `run.monitor.set_speaking(False, now=t)`(한 번씩).
  - `wakes` = 결과가 `"wake"` 인 조각 수, `first_wake` = 첫 그 조각의 `t`. `emergencies` = `len(run.emergencies)`, `first_emergency` = 그 길이가 처음 늘어난 조각의 `t`. `check_text` = `(run.monitor.last_emergency_check or {}).get("text", "") or ""`. `rms` = 구간 조각 전체의 rms(0~1).
- `ment_reductions`: `robot_only` 항목 안에서 `start >= 항목.start + LEARN_SKIP_SEC` 인 멘트마다 `20·log10(rms(on 구간) / rms(off 구간))`.
- `judge` 가 돌려주는 dict(설계 5.4):
  ```python
  {"overlap_stop": {"n", "off", "on", "late": [id…], "pass"},     # pass = on >= off and not late
   "overlap_wake": {"n", "off", "on", "pass"},                    # pass = on >= off
   "user_only":    {"n", "differs": [id…], "level_db_max", "pass"},
   "robot_only":   {"reduce_db_median", "new_events", "pass"},    # pass = median <= -6.0 and new_events <= 0
   "pass": bool}
  ```
  - 잡힘: `expect == "emergency"` 는 `emergencies >= 1`, `"wake"` 는 `wakes >= 1`. `off`/`on` = 잡힌 항목 수.
  - `late`: 겹침 "멈춰" 가운데 둘 다 잡혔고 `on.first_emergency - off.first_emergency > LATE_MAX_SEC` 인 항목.
  - `differs`: 사람만 항목 가운데 잡힘 여부나 `texts` 가 다른 것. `level_db_max` = `max |20·log10(on.rms / off.rms)|`, `LEVEL_TOL_DB` 이하여야 통과.
  - `new_events` = `robot_only` 항목의 `(on.wakes + on.emergencies) - (off.wakes + off.emergencies)`.
  - 줄임 값이 하나도 없으면 `reduce_db_median` 은 `None` 이고 `robot_only` 는 통과가 아니다. 묶음에 항목이 없으면(`n == 0`) 그 묶음은 통과로 본다.
- `tools/aec_replay.py`: `main(argv)` — 인자 `session`(json 경로) · `--env`(기본 저장소 `.env`) · `--out`(기본 json 옆 `replay.json`) · `--force`. 음성 노드가 켜져 있으면 `--force` 없이는 끝낸다(GPU·모델을 같이 쓴다). 모델은 `tools/mic_venue_check.py` 의 `_load_models` 로 한 번 올리고, 값은 `values_from(read_env(env))`, 지연·필터는 `echo_config_from_env(os.environ)` 의 숫자를 쓴다(모드는 무시). 항목별 `ItemResult` 두 벌과 `judge` 결과를 json 으로 쓰고 `format_report` 를 찍는다. 끝 코드는 `0`(판정 통과) / `1`.

- [ ] **Step 1: 실패하는 시험을 쓴다** — `tests/test_aec_replay.py`

```python
import json
import numpy as np
from src.aec_replay import (Item, ItemResult, Ment, Session, frames_for, judge, load_session, ment_reductions,
                            run_item, split_blocks)
from src.echo_cancel import BLOCK, EchoConfig
from src.env_file import values_from
from src.mic_check_replay import Models

DT = 0.08
def res(id, wakes=0, emer=0, fw=None, fe=None, texts=(), rms=0.1):
    return ItemResult(id, wakes, emer, fw, fe, 0.0, 0.0, tuple(texts), "", rms)

def session(items, ments=()):
    return Session(wav=None, ments=tuple(ments), items=tuple(items))

def test_load_session_and_blocks(tmp_path):
    d = {"version": 1, "rate": 16000, "channels": 6, "wav": "set.wav",
         "ments": [{"name": "a.wav", "text": "가", "start": 2.0, "end": 5.0}],
         "items": [{"id": "over_stop_1", "kind": "overlap", "say": "멈춰", "expect": "emergency", "start": 2.0, "end": 7.0}]}
    p = tmp_path / "set.json"; p.write_text(json.dumps(d, ensure_ascii=False))
    s = load_session(p)
    assert s.wav == tmp_path / "set.wav" and s.ments[0] == Ment("a.wav", "가", 2.0, 5.0) and s.items[0].expect == "emergency"
    blocks = split_blocks(np.zeros((BLOCK * 3 + 100, 6), np.int16))
    assert len(blocks) == 3 and blocks[0].shape == (BLOCK, 6)

def test_frames_for_keeps_one_learner_for_the_whole_file():
    made = []
    class Ec:
        def __init__(self, **_): made.append(1)
        def process(self, mic, ref): return (mic // 2).astype(np.int16)
        def close(self): pass
    blocks = [np.full((BLOCK, 6), 1000, np.int16) for _ in range(5)]
    assert all(np.array_equal(f, b[:, 0]) for f, b in zip(frames_for(blocks, "off", EchoConfig("on")), blocks))
    on = frames_for(blocks, "on", EchoConfig("on"), factory=Ec)
    assert len(made) == 1 and all(int(f[0]) == 500 for f in on)

def _models():
    def predict(frame):
        v = int(frame[0])
        return {"a": 0.95 if v == 5000 else 0.0, "b": 0.9 if v == 7000 else 0.0}
    return Models(predict=predict, transcribe=lambda audio: "멈춰")

def _frames(marks):                       # marks: {블록 번호: 값}
    return [np.full(BLOCK, marks.get(i, 0), np.int16) for i in range(200)]

def test_run_item_counts_emergency_inside_the_window():
    item = Item("over_stop_1", "overlap", "멈춰", "emergency", 4.0, 9.0)
    r = run_item(_models(), values_from({}), _frames({70: 7000}), item, [Ment("a.wav", "잠시 멈추겠습니다.", 4.0, 7.0)])
    assert r.emergencies == 1 and 70 * DT <= r.first_emergency <= 76 * DT and r.peak_b == 0.9

def test_run_item_counts_wake_and_ignores_marks_outside_the_window():
    item = Item("over_wake_1", "overlap", "비카야", "wake", 4.0, 9.0)
    r = run_item(_models(), values_from({}), _frames({70: 5000, 71: 5000, 150: 7000}), item, [])
    assert r.wakes == 1 and abs(r.first_wake - 71 * DT) < DT and r.emergencies == 0

ITEMS = [Item("over_stop_1", "overlap", "멈춰", "emergency", 0, 5), Item("over_wake_1", "overlap", "비카야", "wake", 5, 10),
         Item("user_stop_1", "user_only", "멈춰", "emergency", 10, 14), Item("robot_only", "robot_only", None, None, 14, 74)]
GOOD = {"over_stop_1": res("over_stop_1", emer=1, fe=3.0), "over_wake_1": res("over_wake_1", wakes=1, fw=7.0),
        "user_stop_1": res("user_stop_1", emer=1, fe=11.0), "robot_only": res("robot_only")}

def test_judge_passes_when_on_is_no_worse():
    rep = judge(session(ITEMS), GOOD, GOOD, [-8.0, -9.0, -7.0])
    assert rep["pass"] and rep["overlap_stop"] == {"n": 1, "off": 1, "on": 1, "late": [], "pass": True}
    assert rep["robot_only"]["reduce_db_median"] == -8.0

def test_judge_fails_on_a_lost_stop_a_late_stop_or_a_changed_user_item():
    lost = {**GOOD, "over_stop_1": res("over_stop_1")}
    assert not judge(session(ITEMS), GOOD, lost, [-8.0])["pass"]
    late = {**GOOD, "over_stop_1": res("over_stop_1", emer=1, fe=3.16)}
    assert judge(session(ITEMS), GOOD, late, [-8.0])["overlap_stop"]["late"] == ["over_stop_1"]
    quiet = {**GOOD, "user_stop_1": res("user_stop_1", emer=1, fe=11.0, rms=0.08)}     # −1.9 dB
    assert not judge(session(ITEMS), GOOD, quiet, [-8.0])["user_only"]["pass"]
    text = {**GOOD, "user_stop_1": res("user_stop_1", emer=1, fe=11.0, texts=("머",))}
    assert judge(session(ITEMS), GOOD, text, [-8.0])["user_only"]["differs"] == ["user_stop_1"]

def test_judge_fails_on_weak_reduction_or_new_robot_events():
    assert not judge(session(ITEMS), GOOD, GOOD, [-5.0, -5.5])["robot_only"]["pass"]
    noisy = {**GOOD, "robot_only": res("robot_only", wakes=1, fw=20.0)}
    assert judge(session(ITEMS), GOOD, noisy, [-8.0])["robot_only"]["new_events"] == 1

def test_ment_reductions_skip_the_first_4_seconds():
    ments = [Ment("a.wav", "가", 14.0, 16.0), Ment("b.wav", "나", 20.0, 22.0)]
    off = [np.full(BLOCK, 4000, np.int16)] * 1000; on = [np.full(BLOCK, 1000, np.int16)] * 1000
    out = ment_reductions(session(ITEMS, ments), off, on)
    assert len(out) == 1 and abs(out[0] - (-12.04)) < 0.05
```

- [ ] **Step 2: 실패 확인** → `ModuleNotFoundError: src.aec_replay`
- [ ] **Step 3: `src/aec_replay.py` 와 `tools/aec_replay.py` 를 위 Interfaces 대로 만든다.** `format_report` 는 네 묶음의 수(`n`·`off`·`on`·통과)와 줄임 중앙값, 늦은 항목·달라진 항목의 id 를 한국어 표로 낸다.
- [ ] **Step 4: 통과 확인** → `tests/test_aec_replay.py` 전부 통과 · `.venv/bin/python -m tools.aec_replay --help` 가 0 으로 끝난다(모델을 올리지 않는다)
- [ ] **Step 5: 커밋** — `git add src/aec_replay.py tools/aec_replay.py tests/test_aec_replay.py` · `feat(voice): 에코 제거 다시 틀기 — 같은 녹음을 off·on 으로 운영 감시기에 흘려 "멈춰"·"비카야" 사건을 견준다(설계 5.2·5.4)`

---

### Task 8: 전체 확인

**Files:** 없음(확인만)

- [ ] **Step 1: 전체 시험** — `cd ~/wt-aec-voice && .venv/bin/python -m pytest -q -p no:cacheprovider` → 실패 0, `passed` 가 1700 보다 늘었다.
- [ ] **Step 2: 실제 녹음으로 연기 시험**(음성 노드가 꺼져 있을 필요 없음 — 파일만 읽는다):

```bash
cd ~/wt-aec-voice && .venv/bin/python - <<'PY'
import json, numpy as np, soundfile as sf
from src.aec_replay import frames_for, split_blocks
from src.echo_cancel import EchoConfig
d = "/home/ji_w/vica_data/mic_checks/20261118-aec/"
x, _ = sf.read(d + "aec_153103_user1.wav", dtype="int16", always_2d=True)
m = json.load(open(d + "aec_153103_user1.json"))["marks"]
blocks = split_blocks(x); off = frames_for(blocks, "off", EchoConfig("on")); on = frames_for(blocks, "on", EchoConfig("on"))
rms = lambda fs: np.sqrt(np.mean(np.concatenate(fs).astype(np.float64) ** 2))
a, b = int((m["robot_start"] + 4.0) / 0.08), int(m["robot_end"] / 0.08)
ua, ub = int((m["user_start"] + 0.3) / 0.08), int(m["user_end"] / 0.08)
print("로봇(4 s 뒤) 줄임 dB", round(20 * np.log10(rms(on[a:b]) / rms(off[a:b])), 1), "· 사람 변화 dB", round(20 * np.log10(rms(on[ua:ub]) / rms(off[ua:ub])), 1))
PY
```
Expected: 로봇 줄임 −6 dB 이하(11-18 오프라인 측정 −7.7 ~ −8.4), 사람 변화 ±1 dB 안.

- [ ] **Step 3: `off` 가 지금과 같은지 다시 확인** — `git diff after_feedback_waiting -- src/wakeword_monitor.py src/ros_wakeword_node.py` 를 읽어, `echo_stage` 가 `None` 일 때 바뀌는 실행 경로가 `_queue_item(block, False)`(= `block[:, 0].copy()`)·`_frame_from_queue`(1차원 그대로)·`clip_info_with_aec(info, "off")`(같은 객체) 셋뿐인지 본다.
- [ ] **Step 4: 여기서 멈추고 보고한다.** 아래 Task 9 는 사용자와 함께 한다.

---

### Task 9: 사용자와 함께 하는 검증 (단계마다 승인 — 구현 작업자가 혼자 진행하지 않는다)

- [ ] 녹음(약 5분, 주행 끈 상태, 사용자가 말함): `cd ~/wt-aec-voice && .venv/bin/python -m tools.aec_record` → `~/vica_data/aec_sets/<시각>/set.wav·set.json`
- [ ] 다시 틀기: `.venv/bin/python -m tools.aec_replay ~/vica_data/aec_sets/<시각>/set.json` → 결과표를 사용자에게 보이고 합격 숫자를 확정한다(설계 5.4).
- [ ] 합격이면 운영 브랜치에 합치기(사용자 승인, 푸시 없음) → `.env` 에 `VICA_SW_AEC=log` → 음성 노드 재시작은 사용자가 → `~/vica_data/aec_log` 숫자 확인.
- [ ] `VICA_SW_AEC=on` → `aec_log` 숫자와 `~/vica_data/wake_clips` 를 귀로 확인 → 평소 시나리오 1회(설계 5.5).
