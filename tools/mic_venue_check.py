"""마이크 현장 점검기 — 시연장에서 한 번 돌려 '이 장소에서 비카가 제대로 듣는가'를 휴대폰으로 본다.

설계: docs/superpowers/specs/2026-10-12-mic-venue-check-design.md · 사용법: docs/mic-venue-check.md

    cd ~/VICA-smarthandle/vica-voice-llm && .venv/bin/python -m tools.mic_venue_check --name 시연장
    .venv/bin/python -m tools.mic_venue_check --restore     # .env 의 마이크 값 7개를 가장 최근 백업으로
    .venv/bin/python -m tools.mic_venue_check --demo        # 마이크·모델 없이 화면만(가짜 소리)

전제: ⑫·⑬ 꺼짐 · 로봇 정지 · reSpeaker USB · 스피커는 reSpeaker 출력. 도구는 ROS 에 아무것도 보내지 않는다.
가져오기(import)만으로는 .env 를 읽지 않는다 — 시험이 운영 설정에 흔들리지 않게 프로그램으로 실행할 때만 읽는다.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shutil
import signal
import socket
import sys
import time
import wave
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    # ⑫ 와 같게 .env 를 src 가져오기보다 먼저 읽는다 — wakeword_monitor 가 가져오는 순간 환경에서
    # 문턱·청취 창 시간을 읽는다(노드의 09-01 결함과 같은 것).
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")

from src.env_file import EnvEditError, latest_backup, restore_latest, values_from  # noqa: E402
from src.mic_check_replay import Models  # noqa: E402
from src.mic_check_rules import MicValues  # noqa: E402
from src.mic_check_session import Clip, Hardware, Session, previous_robot_level  # noqa: E402
from src.wakeword_monitor import FRAME, SAMPLE_RATE, WakewordMonitor  # noqa: E402
from tools.mic_check_web import DEFAULT_PORT, LiveServer  # noqa: E402

ENV_PATH = ROOT / ".env"
DATA = Path.home() / "vica_data" / "mic_checks"
BACKUP_DIR = DATA / "env_backups"
PAGE = Path(__file__).with_name("mic_check_page.html")
CHANNELS = 6              # reSpeaker 6채널 펌웨어 — ch0 이 칩 처리음(⑫ 가 듣는 소리)
SETTLE_SEC = 2.0          # 칩에 쓴 뒤 AGC·에코가 자리 잡을 시간(잡음은 그 뒤에 잰다)
VOICE_PATTERNS = ("vica_voice.launch.py", "-m src.ros_")   # ⑫·⑬ 중복 검사와 같은 무늬
ROBOT_CLIPS = (           # 설계 3절 — 호출어·긴급어와 닮은 말이 든 실제 멘트, 합 29.3 s
    "mission_msg_approach_onboarding.wav",
    "mission_msg_approach_question.wav",
    "mission_msg_wait_spot_left_default.wav",
    "mission_msg_estopped.wav",
    "mission_msg_paused.wav",
    "mission_msg_obstacle_stop.wav",
    "mission_msg_estop_reject.wav",
    "mission_msg_obstacle_avoid.wav",
)


def voice_stack_running(cmdlines: Sequence[str]) -> list:
    """떠 있는 음성 칸(⑫·⑬) 명령줄들. 이 도구가 마이크·스피커·칩을 혼자 써야 한다."""
    return [line for line in cmdlines if any(p in line for p in VOICE_PATTERNS)]


def _proc_cmdlines() -> list:
    me = os.getpid()
    out = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == me:
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        out.append(raw.replace(b"\0", b" ").decode("utf-8", "replace").strip())
    return out


def load_clips(baked_dir: Path, names: Sequence[str] = ROBOT_CLIPS) -> tuple:
    """(Clip 목록, 빠진 파일 이름 목록). 글자는 manifest.json 에서 — ⑫ 의 자기 말 무시 규칙에 쓴다."""
    baked_dir = Path(baked_dir)
    try:
        manifest = json.loads((baked_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        manifest = {}
    clips, missing = [], []
    for name in names:
        try:
            with wave.open(str(baked_dir / name), "rb") as w:
                pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
                rate = w.getframerate()
        except (OSError, wave.Error, EOFError):
            missing.append(name)
            continue
        clips.append(Clip(name, str(manifest.get(name, "")), pcm, rate))
    return clips, missing


def lan_ip() -> str:
    """휴대폰이 열 주소. 바깥으로 아무것도 보내지 않고 나가는 쪽 주소만 고른다."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def write_chip(values: MicValues) -> list:
    """칩에 AGC·에코를 쓰고 읽어서 확인한다. 문제 목록(빈 목록 = 성공). 마이크 스트림이 닫혀 있을 때만 부를 것."""
    from src import audio_out
    from src.dsp_state import apply_agc_desired_level, apply_echo_tuning

    audio_out.reset_device_cache()   # 재생 스트림이 열려 있으면 칩이 쓰기를 거부한다 — 먼저 닫는다
    problems = []
    if not apply_agc_desired_level(values.agc):
        problems.append(f"AGC {values.agc:.3f} 쓰기 실패")
    wrote = apply_echo_tuning(values.echo_etail, values.echo_nlatten)
    etail = wrote.get("GAMMA_ETAIL")
    if etail is None or abs(float(etail) - values.echo_etail) > 1e-3:
        problems.append(f"에코 꼬리 {values.echo_etail} 쓰기 실패(읽은 값 {etail})")
    nl = wrote.get("NLATTENONOFF")
    if nl is None or int(nl) != int(values.echo_nlatten):
        problems.append(f"에코 메아리 {values.echo_nlatten} 쓰기 실패(읽은 값 {nl})")
    return problems


def _find_input() -> int:
    import sounddevice as sd

    for i, dev in enumerate(sd.query_devices()):
        if "respeaker" in dev["name"].lower() and dev["max_input_channels"] >= CHANNELS:
            return i
    raise SystemExit("reSpeaker 6채널 입력을 찾지 못했습니다 — USB 연결을 확인하세요(다른 마이크로 대신하지 않습니다).")


class Mic:
    """reSpeaker ch0 스트림. 칩 값은 스트림이 닫혀 있을 때만 써지므로 열고 닫기를 여기서 한다."""

    def __init__(self, device: int):
        self.device = device
        self.frames: "queue.Queue[np.ndarray]" = queue.Queue()
        self._stream = None

    @property
    def is_open(self) -> bool:
        return self._stream is not None

    def open(self) -> None:
        import sounddevice as sd

        def callback(indata, frames, time_info, status):  # noqa: ANN001
            block = np.frombuffer(indata, dtype=np.int16).reshape(-1, CHANNELS)
            self.frames.put(block[:, 0].copy())

        _drain(self.frames)   # 닫혀 있는 동안 남은 묵은 조각은 버린다
        self._stream = sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=FRAME, channels=CHANNELS,
                                         dtype="int16", device=self.device, callback=callback)
        self._stream.start()

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


def _chip_apply_for(mic: Mic):
    def apply(values: MicValues) -> list:
        was_open = mic.is_open
        mic.close()
        problems = write_chip(values)
        time.sleep(SETTLE_SEC)
        if was_open:
            mic.open()
        return problems
    return apply


def _place_hint(path: str) -> Optional[str]:
    if not path:
        print("  (장소 귀띔 없음 — --destinations 를 주면 ⑫ 와 같은 귀띔으로 받아씁니다)")
        return None
    from src.destination_loader import build_place_hint, load_destinations
    try:
        return build_place_hint(load_destinations(path))
    except Exception as exc:  # noqa: BLE001
        print(f"  (장소 귀띔 생략: {exc})")
        return None


def _load_models(listen_hint: Optional[str]) -> Models:
    m = WakewordMonitor(on_emergency=lambda e: None, on_user_text=lambda t: None, listen_hint=listen_hint)
    m._load_real()
    return Models.from_loaded(m)


def _drain(q: "queue.Queue") -> int:
    """큐를 비우고 버린 개수를 돌려준다 — 처리 스레드가 오래 막힌 뒤 쌓인 묵은 마이크 조각을 버리려고."""
    dropped = 0
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            return dropped
        dropped += 1


SLOW_STEP_SEC = 0.3       # 한 번 처리에 이보다 오래 걸리면 그동안 쌓인 마이크 조각은 버린다


def _may_drain(session) -> bool:
    """녹음·6단계 도중에는 비우지 않는다 — 버리면 녹음에 빈틈이 생겨 채점이 어긋난다. 밀린 조각은 차례로 처리한다."""
    return session.recording_kind is None and not session.robot_running


def _run_commands(session: Session, server: LiveServer) -> None:
    while True:
        try:
            cmd = server.commands.get_nowait()
        except queue.Empty:
            return
        session.handle(cmd)


def record_dir_for(name: str, demo: bool, now: datetime) -> Path:
    """기록 폴더. 데모 기록은 demo/ 아래에 둔다 — 가짜 로봇 목소리 크기가 다음 진짜 점검의 비교 기준이 되지 않게."""
    safe = re.sub(r"[^\w가-힣-]+", "_", name).strip("_") or "점검"
    return (DATA / "demo" if demo else DATA) / f"{now:%Y%m%d-%H%M}_{safe}"


def _restore_cli() -> int:
    latest = latest_backup(BACKUP_DIR)
    if latest is None:
        print(f"되돌릴 백업이 없습니다 ({BACKUP_DIR})")
        return 1
    answer = input(f".env 의 마이크 값 7개를 {latest.name} 으로 되돌릴까요? "
                   "다른 줄은 그대로이고, 지금 .env 는 .before-restore 로 남깁니다 [y/N] ")
    if answer.strip().lower() != "y":
        print("그대로 둡니다.")
        return 0
    restore_latest(ENV_PATH, BACKUP_DIR)
    print("되돌렸습니다. ⑫를 다시 켜야 적용됩니다.")
    return 0


class _Demo:
    """--demo: 마이크·모델·칩 없이 가짜 소리로 화면을 움직인다(휴대폰 화면 확인용). 진짜 .env 는 건드리지 않는다."""
    WAKE, STOP, YES, NO, VOICE = 20001, 20002, 20003, 20004, 20005

    def __init__(self):
        self.vad = False
        self.rng = np.random.default_rng(0)

    def predict(self, frame: np.ndarray) -> dict:
        v = int(frame[0])
        return {"a": 0.92 if v == self.WAKE else 0.02, "b": 0.88 if v == self.STOP else 0.03}

    def models(self) -> Models:
        def confirm(audio: np.ndarray) -> str:
            return "아니" if np.any(audio == self.NO) else "그래"
        return Models(self.predict, lambda a: "멈춰", lambda a: "화장실로 가자", confirm)

    def frame(self, session: Session) -> np.ndarray:
        kind, pos = session.recording_kind, session.recording_pos
        self.vad = False
        if kind == "call" and pos in (3, 4):
            return np.full(FRAME, self.WAKE, np.int16)
        if kind == "call" and 5 <= pos < 15:
            self.vad = True
            return np.full(FRAME, self.VOICE, np.int16)
        if kind == "answer" and 3 <= pos < 11:
            self.vad = True
            return np.full(FRAME, self.NO if session.recording_expected == "no" else self.YES, np.int16)
        if kind == "stop" and pos == 3:
            return np.full(FRAME, self.STOP, np.int16)
        scale = 900 if session.robot_running else 100
        return (self.rng.standard_normal(FRAME) * scale).astype(np.int16)


def _raise_interrupt(signum, frame) -> None:  # noqa: ANN001
    raise KeyboardInterrupt


def _install_signal_handlers() -> None:
    """창을 닫거나(SIGHUP) kill(SIGTERM) 해도 Ctrl+C 와 같이 finally 가 돌아 칩이 되돌아가게."""
    for sig in (signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(sig, _raise_interrupt)
        except (ValueError, OSError):
            pass


def _run_demo(server: LiveServer, env_values: MicValues, record_dir: Path, clips: list,
              prev: Optional[float], place: str) -> int:
    demo = _Demo()
    demo_env = record_dir / "demo.env"
    shutil.copy(ENV_PATH, demo_env)
    session = Session(models=demo.models(),
                      hw=Hardware(lambda v: [], lambda: demo.vad, lambda pcm, rate: time.sleep(len(pcm) / rate)),
                      publish=server.publish, env_values=env_values, env_path=demo_env, place=place,
                      record_dir=record_dir, backup_dir=record_dir / "env_backups", clips=clips,
                      previous_robot_level=prev)
    print("  [데모] 가짜 소리로 돕니다. 저장·되돌리기는 기록 폴더의 demo.env 에만 씁니다.")
    _install_signal_handlers()
    next_t = time.monotonic()
    try:
        while not session.done:
            _run_commands(session, server)
            session.on_frame(demo.frame(session))
            next_t += FRAME / SAMPLE_RATE
            time.sleep(max(0.0, next_t - time.monotonic()))
    except KeyboardInterrupt:
        print("\n  Ctrl+C — 끝냅니다")
    finally:
        try:
            session.shutdown()
        except Exception as exc:  # noqa: BLE001
            print(f"  ⚠ 6단계 멈추기 문제: {exc}")
        session.close()
        server.stop()
        print(f"  기록: {record_dir}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="마이크 현장 점검기")
    ap.add_argument("--name", default="점검", help="장소 이름(기록 폴더·.env 표시에 쓴다)")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--destinations", default=os.environ.get("VICA_DESTINATIONS_YAML", ""),
                    help="장소 귀띔용 destinations.yaml(⑫ 와 같은 파일). 없으면 귀띔 없이 받아쓴다")
    ap.add_argument("--restore", action="store_true", help=".env 의 마이크 값 7개를 가장 최근 백업으로 되돌리고 끝낸다")
    ap.add_argument("--demo", action="store_true", help="마이크·모델 없이 가짜 소리로 화면만 본다")
    args = ap.parse_args(argv)
    if args.restore:
        return _restore_cli()
    os.chdir(ROOT)   # 호출 모델 경로(models/...)가 저장소 기준이다
    if not args.demo:
        running = voice_stack_running(_proc_cmdlines())
        if running:
            print("⑫·⑬ 음성 칸이 켜져 있습니다. 그 칸을 Ctrl+C 로 끈 뒤 다시 실행하세요:")
            for line in running[:5]:
                print("   ", line[:120])
            return 1
    try:
        env_values = values_from(os.environ)
    except EnvEditError as exc:
        print(exc)
        return 1
    record_dir = record_dir_for(args.name, args.demo, datetime.now())
    record_dir.mkdir(parents=True, exist_ok=True)
    prev = previous_robot_level(DATA, exclude=record_dir)
    clips, missing = load_clips(ROOT / "assets" / "baked")
    for name in missing:
        print(f"  (멘트 {name} 없음 — 건너뜀)")
    server = LiveServer(PAGE.read_text(encoding="utf-8"), port=args.port)
    server.start()
    print(f"\n  휴대폰으로 여세요:  http://{lan_ip()}:{server.port}\n")
    server.publish({"type": "prep", "text": "젯슨이 준비하는 중입니다(모델 올리기 1~2분)…"}, keep="status")
    if args.demo:
        return _run_demo(server, env_values, record_dir, clips, prev, args.name)

    from src import audio_out
    from src.dsp_state import DspState

    dsp = DspState()
    if not dsp.available:
        print("reSpeaker 상태 레지스터(사람 말 표시)를 읽을 수 없습니다 — udev·USB 를 확인하세요.")
        server.stop()
        return 1
    mic = Mic(_find_input())
    chip_apply = _chip_apply_for(mic)
    print("  칩에 .env 값을 씁니다(⑫ 가 켤 때와 같은 일)…")
    problems = chip_apply(env_values)
    for p in problems:
        print(f"  ⚠ {p}")
    print("  모델을 올립니다(1~2분)…")
    models = _load_models(_place_hint(args.destinations))
    session = Session(models=models,
                      hw=Hardware(chip_apply, dsp.speech_detected,
                                  lambda pcm, rate: audio_out.play(pcm, rate, blocking=True),
                                  stop_play=audio_out.stop),
                      publish=server.publish, env_values=env_values, env_path=ENV_PATH, place=args.name,
                      record_dir=record_dir, backup_dir=BACKUP_DIR, clips=clips, previous_robot_level=prev,
                      chip_problems=problems)
    mic.open()
    print("  준비 끝 — 휴대폰 화면을 따라 하세요. 끝낼 때는 화면의 '점검 끝내기' 또는 Ctrl+C.")
    _install_signal_handlers()
    try:
        while not session.done:
            t0 = time.monotonic()
            _run_commands(session, server)
            slow = time.monotonic() - t0 > SLOW_STEP_SEC
            try:
                frame = mic.frames.get(timeout=0.5)
            except queue.Empty:
                if slow and _may_drain(session):
                    _drain(mic.frames)
                continue
            t0 = time.monotonic()
            session.on_frame(frame)
            if (slow or time.monotonic() - t0 > SLOW_STEP_SEC) and _may_drain(session):
                _drain(mic.frames)   # 재채점 등으로 오래 막힌 동안 쌓인 조각은 시간이 어긋나므로 버린다
    except KeyboardInterrupt:
        print("\n  Ctrl+C — 끝냅니다")
    finally:
        # 단계마다 따로 막는다 — 한 단계가 터져도 칩 되돌리기·서버 끄기는 반드시 돈다.
        try:
            session.shutdown()   # 6단계 재생이 돌고 있으면 먼저 끊는다 — 재생 스트림이 열려 있으면 칩 쓰기가 거부된다
        except Exception as exc:  # noqa: BLE001
            print(f"  ⚠ 6단계 멈추기 문제: {exc}")
        try:
            mic.close()
        except Exception as exc:  # noqa: BLE001
            print(f"  ⚠ 마이크 닫기 문제: {exc}")
        try:
            back = write_chip(session.env_values)   # 칩을 .env 값으로(저장했다면 새 값)
            if back:
                print("  ⚠ 칩 되돌리기 문제: " + "; ".join(back))
        except Exception as exc:  # noqa: BLE001
            print(f"  ⚠ 칩 되돌리기 실패: {exc} — ⑫ 를 다시 켜면 .env 에 적힌 값으로 다시 써집니다"
                  "(에코 줄이 없으면 에코는 전원을 껐다 켜야 공장값)")
        try:
            session.close()
        except Exception as exc:  # noqa: BLE001
            print(f"  ⚠ 기록 마무리 문제: {exc}")
        try:
            server.stop()
        except Exception as exc:  # noqa: BLE001
            print(f"  ⚠ 서버 끄기 문제: {exc}")
        print(f"  기록: {record_dir}")
        print("  이제 ⑫를 켜고 시작 로그의 값을 확인한 뒤 시연 대화를 한 번 해 보세요.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
