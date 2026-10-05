#!/usr/bin/env python3
"""로컬 LLM 폴백 후보 비교 벤치 (실험용, 자동화 테스트 아님) — 2026-09-28.

잴 것 3가지 + 메모리:
  ① 응답시간  — parse_intent 한 번의 벽시계 초 (발화 → 의도). 실사용 체감값.
  ② 생성속도  — Ollama 가 보고하는 eval_count / eval_duration (tok/s).
                 입력 처리 속도(prompt_eval)도 따로 남긴다.
  ③ 정확도    — intent·목적지 id (+확인 수락 문항은 need_confirm) 일치율.
  ④ 메모리    — 모델 적재 전후 MemAvailable 감소량·/api/ps 크기·러너 RSS.

실사용과 같은 경로를 탄다: parse_intent(model=...) → ChatOllama(reasoning=False,
json_schema 구조화 출력). 계측은 ollama.Client.chat 을 감싸 마지막 청크의 수치를 읽는다.

사용 (저장소 루트, ollama serve 는 launch 와 같은 환경변수로 먼저 띄울 것):
    .venv/bin/python scripts/local_llm_bench/bench.py exaone3.5:2.4b exaone4-1.2b ...
결과: scripts/local_llm_bench/results/<모델>.json
"""
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# .env 는 클라우드(openai·ollama.com)를 가리킨다 — 로드 전에 로컬로 덮어쓴다.
# load_dotenv 는 이미 있는 환경변수를 바꾸지 않는다.
os.environ["VICA_LLM_PROVIDER"] = "ollama"
os.environ["OLLAMA_HOST"] = "http://localhost:11434"
os.environ["OLLAMA_API_KEY"] = ""

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import ollama  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402

from src.destination_loader import load_destinations  # noqa: E402
from src.langchain_intent_parser import parse_intent  # noqa: E402

HOST = "http://localhost:11434"
ROUNDS = int(os.environ.get("BENCH_ROUNDS", "3"))
OUT = Path(__file__).resolve().parent / "results"

# --- 계측: Ollama 응답의 마지막 청크 수치를 가로챈다 -------------------------
LAST: dict = {}
_orig_chat = ollama.Client.chat


def _metrics(chunk) -> dict:
    d = chunk.model_dump() if hasattr(chunk, "model_dump") else dict(chunk)
    return {k: d.get(k) for k in ("total_duration", "load_duration", "prompt_eval_count",
                                   "prompt_eval_duration", "eval_count", "eval_duration")}


def _chat(self, *args, **kwargs):
    resp = _orig_chat(self, *args, **kwargs)
    if kwargs.get("stream"):
        def gen():
            for c in resp:
                if getattr(c, "done", False):
                    LAST.clear()
                    LAST.update(_metrics(c))
                yield c
        return gen()
    LAST.clear()
    LAST.update(_metrics(resp))
    return resp


ollama.Client.chat = _chat

# --- 문항 ------------------------------------------------------------------
R = "starlight_1f_restroom"
# (분류, 발화, 기대 intent, 기대 목적지 id, 직전 로봇 발화, 기대 need_confirm)
# 기대 intent 가 clarify/unknown 이면 서로 바꿔 답해도 정답(실용상 무해).
CASES = [
    # 기존 bench_models.py 8문항 — 09-19 결과와 이어 보기 위해 그대로 둔다.
    ("목적지", "407호로 안내해줘", "navigate", "engineering_4f_room_407_prof_yoon_jiyoung_office", None, None),
    ("목적지", "윤지영 교수님 사무실로 가줘", "navigate", "engineering_4f_room_407_prof_yoon_jiyoung_office", None, None),
    ("오인식", "407화로 가죠", "navigate", "engineering_4f_room_407_prof_yoon_jiyoung_office", None, None),
    ("목적지", "화장실로 안내해줘", "navigate", R, None, None),
    ("목적지", "안내센터로 가고 싶어요", "navigate", "starlight_1f_information_center", None, None),
    ("별칭", "커피 마시는 곳으로 가줘", "navigate", "starlight_1f_cafe", None, None),
    ("질문", "여기가 몇 층이에요?", "question", None, None, None),
    ("되묻기", "으로 가주세요", "clarify", None, None, None),
    # 보강 16문항
    ("별칭", "책 빌리고 싶은데 어디로 가야 돼요", "navigate", "library_1f_information_desk", None, None),
    ("별칭", "증명서 떼러 왔어요 거기로 데려다 주세요", "navigate", "administration_1f_admin_support_office", None, None),
    ("목적지", "김민준 교수님 뵈러 왔는데 연구실로 가 주세요", "navigate", "engineering_4f_room_402_prof_kim_minjun_office", None, None),
    ("목적지", "경영학과 과사무실로 데려다 줘", "navigate", "administration_4f_room_401_biz_dept_office", None, None),
    ("오인식", "보건시로 가주세요", "navigate", "starlight_3f_health_center", None, None),
    ("오인식", "카패 가고 싶어", "navigate", "starlight_1f_cafe", None, None),
    ("접근불가", "학식 먹으러 가자", "navigate", "student_union_2f_cafeteria", None, None),
    ("목록없음", "주차장으로 가 줘", "clarify", None, None, None),
    ("질문", "너는 이름이 뭐야?", "question", None, None, None),
    ("되묻기", "저기 그 뭐였더라", "clarify", None, None, None),
    ("제어", "아 됐어요 안 가도 돼요", "cancel", None, None, None),
    ("제어", "잠깐 멈춰 줄래?", "pause", None, None, None),
    ("제어", "이제 다시 출발하자", "resume", None, None, None),
    # 확인 대기 중 긴 대답 — 07-19 exaone3.5 의 확인 무한반복을 재는 문항.
    # 첫 단어가 긍정어가 아니라 지름길을 안 타고 LLM 으로 간다.
    ("확인수락", "그쪽으로 데려다 주세요", "navigate", R, "별빛관 1층 화장실로 안내해드릴까요?", False),
    ("확인정정", "카페 말고 보건실로 가 주세요", "navigate", "starlight_3f_health_center",
     "별빛관 1층 카페로 안내해드릴까요?", True),
    ("목적지", "엘리베이터 타는 곳까지 같이 가 줘", "navigate", "engineering_1f_elevator", None, None),
]


# --- 0630 지도 문항 (로봇이 실제로 읽는 목적지 파일, 공개 6곳) — 2026-09-28 ------------
MAP0630 = Path.home() / "vica_data/destinations/vica_map_0630/destinations.yaml"
ENT, WC, INFO = "e5cc4231-334c-48e5-935b-44e5b448e25f", "df052c8d-6690-4500-b60e-d4b6bf087b14", "fd65ad77-0995-44f5-ad4d-7b79b7547510"
ROOM2, REST, WORK = "8156bd53-0617-467a-9a04-f6a6ed41b8c1", "8355bc1c-c04e-4301-a70d-f79547eb00c3", "76490c63-536f-4b39-84c3-6c86d7b86f29"
CASES_0630 = [
    ("목적지", "화장실로 안내해줘", "navigate", WC, None, None),
    ("목적지", "안내소로 가 주세요", "navigate", INFO, None, None),
    ("목적지", "휴게실 가고 싶어요", "navigate", REST, None, None),
    ("목적지", "작업실로 데려다 줘", "navigate", WORK, None, None),
    ("목적지", "입구까지 같이 가 줄래?", "navigate", ENT, None, None),
    ("목적지", "방2로 가자", "navigate", ROOM2, None, None),
    ("별칭", "실험실 방으로 가 줘", "navigate", ROOM2, None, None),
    ("간접", "좀 쉬고 싶은데 쉴 수 있는 곳으로 가 줘", "navigate", REST, None, None),
    ("간접", "나가는 곳으로 안내해 줘", "navigate", ENT, None, None),
    ("간접", "물어볼 게 있는데 안내 데스크로 가 줘", "navigate", INFO, None, None),
    ("오인식", "화장시로 가주세요", "navigate", WC, None, None),
    ("오인식", "휴게시로 가 줘", "navigate", REST, None, None),
    ("오인식", "방이로 가 줘", "navigate", ROOM2, None, None),
    ("오인식", "작업시로 가죠", "navigate", WORK, None, None),
    # 407호는 비공개(authorization private) — 목록에 없으니 매칭되면 안 된다.
    ("목록없음", "407호로 가 줘", "clarify", None, None, None),
    ("목록없음", "세미나실 가고 싶어요", "clarify", None, None, None),
    ("목록없음", "주차장으로 가 줘", "clarify", None, None, None),
    ("질문", "여기가 몇 층이에요?", "question", None, None, None),
    ("질문", "너는 누구야?", "question", None, None, None),
    ("되묻기", "으로 가주세요", "clarify", None, None, None),
    ("되묻기", "저기 그 뭐였더라", "clarify", None, None, None),
    ("제어", "아 됐어요 안 가도 돼요", "cancel", None, None, None),
    ("제어", "잠깐 멈춰 줄래?", "pause", None, None, None),
    ("제어", "이제 다시 출발하자", "resume", None, None, None),
    ("확인수락", "그쪽으로 데려다 주세요", "navigate", WC, "화장실으로 안내해드릴까요?", False),
    ("확인정정", "휴게실 말고 작업실로 가 주세요", "navigate", WORK, "휴게실으로 안내해드릴까요?", True),
]
if os.environ.get("BENCH_SET") == "0630":
    CASES = CASES_0630


def _destinations():
    return load_destinations(MAP0630) if os.environ.get("BENCH_SET") == "0630" else load_destinations()


def _judge(case, result) -> bool:
    _, _, want_intent, want_dest, _, want_confirm = case
    intent_ok = (result.intent == want_intent) or (
        want_intent in ("clarify", "unknown") and result.intent in ("clarify", "unknown"))
    dest_ok = (result.matched_destination_id or None) == want_dest
    confirm_ok = want_confirm is None or result.need_confirm == want_confirm
    return intent_ok and dest_ok and confirm_ok


# --- 메모리 ----------------------------------------------------------------
def _http(path: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(HOST + path, data=json.dumps(body).encode() if body else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read() or b"{}")


def _mem_available_mb() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) // 1024
    return -1


def _swap_used_mb() -> int:
    kv = {l.split(":")[0]: int(l.split()[1]) for l in Path("/proc/meminfo").read_text().splitlines()}
    return (kv["SwapTotal"] - kv["SwapFree"]) // 1024


def _runner_rss_mb() -> int:
    # 모델을 실제로 올린 자식 프로세스(ollama runner / llama-server)의 RSS 합.
    out = subprocess.run(["ps", "-eo", "rss,args"], capture_output=True, text=True).stdout
    return sum(int(l.split(None, 1)[0]) for l in out.splitlines()[1:]
               if ("runner" in l or "llama-server" in l) and "ollama" in l) // 1024


def _llm_own_mem_mb() -> dict:
    """LLM 러너 프로세스 자신의 몫만 (다른 프로세스·Claude·원격 세션과 무관).

    젯슨은 CPU·GPU 가 한 메모리를 나눠 쓴다. GPU 몫은 프로세스 RSS 에 안 잡혀
    jtop(시스템 python)이 nvmap 에서 읽는 프로세스별 GPU 메모리로 잰다.
    CPU 몫은 smaps_rollup 의 익명(Anon) + 파일(모델 가중치 mmap) PSS.
    """
    out = {"gpu_mb": None, "cpu_anon_mb": None, "cpu_file_mb": None}
    ps = subprocess.run(["ps", "-eo", "pid,comm"], capture_output=True, text=True).stdout
    pids = [l.split()[0] for l in ps.splitlines() if l.split()[-1] in ("llama-server", "ollama_llama_se")]
    if not pids:
        return out
    pid = pids[0]
    roll = {l.split(":")[0]: int(l.split()[1]) for l in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines()[1:]}
    out["cpu_anon_mb"] = roll.get("Pss_Anon", 0) // 1024
    out["cpu_file_mb"] = roll.get("Pss_File", 0) // 1024
    # jtop 은 첫 j.ok() 에서 프로세스 목록이 비어 있다(0 으로 읽힘, 09-28) — 세 번 갱신 뒤 읽는다.
    code = ("from jtop import jtop\nwith jtop() as j:\n for _ in range(3): j.ok()\n"
            f" print(sum(p[8] for p in j.processes if p[0]=={pid}))")
    try:
        r = subprocess.run(["/usr/bin/python3", "-c", code], capture_output=True, text=True, timeout=30)
        out["gpu_mb"] = int(r.stdout.strip().splitlines()[-1]) // 1024
    except Exception:
        pass
    return out


def _unload_all() -> None:
    for m in _http("/api/ps").get("models", []):
        _http("/api/generate", {"model": m["name"], "keep_alive": 0})
    time.sleep(5)


# --- 본체 ------------------------------------------------------------------
def _call(model, case, destinations):
    _, text, _, _, prev_ai, _ = case
    history = [AIMessage(prev_ai)] if prev_ai else None
    LAST.clear()
    t0 = time.monotonic()
    result = parse_intent(text, destinations, history=history, model=model)
    wall = time.monotonic() - t0
    m = dict(LAST)
    ok = _judge(case, result)
    tok_s = (m["eval_count"] / m["eval_duration"] * 1e9) if m.get("eval_duration") else None
    pp_s = (m["prompt_eval_count"] / m["prompt_eval_duration"] * 1e9) if m.get("prompt_eval_duration") else None
    return {"text": text, "category": case[0], "ok": ok, "wall_s": round(wall, 3),
            "intent": result.intent, "dest": result.matched_destination_id or None,
            "need_confirm": result.need_confirm, "fallback": result.reply if not m else None,
            "gen_tok_s": round(tok_s, 1) if tok_s else None,
            "prompt_tok_s": round(pp_s, 1) if pp_s else None, **m}


def bench(model: str) -> dict:
    destinations = _destinations()
    _unload_all()
    mem_before = _mem_available_mb()
    print(f"\n=== {model}  (적재 전 가용 {mem_before} MB)")

    cold = _call(model, CASES[0], destinations)  # 첫 호출 = 모델 적재 포함
    print(f" 첫 호출 {cold['wall_s']:.2f}s (적재 {((cold.get('load_duration') or 0) / 1e9):.2f}s)")
    ps = _http("/api/ps").get("models", [])
    mem_loaded = _mem_available_mb()

    rows = []
    for r in range(ROUNDS):
        for case in CASES:
            row = _call(model, case, destinations)
            row["round"] = r
            rows.append(row)
            if r == 0:
                print(f" [{'O' if row['ok'] else 'X'}] {row['wall_s']:5.2f}s "
                      f"{row['gen_tok_s'] or 0:5.1f}tok/s  '{row['text']}' -> {row['intent']} / {row['dest'] or '-'}")
    mem_after = _mem_available_mb()
    own = _llm_own_mem_mb()

    walls = sorted(x["wall_s"] for x in rows)
    gens = [x["gen_tok_s"] for x in rows if x["gen_tok_s"]]
    summary = {
        "model": model,
        "accuracy": round(sum(x["ok"] for x in rows) / len(rows), 3),
        "correct_round0": sum(x["ok"] for x in rows if x["round"] == 0),
        "n_cases": len(CASES),
        "wall_mean_s": round(sum(walls) / len(walls), 3),
        "wall_p50_s": walls[len(walls) // 2],
        "wall_p90_s": walls[int(len(walls) * 0.9)],
        "wall_max_s": walls[-1],
        "cold_first_s": cold["wall_s"],
        "gen_tok_s_mean": round(sum(gens) / len(gens), 1) if gens else None,
        "prompt_tokens_mean": round(sum(x.get("prompt_eval_count") or 0 for x in rows) / len(rows)),
        "out_tokens_mean": round(sum(x.get("eval_count") or 0 for x in rows) / len(rows), 1),
        "mem_available_before_mb": mem_before,
        "mem_drop_loaded_mb": mem_before - mem_loaded,
        "mem_drop_after_mb": mem_before - mem_after,
        "ollama_ps_size_mb": ps[0]["size"] // 2**20 if ps else None,
        "runner_rss_mb": _runner_rss_mb(),
        # LLM 자신의 몫(권장 지표): GPU + CPU 익명 + 가중치 mmap
        "llm_gpu_mb": own["gpu_mb"],
        "llm_cpu_anon_mb": own["cpu_anon_mb"],
        "llm_cpu_file_mb": own["cpu_file_mb"],
        "llm_own_total_mb": sum(v for v in own.values() if v) if own["gpu_mb"] is not None else None,
        "rounds": ROUNDS,
        # 스택 켠 시험용: 끝났을 때 남은 메모리·스왑·부하(1분 평균)
        "mem_available_end_mb": mem_after,
        "swap_used_end_mb": _swap_used_mb(),
        "loadavg_1m": float(Path("/proc/loadavg").read_text().split()[0]),
        "tag": os.environ.get("BENCH_TAG", ""),
    }
    print(f" 정확도 {summary['accuracy'] * 100:.1f}%  응답 평균 {summary['wall_mean_s']:.2f}s "
          f"p90 {summary['wall_p90_s']:.2f}s  생성 {summary['gen_tok_s_mean']} tok/s  "
          f"메모리 -{summary['mem_drop_after_mb']} MB")
    OUT.mkdir(exist_ok=True)
    tag = os.environ.get("BENCH_TAG", "")
    (OUT / f"{model.replace(':', '_').replace('/', '_')}{'_' + tag if tag else ''}.json").write_text(
        json.dumps({"summary": summary, "cold": cold, "rows": rows}, ensure_ascii=False, indent=1))
    return summary


if __name__ == "__main__":
    for name in sys.argv[1:]:
        bench(name)
