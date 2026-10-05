#!/usr/bin/env python3
"""보고서용 한 장 — 벤치(캠퍼스·0630) → 마이크 실물 순서로 그래프·표·주석.

사용: python3 scripts/local_llm_bench/report_full.py <출력.html>
읽는 것: results/<모델>.json(캠퍼스, 스택 꺼짐) · <모델>_stack.json(캠퍼스, 스택 켬) ·
<모델>_0630.json(0630 지도) · own_mem_stack.json(메모리) · field_2026-09-28.json(실물).
"""
import html
import json
import sys
from pathlib import Path

import report as R

RES = R.RES
KEYS = [m[0] for m in R.MODELS]
INFO = {m[0]: {"name": m[1], "quant": m[2], "lic": m[3], "size": m[4], "short": m[5]} for m in R.MODELS}
SLOT = {k: i + 1 for i, k in enumerate(KEYS)}
# 표의 줄 순서 = 판정 순서
ORDER = ["midm2-mini", "gemma4-e2b-text", "qwen3.5-2b-text", "exaone3.5_2.4b", "exaone4-1.2b"]
VERDICT = {"midm2-mini": ("채택", "ok"), "gemma4-e2b-text": ("예비", "part"),
           "exaone3.5_2.4b": ("제외", "over"), "exaone4-1.2b": ("제외", "over"),
           "qwen3.5-2b-text": ("제외", "over")}


def _load(name):
    f = RES / name
    return json.loads(f.read_text()) if f.exists() else None


def summaries(suffix, fallback=None):
    out = {}
    for k in KEYS:
        d = _load(f"{k}{suffix}.json") or (_load(f"{k}{fallback}.json") if fallback else None)
        if d:
            out[k] = d["summary"]
    return out


def single_chart(title, sub, data, unit, fmt, best, note):
    items = [(SLOT[k], INFO[k]["short"], v, f'{INFO[k]["name"]}: {tip}', False) for k, v, tip in data]
    return R.bar_chart(title, sub, items, unit, fmt, best) + f'<p class="note">{note}</p>'


def pair_chart(title, sub, key, unit, fmt, best, tip, note):
    off, on = summaries(""), summaries("_stack")
    items = []
    for k in KEYS:
        if k in off:
            items.append((SLOT[k], f'{INFO[k]["short"]} · 꺼짐', key(off[k]), "스택 꺼짐 — " + tip(k, off[k]), True))
        if k in on:
            items.append((SLOT[k], f'{INFO[k]["short"]} · 켜짐', key(on[k]), "스택 켜짐 — " + tip(k, on[k]), False))
    return R.bar_chart(title, sub, items, unit, fmt, best) + f'<p class="note">{note}</p>'


def sw(k):
    return f'<span class="sw" style="background:var(--s{SLOT[k]})"></span>'


def build():
    s0630 = summaries("_0630", "_0630stack")  # Qwen 은 같은 날 저녁 회차(_0630stack)만 있다
    field = _load("field_2026-09-28.json")["models"]
    own = _load("own_mem_stack.json") or {}

    # ---------- 벤치 1: 캠퍼스 ----------
    b1 = "".join([
        pair_chart("① 응답시간", "말이 끝난 뒤 의도가 나오기까지 걸린 평균 시간 (짧을수록 좋음)",
                   lambda s: s["wall_mean_s"], "초", lambda v: f"{v:.1f}", "low",
                   lambda k, s: f'{INFO[k]["name"]}: 평균 {s["wall_mean_s"]:.2f}초 · p90 {s["wall_p90_s"]:.2f}초 · 최악 {s["wall_max_s"]:.2f}초',
                   "흐린 막대는 주행 스택을 끈 상태, 진한 막대는 켠 상태입니다. 스택을 켜도 느려지지 않았습니다. 네 모델 모두 2초 안팎이라 대화에 지장이 없는 수준입니다."),
        pair_chart("② 생성속도", "답을 써 내려가는 속도 (tokens/s, 높을수록 좋음)",
                   lambda s: s["gen_tok_s_mean"] or 0, "", lambda v: f"{v:.0f}", "high",
                   lambda k, s: f'{INFO[k]["name"]}: {s["gen_tok_s_mean"]} tok/s · 답 길이 {s["out_tokens_mean"]} 토큰',
                   "작은 모델(EXAONE 4.0)이 가장 빠르게 씁니다. 다만 응답시간은 쓰는 속도 × 답 길이로 정해집니다. 그래서 tok/s 순위가 곧 응답시간 순위는 아닙니다(EXAONE 3.5는 빨리 쓰지만 답이 길어 Mi:dm보다 늦습니다)."),
        pair_chart("③ 정확도", "24문항 × 3회, 의도와 목적지를 모두 맞힌 비율",
                   lambda s: s["accuracy"] * 100, " %", lambda v: f"{v:.0f}", "high",
                   lambda k, s: f'{INFO[k]["name"]}: {s["accuracy"] * 100:.1f} %',
                   "세 모델이 83.3 %로 같았습니다. 한 문항이 4.2 %p라서, 이 문항 수로는 점수 차이를 가리기 어렵습니다. 캠퍼스 예시 목적지 28곳은 로봇이 실제로 쓰는 지도가 아니라서, 다음 단계에서 실제 지도로 다시 쟀습니다."),
    ])

    # ---------- 벤치 2: 0630 ----------
    d = lambda f, tip: [(k, f(s0630[k]), tip(s0630[k])) for k in KEYS if k in s0630]
    b2 = "".join([
        single_chart("① 응답시간", "0630 지도, 26문항 × 3회 평균 (짧을수록 좋음)",
                     d(lambda s: s["wall_mean_s"], lambda s: f'평균 {s["wall_mean_s"]:.2f}초 · p90 {s["wall_p90_s"]:.2f}초'),
                     "초", lambda v: f"{v:.1f}", "low",
                     "목적지가 6곳으로 줄어 입력이 짧아지면서 네 모델 모두 캠퍼스 벤치보다 빨라졌습니다. 빠른 순서는 EXAONE 4.0 → Mi:dm → EXAONE 3.5 → Gemma → Qwen3.5 2B입니다. Qwen은 덜 압축된 판(Q8_0)이라 가장 느립니다."),
        single_chart("② 생성속도", "tokens/s (높을수록 좋음)",
                     d(lambda s: s["gen_tok_s_mean"] or 0, lambda s: f'{s["gen_tok_s_mean"]} tok/s'),
                     "", lambda v: f"{v:.0f}", "high",
                     "쓰는 속도 순서는 캠퍼스 벤치와 같습니다. 가장 가벼운 EXAONE 4.0이 약 40 tok/s로 가장 빠릅니다."),
        single_chart("③ 정확도", "26문항 × 3회, 의도와 목적지를 모두 맞힌 비율",
                     d(lambda s: s["accuracy"] * 100, lambda s: f'{s["accuracy"] * 100:.1f} %'),
                     " %", lambda v: f"{v:.0f}", "high",
                     "<b>실제 지도에서는 순위가 갈렸습니다.</b> Gemma가 84.6 %로 1위입니다. Mi:dm(74.4 %)은 “안내소로 가 주세요”를 되물었고, 없는 곳(세미나실·주차장)을 방2·입구로 끼워 맞췄습니다. EXAONE 두 모델은 “쉬고 싶은 곳”을 일시정지로 받는 등 기본 목적지도 자주 놓쳤습니다. 나중에 추가한 Qwen3.5 2B(69.2 %)는 비공개인 407호를 입구로 끼워 맞추고 “안 가도 돼요”(취소)를 놓쳤습니다. Qwen은 Gemma와 같은 회차에 나란히 쟀고, 그 회차에서도 Gemma는 84.6 %로 똑같이 나왔습니다."),
    ])

    # ---------- 실물 ----------
    def fstat(k):
        rows = field.get(k, [])
        judged = [r for r in rows if r["ok"] is not None]
        return rows, judged
    time_data, acc_data, frows = [], [], ""
    for k in KEYS:
        rows, judged = fstat(k)
        if not rows:
            continue
        avg = sum(r["llm_s"] for r in rows) / len(rows)
        ok = sum(r["ok"] for r in judged)
        time_data.append((k, avg, f"LLM 평균 {avg:.2f}초 (판단 {len(rows)}문장)"))
        acc_data.append((k, 100 * ok / len(judged) if judged else 0, f"채점 {ok}/{len(judged)}"))
    for k in ORDER:
        rows, judged = fstat(k)
        if not rows:
            continue
        avg = sum(r["llm_s"] for r in rows) / len(rows)
        ok = sum(r["ok"] for r in judged)
        cat = lambda c: [r for r in judged if r["category"] == c]
        cell = lambda c: (f'{sum(r["ok"] for r in cat(c))} / {len(cat(c))}' if cat(c) else "–")
        cond = {"qwen3.5-2b-text": "8곳", "midm2-mini": "6곳·8곳"}.get(k, "6곳")
        frows += (f'<tr><td>{sw(k)}{html.escape(INFO[k]["name"])}</td><td class="n">{cond}</td><td class="n">{len(rows)}</td>'
                  f'<td class="n">{cell("목적지")}</td><td class="n">{cell("돌려 말하기")}</td><td class="n">{cell("대답")}</td><td class="n">{cell("잡음")}</td>'
                  f'<td class="n b">{ok} / {len(judged)}</td><td class="n">{avg:.1f}초</td>'
                  f'<td class="n">{max(r["llm_s"] for r in rows):.1f}초</td>'
                  f'<td><span class="chip {VERDICT[k][1]}">{VERDICT[k][0]}</span></td></tr>')
    field_detail = ""
    for k in KEYS:
        items = "".join(
            f'<li>“{html.escape(r["text"])}” → {r["intent"]}'
            f'{" (대답 없이 버려짐)" if r["dropped"] else ""} · {r["llm_s"]:.1f}초 '
            f'{"✅" if r["ok"] else ("❌" if r["ok"] is False else "·")}</li>'
            for r in field.get(k, []))
        if items:
            field_detail += f'<details><summary>{sw(k)}{html.escape(INFO[k]["name"])}</summary><ul>{items}</ul></details>'
    b3 = "".join([
        single_chart("① LLM 응답시간", "받아쓰기가 끝난 순간부터 판단이 나오기까지 (짧을수록 좋음)",
                     time_data, "초", lambda v: f"{v:.1f}", "low",
                     "빠른 순서는 EXAONE 4.0 → Mi:dm → EXAONE 3.5 → Gemma → Qwen3.5 2B로 벤치와 같습니다. 판단 품질까지 함께 보면 Mi:dm이 가장 빠르면서 틀린 판단이 없었습니다. 실제 대화에서 사용자가 기다리는 시간은 여기에 말끝 판정 0.9초와 받아쓰기 약 1.6초가 더해집니다."),
        single_chart("② 판단 정확도", "채점 대상 문장(목적지·돌려 말하기·대답) 중 맞힌 비율",
                     acc_data, " %", lambda v: f"{v:.0f}", "high",
                     "<b>표본 크기가 다릅니다</b>(Mi:dm 18문장, 나머지 3~4문장). 막대에 마우스를 올리면 맞힌 수/채점 수가 보입니다. Mi:dm은 가장 많이 시험해 틀린 문장도 드러났습니다(“쉬고 싶은데 어디로 가야 돼?” 2회 되묻기, 대기 시간 “한 5분?”을 식당 안내로, 잡음 “어. 다.”를 식당으로). 표본이 3~4문장인 모델의 비율과 그대로 비교하기는 어렵습니다. 순위보다 <b>어떤 말을 놓쳤는지</b>가 중요합니다. EXAONE 3.5는 정확히 받아 적은 “안내소 가고 싶어요”를, EXAONE 4.0은 “네. 좋아요.”를 놓쳤습니다."),
    ])

    # ---------- 메모리 ----------
    mem_rows = ""
    okey = {"exaone3.5_2.4b": "exaone3.5:2.4b", "midm2-mini": "midm2-mini", "gemma4-e2b-text": "gemma4-e2b-text"}
    for k in KEYS:
        o = own.get(okey.get(k, ""))
        if o:
            mem_rows += (f'<tr><td>{sw(k)}{html.escape(INFO[k]["name"])}</td><td>{INFO[k]["quant"]}</td>'
                         f'<td class="n">{INFO[k]["size"]:.2f} GB</td><td class="n">{o["gpu_mb"] / 1024:.2f}</td>'
                         f'<td class="n">{(o["cpu_anon_mb"] + o["cpu_file_mb"]) / 1024:.2f}</td>'
                         f'<td class="n b">{o["total"] / 1024:.2f} GB</td><td>{INFO[k]["lic"]}</td></tr>')
        else:
            mem_rows += (f'<tr><td>{sw(k)}{html.escape(INFO[k]["name"])}</td><td>{INFO[k]["quant"]}</td>'
                         f'<td class="n">{INFO[k]["size"]:.2f} GB</td><td colspan="3" class="dim">미측정</td><td>{INFO[k]["lic"]}</td></tr>')

    # ---------- 벤치 요약표 ----------
    camp = summaries("")
    camp_on = summaries("_stack")
    bench_rows = ""
    for k in sorted(KEYS, key=lambda x: -s0630[x]["accuracy"]):
        c, c_on, z = camp.get(k), camp_on.get(k), s0630[k]
        cacc = f'{c["accuracy"] * 100:.1f} %' if c else "–"
        cwall = f'{c["wall_mean_s"]:.2f}초' if c else "–"
        conwall = f'{c_on["wall_mean_s"]:.2f}초' if c_on else "–"
        bench_rows += (f'<tr><td>{sw(k)}{html.escape(INFO[k]["name"])}</td>'
                       f'<td class="n">{cacc}</td><td class="n">{cwall}</td><td class="n">{conwall}</td>'
                       f'<td class="n b">{z["accuracy"] * 100:.1f} %</td><td class="n">{z["wall_mean_s"]:.2f}초</td>'
                       f'<td class="n">{z["wall_p90_s"]:.2f}초</td><td class="n">{z["gen_tok_s_mean"]:.0f}</td>'
                       f'<td>{INFO[k]["quant"]}</td><td>{INFO[k]["lic"]}</td></tr>')

    # ---------- 종합 ----------
    rank_rows = ""
    for k in ORDER:
        rows, judged = fstat(k)
        ok = sum(r["ok"] for r in judged)
        v, cls = VERDICT[k]
        camp_acc = f'{camp[k]["accuracy"] * 100:.1f} %' if k in camp else "–"
        field_score = f"{ok} / {len(judged)}" if judged else "–"
        rank_rows += (f'<tr><td>{sw(k)}{html.escape(INFO[k]["name"])}</td>'
                      f'<td class="n">{camp_acc}</td>'
                      f'<td class="n">{s0630[k]["accuracy"] * 100:.1f} %</td>'
                      f'<td class="n">{field_score}</td>'
                      f'<td class="n">{s0630[k]["wall_mean_s"]:.2f}초</td><td>{INFO[k]["lic"]}</td>'
                      f'<td><span class="chip {cls}">{v}</span></td></tr>')

    head = R.TEMPLATE.split("<main>")[0].replace("{{", "{").replace("}}", "}")
    head = head.replace("<title>VICA 로컬 LLM 비교</title>", "<title>VICA 로컬 LLM 비교</title>")
    extra_css = ("<style>.chip.part{background:var(--partbg);color:var(--part)} .step{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px}"
                 ".step div{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px} .step b{display:block;font-size:15px}"
                 ".step span{font-size:13px;color:var(--mute)} h2 .no{font:500 13px 'IBM Plex Mono',monospace;color:var(--accent);margin-right:8px}</style>")
    legend = "".join(f'<span>{sw(k)}{html.escape(INFO[k]["name"])}</span>' for k in KEYS)
    return head + extra_css + f"""<main>
<header>
 <div class="eyebrow">VICA · 네트워크 없는 곳의 음성 안내 · 2026-09-28</div>
 <h1>로컬 LLM 폴백 후보 5종 비교 — Mi:dm 채택</h1>
 <p>인터넷이 끊겼을 때 로봇 안에서 “어디로 가고 싶은지”를 알아듣는 작은 모델을 고르는 시험입니다. 벤치 두 번과 마이크 실물 시험 한 번, 모두 세 단계로 쟀습니다. 결론은 <b>Mi:dm 2.0 Mini로 바꾸는 것</b>입니다. 벤치 점수는 Gemma가 높았지만, 실제로 말을 걸어 본 실물 시험에서 Mi:dm이 장소를 더 잘 짚고 더 빨랐습니다.</p>
 <div class="meta"><span>Jetson Orin NX 16 GB</span><span>Ollama 0.30.6 · GPU 적재</span><span>문맥 4096 · temperature 0 · 생각 모드 끔</span><span>실제 음성 파이프라인과 같은 경로(목적지 목록 프롬프트 → JSON 의도)</span></div>
</header>

<section>
 <h2><span class="no">00</span>시험 개요</h2>
 <div class="step">
  <div><b>벤치 1 · 캠퍼스 예시</b><span>목적지 28곳 · 24문항 × 3회 · 스택 꺼짐/켜짐</span></div>
  <div><b>벤치 2 · 0630 지도</b><span>실제 목적지 6곳 · 26문항 × 3회 · 스택 꺼짐</span></div>
  <div><b>실물 · 마이크</b><span>0630 지도 · 스택 켬 · 사람이 직접 말함 · 16:40~19:00 (새 터미널로 모든 노드를 켠 뒤)</span></div>
 </div>
 <p class="note">벤치는 글자로 질문해 모델의 판단만 봅니다. 실물 시험은 마이크 → whisper 받아쓰기 → LLM 전체를 거칩니다. 실물은 새 터미널에서 모든 노드를 제대로 켠 16:40 이후 기록만 씁니다. 그 전 시험은 일부 노드가 제대로 켜지지 않아 비교에서 뺐습니다. 뒤 단계로 갈수록 실제 사용 조건에 가깝습니다. 결론은 실물 시험에서 사용자가 직접 겪은 판단 품질과 속도를 가장 무겁게 보았습니다.</p>
</section>

<div class="legend">{legend}</div>

<section><h2><span class="no">01</span>벤치 결과 — 글자로 물은 판단 시험</h2>
 <p>음성 없이 문장을 글자로 넣어 모델의 판단만 쟀습니다. 벤치 1은 캠퍼스 예시 목적지 28곳에 24문항, 벤치 2는 로봇이 실제로 쓰는 0630 지도 6곳에 26문항이고, 두 벤치 모두 문항마다 3번씩 물었습니다. 표는 벤치 2 정확도 순서입니다.</p>
 <div class="tbl"><table><thead><tr><th rowspan="2">모델</th><th colspan="3">벤치 1 · 캠퍼스 28곳</th><th colspan="4">벤치 2 · 0630 지도 6곳</th><th rowspan="2">압축</th><th rowspan="2">라이선스</th></tr>
 <tr><th>정확도</th><th>응답(스택 꺼짐)</th><th>응답(스택 켬)</th><th>정확도</th><th>평균 응답</th><th>느린 10 %</th><th>tok/s</th></tr></thead><tbody>{bench_rows}</tbody></table></div>
 <p class="note">정확도 = 의도와 목적지를 모두 맞힌 비율. 응답 = 질문을 넣고 판단이 나오기까지의 평균 초. 느린 10 % = 가장 느린 답 10 %가 걸린 시간(p90). tok/s = 답을 쓰는 속도. Qwen3.5 2B는 나중에 추가해 벤치 2만 쟀습니다(음성 스택을 끈 날 저녁, Gemma와 같은 회차).</p>
 <h3 style="margin-top:18px">1-1 · 캠퍼스 28곳 (흐린 막대 = 스택 꺼짐, 진한 막대 = 스택 켬)</h3>
 <div class="charts">{b1}</div>
 <h3 style="margin-top:18px">1-2 · 실제 0630 지도 6곳</h3>
 <div class="charts">{b2}</div>
</section>

<section><h2><span class="no">02</span>실물 결과 — 마이크로 직접 말하기</h2>
 <p>새 터미널에서 주행 스택과 음성 노드를 모두 제대로 켠 뒤(16:40~19:00), 다섯 모델을 차례로 올려 사람이 직접 말을 걸었습니다. 그 전 시험은 일부 노드가 켜지지 않아 비교에서 뺐습니다.</p>
 <div class="tbl"><table><thead><tr><th>모델</th><th>목적지 목록</th><th>LLM 판단</th><th>목적지</th><th>돌려 말하기</th><th>대답</th><th>잡음 무시</th><th>채점 합계</th><th>평균 응답</th><th>가장 느린</th><th>판정</th></tr></thead><tbody>{frows}</tbody></table></div>
 <p class="note">LLM 판단 = LLM이 직접 판단한 문장 수. 호출어 오인이나 STT 오류로 LLM까지 가지 못한 말, 그리고 “네·아니·음”처럼 코드가 먼저 처리한 짧은 대답은 뺐습니다. 채점은 정답이 분명한 문장만 했습니다(“배가 아파요 → 화장실”처럼 정답을 정하기 어려운 문장은 제외). 응답 = 받아쓰기가 끝난 순간부터 판단이 나오기까지. Qwen과 Mi:dm 2차는 목적지가 8곳으로 늘어난 뒤(민원실·식당 추가) 시험해서 “배고파”의 정답이 식당입니다. Mi:dm은 2차(18:37~19:00)에서 가장 오래 시험했고(22문장), 실제로 네 번 주행해 입구·민원실·식당·휴게실에 도착했습니다. 잡음 무시 = 사용자에게 한 말이 아닌 소리(옆 사람 대화, “어. 다.”)를 목적지로 읽지 않았는지.</p>
 <div class="charts">{b3}</div>
 {field_detail}
</section>

<section><h2><span class="no">03</span>종합 — 세 단계를 한 표로</h2>
 <div class="tbl"><table><thead><tr><th>모델</th><th>벤치 1 정확도</th><th>벤치 2 정확도</th><th>실물 채점</th><th>벤치 2 응답</th><th>라이선스</th><th>판정</th></tr></thead><tbody>{rank_rows}</tbody></table></div>
 <div class="concl">
  <p><b>Mi:dm 2.0 Mini 채택.</b> 가장 오래 시험한 모델입니다(18문장 채점, 13개 정답). 목적지 이름은 모두 맞혔고, 목적만 말한 요청도 10번 중 7번 장소를 짚었습니다(서류 제출 → 민원실, 배고파 → 식당, 다리가 아파 앉고 싶어 → 휴게실). 실제로 네 번 주행해 모두 도착했고, 판단은 평균 1.7초로 Gemma(2.5초)보다 빨랐습니다. 직접 말을 걸어 본 결과 추론과 응답속도 모두 가장 좋았습니다. 상업 라이선스(MIT)이고, 메모리는 Gemma와 같습니다.</p>
  <p><b>Gemma 4 E2B는 예비.</b> 벤치 정확도는 가장 높았지만(0630 84.6 %), 실물에서 장소 유추가 약했습니다. 같은 뜻의 두 문장 중 “나 다리가 너무 아파, 쉬고 싶어”를 휴게실이 아니라 일시정지로 받았고, 판단도 네 모델 중 가장 느렸습니다(약 2.5초).</p>
  <p><b>Mi:dm을 쓸 때 조심할 점.</b> 애매한 말을 목록의 목적지로 끼워 맞추는 경향이 있습니다. 0630 벤치에서는 세미나실을 방2로 읽었고, 실물에서는 대기 시간 대답 “한 5분?”과 잡음 “어. 다.”를 식당 안내로 읽었습니다. “쉬고 싶은데 어디로 가야 돼?”처럼 이유 없이 쉬고 싶다고만 하면 휴게실을 놓쳤습니다. 목적지 안내는 늘 확인 질문을 거치므로 “아니요”로 되돌릴 수 있습니다. 목적지 별칭에 “쉬는 곳”, “서류 제출” 같은 말을 넣으면 연결이 더 좋아질 것으로 봅니다.</p>
  <p><b>EXAONE 3.5 · 4.0은 제외.</b> 기본 목적지나 분명한 승낙을 놓쳤고, 비상업 라이선스입니다.</p>
  <p><b>Qwen3.5 2B도 제외.</b> 같은 회차의 Gemma보다 15 %p 낮고(69.2 % 대 84.6 %), 1초 느리며, GPU 메모리를 1 GB 더 씁니다(2.70 GB 대 1.70 GB). 실물에서도 돌려 말한 휴게실·민원실을 놓치고 “화장실이 있어요”처럼 엉뚱한 장소를 말했으며, 모델을 올리는 데 20초가 걸렸습니다. 공식 Ollama 판은 JSON 형식을 강제하지 못해 그대로는 쓸 수 없고, 형식 강제 틀로 다시 등록해야 합니다.</p>
 </div>
 <p class="note">모델과 무관한 문제가 실물 시험에서 드러났습니다. 호출어 감지기가 정확히 받아 적은 대답(“네. 좋아요.”, “배고파요”)까지 “비카야”로 덮어쓰고, 인사 질문 뒤 짧은 “네”가 “대”로 받아 적혔습니다. 이 문제들 때문에 LLM까지 가지 못한 말이 많았습니다. 모델을 바꿔도 풀리지 않는 귀(STT·호출어) 쪽 과제로 따로 다룹니다. 같은 날 “음” 한마디를 출발 승낙으로 읽던 규칙은 되묻기로 고쳐 실물에서 확인했습니다.</p>
</section>
<section><h2><span class="no">부록</span>메모리 — LLM이 혼자 쥐는 양 (스택 켬)</h2>
 <div class="tbl"><table><thead><tr><th>모델</th><th>압축</th><th>파일</th><th>GPU</th><th>CPU</th><th>LLM 합계</th><th>라이선스</th></tr></thead><tbody>{mem_rows}</tbody></table></div>
 <p class="note">LLM 프로세스 하나가 쥔 메모리입니다. GPU 몫은 jtop, CPU 몫은 프로세스 메모리 지도(PSS)로 쟀습니다. 젯슨은 CPU와 GPU가 메모리를 나눠 씁니다. 스택만 켰을 때 남은 메모리는 5.6 GB였고, 어느 모델을 올리든 스왑이 0.8 GB에서 2.4 GB로 늘었습니다. Mi:dm과 Gemma의 점유량은 거의 같습니다. Qwen3.5 2B는 벤치 중 GPU 몫만 쟀고(2.70 GB, Gemma 1.70 GB), EXAONE 4.0은 재지 않았습니다.</p>
</section>
</main>
"""


if __name__ == "__main__":
    Path(sys.argv[1]).write_text(build())
    print("written", sys.argv[1])
