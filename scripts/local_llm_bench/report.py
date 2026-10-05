#!/usr/bin/env python3
"""bench.py 결과(results/*.json)를 그래프 페이지 한 장으로 굽는다.

사용: python3 scripts/local_llm_bench/report.py <출력.html> [결론 문단 HTML 파일]
그래프 3종(응답시간·생성속도·정확도) + 메모리·분류별 표. 외부 라이브러리 없이 SVG.
"""
import html
import json
import sys
from pathlib import Path

RES = Path(__file__).resolve().parent / "results"

# 표시 이름·라이선스·파일 크기. 순서 = 그래프 순서 = 색 슬롯 순서(고정).
MODELS = [
    ("exaone3.5_2.4b", "EXAONE 3.5 2.4B", "Q4_K_M", "비상업", 1.64, "EXAONE 3.5"),
    ("exaone4-1.2b", "EXAONE 4.0 1.2B", "Q8_0", "비상업", 1.36, "EXAONE 4.0"),
    ("midm2-mini", "Mi:dm 2.0 Mini 2.3B", "Q4_K_M", "MIT", 1.43, "Mi:dm Mini"),
    ("gemma4-e2b-text", "Gemma 4 E2B (현 폴백)", "Q4_K_M", "Apache 2.0", 3.11, "Gemma E2B"),
    ("qwen3.5-2b-text", "Qwen3.5 2B", "Q8_0", "Apache 2.0", 2.74, "Qwen3.5 2B"),
]
OWN_KEY = {"exaone3.5_2.4b": "exaone3.5:2.4b", "midm2-mini": "midm2-mini", "gemma4-e2b-text": "gemma4-e2b-text"}


def load():
    out = []
    own_f = RES / "own_mem_stack.json"
    own = json.loads(own_f.read_text()) if own_f.exists() else {}
    for key, name, quant, lic, size, short in MODELS:
        f = RES / f"{key}.json"
        if f.exists():
            d = json.loads(f.read_text())
            fs = RES / f"{key}_stack.json"
            d["stack"] = json.loads(fs.read_text()) if fs.exists() else None
            d["own"] = own.get(OWN_KEY.get(key, ""))
            out.append({"key": key, "name": name, "quant": quant, "lic": lic, "size": size, "short": short, **d})
    return out


def bar_chart(title, sub, items, unit, fmt, best, extra=None):
    """가로 막대. items = [(slot, name, value, tooltip, faded)]. best = 'low' | 'high'.
    faded = 스택 꺼짐(흐린 막대), 아니면 스택 켜짐(진한 막대)."""
    vmax = max(x[2] for x in items) or 1
    if extra:
        vmax = max(vmax, extra[1])
    vmax *= 1.12
    row_h, left, right, top = 34, 170, 70, 8
    w, h = 640, top + row_h * len(items) + 30
    plot = w - left - right
    pick = (min if best == "low" else max)(items, key=lambda x: x[2])[1]
    s = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{html.escape(title)}">']
    # 격자 4칸
    for i in range(5):
        x = left + plot * i / 4
        val = vmax * i / 4
        s.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{h - 26}" class="grid"/>')
        s.append(f'<text x="{x:.1f}" y="{h - 8}" class="tick" text-anchor="middle">{fmt(val)}</text>')
    for i, (slot, name, v, tip, faded) in enumerate(items):
        y = top + i * row_h + 6
        bw = max(plot * v / vmax, 2)
        s.append(f'<g class="bar" tabindex="0"><title>{html.escape(tip)}</title>'
                 f'<rect x="{left}" y="{y - 4}" width="{plot + right}" height="{row_h - 4}" class="hit"/>'
                 f'<text x="{left - 12}" y="{y + 14}" class="lab{" sub" if faded else ""}" text-anchor="end">{html.escape(name)}</text>'
                 f'<rect x="{left}" y="{y}" width="{bw:.1f}" height="20" rx="4" '
                 f'style="fill:var(--s{slot}){";fill-opacity:.38" if faded else ""}"/>'
                 f'<text x="{left + bw + 8:.1f}" y="{y + 15}" class="val{" best" if name == pick else ""}">'
                 f'{fmt(v)}{unit}</text></g>')
    if extra:
        lab, val = extra
        x = left + plot * val / vmax
        s.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{h - 26}" class="ref"/>'
                 f'<text x="{x - 4:.1f}" y="{top + 10}" class="reflab" text-anchor="end">{html.escape(lab)}</text>')
    s.append("</svg>")
    return (f'<figure class="chart"><figcaption><h3>{html.escape(title)}</h3><p>{sub}</p></figcaption>'
            + "".join(s) + "</figure>")


def page(data, conclusion_html):
    def items(f, tip):
        out = []
        for i, d in enumerate(data):
            out.append((i + 1, f'{d["short"]} · 꺼짐', f(d["summary"]), "스택 꺼짐 — " + tip(d, d["summary"]), True))
            if d.get("stack"):
                st = d["stack"]["summary"]
                out.append((i + 1, f'{d["short"]} · 켜짐', f(st), "스택 켜짐 — " + tip(d, st), False))
        return out
    S = lambda d: d["summary"]
    c_wall = bar_chart(
        "① 응답시간", "말이 끝난 뒤 의도가 나오기까지 걸린 평균 시간 (짧을수록 좋음). 흐린 막대 = 스택 꺼짐, 진한 막대 = 스택 켜짐. 막대에 마우스를 올리면 p90·최악·첫 호출이 보입니다.",
        items(lambda x: x["wall_mean_s"],
              lambda d, x: f'{d["name"]}: 평균 {x["wall_mean_s"]:.2f}초 · p90 {x["wall_p90_s"]:.2f}초 · '
                           f'최악 {x["wall_max_s"]:.2f}초 · 첫 호출 {x["cold_first_s"]:.1f}초'),
        "초", lambda v: f"{v:.1f}", "low")
    c_gen = bar_chart(
        "② 생성속도", "답을 한 글자씩 써 내려가는 속도 (tokens/s, 높을수록 좋음).",
        items(lambda x: x["gen_tok_s_mean"] or 0,
              lambda d, x: f'{d["name"]}: {x["gen_tok_s_mean"]} tok/s · 답 길이 평균 {x["out_tokens_mean"]} 토큰 · '
                           f'입력 {x["prompt_tokens_mean"]} 토큰'),
        "", lambda v: f"{v:.0f}", "high")
    c_acc = bar_chart(
        "③ 정확도", f'24문항 × {data[0]["summary"]["rounds"]}회, 의도·목적지를 모두 맞힌 비율 (높을수록 좋음).',
        items(lambda x: x["accuracy"] * 100,
              lambda d, x: f'{d["name"]}: {x["accuracy"] * 100:.1f} % (1회차 {x["correct_round0"]}/24)'),
        " %", lambda v: f"{v:.0f}", "high")

    def mem_row(i, d):
        o, st = d.get("own"), (d.get("stack") or {}).get("summary")
        sw = f'<span class="sw" style="background:var(--s{i + 1})"></span>{html.escape(d["name"])}'
        if not o:
            return (f'<tr><td>{sw}</td><td>{d["quant"]}</td><td class="n">{d["size"]:.2f} GB</td>'
                    f'<td colspan="5" class="dim">스택 켠 시험 안 함 (정확도 탈락)</td><td>{d["lic"]}</td></tr>')
        return (f'<tr><td>{sw}</td><td>{d["quant"]}</td><td class="n">{d["size"]:.2f} GB</td>'
                f'<td class="n">{o["gpu_mb"] / 1024:.2f}</td><td class="n">{o["cpu_anon_mb"] / 1024:.2f}</td>'
                f'<td class="n">{o["cpu_file_mb"] / 1024:.2f}</td><td class="n b">{o["total"] / 1024:.2f} GB</td>'
                f'<td class="n">{st["cold_first_s"]:.1f} 초</td><td>{d["lic"]}</td></tr>')
    mem_rows = "".join(mem_row(i, d) for i, d in enumerate(data))

    cats = []
    for r in data[0]["rows"]:
        if r["category"] not in cats:
            cats.append(r["category"])
    cat_head = "".join(f"<th>{html.escape(c)}</th>" for c in cats)
    cat_rows = ""
    for i, d in enumerate(data):
        cells = ""
        for c in cats:
            rs = [r for r in d["rows"] if r["category"] == c]
            ok = sum(r["ok"] for r in rs)
            cls = "full" if ok == len(rs) else ("zero" if ok == 0 else "part")
            cells += f'<td class="n {cls}">{ok}/{len(rs)}</td>'
        cat_rows += (f'<tr><td><span class="sw" style="background:var(--s{i + 1})"></span>'
                     f'{html.escape(d["name"])}</td>{cells}</tr>')

    wrong = ""
    for i, d in enumerate(data):
        misses = {}
        for r in d["rows"]:
            if not r["ok"]:
                k = (r["text"], r["intent"], r["dest"] or "-", r["need_confirm"])
                misses[k] = misses.get(k, 0) + 1
        items_html = "".join(
            f'<li>“{html.escape(t)}” → {html.escape(it)} / {html.escape(dst)}'
            f'{" · 확인 다시 물음" if nc else ""} <span class="times">×{n}</span></li>'
            for (t, it, dst, nc), n in misses.items()) or "<li>틀린 문항 없음</li>"
        wrong += (f'<details><summary><span class="sw" style="background:var(--s{i + 1})"></span>'
                  f'{html.escape(d["name"])} — 틀린 답 {sum(misses.values())}건</summary><ul>{items_html}</ul></details>')

    legend = "".join(f'<span><span class="sw" style="background:var(--s{i + 1})"></span>{html.escape(d["name"])}</span>'
                     for i, d in enumerate(data))
    return TEMPLATE.format(legend=legend, c_wall=c_wall, c_gen=c_gen, c_acc=c_acc, mem_rows=mem_rows,
                           cat_head=cat_head, cat_rows=cat_rows, wrong=wrong,
                           conclusion=conclusion_html, rounds=data[0]["summary"]["rounds"])


TEMPLATE = """<title>VICA 로컬 LLM 비교</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+KR:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{{--bg:#f6f7f8;--card:#ffffff;--ink:#14181c;--ink2:#4d5660;--mute:#7a838c;--line:#dfe3e7;--grid:#e8ebee;
--accent:#1f6f8b;--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#c98500;--s5:#e87ba4;--ok:#1b7f4b;--okbg:#e3f3ea;--bad:#b3261e;--badbg:#fbe7e5;--part:#8a5a00;--partbg:#fdf1d9}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{color-scheme:dark;--bg:#121517;--card:#1a1e21;--ink:#eef1f3;--ink2:#b5bdc4;--mute:#8a939b;--line:#2c3237;--grid:#262b30;
--accent:#6cc0dc;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--ok:#6fd49a;--okbg:#17301f;--bad:#ff8a80;--badbg:#3a1a18;--part:#f2c46b;--partbg:#35290f}}}}
:root[data-theme="dark"]{{color-scheme:dark;--bg:#121517;--card:#1a1e21;--ink:#eef1f3;--ink2:#b5bdc4;--mute:#8a939b;--line:#2c3237;--grid:#262b30;
--accent:#6cc0dc;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--ok:#6fd49a;--okbg:#17301f;--bad:#ff8a80;--badbg:#3a1a18;--part:#f2c46b;--partbg:#35290f}}
body{{background:var(--bg);color:var(--ink);font:15px/1.65 "IBM Plex Sans KR",system-ui,sans-serif;padding-inline:16px;padding-block:28px 56px}}
main{{max-width:880px;margin:0 auto;display:grid;gap:28px}}
header .eyebrow{{font:500 12px "IBM Plex Mono",monospace;letter-spacing:.08em;color:var(--accent);text-transform:uppercase}}
h1{{font-size:28px;margin:.2em 0 .3em;text-wrap:balance}} h2{{font-size:19px;margin:0 0 10px}} h3{{font-size:16px;margin:0}}
p{{margin:.3em 0;max-width:68ch;color:var(--ink2)}}
.meta{{display:flex;flex-wrap:wrap;gap:6px 18px;font:13px "IBM Plex Mono",monospace;color:var(--mute)}}
.concl{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:18px 20px}}
.concl p,.concl li{{color:var(--ink)}} .concl ul{{margin:.4em 0;padding-left:1.2em}}
.legend{{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:14px}}
.sw{{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:7px;vertical-align:-1px}}
.charts{{display:grid;gap:22px}}
.chart{{margin:0;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 18px 8px}}
.chart figcaption p{{font-size:13.5px}}
.chart svg{{width:100%;height:auto;display:block;margin-top:8px;font-family:"IBM Plex Sans KR",sans-serif}}
.grid{{stroke:var(--grid);stroke-width:1}} .tick{{fill:var(--mute);font:12px "IBM Plex Mono",monospace}}
.lab{{fill:var(--ink);font-size:13.5px}} .lab.sub{{fill:var(--mute)}} td.b{{font-weight:700}} td.dim{{color:var(--mute)}} .val{{fill:var(--ink2);font:500 13.5px "IBM Plex Mono",monospace}} .val.best{{fill:var(--ink);font-weight:700}}
.hit{{fill:transparent}} .bar:hover .hit,.bar:focus .hit{{fill:var(--grid)}} .bar:focus{{outline:none}}
.ref{{stroke:var(--bad);stroke-width:1.5;stroke-dasharray:4 3}} .reflab{{fill:var(--bad);font-size:12px}}
.tbl{{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:10px}}
table{{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}}
th,td{{padding:9px 12px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}}
th{{font-weight:500;color:var(--mute);font-size:12.5px}} tr:last-child td{{border-bottom:0}}
td.n{{text-align:right;font-family:"IBM Plex Mono",monospace}}
td.full{{color:var(--ok)}} td.zero{{color:var(--bad);font-weight:700}} td.part{{color:var(--part)}}
.chip{{font-size:12px;padding:2px 8px;border-radius:99px}} .chip.ok{{background:var(--okbg);color:var(--ok)}} .chip.over{{background:var(--badbg);color:var(--bad)}}
details{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 16px;margin-bottom:8px}}
summary{{cursor:pointer;font-weight:500}} details ul{{margin:8px 0 4px;padding-left:1.2em;font-size:14px;color:var(--ink2)}}
.times{{font-family:"IBM Plex Mono",monospace;color:var(--mute)}}
.note{{font-size:13.5px;color:var(--mute)}}
@media (max-width:520px){{h1{{font-size:23px}}}}
</style>
<main>
<header>
 <div class="eyebrow">VICA · 네트워크 없는 곳의 음성 안내 · 2026-09-28</div>
 <h1>로컬 LLM 폴백 후보 4종 비교 — Gemma 유지</h1>
 <p>인터넷이 끊겼을 때 로봇 안에서 “어디로 가고 싶은지”를 알아듣는 작은 모델을 고르는 시험입니다. 실제 음성 파이프라인과 같은 경로(목적지 목록이 든 시스템 프롬프트 → JSON 의도)로 24문항을 {rounds}번씩 물었습니다.</p>
 <div class="meta"><span>Jetson Orin NX 16 GB</span><span>Ollama 0.30.6 · GPU 적재</span><span>주행 스택 꺼짐·켜짐 두 번 측정</span><span>문맥 4096 · temperature 0 · 생각 모드 끔</span></div>
</header>
<section class="concl">{conclusion}</section>
<div class="legend">{legend}</div>
<section class="charts">{c_wall}{c_gen}{c_acc}</section>
<section>
 <h2>메모리 — LLM이 혼자 쥐는 양 (스택 켠 상태)</h2>
 <p>Claude·원격 접속 등 다른 프로그램과 상관없이, LLM을 돌리는 프로세스 하나가 쥔 메모리만 셌습니다. 젯슨은 CPU와 GPU가 한 메모리를 나눠 써서, GPU 몫은 jtop이 읽는 프로세스별 GPU 메모리로, CPU 몫은 프로세스의 작업 공간과 디스크에서 빌려 온 가중치로 나눴습니다.</p>
 <div class="tbl"><table><thead><tr><th>모델</th><th>압축</th><th>파일</th><th>GPU</th><th>CPU 작업</th><th>CPU 가중치</th><th>LLM 합계</th><th>첫 호출</th><th>라이선스</th></tr></thead><tbody>{mem_rows}</tbody></table></div>
 <p class="note">스택만 켠 상태에서 남은 메모리는 5.6 GB, 스왑 사용은 0.8 GB였습니다. 세 모델을 차례로 올리고 내리는 동안 스왑이 2.4 GB까지 늘고 되돌아오지 않았습니다. 메모리가 모자라 스택 쪽 일부가 디스크로 밀려났다는 뜻입니다. 응답시간과 정확도는 나빠지지 않았지만, 첫 호출(모델 적재)은 스택 켠 상태에서 12~15초로 길어졌습니다.</p>
</section>
<section>
 <h2>문항 종류별 정답</h2>
 <div class="tbl"><table><thead><tr><th>모델</th>{cat_head}</tr></thead><tbody>{cat_rows}</tbody></table></div>
 <p class="note">오인식 = STT가 잘못 받아 적은 말(“보건시”, “카패”). 확인수락 = “화장실로 안내해드릴까요?” 뒤에 “그쪽으로 데려다 주세요”. 확인정정 = “카페 말고 보건실로”. 짧은 “네/아니요”는 코드가 먼저 처리해 LLM까지 오지 않으므로 넣지 않았습니다.</p>
</section>
<section>
 <h2>틀린 답 모아 보기</h2>
 {wrong}
</section>
</main>
"""

if __name__ == "__main__":
    out = Path(sys.argv[1])
    concl = Path(sys.argv[2]).read_text() if len(sys.argv) > 2 else "<p>결론 작성 전</p>"
    out.write_text(page(load(), concl))
    print("written", out)
