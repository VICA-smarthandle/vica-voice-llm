"""시험은 운영 `.env` 를 따르지 않는다.

`src.langchain_intent_parser` 가 임포트 시점에 `load_dotenv()` 를 부른다.
그래서 젯슨의 운영 설정이 시험 결과를 바꿨고, **임포트 순서에 따라
달라지기까지 했다** — 2026-09-02 에 `.env` 로 barge-in AND 를 켜자 그
설정과 무관한 방향 관문 시험 7건이 무너졌다.

시험이 검증해야 하는 것은 **코드의 기본값**이다. 운영값을 바꿨다고 시험이
빨개지면 그 시험은 회귀를 못 잡는다. `load_dotenv` 는 이미 있는 환경변수를
덮어쓰지 않으므로, 여기서 미리 박아 두면 `.env` 가 무엇이든 같은 값으로
돈다. 특정 값을 시험하려면 그 시험 안에서 monkeypatch 로 모듈 상수를 바꾼다
(예: `TestRequireBothSignals`).
"""
import os

# 코드 기본값과 같은 값. 여기 없는 변수는 시험이 쓰지 않는 것이다.
_DEFAULTS = {
    "VICA_BARGE_REQUIRE_BOTH": "0",
    "VICA_BARGE_MIN_HITS": "5",
    "VICA_BARGE_DOA_GATE": "0",
    "VICA_USER_DOA_CENTER": "",
    "VICA_USER_DOA_WIDTH": "45",
    "VICA_LISTEN_MAX_SEC": "6.0",
    "VICA_LISTEN_END_SEC": "0.8",
    "VICA_LISTEN_MIN_OPEN_SEC": "2.5",
    "VICA_LISTEN_BLIP_VOID_SEC": "0.32",
    "VICA_CONFIRM_WINDOW_SEC": "30.0",
}

for _key, _value in _DEFAULTS.items():
    os.environ[_key] = _value
