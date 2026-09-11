"""시험은 운영 `.env` 를 따르지 않는다."""
import os

_DEFAULTS = {
    "VICA_BARGE_REQUIRE_BOTH": "0",
    "VICA_BARGE_MIN_HITS": "5",
    "VICA_BARGE_DOA_GATE": "0",
    "VICA_USER_DOA_CENTER": "",
    "VICA_USER_DOA_WIDTH": "45",
    "VICA_LISTEN_MAX_SEC": "15.0",
    "VICA_LISTEN_END_SEC": "0.8",
    "VICA_LISTEN_MIN_OPEN_SEC": "2.5",
    "VICA_LISTEN_BLIP_VOID_SEC": "0.32",
    "VICA_CONFIRM_WINDOW_SEC": "30.0",
}

for _key, _value in _DEFAULTS.items():
    os.environ[_key] = _value
