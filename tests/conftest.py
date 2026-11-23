"""시험은 운영 `.env` 를 따르지 않는다."""
import os
import sys
from unittest.mock import MagicMock

sys.modules['vica_interfaces'] = MagicMock()
sys.modules['vica_interfaces.msg'] = MagicMock()

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
    "VICA_APPROACH_MIN_RMS": "",
    "VICA_LISTEN_MIN_RMS": "",
    "VICA_SHORT_ANSWER_MIN_RMS": "",
    "VICA_LISTEN_BAND_MIN": "",
    "VICA_WAKE_THRESHOLD": "",
    "VICA_EMERGENCY_SINGLE_FRAME": "",
    "VICA_MIC_AGC_DESIRED": "0.010",
    "VICA_MIC_AGC_MAX_GAIN": "",
    "VICA_MIC_GAMMA_ETAIL": "",
    "VICA_MIC_NLATTEN": "",
    "VICA_STT_ENGINE": "whisper",
    "VICA_VERIFY_STT_ENGINE": "whisper",
    "VICA_SW_AEC": "off",
    "VICA_TOOLS_TEXT": "off",
    "VICA_TOOLS_AUDIO": "off",
    "VICA_ANSWER": "off",
    "VICA_ANSWER_LOCAL": "off",
    "VICA_RECUT": "off",
    "VICA_TALK": "off",
}

for _key, _value in _DEFAULTS.items():
    os.environ[_key] = _value
