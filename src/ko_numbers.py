"""한글로 읽은 숫자('사백칠호')를 숫자('407호')로 바꾼다 — 비교용 정리."""
from __future__ import annotations

import re
from typing import Optional

_DIGIT = {"영": 0, "공": 0, "일": 1, "이": 2, "삼": 3, "사": 4, "오": 5, "육": 6, "칠": 7, "팔": 8, "구": 9}
_MULT = {"십": 10, "백": 100, "천": 1000}
_RUN = re.compile("[" + "".join(_DIGIT) + "".join(_MULT) + "]+")
_UNITS = "호층번"


def _read_value(run: str) -> Optional[int]:
    """'사백칠' → 407. 숫자 낱자 둘이 곱수 없이 붙으면 모호하니 None."""
    total, current = 0, None
    for ch in run:
        if ch in _DIGIT:
            if current is not None:
                return None
            current = _DIGIT[ch]
        else:
            total += (1 if current is None else current) * _MULT[ch]
            current = None
    return total + (current or 0)


def korean_numbers_to_digits(text: Optional[str]) -> str:
    """글자 속 한글 숫자 읽기를 숫자로 바꾼 사본. 바꿀 게 없으면 그대로."""
    if not text:
        return ""
    out, last = [], 0
    for m in _RUN.finditer(text):
        run = m.group()
        after = text[m.end():m.end() + 1]
        has_mult = any(ch in _MULT for ch in run)
        has_digit = any(ch in _DIGIT for ch in run)
        replacement = None
        if has_mult and (has_digit and len(run) >= 2 or after and after in _UNITS):
            value = _read_value(run)
            replacement = None if value is None else str(value)
        elif not has_mult and after and after in _UNITS:
            replacement = "".join(str(_DIGIT[ch]) for ch in run)
        if replacement is not None:
            out.append(text[last:m.start()])
            out.append(replacement)
            last = m.end()
    out.append(text[last:])
    return "".join(out)
