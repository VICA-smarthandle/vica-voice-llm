# 장애물 안내 녹음 (점검 도구용, 2026-10-08)

사용자가 [장애물 안내 멘트](https://claude.ai/artifact/LyeYU1iyZrH2XffyfqVA8H) 페이지에서 후보 20개를 듣고 고른 것.
설계서: 루트 docs/superpowers/specs/2026-10-08-obstacle-narration-design.md

| 파일 | 문장 | 길이 |
| --- | --- | --- |
| avoid.wav | 앞에 장애물이 있어 피해 갈게요. (후보 A1) | 1.60초 |
| slow.wav  | 앞에 장애물이 있어 천천히 갈게요. (후보 S1) | 1.90초 |

Supertonic-3 F2, speed 1.2, 16 kHz mono, -3 dBFS, 앞뒤 무음 정리. 문장·속도마다 4번 합성해 깨진 것을
거르고 길이가 가운데인 것을 썼다(1.35배는 짧은 문장이 무음으로 깨져 쓰지 않음).
`assets/` 는 git 무시 대상이라 커밋하려면 `git add -f` 가 필요하다.
이 파일이 없으면 scripts/avoid_cue.py 가 같은 목소리·속도로 그 자리에서 만든다.
