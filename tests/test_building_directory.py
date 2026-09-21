"""건물 디렉터리: 다른 층 장소를 알되 갈 수는 없다 → 엘리베이터 제안 (스펙 3.1)."""
from src.building_directory import DirectoryEntry, format_directory_block, load_directory


def test_load_and_skip_bad_rows(tmp_path):
    p = tmp_path / "directory.yaml"
    p.write_text("- name: 세미나실\n  building: 로봇관\n  floor: 3\n- name: 잘못\n  floor: x\n- 5\n", encoding="utf-8")
    assert load_directory(str(p)) == [DirectoryEntry(name="세미나실", building="로봇관", floor=3)]


def test_missing_file_is_empty(tmp_path):
    assert load_directory(str(tmp_path / "none.yaml")) == []


def test_block_lists_only_other_floors():
    entries = [DirectoryEntry("세미나실", "로봇관", 3), DirectoryEntry("407호", "로봇관", 4),
               DirectoryEntry("식당", "학생회관", 1)]
    block = format_directory_block(entries, "로봇관", 4)
    assert block.startswith("\n[다른 층 장소]")
    assert "세미나실: 로봇관 3층" in block and "식당: 학생회관 1층" in block and "407호" not in block


def test_block_empty_when_nothing_else():
    assert format_directory_block([DirectoryEntry("407호", "로봇관", 4)], "로봇관", 4) == ""
    assert format_directory_block([], "로봇관", 4) == ""
