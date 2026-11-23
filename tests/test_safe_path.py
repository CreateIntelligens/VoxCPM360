"""接到固定目錄下的請求名稱，組出來的路徑不能跑出那個目錄。"""

from gateway.safe_path import child_path


def test_child_path_stays_inside_base(tmp_path):
    assert child_path(tmp_path, "abc.wav") == tmp_path / "abc.wav"
    assert child_path(tmp_path, "batch", "0.mp3") == tmp_path / "batch" / "0.mp3"


def test_child_path_rejects_escapes(tmp_path):
    assert child_path(tmp_path, "../etc/passwd") is None
    assert child_path(tmp_path, "a/../../x") is None
    assert child_path(tmp_path, "/etc/passwd") is None
    assert child_path(tmp_path, "") is None
    # 前綴相同但不是子目錄：/tmp/base 與 /tmp/base-evil
    assert child_path(tmp_path, f"../{tmp_path.name}-evil/x") is None
