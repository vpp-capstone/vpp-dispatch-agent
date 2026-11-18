import pytest

from vpp.project_paths import resolve_project_path


@pytest.mark.parametrize("path", [
    "../outside.json", "data/../../outside.json", "/tmp/outside.json",
    "/old/repo/data/processed/../../outside.json", "C:relative.json",
    "C:\\repo\\data\\processed\\..\\outside.json",
])
def test_relocated_paths_do_not_escape_project(tmp_path, path):
    with pytest.raises(ValueError):
        resolve_project_path(tmp_path / "project", path)


def test_relocated_path_cannot_follow_symlink_outside_project(tmp_path):
    root = tmp_path / "project"
    (root / "data").mkdir(parents=True)
    (root / "data/processed").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="범위"):
        resolve_project_path(root, "/Users/old/repo/data/processed/result.json")


def test_native_absolute_paths_inside_project_still_work(tmp_path):
    path = tmp_path / "custom-results/run"
    assert resolve_project_path(tmp_path, path) == path.resolve()
