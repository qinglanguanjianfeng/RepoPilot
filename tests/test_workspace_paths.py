import os
from pathlib import Path

import pytest

from workspace_paths import resolve_workspace_path


def test_normal_file(tmp_path: Path) -> None:
    """普通文件路径应当被允许。"""
    result = resolve_workspace_path(tmp_path, "notes.txt")

    assert result == tmp_path / "notes.txt"


def test_nested_file(tmp_path: Path) -> None:
    """允许尚未创建的嵌套文件路径。"""
    result = resolve_workspace_path(tmp_path, "src/hello.py")

    assert result == tmp_path / "src" / "hello.py"


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "../secret.txt",
        "src/../../secret.txt",
        "../../other_project/config.py",
    ],
)
def test_path_traversal(
    tmp_path: Path,
    unsafe_path: str,
) -> None:
    """拒绝父目录穿越。"""
    with pytest.raises(ValueError):
        resolve_workspace_path(tmp_path, unsafe_path)


@pytest.mark.parametrize("unsafe_path", ["", "   ", "."])
def test_empty_or_directory_path(
    tmp_path: Path,
    unsafe_path: str,
) -> None:
    """拒绝空路径及 Workspace 根目录本身。"""
    with pytest.raises(ValueError):
        resolve_workspace_path(tmp_path, unsafe_path)


@pytest.mark.skipif(os.name != "nt", reason="Windows 专用测试")
@pytest.mark.parametrize(
    "unsafe_path",
    [
        r"C:\Windows\win.ini",
        r"\Windows\win.ini",
        r"C:relative.txt",
        "CON",
        "NUL",
    ],
)
def test_windows_unsafe_path(
    tmp_path: Path,
    unsafe_path: str,
) -> None:
    """拒绝 Windows 绝对路径、带盘符路径及保留名称。"""
    with pytest.raises(ValueError):
        resolve_workspace_path(tmp_path, unsafe_path)


def test_symlink_escape(tmp_path: Path) -> None:
    """不允许通过符号链接访问工作区外部。"""
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    outside = tmp_path / "outside"
    outside.mkdir()

    link = workspace / "linked"

    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("当前系统不允许创建符号链接")

    with pytest.raises(ValueError):
        resolve_workspace_path(workspace, "linked/secret.txt")
