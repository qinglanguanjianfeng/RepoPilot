from pathlib import Path

import pytest

import main


@pytest.fixture
def isolated_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    monkeypatch.setattr(main, "WORKSPACE", tmp_path)
    return tmp_path


def test_list_workspace_root(isolated_workspace: Path) -> None:
    """能够列出根目录，且区分文件和目录。"""

    (isolated_workspace / "notes.txt").write_text(
        "Hello", encoding="utf-8"
    )
    (isolated_workspace / "src").mkdir()

    result = main.execute_tool("list_files", {})

    assert "Directory: ." in result
    assert "[FILE] notes.txt" in result
    assert "[DIR]  src/" in result


def test_list_subdirectory(isolated_workspace: Path) -> None:
    """能够列出指定子目录。"""

    src = isolated_workspace / "src"
    src.mkdir()
    (src / "hello.py").write_text(
        "print('hello')", encoding="utf-8"
    )

    result = main.execute_tool(
        "list_files", {"path": "src"}
    )

    assert "Directory: src" in result
    assert "[FILE] hello.py" in result


def test_list_empty_directory(isolated_workspace: Path) -> None:
    """空目录应返回明确提示。"""

    result = main.execute_tool("list_files", {})

    assert "(空目录)" in result


def test_list_nonexistent_directory(
    isolated_workspace: Path,
) -> None:
    """不存在的目录应返回错误。"""

    result = main.execute_tool(
        "list_files", {"path": "missing"}
    )

    assert result.startswith("ERROR:")


def test_list_file_instead_of_directory(
    isolated_workspace: Path,
) -> None:
    """不能将普通文件当作目录列出。"""

    (isolated_workspace / "notes.txt").write_text(
        "Hello", encoding="utf-8"
    )

    result = main.execute_tool(
        "list_files", {"path": "notes.txt"}
    )

    assert result.startswith("ERROR:")


def test_list_path_traversal(
    isolated_workspace: Path,
) -> None:
    """不能列出 Workspace 外部目录。"""

    result = main.execute_tool(
        "list_files", {"path": "../"}
    )

    assert result.startswith("ERROR:")


def test_list_entries_limit(
    isolated_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """目录列表不能无限制返回条目。"""

    monkeypatch.setattr(main, "MAX_LIST_ENTRIES", 3)

    for i in range(5):
        (isolated_workspace / f"file_{i}.txt").write_text(
            "test", encoding="utf-8"
        )

    result = main.execute_tool("list_files", {})

    assert result.count("[FILE]") == 3
    assert "另有 2 项未显示" in result


def test_list_symlink_not_followed(
    isolated_workspace: Path,
) -> None:
    """可以识别链接，但不能进入链接指向的目录。"""

    outside = isolated_workspace.parent / (
        isolated_workspace.name + "_outside"
    )
    outside.mkdir()

    (outside / "secret.txt").write_text(
        "Secret", encoding="utf-8"
    )

    link = isolated_workspace / "linked"

    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("当前系统不支持创建符号链接")

    # 根目录列表可以显示链接名称，但不会遍历其内容
    root_result = main.execute_tool("list_files", {})

    assert "[LINK] linked" in root_result
    assert "secret.txt" not in root_result

    # 尝试进入链接目录必须被拒绝
    result = main.execute_tool(
        "list_files", {"path": "linked"}
    )

    assert result.startswith("ERROR:")
