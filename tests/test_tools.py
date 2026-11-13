from pathlib import Path

import pytest

import main


@pytest.fixture
def isolated_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """为每个测试提供独立的临时工作目录。"""
    monkeypatch.setattr(main, "WORKSPACE", tmp_path)
    return tmp_path


def test_write_file(isolated_workspace: Path) -> None:
    result = main.execute_tool(
        "write_file",
        {"path": "hello_agent.txt", "content": "Hello RepoPilot"},
    )

    assert result == "OK: hello_agent.txt 写入成功"
    assert (
        isolated_workspace / "hello_agent.txt"
    ).read_text(encoding="utf-8") == "Hello RepoPilot"


def test_read_file(isolated_workspace: Path) -> None:
    file_path = isolated_workspace / "hello_agent.txt"
    file_path.write_text("Hello RepoPilot", encoding="utf-8")

    result = main.execute_tool(
        "read_file",
        {"path": "hello_agent.txt"},
    )

    assert "[L1] Hello RepoPilot" in result
    assert "[EOF]" in result


def test_read_missing_file(isolated_workspace: Path) -> None:
    result = main.execute_tool(
        "read_file",
        {"path": "hello_agent.txt"},
    )

    assert result == "ERROR: 文件不存在或目标不是普通文件"


def test_invalid_path(isolated_workspace: Path) -> None:
    result = main.execute_tool(
        "write_file",
        {
            "path": "../other.txt",
            "content": "Test",
        },
    )

    assert result.startswith("ERROR:")
    assert list(isolated_workspace.iterdir()) == []


def test_invalid_content(isolated_workspace: Path) -> None:
    result = main.execute_tool(
        "write_file",
        {"path": "hello_agent.txt", "content": 123},
    )

    assert result == "ERROR: content 必须是字符串"
    assert not (isolated_workspace / "hello_agent.txt").exists()


def test_unknown_tool(isolated_workspace: Path) -> None:
    result = main.execute_tool(
        "delete_file",
        {"path": "hello_agent.txt"},
    )

    assert result == "Error: 未知工具 delete_file"



def test_write_multiple_files(isolated_workspace: Path) -> None:
    """验证能够写入不同名称的文件。"""

    files = {
        "notes.txt": "Hello RepoPilot",
        "config.json": '{"name": "RepoPilot"}',
    }

    for name, content in files.items():
        result = main.execute_tool(
            "write_file",
            {"path": name, "content": content},
        )

        assert result == f"OK: {name} 写入成功"

        assert (
            isolated_workspace / name
        ).read_text(encoding="utf-8") == content


def test_nested_file(isolated_workspace: Path) -> None:
    """验证自动创建子目录，并能读取嵌套文件。"""

    path = "src/utils/math.py"
    content = "def add(a, b):\n    return a + b\n"

    write_result = main.execute_tool(
        "write_file",
        {"path": path, "content": content},
    )

    assert write_result == f"OK: {path} 写入成功"

    file_path = isolated_workspace / "src" / "utils" / "math.py"

    assert file_path.is_file()
    assert file_path.read_text(encoding="utf-8") == content

    read_result = main.execute_tool(
        "read_file",
        {"path": path},
    )

    assert "[L1] def add(a, b):" in read_result
    assert "[L2]     return a + b" in read_result


def test_reject_absolute_path(isolated_workspace: Path) -> None:
    """验证绝对路径无法通过文件工具执行。"""

    outside = isolated_workspace.parent / "outside.txt"

    result = main.execute_tool(
        "write_file",
        {
            "path": str(outside),
            "content": "Forbidden",
        },
    )

    assert result.startswith("ERROR:")
    assert not outside.exists()


def test_reject_symlink_escape(isolated_workspace: Path) -> None:
    """验证不能利用符号链接写入 Workspace 外部。"""

    outside = isolated_workspace.parent / (
        isolated_workspace.name + "_outside"
    )
    outside.mkdir()

    link = isolated_workspace / "linked"

    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("当前环境无法创建符号链接")

    result = main.execute_tool(
        "write_file",
        {
            "path": "linked/secret.txt",
            "content": "Forbidden",
        },
    )

    assert result.startswith("ERROR:")
    assert not (outside / "secret.txt").exists()
