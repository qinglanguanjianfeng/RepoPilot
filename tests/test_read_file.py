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


def test_default_read(isolated_workspace: Path) -> None:
    """默认只读取前 60 行。"""

    file_path = isolated_workspace / "sample.txt"

    file_path.write_text(
        "\n".join(f"Line {i}" for i in range(1, 66)),
        encoding="utf-8",
    )

    result = main.execute_tool(
        "read_file",
        {"path": "sample.txt"},
    )

    assert "[L1] Line 1" in result
    assert "[L60] Line 60" in result
    assert "[L61] Line 61" not in result
    assert "[MORE]" in result
    assert "start_line=61" in result


def test_read_line_range(isolated_workspace: Path) -> None:
    """能够读取指定的起止行。"""

    file_path = isolated_workspace / "sample.txt"

    file_path.write_text(
        "\n".join(f"Line {i}" for i in range(1, 11)),
        encoding="utf-8",
    )

    result = main.execute_tool(
        "read_file",
        {
            "path": "sample.txt",
            "start_line": 3,
            "end_line": 5,
        },
    )

    assert "[L3] Line 3" in result
    assert "[L4] Line 4" in result
    assert "[L5] Line 5" in result

    assert "[L2] Line 2" not in result
    assert "[L6] Line 6" not in result
    assert "[MORE]" in result


def test_empty_file(isolated_workspace: Path) -> None:
    """空文件应返回明确提示。"""

    (isolated_workspace / "empty.txt").touch()

    result = main.execute_tool(
        "read_file",
        {"path": "empty.txt"},
    )

    assert "(空文件)" in result
    assert "[EOF]" in result


def test_start_line_out_of_range(
    isolated_workspace: Path,
) -> None:
    """起始行超过文件长度时应返回错误。"""

    (isolated_workspace / "sample.txt").write_text(
        "Hello\nWorld",
        encoding="utf-8",
    )

    result = main.execute_tool(
        "read_file",
        {
            "path": "sample.txt",
            "start_line": 100,
        },
    )

    assert result.startswith("ERROR:")


@pytest.mark.parametrize(
    "arguments",
    [
        {"path": "sample.txt", "start_line": 0},
        {"path": "sample.txt", "start_line": True},
        {"path": "sample.txt", "start_line": "2"},
        {"path": "sample.txt", "end_line": 0},
        {"path": "sample.txt", "end_line": 1.5},
        {"path": "sample.txt", "start_line": 5, "end_line": 3},
        {"path": "sample.txt", "start_line": 1, "end_line": 200},
    ],
)
def test_invalid_line_arguments(
    isolated_workspace: Path,
    arguments: dict[str, object],
) -> None:
    """非法行号应被拒绝。"""

    (isolated_workspace / "sample.txt").write_text(
        "Hello",
        encoding="utf-8",
    )

    result = main.execute_tool("read_file", arguments)

    assert result.startswith("ERROR:")


def test_character_limit(
    isolated_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """工具返回内容不能超出字符限制。"""

    monkeypatch.setattr(main, "MAX_READ_CHARS", 300)

    (isolated_workspace / "sample.txt").write_text(
        "\n".join("x" * 70 for _ in range(10)),
        encoding="utf-8",
    )

    result = main.execute_tool(
        "read_file",
        {"path": "sample.txt"},
    )

    assert len(result) <= 300
    assert "[TRUNCATED]" in result
    assert "start_line=" in result


def test_oversized_single_line(
    isolated_workspace: Path,
) -> None:
    """单行超过字符预算时，不应将全文返回。"""

    (isolated_workspace / "sample.txt").write_text(
        "x" * 20000,
        encoding="utf-8",
    )

    result = main.execute_tool(
        "read_file",
        {"path": "sample.txt"},
    )

    assert result.startswith("ERROR:")
    assert len(result) < main.MAX_READ_CHARS
