
import pytest
from openai.types.chat import ChatCompletionToolParam

import main
from tool_registry import ToolRegistry


def make_test_schema(name: str) -> ChatCompletionToolParam:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "用于测试的工具",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    }


def test_register_and_execute() -> None:
    """注册后能正确执行工具。"""

    registry = ToolRegistry()

    registry.register(
        make_test_schema("hello"),
        lambda arguments: "Hello RepoPilot",
    )

    result = registry.execute("hello", {})

    assert result == "Hello RepoPilot"


def test_schemas() -> None:
    """能够获取所有注册工具的 Schema。"""

    registry = ToolRegistry()

    registry.register(
        make_test_schema("tool_a"),
        lambda arguments: "A",
    )
    registry.register(
        make_test_schema("tool_b"),
        lambda arguments: "B",
    )

    schemas = registry.schemas()

    names = {
        schema["function"]["name"]
        for schema in schemas
    }

    assert names == {"tool_a", "tool_b"}


def test_duplicate_registration() -> None:
    """不能注册两个同名工具。"""

    registry = ToolRegistry()

    registry.register(
        make_test_schema("hello"),
        lambda arguments: "A",
    )

    with pytest.raises(ValueError, match="工具已注册"):
        registry.register(
            make_test_schema("hello"),
            lambda arguments: "B",
        )


def test_unknown_tool() -> None:
    """未知工具应返回错误。"""

    registry = ToolRegistry()

    result = registry.execute("missing", {})

    assert result == "ERROR: 未知工具 missing"


def test_handler_error() -> None:
    """Handler 抛出的参数错误应转化为 Observation。"""

    registry = ToolRegistry()

    def failing_handler(arguments: dict[str, object]) -> str:
        raise ValueError("参数错误")

    registry.register(
        make_test_schema("failing_tool"),
        failing_handler,
    )

    result = registry.execute("failing_tool", {})

    assert result == "ERROR: 参数错误"


def test_repopilot_registered_tools() -> None:
    """RepoPilot 中的四个工具都应已注册。"""

    schemas = main.REGISTRY.schemas()

    names = {
        schema["function"]["name"]
        for schema in schemas
    }

    assert names == {
        "write_file",
        "read_file",
        "list_files",
        "search_files",
    }