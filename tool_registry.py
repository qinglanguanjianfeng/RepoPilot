from collections.abc import Callable

from openai.types.chat import ChatCompletionToolParam


# 工具执行函数的统一类型
ToolHandler = Callable[[dict[str, object]], str]


class ToolRegistry:
    """管理工具的 Schema 和执行函数。"""

    def __init__(self) -> None:
        self._tools: dict[
            str,
            tuple[ChatCompletionToolParam, ToolHandler],
        ] = {}

    def register(
        self,
        schema: ChatCompletionToolParam,
        handler: ToolHandler,
    ) -> None:
        """注册工具。"""

        name = schema["function"]["name"]

        if not name.strip():
            raise ValueError("工具名称不能为空")

        if name in self._tools:
            raise ValueError(f"工具已注册: {name}")

        self._tools[name] = (schema, handler)

    def schemas(self) -> list[ChatCompletionToolParam]:
        """返回可提供给 LLM 的工具 Schema。"""

        return [
            schema
            for schema, _ in self._tools.values()
        ]

    def execute(
        self,
        name: str,
        arguments: dict[str, object],
    ) -> str:
        """根据名称执行工具，并返回 Observation。"""

        tool = self._tools.get(name)

        if tool is None:
            return f"ERROR: 未知工具 {name}"

        _, handler = tool

        try:
            return handler(arguments)

        except (ValueError, OSError, UnicodeError) as exc:
            return f"ERROR: {exc}"
