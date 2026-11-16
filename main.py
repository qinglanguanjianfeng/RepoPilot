import json
import os
from pathlib import Path

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionToolParam,
)
from collections.abc import Callable
from openai.types.chat import ChatCompletion

from workspace_paths import resolve_workspace_path
from workspace_search import search_workspace

from tool_registry import ToolRegistry

from openai.types.chat import ChatCompletionToolParam

# ========== 1. 基础配置 ==========

MODEL = "deepseek-flash"
MAX_STEPS = 6


WORKSPACE = Path(__file__).resolve().parent / "workspace"
WORKSPACE.mkdir(exist_ok=True)


# ========== 2. 定义 LLM 可用的工具 ==========
# Tool Schema 负责指导 LLM 如何生成工具调用
def make_tool_schema(
    name: str,
    description: str,
    properties: dict[str, object],
    required: list[str],
) -> ChatCompletionToolParam:
    """创建 Function Calling 使用的工具 Schema。"""

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


# LLM 的最大子文件（多种类）条目数量
MAX_LIST_ENTRIES = 50
DEFAULT_READ_LINES = 60
MAX_READ_LINES = 120
MAX_READ_CHARS = 8000

def get_path_argument(
    arguments: dict[str, object],
    default: str | None = None,
) -> str:
    """获取并检查工具参数中的 path。"""

    path = arguments.get("path", default)

    if not isinstance(path, str):
        raise ValueError("path 必须是字符串")

    return path

def handle_write_file(arguments: dict[str, object]) -> str:
    """执行写入文件工具。"""

    raw_path = get_path_argument(arguments)

    file_path = resolve_workspace_path(
        WORKSPACE, raw_path
    )

    content = arguments.get("content")

    if not isinstance(content, str):
        raise ValueError("content 必须是字符串")

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_path.write_text(content, encoding="utf-8")

    return f"OK: {raw_path} 写入成功"

def handle_read_file(arguments: dict[str, object]) -> str:
    """执行分段读取文件工具。"""

    raw_path = get_path_argument(arguments)

    file_path = resolve_workspace_path(
        WORKSPACE, raw_path
    )

    start_line = arguments.get("start_line", 1)

    if not isinstance(start_line, int) or isinstance(start_line, bool):
        raise ValueError("start_line 必须是整数")

    if start_line < 1:
        raise ValueError("start_line 必须大于等于 1")

    end_line_value = arguments.get("end_line")

    if end_line_value is None:
        end_line = start_line + DEFAULT_READ_LINES - 1
    else:
        if (
            not isinstance(end_line_value, int)
            or isinstance(end_line_value, bool)
        ):
            raise ValueError("end_line 必须是整数")

        end_line = end_line_value

    if end_line < start_line:
        raise ValueError("end_line 不能小于 start_line")

    if end_line - start_line + 1 > MAX_READ_LINES:
        raise ValueError(
            f"单次最多读取 {MAX_READ_LINES} 行，请缩小读取范围"
        )

    return read_file_chunk(
        file_path=file_path,
        display_path=raw_path,
        start_line=start_line,
        end_line=end_line,
    )

def handle_list_files(arguments: dict[str, object]) -> str:
    """执行目录浏览工具。"""

    raw_path = get_path_argument(arguments, default=".")

    directory = resolve_workspace_path(
        WORKSPACE,
        raw_path,
        allow_root=True,
    )

    return list_directory(directory, raw_path)

def handle_search_files(arguments: dict[str, object]) -> str:
    """执行文件内容搜索工具。"""

    keyword = arguments.get("keyword")
    search_path = arguments.get("path", ".")
    extension = arguments.get("extension")

    if not isinstance(keyword, str):
        raise ValueError("keyword 必须是字符串")

    if not isinstance(search_path, str):
        raise ValueError("path 必须是字符串")

    if extension is not None and not isinstance(extension, str):
        raise ValueError("extension 必须是字符串")

    return search_workspace(
        workspace=WORKSPACE,
        keyword=keyword,
        directory=search_path,
        extension=extension,
    )

def list_directory(
    directory: Path,
    relative_path: str,
) -> str:
    """列出一个目录的直接子项，不递归进入子目录。"""

    if not directory.is_dir():
        return "ERROR: 目录不存在或目标不是目录"

    # 按文件名排序，保证输出顺序稳定
    entries = sorted(
        directory.iterdir(),
        key=lambda path: path.name.casefold(),
    )

    if not entries:
        return f"Directory: {relative_path}\n(空目录)"

    lines = [f"Directory: {relative_path}"]

    # 限制返回给 LLM 的条目数量
    for entry in entries[:MAX_LIST_ENTRIES]:
        if entry.is_symlink() or entry.is_junction():
            lines.append(f"[LINK] {entry.name}")

        elif entry.is_dir():
            lines.append(f"[DIR]  {entry.name}/")

        elif entry.is_file():
            lines.append(f"[FILE] {entry.name}")

        else:
            lines.append(f"[OTHER] {entry.name}")

    # 提示模型还有多少项没有返回
    remaining = len(entries) - MAX_LIST_ENTRIES

    if remaining > 0:
        lines.append(f"... 另有 {remaining} 项未显示")

    return "\n".join(lines)

def read_file_chunk(
    file_path: Path,
    display_path: str,
    start_line: int,
    end_line: int,
) -> str:
    """分段读取文本文件，返回带行号且受字符数限制的内容。"""

    if not file_path.is_file():
        return "ERROR: 文件不存在或目标不是普通文件"

    header = (
        f"File: {display_path}\n"
        f"Requested lines: {start_line}-{end_line}\n"
    )

    # 为结尾状态信息预留空间
    char_budget = MAX_READ_CHARS - len(header) - 160

    if char_budget <= 0:
        return "ERROR: 输出字符限制配置不合理"

    output_lines: list[str] = []
    used_chars = 0
    found_start = False
    next_line: int | None = None

    # 逐行读取，不使用 read_text() 加载整个文件
    with file_path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):

            # 跳过起始行之前的内容
            if line_number < start_line:
                continue

            # 超过请求的结束行，停止读取
            if line_number > end_line:
                next_line = line_number
                break

            found_start = True

            # 移除行尾换行符，添加行号
            content = raw_line.rstrip("\r\n")
            formatted = f"[L{line_number}] {content}\n"

            # 检查字符预算
            if used_chars + len(formatted) > char_budget:

                # 单行就超过预算时，明确报错
                if not output_lines:
                    return (
                        f"ERROR: 第 {line_number} 行过长，"
                        "无法在单次字符限制内完整显示。"
                        "当前版本不支持单行内部的分片读取。"
                    )

                # 本行尚未返回，下次应从本行继续
                next_line = line_number
                break

            output_lines.append(formatted)
            used_chars += len(formatted)

    # 文件为空，或者起始行超出范围
    if not found_start:
        if start_line == 1:
            return header + "(空文件)\n[EOF]"

        return "ERROR: start_line 超出文件实际行数"

    # 给模型提供下一步读取提示
    if next_line is None:
        footer = "[EOF] 已到达文件末尾"

    elif next_line > end_line:
        footer = (
            f"[MORE] 文件仍有后续内容，"
            f"可从 start_line={next_line} 继续读取"
        )

    else:
        footer = (
            f"[TRUNCATED] 已达到单次输出字符限制，"
            f"可从 start_line={next_line} 继续读取"
        )

    return header + "".join(output_lines) + footer

def build_tool_registry() -> ToolRegistry:
    """注册 RepoPilot 当前可用的四个工具。"""

    registry = ToolRegistry()

    # 1. write_file
    registry.register(
        make_tool_schema(
            name="write_file",
            description=(
                "在 Workspace 内创建或覆盖文本文件，"
                "可以自动创建父目录。"
            ),
            properties={
                "path": {
                    "type": "string",
                    "description": "Workspace 内的相对文件路径",
                },
                "content": {
                    "type": "string",
                    "description": "要写入的文本内容",
                },
            },
            required=["path", "content"],
        ),
        handle_write_file,
    )

    # 2. read_file
    registry.register(
        make_tool_schema(
            name="read_file",
            description=(
                "按行读取 Workspace 中的 UTF-8 文本文件。"
                "默认读取前 60 行，每次最多读取 120 行，"
                "返回带行号的内容，可根据提示继续读取。"
            ),
            properties={
                "path": {
                    "type": "string",
                    "description": "Workspace 内的相对文件路径",
                },
                "start_line": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "起始行号，默认 1",
                },
                "end_line": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "结束行号，包含该行",
                },
            },
            required=["path"],
        ),
        handle_read_file,
    )

    # 3. list_files
    registry.register(
        make_tool_schema(
            name="list_files",
            description=(
                "列出 Workspace 指定目录的直接子项，"
                "区分文件、目录和链接，不递归遍历。"
                "未提供 path 时默认列出根目录。"
            ),
            properties={
                "path": {
                    "type": "string",
                    "description": "相对目录路径，默认 .",
                },
            },
            required=[],
        ),
        handle_list_files,
    )

    # 4. search_files
    registry.register(
        make_tool_schema(
            name="search_files",
            description=(
                "递归搜索 Workspace 内的文本关键词，"
                "返回相对路径、行号和匹配片段。"
                "支持指定搜索目录和扩展名过滤。"
                "找到目标后可使用 read_file 查看上下文。"
            ),
            properties={
                "keyword": {
                    "type": "string",
                    "description": "要搜索的精确文本",
                },
                "path": {
                    "type": "string",
                    "description": "搜索目录，默认 .",
                },
                "extension": {
                    "type": "string",
                    "description": "可选扩展名，例如 .py",
                },
            },
            required=["keyword"],
        ),
        handle_search_files,
    )

    return registry

REGISTRY = build_tool_registry()

# ========== 3. 实际执行工具 ==========
def execute_tool(
    name: str,
    arguments: dict[str, object],
) -> str:
    """通过 Tool Registry 执行工具。"""

    return REGISTRY.execute(name, arguments)
# ========== 4. Agent Loop ==========

def run_agent(
    user_task: str,
    llm_call: Callable[
        [list[ChatCompletionMessageParam]],
        ChatCompletion,
    ],
    max_steps: int = MAX_STEPS,
) -> str:
    """运行 Agent Loop，由外部提供 LLM 调用函数。"""

    # 1. 校验用户任务
    user_task = user_task.strip()

    if not user_task:
        raise ValueError("用户任务不能为空")

    if max_steps < 1:
        raise ValueError("max_steps 必须大于 0")

    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": (
                "你是一个能够操作 Workspace 内文本文件的 Coding Agent。"
                "你有三个工具："
                "list_files 用于查看目录结构；"
                "read_file 用于读取文本文件；"
                "write_file 用于创建或修改文本文件。"
                "当需要了解有哪些文件时，应先使用 list_files。"
                "所有路径都必须位于 Workspace 内。"
                "文件操作必须通过工具执行，不允许虚构执行结果。"
                "如果工具返回错误，应根据错误信息决定下一步行动。"
                "read_file 支持按行分段读取，"
                "每次最多读取 120 行。"
                "对于较长文件，应根据需要使用 start_line 和 end_line。"
                "如果工具返回 [MORE] 或 [TRUNCATED]，"
                "可以按照提示继续读取。"
                "不能将未读取的文件内容当作已知事实。"
                "当需要定位某个函数、类或字符串时，"
                    "优先使用 search_files 查找位置。"
                    "找到目标后，可以通过 read_file 读取对应行及上下文。"
                    "搜索结果可能因为数量限制而不完整，"
                    "不能将未搜索到的内容直接认定为不存在。"
            ),
        },
        {
            "role": "user",
            "content": user_task,
        },
    ]

    for step in range(1, max_steps + 1):
        print(f"\n========== Step {step} ==========")

        # 1. 通过传入的函数获取模型决策
        response = llm_call(messages)

        if not response.choices:
            raise RuntimeError("模型未返回任何 choices")

        choice = response.choices[0]
        message = choice.message

        if choice.finish_reason == "length":
            return "模型输出被截断，任务停止"

        tool_calls = message.tool_calls or []

        # 2. 模型没有调用工具，返回最终回答
        if not tool_calls:
            return message.content or "模型未返回有效内容"

        # 3. 解析 Tool Calls
        pending: list[tuple[str, str, str]] = []

        for call in tool_calls:
            if call.type != "function":
                raise RuntimeError("当前仅支持 Function Tool Calling")

            pending.append(
                (call.id, call.function.name, call.function.arguments)
            )

        # 4. 记录 Assistant 的 Tool Calls
        history_calls: list[ChatCompletionMessageToolCallParam] = [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": raw_arguments,
                },
            }
            for call_id, name, raw_arguments in pending
        ]

        messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": history_calls,
        })

        # 5. 执行工具并记录 Observation
        for call_id, name, raw_arguments in pending:
            print(f"Tool Call: {name}({raw_arguments})")

            try:
                arguments = json.loads(raw_arguments)

                if not isinstance(arguments, dict):
                    raise ValueError("工具参数必须是 JSON 对象")

                observation = execute_tool(name, arguments)

            except (ValueError, OSError) as exc:
                observation = f"ERROR: {exc}"

            print("Observation:", observation)

            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": observation,
            })

    return f"达到最大执行轮数 {max_steps}，任务停止"


def main() -> None:
    api_key = os.getenv("DEEPSEEK_API_KEY")

    if not api_key:
        raise RuntimeError("未找到 DEEPSEEK_API_KEY")

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        timeout=60.0,
        max_retries=2,
    )

    # DeepSeek API 调用函数
    def call_deepseek(
        messages: list[ChatCompletionMessageParam],
    ) -> ChatCompletion:
        return client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=REGISTRY.schemas(),
            tool_choice="auto",
            temperature=0.0,
            extra_body={"thinking": {"type": "disabled"}},
        )

    # 将模型调用函数传入 Agent Loop
    print("RepoPilot 已启动！输入 exit 或 quit 退出。")

    while True:
        try:
            user_task = input("\nRepoPilot > ").strip()

        except (KeyboardInterrupt, EOFError):
            print("\n已退出 RepoPilot。")
            break

        if user_task.lower() in ("exit", "quit"):
            print("已退出 RepoPilot。")
            break

        if not user_task:
            print("请输入任务内容。")
            continue

        result = run_agent(
            user_task=user_task,
            llm_call=call_deepseek,
        )

        print("\nAgent:", result)

if __name__ == "__main__":
    main()
