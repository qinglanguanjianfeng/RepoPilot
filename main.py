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

# ========== 1. 基础配置 ==========

MODEL = "deepseek-flash"
MAX_STEPS = 6


WORKSPACE = Path(__file__).resolve().parent / "workspace"
WORKSPACE.mkdir(exist_ok=True)


# ========== 2. 定义 LLM 可用的工具 ==========
# Tool Schema 负责指导 LLM 如何生成工具调用
TOOLS: list[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "在 Workspace 内创建或覆盖文本文件。"
                "可以自动创建不存在的父目录。"
                "path 必须是 Workspace 内的相对路径。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "相对于 Workspace 的文件路径，"
                            "例如 notes.txt 或 src/hello.py"
                        ),
                    },
                    "content": {
                        "type": "string",
                        "description": "要写入的文本内容",
                    },
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "按行读取 Workspace 内的 UTF-8 文本文件。"
                "默认读取前 60 行。"
                "可以通过 start_line 和 end_line 指定读取范围，"
                "结束行包含在读取范围内。"
                "每次最多读取 120 行，返回带行号的文本。"
                "当结果提示还有后续内容时，可以继续分段读取。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "相对于 Workspace 的文件路径，"
                            "例如 notes.txt 或 src/hello.py"
                        ),
                    },
                    "start_line": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "起始行号，从 1 开始，默认 1",
                    },
                    "end_line": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "结束行号，包含该行；省略时默认读取 60 行",
                    },
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": (
                "列出 Workspace 指定目录下的文件和子目录。"
                "仅列出当前层级，不递归遍历。"
                "如果未提供 path，则列出 Workspace 根目录。"
                "返回结果包含文件和目录类型标记。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "相对于 Workspace 的目录路径。"
                            "例如 .、src 或 src/utils。"
                            "省略时默认使用根目录。"
                        ),
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    }

]

# LLM 的最大子文件（多种类）条目数量
MAX_LIST_ENTRIES = 50
DEFAULT_READ_LINES = 60
MAX_READ_LINES = 120
MAX_READ_CHARS = 8000

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


# ========== 3. 实际执行工具 ==========

def execute_tool(name: str, arguments: dict[str, object]) -> str:
    """执行 Workspace 内的文本文件操作。"""

    # 1. 检查工具名称
    if name not in ("write_file","read_file", "list_files"):
        return f"Error: 未知工具 {name}"

    # 2. 获取路径
    # list_files 可以不提供 path，默认使用根目录
    default_path = "." if name == "list_files" else None
    raw_path = arguments.get("path", default_path)

    if not isinstance(raw_path, str):
        return "ERROR: path 必须是字符串"

    try:
        # 3. 解析并校验 Workspace 路径
        file_path = resolve_workspace_path(
            WORKSPACE,
            raw_path,
            allow_root=(name == "list_files"),  
            # 当工具是 list_files 时允许根目录，其他工具依然使用原来的限制
        )

        # 4. 目录浏览
        if name == "list_files":
            return list_directory(file_path, raw_path)

        # 5. 写入文件
        if name == "write_file":
            content = arguments.get("content")

            if not isinstance(content, str):
                return "ERROR: content 必须是字符串"

            # 自动创建不存在的父目录
            file_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            file_path.write_text(
                content,
                encoding="utf-8",
            )

            return f"OK: {raw_path} 写入成功"
        # 6. 读取文件
        
        if name == "read_file":

            # 1. 获取起始行
            start_line = arguments.get("start_line", 1)

            if not isinstance(start_line, int) or isinstance(start_line, bool):
                return "ERROR: start_line 必须是整数"

            if start_line < 1:
                return "ERROR: start_line 必须大于等于 1"

            # 2. 获取结束行
            end_line_value = arguments.get("end_line")

            if end_line_value is None:
                end_line = start_line + DEFAULT_READ_LINES - 1

            else:
                if (
                    not isinstance(end_line_value, int)
                    or isinstance(end_line_value, bool)
                ):
                    return "ERROR: end_line 必须是整数"

                end_line = end_line_value

            # 3. 校验读取范围
            if end_line < start_line:
                return "ERROR: end_line 不能小于 start_line"

            requested_lines = end_line - start_line + 1

            if requested_lines > MAX_READ_LINES:
                return (
                    f"ERROR: 单次最多读取 {MAX_READ_LINES} 行，"
                    "请缩小读取范围"
                )

            # 4. 执行分段读取
            return read_file_chunk(
                file_path=file_path,
                display_path=raw_path,
                start_line=start_line,
                end_line=end_line,
            )


    except (ValueError, OSError, UnicodeError) as exc:
        return f"ERROR: {exc}"

    return f"ERROR: 未知工具 {name}"


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
            tools=TOOLS,
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
