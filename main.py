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
                "读取 Workspace 内的 UTF-8 文本文件。"
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
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
]


# ========== 3. 实际执行工具 ==========

def execute_tool(name: str, arguments: dict[str, object]) -> str:
    """执行 Workspace 内的文本文件操作。"""

    # 1. 检查工具名称
    if name not in ("write_file","read_file"):
        return f"Error: 未知工具 {name}"

    # 2. 检查路径参数类型
    raw_path = arguments.get("path")

    if not isinstance(raw_path, str):
        return "ERROR: path 必须是字符串"

    try:
        # 3. 解析并校验 Workspace 路径
        file_path = resolve_workspace_path(
            WORKSPACE,
            raw_path,
        )

        # 4. 写入文件
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
        # 5. 读取文件
        if name == "read_file":
            if not file_path.is_file():
                return "ERROR: 文件不存在"

            return file_path.read_text(
                encoding="utf-8",
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
                "你可以通过 write_file 和 read_file 创建、修改和读取文件。"
                "文件路径必须是 Workspace 内的相对路径，"
                "例如 notes.txt 或 src/hello.py。"
                "不允许访问 Workspace 外的文件。"
                "执行文件操作时必须调用工具，不能虚构执行结果。"
                "如果工具返回错误，应根据错误信息决定如何处理。"
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
