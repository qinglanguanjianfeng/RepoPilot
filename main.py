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

# ========== 1. 基础配置 ==========

MODEL = "deepseek-flash"
MAX_STEPS = 6
ALLOWED_FILE = "hello_agent.txt"

WORKSPACE = Path(__file__).resolve().parent / "workspace"
WORKSPACE.mkdir(exist_ok=True)


# ========== 2. 定义 LLM 可用的工具 ==========

TOOLS: list[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "在工作区中写入文本文件。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "enum": ["hello_agent.txt"],
                    },
                    "content": {
                        "type": "string",
                        "description": "要写入的文本内容",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取工作区内的文本文件。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "enum": ["hello_agent.txt"],
                    },
                },
                "required": ["path"],
            },
        },
    },
]


# ========== 3. 实际执行工具 ==========

def execute_tool(name: str, arguments: dict[str, object]) -> str:
    # MVP 阶段只允许操作指定文件
    if arguments.get("path") != ALLOWED_FILE:
        return "ERROR: 不允许操作该文件"

    file_path = WORKSPACE / ALLOWED_FILE

    # 防止通过已有符号链接访问工作区外的文件
    if file_path.is_symlink():
        return "ERROR: 不允许操作符号链接"

    if name == "write_file":
        content = arguments.get("content")
        if not isinstance(content, str):
            return "ERROR: content 必须是字符串"

        file_path.write_text(content, encoding="utf-8")
        return f"OK: {ALLOWED_FILE} 写入成功"

    if name == "read_file":
        if not file_path.is_file():
            return "ERROR: 文件不存在"

        return file_path.read_text(encoding="utf-8")

    return f"ERROR: 未知工具 {name}"

# ========== 4. Agent Loop ==========

def run_agent(
    llm_call: Callable[
        [list[ChatCompletionMessageParam]],
        ChatCompletion,
    ],
    max_steps: int = MAX_STEPS,
) -> str:
    """运行 Agent Loop，由外部提供 LLM 调用函数。"""

    if max_steps < 1:
        raise ValueError("max_steps 必须大于 0")

    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": (
                "你是一个文件操作 Agent。"
                "必须通过提供的工具实际执行文件操作。"
                "当用户要求创建并验证文件时，"
                "需要先写入文件，再使用 read_file 读取确认。"
                "只有确认任务完成后才能输出最终回答。"
            ),
        },
        {
            "role": "user",
            "content": (
                "请创建 hello_agent.txt，"
                "写入 Hello RepoPilot，"
                "然后读取文件，确认内容正确。"
            ),
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
    result = run_agent(llm_call=call_deepseek)  # 传递函数供 Agent Loop 调用

    print("\nAgent:", result)

if __name__ == "__main__":
    main()
