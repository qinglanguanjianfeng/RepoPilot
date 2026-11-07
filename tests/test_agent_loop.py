import json
from pathlib import Path

import pytest
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessageParam,
)

import main


@pytest.fixture
def isolated_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """使用独立工作目录，避免测试修改真实文件。"""
    monkeypatch.setattr(main, "WORKSPACE", tmp_path)
    return tmp_path


def fake_completion(
    *,
    content: str | None = None,
    tool_name: str | None = None,
    arguments: dict[str, str] | str | None = None,
    call_id: str = "call_1",
) -> ChatCompletion:
    """构造符合 OpenAI SDK 数据结构的模拟模型响应。"""

    message: dict[str, object] = {
        "role": "assistant",
        "content": content,
    }

    finish_reason = "stop"

    if tool_name is not None:
        raw_arguments = (
            arguments
            if isinstance(arguments, str)
            else json.dumps(arguments or {}, ensure_ascii=False)
        )

        message["tool_calls"] = [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": raw_arguments,
                },
            }
        ]

        finish_reason = "tool_calls"

    return ChatCompletion.model_validate(
        {
            "id": "fake-completion",
            "object": "chat.completion",
            "created": 0,
            "model": "fake-model",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish_reason,
                    "message": message,
                }
            ],
        }
    )


def test_agent_loop_success(isolated_workspace: Path) -> None:
    """验证 write -> read -> finish 完整执行流程。"""

    responses = iter([
        fake_completion(
            tool_name="write_file",
            arguments={
                "path": "hello_agent.txt",
                "content": "Hello RepoPilot",
            },
            call_id="call_write",
        ),
        fake_completion(
            tool_name="read_file",
            arguments={"path": "hello_agent.txt"},
            call_id="call_read",
        ),
        fake_completion(content="文件创建并验证成功"),
    ])

    def fake_llm(
        messages: list[ChatCompletionMessageParam],
    ) -> ChatCompletion:
        return next(responses)

    result = main.run_agent(
        user_task="请创建 hello_agent.txt 并验证内容",
        llm_call=fake_llm,
        max_steps=5,
    )

    assert result == "文件创建并验证成功"

    file_path = isolated_workspace / "hello_agent.txt"
    assert file_path.exists()
    assert file_path.read_text(encoding="utf-8") == "Hello RepoPilot"

def test_tool_observation_in_history(
    isolated_workspace: Path,
) -> None:
    """验证工具执行结果会传递到下一轮 LLM。"""

    responses = iter([
        fake_completion(
            tool_name="write_file",
            arguments={
                "path": "hello_agent.txt",
                "content": "Hello RepoPilot",
            },
            call_id="call_write",
        ),
        fake_completion(content="任务完成"),
    ])

    history_snapshots: list[list[ChatCompletionMessageParam]] = []

    def fake_llm(
        messages: list[ChatCompletionMessageParam],
    ) -> ChatCompletion:
        history_snapshots.append(list(messages))
        return next(responses)

    result = main.run_agent(
        user_task="请创建 hello_agent.txt 并验证内容",
        llm_call=fake_llm,
        max_steps=5,
    )

    assert result == "任务完成"
    assert len(history_snapshots) == 2

    # 第二次调用 LLM 时，应包含第一次的工具执行结果
    second_history = history_snapshots[1]

    assistant_message = second_history[-2]
    tool_message = second_history[-1]

    assert assistant_message["role"] == "assistant"
    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == "call_write"
    assert tool_message["content"] == "OK: hello_agent.txt 写入成功"

def test_agent_max_steps(isolated_workspace: Path) -> None:
    """验证 Agent 不会无限执行。"""

    call_count = 0

    def fake_llm(
        messages: list[ChatCompletionMessageParam],
    ) -> ChatCompletion:
        nonlocal call_count
        call_count += 1

        return fake_completion(
            tool_name="read_file",
            arguments={"path": "hello_agent.txt"},
            call_id=f"call_{call_count}",
        )

    result = main.run_agent(
        user_task="请创建 hello_agent.txt 并验证内容",
        llm_call=fake_llm,
        max_steps=2,
    )

    assert call_count == 2
    assert "达到最大执行轮数 2" in result

def test_invalid_tool_arguments(
    isolated_workspace: Path,
) -> None:
    """验证非法工具参数不会让 Agent 直接崩溃。"""

    responses = iter([
        fake_completion(
            tool_name="write_file",
            arguments="{invalid json",
            call_id="call_bad",
        ),
        fake_completion(content="已收到工具参数错误"),
    ])

    history_snapshots: list[list[ChatCompletionMessageParam]] = []

    def fake_llm(
        messages: list[ChatCompletionMessageParam],
    ) -> ChatCompletion:
        history_snapshots.append(list(messages))
        return next(responses)

    result = main.run_agent(
        user_task="请创建 hello_agent.txt 并验证内容",
        llm_call=fake_llm,
        max_steps=5,
    )

    assert result == "已收到工具参数错误"

    # 第二次调用 LLM 时，上一轮工具错误已经进入历史
    tool_message = history_snapshots[1][-1]

    assert tool_message["role"] == "tool"
    assert isinstance(tool_message["content"], str)
    assert tool_message["content"].startswith("ERROR:")

    # 非法参数不应该生成文件
    assert not (isolated_workspace / "hello_agent.txt").exists()

def test_dynamic_task_and_history_isolation() -> None:
    """验证动态任务输入，以及不同任务的历史相互独立。"""

    captured_histories: list[list[ChatCompletionMessageParam]] = []

    def fake_llm(
        messages: list[ChatCompletionMessageParam],
    ) -> ChatCompletion:
        captured_histories.append(list(messages))
        return fake_completion(content="任务完成")

    task_a = "请读取 hello_agent.txt"
    task_b = "请修改 hello_agent.txt 的内容"

    result_a = main.run_agent(
        user_task=task_a,
        llm_call=fake_llm,
    )

    result_b = main.run_agent(
        user_task=task_b,
        llm_call=fake_llm,
    )

    assert result_a == "任务完成"
    assert result_b == "任务完成"

    assert len(captured_histories) == 2

    # 两次任务都应该从独立的消息历史开始
    assert all(len(history) == 2 for history in captured_histories)

    # 两次任务分别包含自己的用户输入
    assert captured_histories[0][1] == {
        "role": "user",
        "content": task_a,
    }

    assert captured_histories[1][1] == {
        "role": "user",
        "content": task_b,
    }


def test_empty_user_task() -> None:
    """验证 Agent 拒绝执行空任务。"""

    def fake_llm(
        messages: list[ChatCompletionMessageParam],
    ) -> ChatCompletion:
        raise AssertionError("空任务不应该触发 LLM 调用")

    with pytest.raises(ValueError, match="用户任务不能为空"):
        main.run_agent(
            user_task="   ",
            llm_call=fake_llm,
        )