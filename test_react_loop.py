"""
S7 第 67-68 天：ReAct 执行循环单元测试

验收标准（来自 Sprint_7.md 第 67-68 天）：
  "启动 Agent，观察后端日志：循环依次处理步骤 1 → 步骤 2 → 步骤 3，
   每个步骤都经历了 Reason -> Act -> Observe 的完整生命周期，
   且日志中打印出模拟的执行结果。"

覆盖场景：
  - 完整循环执行：3 步 DAG，每步 Reason→Act→Observe，全部完成
  - ask_user 人工介入：某步返回 ask_user，循环暂停，用户回复后恢复
  - 最大迭代次数退出：步骤数超过上限，循环停止
  - 上下文裁剪：_build_context_summary 只保留最近 N 步的完整 observation
  - 中断标志暂停：设置 interrupt_flag 后循环 break
  - 工具执行失败：ToolExecutionError 导致步骤标记 failed
  - Reason 失败：LLM 返回非法 JSON 导致步骤标记 failed
"""

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

from app.models.agent import AgentSession, TaskStep
from app.services.agent_session_store import get_agent_session_store
from app.services.agent_state_machine import AgentStateMachine
from app.services.react_loop import (
    _build_context_summary,
    _extract_action_json,
    handle_ask_user_response,
    run_agent,
)
from app.services.tool_executor import ToolExecutor


# ============================================================
# 测试夹具
# ============================================================

class FakeToolExecutor(ToolExecutor):
    """测试用工具执行器，记录调用并返回预设结果"""

    def __init__(self, results=None):
        self.calls = []
        self._results = results or {}

    async def execute(self, tool, params):
        self.calls.append((tool, params))
        return self._results.get(tool, f"{tool} 执行成功（测试模拟）")


def _make_session(steps, goal="测试 ReAct 循环"):
    """创建一个 AgentSession 并保存到 store，返回 session_id"""
    session = AgentSession(user_goal=goal, plan=steps)
    store = get_agent_session_store()
    store.save(session)
    return session.session_id


def _dag_3_steps():
    """3 步 DAG：A -> B -> C"""
    return [
        TaskStep(id="A", description="步骤A", details="初始化", dependencies=[], action="write_file"),
        TaskStep(id="B", description="步骤B", details="编码", dependencies=["A"], action="write_file"),
        TaskStep(id="C", description="步骤C", details="测试", dependencies=["B"], action="run_command"),
    ]


# ============================================================
# 内部函数测试
# ============================================================

def test_extract_action_json_with_markdown():
    """Reason 输出含 ```json 标记时仍能提取 action"""
    text = '```json\n{"tool": "write_file", "params": {"path": "x"}}\n```'
    action = _extract_action_json(text)
    assert action == {"tool": "write_file", "params": {"path": "x"}}


def test_extract_action_json_plain():
    """纯 JSON 文本能正确解析"""
    text = '{"tool": "run_command", "params": {"cmd": "ls"}}'
    action = _extract_action_json(text)
    assert action == {"tool": "run_command", "params": {"cmd": "ls"}}


def test_extract_action_json_invalid():
    """非法 JSON 返回 None"""
    assert _extract_action_json("not json at all") is None
    assert _extract_action_json("") is None


def test_build_context_summary_window():
    """
    上下文裁剪：只保留最近 window 个 done 步骤的完整 observation，
    更早的步骤只保留摘要。
    对应 S7 风险预警："ReAct 循环中的上下文爆炸"。
    """
    steps = [
        TaskStep(id=f"s{i}", description=f"步骤{i}", status="done",
                 observation=f"结果{i}", dependencies=[])
        for i in range(5)
    ]
    session = AgentSession(user_goal="test", plan=steps)
    sm = AgentStateMachine(session)

    summary = _build_context_summary(sm, window=3)

    # 最近 3 步（s2, s3, s4）应包含完整 observation
    assert "结果2" in summary
    assert "结果3" in summary
    assert "结果4" in summary
    # 更早的步骤（s0, s1）不应包含完整 observation
    assert "结果0" not in summary
    assert "结果1" not in summary
    # 但应包含"已省略"提示
    assert "省略" in summary


def test_build_context_summary_empty():
    """无已完成步骤时返回占位文本"""
    steps = [TaskStep(id="s1", description="步骤1", status="pending", dependencies=[])]
    session = AgentSession(user_goal="test", plan=steps)
    sm = AgentStateMachine(session)
    summary = _build_context_summary(sm, window=3)
    assert "暂无" in summary or "已完成" in summary


# ============================================================
# 验收标准：完整循环执行（Reason -> Act -> Observe）
# ============================================================

@pytest.mark.asyncio
async def test_full_loop_executes_all_steps():
    """
    验收标准核心：循环依次处理步骤 A → B → C，
    每个步骤经历 Reason -> Act -> Observe，全部标记为 done。
    """
    session_id = _make_session(_dag_3_steps())
    executor = FakeToolExecutor()

    # Mock Reason 阶段：每步返回 write_file 工具调用
    async def fake_chat(messages, model, **kwargs):
        return json.dumps({"tool": "write_file", "params": {"path": "main.py", "content": "print(1)"}})

    with patch("app.services.react_loop.chat_completion_text", side_effect=fake_chat):
        await run_agent(session_id, tool_executor=executor)

    # 验证所有步骤完成
    store = get_agent_session_store()
    session = store.get(session_id)
    assert session.done_steps == 3
    assert session.total_steps == 3
    assert all(s.status == "done" for s in session.plan)

    # 验证每个步骤都有 observation（Observe 阶段写入）
    for step in session.plan:
        assert step.observation is not None
        assert len(step.observation) > 0

    # 验证工具执行器被调用了 3 次（每步一次 Act）
    assert len(executor.calls) == 3
    assert all(call[0] == "write_file" for call in executor.calls)

    # 验证最终总结
    assert session.final_answer == "所有任务步骤已完成"
    print("✅ 完整循环：A→B→C 三步依次完成，Reason→Act→Observe 生命周期验证通过")


# ============================================================
# ask_user 人工介入
# ============================================================

@pytest.mark.asyncio
async def test_ask_user_pauses_loop():
    """
    当某步 Reason 返回 ask_user 时，循环暂停，设置 pending_question，
    该步骤保持 running 状态。
    """
    steps = [
        TaskStep(id="A", description="选择数据库", details="选择 SQLite 还是 PG", dependencies=[]),
        TaskStep(id="B", description="创建表", details="建表", dependencies=["A"]),
    ]
    session_id = _make_session(steps)

    # 第一步返回 ask_user，第二步返回 write_file
    call_count = {"n": 0}

    async def fake_chat(messages, model, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return json.dumps({"tool": "ask_user", "params": {"question": "使用 SQLite 还是 PostgreSQL？"}})
        return json.dumps({"tool": "write_file", "params": {"path": "db.py", "content": "..."}})

    with patch("app.services.react_loop.chat_completion_text", side_effect=fake_chat):
        await run_agent(session_id, tool_executor=FakeToolExecutor())

    store = get_agent_session_store()
    session = store.get(session_id)

    # 循环暂停，pending_question 已设置
    assert session.pending_question == "使用 SQLite 还是 PostgreSQL？"
    assert session.is_paused is True
    # 发起 ask_user 的步骤 A 保持 running（等待用户回复）
    step_a = next(s for s in session.plan if s.id == "A")
    assert step_a.status == "running"
    # 步骤 B 仍为 pending
    step_b = next(s for s in session.plan if s.id == "B")
    assert step_b.status == "pending"

    print("✅ ask_user：循环暂停，pending_question 已设置验证通过")


@pytest.mark.asyncio
async def test_ask_user_response_resumes_loop():
    """
    用户回复 ask_user 后，handle_ask_user_response 将提问步骤标记为 done，
    并恢复循环继续执行后续步骤。
    """
    steps = [
        TaskStep(id="A", description="选择数据库", dependencies=[]),
        TaskStep(id="B", description="创建表", dependencies=["A"]),
    ]
    session_id = _make_session(steps)

    # 先模拟 ask_user 暂停
    call_count = {"n": 0}

    async def fake_chat(messages, model, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return json.dumps({"tool": "ask_user", "params": {"question": "SQLite 还是 PG？"}})
        return json.dumps({"tool": "write_file", "params": {"path": "db.py"}})

    executor = FakeToolExecutor()
    # 注意：patch 必须覆盖整个测试过程，包括 handle_ask_user_response 启动的新循环
    with patch("app.services.react_loop.chat_completion_text", side_effect=fake_chat):
        await run_agent(session_id, tool_executor=executor)

        # 此时循环已暂停
        store = get_agent_session_store()
        session = store.get(session_id)
        assert session.pending_question is not None

        # 用户回复，恢复循环
        success = await handle_ask_user_response(session_id, "SQLite")
        assert success is True

        # 等待后台循环完成
        await asyncio.sleep(0.5)

    session = store.get(session_id)
    # 步骤 A 已完成（observation 包含用户回答）
    step_a = next(s for s in session.plan if s.id == "A")
    assert step_a.status == "done"
    assert "SQLite" in (step_a.observation or "")
    # 步骤 B 也应完成
    step_b = next(s for s in session.plan if s.id == "B")
    assert step_b.status == "done"
    # pending_question 已清除
    assert session.pending_question is None

    print("✅ ask_user 回复后循环恢复，后续步骤继续执行验证通过")


# ============================================================
# 退出条件：最大迭代次数
# ============================================================

@pytest.mark.asyncio
async def test_max_iterations_exit():
    """
    当步骤数超过 REACT_MAX_ITERATIONS 时，循环停止并记录原因。
    对应 S7 风险预警："防止 AI 陷入死循环"。
    """
    # 创建 20 个无依赖步骤（可并行调度，但单线程循环逐个执行）
    steps = [
        TaskStep(id=f"s{i}", description=f"步骤{i}", dependencies=[])
        for i in range(20)
    ]
    session_id = _make_session(steps)

    async def fake_chat(messages, model, **kwargs):
        return json.dumps({"tool": "write_file", "params": {"path": "x"}})

    with patch("app.services.react_loop.chat_completion_text", side_effect=fake_chat):
        await run_agent(session_id, tool_executor=FakeToolExecutor())

    store = get_agent_session_store()
    session = store.get(session_id)

    # 完成的步骤数应等于最大迭代次数（每迭代处理一步）
    assert session.done_steps == 15
    assert "最大迭代次数" in (session.final_answer or "")

    print("✅ 最大迭代次数退出：15 步后循环停止验证通过")


# ============================================================
# 中断标志暂停
# ============================================================

@pytest.mark.asyncio
async def test_interrupt_flag_pauses_loop():
    """
    设置 interrupt_flag=True 后，循环在下一次迭代顶部 break。
    对应 S7 第 69-70 天暂停机制。

    当 pause 发生在 _reason 期间时，当前 running 步骤会被回收为 pending，
    供 resume 后重试。
    """
    steps = [
        TaskStep(id="A", description="A", dependencies=[]),
        TaskStep(id="B", description="B", dependencies=["A"]),
        TaskStep(id="C", description="C", dependencies=["B"]),
    ]
    session_id = _make_session(steps)

    # 模拟 Reason 调用：第一步正常返回，第二步时设置中断标志
    call_count = {"n": 0}

    async def fake_chat(messages, model, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return json.dumps({"tool": "write_file", "params": {"path": "a"}})
        # 第二步之前设置中断标志（模拟 pause 端点）
        store = get_agent_session_store()
        session = store.get(session_id)
        session.interrupt_flag = True
        store.save(session)
        return json.dumps({"tool": "write_file", "params": {"path": "b"}})

    with patch("app.services.react_loop.chat_completion_text", side_effect=fake_chat):
        await run_agent(session_id, tool_executor=FakeToolExecutor())

    store = get_agent_session_store()
    session = store.get(session_id)

    # 步骤 A 完成（第一步正常执行）
    step_a = next(s for s in session.plan if s.id == "A")
    assert step_a.status == "done"
    # 步骤 B 被回收为 pending（pause 发生在其 _reason 期间）
    step_b = next(s for s in session.plan if s.id == "B")
    assert step_b.status == "pending"
    # 步骤 C 没有执行
    step_c = next(s for s in session.plan if s.id == "C")
    assert step_c.status == "pending"
    # 仅完成了 A
    assert session.done_steps == 1

    print("✅ 中断标志：循环在检测到 interrupt_flag 后停止，running 步骤回收为 pending")


# ============================================================
# 工具执行失败
# ============================================================

@pytest.mark.asyncio
async def test_tool_execution_failure_marks_step_failed():
    """
    工具执行抛出 ToolExecutionError 时，步骤标记为 failed，
    循环继续尝试下一个可执行步骤。
    """
    from app.services.tool_executor import ToolExecutionError

    class FailingExecutor(ToolExecutor):
        async def execute(self, tool, params):
            raise ToolExecutionError("模拟工具执行失败")

    steps = [
        TaskStep(id="A", description="A", dependencies=[]),
        TaskStep(id="B", description="B", dependencies=[]),  # 无依赖，可独立执行
    ]
    session_id = _make_session(steps)

    async def fake_chat(messages, model, **kwargs):
        return json.dumps({"tool": "write_file", "params": {"path": "x"}})

    with patch("app.services.react_loop.chat_completion_text", side_effect=fake_chat):
        await run_agent(session_id, tool_executor=FailingExecutor())

    store = get_agent_session_store()
    session = store.get(session_id)

    # 两个步骤都应标记为 failed
    assert all(s.status == "failed" for s in session.plan)
    for step in session.plan:
        assert "工具执行失败" in (step.observation or "")

    print("✅ 工具执行失败：步骤标记 failed，循环继续验证通过")


# ============================================================
# Reason 失败（LLM 返回非法 JSON）
# ============================================================

@pytest.mark.asyncio
async def test_reason_failure_marks_step_failed():
    """
    Reason 阶段 LLM 返回非法 JSON 时，_reason 返回 None，
    步骤标记为 failed。
    """
    steps = [
        TaskStep(id="A", description="A", dependencies=[]),
    ]
    session_id = _make_session(steps)

    async def fake_chat(messages, model, **kwargs):
        return "这不是 JSON"

    with patch("app.services.react_loop.chat_completion_text", side_effect=fake_chat):
        await run_agent(session_id, tool_executor=FakeToolExecutor())

    store = get_agent_session_store()
    session = store.get(session_id)

    step_a = session.plan[0]
    assert step_a.status == "failed"
    assert "Reason" in (step_a.observation or "")

    print("✅ Reason 失败：步骤标记 failed 验证通过")


# ============================================================
# DAG 依赖调度验证
# ============================================================

@pytest.mark.asyncio
async def test_loop_respects_dag_dependencies():
    """
    循环按 DAG 依赖顺序执行步骤：B 依赖 A，必须等 A 完成后才能执行 B。
    """
    steps = [
        TaskStep(id="A", description="A", dependencies=[]),
        TaskStep(id="B", description="B", dependencies=["A"]),
        TaskStep(id="C", description="C", dependencies=["B"]),
    ]
    session_id = _make_session(steps)
    executor = FakeToolExecutor()

    executed_order = []

    async def fake_chat(messages, model, **kwargs):
        # 从 messages 中提取当前步骤 ID
        for msg in messages:
            if msg["role"] == "user":
                # "请执行步骤 A：..."
                import re
                m = re.search(r"步骤\s+([^：:\s]+)", msg["content"])
                if m:
                    executed_order.append(m.group(1))
        return json.dumps({"tool": "write_file", "params": {"path": "x"}})

    with patch("app.services.react_loop.chat_completion_text", side_effect=fake_chat):
        await run_agent(session_id, tool_executor=executor)

    # 执行顺序应为 A -> B -> C
    assert executed_order == ["A", "B", "C"]

    print("✅ DAG 依赖调度：A→B→C 顺序执行验证通过")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
