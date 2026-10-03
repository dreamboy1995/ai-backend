"""
S7 第 61-62 天：Agent 状态机单元测试

验收标准（来自 Sprint_7.md）：
  "创建包含 3 个步骤（B 依赖 A，C 依赖 B）的 AgentSession，
   调用 get_next_runnable_step 返回 A，标记 A 完成后再次调用返回 B，
   符合 DAG 依赖逻辑。"

额外覆盖：
  - 状态迁移合法性
  - DAG 校验（依赖 ID 不存在 / 循环依赖）
  - failed 步骤阻塞后续调度
  - 并发步骤调度（多个无依赖步骤同时 runnable）
  - 会话存储（Redis 降级内存）
  - MockToolExecutor 执行
"""

import asyncio
import time

from app.models.agent import AgentSession, TaskStep
from app.services.agent_session_store import (
    AgentSessionStore,
    get_agent_session_store,
)
from app.services.agent_state_machine import (
    AgentStateMachine,
    InvalidPlanError,
    StateTransitionError,
)
from app.services.tool_executor import MockToolExecutor


def _make_dag_session() -> AgentSession:
    """
    创建验收标准中的 3 步 DAG：
      A (无依赖) -> B (依赖 A) -> C (依赖 B)
    """
    a = TaskStep(id="A", description="步骤A", dependencies=[])
    b = TaskStep(id="B", description="步骤B", dependencies=["A"])
    c = TaskStep(id="C", description="步骤C", dependencies=["B"])
    return AgentSession(user_goal="测试 DAG", plan=[a, b, c])


# ============================================================
# 验收标准：DAG 依赖调度
# ============================================================

def test_dag_runnable_order():
    """
    验收标准核心：3 步 DAG（B 依赖 A，C 依赖 B），
    get_next_runnable_step 依次返回 A -> B -> C。
    """
    session = _make_dag_session()
    sm = AgentStateMachine(session)

    # 初始：A 是唯一可执行步骤
    step = sm.get_next_runnable_step()
    assert step is not None and step.id == "A", "第一步应返回 A"
    assert step.status == "pending"

    # 标记 A 完成
    sm.mark_running("A")
    sm.mark_done("A", observation="A 执行成功")

    # A 完成后，B 变为可执行
    step = sm.get_next_runnable_step()
    assert step is not None and step.id == "B", "A 完成后应返回 B"

    # 标记 B 完成
    sm.mark_running("B")
    sm.mark_done("B", observation="B 执行成功")

    # B 完成后，C 变为可执行
    step = sm.get_next_runnable_step()
    assert step is not None and step.id == "C", "B 完成后应返回 C"

    # 标记 C 完成
    sm.mark_running("C")
    sm.mark_done("C", observation="C 执行成功")

    # 全部完成，无更多可执行步骤
    assert sm.get_next_runnable_step() is None
    assert sm.is_all_done()
    assert session.progress == 1.0
    print("✅ DAG 依赖调度顺序验证通过: A -> B -> C")


def test_dependency_blocks_runnable():
    """
    依赖未完成时，依赖它的步骤不应被调度。
    B 依赖 A，A 未完成时 get_next_runnable_step 不应返回 B。
    """
    session = _make_dag_session()
    sm = AgentStateMachine(session)

    # A 是唯一可执行的，B/C 因依赖未满足不可执行
    runnable = sm.get_runnable_steps()
    assert len(runnable) == 1
    assert runnable[0].id == "A"

    # 标记 A 为 running（未 done），B 仍不可执行
    sm.mark_running("A")
    assert sm.get_next_runnable_step() is None, "A running 时 B 仍不可执行"

    print("✅ 依赖未完成时阻塞调度验证通过")


# ============================================================
# 状态迁移
# ============================================================

def test_state_transitions_valid():
    """合法状态迁移：pending -> running -> done"""
    session = _make_dag_session()
    sm = AgentStateMachine(session)

    step = sm.mark_running("A")
    assert step.status == "running"

    step = sm.mark_done("A", observation="ok")
    assert step.status == "done"
    assert step.observation == "ok"
    print("✅ 合法状态迁移验证通过")


def test_state_transition_invalid():
    """非法状态迁移应抛异常"""
    session = _make_dag_session()
    sm = AgentStateMachine(session)

    # pending 不能直接 mark_done
    try:
        sm.mark_done("A")
        assert False, "pending -> done 应抛异常"
    except StateTransitionError:
        pass

    # done 不能再 mark_running
    sm.mark_running("A")
    sm.mark_done("A")
    try:
        sm.mark_running("A")
        assert False, "done -> running 应抛异常"
    except StateTransitionError:
        pass

    print("✅ 非法状态迁移抛异常验证通过")


def test_failed_transition():
    """running -> failed 迁移"""
    session = _make_dag_session()
    sm = AgentStateMachine(session)

    sm.mark_running("A")
    step = sm.mark_failed("A", "工具调用超时")
    assert step.status == "failed"
    assert "工具调用超时" in (step.observation or "")
    assert sm.is_any_failed()
    print("✅ failed 状态迁移验证通过")


def test_blocked_and_reset():
    """blocked 状态与重置重试"""
    session = _make_dag_session()
    sm = AgentStateMachine(session)

    sm.mark_running("A")
    step = sm.mark_blocked("A", "等待用户确认")
    assert step.status == "blocked"

    # blocked 步骤不可被调度
    assert sm.get_next_runnable_step() is None

    # 重置为 pending
    step = sm.reset_step("A")
    assert step.status == "pending"
    assert step.retry_count == 1
    assert step.observation is None

    # 重置后可再次被调度
    assert sm.get_next_runnable_step().id == "A"
    print("✅ blocked 与 reset 验证通过")


# ============================================================
# DAG 校验（对应 S7 风险预警：依赖 ID 强校验 + 循环依赖检测）
# ============================================================

def test_validate_plan_missing_dependency():
    """依赖 ID 不存在应抛 InvalidPlanError"""
    a = TaskStep(id="A", description="A", dependencies=[])
    b = TaskStep(id="B", description="B", dependencies=["non_existent"])
    session = AgentSession(user_goal="test", plan=[a, b])
    sm = AgentStateMachine(session)

    try:
        sm.validate_plan()
        assert False, "依赖不存在应抛异常"
    except InvalidPlanError as e:
        assert "non_existent" in str(e)
    print("✅ 依赖 ID 不存在校验通过")


def test_validate_plan_circular_dependency():
    """循环依赖应抛 InvalidPlanError"""
    a = TaskStep(id="A", description="A", dependencies=["B"])
    b = TaskStep(id="B", description="B", dependencies=["A"])
    session = AgentSession(user_goal="test", plan=[a, b])
    sm = AgentStateMachine(session)

    try:
        sm.validate_plan()
        assert False, "循环依赖应抛异常"
    except InvalidPlanError as e:
        assert "循环依赖" in str(e)
    print("✅ 循环依赖检测通过")


def test_validate_plan_valid():
    """合法 DAG 校验通过"""
    session = _make_dag_session()
    sm = AgentStateMachine(session)
    sm.validate_plan()  # 不应抛异常
    print("✅ 合法 DAG 校验通过")


# ============================================================
# 多步骤并行调度（无依赖的步骤同时可执行）
# ============================================================

def test_parallel_runnable_steps():
    """多个无依赖步骤应同时为 runnable"""
    a = TaskStep(id="A", description="A", dependencies=[])
    b = TaskStep(id="B", description="B", dependencies=[])
    c = TaskStep(id="C", description="C", dependencies=["A"])
    session = AgentSession(user_goal="test", plan=[a, b, c])
    sm = AgentStateMachine(session)

    runnable = sm.get_runnable_steps()
    assert len(runnable) == 2
    ids = {s.id for s in runnable}
    assert ids == {"A", "B"}
    print("✅ 并行可执行步骤调度通过")


# ============================================================
# 会话存储（Redis 降级内存）
# ============================================================

def test_session_store_memory_fallback():
    """
    会话存储：保存后能取回，删除后不存在。
    （Redis 不可用时自动降级到内存，本测试验证降级路径）
    """
    store = AgentSessionStore(ttl_seconds=60)
    session = _make_dag_session()

    store.save(session)
    assert store.exists(session.session_id)

    fetched = store.get(session.session_id)
    assert fetched is not None
    assert fetched.session_id == session.session_id
    assert len(fetched.plan) == 3

    store.delete(session.session_id)
    assert not store.exists(session.session_id)
    print("✅ 会话存储（内存降级）验证通过")


# ============================================================
# MockToolExecutor
# ============================================================

def test_mock_tool_executor():
    """Mock 工具执行器返回模拟成功消息"""
    executor = MockToolExecutor()

    obs = asyncio.run(executor.execute("write_file", {"path": "main.py"}))
    assert "main.py" in obs
    assert "模拟" in obs

    obs = asyncio.run(executor.execute("run_command", {"cmd": "ls"}))
    assert "ls" in obs
    print("✅ MockToolExecutor 验证通过")


# ============================================================
# 端到端：完整执行一遍 3 步 DAG
# ============================================================

def test_full_dag_execution():
    """
    端到端：通过状态机完整执行 A -> B -> C 三步，
    验证 progress、is_all_done、current_step_index 等字段。
    """
    session = _make_dag_session()
    sm = AgentStateMachine(session)
    sm.validate_plan()

    assert session.progress == 0.0
    assert session.current_step_index == -1

    # 执行 A
    step = sm.get_next_runnable_step()
    sm.mark_running(step.id)
    assert session.current_step_index == 0
    sm.mark_done(step.id, "A done")
    assert session.progress == 1 / 3

    # 执行 B
    step = sm.get_next_runnable_step()
    sm.mark_running(step.id)
    assert session.current_step_index == 1
    sm.mark_done(step.id, "B done")
    assert session.progress == 2 / 3

    # 执行 C
    step = sm.get_next_runnable_step()
    sm.mark_running(step.id)
    sm.mark_done(step.id, "C done")
    assert session.progress == 1.0

    assert sm.is_all_done()
    assert not sm.is_any_failed()
    print("✅ 端到端 3 步 DAG 执行验证通过")


if __name__ == "__main__":
    test_dag_runnable_order()
    test_dependency_blocks_runnable()
    test_state_transitions_valid()
    test_state_transition_invalid()
    test_failed_transition()
    test_blocked_and_reset()
    test_validate_plan_missing_dependency()
    test_validate_plan_circular_dependency()
    test_validate_plan_valid()
    test_parallel_runnable_steps()
    test_session_store_memory_fallback()
    test_mock_tool_executor()
    test_full_dag_execution()
    print("\n" + "=" * 60)
    print("全部 Agent 状态机测试通过！")
    print("=" * 60)
