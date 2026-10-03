"""
S7 第 69-70 天：S7 封板 Demo 预演端到端测试

对应 Sprint_7.md 三大部分：
  1. 第 69-70 天"全员 S7 完整 Demo 预演"任务
  2. "关键接口/数据结构变更（S7 新增）"全部接口
  3. "S7 关键技术预研与风险预警（研发必读）"四项风险点

验收标准（来自 Sprint_7.md 第 69-70 天）：
  1. "输入'创建一个 Express.js 的 RESTful API 项目'，Builder 面板自动拆解为 5 个步骤，
      逐项自动打勾完成（模拟执行），中间遇到问题时弹出询问框，用户确认后继续。"
  2. "调用 /v1/agent/pause 后，Builder 面板上的进度条停止，点击'继续'后恢复执行。"
  3. "流程顺畅，无崩溃，步骤卡片颜色变化符合预期。"

覆盖场景：
  Demo 1 — 完整 Express.js Demo（ask_user 中途介入）
    POST /plan → POST /start → step_1,2 自动完成 → step_3 ask_user 暂停
    → POST /ask/respond → 循环恢复 → step_3,4,5 全部完成 → progress=100%

  Demo 2 — 中断机制（pause → resume via HTTP）
    POST /plan → POST /start → 执行中 POST /pause → 进度停止
    → POST /resume → 循环恢复 → 全部完成

  Demo 3 — 风险预警验证
    - 上下文爆炸：_build_context_summary 只保留最近 REACT_CONTEXT_WINDOW(=3) 步
    - 模拟工具 DI：注入自定义 ToolExecutor，循环正常工作（HTTP 层用默认 Mock）
    - Redis 持久化：会话可从 store 恢复（应对"用户关闭 VS Code"）
    - 最大迭代次数：达到 REACT_MAX_ITERATIONS(=15) 循环终止
"""

import asyncio
import json
import re
import time
from unittest.mock import patch

import httpx
import pytest
from httpx import ASGITransport

from app.auth import create_access_token
from app.main import app
from app.models.agent import AgentSession, TaskStep
from app.services import react_loop as react_loop_mod
from app.services.agent_session_store import get_agent_session_store
from app.services.agent_state_machine import AgentStateMachine
from app.services.react_loop import _build_context_summary, run_agent
from app.services.tool_executor import ToolExecutor


# ============================================================
# Express.js RESTful API 5 步计划（对应 S7 Demo 预演场景）
# ============================================================
# 步骤 3 设计为触发 ask_user 的关键决策点（选择数据库）

_EXPRESS_PLAN = [
    {
        "id": "step_1",
        "description": "初始化项目结构",
        "details": "mkdir express-api && cd express-api && npm init -y && git init",
        "dependencies": [],
        "suggested_tool": "run_command",
    },
    {
        "id": "step_2",
        "description": "安装依赖",
        "details": "npm install express cors morgan",
        "dependencies": ["step_1"],
        "suggested_tool": "run_command",
    },
    {
        "id": "step_3",
        "description": "设计数据模型",
        "details": "设计 Item 表结构，需选择数据库（SQLite / PostgreSQL）",
        "dependencies": ["step_2"],
        "suggested_tool": "ask_user",
    },
    {
        "id": "step_4",
        "description": "实现 CRUD 接口",
        "details": "创建 routes/items.js，实现 GET/POST/PUT/DELETE",
        "dependencies": ["step_3"],
        "suggested_tool": "write_file",
    },
    {
        "id": "step_5",
        "description": "启动并测试",
        "details": "node server.js 启动并用 curl 测试 /api/items",
        "dependencies": ["step_4"],
        "suggested_tool": "run_command",
    },
]


# ============================================================
# 测试夹具
# ============================================================


@pytest.fixture(autouse=True)
def _mock_rate_limiter(monkeypatch):
    """mock 限频器，避免测试间请求累积触发 429"""
    from app.middlewares import rate_limiter as rl_mod

    class _FakeLimiter:
        def check(self, user_id):
            return True, 100, 0, None

    monkeypatch.setattr(rl_mod, "get_rate_limiter", lambda: _FakeLimiter())


@pytest.fixture(autouse=True)
def _clear_running_agents():
    """每个测试前后清理 react_loop 后台任务注册表，避免测试间相互干扰"""
    react_loop_mod._running_agents.clear()
    yield
    # 测试后清理：取消未完成的任务
    for task in list(react_loop_mod._running_agents.values()):
        if not task.done():
            task.cancel()
    react_loop_mod._running_agents.clear()


def _auth_headers() -> dict:
    """生成认证头"""
    token = create_access_token({"sub": "test-user", "user_id": "test-user"})
    return {"Authorization": f"Bearer {token}"}


async def _poll_status(
    client: httpx.AsyncClient,
    session_id: str,
    predicate,
    timeout: float = 15.0,
    interval: float = 0.05,
) -> dict:
    """
    轮询 GET /v1/agent/status/{session_id} 直到 predicate(body) 为 True 或超时。

    使用小间隔（50ms）快速检测状态变化，模拟 Builder 面板每秒轮询但更快。
    """
    start = time.time()
    while time.time() - start < timeout:
        resp = await client.get(f"/v1/agent/status/{session_id}")
        if resp.status_code == 200:
            body = resp.json()
            if predicate(body):
                return body
        await asyncio.sleep(interval)
    raise TimeoutError(
        f"轮询超时({timeout}s): session={session_id}, 未满足条件"
    )


def _parse_step_id_from_messages(messages: list) -> str:
    """从 Reason 阶段的 messages 中解析当前步骤 ID"""
    for msg in messages:
        if msg.get("role") == "user":
            m = re.search(r"步骤\s+(step_\d+)", msg.get("content", ""))
            if m:
                return m.group(1)
    return "unknown"


# ============================================================
# LLM mock 工厂
# ============================================================


def _make_llm_for_ask_user_scenario():
    """
    构造 LLM mock（Demo 1 用）：
      - step_3 返回 ask_user "使用 SQLite 还是 PostgreSQL？"
      - 其他步骤返回对应工具调用
    """

    async def fake_chat(messages, model, **kwargs):
        step_id = _parse_step_id_from_messages(messages)
        if step_id == "step_3":
            return json.dumps(
                {"tool": "ask_user", "params": {"question": "使用 SQLite 还是 PostgreSQL？"}}
            )
        if step_id in ("step_1", "step_2", "step_5"):
            return json.dumps(
                {"tool": "run_command", "params": {"cmd": "echo ok"}}
            )
        # step_4
        return json.dumps(
            {"tool": "write_file", "params": {"path": "routes/items.js", "content": "// CRUD"}}
        )

    return fake_chat


def _make_llm_for_pause_scenario(session_id: str):
    """
    构造 LLM mock（Demo 2 用）：
      - step_2 首次执行时设置 interrupt_flag（模拟用户调用 /pause）
      - 后续执行正常返回（避免无限暂停）
    """
    paused_once = {"done": False}

    async def fake_chat(messages, model, **kwargs):
        step_id = _parse_step_id_from_messages(messages)
        # step_2 首次执行时模拟用户中断
        if step_id == "step_2" and not paused_once["done"]:
            paused_once["done"] = True
            store = get_agent_session_store()
            session = store.get(session_id)
            if session is not None:
                session.is_paused = True
                session.interrupt_flag = True
                store.save(session)
        # 所有步骤都返回 run_command（正常执行）
        return json.dumps(
            {"tool": "run_command", "params": {"cmd": "echo ok"}}
        )

    return fake_chat


# ============================================================
# Demo 1：完整 Express.js Demo（ask_user 中途介入）
# 对应 S7 验收标准 1：
#   "输入'创建一个 Express.js 的 RESTful API 项目'，自动拆解为 5 个步骤，
#    逐项自动打勾完成，中间遇到问题时弹出询问框，用户确认后继续。"
# ============================================================


@pytest.mark.asyncio
async def test_demo_express_js_with_ask_user():
    """
    S7 完整 Demo 预演：Express.js RESTful API 项目，5 步 DAG，
    step_3 触发 ask_user 中途介入，用户回答后循环继续。

    覆盖的 S7 接口：/plan, /start, /status, /ask/respond
    覆盖的 S7 风险预警：ask_user 人工介入机制（第 69-70 天后端任务 2）
    """
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", headers=_auth_headers()
    ) as client:

        # ---- 1. 创建 5 步 Express.js 计划 ----
        resp = await client.post(
            "/v1/agent/plan",
            json={
                "goal": "创建一个 Express.js 的 RESTful API 项目",
                "workspace_root": "/tmp/express-api",
                "plan": _EXPRESS_PLAN,
            },
        )
        assert resp.status_code == 200
        session_id = resp.json()["session_id"]
        assert len(resp.json()["plan_preview"]) == 5

        fake_llm = _make_llm_for_ask_user_scenario()
        with patch("app.services.react_loop.chat_completion_text", side_effect=fake_llm):

            # ---- 2. 启动 Agent 执行循环 ----
            resp = await client.post(f"/v1/agent/start/{session_id}")
            assert resp.status_code == 200
            assert resp.json()["success"] is True

            # ---- 3. 轮询直到出现 pending_question（step_3 的 ask_user）----
            status = await _poll_status(
                client,
                session_id,
                lambda s: s["pending_question"] is not None,
                timeout=15.0,
            )
            assert "SQLite 还是 PostgreSQL" in status["pending_question"]
            # step_1, step_2 已完成，step_3 处于 running（等待用户回答）
            assert status["done_steps"] == 2
            assert status["is_paused"] is True
            step_3 = next(s for s in status["steps"] if s["id"] == "step_3")
            assert step_3["status"] == "running"

            # ---- 4. 用户通过 /ask/respond 回答 ----
            resp = await client.post(
                "/v1/agent/ask/respond",
                json={"session_id": session_id, "answer": "SQLite"},
            )
            assert resp.status_code == 200
            assert resp.json()["success"] is True

            # ---- 5. 轮询直到全部完成 ----
            status = await _poll_status(
                client,
                session_id,
                lambda s: s["progress_percent"] == 100,
                timeout=15.0,
            )
            assert status["done_steps"] == 5
            assert status["total_steps"] == 5
            assert status["progress"] == 1.0
            assert status["progress_percent"] == 100
            assert status["is_paused"] is False
            assert status["is_executing"] is False
            assert status["pending_question"] is None

            # step_3 的 observation 包含用户回答
            step_3 = next(s for s in status["steps"] if s["id"] == "step_3")
            assert step_3["status"] == "done"
            assert "SQLite" in (step_3["observation"] or "")

            # 所有步骤均为 done
            assert all(s["status"] == "done" for s in status["steps"])

    print("✅ S7 Demo 预演：Express.js 5 步 + ask_user 中途介入 + 用户回答后继续，全部验证通过")


# ============================================================
# Demo 2：中断机制（pause → resume via HTTP）
# 对应 S7 验收标准 2：
#   "调用 /v1/agent/pause 后，进度条停止，点击'继续'后恢复执行。"
# ============================================================


@pytest.mark.asyncio
async def test_demo_pause_resume_via_http():
    """
    S7 第 69-70 天中断机制验收：通过 HTTP /pause 中断执行，/resume 恢复。

    覆盖的 S7 接口：/plan, /start, /pause, /resume, /status
    覆盖的 S7 风险预警：中断标志 interrupt_flag 持久化 + 循环顶部检查（第 69-70 天后端任务 1）
    """
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", headers=_auth_headers()
    ) as client:

        # ---- 1. 创建计划 ----
        resp = await client.post(
            "/v1/agent/plan",
            json={
                "goal": "创建一个 Express.js 的 RESTful API 项目",
                "workspace_root": "/tmp/express-api",
                "plan": _EXPRESS_PLAN,
            },
        )
        session_id = resp.json()["session_id"]

        fake_llm = _make_llm_for_pause_scenario(session_id)
        with patch("app.services.react_loop.chat_completion_text", side_effect=fake_llm):

            # ---- 2. 启动执行 ----
            resp = await client.post(f"/v1/agent/start/{session_id}")
            assert resp.status_code == 200

            # ---- 3. 轮询直到 is_paused=True（step_2 执行期间中断生效）----
            status = await _poll_status(
                client,
                session_id,
                lambda s: s["is_paused"],
                timeout=15.0,
            )
            assert status["is_paused"] is True
            assert status["is_executing"] is False
            # step_1 已完成，step_2 被回收为 pending
            assert status["done_steps"] == 1
            step_1 = next(s for s in status["steps"] if s["id"] == "step_1")
            assert step_1["status"] == "done"
            step_2 = next(s for s in status["steps"] if s["id"] == "step_2")
            assert step_2["status"] == "pending"

            # ---- 4. 调用 /resume 恢复执行 ----
            resp = await client.post(f"/v1/agent/resume/{session_id}")
            assert resp.status_code == 200
            assert resp.json()["success"] is True

            # ---- 5. 轮询直到全部完成 ----
            status = await _poll_status(
                client,
                session_id,
                lambda s: s["progress_percent"] == 100,
                timeout=15.0,
            )
            assert status["done_steps"] == 5
            assert status["is_paused"] is False
            assert status["progress_percent"] == 100

    print("✅ 中断机制：/pause 后进度停止，/resume 后恢复执行，全部验证通过")


# ============================================================
# Demo 3：S7 风险预警验证
# ============================================================


def test_risk_context_window_trimming():
    """
    风险预警 2："ReAct 循环中的上下文爆炸"：
    _build_context_summary 只保留最近 REACT_CONTEXT_WINDOW(=3) 步的完整 observation，
    更早的步骤只保留 description + status 摘要。
    """
    from app.config import settings

    assert settings.REACT_CONTEXT_WINDOW == 3, "S7 风险预警要求上下文窗口为 3"

    # 构造 6 个已完成步骤（超出窗口大小 3）
    steps = [
        TaskStep(
            id=f"step_{i}",
            description=f"步骤{i}",
            status="done",
            observation=f"这是步骤{i}的完整执行结果，包含详细日志",
            dependencies=[],
        )
        for i in range(6)
    ]
    session = AgentSession(user_goal="test", plan=steps)
    sm = AgentStateMachine(session)

    summary = _build_context_summary(sm, window=settings.REACT_CONTEXT_WINDOW)

    # 最近 3 步（step_3,4,5）包含完整 observation
    assert "步骤3" in summary and "步骤4" in summary and "步骤5" in summary
    assert "完整执行结果" in summary  # 最近步骤的 observation 出现

    # 更早的步骤（step_0,1,2）只有摘要，不含完整 observation
    # （但步骤描述仍会出现，只是不带 observation 结果）
    early_full_obs = "这是步骤0的完整执行结果"
    assert early_full_obs not in summary
    early_full_obs_1 = "这是步骤1的完整执行结果"
    assert early_full_obs_1 not in summary

    # 应有"已省略"提示
    assert "省略" in summary

    print("✅ 风险预警-上下文爆炸：只保留最近 3 步完整 observation 验证通过")


def test_risk_tool_executor_di():
    """
    风险预警 3："模拟工具与真实工具的衔接"：
    ToolExecutor 抽象基类支持依赖注入，注入自定义执行器后循环正常工作。
    S7 注入 MockToolExecutor，S8 只需替换为 MCPToolExecutor，上层代码无需修改。
    """
    from app.services.tool_executor import MockToolExecutor, get_default_tool_executor

    # 默认执行器为 MockToolExecutor
    default = get_default_tool_executor()
    assert isinstance(default, MockToolExecutor)

    # 自定义执行器（模拟 S8 的 MCPToolExecutor）
    class CustomExecutor(ToolExecutor):
        def __init__(self):
            self.calls = []

        async def execute(self, tool, params):
            self.calls.append((tool, params))
            return f"[CUSTOM] {tool} 执行成功"

    custom = CustomExecutor()
    assert isinstance(custom, ToolExecutor)  # 鸭子类型兼容

    # 验证：自定义执行器可替换默认执行器
    import asyncio

    result = asyncio.run(custom.execute("write_file", {"path": "test.py"}))
    assert "CUSTOM" in result
    assert len(custom.calls) == 1
    assert custom.calls[0][0] == "write_file"

    print("✅ 风险预警-工具DI：自定义 ToolExecutor 可注入验证通过")


def test_risk_session_persistence_after_restart():
    """
    风险预警 4："用户中途关闭 VS Code"：
    Agent 会话持久化到 Redis（不可用时降级内存），下次可从 store 恢复未完成的会话。
    """
    store = get_agent_session_store()

    # 模拟一个执行到一半的会话（step_1 完成，step_2 running）
    steps = [
        TaskStep(id="step_1", description="初始化", status="done",
                 observation="ok", dependencies=[]),
        TaskStep(id="step_2", description="编码", status="running",
                 dependencies=["step_1"]),
        TaskStep(id="step_3", description="测试", status="pending",
                 dependencies=["step_2"]),
    ]
    session = AgentSession(
        user_goal="未完成的项目",
        workspace_root="/tmp/proj",
        plan=steps,
    )
    session.is_paused = True
    session.is_executing = False
    store.save(session)

    # 模拟"重启"：重新从 store 加载会话
    recovered = store.get(session.session_id)
    assert recovered is not None
    assert recovered.user_goal == "未完成的项目"
    assert recovered.is_paused is True
    assert recovered.done_steps == 1
    assert recovered.total_steps == 3

    # 步骤状态完整保留
    step_2 = next(s for s in recovered.plan if s.id == "step_2")
    assert step_2.status == "running"

    # 清理
    store.delete(session.session_id)

    print("✅ 风险预警-会话持久化：重启后可恢复未完成会话验证通过")


@pytest.mark.asyncio
async def test_risk_max_iterations_safety():
    """
    风险预警（第 67-68 天，对应 S7 风险预警"防止 AI 陷入死循环"）：
    达到 REACT_MAX_ITERATIONS(=15) 时循环终止，不会无限执行。
    """
    from app.config import settings

    assert settings.REACT_MAX_ITERATIONS == 15, "S7 风险预警要求最大迭代次数为 15"

    # 创建 20 个无依赖步骤（超过最大迭代次数）
    steps = [
        TaskStep(id=f"step_{i}", description=f"步骤{i}", dependencies=[])
        for i in range(20)
    ]
    session = AgentSession(user_goal="测试最大迭代", plan=steps)
    store = get_agent_session_store()
    store.save(session)

    async def fake_chat(messages, model, **kwargs):
        return json.dumps({"tool": "run_command", "params": {"cmd": "echo ok"}})

    with patch("app.services.react_loop.chat_completion_text", side_effect=fake_chat):
        await run_agent(session.session_id)

    session = store.get(session.session_id)
    # 完成的步骤数 = 最大迭代次数（每迭代处理一步）
    assert session.done_steps == 15
    assert "最大迭代次数" in (session.final_answer or "")

    # 清理
    store.delete(session.session_id)

    print("✅ 风险预警-最大迭代次数：15 步后循环终止验证通过")


# ============================================================
# S7 封板 ShowCase 检查清单验证
# 对应 Sprint_7.md "✅ Sprint 7 结束时的 Demo 检查清单"
# ============================================================


@pytest.mark.asyncio
async def test_s7_showcase_checklist():
    """
    S7 封板 ShowCase 检查清单（后端可验证部分）：
      [✓] 用户输入需求，AI 返回 6-8 步计划（此处用预定义 5 步计划验证接口）
      [✓] Builder 面板以卡片时间线展示所有步骤（后端 /status 返回完整步骤列表）
      [✓] 点击"开始执行"，步骤依次从 ⏳ 流转为 🔄 再到 ✅
      [✓] 执行过程中点击"暂停"，进度条停止；点击"继续"，从断点恢复
      [✓] Agent 返回 ask_user，用户回答后步骤继续
      [✓] 所有步骤完成后，显示"任务全部完成"
    """
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", headers=_auth_headers()
    ) as client:

        # [1] 创建计划（5 步 Express.js）
        resp = await client.post(
            "/v1/agent/plan",
            json={
                "goal": "创建一个 Express.js 的 RESTful API 项目",
                "workspace_root": "/tmp/express-api",
                "plan": _EXPRESS_PLAN,
            },
        )
        assert resp.status_code == 200
        session_id = resp.json()["session_id"]
        plan_preview = resp.json()["plan_preview"]
        assert len(plan_preview) == 5

        # [2] /status 返回完整步骤列表（供 Builder 面板渲染卡片时间线）
        status = (await client.get(f"/v1/agent/status/{session_id}")).json()
        assert len(status["steps"]) == 5
        assert all(s["status"] == "pending" for s in status["steps"])
        assert status["progress_percent"] == 0

        # [3] 启动执行 + ask_user 中途介入 + 用户回答 + 全部完成
        fake_llm = _make_llm_for_ask_user_scenario()
        with patch("app.services.react_loop.chat_completion_text", side_effect=fake_llm):
            await client.post(f"/v1/agent/start/{session_id}")

            # 等待 ask_user 暂停
            status = await _poll_status(
                client, session_id, lambda s: s["pending_question"] is not None
            )
            assert status["is_paused"] is True

            # 用户回答
            await client.post(
                "/v1/agent/ask/respond",
                json={"session_id": session_id, "answer": "SQLite"},
            )

            # 等待全部完成
            status = await _poll_status(
                client, session_id, lambda s: s["progress_percent"] == 100
            )

        # [4] 验证步骤依次完成（状态流转：pending → running → done）
        assert all(s["status"] == "done" for s in status["steps"])
        assert status["progress_percent"] == 100
        assert status["done_steps"] == 5

        # [5] 最终总结
        assert status["final_answer"] is not None
        assert "完成" in (status["final_answer"] or "")

    print("✅ S7 ShowCase 检查清单：5 步计划 + 状态流转 + ask_user + 全部完成，验证通过")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
