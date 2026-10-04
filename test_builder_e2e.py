"""
Builder 模式端到端测试

模拟用户在 VSCode Builder 面板输入"创建一个 Express.js 的 RESTful API 项目"，
点击"生成计划"后整个生命周期的端到端流程：

  1. POST /v1/agent/plan（不传 plan 字段，触发后端 Planner 生成 DAG）
  2. GET  /v1/agent/status/{session_id} 验证初始状态（5 步 pending，进度 0%）
  3. POST /v1/agent/start/{session_id} 启动 ReAct 后台循环
  4. 轮询 GET /v1/agent/status 观察步骤逐步完成
  5. 验证最终 progress=100%、end_reason=completed
  6. ask_user 人工介入分支：某步 Reason 返回 ask_user → pending_question 透出 →
     POST /v1/agent/ask/respond → 循环恢复 → completed
  7. 错误路径：未登录 401 / 无 plan 字段且 Planner 失败 502 / 不存在的 session 404

设计要点：
  - 不调用真实 LLM：monkeypatch app.services.planner.chat_completion_text
    与 app.services.react_loop.chat_completion_text，按调用上下文返回不同预设 JSON。
  - 不依赖 Redis：app.services.agent_session_store 内存降级已支持。
  - 后台 asyncio 循环同步化：mock start_agent_loop 让其 await run_agent 完成再返回，
    这样 TestClient 的同步调用能等到循环结束、状态已落盘。
  - 复用项目现有测试范式：参考 test_api_agent.py / test_react_loop.py。
"""

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.auth import create_access_token
from app.main import app
from app.models.agent import END_REASON_ASK_USER, END_REASON_COMPLETED
from app.services.agent_session_store import get_agent_session_store
from app.services.react_loop import run_agent
from app.services.tool_executor import ToolExecutor, set_default_tool_executor


# ============================================================
# 用户输入：模拟 Builder 面板提交的"创建一个 Express.js 的 RESTful API 项目"
# ============================================================

USER_GOAL = "创建一个 Express.js 的 RESTful API 项目"
WORKSPACE_ROOT = "/tmp/express-api"

# Planner 返回的 5 步 DAG（与用户在 Builder 面板看到的一致）
PLANNER_PLAN_JSON = json.dumps({
    "plan": [
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
            "description": "编写入口文件",
            "details": "创建 server.js，配置 Express 中间件与基础路由",
            "dependencies": ["step_2"],
            "suggested_tool": "write_file",
        },
        {
            "id": "step_4",
            "description": "实现 CRUD 接口",
            "details": "实现 /api/items 的 GET/POST/PUT/DELETE 接口",
            "dependencies": ["step_3"],
            "suggested_tool": "write_file",
        },
        {
            "id": "step_5",
            "description": "启动并测试",
            "details": "node server.js 启动并用 curl 测试接口",
            "dependencies": ["step_4"],
            "suggested_tool": "run_command",
        },
    ]
})


# ============================================================
# 测试夹具
# ============================================================

class _FakeToolExecutor(ToolExecutor):
    """测试用工具执行器：记录调用、返回硬编码 observation"""

    def __init__(self):
        self.calls = []

    async def execute(self, tool, params):
        self.calls.append((tool, params))
        return f"[mock] {tool} 执行成功，params={params}"


@pytest.fixture()
def client(monkeypatch):
    """
    带认证头的 TestClient。
    mock 限频器，避免测试间请求累积触发 429。
    """
    from app.middlewares import rate_limiter as rl_mod

    class _FakeLimiter:
        def check(self, user_id):
            return True, 100, 0, None

    monkeypatch.setattr(rl_mod, "get_rate_limiter", lambda: _FakeLimiter())

    token = create_access_token({"sub": "test-user", "user_id": "test-user"})
    headers = {"Authorization": f"Bearer {token}"}
    with TestClient(app) as c:
        c.headers.update(headers)
        yield c


def _mock_start_agent_loop_sync(monkeypatch):
    """
    将 start_agent_loop 改造为同步执行 run_agent，等循环跑完再返回。
    这样 TestClient 的同步 POST /start 调用返回时，循环已经结束、状态已落盘，
    便于后续 GET /status 直接验证终态，无需复杂的轮询等待。

    同时也覆盖了"循环作为后台任务"的真实路径——run_agent 本身被原样调用。
    """

    async def _sync_start(session_id, tool_executor=None):
        # 复用 react_loop 的真实实现，让测试覆盖 ReAct 主循环代码路径
        await run_agent(session_id, tool_executor=tool_executor)
        return True

    monkeypatch.setattr("app.api.agent.start_agent_loop", _sync_start)


# ============================================================
# 场景 1：用户输入需求 → 创建计划 → 查询状态
# ============================================================

def test_builder_e2e_create_plan_and_query_status(client, monkeypatch):
    """
    模拟 Builder 面板点击"生成计划"：
      - 前端 POST /v1/agent/plan（无 plan 字段）
      - 后端 Planner 调用 LLM 拆解为 5 步 DAG
      - 返回 session_id + plan_preview（5 项）
      - 前端立即 GET /status 拿完整状态，渲染 5 张卡片
    """
    # mock Planner 的 LLM 调用
    monkeypatch.setattr(
        "app.services.planner.chat_completion_text",
        AsyncMock(return_value=PLANNER_PLAN_JSON),
    )

    # === 用户在 Builder 面板点击"生成计划" ===
    resp = client.post("/v1/agent/plan", json={
        "goal": USER_GOAL,
        "workspace_root": WORKSPACE_ROOT,
        "model": "glm-4.5-air",
        # 注意：不传 plan，触发后端 Planner
    })
    assert resp.status_code == 200, f"创建计划失败: {resp.text}"
    body = resp.json()

    assert body["session_id"], "应返回 session_id"
    assert len(body["plan_preview"]) == 5, "应返回 5 步预览"
    assert body["plan_preview"][0]["description"] == "初始化项目结构"

    session_id = body["session_id"]

    # === Builder 面板立即轮询 /status 拿完整状态 ===
    status_resp = client.get(f"/v1/agent/status/{session_id}")
    assert status_resp.status_code == 200
    status = status_resp.json()

    assert status["session_id"] == session_id
    assert status["user_goal"] == USER_GOAL
    assert status["workspace_root"] == WORKSPACE_ROOT
    assert status["total_steps"] == 5
    assert status["done_steps"] == 0
    assert status["progress"] == 0.0
    assert status["progress_percent"] == 0
    assert status["is_executing"] is False
    assert status["is_paused"] is False
    assert status["end_reason"] == "idle"  # 计划已生成但未启动

    # 5 个步骤全部 pending，状态机/前端可正确渲染卡片
    for step in status["steps"]:
        assert step["status"] == "pending"

    print("✅ 场景 1：用户输入需求 → 生成计划 → 状态查询 验证通过")


# ============================================================
# 场景 2：启动执行 → 后台 ReAct 循环 → 全部完成
# ============================================================

def test_builder_e2e_start_and_complete_all_steps(client, monkeypatch):
    """
    模拟用户点击"开始执行"按钮：
      - 前端 POST /v1/agent/start/{session_id}
      - 后端启动 ReAct 循环（Reason → Act → Observe）
      - 前端每秒轮询 /status 观察步骤逐步 done
      - 最终 progress=100%，end_reason=completed
    """
    # 1. 先创建计划（mock Planner）
    monkeypatch.setattr(
        "app.services.planner.chat_completion_text",
        AsyncMock(return_value=PLANNER_PLAN_JSON),
    )
    plan_resp = client.post("/v1/agent/plan", json={
        "goal": USER_GOAL,
        "workspace_root": WORKSPACE_ROOT,
    })
    session_id = plan_resp.json()["session_id"]

    # 2. mock ReAct 循环的 Reason 阶段：每步返回 write_file 调用
    #    ToolExecutor 用 _FakeToolExecutor 记录调用——通过 set_default_tool_executor
    #    注入到 react_loop 全局，让 run_agent 在 tool_executor=None 时拿到本测试的实例
    fake_executor = _FakeToolExecutor()
    set_default_tool_executor(fake_executor)

    async def _react_chat(messages, model, **kwargs):
        # 每步 Reason 都决定调用 write_file
        return json.dumps({"tool": "write_file",
                           "params": {"path": "main.js", "content": "..."}})

    monkeypatch.setattr("app.services.react_loop.chat_completion_text",
                        AsyncMock(side_effect=_react_chat))
    # 后台循环同步化，便于同步断言
    _mock_start_agent_loop_sync(monkeypatch)

    # 3. 用户点击"开始执行"
    start_resp = client.post(f"/v1/agent/start/{session_id}")
    assert start_resp.status_code == 200
    assert start_resp.json()["success"] is True

    # 4. 模拟前端轮询：此时循环已跑完，应直接看到终态
    final = client.get(f"/v1/agent/status/{session_id}").json()
    assert final["done_steps"] == 5
    assert final["total_steps"] == 5
    assert final["progress"] == 1.0
    assert final["progress_percent"] == 100
    assert final["end_reason"] == END_REASON_COMPLETED
    assert final["is_executing"] is False
    assert final["final_answer"] == "所有任务步骤已完成"

    # 每步都有 observation
    for step in final["steps"]:
        assert step["status"] == "done"
        assert step["observation"]

    # 工具执行器被调用 5 次（每步一次）
    assert len(fake_executor.calls) == 5
    assert all(call[0] == "write_file" for call in fake_executor.calls)

    print("✅ 场景 2：开始执行 → 5 步全部完成 → progress=100% 验证通过")


# ============================================================
# 场景 3：ask_user 人工介入 → 用户回答 → 循环恢复 → 完成
# ============================================================

def test_builder_e2e_ask_user_intervention_and_resume(client, monkeypatch):
    """
    模拟 Agent 在执行中遇到决策点向用户提问：
      - 第 1 步 Reason 返回 ask_user
      - 后端设置 pending_question，循环暂停
      - 前端 GET /status 看到 pending_question，渲染输入框
      - 用户回答后 POST /v1/agent/ask/respond
      - 循环恢复，继续执行剩余步骤，最终 completed
    """
    # 1. 创建计划
    monkeypatch.setattr(
        "app.services.planner.chat_completion_text",
        AsyncMock(return_value=PLANNER_PLAN_JSON),
    )
    session_id = client.post("/v1/agent/plan", json={
        "goal": USER_GOAL,
        "workspace_root": WORKSPACE_ROOT,
    }).json()["session_id"]

    # 2. mock ReAct 循环：第 1 次调用返回 ask_user，后续返回 write_file
    call_count = {"n": 0}

    async def _react_chat(messages, model, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return json.dumps({"tool": "ask_user",
                               "params": {"question": "使用 SQLite 还是 PostgreSQL？"}})
        return json.dumps({"tool": "write_file",
                           "params": {"path": "x.js", "content": "..."}})

    monkeypatch.setattr("app.services.react_loop.chat_completion_text",
                        AsyncMock(side_effect=_react_chat))
    _mock_start_agent_loop_sync(monkeypatch)

    # 3. 启动 → 循环跑完后应处于 ask_user 暂停态
    client.post(f"/v1/agent/start/{session_id}")

    paused = client.get(f"/v1/agent/status/{session_id}").json()
    assert paused["pending_question"] == "使用 SQLite 还是 PostgreSQL？"
    assert paused["is_paused"] is True
    assert paused["end_reason"] == END_REASON_ASK_USER

    # 4. 用户回答 → 后端恢复循环
    respond_resp = client.post("/v1/agent/ask/respond", json={
        "session_id": session_id,
        "answer": "SQLite",
    })
    assert respond_resp.status_code == 200

    # 5. 循环恢复后跑完剩余步骤，最终 completed
    final = client.get(f"/v1/agent/status/{session_id}").json()
    assert final["end_reason"] == END_REASON_COMPLETED
    assert final["progress_percent"] == 100
    assert final["pending_question"] is None

    print("✅ 场景 3：ask_user 介入 → 用户回答 → 循环恢复 → 完成 验证通过")


# ============================================================
# 场景 4：错误路径——未登录 / Planner 失败 / 不存在的 session
# ============================================================

def test_builder_e2e_error_paths(client, monkeypatch):
    """
    错误路径覆盖，便于排查用户"点击无反应"是否源于后端 401/502/404：
      - 无 Authorization 头 → 401
      - Planner LLM 失败 → 502
      - 查询不存在的 session → 404
    """
    # 1. 未登录
    no_auth_client = TestClient(app)
    resp = no_auth_client.post("/v1/agent/plan", json={"goal": USER_GOAL})
    assert resp.status_code == 401, "无 Authorization 头应返回 401"

    # 2. Planner 失败（LLM 一直返回非 JSON）→ 502
    monkeypatch.setattr(
        "app.services.planner.chat_completion_text",
        AsyncMock(return_value="not a json"),
    )
    resp = client.post("/v1/agent/plan", json={
        "goal": USER_GOAL,
        "workspace_root": WORKSPACE_ROOT,
    })
    assert resp.status_code == 502, f"Planner 失败应返回 502，实际: {resp.status_code}"
    # 后端统一异常响应字段为 message（对齐 test_api_agent.py 的 400 校验）
    body = resp.json()
    assert "Planner" in body.get("message") or "Planner" in body.get("detail", "")

    # 3. 查询不存在的 session → 404
    resp = client.get("/v1/agent/status/nonexistent-session-id")
    assert resp.status_code == 404

    print("✅ 场景 4：错误路径（401/502/404）验证通过")


# ============================================================
# 场景 5：模拟"点击生成计划后无日志"——后端服务未启动时前端 fetch 失败
# ============================================================

def test_builder_e2e_diagnostic_logs_visible(client, monkeypatch, caplog):
    """
    验证：用户报告"点击生成计划后无日志"时，至少后端 /plan 接口被调用应有 INFO 日志。
    若后端服务根本没启动，前端 fetch 会失败但扩展端入口日志仍应输出（已在
    builderPanel.ts:_handleWebviewMessage 的 console.log 处覆盖）。

    本测试确认：当请求到达后端时，agent.py 的 logger.info 一定会输出，
    可作为"日志是否到达后端"的判定基线。
    """
    import logging
    caplog.set_level(logging.INFO, logger="app.api.agent")

    monkeypatch.setattr(
        "app.services.planner.chat_completion_text",
        AsyncMock(return_value=PLANNER_PLAN_JSON),
    )
    client.post("/v1/agent/plan", json={
        "goal": USER_GOAL,
        "workspace_root": WORKSPACE_ROOT,
    })

    # 后端必须输出"计划创建成功"日志——若用户在 Builder 面板点击后看不到任何日志，
    # 且后端日志中也没有这条，说明请求根本未到达后端（前端 fetch 失败/路由错/服务未起）。
    assert any("计划创建成功" in r.message for r in caplog.records), (
        "后端 agent.py 的 '计划创建成功' INFO 日志未输出——"
        "若用户看不到任何日志，根因在前端（webview JS 未执行 / 后端服务未启动 / "
        "fetch 到错误端口），而非后端处理逻辑"
    )

    print("✅ 场景 5：后端 /plan 接口被命中时必输出 '计划创建成功' 日志 验证通过")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
