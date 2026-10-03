"""
S7 第 65-66 天：Agent API 集成测试（Builder 面板 UI 与状态同步）

验收标准（来自 Sprint_7.md 第 65-66 天）：
  后端：
    1. /v1/agent/plan 接收用户需求，调用 Planner，创建 AgentSession 并存入 Redis，返回 session_id。
    2. /v1/agent/status/{session_id} 返回当前 AgentSession 的完整状态（含所有步骤）。
    3. 使用 Postman 调用 /v1/agent/plan，返回的 session_id 可用于查询状态。

  前端联动验收（后端需支撑）：
    - 后端返回包含 5 个步骤的计划后，Builder 面板渲染出 5 张卡片，进度条显示 0/5。
    - 当后端将步骤 1 状态改为 done 后，前端轮询到更新，卡片变绿，进度条变为 1/5。

覆盖场景：
  - POST /v1/agent/plan：传入预拆解计划，返回 session_id + plan_preview
  - GET  /v1/agent/status/{id}：返回完整状态，步骤列表、进度、执行标志
  - 进度同步：手动将步骤标记 done 后，status 返回更新后的进度（模拟前端轮询场景）
  - 404：查询不存在的 session_id
  - 400：传入含循环依赖的非法计划
  - POST /v1/agent/start|pause|resume：执行控制标志切换
  - POST /v1/agent/ask/respond：无待回答问题时返回 400
"""

import pytest
from fastapi.testclient import TestClient

from app.auth import create_access_token
from app.main import app
from app.models.agent import TaskStep
from app.services.agent_session_store import get_agent_session_store
from app.services.agent_state_machine import AgentStateMachine


# ============================================================
# 测试夹具
# ============================================================

@pytest.fixture()
def client(monkeypatch):
    """
    带认证头的 TestClient。
    同时 mock 限频器，避免测试间请求累积触发 429。
    """
    # mock 限频器：始终允许
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


# 一个合法的 5 步 DAG 计划（用于传入 /plan，避免调用真实 LLM）
def _valid_plan_payload():
    return {
        "goal": "做一个 Express.js 的 RESTful API 项目",
        "workspace_root": "/tmp/express-api",
        "model": "glm-4.5-air",
        "plan": [
            {"id": "step_1", "description": "初始化项目结构",
             "details": "mkdir express-api && cd express-api && npm init -y && git init",
             "dependencies": [], "suggested_tool": "run_command"},
            {"id": "step_2", "description": "安装依赖",
             "details": "npm install express cors morgan",
             "dependencies": ["step_1"], "suggested_tool": "run_command"},
            {"id": "step_3", "description": "编写入口文件",
             "details": "创建 server.js，配置 Express 中间件与基础路由",
             "dependencies": ["step_2"], "suggested_tool": "write_file"},
            {"id": "step_4", "description": "实现 CRUD 接口",
             "details": "实现 /api/items 的 GET/POST/PUT/DELETE 接口",
             "dependencies": ["step_3"], "suggested_tool": "write_file"},
            {"id": "step_5", "description": "启动并测试",
             "details": "node server.js 启动并用 curl 测试接口",
             "dependencies": ["step_4"], "suggested_tool": "run_command"},
        ],
    }


# ============================================================
# 验收标准 1：POST /v1/agent/plan
# ============================================================

def test_create_plan_returns_session_id(client):
    """
    验收标准：调用 /v1/agent/plan，返回 session_id + plan_preview，
    且 session_id 可用于后续查询状态。
    """
    resp = client.post("/v1/agent/plan", json=_valid_plan_payload())
    assert resp.status_code == 200
    body = resp.json()

    assert "session_id" in body and body["session_id"]
    assert "plan_preview" in body
    assert len(body["plan_preview"]) == 5
    # 预览只含 id + description
    for item in body["plan_preview"]:
        assert "id" in item and "description" in item

    print("✅ POST /v1/agent/plan 返回 session_id + 5 步预览验证通过")


def test_create_plan_without_plan_calls_planner(client, monkeypatch):
    """
    未传入 plan 时应调用 Planner 生成任务图。
    此处 mock app.api.agent.planner_plan 返回固定步骤，验证接口能正确接回。
    """
    from app.api import agent as agent_api

    async def fake_plan(goal, workspace_root="", model="glm-4.5-air"):
        return [
            TaskStep(id="s1", description="初始化", dependencies=[]),
            TaskStep(id="s2", description="编码", dependencies=["s1"]),
            TaskStep(id="s3", description="测试", dependencies=["s2"]),
            TaskStep(id="s4", description="部署", dependencies=["s3"]),
            TaskStep(id="s5", description="上线", dependencies=["s4"]),
        ]

    monkeypatch.setattr(agent_api, "planner_plan", fake_plan)

    resp = client.post("/v1/agent/plan", json={
        "goal": "做个项目",
        "workspace_root": "",
        "model": "glm-4.5-air",
    })
    assert resp.status_code == 200
    assert len(resp.json()["plan_preview"]) == 5
    print("✅ 未传 plan 时调用 Planner 生成任务图验证通过")


def test_create_plan_invalid_circular_dependency(client):
    """
    传入含循环依赖的计划应返回 400。
    对应 S7 风险预警："大模型经常在 dependencies 里填错 ID，后端必须做强校验"。
    """
    bad_plan = {
        "goal": "循环依赖测试",
        "plan": [
            {"id": "a", "description": "A", "dependencies": ["b"]},
            {"id": "b", "description": "B", "dependencies": ["a"]},
            {"id": "c", "description": "C", "dependencies": ["b"]},
            {"id": "d", "description": "D", "dependencies": ["c"]},
            {"id": "e", "description": "E", "dependencies": ["d"]},
        ],
    }
    resp = client.post("/v1/agent/plan", json=bad_plan)
    assert resp.status_code == 400
    assert "循环依赖" in resp.json()["message"]
    print("✅ 循环依赖计划返回 400 验证通过")


def test_create_plan_missing_dependency(client):
    """依赖了不存在的步骤 ID 应返回 400"""
    bad_plan = {
        "goal": "缺失依赖测试",
        "plan": [
            {"id": "a", "description": "A", "dependencies": []},
            {"id": "b", "description": "B", "dependencies": ["non_existent"]},
            {"id": "c", "description": "C", "dependencies": ["b"]},
            {"id": "d", "description": "D", "dependencies": ["c"]},
            {"id": "e", "description": "E", "dependencies": ["d"]},
        ],
    }
    resp = client.post("/v1/agent/plan", json=bad_plan)
    assert resp.status_code == 400
    assert "non_existent" in resp.json()["message"]
    print("✅ 依赖不存在的步骤 ID 返回 400 验证通过")


# ============================================================
# 验收标准 2：GET /v1/agent/status/{session_id}
# ============================================================

def test_get_status_returns_full_session(client):
    """
    验收标准：/v1/agent/status/{session_id} 返回 AgentSession 完整状态，
    包含所有步骤、进度、执行标志等（供 Builder 面板轮询）。
    """
    plan_resp = client.post("/v1/agent/plan", json=_valid_plan_payload())
    session_id = plan_resp.json()["session_id"]

    status_resp = client.get(f"/v1/agent/status/{session_id}")
    assert status_resp.status_code == 200
    body = status_resp.json()

    # 基础字段
    assert body["session_id"] == session_id
    assert body["user_goal"] == "做一个 Express.js 的 RESTful API 项目"
    assert body["workspace_root"] == "/tmp/express-api"
    assert body["total_steps"] == 5
    assert body["done_steps"] == 0

    # 所有步骤初始为 pending
    assert len(body["steps"]) == 5
    for step in body["steps"]:
        assert step["status"] == "pending"
        assert "id" in step and "description" in step and "details" in step

    # 进度初始为 0
    assert body["progress"] == 0.0
    assert body["progress_percent"] == 0
    assert body["is_executing"] is False
    assert body["is_paused"] is False
    assert body["current_step_index"] == -1
    print("✅ GET /v1/agent/status 返回完整会话状态验证通过")


def test_get_status_not_found(client):
    """查询不存在的 session_id 返回 404"""
    resp = client.get("/v1/agent/status/nonexistent-session-id")
    assert resp.status_code == 404
    print("✅ 查询不存在会话返回 404 验证通过")


def test_status_progress_updates_when_step_done(client):
    """
    前端联动验收：当后端将步骤 1 标记为 done 后，
    轮询 /status 应返回更新后的进度（1/5，progress_percent=20）。

    模拟 ReAct 循环执行后的状态变更：直接通过状态机修改步骤状态并保存，
    验证前端轮询能拿到最新状态。
    """
    plan_resp = client.post("/v1/agent/plan", json=_valid_plan_payload())
    session_id = plan_resp.json()["session_id"]

    # 直接操作状态机：标记 step_1 完成（模拟 ReAct 循环执行）
    store = get_agent_session_store()
    session = store.get(session_id)
    sm = AgentStateMachine(session)
    sm.mark_running("step_1")
    sm.mark_done("step_1", observation="项目初始化成功")
    store.save(session)

    # 前端轮询拿到更新后的状态
    status_resp = client.get(f"/v1/agent/status/{session_id}")
    assert status_resp.status_code == 200
    body = status_resp.json()

    assert body["done_steps"] == 1
    assert body["total_steps"] == 5
    assert body["progress"] == pytest.approx(0.2, abs=0.01)
    assert body["progress_percent"] == 20

    # step_1 状态变为 done，observation 已写入
    step_1 = next(s for s in body["steps"] if s["id"] == "step_1")
    assert step_1["status"] == "done"
    assert "项目初始化成功" in (step_1["observation"] or "")

    print("✅ 步骤完成后 status 进度更新（1/5, 20%）验证通过")


# ============================================================
# 执行控制接口：start / pause / resume
# ============================================================

def test_start_pause_resume_flow(client):
    """
    执行控制流程：start -> is_executing=True；
    pause -> is_paused=True, is_executing=False；
    resume -> is_paused=False, is_executing=True。
    """
    plan_resp = client.post("/v1/agent/plan", json=_valid_plan_payload())
    session_id = plan_resp.json()["session_id"]

    # start
    resp = client.post(f"/v1/agent/start/{session_id}")
    assert resp.status_code == 200
    assert resp.json()["success"] is True
    status = client.get(f"/v1/agent/status/{session_id}").json()
    assert status["is_executing"] is True
    assert status["is_paused"] is False

    # pause
    resp = client.post(f"/v1/agent/pause/{session_id}")
    assert resp.status_code == 200
    status = client.get(f"/v1/agent/status/{session_id}").json()
    assert status["is_paused"] is True
    assert status["is_executing"] is False

    # resume
    resp = client.post(f"/v1/agent/resume/{session_id}")
    assert resp.status_code == 200
    status = client.get(f"/v1/agent/status/{session_id}").json()
    assert status["is_paused"] is False
    assert status["is_executing"] is True

    print("✅ start -> pause -> resume 状态流转验证通过")


def test_start_idempotent(client):
    """重复 start 不应报错，返回 '已在执行中'"""
    plan_resp = client.post("/v1/agent/plan", json=_valid_plan_payload())
    session_id = plan_resp.json()["session_id"]

    client.post(f"/v1/agent/start/{session_id}")
    resp = client.post(f"/v1/agent/start/{session_id}")
    assert resp.status_code == 200
    assert resp.json()["success"] is True
    print("✅ 重复 start 幂等验证通过")


def test_control_endpoints_not_found(client):
    """对不存在的 session 调用 start/pause/resume 返回 404"""
    for path in ["/start", "/pause", "/resume"]:
        resp = client.post(f"/v1/agent{path}/nonexistent-id")
        assert resp.status_code == 404, f"{path} 应返回 404"
    print("✅ 控制接口对不存在会话返回 404 验证通过")


# ============================================================
# 人工介入接口：ask/respond
# ============================================================

def test_ask_respond_without_pending_question(client):
    """没有待回答问题时调用 ask/respond 返回 400"""
    plan_resp = client.post("/v1/agent/plan", json=_valid_plan_payload())
    session_id = plan_resp.json()["session_id"]

    resp = client.post("/v1/agent/ask/respond", json={
        "session_id": session_id,
        "answer": "SQLite",
    })
    assert resp.status_code == 400
    assert "待回答" in resp.json()["message"]
    print("✅ 无待回答问题时 ask/respond 返回 400 验证通过")


# ============================================================
# 端到端验收：完整 5 步计划 + 状态轮询
# ============================================================

def test_e2e_plan_and_polling(client):
    """
    端到端验收：
      1. POST /plan 创建 5 步计划
      2. GET /status 看到 5 个 pending 步骤，进度 0/5
      3. 逐步标记步骤 done，每次轮询 /status 验证进度递增
      4. 全部完成后进度 100%
    """
    plan_resp = client.post("/v1/agent/plan", json=_valid_plan_payload())
    session_id = plan_resp.json()["session_id"]
    store = get_agent_session_store()

    step_ids = ["step_1", "step_2", "step_3", "step_4", "step_5"]

    for i, step_id in enumerate(step_ids, 1):
        # 模拟 ReAct 循环执行该步骤
        session = store.get(session_id)
        sm = AgentStateMachine(session)
        sm.mark_running(step_id)
        sm.mark_done(step_id, observation=f"{step_id} 执行成功")
        store.save(session)

        # 前端轮询
        body = client.get(f"/v1/agent/status/{session_id}").json()
        assert body["done_steps"] == i
        assert body["total_steps"] == 5
        assert body["progress_percent"] == int(round(i / 5 * 100))
        # 当前步骤状态为 done
        step = next(s for s in body["steps"] if s["id"] == step_id)
        assert step["status"] == "done"

    # 全部完成
    final = client.get(f"/v1/agent/status/{session_id}").json()
    assert final["progress"] == 1.0
    assert final["progress_percent"] == 100
    assert final["done_steps"] == 5

    print("✅ 端到端：5 步计划逐步完成 + 轮询进度递增验证通过")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
