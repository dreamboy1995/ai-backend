"""
S7 第 61-62 天：Agent API 路由

实现 S7 新增的 Agent 接口（骨架）：
  - POST /v1/agent/plan          : 创建任务计划，返回 session_id
  - GET  /v1/agent/status/{id}   : 查询会话完整状态（供 Builder 面板轮询）
  - POST /v1/agent/start/{id}    : 启动 Agent 执行循环（S7 第 67-68 天实现 ReAct 循环）
  - POST /v1/agent/pause/{id}    : 暂停执行（S7 第 69-70 天实现中断机制）
  - POST /v1/agent/resume/{id}   : 继续执行
  - POST /v1/agent/ask/respond   : 人工介入回复

说明：
  - Planner（任务规划器）在第 63-64 天实现，当前 /plan 接受调用方传入的预拆解计划，
    或生成一个默认占位计划，用于联调状态机与 UI。
  - ReAct 循环在第 67-68 天实现，当前 start/pause/resume 仅做状态字段切换，
    为后续循环逻辑铺路。
"""

import logging

from fastapi import APIRouter, HTTPException

from app.models.agent import (
    AgentActionResponse,
    AgentSession,
    AgentStatusResponse,
    AskUserRespondRequest,
    AskUserRespondResponse,
    PlanPreviewItem,
    PlanRequest,
    PlanResponse,
    TaskStep,
)
from app.services.agent_session_store import get_agent_session_store
from app.services.agent_state_machine import (
    AgentStateMachine,
    InvalidPlanError,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def _get_session_or_404(session_id: str) -> AgentSession:
    """从存储中获取会话，不存在则抛 404"""
    session = get_agent_session_store().get(session_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail=f"Agent 会话不存在: {session_id}",
        )
    return session


@router.post("/plan", response_model=PlanResponse)
async def create_plan(req: PlanRequest):
    """
    创建 Agent 任务计划。

    S7 第 63-64 天将接入 Planner，由大模型将自然语言需求拆解为 DAG 任务图。
    当前阶段（第 61-62 天）支持两种方式：
      1. 调用方在 req.plan 中直接传入预拆解的步骤列表（测试/联调用）。
      2. 未传入时，后端生成一个默认占位计划，用于验证状态机与 UI 联通。
    """
    plan_steps = req.plan
    if plan_steps is None:
        # 生成默认占位计划（S7 第 63-64 天替换为真实 Planner 输出）
        plan_steps = _generate_default_plan(req.goal)

    session = AgentSession(
        user_goal=req.goal,
        workspace_root=req.workspace_root,
        model=req.model,
        plan=plan_steps,
    )

    # 校验任务图合法性（依赖 ID 存在性 + 循环依赖检测）
    sm = AgentStateMachine(session)
    try:
        sm.validate_plan()
    except InvalidPlanError as e:
        raise HTTPException(status_code=400, detail=f"任务计划不合法: {e}")

    # 持久化会话（Redis + 内存降级）
    get_agent_session_store().save(session)

    logger.info(
        f"[Agent] 计划创建成功: session={session.session_id}, "
        f"steps={len(plan_steps)}, goal={req.goal[:50]}"
    )

    return PlanResponse(
        session_id=session.session_id,
        plan_preview=[
            PlanPreviewItem(id=s.id, description=s.description)
            for s in plan_steps
        ],
    )


@router.get("/status/{session_id}", response_model=AgentStatusResponse)
async def get_status(session_id: str):
    """
    查询 Agent 会话完整状态。

    供 Builder 面板每秒轮询刷新，返回所有步骤的状态、观察结果与整体进度。
    """
    session = _get_session_or_404(session_id)
    return AgentStatusResponse(
        session_id=session.session_id,
        user_goal=session.user_goal,
        workspace_root=session.workspace_root,
        steps=session.plan,
        current_step_index=session.current_step_index,
        final_answer=session.final_answer,
        progress=session.progress,
        is_executing=session.is_executing,
        is_paused=session.is_paused,
        pending_question=session.pending_question,
        total_steps=session.total_steps,
        done_steps=session.done_steps,
    )


@router.post("/start/{session_id}", response_model=AgentActionResponse)
async def start_agent(session_id: str):
    """
    启动 Agent 执行循环。

    S7 第 67-68 天将实现真正的 ReAct 循环（Reason -> Act -> Observe）。
    当前阶段仅设置 is_executing 标志，为后续循环铺路。
    """
    session = _get_session_or_404(session_id)
    if session.is_executing:
        return AgentActionResponse(success=True, message="Agent 已在执行中")
    session.is_executing = True
    session.is_paused = False
    session.interrupt_flag = False
    get_agent_session_store().save(session)
    logger.info(f"[Agent] 启动执行: session={session_id}")
    return AgentActionResponse(success=True, message="Agent 执行已启动")


@router.post("/pause/{session_id}", response_model=AgentActionResponse)
async def pause_agent(session_id: str):
    """
    暂停 Agent 执行。

    S7 第 69-70 天：在 Redis 中存储 interrupt_flag，
    run_agent 循环顶部检查该标志后 break。
    """
    session = _get_session_or_404(session_id)
    session.is_paused = True
    session.interrupt_flag = True
    session.is_executing = False
    get_agent_session_store().save(session)
    logger.info(f"[Agent] 暂停执行: session={session_id}")
    return AgentActionResponse(success=True, message="Agent 已暂停")


@router.post("/resume/{session_id}", response_model=AgentActionResponse)
async def resume_agent(session_id: str):
    """继续 Agent 执行（清除暂停与中断标志）"""
    session = _get_session_or_404(session_id)
    session.is_paused = False
    session.interrupt_flag = False
    session.is_executing = True
    get_agent_session_store().save(session)
    logger.info(f"[Agent] 继续执行: session={session_id}")
    return AgentActionResponse(success=True, message="Agent 已恢复执行")


@router.post("/ask/respond", response_model=AskUserRespondResponse)
async def respond_to_ask_user(req: AskUserRespondRequest):
    """
    人工介入回复。

    当 Agent 在某步骤返回 ask_user 工具时，循环暂停并等待用户输入。
    用户通过此接口提交回答后，循环继续。
    """
    session = _get_session_or_404(req.session_id)
    if not session.pending_question:
        raise HTTPException(
            status_code=400,
            detail="当前会话没有待回答的问题",
        )
    # 清除待回答问题，标记中断标志为 False（循环可继续）
    session.pending_question = None
    session.interrupt_flag = False
    session.is_executing = True
    session.is_paused = False
    get_agent_session_store().save(session)
    logger.info(
        f"[Agent] 用户回复 ask_user: session={req.session_id}, answer={req.answer[:50]}"
    )
    return AskUserRespondResponse(success=True, message="回复已接收，Agent 继续执行")


# ------------------------------------------------------------------
# 默认占位计划生成器（S7 第 63-64 天替换为 Planner）
# ------------------------------------------------------------------

def _generate_default_plan(goal: str) -> list[TaskStep]:
    """
    生成默认占位计划，用于在 Planner 未实现前验证状态机与 UI。

    真实 Planner 会在第 63-64 天接入，届时此方法将被移除。
    """
    return [
        TaskStep(
            id="step_1",
            description="分析需求",
            details=f"分析用户需求：{goal}，确定技术栈与项目结构",
            dependencies=[],
            action="search_code",
            action_input={"query": goal},
        ),
        TaskStep(
            id="step_2",
            description="初始化项目",
            details="创建项目目录结构，初始化版本控制与配置文件",
            dependencies=["step_1"],
            action="run_command",
            action_input={"cmd": "mkdir project && cd project && git init"},
        ),
        TaskStep(
            id="step_3",
            description="实现核心功能",
            details="根据需求实现核心业务逻辑",
            dependencies=["step_2"],
            action="write_file",
            action_input={"path": "src/main.py", "content": "# TODO: implement"},
        ),
    ]
