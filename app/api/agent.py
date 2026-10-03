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
  - Planner（任务规划器）在第 63-64 天实现，/plan 未传入预拆解计划时由大模型生成。
  - ReAct 循环在第 67-68 天实现，start/pause/resume/ask/respond 已与
    react_loop.run_agent 打通：start 启动后台循环，pause 设置中断标志，
    resume 重启循环，ask/respond 处理人工介入后恢复执行。
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
)
from app.services.agent_session_store import get_agent_session_store
from app.services.agent_state_machine import (
    AgentStateMachine,
    InvalidPlanError,
)
from app.services.planner import PlannerError, plan as planner_plan
from app.services.react_loop import (
    handle_ask_user_response,
    is_agent_running,
    resume_agent_loop,
    start_agent_loop,
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

    S7 第 63-64 天已接入 Planner：当调用方未传入预拆解计划时，
    由大模型将自然语言需求拆解为 DAG 任务图（含步骤数量校验、依赖合法性校验、
    解析失败自动重试）。
    调用方也可在 req.plan 中直接传入预拆解的步骤列表（测试/联调用）。
    """
    plan_steps = req.plan
    if plan_steps is None:
        # 调用 Planner 生成任务计划（S7 第 63-64 天）
        try:
            plan_steps = await planner_plan(
                goal=req.goal,
                workspace_root=req.workspace_root,
                model=req.model,
            )
        except PlannerError as e:
            raise HTTPException(
                status_code=502,
                detail=f"Planner 生成任务计划失败: {e}",
            )

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
        progress_percent=int(round(session.progress * 100)),
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

    S7 第 67-68 天：触发 ReAct 主循环（Reason -> Act -> Observe）。
    循环作为后台 asyncio.Task 运行，HTTP 接口立即返回，
    Builder 面板通过 /status 轮询观察步骤状态流转。
    """
    session = _get_session_or_404(session_id)

    if is_agent_running(session_id):
        return AgentActionResponse(success=True, message="Agent 已在执行中")

    # 清除可能残留的中断/暂停标志
    session.is_executing = True
    session.is_paused = False
    session.interrupt_flag = False
    session.pending_question = None
    get_agent_session_store().save(session)

    started = await start_agent_loop(session_id)
    if started:
        logger.info(f"[Agent] 启动执行循环: session={session_id}")
        return AgentActionResponse(success=True, message="Agent 执行已启动")
    else:
        return AgentActionResponse(success=True, message="Agent 已在执行中")


@router.post("/pause/{session_id}", response_model=AgentActionResponse)
async def pause_agent(session_id: str):
    """
    暂停 Agent 执行。

    设置 is_paused=True / interrupt_flag=True，并立即将 is_executing 置为 False，
    让 Builder 面板能立即看到"已暂停"状态。
    ReAct 循环在每次迭代顶部从存储同步控制标志，检测到中断标志后 break，
    其 finally 块也会将 is_executing 置为 False（幂等）。
    """
    session = _get_session_or_404(session_id)
    session.is_paused = True
    session.interrupt_flag = True
    session.is_executing = False
    get_agent_session_store().save(session)
    logger.info(f"[Agent] 请求暂停执行: session={session_id}")
    return AgentActionResponse(success=True, message="Agent 已暂停，循环将在当前步骤结束后停止")


@router.post("/resume/{session_id}", response_model=AgentActionResponse)
async def resume_agent(session_id: str):
    """
    继续 Agent 执行（清除暂停与中断标志，重新启动循环）。

    无论旧循环是否仍在运行（可能卡在 LLM 调用中），都先清除暂停标志。
    若旧循环已结束，则启动新循环；若旧循环仍在运行，它会在下一次迭代
    顶部检测到 interrupt_flag=False 后继续执行。
    """
    session = _get_session_or_404(session_id)

    # 先清除暂停/中断标志，保存到存储（循环会从存储同步这些标志）
    session.is_paused = False
    session.interrupt_flag = False
    session.is_executing = True
    session.pending_question = None
    get_agent_session_store().save(session)

    if is_agent_running(session_id):
        # 旧循环仍在运行，让它继续（下一次迭代会检测到标志已清除）
        logger.info(f"[Agent] 继续执行（旧循环仍在运行）: session={session_id}")
        return AgentActionResponse(success=True, message="Agent 已恢复执行")

    resumed = await resume_agent_loop(session_id)
    if resumed:
        logger.info(f"[Agent] 继续执行循环: session={session_id}")
        return AgentActionResponse(success=True, message="Agent 已恢复执行")
    else:
        return AgentActionResponse(success=False, message="Agent 恢复失败")


@router.post("/ask/respond", response_model=AskUserRespondResponse)
async def respond_to_ask_user(req: AskUserRespondRequest):
    """
    人工介入回复。

    当 Agent 在某步骤返回 ask_user 工具时，循环暂停并等待用户输入。
    用户通过此接口提交回答后，后端将发起提问的步骤标记为 done
    （observation 记录用户回答），然后恢复 ReAct 循环继续执行后续步骤。
    """
    session = _get_session_or_404(req.session_id)
    if not session.pending_question:
        raise HTTPException(
            status_code=400,
            detail="当前会话没有待回答的问题",
        )

    success = await handle_ask_user_response(req.session_id, req.answer)
    if success:
        logger.info(
            f"[Agent] 用户回复 ask_user: session={req.session_id}, "
            f"answer={req.answer[:50]}"
        )
        return AskUserRespondResponse(
            success=True, message="回复已接收，Agent 继续执行"
        )
    else:
        raise HTTPException(
            status_code=500,
            detail="处理用户回复失败",
        )

