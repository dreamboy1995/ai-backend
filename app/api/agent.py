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

from app.config import settings
from app.models.agent import (
    AgentActionResponse,
    AgentSession,
    AgentStatusResponse,
    AskUserRespondRequest,
    AskUserRespondResponse,
    END_REASON_CONFIRMING,
    END_REASON_IDLE,
    PlanPreviewItem,
    PlanRequest,
    PlanResponse,
    TERMINAL_END_REASONS,
    ToolConfirmRequest,
    ToolConfirmResponse,
)
from app.services.agent_session_store import get_agent_session_store
from app.services.agent_state_machine import (
    AgentStateMachine,
    InvalidPlanError,
)
from app.services.planner import PlannerError, plan as planner_plan
from app.services.react_loop import (
    handle_ask_user_response,
    handle_tool_confirm,
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


def _ensure_session_alive(session: AgentSession) -> None:
    """
    检查会话是否已进入终态（已死亡），若是则返回 409 Conflict。

    终态原因：completed / max_iter / timeout / error。
    这些状态下不允许 start / resume / pause，防止前端误点击触发异常。
    """
    if session.end_reason in TERMINAL_END_REASONS:
        raise HTTPException(
            status_code=409,
            detail=(
                f"会话已结束（{session.end_reason}），无法执行该操作。"
                f"结束原因：{session.end_message or '未知'}"
            ),
        )


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
        end_reason=END_REASON_IDLE,
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
        end_reason=session.end_reason,
        end_message=session.end_message,
        # S8 确认链路：前端 Builder 面板据此弹出「确认写入」浮层
        pending_confirmation_id=session.pending_confirmation_id,
        pending_confirmation_prompt=session.pending_confirmation_prompt,
        pending_confirmation_preview=session.pending_confirmation_preview,
        pending_confirmation_tool=session.pending_confirmation_tool,
        # S8 第 79-80 天：熔断状态
        consecutive_failures=session.consecutive_failures,
        is_fused=session.end_reason == "fused",
        fused_threshold=settings.TOOL_FAIL_FUSE_LIMIT,
        # S8 第 79-80 天：沙箱模式
        sandbox_mode=session.sandbox_mode,
        # S9 第 83-84 天：全局自修复熔断状态
        total_retries_used=session.total_retries_used,
        max_total_retries=session.max_total_retries,
        # S9 第 85-86 天：测试沙箱集成
        test_results=session.test_results,
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
    # 会话已死亡（终态）时拒绝启动，防止前端误点击触发异常
    _ensure_session_alive(session)

    if is_agent_running(session_id):
        return AgentActionResponse(success=True, message="Agent 已在执行中")

    # 清除可能残留的中断/暂停标志及结束标记
    session.is_executing = True
    session.is_paused = False
    session.interrupt_flag = False
    session.pending_question = None
    session.end_reason = None
    session.end_message = None
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
    # 会话已死亡（终态）时拒绝暂停，防止前端误点击触发异常
    _ensure_session_alive(session)
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
    # 会话已死亡（终态）时拒绝恢复，防止前端误点击触发异常
    _ensure_session_alive(session)

    # 先清除暂停/中断标志，保存到存储（循环会从存储同步这些标志）
    session.is_paused = False
    session.interrupt_flag = False
    session.is_executing = True
    session.pending_question = None
    # 恢复执行时清除结束标记
    session.end_reason = None
    session.end_message = None
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


@router.post("/confirm", response_model=ToolConfirmResponse)
async def confirm_tool_execution(req: ToolConfirmRequest):
    """
    S8 确认链路：处理用户对工具执行的确认/拒绝。

    当 Agent 执行 write_file / run_command / git_commit 等需要确认的工具时，
    ReAct 循环暂停，将 confirmation_id 写入 AgentSession.pending_confirmation_id。
    Builder 面板轮询 status 接口发现非空后弹出「确认写入」浮层，
    用户点击「允许」或「拒绝」后调用本接口恢复执行。

    执行流程（见 react_loop.handle_tool_confirm）：
      1. 校验 confirmation_id 与会话一致（防止伪造请求）。
      2. allow → 调用 tool_registry.confirm_tool 真正落盘；
         deny → 直接标记步骤 failed。
      3. 清除 pending_confirmation_* 字段，重启 ReAct 循环。
    """
    session = _get_session_or_404(req.session_id)

    success, result_summary = await handle_tool_confirm(
        session_id=req.session_id,
        confirmation_id=req.confirmation_id,
        action=req.action,
    )

    if success:
        logger.info(
            f"[Agent] 用户确认工具执行: session={req.session_id}, "
            f"action={req.action}, confirmation_id={req.confirmation_id}"
        )
        action_label = "允许" if req.action == "allow" else "拒绝"
        return ToolConfirmResponse(
            success=True,
            message=f"已{action_label}工具执行，Agent 继续运行",
            result_summary=result_summary,
        )
    else:
        raise HTTPException(
            status_code=400,
            detail=result_summary or "确认操作失败",
        )

