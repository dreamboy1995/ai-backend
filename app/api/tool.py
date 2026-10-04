"""
S8 第 71-72 天：工具执行 API 路由

对应 Sprint_8.md「关键接口/数据结构变更（S8 新增）」：

  POST /v1/tool/execute
    Request:  { session_id: string; tool_call: ToolCall }
    Response: ToolResult

  POST /v1/tool/confirm
    Request:  { session_id: string; confirmation_id: string; action: 'allow' | 'deny' }
    Response: { success: boolean; result?: ToolResult }

设计要点：
  - /execute 不直接执行危险操作（write_file / run_command / git_commit），
    而是返回 requires_confirmation=True + confirmation_id，由插件弹出确认框。
  - /confirm 接收用户的 allow/deny 决策，才真正执行（或拒绝）。
  - session_id 透传给 tool_registry 用于审计日志。
  - workspace_root 从 AgentSession 中获取（若 session_id 对应存在的会话），
    否则降级为进程当前工作目录。
"""

import logging
import os

from fastapi import APIRouter, HTTPException

from app.models.tool import (
    ToolConfirmRequest,
    ToolConfirmResponse,
    ToolExecuteRequest,
    ToolResult,
)
from app.services.agent_session_store import get_agent_session_store
from app.services.tool_registry import confirm_tool, execute_tool

logger = logging.getLogger(__name__)
router = APIRouter()


def _get_workspace_root(session_id: str) -> str:
    """
    从 AgentSession 获取 workspace_root；会话不存在时降级为 cwd。

    S8 统一使用相对路径（如 src/main.py），需要 workspace_root 来解析绝对路径。
    """
    try:
        session = get_agent_session_store().get(session_id)
        if session is not None and session.workspace_root:
            return session.workspace_root
    except Exception as e:
        logger.debug(f"[ToolAPI] 从会话获取 workspace_root 失败: {e}")
    # 降级：进程当前工作目录
    return os.getcwd()


@router.post("/execute", response_model=ToolResult)
async def execute_tool_api(req: ToolExecuteRequest) -> ToolResult:
    """
    执行工具调用。

    - 对于 read_file / grep_search 等只读工具，直接执行并返回结果。
    - 对于 write_file / run_command / git_commit 等写操作，
      返回 requires_confirmation=True + confirmation_id，
      插件端弹出确认框后调用 /confirm。
    """
    workspace_root = _get_workspace_root(req.session_id)
    result = await execute_tool(
        tool_call=req.tool_call,
        workspace_root=workspace_root,
        session_id=req.session_id,
    )
    return result


@router.post("/confirm", response_model=ToolConfirmResponse)
async def confirm_tool_api(req: ToolConfirmRequest) -> ToolConfirmResponse:
    """
    处理用户对工具的确认操作。

    - action='allow'：真正执行该工具（写入文件 / 运行命令 / 提交 Git）。
    - action='deny'：拒绝执行，返回失败结果。
    """
    result = await confirm_tool(
        confirmation_id=req.confirmation_id,
        action=req.action,
        session_id=req.session_id,
    )
    return ToolConfirmResponse(success=result.success, result=result)
