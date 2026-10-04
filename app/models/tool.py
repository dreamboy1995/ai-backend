"""
S8 第 71-72 天：工具调用标准 Schema（MCP 协议适配层数据结构）

对应 Sprint_8.md「关键接口/数据结构变更（S8 新增）」：

  POST /v1/tool/execute
  Request:  { session_id: string; tool_call: ToolCall }
  Response: ToolResult

  POST /v1/tool/confirm
  Request:  { session_id: string; confirmation_id: string; action: 'allow' | 'deny' }
  Response: { success: boolean; result?: ToolResult }

设计要点（兼顾 S8 风险预警）：
1. ToolCall.tool_name 使用 Literal 限定合法工具名，防止非法工具调用。
2. ToolResult.requires_confirmation 标记需要用户手动确认的操作（如 write_file、run_command），
   配合 confirmation_id 实现「先预览 Diff，再确认执行」的安全护栏。
3. confirmation_id 由后端生成并持久化到 PendingConfirmationStore，
   防止前端伪造确认请求（S8 风险预警：命令注入 / 越权执行）。
"""

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field


# ============================================================
# 工具名称枚举
# ============================================================
# S8 第 71-72 天定义的标准工具集。
# 新增工具需同步更新此处、tool_registry.py 的注册表、以及 S8 文档。
ToolName = Literal[
    "read_file",      # 读取文件内容（含行号定位）
    "write_file",     # 写入文件（需确认 + Diff 预览）
    "run_command",    # 执行终端命令（需确认 + 危险命令黑名单）
    "grep_search",    # 代码正则搜索（ripgrep / Python 降级）
    "git_commit",     # Git 提交（需确认 + 变更文件列表预览）
]


class ToolCall(BaseModel):
    """
    工具调用请求（Agent 内部与插件端共用）。

    - tool_name:  工具名称（Literal 限定），非法值在 Pydantic 校验阶段即被拒绝。
    - arguments:  工具参数字典，由各工具 handler 自行校验必填字段。
                  例如 read_file 需要 {"file_path": "...", "start_line": 1, "end_line": 50}。
    """

    tool_name: ToolName
    arguments: Dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """
    工具执行结果。

    - success:             是否执行成功。
    - output:              成功时的输出文本（文件内容 / 命令 stdout / 搜索结果等）。
    - error:               失败时的错误描述。
    - requires_confirmation: 是否需要用户手动确认。为 True 时，
                  插件端应弹出确认框，用户确认后调用 /v1/tool/confirm。
    - confirmation_prompt: 确认框展示给用户的提示文本（如 "即将写入文件 main.py，是否继续？"）。
    - confirmation_id:     后端生成的确认凭证，用户确认时需原样回传。
                  仅当 requires_confirmation=True 时有值。
    """

    success: bool
    output: str = ""
    error: Optional[str] = None
    requires_confirmation: bool = False
    confirmation_prompt: Optional[str] = None
    confirmation_id: Optional[str] = None


# ============================================================
# API 请求/响应 DTO
# ============================================================

class ToolExecuteRequest(BaseModel):
    """POST /v1/tool/execute 请求体"""

    session_id: str = Field(..., min_length=1, description="Agent 会话 ID")
    tool_call: ToolCall


class ToolConfirmRequest(BaseModel):
    """POST /v1/tool/confirm 请求体"""

    session_id: str = Field(..., min_length=1, description="Agent 会话 ID")
    confirmation_id: str = Field(..., min_length=1, description="工具执行返回的确认凭证")
    action: Literal["allow", "deny"] = Field(..., description="allow=允许执行，deny=拒绝执行")


class ToolConfirmResponse(BaseModel):
    """POST /v1/tool/confirm 响应体"""

    success: bool
    result: Optional[ToolResult] = None
