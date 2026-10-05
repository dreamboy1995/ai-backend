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
    "run_tests",      # S9：运行项目测试套件（pytest / jest 自动检测）
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
    - output:              成功时的输出。普通工具为 str（文件内容 / 命令 stdout / 搜索结果等）；
                  需要确认的工具（write_file / git_commit）为结构化 dict，格式见下：
                    * write_file: {"diff": {"files": [{path, old_content, new_content, diff}], "explanation": "..."}}
                    * git_commit: {"commit_message": "...", "git_changes": [{path, status, additions, deletions}], "diff": {...同上}}
                    * run_command: 纯 str，如 "npm install flask"
    - error:               失败时的错误描述。
    - requires_confirmation: 是否需要用户手动确认。为 True 时，
                  插件端应弹出确认框，用户确认后调用 /v1/tool/confirm。
    - confirmation_prompt: 确认框展示给用户的提示文本（如 "即将写入文件 main.py，是否继续？"）。
    - confirmation_id:     后端生成的确认凭证，用户确认时需原样回传。
                  仅当 requires_confirmation=True 时有值。
    - requires_interaction: S8 第 75-76 天新增。True 表示命令需要交互式输入
                  （如 npm init 无 --yes、python REPL 等），Agent 自动执行会卡住。
                  此时由插件提示用户在真实终端手动完成，再告知 Agent 继续。
                  对应 S8 风险预警："终端命令的交互式输入"。
    """

    success: bool
    output: Any = ""
    error: Optional[str] = None
    requires_confirmation: bool = False
    confirmation_prompt: Optional[str] = None
    confirmation_id: Optional[str] = None
    requires_interaction: bool = False


# ============================================================
# 终端日志流式消息（S8 第 75-76 天 + S9 扩展）
# ============================================================
# 对应 Sprint_8.md「关键接口/数据结构变更」：
#   ws://localhost:3000/v1/agent/stream/{session_id}
#   interface StreamMessage {
#     type: 'stdout' | 'stderr' | 'system' | 'repair_attempt';  // S9 新增
#     content: string;  // 可能包含 ANSI 颜色码
#     timestamp: string;
#     extra?: object;   // S9 新增：复杂事件的结构化 payload
#   }
#
# - stdout/stderr: 命令进程的实时输出（按行推送，可能含 ANSI 颜色码）
# - system:        系统事件（启动提示、退出码、超时杀进程等元信息）
# - repair_attempt:S9 第 83-84 天：自修复尝试事件（Builder 面板时间线数据源）
#                    content 为摘要字符串，extra 携带完整结构化 RepairAttempt payload
# - timestamp:     ISO 8601 UTC 时间戳，便于插件端按时间排序

StreamMessageType = Literal["stdout", "stderr", "system", "repair_attempt", "test_run"]


class StreamMessage(BaseModel):
    """终端日志流式消息（通过 WebSocket 推送给插件端）"""

    type: StreamMessageType = Field(..., description="消息类型：stdout/stderr/system/repair_attempt")
    content: str = Field(..., description="输出内容或事件摘要（文本）")
    timestamp: str = Field(..., description="ISO 8601 UTC 时间戳")
    # S9 第 83-84 天：复杂事件的结构化 payload（可选）
    # repair_attempt 事件使用该字段携带完整 RepairAttempt.to_sse_dict() 数据
    extra: Optional[Dict[str, Any]] = Field(
        default=None,
        description="结构化 payload（repair_attempt 等复杂事件使用）",
    )


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
