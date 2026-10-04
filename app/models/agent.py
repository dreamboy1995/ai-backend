"""
S7 第 61-62 天：Agent 核心数据结构定义

定义 Agent 状态机所需的 Pydantic 模型，以及 S7 新增接口的请求/响应 DTO。

关键设计点（兼顾 S7 风险预警）：
1. TaskStep.status 使用 Literal 限定合法状态，避免非法状态写入。
2. AgentSession 持久化到 Redis（见 agent_session_store.py），应对
   "用户中途关闭 VS Code" 导致会话丢失的风险。
3. 预留 is_paused / interrupt_flag / pending_question 字段，为
   第 69-70 天的暂停/人工介入机制铺路，当前阶段状态机已能识别这些字段。
4. ToolExecutor 抽象基类在 app/services/tool_executor.py 中定义，
   采用依赖注入（DI）：S7 注入 MockToolExecutor，S8 替换为 MCPToolExecutor，
   上层循环代码无需修改。
"""

import time
import uuid
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ============================================================
# Agent 核心状态数据结构
# ============================================================

StepStatus = Literal["pending", "running", "done", "failed", "blocked"]
SUGGESTED_TOOLS = Literal["write_file", "run_command", "search_code", "ask_user"]

# ============================================================
# 会话结束原因常量
# ============================================================
# idle: 刚创建，尚未开始执行
# paused: 用户主动暂停（可恢复）
# ask_user: 等待用户回答问题（可恢复）
# confirming: 等待用户确认工具执行（可恢复，S8 确认链路）
# completed: 所有步骤正常完成
# max_iter: 达到最大迭代次数（终态，不可恢复）
# timeout: 执行超时（终态，不可恢复）
# error: 执行异常（终态，不可恢复）
END_REASON_IDLE = "idle"
END_REASON_PAUSED = "paused"
END_REASON_ASK_USER = "ask_user"
END_REASON_CONFIRMING = "confirming"
END_REASON_COMPLETED = "completed"
END_REASON_MAX_ITER = "max_iter"
END_REASON_TIMEOUT = "timeout"
END_REASON_ERROR = "error"

# 终态原因：会话已死亡，不允许再次 start/resume/pause
TERMINAL_END_REASONS = {END_REASON_COMPLETED, END_REASON_MAX_ITER, END_REASON_TIMEOUT, END_REASON_ERROR}
# 可恢复原因：暂停/等待用户/等待确认，允许 resume
RESUMABLE_END_REASONS = {END_REASON_PAUSED, END_REASON_ASK_USER, END_REASON_CONFIRMING}


class TaskStep(BaseModel):
    """
    Agent 任务图中的单个步骤节点。

    每个步骤对应一个可执行动作（工具调用），通过 dependencies 形成 DAG。
    状态机（agent_state_machine.py）负责驱动步骤在 pending -> running -> done/failed 之间流转。
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="步骤唯一 ID（uuid）")
    description: str = Field(..., min_length=1, description="步骤简短描述（10 字以内）")
    details: str = Field(default="", description="步骤详细说明，包括具体技术选型")
    status: StepStatus = Field(default="pending", description="步骤状态")
    dependencies: List[str] = Field(default_factory=list, description="依赖的前置步骤 ID 列表")
    action: str = Field(default="", description="工具名称，如 write_file / run_command")
    action_input: Dict[str, Any] = Field(default_factory=dict, description="工具参数")
    observation: Optional[str] = Field(default=None, description="执行结果反馈（Observe 阶段写入）")
    retry_count: int = Field(default=0, ge=0, description="已重试次数")
    # 建议使用的工具（Planner 输出字段，状态机不直接使用，仅供 UI 展示与调试）
    suggested_tool: Optional[SUGGESTED_TOOLS] = Field(default=None, description="Planner 建议的工具")

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "step_1",
                "description": "初始化项目结构",
                "details": "创建前后端分离目录，初始化 package.json",
                "status": "pending",
                "dependencies": [],
                "action": "run_command",
                "action_input": {"cmd": "mkdir backend frontend && cd backend && npm init -y"},
                "observation": None,
                "retry_count": 0,
                "suggested_tool": "run_command",
            }
        }
    }


class AgentSession(BaseModel):
    """
    Agent 会话：承载一次完整的"需求拆解 -> 分步执行"流程。

    持久化到 Redis（JSON），Redis 不可用时降级为内存存储。
    用户中途关闭 VS Code 后，下次打开插件可从 Redis 恢复未完成的会话。
    """

    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="会话唯一 ID")
    user_goal: str = Field(..., min_length=1, description="原始用户需求")
    workspace_root: str = Field(default="", description="工作区根路径")
    model: str = Field(default="glm-4.5-air", description="执行 Agent 使用的模型")
    plan: List[TaskStep] = Field(default_factory=list, description="任务步骤列表（DAG）")
    current_step_index: int = Field(default=-1, description="当前正在执行的步骤索引，-1 表示未开始")
    final_answer: Optional[str] = Field(default=None, description="所有步骤完成后的最终总结")
    # 执行控制字段（第 69-70 天暂停/人工介入机制使用，状态机已识别）
    is_paused: bool = Field(default=False, description="是否已暂停")
    is_executing: bool = Field(default=False, description="Agent 循环是否正在执行")
    interrupt_flag: bool = Field(default=False, description="中断标志，循环顶部检查")
    pending_question: Optional[str] = Field(default=None, description="待用户回答的问题（ask_user 工具）")
    # S8 确认链路：write_file / run_command 等工具需要用户确认后再真正执行
    # 当 react_loop 检测到 ToolResult.requires_confirmation=True 时，
    # 暂停循环并填充以下字段，等待前端 Builder 面板弹确认浮层后回传 confirmation_id。
    pending_confirmation_id: Optional[str] = Field(
        default=None, description="工具确认凭证 ID（一次性，由 tool_registry 生成）"
    )
    pending_confirmation_prompt: Optional[str] = Field(
        default=None, description="确认浮层展示给用户的提示文本"
    )
    pending_confirmation_preview: Optional[str] = Field(
        default=None, description="确认前的预览内容（Diff / 命令 / 文件列表）"
    )
    pending_confirmation_tool: Optional[str] = Field(
        default=None, description="待确认的工具名（write_file / run_command / git_commit）"
    )
    # 会话结束原因与描述（持久化到 Redis，供前端判断会话是否已死亡）
    # 取值见 END_REASON_* 常量；None 表示运行中或未开始
    end_reason: Optional[str] = Field(default=None, description="会话结束原因")
    end_message: Optional[str] = Field(default=None, description="会话结束的详细描述")
    # 时间戳
    created_at: float = Field(default_factory=time.time, description="创建时间戳")
    updated_at: float = Field(default_factory=time.time, description="最后更新时间戳")

    @property
    def total_steps(self) -> int:
        """总步骤数"""
        return len(self.plan)

    @property
    def done_steps(self) -> int:
        """已完成步骤数"""
        return sum(1 for s in self.plan if s.status == "done")

    @property
    def progress(self) -> float:
        """整体进度（0.0 ~ 1.0）"""
        if self.total_steps == 0:
            return 0.0
        return self.done_steps / self.total_steps


# ============================================================
# S7 新增接口请求/响应 DTO
# ============================================================

class PlanRequest(BaseModel):
    """
    POST /v1/agent/plan 请求体。

    接收用户需求，由 Planner 拆解为任务图并创建 AgentSession。
    S7 第 63-64 天实现 Planner；第 61-62 天接口已就位，
    当未传入 plan 时后端会生成默认占位计划用于联调。
    """

    goal: str = Field(..., min_length=1, description="用户原始开发需求")
    workspace_root: str = Field(default="", description="工作区根路径")
    model: str = Field(default="glm-4.5-air", description="使用的模型")
    # 允许调用方直接传入预拆解的步骤（测试/联调用）；
    # 正常流程由 Planner 生成，此字段为空。
    plan: Optional[List[TaskStep]] = Field(default=None, description="预拆解的步骤列表（可选）")


class PlanPreviewItem(BaseModel):
    """计划预览项（仅返回 id + description，供前端快速展示）"""

    id: str
    description: str


class PlanResponse(BaseModel):
    """POST /v1/agent/plan 响应体"""

    session_id: str
    plan_preview: List[PlanPreviewItem]


class AgentStatusResponse(BaseModel):
    """
    GET /v1/agent/status/{session_id} 响应体。

    返回 AgentSession 完整状态，供 Builder 面板每秒轮询刷新。
    包含每一步的状态、观察结果，以及整体进度。

    S8 确认链路新增字段：
      - pending_confirmation_id / _prompt / _preview / _tool
        当会话处于「等待用户确认工具执行」状态时非空，
        Builder 面板据此弹出「确认写入」浮层，用户确认后
        调用 /v1/agent/confirm 恢复执行。

    进度字段：
      - progress: 0.0 ~ 1.0 的小数进度
      - progress_percent: 0 ~ 100 的整数进度（Builder 面板直接使用，
        对应 S7 接口规范 BuilderState.progress: number // 0-100）
    """

    session_id: str
    user_goal: str
    workspace_root: str
    steps: List[TaskStep]
    current_step_index: int
    final_answer: Optional[str]
    progress: float
    progress_percent: int = Field(description="整体进度百分比 0-100，供 Builder 面板直接渲染进度条")
    is_executing: bool
    is_paused: bool
    pending_question: Optional[str]
    total_steps: int
    done_steps: int
    end_reason: Optional[str] = Field(default=None, description="会话结束原因，None 表示运行中或未开始")
    end_message: Optional[str] = Field(default=None, description="会话结束的详细描述")
    # S8 确认链路：供前端 Builder 面板弹「确认写入」浮层
    pending_confirmation_id: Optional[str] = Field(default=None, description="工具确认凭证 ID")
    pending_confirmation_prompt: Optional[str] = Field(default=None, description="确认浮层提示文本")
    pending_confirmation_preview: Optional[str] = Field(default=None, description="确认前预览（Diff/命令等）")
    pending_confirmation_tool: Optional[str] = Field(default=None, description="待确认工具名")


class ToolConfirmRequest(BaseModel):
    """POST /v1/agent/confirm 请求体（Builder 面板弹确认浮层后回传）"""

    session_id: str = Field(..., min_length=1)
    confirmation_id: str = Field(..., min_length=1, description="从 status 接口拿到的 pending_confirmation_id")
    action: Literal["allow", "deny"] = Field(..., description="allow=允许执行，deny=拒绝执行")


class ToolConfirmResponse(BaseModel):
    """POST /v1/agent/confirm 响应体"""

    success: bool
    message: str
    # 可选：执行结果摘要（allow 时填充，便于前端展示执行结果）
    result_summary: Optional[str] = Field(default=None, description="执行结果文本摘要")


class AskUserRespondRequest(BaseModel):
    """POST /v1/agent/ask/respond 请求体（人工介入回复）"""

    session_id: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1, description="用户对 ask_user 问题的回答")


class AskUserRespondResponse(BaseModel):
    """POST /v1/agent/ask/respond 响应体"""

    success: bool
    message: str


class AgentActionResponse(BaseModel):
    """start / pause / resume 等控制类接口的通用响应"""

    success: bool
    message: str
