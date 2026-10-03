"""
S7 第 63-64 天：Planner 任务规划器（Prompt 工程核心）

将用户的自然语言开发需求拆解为结构化的 DAG 任务图（List[TaskStep]）。

核心能力：
1. 强制结构化输出：System Prompt + response_format=json_object 双保险，
   让模型返回严格 JSON 对象 {"plan": [step, ...]}。
2. DAG 强校验：依赖 ID 存在性 + 无循环依赖（复用 AgentStateMachine.validate_plan）。
   对应 S7 风险预警："大模型经常在 dependencies 里填错 ID，后端必须做强校验"。
3. 步骤数量校验：5-10 步（S7 风险预警：Planner 输出不稳定，步骤数波动大）。
4. 解析/校验失败自动重试一次（共最多 PLANNER_MAX_RETRIES 次尝试）。
5. Prompt 记录持久化：每次 Planner 的输入输出写入本地 JSON 文件，
   供后续 P4 阶段微调模型使用。
"""

import json
import logging
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import settings
from app.models.agent import AgentSession, TaskStep
from app.services.agent_state_machine import (
    AgentStateMachine,
    InvalidPlanError,
)
from app.services.llm import AdapterError, chat_completion_text

logger = logging.getLogger(__name__)


class PlannerError(Exception):
    """Planner 生成任务计划失败（重试耗尽或模型不可用）"""
    pass


# ============================================================
# System Prompt：强制结构化输出
# ============================================================
#
# 设计要点（对应 S7 第 63-64 天验收标准 + 风险预警）：
#   - 明确输出为 JSON 对象 {"plan": [...]}（json_object 模式要求顶层为对象，不能是数组）。
#   - 给死步骤数范围 5-10 步。
#   - "从零生成项目"第一步必须是初始化项目结构。
#   - 前后端任务拆到不同步骤。
#   - dependencies 只能引用已定义的步骤 id，不能循环依赖。
#   - 只返回 JSON，不要解释或 markdown 标记。
_PLANNER_SYSTEM_PROMPT = """你是一个顶级的全栈项目架构师。用户会给你一个开发需求，你需要将其拆解为具体的、可执行的开发步骤。

**输出要求**：
你必须返回一个严格的 JSON 对象，格式如下，不要输出任何解释、问候语或 markdown 代码围栏标记：
{
  "plan": [
    {
      "id": "step_1",
      "description": "用 10 个字以内概括这一步",
      "details": "详细说明这一步要做什么，包括具体技术选型",
      "dependencies": [],
      "suggested_tool": "write_file"
    }
  ]
}

字段说明：
- id: 步骤唯一标识，建议用 step_1, step_2, ... 格式。
- description: 步骤简短描述，10 个字以内。
- details: 详细说明这一步要做什么，包括具体技术选型和命令。
- dependencies: 依赖的前置步骤 ID 列表，无依赖则为空数组 []。只能引用前面已定义的步骤 id。
- suggested_tool: 建议使用的工具，必须是以下之一："write_file" | "run_command" | "search_code" | "ask_user"。

**拆解原则**：
1. 步骤数严格控制在 5 到 10 步之间，不能多也不能少。
2. 对于"从零生成项目"类型的需求，第一步必须是"初始化项目结构"（如创建目录、git init、初始化 package.json 等）。
3. 前端和后端任务必须拆分到不同步骤，不要混在同一个步骤里。
4. dependencies 只能引用本计划中已定义的步骤 id，不能引用不存在的 id，也不能形成循环依赖。
5. 步骤之间按执行顺序排列，有依赖关系的步骤排在被依赖步骤之后。

**示例**：
用户需求："做一个 Todo 全栈应用"
输出：
{
  "plan": [
    {"id":"step_1","description":"初始化项目结构","details":"创建 todo-app 目录，初始化前后端分离结构，执行 git init 和 npm init","dependencies":[],"suggested_tool":"run_command"},
    {"id":"step_2","description":"设计数据库表","details":"设计 Todo 表的字段 (id, title, completed, created_at)，选择 SQLite 作为开发数据库","dependencies":["step_1"],"suggested_tool":"write_file"},
    {"id":"step_3","description":"实现后端 API","details":"使用 Express.js 实现 Todo 的增删改查 RESTful 接口，连接 SQLite 数据库","dependencies":["step_2"],"suggested_tool":"write_file"},
    {"id":"step_4","description":"搭建前端页面","details":"使用 React + Vite 搭建前端，实现 Todo 列表展示、新增、删除、勾选完成功能","dependencies":["step_1"],"suggested_tool":"write_file"},
    {"id":"step_5","description":"联调与测试","details":"前后端联调，验证 Todo 的增删改查功能完整可用","dependencies":["step_3","step_4"],"suggested_tool":"run_command"}
  ]
}
"""

# 重试时追加给模型的强化指令
_RETRY_INSTRUCTION_JSON = (
    "你上一次的输出不是合法的 JSON 对象。请严格只返回一个合法的 JSON 对象，"
    "不要任何解释、问候语或 markdown 代码围栏标记。JSON 必须包含 plan 数组，"
    "每个元素含 id、description、details、dependencies、suggested_tool 字段。"
)


# ============================================================
# 内部辅助函数
# ============================================================

def _build_planner_messages(goal: str, workspace_root: str = "") -> List[Dict[str, str]]:
    """构建 Planner 的 messages 列表。"""
    user_content = f"用户需求：{goal}"
    if workspace_root:
        user_content += f"\n工作区根路径：{workspace_root}"
    return [
        {"role": "system", "content": _PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    从模型输出文本中提取 JSON 对象。

    应对 S7 风险预警：模型即使加了 response_format=json_object，
    有时仍会在 JSON 前后加 ```json 标记或解释文字。
    用正则提取第一个 { 到最后一个 } 之间的内容并解析。
    """
    if not text:
        return None
    # 去除 markdown 代码围栏
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    # 找到第一个 { 和最后一个 }
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    json_str = text[start:end + 1]
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"[Planner] JSON 解析失败: {e}")
        return None


def _convert_to_task_steps(raw_plan: Any) -> List[TaskStep]:
    """
    将模型返回的 raw plan（list of dict）转换为 TaskStep 列表。

    - 缺失 id 时自动生成 uuid。
    - 缺失 details 时用 description 兜底。
    - dependencies 非 list 时重置为空 list。
    - suggested_tool 映射到 action 字段（ReAct 循环使用 action）。
    """
    if not isinstance(raw_plan, list):
        raise ValueError("plan 字段必须是数组")

    steps: List[TaskStep] = []
    for idx, item in enumerate(raw_plan):
        if not isinstance(item, dict):
            raise ValueError(f"第 {idx} 个步骤不是对象")

        step_id = item.get("id") or f"step_{idx + 1}"
        description = str(item.get("description", "")).strip()
        if not description:
            raise ValueError(f"步骤 {step_id} 缺少 description")

        details = str(item.get("details", "")).strip() or description

        deps = item.get("dependencies", [])
        if not isinstance(deps, list):
            deps = []
        dependencies = [str(d) for d in deps]

        suggested_tool = item.get("suggested_tool")
        # 校验 suggested_tool 合法性
        valid_tools = {"write_file", "run_command", "search_code", "ask_user"}
        if suggested_tool is not None and suggested_tool not in valid_tools:
            logger.warning(
                f"[Planner] 步骤 {step_id} 的 suggested_tool='{suggested_tool}' "
                f"不在合法集合中，置空"
            )
            suggested_tool = None

        steps.append(
            TaskStep(
                id=str(step_id),
                description=description,
                details=details,
                dependencies=dependencies,
                action=str(suggested_tool) if suggested_tool else "",
                suggested_tool=suggested_tool,
            )
        )
    return steps


def _save_planner_log(
    goal: str,
    messages: List[Dict[str, str]],
    raw_output: str,
    steps: List[TaskStep],
    model: str,
    attempt: int,
    success: bool,
    error: Optional[str] = None,
) -> None:
    """
    保存 Planner 的输入输出记录到本地 JSON 文件。

    对应 S7 第 63-64 天任务："保留用户 Prompt 原始记录，供后续 P4 阶段微调模型使用"。
    记录文件命名：{timestamp}_{uuid8}.json，存于 settings.PLANNER_LOG_DIR。
    """
    try:
        log_dir = Path(settings.PLANNER_LOG_DIR)
        log_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{ts}_{uuid.uuid4().hex[:8]}.json"

        record = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "goal": goal,
            "attempt": attempt,
            "success": success,
            "error": error,
            "messages": messages,
            "raw_output": raw_output,
            "parsed_steps": [s.model_dump() for s in steps] if success else None,
        }
        log_file.write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.debug(f"[Planner] Prompt 记录已保存: {log_file}")
    except Exception as e:
        # 记录保存失败不应影响主流程
        logger.warning(f"[Planner] Prompt 记录保存失败: {e}")


# ============================================================
# 公开接口
# ============================================================

async def plan(
    goal: str,
    workspace_root: str = "",
    model: str = "glm-4.5-air",
) -> List[TaskStep]:
    """
    调用 Planner 将用户需求拆解为任务步骤列表。

    Args:
        goal:           用户原始开发需求
        workspace_root: 工作区根路径（可选，注入 Prompt 供模型参考）
        model:          使用的模型 ID

    Returns:
        校验通过的 TaskStep 列表（DAG，已通过依赖合法性校验）

    Raises:
        PlannerError: 重试耗尽后仍无法生成合法计划
    """
    messages = _build_planner_messages(goal, workspace_root)
    last_error: Optional[str] = None

    for attempt in range(1, settings.PLANNER_MAX_RETRIES + 1):
        try:
            logger.info(
                f"[Planner] 开始生成任务计划: attempt={attempt}/{settings.PLANNER_MAX_RETRIES}, "
                f"model={model}, goal={goal[:50]}"
            )

            # 调用模型（json_object 模式强制结构化输出）
            raw_output = await chat_completion_text(
                messages=messages,
                model=model,
                temperature=settings.PLANNER_TEMPERATURE,
                timeout=settings.PLANNER_TIMEOUT_SECONDS,
                max_tokens=settings.PLANNER_MAX_TOKENS,
                response_format={"type": "json_object"},
            )

            # 解析 JSON
            parsed = _extract_json_object(raw_output)
            if parsed is None:
                last_error = "模型输出不是合法 JSON"
                logger.warning(f"[Planner] JSON 解析失败，准备重试: attempt={attempt}")
                messages.append({"role": "user", "content": _RETRY_INSTRUCTION_JSON})
                continue

            raw_plan = parsed.get("plan")
            if raw_plan is None:
                # 兼容模型直接返回数组的情况
                if isinstance(parsed, list):
                    raw_plan = parsed
                else:
                    last_error = "JSON 中缺少 plan 字段"
                    logger.warning(f"[Planner] 缺少 plan 字段，准备重试: attempt={attempt}")
                    messages.append({"role": "user", "content": _RETRY_INSTRUCTION_JSON})
                    continue

            # 转换为 TaskStep
            try:
                steps = _convert_to_task_steps(raw_plan)
            except ValueError as e:
                last_error = f"步骤字段不合法: {e}"
                logger.warning(f"[Planner] {last_error}，准备重试: attempt={attempt}")
                messages.append({
                    "role": "user",
                    "content": f"你上一次的输出步骤字段不合法：{e}。请重新生成合法的 JSON。",
                })
                continue

            # 步骤数量校验（S7 风险预警：Planner 输出不稳定）
            if len(steps) < settings.PLANNER_MIN_STEPS or len(steps) > settings.PLANNER_MAX_STEPS:
                last_error = (
                    f"步骤数量 {len(steps)} 超出范围 "
                    f"[{settings.PLANNER_MIN_STEPS}, {settings.PLANNER_MAX_STEPS}]"
                )
                logger.warning(f"[Planner] {last_error}，准备重试: attempt={attempt}")
                messages.append({
                    "role": "user",
                    "content": (
                        f"你上一次返回了 {len(steps)} 个步骤，请严格控制在 "
                        f"{settings.PLANNER_MIN_STEPS}-{settings.PLANNER_MAX_STEPS} 步之间，"
                        f"不要多也不要少。"
                    ),
                })
                continue

            # DAG 合法性校验（依赖 ID 存在性 + 无循环依赖）
            session = AgentSession(user_goal=goal, plan=steps)
            sm = AgentStateMachine(session)
            try:
                sm.validate_plan()
            except InvalidPlanError as e:
                last_error = f"DAG 校验失败: {e}"
                logger.warning(f"[Planner] {last_error}，准备重试: attempt={attempt}")
                messages.append({
                    "role": "user",
                    "content": (
                        f"你上一次的任务图不合法：{e}。请重新生成，确保 dependencies 中的 "
                        f"每个 ID 都真实存在，且不存在循环依赖。"
                    ),
                })
                continue

            # 所有校验通过
            _save_planner_log(
                goal=goal,
                messages=messages,
                raw_output=raw_output,
                steps=steps,
                model=model,
                attempt=attempt,
                success=True,
            )
            logger.info(
                f"[Planner] 任务计划生成成功: attempt={attempt}, "
                f"steps={len(steps)}"
            )
            return steps

        except AdapterError as e:
            # 模型调用异常（限流/超时/网络等），不重试模型本身的错误
            last_error = f"模型调用失败: {e}"
            logger.error(f"[Planner] {last_error}")
            break

    # 所有尝试均失败
    _save_planner_log(
        goal=goal,
        messages=messages,
        raw_output="",
        steps=[],
        model=model,
        attempt=settings.PLANNER_MAX_RETRIES,
        success=False,
        error=last_error,
    )
    raise PlannerError(f"Planner 生成任务计划失败: {last_error}")
