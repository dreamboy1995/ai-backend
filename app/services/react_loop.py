"""
S7 第 67-68 天：ReAct 执行循环骨架

这是 Agent 的"心脏"——它不停地在 思考（Reason）→ 行动（Act）→ 观察（Observe） 之间循环。

核心能力：
1. run_agent(session_id)：ReAct 主循环，依次执行 DAG 中的可执行步骤。
   - Reason：根据用户目标 + 当前步骤 + 上下文，让 LLM 决定调用哪个工具及参数。
   - Act：调用 ToolExecutor.execute(tool, params) 执行工具（S7 为 Mock，S8 换 MCP）。
   - Observe：将工具返回结果作为 observation 写入步骤，标记 done。
2. ask_user 人工介入：当 Reason 返回 {"tool": "ask_user", "params": {"question": "..."}} 时，
   循环暂停，将问题推送给 Builder 面板，等待用户回复后恢复。
3. 退出条件（对应 S7 风险预警，防止死循环）：
   - 最大迭代次数 REACT_MAX_ITERATIONS（默认 15）
   - 总超时 REACT_TOTAL_TIMEOUT_SECONDS（默认 120s）
   - 中断标志 interrupt_flag（pause / ask_user 设置）
   - 无可用步骤（全部 done 或存在 failed/blocked）
4. 上下文裁剪（对应 S7 风险预警"上下文爆炸"）：
   - 只保留最近 REACT_CONTEXT_WINDOW（默认 3）个已完成步骤的完整 observation，
     更早的步骤只保留摘要（description + status），避免 Prompt 无限膨胀。
5. 依赖注入：通过 tool_executor 参数注入 ToolExecutor 实例，S7 用 MockToolExecutor，
   S8 替换为 MCPToolExecutor 时上层循环代码无需修改。

设计参考：app/services/planner.py 的 JSON 解析 + 重试模式，
以及 app/lifespan.py 的 asyncio.create_task 后台任务模式。
"""

import asyncio
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

from app.config import settings
from app.models.agent import (
    AgentSession,
    END_REASON_ASK_USER,
    END_REASON_COMPLETED,
    END_REASON_CONFIRMING,
    END_REASON_ERROR,
    END_REASON_FUSED,
    END_REASON_MAX_ITER,
    END_REASON_PAUSED,
    END_REASON_TIMEOUT,
    TaskStep,
    TERMINAL_END_REASONS,
)
from app.models.tool import ToolResult
from app.services.agent_session_store import get_agent_session_store
from app.services.agent_state_machine import AgentStateMachine
from app.services.llm import AdapterError, chat_completion_text
from app.services.tool_executor import ToolExecutionError, ToolExecutor, get_default_tool_executor

logger = logging.getLogger(__name__)


# ============================================================
# 后台任务注册表
# ============================================================
# 记录正在运行的 Agent 循环任务：session_id -> asyncio.Task
# 用于避免同一会话重复启动循环，以及在 pause 时识别运行中的任务。
_running_agents: Dict[str, "asyncio.Task[None]"] = {}
_running_agents_lock = asyncio.Lock()


def is_agent_running(session_id: str) -> bool:
    """判断指定会话的 Agent 循环是否正在运行"""
    task = _running_agents.get(session_id)
    return task is not None and not task.done()


# ============================================================
# Reason 阶段 Prompt 构建
# ============================================================

_REASON_SYSTEM_PROMPT_TEMPLATE = """你是一个专业的开发 Agent，正在执行一个项目开发任务。

**用户需求**：
{user_goal}

**当前需要执行的步骤**：
- 步骤 ID：{step_id}
- 描述：{step_description}
- 详细说明：{step_details}
- 建议工具：{suggested_tool}

**已完成步骤的上下文（摘要）**：
{context}

请决定这一步要调用的工具及其参数。你必须返回**且只返回一个**严格的 JSON 对象，不要任何解释、问候语或 markdown 代码围栏标记：
{{
  "tool": "工具名称",
  "params": {{...}}
}}

**可用工具**：
- write_file：写文件，参数 {{ "path": "文件路径", "content": "文件内容" }}
- run_command：运行命令，参数 {{ "cmd": "要执行的命令" }}
- search_code：搜索代码，参数 {{ "query": "搜索关键词" }}
- ask_user：向用户提问（遇到关键决策时使用），参数 {{ "question": "要问用户的问题" }}

**强制规则（必须遵守）**：
1. 如果「建议工具」是 ask_user，**必须**返回 ask_user，把步骤描述中的决策点整理成一个清晰的问题向用户提问，**不得自行替用户做决定**。
2. 即使「建议工具」不是 ask_user，但只要本步骤涉及技术选型 / 架构决策 / 外部依赖选择 / 配置参数，且存在两个及以上合理方案，也**必须**返回 ask_user。
3. 只有在明确无选型空间的纯执行步骤，才返回 write_file / run_command / search_code。
4. ask_user 的 question 应当给出可选方案供用户选择（例如："使用 SQLite 还是 PostgreSQL？"）。
5. **每次只能返回一个 JSON 对象**。即使一个步骤需要写多个文件或执行多条命令，也必须选择最合适的那一个先返回；后续操作会在下一个步骤中继续。**严禁输出两个或多个连续的 JSON 对象**。

**示例**：
- 步骤"选择数据库方案"，建议工具 ask_user → 返回 {{"tool": "ask_user", "params": {{"question": "使用 SQLite 还是 PostgreSQL？"}}}}
- 步骤"实现后端 API"，建议工具 write_file，无选型 → 返回 {{"tool": "write_file", "params": {{"path": "...", "content": "..."}}}}

只返回一个 JSON 对象本身。
"""


def _build_context_summary(sm: AgentStateMachine, window: int) -> str:
    """
    构建已完成步骤的上下文摘要（应对上下文爆炸）。

    策略（对应 S7 风险预警）：
      - 最近 window 个 done 步骤：保留完整 observation
      - 更早的 done 步骤：只保留 description + status（省略详细结果）

    Args:
        sm:     状态机实例
        window: 保留完整 observation 的最近步骤数

    Returns:
        格式化的上下文字符串
    """
    done_steps = [s for s in sm.session.plan if s.status == "done"]
    if not done_steps:
        return "（暂无已完成步骤）"

    if window <= 0:
        recent: List[TaskStep] = []
        older = done_steps
    else:
        recent = done_steps[-window:]
        older = done_steps[:-window]

    lines: List[str] = []
    if older:
        lines.append(f"（更早的 {len(older)} 个步骤已省略详细结果）")
        for s in older:
            lines.append(f"- {s.id}: {s.description} [{s.status}]")
    for s in recent:
        lines.append(f"- {s.id}: {s.description} [{s.status}]")
        if s.observation:
            # 截断过长的 observation，避免单条结果撑爆 Prompt
            obs = s.observation
            if len(obs) > 500:
                obs = obs[:500] + "...(截断)"
            lines.append(f"  结果: {obs}")
    return "\n".join(lines)


def _build_reason_messages(
    session: AgentSession, step: TaskStep, sm: AgentStateMachine
) -> List[Dict[str, str]]:
    """
    构建 Reason 阶段的 messages 列表。

    将用户目标、当前步骤详情、上下文摘要注入 System Prompt，
    引导 LLM 返回结构化的工具调用决策。
    """
    context = _build_context_summary(sm, window=settings.REACT_CONTEXT_WINDOW)
    suggested = step.suggested_tool or step.action or "未指定"

    system_prompt = _REASON_SYSTEM_PROMPT_TEMPLATE.format(
        user_goal=session.user_goal,
        step_id=step.id,
        step_description=step.description,
        step_details=step.details or step.description,
        suggested_tool=suggested,
        context=context,
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"请执行步骤 {step.id}：{step.description}"},
    ]


def _find_first_json_object(text: str) -> Optional[str]:
    """
    使用括号深度跟踪，从文本中定位第一个完整的 JSON 对象（{ ... }）。

    处理 LLM 返回多个连续 JSON 对象的场景（如同时想写两个文件时输出
    {json1}\\n\\n{json2}），避免 rfind('}') 把所有对象都吞进去导致
    json.loads 报 "Extra data" 错误。

    Returns:
        第一个完整 JSON 对象的子串（不含周围文本），找不到则返回 None。
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]

        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return None  # 括号未闭合


def _extract_action_json(text: str) -> Optional[Dict[str, Any]]:
    """
    从 Reason 阶段模型输出中提取 action JSON。

    应对 S7 风险预警：模型即使加了 response_format=json_object，
    有时仍会在 JSON 前后加 ```json 标记或解释文字，或一次返回
    多个连续 JSON 对象。

    额外的鲁棒性处理：
      - 去除 markdown 代码围栏
      - 只取第一个完整 JSON 对象（括号深度跟踪），忽略后续多余对象
      - 移除对象/数组的尾随逗号（模型常见错误）
      - 清理不可见控制字符（保留 \\n \\r \\t）

    返回 {"tool": "...", "params": {...}} 或 None（解析失败）。
    """
    if not text:
        return None

    # 去除 markdown 代码围栏
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip()

    # 用括号深度跟踪定位第一个完整 JSON 对象
    json_str = _find_first_json_object(text)
    if json_str is None:
        logger.warning("[ReAct] Reason 输出中未找到合法 JSON 对象")
        return None

    # 规范化 1：移除尾随逗号（,} 或 ,]），模型常见输出错误
    json_str = re.sub(r",(\s*[}\]])", r"\1", json_str)

    # 规范化 2：清理不可见控制字符（保留 \n \r \t）
    json_str = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", json_str)

    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError as e:
        # 展示错误位置附近的上下文，便于定位（而非只截取前 200 字符）
        pos = e.pos
        ctx_start = max(0, pos - 80)
        ctx_end = min(len(json_str), pos + 80)
        context = json_str[ctx_start:ctx_end]
        # 用 repr 展示，避免换行/控制字符干扰日志格式
        logger.warning(
            f"[ReAct] Reason JSON 解析失败: {e}, "
            f"pos={pos}, context={context!r}"
        )
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


# Reason 重试时追加给模型的纠偏指令
_REASON_RETRY_INSTRUCTION = (
    "你上一次的输出不是合法的 JSON 对象。请严格只返回一个合法的 JSON 对象，"
    "不要任何解释、问候语或 markdown 代码围栏标记。"
    "如果 content 字段包含代码，其中的双引号必须转义为 \\\"，换行必须转义为 \\n。"
    "JSON 格式：{\"tool\": \"工具名称\", \"params\": {...}}"
    "重要：一次只能输出一个 JSON 对象，严禁输出两个或多个连续的 JSON 对象。"
    "如果有多个操作要做，选一个最重要的先返回，后续操作会在下一步继续。"
)


async def _reason(
    session: AgentSession, step: TaskStep, sm: AgentStateMachine
) -> Optional[Dict[str, Any]]:
    """
    Reason（思考）阶段：让 LLM 决定当前步骤要调用的工具及参数。

    支持重试：当模型输出解析失败时，追加纠偏指令后重试，
    最多 REACT_MAX_RETRIES 次（含首次）。

    Returns:
        {"tool": "...", "params": {...}} 或 None（模型调用失败 / 重试耗尽仍解析失败）
    """
    messages = _build_reason_messages(session, step, sm)

    for attempt in range(1, settings.REACT_MAX_RETRIES + 1):
        try:
            raw_output = await chat_completion_text(
                messages=messages,
                model=session.model,
                temperature=settings.REACT_TEMPERATURE,
                timeout=settings.REACT_STEP_TIMEOUT_SECONDS,
                max_tokens=settings.REACT_MAX_TOKENS,
                response_format={"type": "json_object"},
            )
        except AdapterError as e:
            logger.error(f"[ReAct] Reason 模型调用失败 step={step.id}: {e}")
            return None
        except Exception as e:
            logger.error(f"[ReAct] Reason 异常 step={step.id}: {e}", exc_info=True)
            return None

        action = _extract_action_json(raw_output)
        if action is not None:
            if attempt > 1:
                logger.info(
                    f"[ReAct] Reason 重试成功 step={step.id}, attempt={attempt}"
                )
            return action

        logger.warning(
            f"[ReAct] Reason 未返回合法 action step={step.id}, "
            f"attempt={attempt}/{settings.REACT_MAX_RETRIES}"
        )
        if attempt < settings.REACT_MAX_RETRIES:
            messages.append({"role": "user", "content": _REASON_RETRY_INSTRUCTION})

    return None


# ============================================================
# ReAct 主循环
# ============================================================

def _reload_control_flags(session: AgentSession, store) -> None:
    """
    从存储重新加载控制标志（is_paused / interrupt_flag / pending_question），
    不覆盖步骤状态（避免丢失循环内的 mark_running/mark_done 变更）。

    背景：pause 端点会修改会话的 is_paused / interrupt_flag 并保存到 store，
    但循环持有的是本地 session 引用。为了让循环能感知到外部的暂停请求，
    每次迭代顶部需从 store 同步控制标志。
    """
    fresh = store.get(session.session_id)
    if fresh is None:
        return
    session.is_paused = fresh.is_paused
    session.interrupt_flag = fresh.interrupt_flag
    # pending_question 也同步（虽然 ask_user 由循环自身设置，但外部可能清除）
    session.pending_question = fresh.pending_question


def _save_preserving_flags(session: AgentSession, store) -> None:
    """
    保存会话，但先从存储同步控制标志，避免循环的本地 session（控制标志可能已过期）
    覆盖外部（pause 端点）设置的 is_paused / interrupt_flag / pending_question。

    注意：ask_user 场景下循环自身设置了 pending_question/is_paused，
    此时不应调用本方法（应直接 store.save），否则会覆盖循环设置的值。
    """
    _reload_control_flags(session, store)
    store.save(session)


def _mark_session_end(
    session: AgentSession, end_reason: str, end_message: Optional[str] = None
) -> None:
    """
    统一记录会话结束原因与描述。

    所有退出点（暂停/完成/最大迭代/超时/异常，以及 ask_user）都应调用本方法，
    确保 end_reason / end_message 字段被持久化，供前端判断会话是否已死亡。

    Args:
        session:     会话对象
        end_reason:  结束原因（见 END_REASON_* 常量）
        end_message: 结束的详细描述，None 时不覆盖已有值
    """
    session.end_reason = end_reason
    if end_message is not None:
        session.end_message = end_message
    session.updated_at = time.time()


def _fail_remaining_steps(sm: AgentStateMachine, reason: str) -> None:
    """
    将剩余 pending / running 步骤标记为 failed。

    仅在终态退出（completed 存在失败步骤 / max_iter / timeout / error）时调用，
    可恢复退出（paused / ask_user）不调用，以便 resume 后继续执行。
    """
    sm.fail_remaining_steps(reason)


async def run_agent(
    session_id: str,
    tool_executor: Optional[ToolExecutor] = None,
) -> None:
    """
    ReAct 主循环：Reason → Act → Observe。

    流程：
      1. 从存储加载会话，创建状态机。
      2. 循环：
         a. 从存储同步控制标志（pause / ask_user 由外部设置）
         b. 检查中断标志 / 暂停 → break
         c. 检查最大迭代次数 / 总超时 → break
         d. 获取下一个可执行步骤；无则 break
         e. Reason：LLM 决定工具调用
         f. 若 tool == ask_user：设置 pending_question，暂停循环，break
         g. Act：执行工具
         h. Observe：标记步骤 done，写入 observation
         i. 保存会话
      3. 循环结束：根据退出原因更新控制标志，保存会话。

    Args:
        session_id:    会话 ID
        tool_executor: 工具执行器（依赖注入），None 时使用默认 MockToolExecutor
    """
    if tool_executor is None:
        tool_executor = get_default_tool_executor()

    store = get_agent_session_store()
    session = store.get(session_id)
    if session is None:
        logger.error(f"[ReAct] 会话不存在，循环终止: {session_id}")
        return

    sm = AgentStateMachine(session)
    start_time = time.time()
    iteration = 0
    exit_reason = END_REASON_COMPLETED  # 默认 completed
    end_message: Optional[str] = None  # 会话结束描述，对应 end_reason

    # S8 第 79-80 天：同步会话沙箱模式 + 熔断计数重置
    try:
        from app.services.sandbox_orchestrator import get_sandbox_manager
        session.sandbox_mode = get_sandbox_manager().chosen_mode
    except Exception:
        session.sandbox_mode = "host"  # 降级
    # 循环启动时重置连续失败计数（可能是熔断恢复后的新循环）
    session.consecutive_failures = 0
    store.save(session)

    logger.info(
        f"[ReAct] 循环开始: session={session_id}, "
        f"total_steps={session.total_steps}, model={session.model}"
    )

    try:
        while True:
            iteration += 1

            # 同步外部设置的控制标志（pause / ask_user 响应）
            _reload_control_flags(session, store)

            # ---- 退出条件 1：中断标志（pause / ask_user 触发）----
            if session.interrupt_flag or session.is_paused:
                exit_reason = END_REASON_PAUSED
                end_message = "用户暂停了执行"
                logger.info(
                    f"[ReAct] 检测到中断标志，循环暂停: "
                    f"session={session_id}, iteration={iteration}"
                )
                break

            # ---- 退出条件 2：最大迭代次数（防止死循环）----
            if iteration > settings.REACT_MAX_ITERATIONS:
                exit_reason = END_REASON_MAX_ITER
                msg = (
                    f"达到最大迭代次数 {settings.REACT_MAX_ITERATIONS}，Agent 停止执行"
                )
                session.final_answer = msg
                end_message = msg
                logger.warning(f"[ReAct] {msg}: session={session_id}")
                break

            # ---- 退出条件 3：总超时 ----
            elapsed = time.time() - start_time
            if elapsed > settings.REACT_TOTAL_TIMEOUT_SECONDS:
                exit_reason = END_REASON_TIMEOUT
                msg = (
                    f"执行超时（{settings.REACT_TOTAL_TIMEOUT_SECONDS}s），Agent 停止执行"
                )
                session.final_answer = msg
                end_message = msg
                logger.warning(f"[ReAct] {msg}: session={session_id}")
                break

            # ---- 获取下一个可执行步骤 ----
            step = sm.get_next_runnable_step()
            if step is None:
                exit_reason = END_REASON_COMPLETED
                if sm.is_all_done():
                    session.final_answer = "所有任务步骤已完成"
                    end_message = "所有任务步骤已完成"
                    logger.info(f"[ReAct] 所有步骤完成: session={session_id}")
                else:
                    msg = "无可用执行步骤（存在失败或阻塞的步骤）"
                    session.final_answer = msg
                    end_message = msg
                    logger.info(
                        f"[ReAct] 无可用步骤: session={session_id}, "
                        f"summary={sm.summary()}"
                    )
                break

            # 标记步骤为 running（状态机要求 pending -> running -> done/failed）
            sm.mark_running(step.id)
            _save_preserving_flags(session, store)

            # ---- Reason（思考）----
            logger.info(
                f"[ReAct] >> Reason: step={step.id} ({step.description})"
            )
            action = await _reason(session, step, sm)

            # _reason 可能耗时较长（LLM 调用），期间用户可能调用了 pause。
            # 重新同步控制标志，若已请求暂停则回收当前 running 步骤并退出。
            _reload_control_flags(session, store)
            if session.interrupt_flag or session.is_paused:
                sm.reset_step(step.id)  # running -> pending，供 resume 后重试
                exit_reason = END_REASON_PAUSED
                end_message = "用户暂停了执行"
                _save_preserving_flags(session, store)
                logger.info(
                    f"[ReAct] Reason 期间检测到暂停请求，回收步骤 {step.id} 并退出"
                )
                break

            if action is None:
                sm.mark_failed(step.id, "Reason 阶段未返回合法的工具调用")
                _save_preserving_flags(session, store)
                logger.warning(f"[ReAct] 步骤 {step.id} Reason 失败，标记为 failed")
                continue

            tool = str(action.get("tool", "")).strip()
            params = action.get("params", {})
            if not isinstance(params, dict):
                params = {}
            logger.info(
                f"[ReAct] Reason 结果: step={step.id}, tool={tool}, params={params}"
            )

            # ---- 处理 ask_user（人工介入）----
            # 当 Agent 遇到关键决策时，返回 {"tool": "ask_user", "params": {"question": "..."}}
            # 此时循环暂停，将问题推送给 Builder 面板，等待用户输入后再继续。
            if tool == "ask_user":
                exit_reason = END_REASON_ASK_USER
                question = str(params.get("question", "")).strip() or "请提供输入"
                session.pending_question = question
                session.is_executing = False
                session.is_paused = True
                session.interrupt_flag = True
                end_message = f"等待用户回答问题：{question}"
                store.save(session)  # ask_user 由循环自身设置标志，直接保存
                logger.info(
                    f"[ReAct] 步骤 {step.id} 请求人工介入，循环暂停: "
                    f"question={question}"
                )
                break  # 暂停循环，等待 /ask/respond 恢复

            # ---- Act（行动）----
            logger.info(f"[ReAct] >> Act: step={step.id}, tool={tool}")
            try:
                # 将 workspace_root / session_id 传入 execute，
                # 供 MCPToolExecutor 解析相对路径、写入审计日志
                result: ToolResult = await tool_executor.execute(
                    tool,
                    params,
                    workspace_root=session.workspace_root,
                    session_id=session.session_id,
                )
            except ToolExecutionError as e:
                logger.error(f"[ReAct] 工具执行失败 step={step.id}: {e}")
                sm.mark_failed(step.id, f"工具执行失败: {e}")
                _save_preserving_flags(session, store)
                continue
            except Exception as e:
                logger.error(
                    f"[ReAct] 工具执行异常 step={step.id}: {e}", exc_info=True
                )
                sm.mark_failed(step.id, f"工具执行异常: {e}")
                _save_preserving_flags(session, store)
                continue

            # ---- 确认检测（S8 确认链路）----
            # 当 ToolResult.requires_confirmation=True 时，
            # 暂停循环，把 confirmation_id / prompt / preview 写入会话，
            # 由前端 Builder 面板弹出「确认写入」浮层，用户确认后通过
            # /v1/agent/confirm 触发 handle_tool_confirm 恢复执行。
            if result.requires_confirmation:
                exit_reason = END_REASON_CONFIRMING
                session.is_executing = False
                session.is_paused = True
                session.interrupt_flag = True
                session.pending_confirmation_id = result.confirmation_id
                session.pending_confirmation_prompt = result.confirmation_prompt
                # run_command 场景 output 已是纯 str；write_file / git_commit 是 dict → json.dumps
                if isinstance(result.output, str):
                    session.pending_confirmation_preview = result.output
                else:
                    session.pending_confirmation_preview = json.dumps(result.output, ensure_ascii=False)
                session.pending_confirmation_tool = tool
                end_message = (
                    f"等待用户确认工具执行：{tool} - "
                    f"{result.confirmation_prompt or '(无提示)'}"
                )
                store.save(session)  # 确认状态由循环自身设置，直接保存
                logger.info(
                    f"[ReAct] 步骤 {step.id} 需要用户确认，循环暂停: "
                    f"tool={tool}, confirmation_id={result.confirmation_id}"
                )
                break  # 暂停循环，等待 /v1/agent/confirm 恢复

            # ---- Observe（观察）----
            observation = result.output or f"工具 {tool} 执行成功"

            if result.requires_interaction:
                # 需要用户手动执行（如 python REPL / npm init 无 --yes）
                # 不计入连续失败——这是正常的"需要人工介入"场景，不是 Agent 执行失败
                err_msg = result.error or "命令需要交互式输入"
                sm.mark_failed(step.id, err_msg)
                session.consecutive_failures = 0  # 不算连续失败，重置
                _save_preserving_flags(session, store)
                logger.info(
                    f"[ReAct] 步骤 {step.id} 需要交互式输入，提示用户手动执行: {tool}"
                )
                continue

            if not result.success:
                # 执行器返回 success=False（真实失败：命令 exit_code!=0 / 文件权限不足 / 黑名单拦截等）
                err_msg = result.error or "工具执行失败"
                sm.mark_failed(step.id, err_msg)

                # S8 第 79-80 天：熔断机制——连续失败计数
                # deny 不算（Agent 无法控制用户决策），交互式不算（已在上面处理）
                session.consecutive_failures += 1
                logger.warning(
                    f"[ReAct] 工具执行失败 step={step.id}: {err_msg}. "
                    f"连续失败={session.consecutive_failures}/{settings.TOOL_FAIL_FUSE_LIMIT}"
                )

                # 达到熔断阈值 → 暂停循环，提示人工介入
                if session.consecutive_failures >= settings.TOOL_FAIL_FUSE_LIMIT:
                    exit_reason = END_REASON_FUSED
                    fuse_msg = (
                        f"Agent 连续 {settings.TOOL_FAIL_FUSE_LIMIT} 次执行失败，"
                        f"触发熔断，已自动暂停。请人工检查后再恢复执行。"
                    )
                    session.final_answer = fuse_msg
                    end_message = fuse_msg
                    session.is_executing = False
                    session.is_paused = True
                    session.interrupt_flag = True
                    _save_preserving_flags(session, store)
                    logger.warning(
                        f"[ReAct] 熔断触发: session={session_id}, "
                        f"failures={session.consecutive_failures}, step={step.id}"
                    )
                    break

                _save_preserving_flags(session, store)
                continue

            # success=True：重置连续失败计数
            sm.mark_done(step.id, observation)
            session.consecutive_failures = 0
            _save_preserving_flags(session, store)
            logger.info(
                f"[ReAct] << Observe: step={step.id} done, "
                f"observation={observation[:80]}"
            )

    except Exception as e:
        # 循环顶层兜底，避免未捕获异常导致后台任务静默失败
        exit_reason = END_REASON_ERROR
        msg = f"Agent 执行异常: {e}"
        logger.error(f"[ReAct] 循环未捕获异常: session={session_id}: {e}", exc_info=True)
        session.final_answer = msg
        end_message = msg

    finally:
        # 循环结束，根据退出原因更新控制标志
        session.is_executing = False
        if exit_reason in (END_REASON_ASK_USER, END_REASON_PAUSED, END_REASON_CONFIRMING):
            # ask_user / 用户主动暂停 / 等待工具确认：保持暂停状态，等待 resume / ask/respond / confirm
            pass
        else:
            # completed / max_iter / timeout / error：清除暂停标志，进入终态
            session.is_paused = False
            session.interrupt_flag = False

        # 统一标记会话结束原因与描述（所有退出点都走这里）
        _mark_session_end(session, exit_reason, end_message)

        # 终态退出时，将剩余 pending/running 步骤标记为 failed，
        # 避免前端看到"卡住的 pending 步骤"。
        # 可恢复退出（paused / ask_user）保留 pending 状态以便 resume 后继续。
        if exit_reason in TERMINAL_END_REASONS:
            _fail_remaining_steps(
                sm, end_message or "会话结束，步骤未执行"
            )

        store.save(session)
        logger.info(
            f"[ReAct] 循环结束: session={session_id}, reason={exit_reason}, "
            f"iterations={iteration}, progress={session.done_steps}/{session.total_steps}"
        )


# ============================================================
# 后台任务启动 / 恢复
# ============================================================

async def start_agent_loop(
    session_id: str,
    tool_executor: Optional[ToolExecutor] = None,
) -> bool:
    """
    启动 Agent 循环（作为后台 asyncio.Task）。

    - 若同一会话已有正在运行的循环，返回 False（不重复启动）。
    - 否则创建后台任务并注册到 _running_agents。

    Args:
        session_id:    会话 ID
        tool_executor: 工具执行器（依赖注入）

    Returns:
        True 表示成功启动新循环；False 表示已有循环在运行。
    """
    async with _running_agents_lock:
        if is_agent_running(session_id):
            logger.info(f"[ReAct] 会话 {session_id} 已有正在运行的循环，跳过启动")
            return False

        task = asyncio.create_task(run_agent(session_id, tool_executor))

        def _on_done(t: "asyncio.Task[None]") -> None:
            _running_agents.pop(session_id, None)
            if t.cancelled():
                logger.info(f"[ReAct] 会话 {session_id} 的循环任务被取消")
            elif t.exception() is not None:
                logger.error(
                    f"[ReAct] 会话 {session_id} 的循环任务异常: {t.exception()}",
                    exc_info=t.exception(),
                )

        task.add_done_callback(_on_done)
        _running_agents[session_id] = task
        logger.info(f"[ReAct] 会话 {session_id} 的循环已启动为后台任务")
        return True


async def resume_agent_loop(
    session_id: str,
    tool_executor: Optional[ToolExecutor] = None,
) -> bool:
    """
    恢复 Agent 循环（在 pause 或 ask_user 之后）。

    清除中断/暂停标志，然后调用 start_agent_loop 重新启动循环。
    对于 ask_user 场景，调用方需先将发起提问的步骤标记为 done
    （observation 为用户回答），使下一轮循环能 pick up 后续步骤。

    Returns:
        True 表示成功恢复；False 表示会话不存在或已有循环在运行。
    """
    store = get_agent_session_store()
    session = store.get(session_id)
    if session is None:
        logger.error(f"[ReAct] resume 失败，会话不存在: {session_id}")
        return False

    session.is_paused = False
    session.interrupt_flag = False
    session.is_executing = True
    session.pending_question = None
    # 恢复执行时清除结束标记，表示会话重新进入运行态
    session.end_reason = None
    session.end_message = None
    store.save(session)

    return await start_agent_loop(session_id, tool_executor)


# ============================================================
# ask_user 人工介入：处理用户回复
# ============================================================

async def handle_ask_user_response(
    session_id: str,
    answer: str,
    tool_executor: Optional[ToolExecutor] = None,
) -> bool:
    """
    处理用户对 ask_user 问题的回复。

    流程：
      1. 找到当前处于 running 状态的步骤（即发起 ask_user 的步骤）。
      2. 将其标记为 done，observation 记录用户回答。
      3. 清除 pending_question，恢复循环。

    Args:
        session_id:    会话 ID
        answer:        用户回答内容
        tool_executor: 工具执行器（可选，恢复循环时传入；None 用全局默认）

    Returns:
        True 表示处理成功；False 表示会话不存在或无待回答问题。
    """
    store = get_agent_session_store()
    session = store.get(session_id)
    if session is None:
        return False
    if not session.pending_question:
        logger.warning(
            f"[ReAct] handle_ask_user_response: 会话 {session_id} 无待回答问题"
        )
        return False

    sm = AgentStateMachine(session)
    # 找到发起 ask_user 的步骤（处于 running 状态）
    running_steps = [s for s in session.plan if s.status == "running"]
    if running_steps:
        ask_step = running_steps[0]
        observation = f"[USER_ANSWER] {answer}"
        sm.mark_done(ask_step.id, observation)
        logger.info(
            f"[ReAct] ask_user 步骤 {ask_step.id} 已完成，"
            f"用户回答: {answer[:80]}"
        )
    else:
        logger.warning(
            f"[ReAct] handle_ask_user_response: 会话 {session_id} "
            f"无 running 步骤，跳过 mark_done"
        )

    # 清除待回答问题，恢复循环
    session.pending_question = None
    session.is_paused = False
    session.interrupt_flag = False
    session.is_executing = True
    # 用户回复后继续执行，清除结束标记
    session.end_reason = None
    session.end_message = None
    store.save(session)

    return await start_agent_loop(session_id, tool_executor)


# ============================================================
# S8 确认链路：处理用户对工具执行的确认
# ============================================================

async def handle_tool_confirm(
    session_id: str,
    confirmation_id: str,
    action: str,
    tool_executor: Optional[ToolExecutor] = None,
) -> tuple[bool, Optional[str]]:
    """
    处理用户对工具执行的确认操作（S8 核心确认链路）。

    流程：
      1. 校验会话存在 + 有 pending_confirmation + confirmation_id 一致。
      2. 调用 tool_registry.confirm_tool 真正执行（allow）或拒绝（deny）。
         - MockToolExecutor 模式下无真实落盘，返回模拟结果。
         - MCPToolExecutor 模式下从 PendingConfirmationStore 取出原始
           ToolCall，调用对应的 _do_* handler。
      3. 找到当前 running 步骤，根据执行结果标记 done / failed。
      4. 清除 pending_confirmation_* 字段，恢复 ReAct 循环。

    Args:
        session_id:       会话 ID
        confirmation_id: 从 status 接口拿到的确认凭证（一次性，幂等校验）
        action:           'allow' 或 'deny'
        tool_executor:    工具执行器（Mock 模式下用于生成模拟结果）；
                          MCP 模式下 confirm_tool 已自行查 registry，无需 executor。

    Returns:
        (True, result_summary) 表示成功；(False, error_msg) 表示失败。
    """
    store = get_agent_session_store()
    session = store.get(session_id)
    if session is None:
        return False, "会话不存在"
    if not session.pending_confirmation_id:
        logger.warning(
            f"[ReAct] handle_tool_confirm: 会话 {session_id} 无待确认的工具"
        )
        return False, "当前会话没有待确认的工具操作"
    if session.pending_confirmation_id != confirmation_id:
        logger.warning(
            f"[ReAct] handle_tool_confirm: confirmation_id 不匹配 "
            f"(请求={confirmation_id}, 会话={session.pending_confirmation_id})"
        )
        return False, "确认凭证与会话不匹配"

    # ---- 真正执行 / 拒绝 ----
    tool_name = session.pending_confirmation_tool or "<unknown>"
    if action == "deny":
        result_summary = "用户拒绝执行该工具操作"
        logger.info(
            f"[ReAct] 用户拒绝工具执行: session={session_id}, tool={tool_name}"
        )
        observation = "[USER_DENIED] 用户拒绝执行该操作"
        step_status = "failed"
        err_msg = result_summary
    else:
        # allow：调用真实确认流程
        # 优先走 tool_registry.confirm_tool（MCP 模式），
        # 若 confirmation_id 未命中 registry（Mock 模式下 registry 没存），
        # 则降级为模拟成功结果。
        from app.services.tool_registry import confirm_tool as registry_confirm
        registry_result = await registry_confirm(
            confirmation_id=confirmation_id,
            action="allow",
            session_id=session_id,
        )

        if registry_result.success:
            result_summary = registry_result.output or f"工具 {tool_name} 执行成功"
            observation = result_summary
            step_status = "done"
        elif registry_result.error == "确认凭证不存在或已过期":
            # 降级：Mock 模式下 registry 没存 confirmation_id，模拟成功
            result_summary = (
                f"[Mock] 工具 {tool_name} 确认后执行成功（模拟）"
            )
            observation = result_summary
            step_status = "done"
            logger.info(
                f"[ReAct] 降级为 Mock 确认执行: tool={tool_name}, "
                f"session={session_id}"
            )
        else:
            # 其他真实失败
            result_summary = registry_result.error or "工具执行失败"
            logger.warning(
                f"[ReAct] 工具确认后执行失败: session={session_id}, "
                f"tool={tool_name}, err={result_summary}"
            )
            observation = f"[CONFIRM_FAILED] {result_summary}"
            step_status = "failed"
            err_msg = result_summary

    # ---- 更新步骤状态 ----
    sm = AgentStateMachine(session)
    running_steps = [s for s in session.plan if s.status == "running"]
    if running_steps:
        target_step = running_steps[0]
        if step_status == "done":
            sm.mark_done(target_step.id, observation)
        else:
            sm.mark_failed(target_step.id, err_msg)
        logger.info(
            f"[ReAct] 步骤 {target_step.id} 因用户确认{action}而{step_status}: "
            f"session={session_id}"
        )
    else:
        logger.warning(
            f"[ReAct] handle_tool_confirm: 会话 {session_id} "
            f"无 running 步骤，跳过 mark_{step_status}"
        )

    # ---- 清除确认字段，恢复循环 ----
    session.pending_confirmation_id = None
    session.pending_confirmation_prompt = None
    session.pending_confirmation_preview = None
    session.pending_confirmation_tool = None
    session.is_paused = False
    session.interrupt_flag = False
    session.is_executing = True
    session.end_reason = None
    session.end_message = None
    store.save(session)

    # 重启 ReAct 循环继续后续步骤
    await start_agent_loop(session_id, tool_executor)

    return True, result_summary
