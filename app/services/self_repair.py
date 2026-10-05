"""
S9 第 83-84 天：自修复循环引擎

这是 Agent "死磕到底"能力的核心——当某个步骤执行失败时，Agent 不会放弃，
而是主动解析错误、生成修复方案、重新执行，直到成功或达到重试上限。

核心流程（见 repair_loop 函数）：
  1. 错误解析：调用 error_parser.parse_error 将 stderr 转为 ParsedError
  2. 修复 Prompt：将错误 + 原始调用 + 代码片段打包给 LLM
  3. 修复生成：LLM 返回 write_file / run_command 的修复 action
  4. 修复执行：调用 tool_executor 执行修复 action（修改文件或换命令参数）
  5. 验证：重新执行原始失败的步骤验证修复是否有效
  6. 循环：成功则退出，失败则进入下一轮，直到 retry_count >= max_retries

风险预警防护（对应 S9 关键技术预研）：
  - 全局熔断：session.total_retries_used >= max_total_retries → 强制终止（自杀开关）
  - 修复无效：连续 N 次产生相同 Diff → 提前终止（模型"摆烂"）
  - 拆东墙补西墙：最近 N 次错误类型全不同 → 提前终止（模型在引入新错误）
  - 上下文隔离：修复 Prompt 只携带错误 + 代码片段，不带历史对话

设计要点：
  - 独立模块：与 react_loop.py 解耦，由 react_loop 在步骤失败时调用 repair_loop
  - 事件推送：每次修复尝试通过 WebSocket 推 repair_attempt 事件（Builder 面板时间线数据源）
  - 纯异步：所有外部调用（LLM / ToolExecutor / SSE）都是 async
  - 线程安全：不持有长生命周期锁——每次 repair 内部都是单次调用

Sprint_9.md 验收标准：
  "在单元测试中模拟一个'文件导入错误'的场景，Agent 在第 2 次重试时成功修正了 import 路径，
   任务继续。后端日志清晰打印出 Retry 1/3 failed -> Retry 2/3 success。"
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from app.config import settings
from app.models.agent import AgentSession, RepairAttempt, TaskStep
from app.models.tool import ToolResult
from app.services.diff_generator import (
    determine_context_lines,
    generate_unified_diff,
    read_original_content,
)
from app.services.error_parser import ParsedError, parse_error
from app.services.llm import AdapterError, chat_completion_text
from app.services.tool_executor import ToolExecutionError, ToolExecutor

logger = logging.getLogger(__name__)


# ============================================================
# S9 第 85-86 天：测试沙箱在自修复中的集成
# ============================================================

# 缓存标记：项目是否有测试文件（每 session 只检测一次，后续修复循环复用结果）
_session_has_tests_cache: Dict[str, bool] = {}


async def _should_run_tests(session: AgentSession, step: TaskStep) -> bool:
    """
    判断本次修复后是否应该自动运行测试。

    条件（全部满足时返回 True）：
      1. 全局启用测试沙箱（settings.TEST_RUNNER_ENABLED）
      2. 修复工具是 write_file（代码修改场景）
      3. 项目中存在测试文件（has_tests_in_project）

    返回 False 时表示跳过 run_tests。
    """
    if not settings.TEST_RUNNER_ENABLED:
        return False
    # run_command 场景（比如改命令参数）不触发测试
    # 只有 write_file 代码修改场景才触发
    repair_tool = None  # caller 会传入
    # 项目有测试文件检测（带缓存）
    session_id = session.session_id
    if session_id not in _session_has_tests_cache:
        try:
            from app.services.test_runner import has_tests_in_project
            _session_has_tests_cache[session_id] = await has_tests_in_project(
                session.workspace_root
            )
            logger.info(
                f"[SelfRepair] 测试文件检测（首次）: session={session_id}, "
                f"has_tests={_session_has_tests_cache[session_id]}"
            )
        except Exception as e:
            logger.warning(f"[SelfRepair] 测试文件检测异常，跳过 run_tests: {e}")
            _session_has_tests_cache[session_id] = False
            return False
    return _session_has_tests_cache[session_id]


# ============================================================
# S9 第 85-86 天：测试结果持久化辅助函数
# ============================================================

def _append_test_result_to_session(session: AgentSession, result) -> None:
    """
    将 TestRunResult 追加到 session.test_results 并持久化到 Redis。

    追加的结构为 TestRunResult.model_dump()（含 success, summary, failures,
    passed/failed/errors/skipped/total, framework, duration_ms 等完整字段）。
    调用方不需要单独再调用 store.save(session) —— 本函数内部已处理。

    容错：任何异常只记 warning，不中断自修复主流程（测试结果持久化是旁路）。
    """
    from app.services.agent_session_store import get_agent_session_store

    try:
        record = result.model_dump()
        session.test_results.append(record)
        store = get_agent_session_store()
        store.save(session)
        logger.debug(
            f"[SelfRepair] 测试结果已追加到 session: session={session.session_id}, "
            f"total_records={len(session.test_results)}"
        )
    except Exception as e:
        logger.warning(
            f"[SelfRepair] 测试结果持久化失败（不影响自修复流程）: {e}"
        )


async def _run_tests_after_repair(
    session: AgentSession,
    step: TaskStep,
    attempt: int,
    max_retries: int,
) -> Tuple[Optional["ParsedError"], bool]:
    """
    修复成功执行后自动运行测试（S9 风险预警：大模型"幻觉修复"的验收裁判）。

    流程：
      1. 调用 run_tests 运行测试套件。
      2. 测试通过 → 返回 (None, False)，正常走后续验证流程。
      3. 测试失败 → 将测试失败信息转为 ParsedError，返回 (parsed_error, True)。
         调用方将用新的 parsed_error 驱动下一轮修复 Prompt（相当于测试失败
         也算一种"错误来源"）。

    Returns:
        (new_parsed_error, tests_failed):
        - tests_failed=True 表示测试失败，调用方应用新 parsed_error 替换旧的
        - tests_failed=False 表示测试通过或跳过
    """
    from app.services.test_runner import run_tests, TestRunResult, result_to_summary

    logger.info(
        f"[SelfRepair] 自动触发 run_tests（代码修改后）: "
        f"step={step.id}, attempt={attempt}/{max_retries}"
    )

    try:
        result: TestRunResult = await run_tests(
            workspace_root=session.workspace_root,
            session_id=session.session_id,
        )
    except Exception as e:
        logger.warning(
            f"[SelfRepair] run_tests 执行异常（跳过，非致命）: {e}"
        )
        return None, False

    summary = result_to_summary(result)
    logger.info(
        f"[SelfRepair] run_tests 结果: passed={result.passed}, "
        f"failed={result.failed}, errors={result.errors}, summary={summary}"
    )

    # ---- S9 第 85-86 天：将测试结果追加到 session.test_results 并持久化 ----
    # 让前端 Builder 面板通过 /v1/agent/status/{id} 拿到测试时间线数据源。
    # 无论测试通过、失败、超时还是框架不可用，都记录下来供 UI 渲染。
    _append_test_result_to_session(session, result)

    # 测试通过或超时但已有基本通过 → 不干扰自修复流程
    if result.success:
        return None, False

    # 超时提示：不强制将超时当作错误来源（测试只是没跑完，不一定真有 bug）
    if result.timed_out:
        logger.warning(
            f"[SelfRepair] 测试执行超时，跳过作为错误来源: {summary}"
        )
        return None, False

    # 框架不可用 → 跳过（无法跑测试）
    if result.framework == "unknown" and result.error_message:
        logger.warning(
            f"[SelfRepair] 测试框架不可用，跳过: {result.error_message}"
        )
        return None, False

    # 测试失败 → 构造 ParsedError 注入修复循环
    # 用第一个失败用例的 error 作为核心信息
    first_failure = result.failures[0] if result.failures else None
    error_msg_parts = [f"测试失败: {summary}"]
    if first_failure:
        error_msg_parts.append(f"首个失败: {first_failure.test_name}")
        if first_failure.error:
            error_msg_parts.append(f"错误信息: {first_failure.error}")

    test_error = ParsedError(
        error_type="TestFailure",
        error_message="\n".join(error_msg_parts),
        file_path=first_failure.file if first_failure else None,
        line_number=first_failure.line if first_failure else None,
        language="python" if result.framework in ("pytest", "unittest") else "javascript",
        code_snippet="\n".join(
            f"- {tf.test_name}: {tf.error[:80]}"
            for tf in result.failures[:5]
        ),
    )

    logger.warning(
        f"[SelfRepair] 测试失败作为新错误来源: {test_error.error_type}: "
        f"{test_error.error_message[:200]}"
    )
    return test_error, True


# ============================================================
# 工具函数：构建修复 Prompt
# ============================================================

def _truncate_code_snippet(snippet: str, max_lines: int) -> str:
    """
    将过长的代码片段截断到指定行数。

    策略：保留末尾 max_lines 行（最靠近错误位置的上下文），
    这样即使原代码 1000 行，修复 Prompt 里也只放最后 50 行。
    """
    if not snippet:
        return ""
    lines = snippet.splitlines()
    if len(lines) <= max_lines:
        return snippet
    return "\n".join(lines[-max_lines:])


def _build_repair_messages(
    session: AgentSession,
    step: TaskStep,
    parsed_error: ParsedError,
) -> List[Dict[str, str]]:
    """
    构建自修复子循环的 messages 列表。

    关键设计（S9 Sprint_9.md 第 83-84 天"上下文隔离"要求）：
    - 修复 Prompt 只携带：错误信息 + 原始工具调用 + 代码片段
    - 不带 Agent 的完整执行历史，防止模型"分心"
    - 使用独立的 System Prompt，不与 Reason 阶段 Prompt 混用

    Args:
        session:       当前 AgentSession（用于拿 workspace_root / model）
        step:          失败的步骤（用于拿 action / action_input）
        parsed_error:  结构化的错误对象（error_parser 输出）

    Returns:
        messages 列表（可直接传给 chat_completion_text）
    """
    # 选择合适的 Prompt 模板（有/无 file_path 版本）
    template = settings.SELF_REPAIR_PROMPT_TEMPLATE if parsed_error.file_path else settings.SELF_REPAIR_PROMPT_NO_FILE

    # 处理原始工具参数字符串（用于 Prompt 展示）
    raw_params = step.action_input
    if isinstance(raw_params, dict):
        params_str = json.dumps(raw_params, ensure_ascii=False)
    else:
        params_str = str(raw_params)

    # 截断代码片段
    code_snippet = _truncate_code_snippet(
        parsed_error.code_snippet or "",
        settings.SELF_REPAIR_CODE_SNIPPET_MAX_LINES,
    )

    # 填充模板占位符
    prompt_text = template.format(
        error_type=parsed_error.error_type,
        error_message=parsed_error.error_message[:500],  # 错误信息截断防止过长
        file_path=parsed_error.file_path or "(unknown)",
        line_number=parsed_error.line_number or "?",
        language=parsed_error.language or "unknown",
        tool_name=step.action or "(unknown tool)",
        tool_params=params_str[:1000],
        code_snippet=code_snippet if code_snippet else "(错误栈中未检测到源码行)",
    )

    return [
        {"role": "system", "content": prompt_text},
        {"role": "user", "content": "请决定如何修复这个错误，返回修复方案的 JSON。"},
    ]


# ============================================================
# 工具函数：从 LLM 输出提取修复 action
# ============================================================

def _extract_repair_action(text: str) -> Optional[Dict[str, Any]]:
    """
    从自修复阶段的 LLM 输出中提取 {tool, params} JSON。

    复用 react_loop._extract_action_json 的鲁棒性逻辑：
    - 去除 markdown 代码围栏
    - 括号深度跟踪定位第一个完整 JSON
    - 移除尾随逗号
    - 清理不可见控制字符
    """
    import re as _re

    if not text:
        return None

    text = _re.sub(r"```(?:json)?", "", text, flags=_re.IGNORECASE).strip()

    # 括号深度跟踪
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    json_end = -1
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
                json_end = i + 1
                break

    if json_end == -1:
        return None

    json_str = text[start:json_end]
    # 规范化
    json_str = _re.sub(r",(\s*[}\]])", r"\1", json_str)
    json_str = _re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", json_str)

    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(
            f"[SelfRepair] 修复 JSON 解析失败: {e}, "
            f"context={json_str[max(0, e.pos - 50):e.pos + 50]!r}"
        )
        return None

    if not isinstance(parsed, dict):
        return None
    tool = parsed.get("tool")
    params = parsed.get("params")
    if not tool or not isinstance(params, dict):
        return None

    # 安全校验：修复工具只允许 write_file / run_command（自修复只做这两件事）
    if tool not in ("write_file", "run_command"):
        logger.warning(f"[SelfRepair] 修复工具不被允许: {tool}，只允许 write_file/run_command")
        return None

    return {"tool": str(tool), "params": params}


# ============================================================
# 工具函数：Diff 生成与比较（风险预警用）
# ============================================================

def _generate_repair_diff(
    file_path: str,
    old_content: str,
    new_content: str,
) -> str:
    """
    为自修复生成文件的 Unified Diff。

    复用 diff_generator 的能力：动态上下文行数（小文件完整、大文件只保留变更附近）。
    """
    if old_content == new_content:
        return ""
    effective_context = determine_context_lines(
        old_content if len(old_content) >= len(new_content) else new_content
    )
    return generate_unified_diff(old_content, new_content, file_path, context_lines=effective_context)


def _is_too_many_same_diffs(repair_history: List[RepairAttempt]) -> bool:
    """
    风险预警：连续 N 次修复产生相同 Diff → 判定为"修复无效"（模型"摆烂"）。

    检查最近 N 次 repair_history（不含原始失败记录），
    如果它们的 diff 字符串完全相同，说明模型没有在做任何实质改动。
    """
    n = settings.SELF_REPAIR_SAME_DIFF_MAX
    if len(repair_history) < n:
        return False

    recent_diffs = [ra.diff for ra in repair_history[-n:] if ra.diff]
    # 全部相同 → 无效
    if len(recent_diffs) >= n and len(set(recent_diffs)) == 1:
        logger.warning(
            f"[SelfRepair] 检测到连续 {n} 次相同 Diff → 修复无效，提前终止"
        )
        return True

    return False


def _is_too_many_different_errors(repair_history: List[RepairAttempt]) -> bool:
    """
    风险预警：最近 N 次修复的错误类型全部不同 → 判定模型在"拆东墙补西墙"。

    意味着模型每修一个错误就引入一个新的不同错误。
    """
    n = settings.SELF_REPAIR_DIFFERENT_ERROR_MAX
    if len(repair_history) < n:
        return False

    recent_errors = [ra.error_summary for ra in repair_history[-n:] if ra.error_summary]
    # 全部不同（没有两个相同的 error_summary）
    if len(recent_errors) >= n and len(set(recent_errors)) == len(recent_errors):
        logger.warning(
            f"[SelfRepair] 最近 {n} 次错误类型全不同 → 模型在拆东墙补西墙，终止修复"
        )
        return True

    return False


# ============================================================
# 事件推送：repair_attempt 事件
# ============================================================

async def _publish_repair_attempt(session_id: str, attempt: RepairAttempt) -> None:
    """
    将 RepairAttempt 通过 WebSocket 推送给 Builder 面板。

    对应 Sprint_9.md "后端在 SSE 流中新增 repair_attempt 事件"。
    这里复用 command_executor 的 StreamManager，使用扩展后的 StreamMessage
    （type="repair_attempt"，extra 携带结构化 payload）。

    消息结构：
      {"type":"repair_attempt","content":"Retry 1/3 failed: ModuleNotFoundError",
       "timestamp":"...","extra":{...完整 RepairAttempt payload...}}
    """
    try:
        from app.services.command_executor import get_stream_manager
        from app.services.command_executor import _now_iso
        from app.models.tool import StreamMessage

        manager = get_stream_manager()
        payload = attempt.to_sse_dict()
        content = (
            f"Retry {attempt.attempt_number}/{attempt.max_retries} "
            f"{attempt.result}: {attempt.error_summary[:100]}"
        )
        msg = StreamMessage(
            type="repair_attempt",
            content=content,
            timestamp=_now_iso(),
            extra=payload,
        )
        await manager.publish(session_id, msg)
    except Exception as e:
        # 推送失败不影响主流程，只记 warning
        logger.warning(
            f"[SelfRepair] repair_attempt 事件推送失败（非致命）: {e}"
        )


# ============================================================
# 核心函数：自修复循环入口
# ============================================================

async def repair_loop(
    session: AgentSession,
    step: TaskStep,
    original_tool_result: ToolResult,
    tool_executor: ToolExecutor,
    workspace_root: str,
) -> Tuple[bool, str]:
    """
    自修复子循环主入口。

    当 react_loop 中某个步骤执行失败（ToolResult.success == False）时调用。
    负责：错误解析 → Prompt 构建 → LLM 生成修复方案 → 执行修复 → 验证。

    Args:
        session:              当前 AgentSession（含 total_retries_used 熔断计数）
        step:                 失败的 TaskStep（含 action/action_input/max_retries）
        original_tool_result: 原始失败步骤的 ToolResult（含 stderr/error）
        tool_executor:        工具执行器（复用 react_loop 注入的那个）
        workspace_root:       工作区根目录（用于文件 Diff / write_file 路径安全）

    Returns:
        (success, final_observation):
        - success=True  : 自修复成功（步骤现在可以重新标记为 done）
        - success=False : 自修复失败（应 mark_failed）
        - final_observation: 修复成功后重新执行原始步骤得到的 observation
                             或修复失败时的错误描述（供 observation 使用）

    副作用：
        - 修改 step.repair_history：追加每次 RepairAttempt
        - 修改 step.retry_count：成功时设置为实际尝试次数
        - 修改 session.total_retries_used：累加本次尝试次数
        - 修改 step.last_parsed_error：写入结构化错误（S9 status 接口可返回）
        - 通过 WebSocket 推送 repair_attempt 事件（每次尝试一条）
    """
    # ---- 前置：全局开关 & 步骤级开关 ----
    if not settings.SELF_REPAIR_ENABLED:
        logger.info(f"[SelfRepair] 全局自修复开关关闭，跳过 step={step.id}")
        return False, "自修复循环全局禁用"

    if step.disable_self_repair:
        logger.info(f"[SelfRepair] 步骤 {step.id} 标记 disable_self_repair，跳过")
        return False, "步骤标记为禁用自修复"

    max_retries = step.max_retries if step.max_retries > 0 else settings.SELF_REPAIR_MAX_RETRIES
    if max_retries <= 0:
        return False, "步骤 max_retries <= 0，不执行自修复"

    # ---- 全局熔断检查 ----
    if session.total_retries_used >= session.max_total_retries:
        msg = (
            f"全局自修复熔断已触发 "
            f"({session.total_retries_used}/{session.max_total_retries})，"
            f"停止所有修复"
        )
        logger.warning(f"[SelfRepair] {msg}")
        return False, msg

    # ---- 错误解析 ----
    # 优先用 original_tool_result.stderr，降级用 error，再降级用 output
    raw_err = (
        getattr(original_tool_result, "stderr", None)
        or original_tool_result.error
        or (original_tool_result.output if not original_tool_result.success else None)
        or ""
    )
    parsed_error = parse_error(str(raw_err), cwd=workspace_root or "")

    # 写入 step.last_parsed_error（S9 status 接口返回供 Builder 面板展示）
    step.last_parsed_error = parsed_error.model_dump()

    logger.info(
        f"[SelfRepair] 开始自修复: step={step.id}, "
        f"tool={step.action}, error={parsed_error.error_type}: {parsed_error.error_message[:80]}, "
        f"max_retries={max_retries}, total_used={session.total_retries_used}/{session.max_total_retries}"
    )

    # ---- 修复子循环 ----
    attempt = 0
    last_verify_error: str = ""
    last_diff: str = ""
    success = False

    for attempt in range(1, max_retries + 1):
        # ---- 全局熔断二次检查（修复过程中 total 可能被其他步骤消耗）----
        if session.total_retries_used >= session.max_total_retries:
            logger.warning(
                f"[SelfRepair] 全局熔断触发：session.total_retries_used="
                f"{session.total_retries_used}/{session.max_total_retries}，"
                f"终止 step={step.id} 的自修复"
            )
            # 记录一条 fused 状态的 RepairAttempt
            ra = RepairAttempt(
                attempt_number=attempt,
                max_retries=max_retries,
                error_summary=str(parsed_error),
                error_file=parsed_error.file_path,
                error_line=parsed_error.line_number,
                diff="",
                result="fused",
                result_summary="全局自修复熔断触发",
            )
            step.repair_history.append(ra)
            await _publish_repair_attempt(session.session_id, ra)
            break

        # ---- 构建修复 Prompt ----
        messages = _build_repair_messages(session, step, parsed_error)

        # ---- 让 LLM 生成修复 action ----
        repair_action: Optional[Dict[str, Any]] = None
        try:
            raw_output = await chat_completion_text(
                messages=messages,
                model=session.model,
                temperature=settings.SELF_REPAIR_TEMPERATURE,
                timeout=settings.SELF_REPAIR_STEP_TIMEOUT_SECONDS,
                max_tokens=settings.SELF_REPAIR_MAX_TOKENS,
                response_format={"type": "json_object"},
            )
            repair_action = _extract_repair_action(raw_output)
        except AdapterError as e:
            logger.error(f"[SelfRepair] LLM 调用失败 step={step.id}: {e}")
        except Exception as e:
            logger.error(f"[SelfRepair] LLM 异常 step={step.id}: {e}", exc_info=True)

        if repair_action is None:
            ra = RepairAttempt(
                attempt_number=attempt,
                max_retries=max_retries,
                error_summary=str(parsed_error),
                error_file=parsed_error.file_path,
                error_line=parsed_error.line_number,
                diff="",
                result="failed",
                result_summary="LLM 未返回合法修复方案",
            )
            step.repair_history.append(ra)
            await _publish_repair_attempt(session.session_id, ra)
            last_verify_error = "LLM 未返回合法修复方案"
            continue

        tool_name = repair_action["tool"]
        tool_params = repair_action["params"]
        logger.info(
            f"[SelfRepair] step={step.id} 第 {attempt}/{max_retries} 次修复: "
            f"repair_tool={tool_name}, params={tool_params}"
        )

        # ---- 执行修复 action ----
        # 生成 diff（仅对 write_file 有意义）
        diff_text = ""
        repair_tool_result: Optional[ToolResult] = None

        try:
            if tool_name == "write_file":
                file_path = tool_params.get("path", "")
                old_content = read_original_content(
                    file_path, workspace_root
                )
                new_content = tool_params.get("content", "")
                diff_text = _generate_repair_diff(
                    file_path, old_content, new_content
                )
                last_diff = diff_text

            repair_tool_result = await tool_executor.execute(
                tool_name,
                tool_params,
                workspace_root=workspace_root,
                session_id=session.session_id,
            )
        except ToolExecutionError as e:
            logger.warning(f"[SelfRepair] 修复工具执行失败 step={step.id}: {e}")
            diff_text = ""
        except Exception as e:
            logger.warning(f"[SelfRepair] 修复工具异常 step={step.id}: {e}", exc_info=True)
            diff_text = ""

        # ---- 风险预警：修复无效检查 ----
        # 修复 action 执行了（不管成功还是失败），都记录一条 RepairAttempt
        ra = RepairAttempt(
            attempt_number=attempt,
            max_retries=max_retries,
            error_summary=str(parsed_error),
            error_file=parsed_error.file_path,
            error_line=parsed_error.line_number,
            diff=diff_text,
            result="failed",  # 先假设失败，验证后再改
            result_summary=f"执行 {tool_name} 进行修复",
        )

        # ============================================================
        # S9 第 85-86 天：测试沙箱自动触发（代码修改后）
        # ============================================================
        # 当修复工具是 write_file（代码修改）且项目有测试文件时，
        # 自动运行测试套件。测试失败作为新的"错误"来源注入下一轮修复 Prompt。
        # 对应 S9 风险预警："大模型幻觉修复 → 测试是唯一可靠验收裁判"
        test_error_injected = False
        if (
            repair_tool_result
            and repair_tool_result.success
            and tool_name == "write_file"
        ):
            tests_should_run = await _should_run_tests(session, step)
            if tests_should_run:
                new_parsed_err, tests_failed = await _run_tests_after_repair(
                    session, step, attempt, max_retries
                )
                if tests_failed and new_parsed_err is not None:
                    # 注入测试失败作为新的错误来源
                    parsed_error = new_parsed_err
                    last_verify_error = f"测试失败（{new_parsed_err.error_type}）"
                    test_error_injected = True
                    ra.result_summary += f" | run_tests 失败"
                    # 测试失败已经是"错误"了，跳过原始步骤的 verify——
                    # 直接走下一轮修复，Prompt 用测试失败信息
                    verify_result = None
                else:
                    # 测试通过或跳过，正常继续原始步骤验证
                    pass

        # ---- 验证：重新执行原始失败的步骤 ----
        verify_result: Optional[ToolResult] = None
        if not test_error_injected and repair_tool_result and repair_tool_result.success:
            try:
                verify_result = await tool_executor.execute(
                    step.action,
                    step.action_input,
                    workspace_root=workspace_root,
                    session_id=session.session_id,
                )
            except ToolExecutionError as e:
                last_verify_error = f"验证步骤执行失败: {e}"
                logger.warning(f"[SelfRepair] 验证步骤异常 step={step.id}: {e}")
            except Exception as e:
                last_verify_error = f"验证步骤执行异常: {e}"
                logger.warning(f"[SelfRepair] 验证步骤异常 step={step.id}: {e}", exc_info=True)

        # ---- 判断本次修复是否成功 ----
        if verify_result and verify_result.success:
            ra.result = "success"
            ra.result_summary = (
                f"修复成功！重新执行 {step.action} 返回 success。"
                f"observation={verify_result.output[:100] if verify_result.output else '(无)'}"
            )
            success = True
            # 累加全局重试次数（成功也算消耗了 attempt 次机会）
            session.total_retries_used += attempt
            step.retry_count = attempt
            step.repair_history.append(ra)
            await _publish_repair_attempt(session.session_id, ra)
            logger.info(
                f"[SelfRepair] ✅ 修复成功 step={step.id}: "
                f"Retry {attempt}/{max_retries} success"
            )
            # 更新最后 parsed error
            return True, verify_result.output or f"工具 {step.action} 执行成功（自修复后）"

        # 本次修复仍失败
        ra.result = "failed"
        ra.result_summary = (
            f"修复后验证仍失败。最新错误: "
            f"{(verify_result.error if verify_result else '执行异常') or '(未知)'}"
        )
        step.repair_history.append(ra)
        await _publish_repair_attempt(session.session_id, ra)

        # 更新 parsed_error 为最新一次 verify 错误（下次 Prompt 用新错误信息）
        if verify_result:
            verify_err = (
                getattr(verify_result, "stderr", None)
                or verify_result.error
                or str(verify_result.output or "")
            )
            if verify_err:
                parsed_error = parse_error(str(verify_err), cwd=workspace_root or "")
                last_verify_error = str(parsed_error)

        # ---- 风险预警：连续相同 Diff 检查 ----
        if _is_too_many_same_diffs(step.repair_history):
            ra_skip = RepairAttempt(
                attempt_number=attempt,
                max_retries=max_retries,
                error_summary="(风险预警触发：连续相同 Diff)",
                result="skipped",
                result_summary="连续修复产生相同 Diff → 模型无实质改动，提前终止",
            )
            step.repair_history.append(ra_skip)
            await _publish_repair_attempt(session.session_id, ra_skip)
            break

        # ---- 风险预警：拆东墙补西墙检查 ----
        if _is_too_many_different_errors(step.repair_history):
            ra_skip = RepairAttempt(
                attempt_number=attempt,
                max_retries=max_retries,
                error_summary="(风险预警触发：错误类型全不同)",
                result="skipped",
                result_summary="连续修复引入全新错误 → 模型在拆东墙补西墙，提前终止",
            )
            step.repair_history.append(ra_skip)
            await _publish_repair_attempt(session.session_id, ra_skip)
            break

        logger.warning(
            f"[SelfRepair] step={step.id} Retry {attempt}/{max_retries} failed"
        )

    # ---- 循环结束：失败 ----
    # 累加全局重试次数
    session.total_retries_used += attempt
    step.retry_count = attempt
    step.last_parsed_error = parsed_error.model_dump()

    logger.warning(
        f"[SelfRepair] ❌ 自修复失败 step={step.id}: "
        f"重试 {attempt}/{max_retries} 次均未成功。"
        f"final_error={last_verify_error[:200]}"
    )
    return False, (
        f"自修复失败（{attempt} 次重试耗尽）。"
        f"最新错误：{last_verify_error[:300]}"
    )
