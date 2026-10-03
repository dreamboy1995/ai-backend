"""
S7 第 61-62 天：Agent 状态机

实现 Agent 任务图（DAG）的状态迁移与可执行步骤调度。

核心能力：
- mark_running(step_id)        : pending -> running
- mark_done(step_id, obs)      : running -> done，写入 observation
- mark_failed(step_id, err)    : running -> failed
- mark_blocked(step_id, reason): pending/running -> blocked（依赖无法满足或人工介入）
- get_next_runnable_step()     : 返回依赖全部完成的第一个 pending 步骤

DAG 合法性校验：
- validate_plan()：校验所有 dependency ID 真实存在、无循环依赖。
  对应 S7 风险预警："大模型经常在 dependencies 里填错 ID（拼写错误或引用不存在的步骤），
  后端必须在解析时做强校验"。
"""

import logging
import threading
import time
from typing import Dict, List, Optional, Set

from app.models.agent import AgentSession, StepStatus, TaskStep

logger = logging.getLogger(__name__)


class InvalidPlanError(ValueError):
    """任务图不合法（依赖 ID 不存在 / 存在循环依赖）"""
    pass


class StateTransitionError(ValueError):
    """非法状态迁移（如对 done 的步骤调用 mark_running）"""
    pass


class AgentStateMachine:
    """
    Agent 状态机。

    线程安全：所有修改 plan 的操作都持有 _lock，
    避免 ReAct 循环与 HTTP 轮询并发修改同一会话导致竞态。
    """

    def __init__(self, session: AgentSession):
        self._session = session
        self._lock = threading.Lock()
        # 步骤 ID -> 索引的映射，加速查找
        self._index: Dict[str, int] = {step.id: i for i, step in enumerate(session.plan)}

    # ------------------------------------------------------------------
    # 公开属性 / 查询
    # ------------------------------------------------------------------

    @property
    def session(self) -> AgentSession:
        return self._session

    def get_step(self, step_id: str) -> Optional[TaskStep]:
        """根据 ID 获取步骤，不存在返回 None"""
        idx = self._index.get(step_id)
        if idx is None:
            return None
        return self._session.plan[idx]

    def is_all_done(self) -> bool:
        """所有步骤均为 done 或 blocked（blocked 视为终态，不计入未完成）"""
        return all(s.status in ("done", "blocked") for s in self._session.plan)

    def is_any_failed(self) -> bool:
        """是否存在 failed 步骤"""
        return any(s.status == "failed" for s in self._session.plan)

    # ------------------------------------------------------------------
    # DAG 校验（对应 S7 风险预警：依赖 ID 强校验 + 循环依赖检测）
    # ------------------------------------------------------------------

    def validate_plan(self) -> None:
        """
        校验任务图合法性：
          1. 所有 dependency ID 必须真实存在于 plan 中。
          2. 不存在循环依赖。

        Raises:
            InvalidPlanError: 依赖 ID 不存在或存在循环依赖。
        """
        step_ids = set(self._index.keys())

        # 1. 校验依赖 ID 存在
        for step in self._session.plan:
            for dep_id in step.dependencies:
                if dep_id not in step_ids:
                    raise InvalidPlanError(
                        f"步骤 '{step.id}' 依赖了不存在的步骤 ID: '{dep_id}'"
                    )
                if dep_id == step.id:
                    raise InvalidPlanError(
                        f"步骤 '{step.id}' 不能依赖自身"
                    )

        # 2. 循环依赖检测（DFS 三色标记法）
        # WHITE=未访问, GRAY=访问中, BLACK=已完成
        WHITE, GRAY, BLACK = 0, 1, 2
        color: Dict[str, int] = {sid: WHITE for sid in step_ids}

        def _dfs(node: str, path: List[str]) -> None:
            color[node] = GRAY
            path.append(node)
            step = self.get_step(node)
            if step is None:
                return
            for dep in step.dependencies:
                if color[dep] == GRAY:
                    cycle = " -> ".join(path + [dep])
                    raise InvalidPlanError(f"检测到循环依赖: {cycle}")
                if color[dep] == WHITE:
                    _dfs(dep, path)
            path.pop()
            color[node] = BLACK

        for sid in step_ids:
            if color[sid] == WHITE:
                _dfs(sid, [])

        logger.debug(
            f"[StateMachine] 任务图校验通过: "
            f"session={self._session.session_id}, steps={len(self._session.plan)}"
        )

    # ------------------------------------------------------------------
    # 可执行步骤调度（DAG 拓扑调度）
    # ------------------------------------------------------------------

    def get_next_runnable_step(self) -> Optional[TaskStep]:
        """
        返回下一个可执行的步骤。

        可执行条件：
          - 状态为 pending
          - 所有依赖步骤的状态均为 done

        返回 plan 中第一个满足条件的步骤（按 plan 顺序），无则返回 None。

        注意：若存在 failed 步骤且其是某个 pending 步骤的依赖，
        该 pending 步骤永远不会被返回（依赖未完成），调用方需自行决定
        是否重试 failed 步骤或标记为 blocked。
        """
        with self._lock:
            for step in self._session.plan:
                if step.status != "pending":
                    continue
                if self._dependencies_satisfied_locked(step):
                    return step
            return None

    def _dependencies_satisfied_locked(self, step: TaskStep) -> bool:
        """判断步骤的所有依赖是否都已完成（需在持有 _lock 时调用）"""
        if not step.dependencies:
            return True
        for dep_id in step.dependencies:
            dep = self.get_step(dep_id)
            if dep is None:
                return False
            if dep.status != "done":
                return False
        return True

    def get_runnable_steps(self) -> List[TaskStep]:
        """返回当前所有可执行的步骤（供并行调度使用，S7 单线程执行时用 get_next_runnable_step）"""
        with self._lock:
            return [
                s for s in self._session.plan
                if s.status == "pending" and self._dependencies_satisfied_locked(s)
            ]

    # ------------------------------------------------------------------
    # 状态迁移
    # ------------------------------------------------------------------

    def mark_running(self, step_id: str) -> TaskStep:
        """
        将步骤标记为 running。

        合法迁移: pending -> running
        若步骤已为 running 则幂等返回；其他状态抛出 StateTransitionError。
        """
        with self._lock:
            step = self.get_step(step_id)
            if step is None:
                raise StateTransitionError(f"步骤不存在: {step_id}")
            if step.status == "running":
                return step
            if step.status != "pending":
                raise StateTransitionError(
                    f"步骤 '{step_id}' 状态为 {step.status}，无法标记为 running"
                    f"（仅 pending 可迁移到 running）"
                )
            step.status = "running"
            self._session.current_step_index = self._index[step_id]
            self._session.updated_at = time.time()
            logger.info(
                f"[StateMachine] 步骤开始执行: {step_id} "
                f"({step.description})"
            )
            return step

    def mark_done(self, step_id: str, observation: str = "") -> TaskStep:
        """
        将步骤标记为 done，并写入 observation。

        合法迁移: running -> done
        """
        with self._lock:
            step = self.get_step(step_id)
            if step is None:
                raise StateTransitionError(f"步骤不存在: {step_id}")
            if step.status == "done":
                return step
            if step.status != "running":
                raise StateTransitionError(
                    f"步骤 '{step_id}' 状态为 {step.status}，无法标记为 done"
                    f"（仅 running 可迁移到 done）"
                )
            step.status = "done"
            step.observation = observation
            self._session.updated_at = time.time()
            logger.info(
                f"[StateMachine] 步骤完成: {step_id} "
                f"({step.description})"
            )
            return step

    def mark_failed(self, step_id: str, error_msg: str) -> TaskStep:
        """
        将步骤标记为 failed，并记录错误信息到 observation。

        合法迁移: running -> failed
        """
        with self._lock:
            step = self.get_step(step_id)
            if step is None:
                raise StateTransitionError(f"步骤不存在: {step_id}")
            if step.status == "failed":
                return step
            if step.status != "running":
                raise StateTransitionError(
                    f"步骤 '{step_id}' 状态为 {step.status}，无法标记为 failed"
                    f"（仅 running 可迁移到 failed）"
                )
            step.status = "failed"
            step.observation = f"[ERROR] {error_msg}"
            self._session.updated_at = time.time()
            logger.warning(
                f"[StateMachine] 步骤失败: {step_id} "
                f"({step.description}): {error_msg}"
            )
            return step

    def mark_blocked(self, step_id: str, reason: str) -> TaskStep:
        """
        将步骤标记为 blocked（依赖无法满足 / 等待人工介入）。

        合法迁移: pending -> blocked, running -> blocked, failed -> blocked
        """
        with self._lock:
            step = self.get_step(step_id)
            if step is None:
                raise StateTransitionError(f"步骤不存在: {step_id}")
            if step.status in ("done", "blocked"):
                return step
            step.status = "blocked"
            step.observation = f"[BLOCKED] {reason}"
            self._session.updated_at = time.time()
            logger.info(
                f"[StateMachine] 步骤阻塞: {step_id} "
                f"({step.description}): {reason}"
            )
            return step

    def reset_step(self, step_id: str) -> TaskStep:
        """
        重置步骤为 pending（用于重试 failed 步骤，或暂停时回收 running 步骤）。

        合法迁移: failed -> pending, blocked -> pending, running -> pending
        - failed/blocked 重试时 retry_count + 1
        - running 回收（暂停场景）不增加 retry_count，因为该步骤尚未真正执行失败
        """
        with self._lock:
            step = self.get_step(step_id)
            if step is None:
                raise StateTransitionError(f"步骤不存在: {step_id}")
            if step.status == "pending":
                return step
            if step.status not in ("failed", "blocked", "running"):
                raise StateTransitionError(
                    f"步骤 '{step_id}' 状态为 {step.status}，无法重置"
                    f"（仅 failed/blocked/running 可重置为 pending）"
                )
            was_running = step.status == "running"
            step.status = "pending"
            step.observation = None
            if not was_running:
                step.retry_count += 1
            self._session.updated_at = time.time()
            logger.info(
                f"[StateMachine] 步骤重置为 pending: {step_id} "
                f"(from={'running' if was_running else 'failed/blocked'}, "
                f"retry_count={step.retry_count})"
            )
            return step

    def fail_remaining_steps(self, reason: str = "会话结束，步骤未执行") -> List[TaskStep]:
        """
        将所有 pending / running 步骤标记为 failed。

        用于会话终态退出（completed 存在失败步骤 / max_iter / timeout / error）时，
        清理无法再执行的剩余步骤，避免前端看到"卡住的 pending"。

        注意：done / failed / blocked 步骤保持原样，不重复处理。
        """
        with self._lock:
            failed: List[TaskStep] = []
            for step in self._session.plan:
                if step.status in ("pending", "running"):
                    step.status = "failed"
                    step.observation = f"[ERROR] {reason}"
                    failed.append(step)
            if failed:
                self._session.updated_at = time.time()
                logger.info(
                    f"[StateMachine] 会话结束，标记 {len(failed)} 个未完成步骤为 failed: "
                    f"{[s.id for s in failed]}"
                )
            return failed

    # ------------------------------------------------------------------
    # 整体进度
    # ------------------------------------------------------------------

    def progress(self) -> float:
        """返回整体进度（0.0 ~ 1.0）"""
        return self._session.progress

    def summary(self) -> Dict[str, int]:
        """返回各状态步骤数量统计"""
        counts: Dict[str, int] = {
            "pending": 0, "running": 0, "done": 0, "failed": 0, "blocked": 0
        }
        for s in self._session.plan:
            counts[s.status] += 1
        return counts
