"""
会话管理服务（第 11-12 天：后端会话管理 & 历史记忆）

提供基于内存的会话存储（生产环境建议替换为 Redis），
支持多轮对话历史的滑动窗口裁剪，避免 Token 无限膨胀。

核心能力：
- create(session_id)：创建新会话
- append(session_id, msg)：追加单条消息
- get_history(session_id)：获取裁剪后的历史消息
- delete(session_id)：删除会话
- 会话自动过期（TTL）
- 滑动窗口裁剪：保留 System Prompt + 最近 N 轮对话，超出 Token 预算时丢弃最旧消息
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from app.config import settings

logger = logging.getLogger(__name__)

# 尝试加载 tiktoken 进行精确 Token 计数
try:
    import tiktoken
    _ENCODING = tiktoken.get_encoding("cl100k_base")
    _TIKTOKEN_AVAILABLE = True
except Exception as e:  # pragma: no cover - tiktoken 不可用时降级
    logger.warning(f"tiktoken 不可用，将使用字符长度估算 Token：{e}")
    _ENCODING = None
    _TIKTOKEN_AVAILABLE = False


def count_tokens(messages: List[dict]) -> int:
    """
    精确计算消息列表的 Token 数量。
    使用 tiktoken 的 cl100k_base 编码；若不可用则降级为字符长度 / 4 估算。
    """
    if not messages:
        return 0

    if _TIKTOKEN_AVAILABLE and _ENCODING is not None:
        total = 0
        for msg in messages:
            content = msg.get("content", "") or ""
            # 每条消息固定开销（role 等），约 4 token
            total += 4
            total += len(_ENCODING.encode(content))
        # 收尾开销
        total += 2
        return total

    # 降级估算
    total_chars = sum(len(m.get("content", "") or "") for m in messages)
    return total_chars // 4


class SessionService:
    """
    会话管理服务（内存实现）。

    设计说明：
    - 使用内存 dict 存储会话消息与过期时间，适合单机开发/小规模部署。
    - 生产环境可通过替换 _store / _expiry 的底层实现迁移到 Redis，
      对外接口（create / append / get_history / delete）保持不变。
    """

    def __init__(
        self,
        ttl_seconds: int = None,
        token_budget: int = None,
        token_margin: float = None,
        max_rounds: int = None,
    ):
        self._ttl = ttl_seconds if ttl_seconds is not None else settings.SESSION_TTL_SECONDS
        self._token_budget = token_budget if token_budget is not None else settings.SESSION_TOKEN_BUDGET
        self._token_margin = token_margin if token_margin is not None else settings.SESSION_TOKEN_MARGIN
        self._max_rounds = max_rounds if max_rounds is not None else settings.SESSION_MAX_ROUNDS

        # 触发裁剪的 Token 阈值 = 预算 * (1 - 余量比例)，留 20% 余量避免超限
        self._token_threshold = int(self._token_budget * (1 - self._token_margin))

        # 会话存储：session_id -> 消息列表
        self._store: Dict[str, List[dict]] = {}
        # 会话过期时间：session_id -> datetime
        self._expiry: Dict[str, datetime] = {}
        # 线程锁，保证并发安全
        self._lock = threading.Lock()

        logger.info(
            f"SessionService 初始化完成："
            f"TTL={self._ttl}s, Token预算={self._token_budget}, "
            f"裁剪阈值={self._token_threshold}, 保留轮数={self._max_rounds}, "
            f"tiktoken={'启用' if _TIKTOKEN_AVAILABLE else '降级(字符估算)'}"
        )

    # ------------------------------------------------------------------
    # 基础会话操作
    # ------------------------------------------------------------------
    def create(self, session_id: str) -> None:
        """创建一个空会话。若会话已存在则刷新过期时间。"""
        with self._lock:
            if session_id not in self._store:
                self._store[session_id] = []
                logger.info(f"[Session] 创建新会话: {session_id}")
            self._refresh_expiry_locked(session_id)

    def append(self, session_id: str, msg: dict) -> None:
        """
        向会话追加一条消息。
        若会话不存在则自动创建。
        若追加的消息与会话最后一条完全相同（重试场景），则跳过避免重复。
        """
        with self._lock:
            if session_id not in self._store:
                self._store[session_id] = []
            history = self._store[session_id]
            # 去重：防止客户端重试导致同一条消息重复入库
            if history and self._msg_equal(history[-1], msg):
                logger.debug(f"[Session] 跳过重复消息: session={session_id}")
                self._refresh_expiry_locked(session_id)
                return
            history.append(msg)
            self._refresh_expiry_locked(session_id)
            logger.debug(
                f"[Session] 追加消息: session={session_id}, "
                f"role={msg.get('role')}, 当前历史条数={len(history)}"
            )

    def get_history(self, session_id: str, reserved_tokens: int = 0) -> List[dict]:
        """
        获取裁剪后的会话历史。
        - 若会话不存在或已过期，返回空列表。
        - 裁剪策略：保留 System 消息 + 最近 N 轮对话，超出 Token 预算时丢弃最旧非 System 消息。
        - reserved_tokens：为系统提示词（含上下文 XML）预留的 Token 数，会从裁剪阈值中扣除，
          保证「系统提示词 + 历史」总 Token 不超限（S2 第 15-16 天要求）。
        """
        with self._lock:
            if session_id not in self._store:
                return []
            if self._is_expired_locked(session_id):
                logger.info(f"[Session] 会话已过期，自动清理: {session_id}")
                self._delete_locked(session_id)
                return []

            raw_history = list(self._store[session_id])
            self._refresh_expiry_locked(session_id)

        # 在锁外执行裁剪（裁剪不涉及共享状态修改）
        trimmed = self._trim_history(raw_history, reserved_tokens=reserved_tokens)
        logger.info(
            f"[Session] 裁剪历史: session={session_id}, "
            f"原始条数={len(raw_history)}, 裁剪后条数={len(trimmed)}, "
            f"原始Token={count_tokens(raw_history)}, 裁剪后Token={count_tokens(trimmed)}, "
            f"预留系统提示词Token={reserved_tokens}"
        )
        return trimmed

    def delete(self, session_id: str) -> None:
        """删除指定会话。"""
        with self._lock:
            self._delete_locked(session_id)

    def exists(self, session_id: str) -> bool:
        """判断会话是否存在且未过期。"""
        with self._lock:
            if session_id not in self._store:
                return False
            if self._is_expired_locked(session_id):
                self._delete_locked(session_id)
                return False
            return True

    # ------------------------------------------------------------------
    # 滑动窗口裁剪算法
    # ------------------------------------------------------------------
    def _trim_history(self, messages: List[dict], reserved_tokens: int = 0) -> List[dict]:
        """
        滑动窗口裁剪：
        1. System 消息始终保留。
        2. 非 System 消息仅保留最近 max_rounds * 2 条（即最近 N 轮 user+assistant）。
        3. 若仍超过 Token 阈值，从最旧的非 System 消息开始逐条丢弃，直到低于阈值。

        reserved_tokens：为系统提示词（含上下文 XML）预留的 Token，从裁剪阈值中扣除。
        实际可用阈值 = max(0, token_threshold - reserved_tokens)。
        """
        if not messages:
            return []

        # 扣除为系统提示词预留的 Token，得到历史消息可用阈值
        effective_threshold = max(0, self._token_threshold - reserved_tokens)

        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]

        # 步骤 2：按轮数裁剪
        max_non_system = self._max_rounds * 2
        if len(non_system) > max_non_system:
            dropped = len(non_system) - max_non_system
            non_system = non_system[-max_non_system:]
            logger.debug(
                f"[Session] 按轮数裁剪: 丢弃最旧 {dropped} 条非System消息, "
                f"保留最近 {len(non_system)} 条"
            )

        trimmed = system_msgs + non_system

        # 步骤 3：按 Token 预算裁剪（使用扣除预留后的阈值）
        current_tokens = count_tokens(trimmed)
        if current_tokens > effective_threshold:
            logger.debug(
                f"[Session] Token 超限: 当前 {current_tokens} > 有效阈值 {effective_threshold} "
                f"(总阈值 {self._token_threshold} - 预留 {reserved_tokens}), "
                f"开始逐条丢弃最旧非System消息"
            )
            while current_tokens > effective_threshold and non_system:
                non_system.pop(0)
                trimmed = system_msgs + non_system
                current_tokens = count_tokens(trimmed)
            logger.debug(
                f"[Session] Token 裁剪完成: 剩余 {len(trimmed)} 条, "
                f"Token={current_tokens}"
            )

        return trimmed

    # ------------------------------------------------------------------
    # 内部辅助方法（需在持有 _lock 的情况下调用）
    # ------------------------------------------------------------------
    def _refresh_expiry_locked(self, session_id: str) -> None:
        """刷新会话过期时间。"""
        self._expiry[session_id] = datetime.utcnow() + timedelta(seconds=self._ttl)

    def _is_expired_locked(self, session_id: str) -> bool:
        """判断会话是否已过期。"""
        expiry = self._expiry.get(session_id)
        if expiry is None:
            return True
        return datetime.utcnow() > expiry

    def _delete_locked(self, session_id: str) -> None:
        """删除会话（内部方法，需持有锁）。"""
        self._store.pop(session_id, None)
        self._expiry.pop(session_id, None)
        logger.debug(f"[Session] 删除会话: {session_id}")

    @staticmethod
    def _msg_equal(a: dict, b: dict) -> bool:
        """判断两条消息是否相同（用于去重）。"""
        return a.get("role") == b.get("role") and a.get("content") == b.get("content")


# 全局单例
_session_service: Optional[SessionService] = None


def get_session_service() -> SessionService:
    """获取 SessionService 单例。"""
    global _session_service
    if _session_service is None:
        _session_service = SessionService()
    return _session_service
