"""
S7 第 61-62 天：AgentSession 存储服务

将 AgentSession 持久化到 Redis（JSON 序列化），Redis 不可用时自动降级为内存存储。

对应 S7 风险预警：
  "用户中途关闭 VS Code：如果 Builder 执行到一半用户关了 VS Code，
   Agent 会话应该持久化在 Redis 中。下次打开插件时，检测到未完成的 Session，
   提示用户'有未完成的构建任务，是否继续？'。"

设计参考：app/services/code_index/index_service.py 的 Redis + 内存降级模式，
以及 app/services/session.py 的单例模式。
"""

import json
import logging
import threading
import time
from typing import Dict, Optional

from app.models.agent import AgentSession

logger = logging.getLogger(__name__)

# Redis key 前缀
AGENT_SESSION_KEY_PREFIX = "agent:session"
# 默认 TTL：24 小时（Agent 构建任务可能较长，给足过期时间）
DEFAULT_TTL_SECONDS = 24 * 3600


class AgentSessionStore:
    """
    AgentSession 存储（Redis + 内存降级）。

    线程安全：内存降级路径使用 _lock 保证并发安全；
    Redis 路径依赖 Redis 自身的原子性。
    """

    def __init__(self, ttl_seconds: int = DEFAULT_TTL_SECONDS):
        self._ttl = ttl_seconds
        self._mem_store: Dict[str, AgentSession] = {}
        self._mem_expiry: Dict[str, float] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # 内部：Redis 客户端（延迟导入，避免循环依赖）
    # ------------------------------------------------------------------

    @staticmethod
    def _get_redis():
        try:
            from app.services.redis_client import get_redis_sync
            return get_redis_sync()
        except Exception as e:
            logger.debug(f"[AgentSessionStore] 获取 Redis 客户端失败: {e}")
            return None

    @staticmethod
    def _key(session_id: str) -> str:
        return f"{AGENT_SESSION_KEY_PREFIX}:{session_id}"

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def save(self, session: AgentSession) -> None:
        """
        保存会话到 Redis（失败则降级到内存）。
        """
        payload = session.model_dump_json()
        r = self._get_redis()
        if r is not None:
            try:
                r.setex(self._key(session.session_id), self._ttl, payload)
                logger.debug(
                    f"[AgentSessionStore] 会话已写入 Redis: {session.session_id}"
                )
                return
            except Exception as e:
                logger.warning(
                    f"[AgentSessionStore] Redis 写入失败，降级到内存: {e}"
                )

        # 内存降级
        with self._lock:
            self._mem_store[session.session_id] = session
            self._mem_expiry[session.session_id] = time.time() + self._ttl
            logger.debug(
                f"[AgentSessionStore] 会话已写入内存（降级）: {session.session_id}"
            )

    def get(self, session_id: str) -> Optional[AgentSession]:
        """
        读取会话。优先 Redis，未命中则查内存。
        """
        r = self._get_redis()
        if r is not None:
            try:
                raw = r.get(self._key(session_id))
                if raw:
                    session = AgentSession.model_validate_json(raw)
                    return session
            except Exception as e:
                logger.warning(
                    f"[AgentSessionStore] Redis 读取失败，尝试内存降级: {e}"
                )

        # 内存降级
        with self._lock:
            session = self._mem_store.get(session_id)
            if session is None:
                return None
            # 过期检查
            expiry = self._mem_expiry.get(session_id, 0)
            if time.time() > expiry:
                self._mem_store.pop(session_id, None)
                self._mem_expiry.pop(session_id, None)
                logger.info(f"[AgentSessionStore] 内存会话已过期: {session_id}")
                return None
            return session

    def delete(self, session_id: str) -> None:
        """删除会话"""
        r = self._get_redis()
        if r is not None:
            try:
                r.delete(self._key(session_id))
            except Exception as e:
                logger.debug(f"[AgentSessionStore] Redis 删除失败: {e}")

        with self._lock:
            self._mem_store.pop(session_id, None)
            self._mem_expiry.pop(session_id, None)

    def exists(self, session_id: str) -> bool:
        """判断会话是否存在"""
        return self.get(session_id) is not None

    def refresh_ttl(self, session_id: str) -> None:
        """刷新会话 TTL（每次状态更新后调用，避免活跃会话过期）"""
        r = self._get_redis()
        if r is not None:
            try:
                r.expire(self._key(session_id), self._ttl)
            except Exception as e:
                logger.debug(f"[AgentSessionStore] Redis 刷新 TTL 失败: {e}")

        with self._lock:
            if session_id in self._mem_store:
                self._mem_expiry[session_id] = time.time() + self._ttl


# 全局单例
_store: Optional[AgentSessionStore] = None


def get_agent_session_store() -> AgentSessionStore:
    """获取 AgentSessionStore 单例。"""
    global _store
    if _store is None:
        _store = AgentSessionStore()
    return _store
