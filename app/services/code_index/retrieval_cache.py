"""
检索结果缓存层（S5 第 49-50 天：性能压测 & Redis 缓存）

对 hybrid_search（向量 + BM25 + 符号 + RRF + Cross-Encoder）的结果做缓存，
降低重复查询的压力。当压测 P95 延迟超标时，缓存能显著减少重复计算。

核心策略（S5 关键技术预研与风险预警）：
  - 缓存粒度：对相同 Query（近义词归一化后）+ top_k 缓存检索结果 5 分钟。
  - 近义词归一化：小写化、去除首尾空白、折叠连续空白、去除末尾标点，
    使 "排序算法" / "排序算法。" / " 排序算法 " 命中同一缓存。
  - 存储后端：优先 Redis（与 redis_client 一致的降级策略），
    Redis 不可用时自动降级为内存 OrderedDict（带 TTL + LRU 淘汰）。
  - 失效策略：索引更新（文件增删改）时调用 invalidate() 清空缓存，
    避免返回过期结果。

与 redis_client.py 的关系：
  - 复用 get_redis_async() 获取 Redis 连接，不重复管理连接生命周期。
  - 序列化：hybrid_search 返回 List[dict]，用 JSON 序列化存入 Redis。
"""

import json
import logging
import re
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from app.config import settings

logger = logging.getLogger(__name__)


# ============================================================
# Query 归一化（近义词归一化）
# ============================================================

# 折叠连续空白字符
_WS_RE = re.compile(r"\s+")
# 去除末尾常见标点（中英文句号、问号、感叹号、逗号、分号）
_TRAILING_PUNCT_RE = re.compile(r"[。.？?！!，,；;]+$")


def normalize_query(query: str) -> str:
    """
    对用户 Query 做近义词归一化，使语义相同的查询命中同一缓存。

    规则：
      1. 去除首尾空白
      2. 折叠连续空白为单个空格
      3. 转小写（英文大小写不敏感）
      4. 去除末尾标点（避免 "排序算法" 与 "排序算法。" 被当作不同查询）

    Args:
        query: 原始用户查询

    Returns:
        归一化后的查询字符串
    """
    if not query:
        return ""
    q = query.strip()
    q = _WS_RE.sub(" ", q)
    q = q.lower()
    q = _TRAILING_PUNCT_RE.sub("", q)
    return q.strip()


def make_cache_key(query: str, top_k: int) -> str:
    """
    生成缓存键。

    键格式：retrieval:{top_k}:{normalized_query}
    top_k 纳入键中，因为不同 top_k 的检索结果不同。

    Args:
        query: 原始用户查询
        top_k: 检索返回数量

    Returns:
        缓存键字符串
    """
    return f"retrieval:{top_k}:{normalize_query(query)}"


# ============================================================
# 内存缓存降级实现（Redis 不可用时使用）
# ============================================================


class _MemoryCache:
    """
    内存 LRU + TTL 缓存（Redis 不可用时的降级方案）。

    特性：
      - 最大条目数限制（MAX_ENTRIES），超出时淘汰最久未访问的条目（LRU）
      - 每条目带 TTL，过期自动失效（惰性删除：get 时检查）
      - 线程安全（使用锁）
    """

    def __init__(self, max_entries: int = 512):
        self._max_entries = max_entries
        self._store: "OrderedDict[str, tuple]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[List[Dict[str, Any]]]:
        with self._lock:
            item = self._store.get(key)
            if item is None:
                return None
            value, expire_at = item
            if expire_at > 0 and time.time() > expire_at:
                # 已过期，删除
                self._store.pop(key, None)
                return None
            # 命中：移到末尾（LRU 最近使用）
            self._store.move_to_end(key)
            return value

    def set(self, key: str, value: List[Dict[str, Any]], ttl_seconds: int) -> None:
        with self._lock:
            expire_at = time.time() + ttl_seconds if ttl_seconds > 0 else 0
            self._store[key] = (value, expire_at)
            self._store.move_to_end(key)
            # 超出上限时淘汰最久未使用的（队首）
            while len(self._store) > self._max_entries:
                self._store.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def count(self) -> int:
        with self._lock:
            return len(self._store)


# ============================================================
# 检索缓存主类
# ============================================================


class RetrievalCache:
    """
    检索结果缓存。

    使用方式：
        cache = RetrievalCache()
        key = make_cache_key("排序算法", top_k=5)
        results = cache.get(key)
        if results is None:
            results = hybrid_search("排序算法", top_k=5)
            cache.set(key, results, ttl_seconds=300)
    """

    def __init__(
        self,
        enabled: bool = None,
        ttl_seconds: int = None,
        memory_max_entries: int = 512,
    ):
        try:
            self._enabled = (
                enabled if enabled is not None else settings.RETRIEVAL_CACHE_ENABLED
            )
            self._ttl = (
                ttl_seconds if ttl_seconds is not None else settings.RETRIEVAL_CACHE_TTL_SECONDS
            )
        except Exception:
            self._enabled = enabled if enabled is not None else True
            self._ttl = ttl_seconds if ttl_seconds is not None else 300

        self._memory = _MemoryCache(max_entries=memory_max_entries)
        self._lock = threading.Lock()
        self._redis_unavailable = False
        # Singleflight：per-key 锁，防止缓存击穿（同一 key 的并发请求只计算一次）
        self._key_locks: Dict[str, threading.Lock] = {}
        self._key_locks_lock = threading.Lock()

    # ------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def ttl(self) -> int:
        return self._ttl

    def get(self, key: str) -> Optional[List[Dict[str, Any]]]:
        """
        从缓存获取检索结果。

        优先查 Redis，Redis 不可用/未命中时查内存降级缓存。

        Args:
            key: make_cache_key() 生成的缓存键

        Returns:
            缓存命中时返回 List[dict]，未命中或缓存未启用时返回 None
        """
        if not self._enabled:
            return None

        # 1. 尝试 Redis
        if not self._redis_unavailable:
            try:
                from app.services.redis_client import get_redis_async
                redis = get_redis_async()
                if redis is not None:
                    # 同步包装：在调用线程中同步执行 redis 命令
                    # （hybrid_search 本身是同步函数，运行在线程池中）
                    import asyncio
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = None
                    if loop and loop.is_running():
                        # 已有运行中的事件循环，用 run_coroutine_threadsafe
                        raw = asyncio.run_coroutine_threadsafe(
                            redis.get(key), loop
                        ).result(timeout=1.0)
                    else:
                        raw = asyncio.run(redis.get(key))
                    if raw is not None:
                        try:
                            return json.loads(raw)
                        except (json.JSONDecodeError, TypeError):
                            logger.debug(f"[RetrievalCache] Redis 缓存反序列化失败: {key}")
            except Exception as e:
                # Redis 故障时标记为不可用，后续请求直接走内存缓存
                logger.debug(f"[RetrievalCache] Redis 读取失败，降级内存缓存: {e}")
                self._redis_unavailable = True

        # 2. 内存降级缓存
        return self._memory.get(key)

    def set(self, key: str, value: List[Dict[str, Any]]) -> None:
        """
        将检索结果写入缓存（Redis + 内存双写）。

        Args:
            key:   缓存键
            value: 检索结果（List[dict]，JSON 可序列化）
        """
        if not self._enabled:
            return

        # 1. 写 Redis（失败时静默降级，不影响主流程）
        if not self._redis_unavailable:
            try:
                from app.services.redis_client import get_redis_async
                redis = get_redis_async()
                if redis is not None:
                    import asyncio
                    serialized = json.dumps(value, ensure_ascii=False)
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = None
                    if loop and loop.is_running():
                        asyncio.run_coroutine_threadsafe(
                            redis.setex(key, self._ttl, serialized), loop
                        ).result(timeout=1.0)
                    else:
                        asyncio.run(redis.setex(key, self._ttl, serialized))
            except Exception as e:
                logger.debug(f"[RetrievalCache] Redis 写入失败，降级内存缓存: {e}")
                self._redis_unavailable = True

        # 2. 写内存降级缓存
        self._memory.set(key, value, self._ttl)

    # ------------------------------------------------------------
    # Singleflight：防缓存击穿
    # ------------------------------------------------------------

    def _get_key_lock(self, key: str) -> threading.Lock:
        """获取指定 key 的单飞锁（不存在则创建）。"""
        with self._key_locks_lock:
            lock = self._key_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._key_locks[key] = lock
            return lock

    def get_or_compute(
        self,
        key: str,
        compute_fn,
    ) -> List[Dict[str, Any]]:
        """
        带 Singleflight 的"缓存或计算"。

        防止缓存击穿：当多个并发线程请求同一 key 且缓存未命中时，
        只有一个线程执行 compute_fn，其余线程等待其结果，
        避免高并发下对同一 Query 重复执行昂贵的 Embedding + Rerank。

        流程：
          1. 查缓存，命中直接返回
          2. 获取该 key 的单飞锁
          3. 双重检查缓存（可能在等待锁期间已被其他线程写入）
          4. 仍未命中则执行 compute_fn，写入缓存后返回

        Args:
            key:        缓存键
            compute_fn: 缓存未命中时的计算函数，返回 List[dict]

        Returns:
            检索结果（来自缓存或 compute_fn）
        """
        # 1. 快速路径：缓存命中
        cached = self.get(key)
        if cached is not None:
            return cached

        # 2. 获取单飞锁（同一 key 的并发请求在此排队）
        key_lock = self._get_key_lock(key)
        with key_lock:
            # 3. 双重检查：等待锁期间可能已被其他线程写入缓存
            cached = self.get(key)
            if cached is not None:
                return cached
            # 4. 执行计算并写入缓存
            result = compute_fn()
            self.set(key, result)
            return result

    def invalidate(self) -> int:
        """
        清空所有检索缓存。

        在索引更新（文件增删改）后调用，避免返回过期的检索结果。

        Returns:
            清理的内存缓存条目数
        """
        cleared = 0
        # 清空内存缓存
        cleared = self._memory.count()
        self._memory.clear()
        # 尝试清空 Redis 中所有 retrieval:* 键
        if not self._redis_unavailable:
            try:
                from app.services.redis_client import get_redis_async
                redis = get_redis_async()
                if redis is not None:
                    import asyncio
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = None
                    if loop and loop.is_running():
                        keys = asyncio.run_coroutine_threadsafe(
                            redis.keys("retrieval:*"), loop
                        ).result(timeout=2.0)
                        if keys:
                            asyncio.run_coroutine_threadsafe(
                                redis.delete(*keys), loop
                            ).result(timeout=2.0)
                    else:
                        keys = asyncio.run(redis.keys("retrieval:*"))
                        if keys:
                            asyncio.run(redis.delete(*keys))
            except Exception as e:
                logger.debug(f"[RetrievalCache] Redis 缓存清理失败: {e}")
                self._redis_unavailable = True
        logger.info(f"[RetrievalCache] 缓存已清空（内存 {cleared} 条）")
        return cleared


# ============================================================
# 单例工厂
# ============================================================

_retrieval_cache: Optional[RetrievalCache] = None


def get_retrieval_cache() -> RetrievalCache:
    """获取 RetrievalCache 单例（配置从 settings 读取）。"""
    global _retrieval_cache
    if _retrieval_cache is None:
        _retrieval_cache = RetrievalCache()
    return _retrieval_cache
