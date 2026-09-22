"""
配额服务（S3 第 21-22 天：限频与配额系统）

核心能力：
- 记录用户当日 Token 消耗量（Redis incrby / 内存 dict）。
- 提供用量查询接口所需数据：used_tokens, quota_limit, percentage。
- 每日自动重置（Redis key 设过期，内存按日期重置）。

设计说明：
- Redis 模式：key 为 quota:{user_id}:{date}，使用 incrby 累加，
  设置当日剩余秒数为 TTL，实现自动过期重置。
- 内存降级：dict 存储当日消耗，按日期判断是否需重置。
- user_id 取自 JWT 的 sub 字段的 SHA256 哈希（与 RateLimiter 一致）。
- /v1/user/usage 接口返回的数据结构遵循 S3 契约：
  { user_id, used_tokens_today, quota_limit_per_day, percentage, reset_at }
"""

import hashlib
import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict

from app.config import settings
from app.services.redis_client import get_redis_sync

logger = logging.getLogger(__name__)


def _hash_user_id(user_id: str) -> str:
    """对 user_id 做 SHA256 哈希，避免原始标识出现在 Redis key 中。"""
    return hashlib.sha256(user_id.encode()).hexdigest()[:32]


def _get_utc_date_str() -> str:
    """获取 UTC 日期字符串（YYYYMMDD），用于构建日级 Redis key。"""
    return datetime.utcnow().strftime("%Y%m%d")


def _get_reset_at_iso() -> str:
    """获取次日 UTC 0 点的 ISO 8601 时间戳（配额重置时间）。"""
    now = datetime.utcnow()
    tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return tomorrow.strftime("%Y-%m-%dT00:00:00Z")


def _get_seconds_until_midnight() -> int:
    """获取当前到 UTC 午夜剩余秒数（用于 Redis key TTL）。"""
    now = datetime.utcnow()
    tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    remaining = int((tomorrow - now).total_seconds())
    return max(remaining, 1)


class QuotaService:
    """
    配额服务（Redis / 内存双模式）。

    记录与查询用户每日 Token 消耗量。
    """

    def __init__(self, quota_limit_per_day: int = None):
        self._quota_limit = quota_limit_per_day or settings.QUOTA_LIMIT_PER_DAY
        self._redis = get_redis_sync()

        # 内存降级: user_hash -> { date: used_tokens }
        self._in_memory: Dict[str, Dict[str, int]] = {}
        self._lock = threading.Lock()

        if self._redis is not None:
            logger.info(
                f"[QuotaService] Redis 模式: 每日配额={self._quota_limit}"
            )
        else:
            logger.info(
                f"[QuotaService] 内存降级模式: 每日配额={self._quota_limit}"
            )

    def _redis_key(self, user_hash: str, date_str: str) -> str:
        """构建 Redis key: quota:{user_hash}:{date}"""
        return f"quota:{user_hash}:{date_str}"

    def record_usage(self, user_id: str, tokens: int) -> None:
        """
        记录用户 Token 消耗量。

        - Redis 模式: incrby 累加，并确保 TTL 指向 UTC 午夜。
        - 内存模式: 按日期累加，跨日自动重置。
        """
        if tokens <= 0:
            return

        user_hash = _hash_user_id(user_id)

        if self._redis is not None:
            self._record_redis(user_hash, tokens)
        else:
            self._record_in_memory(user_hash, tokens)

    def _record_redis(self, user_hash: str, tokens: int) -> None:
        """Redis 模式记录。"""
        date_str = _get_utc_date_str()
        key = self._redis_key(user_hash, date_str)
        try:
            pipe = self._redis.pipeline()
            pipe.incrby(key, tokens)
            # 仅当 key 为新创建时设置 TTL（nx=True 语义下 expire 会失败）
            # 使用 expire（幂等），确保 key 不泄漏
            pipe.expire(key, _get_seconds_until_midnight())
            pipe.execute()
        except Exception as e:
            logger.warning(
                f"[QuotaService] Redis 记录失败，降级为内存模式: {e}"
            )
            self._redis = None
            self._record_in_memory(user_hash, tokens)

    def _record_in_memory(self, user_hash: str, tokens: int) -> None:
        """内存模式记录。"""
        date_str = _get_utc_date_str()
        with self._lock:
            if user_hash not in self._in_memory:
                self._in_memory[user_hash] = {}
            user_data = self._in_memory[user_hash]

            # 跨日重置
            if date_str not in user_data:
                # 清理旧日期数据
                user_data.clear()
                user_data[date_str] = 0

            user_data[date_str] = user_data.get(date_str, 0) + tokens

    def get_usage(self, user_id: str) -> dict:
        """
        获取用户用量数据。

        返回 S3 契约约定的结构：
        {
            user_id: str,
            used_tokens_today: int,
            quota_limit_per_day: int,
            percentage: float,  # 0.0 ~ 1.0
            reset_at: str      # ISO 8601
        }
        """
        user_hash = _hash_user_id(user_id)

        if self._redis is not None:
            used = self._get_usage_redis(user_hash)
        else:
            used = self._get_usage_in_memory(user_hash)

        percentage = min(used / self._quota_limit, 1.0) if self._quota_limit > 0 else 0.0

        return {
            "user_id": user_hash,
            "used_tokens_today": used,
            "quota_limit_per_day": self._quota_limit,
            "percentage": round(percentage, 4),
            "reset_at": _get_reset_at_iso(),
        }

    def _get_usage_redis(self, user_hash: str) -> int:
        """Redis 模式查询。"""
        date_str = _get_utc_date_str()
        key = self._redis_key(user_hash, date_str)
        try:
            val = self._redis.get(key)
            return int(val) if val else 0
        except Exception as e:
            logger.warning(
                f"[QuotaService] Redis 查询失败，降级为内存模式: {e}"
            )
            self._redis = None
            return self._get_usage_in_memory(user_hash)

    def _get_usage_in_memory(self, user_hash: str) -> int:
        """内存模式查询。"""
        date_str = _get_utc_date_str()
        with self._lock:
            user_data = self._in_memory.get(user_hash)
            if user_data is None:
                return 0
            return user_data.get(date_str, 0)

    def cleanup_memory(self) -> int:
        """
        清理内存中过期的用户数据（供后台任务调用）。

        返回清理的用户数量。
        """
        date_str = _get_utc_date_str()
        removed = 0
        with self._lock:
            stale_keys = []
            for user_hash, user_data in self._in_memory.items():
                # 清理非当天的数据
                stale_dates = [d for d in user_data if d != date_str]
                for d in stale_dates:
                    del user_data[d]
                if not user_data:
                    stale_keys.append(user_hash)
            for key in stale_keys:
                del self._in_memory[key]
                removed += 1
        return removed


# 全局单例
_quota_service: Optional[QuotaService] = None


def get_quota_service() -> QuotaService:
    """获取 QuotaService 单例。"""
    global _quota_service
    if _quota_service is None:
        _quota_service = QuotaService()
    return _quota_service
