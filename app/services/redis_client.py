"""
Redis 客户端模块（S3 第 21-22 天：限频与配额系统）

提供 Redis 连接管理，未配置 Redis 时自动降级为 None，
由调用方（RateLimiter / QuotaService）走内存降级逻辑。

设计说明：
- 与 SessionService 的 tiktoken 降级策略一致：依赖不可用时自动降级，服务不中断。
- Redis 连接延迟初始化（首次使用时建立），避免启动期阻塞。
- 提供 async 和 sync 两种访问方式：
  - async Redis（redis.asyncio）：供中间件 / 路由使用
  - sync Redis（redis.Redis）：供非异步场景使用（如 Lua 脚本注册）
"""

import logging
from typing import Optional

from app.config import settings

logger = logging.getLogger(__name__)

# 全局 Redis 客户端单例
_redis_async_client = None
_redis_sync_client = None
_redis_available: Optional[bool] = None


def _is_redis_enabled() -> bool:
    """判断是否启用 Redis。"""
    return settings.REDIS_ENABLED


def get_redis_async():
    """
    获取异步 Redis 客户端（redis.asyncio.Redis）。

    未启用 Redis 或连接失败时返回 None，调用方走内存降级逻辑。
    延迟初始化：首次调用时建立连接。
    """
    global _redis_async_client, _redis_available

    if not _is_redis_enabled():
        if _redis_available is None:
            logger.info("[RedisClient] REDIS_ENABLED=false，使用内存降级模式")
            _redis_available = False
        return None

    if _redis_async_client is not None:
        return _redis_async_client

    if _redis_available is False:
        # 之前已确认 Redis 不可用，不重复尝试
        return None

    # 延迟导入，未安装 redis 包时不影响启动
    try:
        import redis.asyncio as aioredis
    except ImportError:
        logger.warning("[RedisClient] redis 包未安装，使用内存降级模式")
        _redis_available = False
        return None

    try:
        _redis_async_client = aioredis.from_url(
            settings.REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=2.0,
            socket_timeout=2.0,
            retry_on_timeout=True,
        )
        _redis_available = True
        logger.info(f"[RedisClient] 异步 Redis 客户端已初始化: {settings.REDIS_URL}")
        return _redis_async_client
    except Exception as e:
        logger.warning(f"[RedisClient] Redis 连接失败，降级为内存模式: {e}")
        _redis_available = False
        return None


def get_redis_sync():
    """
    获取同步 Redis 客户端（redis.Redis），用于 Lua 脚本注册等同步场景。

    未启用或连接失败时返回 None。
    """
    global _redis_sync_client, _redis_available

    if not _is_redis_enabled():
        return None

    if _redis_sync_client is not None:
        return _redis_sync_client

    if _redis_available is False:
        return None

    try:
        import redis as sync_redis
    except ImportError:
        _redis_available = False
        return None

    try:
        _redis_sync_client = sync_redis.from_url(
            settings.REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=2.0,
            socket_timeout=2.0,
        )
        _redis_available = True
        logger.info(f"[RedisClient] 同步 Redis 客户端已初始化: {settings.REDIS_URL}")
        return _redis_sync_client
    except Exception as e:
        logger.warning(f"[RedisClient] 同步 Redis 连接失败，降级为内存模式: {e}")
        _redis_available = False
        return None


def is_redis_available() -> bool:
    """判断 Redis 是否可用（已初始化且连接成功）。"""
    if _redis_available is None:
        # 触发一次初始化尝试
        get_redis_sync()
    return _redis_available is True


async def close_redis():
    """关闭 Redis 连接，供应用关闭时调用。"""
    global _redis_async_client, _redis_sync_client
    if _redis_async_client is not None:
        try:
            await _redis_async_client.aclose()
        except Exception:
            pass
        _redis_async_client = None
    if _redis_sync_client is not None:
        try:
            _redis_sync_client.close()
        except Exception:
            pass
        _redis_sync_client = None
