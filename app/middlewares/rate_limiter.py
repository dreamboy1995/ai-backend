"""
限频中间件（S3 第 21-22 天：限频与配额系统）

核心能力：
- 按 user_id 滑动窗口限频：每分钟 20 次，每天 500 次。
- 超限时返回 HTTP 429，Response Header 携带 X-RateLimit-Reset（剩余重置秒数）。
- Redis 模式：使用 Lua 脚本保证「清理过期 + 计数 + 写入」的原子性。
- 内存降级模式：使用 threading.Lock 保证并发安全（与 SessionService 一致）。

设计说明：
- user_id 取自 JWT 的 sub 字段（由 auth 中间件写入 request.state.user_payload）。
- 中间件注册顺序：rate_limiter 为最内层（先于 auth 注册），
  执行顺序为 request_id -> auth -> rate_limiter -> handler，
  保证 auth 已校验 token 并写入 user_payload。
- 仅对需要限频的路径（/v1/ 前缀）生效，公开路径与 auth 接口跳过。
- Redis 不可用时自动降级为内存滑动窗口，不影响服务可用性。
"""

import logging
import threading
import time
import uuid
from collections import deque
from typing import Optional, Tuple

from fastapi import Request, status
from fastapi.responses import JSONResponse

from app.config import settings
from app.services.redis_client import get_redis_sync, is_redis_available

logger = logging.getLogger(__name__)

# Response Header 名称
RATE_LIMIT_RESET_HEADER = "X-RateLimit-Reset"
RATE_LIMIT_REMAINING_HEADER = "X-RateLimit-Remaining"
RATE_LIMIT_LIMIT_HEADER = "X-RateLimit-Limit"

# 需要限频的路径前缀（仅对 /v1/ 下的业务接口限频）
RATE_LIMITED_PREFIX = "/v1/"

# 不限频的路径（状态查询等轻量接口，不应消耗用户的请求配额）
RATE_LIMIT_EXEMPT_PATHS = {"/v1/user/usage"}

# 窗口定义：(窗口秒数, 最大请求数, 配置项)
_WINDOW_MINUTE = 60       # 每分钟窗口
_WINDOW_DAY = 86400       # 每天窗口（24h）

# 滑动窗口 Lua 脚本（Redis 模式）
# 使用 Sorted Set 实现滑动窗口：
# - score 为请求时间戳（秒），member 为唯一标识
# - 先移除窗口外的旧记录，再计数，未超限则写入
# 返回: {是否允许(1/0), 剩余请求数, 重置剩余秒数}
_SLIDING_WINDOW_LUA = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local max_requests = tonumber(ARGV[3])
local member = ARGV[4]

-- 移除窗口外的记录
local cutoff = now - window
redis.call('ZREMRANGEBYSCORE', key, 0, cutoff)

-- 当前窗口内请求数
local count = redis.call('ZCARD', key)

if count >= max_requests then
    -- 超限：计算最早记录的过期时间（即重置时间）
    local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
    local reset_seconds = 0
    if oldest and #oldest >= 2 then
        local oldest_score = tonumber(oldest[2])
        reset_seconds = math.ceil(oldest_score + window - now)
        if reset_seconds < 0 then reset_seconds = 0 end
    end
    return {0, 0, reset_seconds}
end

-- 未超限：写入当前请求
redis.call('ZADD', key, now, member)
-- 设置 key 过期时间（窗口大小 + 缓冲），避免内存泄漏
redis.call('EXPIRE', key, window + 10)

local remaining = max_requests - count - 1
-- 计算重置时间（最早记录的过期时间）
local reset_seconds = window
local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
if oldest and #oldest >= 2 then
    local oldest_score = tonumber(oldest[2])
    reset_seconds = math.ceil(oldest_score + window - now)
    if reset_seconds < 0 then reset_seconds = 0 end
end

return {1, remaining, reset_seconds}
"""


class InMemorySlidingWindow:
    """
    内存滑动窗口（Redis 不可用时的降级实现）。

    使用 deque 存储请求时间戳，保证滑动窗口语义。
    线程安全（threading.Lock），与 SessionService 一致的并发策略。
    """

    def __init__(self):
        # key -> deque[float]（请求时间戳列表）
        self._windows: dict[str, deque] = {}
        self._lock = threading.Lock()

    def check_and_record(
        self, key: str, window_seconds: int, max_requests: int, now: Optional[float] = None
    ) -> Tuple[bool, int, int]:
        """
        检查并记录请求。

        返回: (allowed, remaining, reset_seconds)
        - allowed: 是否允许通过
        - remaining: 剩余请求数
        - reset_seconds: 重置剩余秒数
        """
        if now is None:
            now = time.time()

        with self._lock:
            if key not in self._windows:
                self._windows[key] = deque()

            dq = self._windows[key]
            cutoff = now - window_seconds

            # 移除窗口外的旧记录
            while dq and dq[0] <= cutoff:
                dq.popleft()

            count = len(dq)
            if count >= max_requests:
                # 超限，计算重置时间
                reset_seconds = 0
                if dq:
                    reset_seconds = int(dq[0] + window_seconds - now) + 1
                    if reset_seconds < 0:
                        reset_seconds = 0
                return False, 0, reset_seconds

            # 未超限，记录当前请求
            dq.append(now)
            remaining = max_requests - count - 1

            # 重置时间 = 最早记录 + 窗口 - 现在
            reset_seconds = int(dq[0] + window_seconds - now) + 1
            if reset_seconds < 0:
                reset_seconds = 0

            return True, remaining, reset_seconds

    def cleanup(self):
        """清理所有空的窗口（供后台任务调用）。"""
        now = time.time()
        with self._lock:
            empty_keys = []
            for key, dq in self._windows.items():
                # 移除过期记录
                cutoff = now - _WINDOW_DAY  # 保留天窗口的数据
                while dq and dq[0] <= cutoff:
                    dq.popleft()
                if not dq:
                    empty_keys.append(key)
            for key in empty_keys:
                del self._windows[key]


class RateLimiter:
    """
    限频器（Redis / 内存双模式）。

    对每个 user_id 维护两个滑动窗口：
    - 分钟窗口：每分钟 max_per_minute 次
    - 天窗口：每天 max_per_day 次
    任一窗口超限即返回 429。
    """

    def __init__(
        self,
        max_per_minute: int = None,
        max_per_day: int = None,
    ):
        self._max_per_minute = max_per_minute or settings.RATE_LIMIT_PER_MINUTE
        self._max_per_day = max_per_day or settings.RATE_LIMIT_PER_DAY

        # Redis Lua 脚本 SHA（注册后缓存，避免重复传输脚本）
        self._lua_sha: Optional[str] = None
        self._redis = get_redis_sync()

        # 内存降级
        self._in_memory = InMemorySlidingWindow()

        if self._redis is not None:
            self._register_lua_script()
            logger.info(
                f"[RateLimiter] Redis 模式: 分钟限={self._max_per_minute}, "
                f"天限={self._max_per_day}"
            )
        else:
            logger.info(
                f"[RateLimiter] 内存降级模式: 分钟限={self._max_per_minute}, "
                f"天限={self._max_per_day}"
            )

    def _register_lua_script(self):
        """注册 Lua 脚本到 Redis，缓存 SHA。"""
        if self._redis is None:
            return
        try:
            self._lua_sha = self._redis.script_load(_SLIDING_WINDOW_LUA)
            logger.info(f"[RateLimiter] Lua 脚本已注册, SHA={self._lua_sha[:16]}...")
        except Exception as e:
            logger.warning(f"[RateLimiter] Lua 脚本注册失败，降级为内存模式: {e}")
            self._redis = None
            self._lua_sha = None

    def _check_redis(
        self, key: str, window_seconds: int, max_requests: int, now: float
    ) -> Tuple[bool, int, int]:
        """Redis 模式检查（使用 Lua 脚本保证原子性）。"""
        member = f"{now}:{uuid.uuid4().hex}"
        try:
            result = self._redis.evalsha(
                self._lua_sha,
                1,
                key,
                str(now),
                str(window_seconds),
                str(max_requests),
                member,
            )
            # result = [allowed, remaining, reset_seconds]
            allowed = bool(result[0])
            remaining = int(result[1])
            reset_seconds = int(result[2])
            return allowed, remaining, reset_seconds
        except Exception as e:
            logger.warning(
                f"[RateLimiter] Redis evalsha 失败，降级为内存模式: {e}"
            )
            # Redis 故障，降级为内存
            self._redis = None
            self._lua_sha = None
            return self._check_in_memory(key, window_seconds, max_requests, now)

    def _check_in_memory(
        self, key: str, window_seconds: int, max_requests: int, now: float
    ) -> Tuple[bool, int, int]:
        """内存模式检查。"""
        return self._in_memory.check_and_record(
            key, window_seconds, max_requests, now
        )

    def check(self, user_id: str) -> Tuple[bool, int, int, Optional[int]]:
        """
        检查 user_id 是否被限频。

        返回: (allowed, remaining, reset_seconds, limited_window)
        - allowed: 是否允许通过
        - remaining: 剩余请求数（取两个窗口的最小值）
        - reset_seconds: 重置剩余秒数（取触发的窗口）
        - limited_window: 被限频的窗口标识（1=分钟, 2=天, None=未限频）
        """
        now = time.time()

        # Redis key 前缀
        minute_key = f"rate_limit:{user_id}:minute"
        day_key = f"rate_limit:{user_id}:day"

        if self._redis is not None and self._lua_sha is not None:
            # Redis 模式：先检查分钟窗口，再检查天窗口
            m_allowed, m_remaining, m_reset = self._check_redis(
                minute_key, _WINDOW_MINUTE, self._max_per_minute, now
            )
            if not m_allowed:
                return False, 0, m_reset, 1

            d_allowed, d_remaining, d_reset = self._check_redis(
                day_key, _WINDOW_DAY, self._max_per_day, now
            )
            if not d_allowed:
                return False, 0, d_reset, 2

            remaining = min(m_remaining, d_remaining)
            # 非限频请求显示分钟窗口的重置时间（更接近下次配额恢复的实际时间）
            reset_seconds = m_reset if m_reset > 0 else d_reset
            return True, remaining, reset_seconds, None
        else:
            # 内存模式
            m_allowed, m_remaining, m_reset = self._check_in_memory(
                minute_key, _WINDOW_MINUTE, self._max_per_minute, now
            )
            if not m_allowed:
                return False, 0, m_reset, 1

            d_allowed, d_remaining, d_reset = self._check_in_memory(
                day_key, _WINDOW_DAY, self._max_per_day, now
            )
            if not d_allowed:
                return False, 0, d_reset, 2

            remaining = min(m_remaining, d_remaining)
            reset_seconds = m_reset if m_reset > 0 else d_reset
            return True, remaining, reset_seconds, None


# 全局单例
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """获取 RateLimiter 单例。"""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter


def _get_user_id(request: Request) -> Optional[str]:
    """
    从 request.state 获取 user_id（由 auth 中间件写入）。

    user_id 取自 JWT 的 sub 字段。
    若 auth 中间件未写入（如公开路径），返回 None。
    """
    payload = getattr(request.state, "user_payload", None)
    if payload is None:
        return None
    user_id = payload.get("sub")
    if user_id:
        # 使用哈希值作为 Redis key 的一部分，避免原始 api_key 暴露在 key 中
        import hashlib
        return hashlib.sha256(user_id.encode()).hexdigest()[:32]
    return None


async def rate_limiter_middleware(request: Request, call_next):
    """
    限频中间件。

    执行顺序：request_id -> auth -> rate_limiter -> handler
    （本中间件先于 auth 注册，为最内层，在 auth 之后执行）

    仅对 /v1/ 前缀路径限频，公开路径与 auth 接口跳过。
    """
    path = request.url.path

    # 仅对 /v1/ 前缀的业务接口限频；排除状态查询等轻量接口
    if not path.startswith(RATE_LIMITED_PREFIX) or path in RATE_LIMIT_EXEMPT_PATHS:
        return await call_next(request)

    # 获取 user_id（由 auth 中间件写入 request.state）
    user_id = _get_user_id(request)
    if not user_id:
        # auth 中间件未通过（会返回 401），不限频
        return await call_next(request)

    limiter = get_rate_limiter()
    allowed, remaining, reset_seconds, limited_window = limiter.check(user_id)

    if not allowed:
        logger.warning(
            f"[RateLimiter] 限频触发: user_id={user_id[:8]}..., "
            f"path={path}, window={'minute' if limited_window == 1 else 'day'}, "
            f"reset={reset_seconds}s"
        )
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "code": 429,
                "message": "请求频率超限，请稍后再试",
                "path": path,
            },
            headers={
                RATE_LIMIT_RESET_HEADER: str(reset_seconds),
                RATE_LIMIT_REMAINING_HEADER: "0",
                RATE_LIMIT_LIMIT_HEADER: str(
                    settings.RATE_LIMIT_PER_MINUTE
                    if limited_window == 1
                    else settings.RATE_LIMIT_PER_DAY
                ),
                "Retry-After": str(reset_seconds),
            },
        )

    # 未被限频，继续执行，并在响应头中附带限频信息
    response = await call_next(request)
    response.headers[RATE_LIMIT_REMAINING_HEADER] = str(remaining)
    response.headers[RATE_LIMIT_LIMIT_HEADER] = str(settings.RATE_LIMIT_PER_MINUTE)
    if reset_seconds > 0:
        response.headers[RATE_LIMIT_RESET_HEADER] = str(reset_seconds)
    return response
