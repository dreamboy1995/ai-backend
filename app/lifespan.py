import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.config import settings
from app.services.session import get_session_service

logger = logging.getLogger(__name__)


async def _session_cleanup_loop():
    """
    后台任务：定期清理过期的会话（S2 第 19-20 天）。

    内存实现的 TTL 不会像 Redis 那样在服务端自动过期，需要主动扫描清理，
    否则过期会话会一直占用内存（关闭 VS Code / 用户离开后无人触发被动清理）。

    每 SESSION_CLEANUP_INTERVAL_SECONDS 秒执行一次 SessionService.cleanup_expired()。
    """
    interval = settings.SESSION_CLEANUP_INTERVAL_SECONDS
    logger.info(f"[Lifespan] 启动会话清理后台任务，执行间隔 {interval}s")
    # 启动时先初始化 SessionService 单例（打印其配置日志，便于验收）
    get_session_service()
    while True:
        try:
            await asyncio.sleep(interval)
            removed = get_session_service().cleanup_expired()
            logger.debug(
                f"[Lifespan] 会话清理任务执行: 清理 {removed} 个, "
                f"剩余 {get_session_service().count()} 个"
            )
        except asyncio.CancelledError:
            logger.info("[Lifespan] 会话清理任务已取消")
            raise
        except Exception as e:
            # 后台任务异常不应导致服务退出，记录后继续下一轮
            logger.error(f"[Lifespan] 会话清理任务异常: {e}", exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting AI Backend application")
    cleanup_task = asyncio.create_task(_session_cleanup_loop())
    try:
        yield
    finally:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass
        logger.info("Shutting down AI Backend application")
