import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi import HTTPException
import uvicorn

from app.logging_config import setup_logging
from app.config import settings
from app.lifespan import lifespan
from app.middlewares.auth import auth_middleware
from app.middlewares.request_id import request_id_middleware
from app.exception_handlers import (
    request_validation_exception_handler,
    http_exception_handler,
    unhandled_exception_handler,
)
from app.api.chat import router as chat_router
from app.api.auth import router as auth_router
from app.api.files import router as files_router
from app.api.search import router as search_router
from app.api.terminal import router as terminal_router

# 日志配置必须在所有日志调用之前执行
setup_logging()
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """创建并配置 FastAPI 应用"""
    # 根据环境决定是否启用文档
    docs_url = None if settings.is_production else "/docs"
    redoc_url = None if settings.is_production else "/redoc"
    openapi_url = None if settings.is_production else "/openapi.json"

    app = FastAPI(
        title="AI Backend",
        description="FastAPI + ZAI 后端服务",
        version="0.1.0",
        docs_url=docs_url,
        redoc_url=redoc_url,
        openapi_url=openapi_url,
        lifespan=lifespan,
    )

    # CORS 中间件 - 生产环境使用具体域名
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # JWT 认证中间件
    app.middleware("http")(auth_middleware)
    # X-Request-ID 链路追踪中间件（S2 第 19-20 天）
    # 注：app.middleware("http") 后注册的为最外层，故 request_id 先于 auth 执行，
    # 使 auth 中间件的日志也能带上 request_id。
    app.middleware("http")(request_id_middleware)

    # 全局异常处理器
    app.add_exception_handler(RequestValidationError, request_validation_exception_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)

    # 注册路由
    app.include_router(auth_router, tags=["认证"])
    app.include_router(chat_router, prefix="/v1", tags=["聊天"])
    app.include_router(files_router, prefix="/v1/files", tags=["文件操作"])
    app.include_router(search_router, prefix="/v1/search", tags=["代码搜索"])
    app.include_router(terminal_router, prefix="/v1/terminal", tags=["终端"])

    # 基础路由
    @app.get("/", tags=["默认"])
    async def root():
        return {"message": "AI Backend is running"}

    @app.get("/health", tags=["健康检查"])
    async def health():
        return {
            "status": "ok",
            "port": settings.PORT,
        }

    # 调试路由 - 仅非生产环境注册
    if not settings.is_production:
        from app.api.debug import router as debug_router
        app.include_router(debug_router, tags=["调试"])

    return app


app = create_app()


def check_port_available(host: str, port: int) -> bool:
    """检查端口是否可用"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


if __name__ == "__main__":
    if not check_port_available(settings.HOST, settings.PORT):
        logger.error(f"端口 {settings.PORT} 已被占用，请检查是否有其他进程占用或修改 .env 中的 PORT 配置")
        exit(1)
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
    )
