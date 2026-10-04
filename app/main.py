import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi import HTTPException
import uvicorn

from app.logging_config import setup_logging
from app.config import settings
from app.lifespan import lifespan
from app.middlewares.auth import auth_middleware
from app.middlewares.request_id import request_id_middleware
from app.middlewares.rate_limiter import rate_limiter_middleware
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
from app.api.user import router as user_router
from app.api.models import router as models_router
from app.api.index import router as index_router
from app.api.graph import router as graph_router
from app.api.symbols import router as symbols_router
from app.api.cue import router as cue_router
from app.api.agent import router as agent_router
from app.api.tool import router as tool_router
from app.api.stream import router as stream_router
from app.api.sandbox import router as sandbox_router

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

    # S6 第 55-56 天：Gzip 压缩中间件（风险预警应对）
    # 多文件修改场景下 Diff 数据可能达到数 MB，开启 Gzip 可显著压缩 SSE 包体积。
    # minimum_size=1024 表示仅压缩超过 1KB 的响应，避免小响应的压缩开销。
    # GZipMiddleware 对 StreamingResponse（SSE）同样生效，会逐块压缩流内容。
    app.add_middleware(GZipMiddleware, minimum_size=1024)

    # 中间件注册顺序说明：
    # app.middleware("http") 先注册者为最内层（后执行），后注册者为最外层（先执行）。
    # 期望执行顺序：request_id -> auth -> rate_limiter -> handler
    #   1. request_id 最先执行，使所有后续中间件日志带上 request_id
    #   2. auth 校验 JWT 并将 payload 存入 request.state.user_payload
    #   3. rate_limiter 从 request.state 读取 user_id 做限频（S3 第 21-22 天）
    # 因此注册顺序：rate_limiter（最内） -> auth -> request_id（最外）

    # S3 第 21-22 天：限频中间件（最内层，在 auth 之后执行）
    app.middleware("http")(rate_limiter_middleware)
    # JWT 认证中间件
    app.middleware("http")(auth_middleware)
    # X-Request-ID 链路追踪中间件（S2 第 19-20 天）
    app.middleware("http")(request_id_middleware)

    # 全局异常处理器
    app.add_exception_handler(RequestValidationError, request_validation_exception_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)

    # 注册路由
    app.include_router(auth_router, tags=["认证"])
    app.include_router(chat_router, prefix="/v1/chat", tags=["聊天"])
    app.include_router(files_router, prefix="/v1/files", tags=["文件操作"])
    app.include_router(search_router, prefix="/v1/search", tags=["代码搜索"])
    app.include_router(terminal_router, prefix="/v1/terminal", tags=["终端"])
    # S3 第 21-22 天：用户用量查询接口
    app.include_router(user_router, prefix="/v1/user", tags=["用户"])
    # S3 第 23-24 天：模型列表接口
    app.include_router(models_router, prefix="/v1/models", tags=["模型"])
    # S4 第 31-32 天：代码索引控制接口
    app.include_router(index_router, prefix="/v1/index", tags=["代码索引"])
    # S4 第 39-40 天：依赖关系图查询接口
    app.include_router(graph_router, prefix="/v1/graph", tags=["依赖图"])
    # S5 第 43-44 天：符号实时补全接口（# 输入触发）
    app.include_router(symbols_router, prefix="/v1/symbols", tags=["符号搜索"])
    # S6 第 57-58 天：Cue 编辑位置预测接口（启发式规则后端辅助）
    app.include_router(cue_router, prefix="/v1/cue", tags=["Cue 预测"])
    # S7 第 61-62 天：Agent 任务规划与执行接口（状态机骨架）
    app.include_router(agent_router, prefix="/v1/agent", tags=["Agent"])
    # S8 第 71-72 天：工具执行与确认接口（MCP 协议适配层）
    app.include_router(tool_router, prefix="/v1/tool", tags=["工具执行"])
    # S8 第 75-76 天：终端日志流式接口（WebSocket）
    # 路径前缀 /v1，使最终 URL 为 /v1/agent/stream/{session_id}，
    # 与 Sprint_8.md 关键接口定义一致。
    app.include_router(stream_router, prefix="/v1", tags=["流式日志"])
    # S8 第 79-80 天：沙箱状态接口
    app.include_router(sandbox_router, prefix="/v1/sandbox", tags=["沙箱"])

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
