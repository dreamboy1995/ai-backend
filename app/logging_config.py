import logging

from app.middlewares.request_id import get_request_id_filter


def setup_logging():
    """
    配置日志，必须在所有模块日志调用之前执行。

    日志格式注入 request_id 字段（S2 第 19-20 天链路追踪）：
    - 通过 _RequestIdFilter 将 contextvars 中的 request_id 写入每条 LogRecord。
    - 不在请求上下文中时（启动期、后台清理任务）显示 "-"。
    - 所有日志（含中间件、路由、SSE 生成器、SessionService 等）统一带上 req=<id>。
    """
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | req=%(request_id)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    # 注入 request_id 字段，缺失时由 filter 设为 "-"
    handler.addFilter(get_request_id_filter())

    root = logging.getLogger()
    # 清理 basicConfig 或重复调用残留的 handler，避免日志重复输出
    root.handlers = [handler]
    root.setLevel(logging.INFO)
