import logging


def setup_logging():
    """配置日志，必须在所有模块日志调用之前执行。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
