import logging
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    PORT: int = 3000
    HOST: str = "127.0.0.1"
    ZAI_API_KEY: str  # 必填，无默认值，缺失时 Pydantic 启动即报错
    JWT_SECRET: str  # 必填，无默认值
    JWT_ALGORITHM: str = "HS256"
    ENVIRONMENT: str = "development"
    RELOAD: bool = False
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    # 会话管理相关配置（第 11-12 天：后端会话管理 & 历史记忆）
    SESSION_TTL_SECONDS: int = 3600       # 会话过期时间，默认 1 小时
    SESSION_TOKEN_BUDGET: int = 8000      # 总 Token 预算（为模型预留余量）
    SESSION_TOKEN_MARGIN: float = 0.2     # Token 余量比例，触发裁剪阈值 = budget * (1 - margin)
    SESSION_MAX_ROUNDS: int = 5           # 保留的最近对话轮数（1 轮 = user + assistant）

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT.lower() == "production"


settings = Settings()
