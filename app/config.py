import logging
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    PORT: int = 3000
    HOST: str = "127.0.0.1"
    ZAI_API_KEY: str  # 必填，无默认值，缺失时 Pydantic 启动即报错
    JWT_SECRET: str  # 必填，无默认值
    JWT_ALGORITHM: str = "HS256"

    # S3 第 23-24 天：多厂商适配器 API Key
    # 各厂商 Key 可选配置；未配置的厂商对应模型仍会在 /v1/models 返回，
    # 但实际调用时会返回服务不可用错误
    DEEPSEEK_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    ENVIRONMENT: str = "development"
    RELOAD: bool = False
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    # 会话管理相关配置（第 11-12 天：后端会话管理 & 历史记忆）
    SESSION_TTL_SECONDS: int = 3600       # 会话过期时间，默认 1 小时（S2 第 19-20 天要求 TTL=1 小时）
    SESSION_TOKEN_BUDGET: int = 8000      # 总 Token 预算（为模型预留余量）
    SESSION_TOKEN_MARGIN: float = 0.2     # Token 余量比例，触发裁剪阈值 = budget * (1 - margin)
    SESSION_MAX_ROUNDS: int = 5           # 保留的最近对话轮数（1 轮 = user + assistant）
    # 主动清理过期会话的后台任务执行间隔（S2 第 19-20 天：会话过期机制）
    # 内存实现不像 Redis 那样自动过期，需要定期扫描清理，避免过期会话占用内存
    SESSION_CLEANUP_INTERVAL_SECONDS: int = 300

    # 上下文拼装相关配置（第 15-16 天：后端上下文拼装 & 系统提示词工程）
    SESSION_CONTEXT_TOKEN_RATIO: float = 0.5  # 上下文（System Prompt + 文件内容）占用 Token 阈值的比例
    # 上下文预算 = 阈值(6400) * 0.5 = 3200，剩余 3200 留给对话历史

    # Redis 配置（S3 第 21-22 天：限频与配额系统）
    # 未配置 Redis 时自动降级为内存实现（与 SessionService 一致的降级策略）
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_ENABLED: bool = False  # 默认关闭，设为 True 时启用 Redis

    # 限频配置（S3 第 21-22 天：按 user_id 滑动窗口限频）
    RATE_LIMIT_PER_MINUTE: int = 20   # 每分钟请求上限
    RATE_LIMIT_PER_DAY: int = 500     # 每天请求上限

    # 配额配置（S3 第 21-22 天：每日 Token 消耗配额）
    QUOTA_LIMIT_PER_DAY: int = 100000  # 每日 Token 配额上限

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT.lower() == "production"


settings = Settings()
