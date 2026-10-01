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

    # S3 第 28-29 天：请求超时与 max_tokens 配置
    # 普通对话（chat 模式）使用较短超时，单文件生成（new 模式）使用更长超时，
    # 因为完整文件生成往往比对话需要更多推理时间。
    CHAT_TIMEOUT_SECONDS: float = 60.0       # 普通对话超时（秒）
    NEW_FILE_TIMEOUT_SECONDS: float = 120.0  # /new 单文件生成超时（秒）
    CHAT_MAX_TOKENS: int = 4096              # 普通对话默认 max_tokens
    NEW_FILE_MAX_TOKENS: int = 8192          # /new 单文件生成默认 max_tokens

    # S4 第 33-34 天：代码向量化配置
    # embedding_mode:
    #   - "local"  : 使用本地 sentence-transformers 模型（all-MiniLM-L6-v2，384 维）
    #   - "remote" : 调用远程 Embedding API（OpenAI text-embedding-3-small，1536 维）
    #   - "auto"   : 优先本地，加载失败或过慢时自动降级到远程
    EMBEDDING_MODE: str = "auto"
    EMBEDDING_LOCAL_MODEL: str = "all-MiniLM-L6-v2"   # 本地模型名（~80MB，CPU 可跑）
    # 本地模型加载超时（秒）。首次运行需从 HuggingFace 下载模型（~80MB），
    # 故默认设为 120s；模型缓存后加载通常 < 3s。
    EMBEDDING_LOCAL_LOAD_TIMEOUT: float = 120.0
    EMBEDDING_API_KEY: str = ""
    EMBEDDING_REMOTE_MODEL: str = "text-embedding-3-small"
    EMBEDDING_REMOTE_API_BASE: str = "https://api.openai.com/v1"
    EMBEDDING_REMOTE_BATCH_SIZE: int = 64
    EMBEDDING_CACHE_DB: str = ".ai_index/embeddings_cache.sqlite"  # 远程向量缓存库
    # 向量模型版本由后端自动检测（如 all-MiniLM-L6-v2-384 / text-embedding-3-small-1536），
    # 写入每个 Chunk 的 embedding_version 字段，用于维度迁移（S4 风险预警）。

    # S4 第 35-36 天：向量数据库（LanceDB）配置
    # LanceDB 为纯文件嵌入式存储，db_path 为目录路径；无需外部服务。
    VECTOR_STORE_DB_PATH: str = ".ai_index/lancedb"
    VECTOR_STORE_TABLE_NAME: str = "code_chunks"
    # 向量搜索默认返回 top_k
    VECTOR_SEARCH_TOP_K: int = 10

    # S4 第 37-38 天：全量索引 & 增量更新机制
    # 每处理多少个文件后批量提交向量（切片→向量化→删旧→插新），减少 IO 开销
    INDEX_BATCH_SIZE: int = 10
    # Redis 中索引进度 key 的前缀（完整 key = {prefix}:{job_id}）
    INDEX_REDIS_KEY_PREFIX: str = "index:status"
    # 索引进度在 Redis 中的过期时间（秒）
    INDEX_REDIS_TTL: int = 3600
    # 索引完成标记文件路径（相对工作区根目录），供插件检测当前仓库是否已索引
    INDEX_MARKER_FILE: str = ".ai_index/index_done"

    # S4 第 39-40 天：依赖关系图（Call Graph）配置
    # MVP 阶段采用 JSON 文件 + 内存缓存，避免引入 Neo4j 中间件
    # 全量索引结束后整体持久化；增量更新时局部修改后重新持久化
    DEPENDENCY_GRAPH_FILE: str = ".ai_index/dependency_graph.json"
    # 依赖图 BFS 查询默认深度（GET /v1/graph/related?depth=...）
    GRAPH_DEFAULT_DEPTH: int = 2
    # 依赖图 BFS 查询最大深度（防止全图遍历）
    GRAPH_MAX_DEPTH: int = 5

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT.lower() == "production"


settings = Settings()
