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

    # S6 第 51-52 天：Inline Chat 模式超时与 max_tokens 配置
    # Inline 模式下 AI 需要输出修改后的完整代码，类似 /new 单文件生成，
    # 故使用较长超时和较大 max_tokens，确保完整代码能输出完毕。
    INLINE_CHAT_TIMEOUT_SECONDS: float = 120.0  # Inline Chat 超时（秒）
    INLINE_CHAT_MAX_TOKENS: int = 8192          # Inline Chat 默认 max_tokens

    # S6 第 59-60 天：多文件 JSON 输出超时保护
    # 当 response_format=json_object（多文件修改场景）时，模型需要同时思考
    # 多个文件的修改，容易出现"思考时间过长"导致用户长时间等待。
    # 为避免用户无感知地等待，对 JSON 模式单独设置更短的超时（30 秒），
    # 超时后强制终止并返回友好提示 "生成时间过长，请简化需求重试"。
    # 该超时覆盖 mode 维度的超时（chat=60 / new=120 / inline=120），
    # 即只要开启 JSON Mode，就以本配置为准。
    JSON_MODE_TIMEOUT_SECONDS: float = 30.0     # 多文件 JSON 输出超时（秒）
    JSON_MODE_TIMEOUT_MESSAGE: str = "生成时间过长，请简化需求重试"  # 超时友好提示

    # S4 第 33-34 天：代码向量化配置
    # embedding_mode:
    #   - "local"  : 使用本地 sentence-transformers 模型（all-MiniLM-L6-v2，384 维）
    #   - "remote" : 调用远程 Embedding API（OpenAI text-embedding-3-small，1536 维）
    #   - "auto"   : 优先本地，加载失败或过慢时自动降级到远程
    EMBEDDING_MODE: str = "local"
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
    # 符号表持久化文件路径（相对工作区根目录）
    # 解决后端重启后内存符号表丢失导致 /v1/symbols/search 返回空的问题。
    # 索引完成后整体持久化；增量更新时重新写入；启动时从该文件恢复（含 mtime 陈旧校验）。
    SYMBOLS_FILE: str = ".ai_index/symbols.json"
    # 依赖图 BFS 查询默认深度（GET /v1/graph/related?depth=...）
    GRAPH_DEFAULT_DEPTH: int = 2
    # 依赖图 BFS 查询最大深度（防止全图遍历）
    GRAPH_MAX_DEPTH: int = 5

    # S5 第 41-42 天：BM25 关键词检索配置
    # BM25 索引序列化文件路径（pickle），避免每次启动都重建
    BM25_INDEX_PATH: str = ".ai_cache/bm25_index.pkl"
    # BM25 检索默认返回 top_k
    BM25_DEFAULT_TOP_K: int = 20
    # 是否跳过测试文件（路径含 test/ 或文件名以 test_ 开头），
    # 降低大仓库 BM25 词袋矩阵的磁盘占用（S5 风险预警）
    BM25_SKIP_TEST_FILES: bool = True
    # 仅索引最近 N 个月修改过的文件（0 表示不限制）。
    # 大型仓库（10万+文件）的 BM25 词袋矩阵可能膨胀到数 GB，
    # 通过限制时间窗口控制索引规模（S5 风险预警）。
    BM25_RECENT_MONTHS: int = 0
    # BM25Okapi 的 k1 参数（词频饱和度，经典值 1.5）
    BM25_K1: float = 1.5
    # BM25Okapi 的 b 参数（文档长度归一化，经典值 0.75）
    BM25_B: float = 0.75

    # S5 第 43-44 天：三路检索融合 + RRF + Cross-Encoder 重排序配置
    # RRF（倒数排名融合）公式的 k 值：RRF_score(doc) = Σ 1 / (k + rank_i(doc))
    # k=60 为经典值；k 越大，多路命中加成越明显，k 越小排名靠前加成越强
    RRF_K: int = 60
    # RRF 融合后保留的候选数（来自向量/BM25/符号三路融合 Top-N）
    # S5 任务文档要求 Top-30 候选输入 Cross-Encoder
    RRF_CANDIDATE_K: int = 30
    # 符号精确检索返回 Top-K（S5 任务文档：Top-10）
    SYMBOL_SEARCH_TOP_K: int = 10
    # 混合检索（hybrid_search）最终返回数（注入 Prompt 的片段数）
    HYBRID_SEARCH_TOP_K: int = 5

    # Cross-Encoder 重排序模型（约 90MB，CPU 可跑）
    # cross-encoder/ms-marco-MiniLM-L-6-v2 是 MS-MARCO 排行榜轻量级基线，
    # 对代码场景虽非最优，但作为 P2 阶段 MVP 足够；后续可换为
    # cross-encoder/ms-code-mistral-7b-v2 等（需 GPU）
    RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # 重排序超时熔断阈值（毫秒）。超过则跳过重排序，直接返回 RRF 结果
    # S5 任务文档要求 300ms；CPU 上对 30 候选逐对推理可能 1s+，故默认 300ms 触发熔断
    RERANK_TIMEOUT_MS: int = 300
    # 重排序候选数上限（性能保护）。S5 风险预警：Top-30 在 CPU 上慢，
    # 限定到 ≤ 10 个候选重排序，平衡延迟与精度
    RERANK_MAX_CANDIDATES: int = 10
    # 是否启用重排序（False 时仅走 RRF，跳过 Cross-Encoder，用于压测/降级）
    RERANK_ENABLED: bool = True

    # S5 第 49-50 天：检索结果缓存（Redis + 内存降级）
    # 对 hybrid_search 结果做缓存，降低重复查询压力。
    # 缓存键 = retrieval:{top_k}:{normalize_query(query)}，TTL 5 分钟。
    # Redis 不可用时自动降级为内存 OrderedDict（LRU + TTL）。
    # 索引更新（文件增删改）时调用 RetrievalCache.invalidate() 清空缓存。
    RETRIEVAL_CACHE_ENABLED: bool = True
    RETRIEVAL_CACHE_TTL_SECONDS: int = 300  # 5 分钟

    # S6 第 57-58 天：Cue 编辑位置预测配置
    # 单次 /v1/cue/suggest 返回的最大建议数量上限。
    # 风险预警（Cue 规则误报）：启发式规则必然有误报，限制返回数量
    # 避免在编辑器上渲染过多灰色箭头干扰用户（弱视觉设计由插件实现）。
    CUE_SUGGEST_MAX: int = 20

    # S5 第 47-48 天：智能上下文组装器配置
    # 上下文组装器对 hybrid_search 召回的 Top-K Chunk 做智能压缩与 Token 预算控制。
    #
    # 动态 Token 预算（S5 风险预警：token 预算不能写死，必须根据模型动态调整）：
    #   组装器预算 = min(model_context_window * CONTEXT_ASSEMBLY_FILL_RATIO,
    #                    CONTEXT_ASSEMBLY_MAX_BUDGET)
    #   - fill_ratio=0.7：留 30% 给对话历史和模型输出（S5 风险预警建议）
    #   - max_budget=8000：硬上限，避免超大模型下上下文膨胀导致推理变慢/成本飙升
    #   若模型上下文窗口未知（不在 MODEL_REGISTRY 中），则回退到 CONTEXT_ASSEMBLY_DEFAULT_BUDGET。
    CONTEXT_ASSEMBLY_FILL_RATIO: float = 0.7
    CONTEXT_ASSEMBLY_MAX_BUDGET: int = 8000
    CONTEXT_ASSEMBLY_DEFAULT_BUDGET: int = 8000
    # 光标所在文件的 Chunk 权重加成（+30%），让用户正在编辑的文件相关片段排在前面
    CONTEXT_ASSEMBLY_CURSOR_BOOST: float = 0.3
    # 压缩阈值：函数/类 Chunk 行数超过该值时触发智能压缩（保留签名+注释+头尾）
    CONTEXT_ASSEMBLY_COMPRESS_LINES: int = 100
    # 压缩时保留的头部行数 / 尾部行数
    CONTEXT_ASSEMBLY_HEAD_LINES: int = 10
    CONTEXT_ASSEMBLY_TAIL_LINES: int = 10

    # S7 第 63-64 天：Planner 任务规划器配置
    # 步骤数量范围（S7 风险预警：Planner 输出不稳定，同一需求每次生成的步骤数波动大，
    # 必须在 System Prompt 中给死范围，并在解析后校验，过少或过多触发重新生成）
    PLANNER_MIN_STEPS: int = 5
    PLANNER_MAX_STEPS: int = 10
    # Planner 调用最大尝试次数（首次 + 重试）。解析失败或校验失败时重试一次。
    PLANNER_MAX_RETRIES: int = 2
    # Planner 单次模型调用超时（秒）。规划需要较充分推理，给较长超时。
    PLANNER_TIMEOUT_SECONDS: float = 120.0
    # Planner 采样温度。较低温度保证输出稳定、可复现。
    PLANNER_TEMPERATURE: float = 0.3
    # Planner 推理输出 max_tokens
    PLANNER_MAX_TOKENS: int = 4096
    # Planner Prompt 输入输出记录目录（供后续 P4 阶段微调模型使用）
    PLANNER_LOG_DIR: str = ".ai_planner_logs"

    # S7 第 67-68 天：ReAct 执行循环配置
    # 最大迭代次数（防止 AI 陷入死循环，比如反复重试同一个失败步骤）。
    # 对应 S7 风险预警："ReAct 循环的退出条件：必须设置最大迭代次数（如 15 次）"
    REACT_MAX_ITERATIONS: int = 15
    # 单次会话总超时（秒），防止长时间运行的 Agent 占用资源。
    # 对应 S7 风险预警："总超时（2 分钟）"
    REACT_TOTAL_TIMEOUT_SECONDS: float = 120.0
    # 单步 Reason 模型调用超时（秒）。单步推理不需要太长，给 60s。
    REACT_STEP_TIMEOUT_SECONDS: float = 60.0
    # Reason 阶段采样温度。较低温度让工具选择更稳定、可复现。
    REACT_TEMPERATURE: float = 0.2
    # Reason 阶段 max_tokens
    REACT_MAX_TOKENS: int = 1024
    # 上下文窗口（应对上下文爆炸）：只保留最近 N 个已完成步骤的完整 observation，
    # 更早的步骤只保留摘要（description + status）。
    # 对应 S7 风险预警："ReAct 循环中的上下文爆炸……只保留最近 3 步的完整观察结果"
    REACT_CONTEXT_WINDOW: int = 3

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT.lower() == "production"


settings = Settings()
