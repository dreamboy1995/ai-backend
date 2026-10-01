from typing import List, Dict, Optional, Literal, Any
from pydantic import BaseModel, Field, field_validator
import time


# ============================================================
# S4 第 31-32 天：代码索引接口数据结构
# ============================================================

class IndexStartRequest(BaseModel):
    """触发全量索引请求"""
    workspace_root: str = Field(..., min_length=1, description="工作区根路径")
    force_rebuild: bool = Field(default=False, description="是否强制重建索引")
    priority_files: Optional[List[str]] = Field(
        default=None,
        description=(
            "优先索引的文件列表（相对路径），如用户当前打开的文件。"
            "实现'即用即索引'策略：优先处理这些文件后，后台静默索引剩余文件。"
        )
    )


class IndexStartResponse(BaseModel):
    """触发全量索引响应"""
    job_id: str
    total_files: int


class IndexStatusResponse(BaseModel):
    """查询索引进度响应"""
    status: Literal["idle", "indexing", "done", "error"]
    total: int = 0
    processed: int = 0
    percentage: float = 0.0  # 进度比例，0.0 ~ 1.0（前端展示时 * 100 转百分比）
    total_symbols: int = 0
    message: Optional[str] = None


class IndexUpdateRequest(BaseModel):
    """增量更新通知请求（由插件在文件保存时调用）"""
    file_path: str = Field(..., min_length=1, description="变更的文件路径（相对路径）")
    action: Literal["modified", "deleted", "renamed"]


class IndexUpdateResponse(BaseModel):
    """增量更新响应"""
    success: bool
    message: str
    symbols_count: int = 0


# ============================================================
# S4 第 35-36 天：代码语义搜索接口数据结构
# ============================================================

class SearchResultItem(BaseModel):
    """单条语义搜索结果"""
    id: str
    file_path: str
    symbol_name: str
    chunk_type: str
    content: str
    start_line: int
    end_line: int
    embedding_version: str = ""
    distance: float = 0.0          # LanceDB L2 距离（越小越相似）
    score: float = 0.0             # 归一化相似度（0~1，越大越相似）


class SearchResponse(BaseModel):
    """语义搜索响应"""
    query: str
    top_k: int
    total: int                     # 实际返回的结果数
    results: List[SearchResultItem]


# ============================================================
# S4 第 39-40 天：依赖关系图接口数据结构
# ============================================================

class GraphEdgeInfo(BaseModel):
    """图边信息"""
    source: str
    target: str
    edge_type: str          # "import" | "call"
    line: int = 0
    raw: str = ""           # 原始引用文本


class GraphRelatedItem(BaseModel):
    """关联文件项"""
    file_path: str
    direction: str          # "upstream" | "downstream"
    depth: int
    edge: Optional[GraphEdgeInfo] = None


class GraphRelatedResponse(BaseModel):
    """关联文件查询响应"""
    file_path: str
    depth: int
    upstream: List[GraphRelatedItem] = []      # 被查询文件依赖的文件（import 的目标）
    downstream: List[GraphRelatedItem] = []    # 依赖被查询文件的文件（import 的来源）
    total: int = 0


class GraphImportsResponse(BaseModel):
    """文件 import 关系响应（用于验收：返回该文件 import 的所有本地模块名）"""
    file_path: str
    imports: List[dict] = []


# ============================================================
# S5 第 43-44 天：符号搜索接口数据结构（GET /v1/symbols/search）
# ============================================================

class SymbolItem(BaseModel):
    """单条符号搜索结果（供 # 符号下拉框实时补全）"""
    name: str                                # 符号名（如 DataProcessor）
    type: str                                 # class / function / variable
    file_path: str                            # 相对路径
    line: int                                 # 起始行号（1-based）


class SymbolSearchResponse(BaseModel):
    """符号搜索响应"""
    query: str
    limit: int
    total: int
    symbols: List[SymbolItem] = []


# ============================================================
# S5 第 43-44 天：Chat 接口 retrieval_config & SSE references
# ============================================================

class RetrievalConfig(BaseModel):
    """
    Chat 请求的检索配置（S5 关键接口变更）。

    - auto_context:      是否自动从索引中检索相关代码并注入 Prompt（S5 默认开启）
    - top_k:             最终注入 Prompt 的片段数（建议 3~5）
    - include_references: 是否在 SSE 流中返回 references 元数据块
                         （前端展示"📎 参考了 N 个代码片段"用）
    """
    auto_context: bool = True
    top_k: int = Field(default=5, ge=1, le=20)
    include_references: bool = True


class ReferenceItem(BaseModel):
    """
    SSE 流 references 元数据块中的单条引用（S5 关键接口变更）。

    前端在 AI 回复上方展示"📎 参考了 N 个代码片段"，点击可跳转到对应文件行号。
    """
    file: str           # 相对路径（如 src/main.py）
    lines: str          # 行号范围字符串（如 "12-45"）
    score: float = 0.0  # 相关性分数（rerank_score 或 rrf_score，越大越相关）
    symbol: str = ""    # 命中的符号名（便于前端展示函数/类名）


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


# S2 第 13-14 天新增：上下文条目，用于 @文件 / @选中代码 / 隐式上下文
# 关键技术点：content_snippet 应为插件端截断后的内容（头尾各 200 行 + 光标附近 50 行），
# 后端再做防御性长度校验，避免恶意/异常的大文件直接爆 Token 预算。
class ContextItem(BaseModel):
    """
    上下文条目（S2 第 13-14 天）。

    - type='file'        : 用户主动 @ 的整个文件
    - type='selection'   : 用户在编辑器中选中的代码片段
    - type='implicit'    : 插件自动附带的当前激活文件（不显示 @ 标签）
    """
    type: Literal["file", "selection", "implicit"]
    file_path: str = Field(..., min_length=1, description="文件路径，必填且非空")
    content_snippet: str = Field(
        ...,
        description="插件端截断后的文件/代码片段内容。ContextBuilder 会将其格式化为 XML 标签插入 System Prompt"
    )
    language: Optional[str] = Field(
        default=None,
        description="文件语言（如 python/javascript），用于代码块语言标签"
    )

    @field_validator("content_snippet")
    @classmethod
    def _validate_snippet_length(cls, v: str) -> str:
        # 防御性截断：单条上下文 snippet 上限 50000 字符（约 12500 token）。
        # 正常插件端截断后远小于此值；超过则说明客户端未做截断，直接拒绝以保护 Token 预算。
        max_chars = 50_000
        if len(v) > max_chars:
            raise ValueError(
                f"content_snippet 过长（{len(v)} 字符 > {max_chars}），"
                f"请在插件端先做截断（头尾各 200 行 + 光标附近 50 行）后再发送"
            )
        return v


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: str = "glm-4.5-air"
    temperature: float = 0.7
    stream: bool = True
    max_tokens: Optional[int] = None
    # S2 新增：会话 ID，用于多轮对话历史记忆（第 11-12 天）
    session_id: Optional[str] = None
    # S2 第 13-14 天新增：上下文数组，承载 @文件 / @选中代码 / 隐式上下文。
    # ContextBuilder 会消费此字段拼装 System Prompt（第 15-16 天已实现）。
    contexts: Optional[List[ContextItem]] = None
    # S3 第 28-29 天新增：请求模式。
    # - "chat": 普通对话，使用 60s 超时、默认 max_tokens=4096。
    # - "new":  单文件生成（/new 指令），使用 120s 超时、默认 max_tokens=8192，
    #           因为完整文件生成比对话需要更多推理时间。
    mode: Literal["chat", "new"] = "chat"
    # S5 第 43-44 天新增：检索配置。auto_context=true 时，后端用 hybrid_search
    # （向量 + BM25 + 符号 + RRF + Cross-Encoder 重排序）检索相关代码注入 System Prompt。
    # include_references=true 时，SSE 流首推 type:meta 的 references 数据块，
    # 前端在 AI 回复上方展示"📎 参考了 N 个代码片段"。
    retrieval_config: Optional[RetrievalConfig] = None


class Delta(BaseModel):
    """choices[].delta 中的增量内容"""
    role: Optional[str] = None
    content: Optional[str] = None


class Choice(BaseModel):
    index: int = 0
    delta: Optional[Delta] = None
    finish_reason: Optional[str] = None
    error: Optional[Dict[str, Any]] = None


class Usage(BaseModel):
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class ChatChunk(BaseModel):
    """
    标准 SSE 流式 DTO，一条 chunk 对应一次 data: 推送
    例：{"id":"chatcmpl-xxx","object":"chat.completion.chunk",
         "created":1699999999,"model":"glm-4.5-air",
         "choices":[{"index":0,"delta":{"content":"你"},"finish_reason":null}]}
    """
    id: str
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[Choice]
    usage: Optional[Usage] = None


class ChatMetaChunk(BaseModel):
    """
    SSE 流的元数据块（S5 关键接口变更）。

    在第一个 content chunk 之前推送一条 type:meta 数据，承载 references
    引用列表，供前端在 AI 回复上方渲染"📎 参考了 N 个代码片段"。

    前端状态机：先收到 type:meta 时存储引用列表，流结束时统一渲染，
    避免 Webview 未渲染完毕时引用信息丢失（S5 风险预警应对）。

    SSE 推送格式：
        data: {"type":"meta","references":[
            {"file":"src/main.py","lines":"12-45","score":0.92,"symbol":"foo"}
        ]}
    """
    type: Literal["meta"] = "meta"
    references: List[ReferenceItem] = []
