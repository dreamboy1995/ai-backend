"""
向量化客户端（S4 第 33-34 天）

将代码切片（CodeChunk）的文本内容转换为向量，供向量数据库检索使用。

支持两种后端：
  1. 本地后端（默认）：使用 sentence-transformers 加载 all-MiniLM-L6-v2
     （~80MB，CPU 可跑，输出 384 维向量）。
  2. 远程后端：调用 OpenAI text-embedding-3-small API（1536 维），
     并将结果缓存到本地 SQLite，避免重复计算。

降级策略（auto 模式）：
  - 优先尝试本地模型。
  - 若 sentence-transformers 未安装、模型加载失败或加载耗时超过阈值，
    自动降级到远程 API。
  - 远程 API 调用失败时，抛出明确异常，由上层决定是否中断索引。

风险应对（S4 关键技术预研）：
  - 本地模型内存占用（~80MB）：对大多数开发机可接受。
  - 向量维度不兼容：通过 embedding_version 字段标识，换模型时可重建索引。
  - 首次索引耗时：远程模式下通过 SQLite 缓存避免重复向量化。
"""

import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from typing import List, Optional

from .models import CodeChunk

logger = logging.getLogger(__name__)


# ============================================================
# 远程向量 SQLite 缓存
# ============================================================

class EmbeddingCache:
    """
    远程 Embedding 结果的本地缓存（SQLite）。

    以 (content_hash, model) 为键缓存向量，避免对相同文本重复调用远程 API。
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    content_hash TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _hash(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def get(self, content: str, model: str) -> Optional[List[float]]:
        key = self._hash(content)
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                row = conn.execute(
                    "SELECT embedding FROM embeddings WHERE content_hash = ? AND model = ?",
                    (key, model),
                ).fetchone()
                if row:
                    return json.loads(row[0])
                return None
            finally:
                conn.close()

    def put(self, content: str, model: str, embedding: List[float]):
        key = self._hash(content)
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO embeddings (content_hash, model, embedding, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (key, model, json.dumps(embedding), time.time()),
                )
                conn.commit()
            finally:
                conn.close()

    def get_batch(self, contents: List[str], model: str) -> List[Optional[List[float]]]:
        """批量查询缓存，返回与 contents 等长的列表（未命中为 None）"""
        result = [None] * len(contents)
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                for i, content in enumerate(contents):
                    key = self._hash(content)
                    row = conn.execute(
                        "SELECT embedding FROM embeddings WHERE content_hash = ? AND model = ?",
                        (key, model),
                    ).fetchone()
                    if row:
                        result[i] = json.loads(row[0])
            finally:
                conn.close()
        return result


# ============================================================
# 本地后端：sentence-transformers
# ============================================================

class LocalEmbeddingBackend:
    """
    本地向量化后端，基于 sentence-transformers。

    模型 all-MiniLM-L6-v2 输出 384 维向量，CPU 可跑。
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._dimension: Optional[int] = None

    def load(self, timeout: float = 3.0) -> bool:
        """
        加载本地模型。

        Args:
            timeout: 加载超时秒数。超时返回 False（调用方应降级到远程）。

        Returns:
            True 表示加载成功，False 表示失败或超时。
        """
        if self._model is not None:
            return True

        result = {"model": None, "error": None}

        def _load():
            try:
                # S5 修复：强制 HuggingFace 离线模式，避免 sentence-transformers
                # 加载本地缓存模型时联网检查元数据导致超时（网络受限环境下 120s 不够）。
                # 模型已缓存于 ~/.cache/huggingface/hub/，离线加载 7s 内完成。
                import os
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                from sentence_transformers import SentenceTransformer
                result["model"] = SentenceTransformer(self.model_name)
            except Exception as e:
                result["error"] = e

        thread = threading.Thread(target=_load, daemon=True)
        start = time.time()
        thread.start()
        thread.join(timeout=timeout)

        elapsed = time.time() - start
        if thread.is_alive():
            logger.warning(
                f"[Embedding] 本地模型加载超时（{elapsed:.1f}s > {timeout}s），"
                f"将降级到远程 API"
            )
            return False

        if result["error"] is not None:
            logger.warning(
                f"[Embedding] 本地模型加载失败: {result['error']}，将降级到远程 API"
            )
            return False

        if result["model"] is None:
            return False

        self._model = result["model"]
        # 探测向量维度
        try:
            test_vec = self._model.encode(["test"], show_progress_bar=False)
            self._dimension = len(test_vec[0])
        except Exception as e:
            logger.warning(f"[Embedding] 维度探测失败: {e}")
            self._dimension = 384  # all-MiniLM-L6-v2 默认维度

        logger.info(
            f"[Embedding] 本地模型加载成功: {self.model_name} "
            f"(维度={self._dimension}, 耗时={elapsed:.1f}s)"
        )
        return True

    @property
    def dimension(self) -> int:
        return self._dimension or 384

    @property
    def version(self) -> str:
        return f"{self.model_name}-{self.dimension}"

    def embed(self, texts: List[str]) -> List[List[float]]:
        if self._model is None:
            raise RuntimeError("本地模型未加载")
        # S4 性能优化：关闭进度条，batch_size=32
        vectors = self._model.encode(
            texts,
            show_progress_bar=False,
            batch_size=32,
            convert_to_numpy=True,
        )
        return vectors.tolist()


# ============================================================
# 远程后端：OpenAI Embedding API
# ============================================================

class RemoteEmbeddingBackend:
    """
    远程向量化后端，调用 OpenAI text-embedding-3-small API。

    结果缓存到 SQLite，避免重复调用。
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_base: str = "https://api.openai.com/v1",
        api_key: str = "",
        cache_db: str = ".ai_index/embeddings_cache.sqlite",
        batch_size: int = 64,
    ):
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.batch_size = batch_size
        self._cache = EmbeddingCache(cache_db)
        self._dimension: Optional[int] = None

    @property
    def dimension(self) -> int:
        return self._dimension or 1536  # text-embedding-3-small 默认 1536 维

    @property
    def version(self) -> str:
        return f"{self.model}-{self.dimension}"

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        # 1. 查缓存
        cached = self._cache.get_batch(texts, self.model)
        missing_indices = [i for i, v in enumerate(cached) if v is None]
        missing_texts = [texts[i] for i in missing_indices]

        if missing_texts:
            # 2. 调用远程 API（分批）
            remote_vectors = self._call_api_batch(missing_texts)
            # 3. 写缓存
            for idx, vec in zip(missing_indices, remote_vectors):
                self._cache.put(texts[idx], self.model, vec)
                cached[idx] = vec

        # 探测维度（取第一个非空向量）
        if self._dimension is None:
            for v in cached:
                if v:
                    self._dimension = len(v)
                    break

        return cached

    def _call_api_batch(self, texts: List[str]) -> List[List[float]]:
        """分批调用远程 Embedding API"""
        import httpx

        if not self.api_key:
            raise RuntimeError(
                "远程 Embedding API Key 未配置（OPENAI_API_KEY 或 EMBEDDING_API_KEY）"
            )

        all_vectors: List[List[float]] = []
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            try:
                with httpx.Client(timeout=30.0) as client:
                    resp = client.post(
                        f"{self.api_base}/embeddings",
                        headers=headers,
                        json={"model": self.model, "input": batch},
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    # 按 index 排序后取 embedding
                    vectors = [None] * len(batch)
                    for item in data.get("data", []):
                        idx = item.get("index")
                        if idx is not None and 0 <= idx < len(batch):
                            vectors[idx] = item["embedding"]
                    # 兜底：若顺序已正确
                    if any(v is None for v in vectors):
                        vectors = [item["embedding"] for item in data.get("data", [])]
                    all_vectors.extend(vectors)
            except httpx.HTTPStatusError as e:
                raise RuntimeError(
                    f"远程 Embedding API 返回错误: {e.response.status_code} {e.response.text}"
                ) from e
            except httpx.RequestError as e:
                raise RuntimeError(f"远程 Embedding API 请求失败: {e}") from e

        return all_vectors


# ============================================================
# 统一客户端
# ============================================================

class EmbeddingClient:
    """
    向量化统一客户端，自动选择本地或远程后端。

    使用方式：
        client = EmbeddingClient.from_settings()
        chunks = client.embed_chunks(chunks)
    """

    def __init__(
        self,
        mode: str = "auto",
        local_model: str = "all-MiniLM-L6-v2",
        local_load_timeout: float = 120.0,
        remote_model: str = "text-embedding-3-small",
        remote_api_base: str = "https://api.openai.com/v1",
        remote_api_key: str = "",
        cache_db: str = ".ai_index/embeddings_cache.sqlite",
    ):
        self.mode = mode
        self._local = LocalEmbeddingBackend(local_model)
        self._remote = RemoteEmbeddingBackend(
            model=remote_model,
            api_base=remote_api_base,
            api_key=remote_api_key,
            cache_db=cache_db,
        )
        self._local_load_timeout = local_load_timeout
        self._active_backend = None  # "local" | "remote"

    @classmethod
    def from_settings(cls) -> "EmbeddingClient":
        from app.config import settings
        # API Key 读取优先级：EMBEDDING_API_KEY > OPENAI_API_KEY > ZAI_API_KEY
        # （ZAI_API_KEY 为智谱 AI，其 Embedding 接口兼容 OpenAI 协议）
        api_key = settings.EMBEDDING_API_KEY or settings.OPENAI_API_KEY or settings.ZAI_API_KEY
        return cls(
            mode=settings.EMBEDDING_MODE,
            local_model=settings.EMBEDDING_LOCAL_MODEL,
            local_load_timeout=settings.EMBEDDING_LOCAL_LOAD_TIMEOUT,
            remote_model=settings.EMBEDDING_REMOTE_MODEL,
            remote_api_base=settings.EMBEDDING_REMOTE_API_BASE,
            remote_api_key=api_key,
            cache_db=settings.EMBEDDING_CACHE_DB,
        )

    def _ensure_backend(self):
        """确保已选定并加载后端"""
        if self._active_backend is not None:
            return

        if self.mode in ("local", "auto"):
            if self._local.load(timeout=self._local_load_timeout):
                self._active_backend = "local"
                return

        if self.mode in ("remote", "auto"):
            self._active_backend = "remote"
            return

        raise RuntimeError(
            f"无法加载任何 Embedding 后端（mode={self.mode}）。"
            f"请安装 sentence-transformers 或配置远程 API Key。"
        )

    @property
    def dimension(self) -> int:
        self._ensure_backend()
        if self._active_backend == "local":
            return self._local.dimension
        return self._remote.dimension

    @property
    def version(self) -> str:
        """
        向量模型版本标识，用于维度迁移（S4 风险预警）。

        始终返回当前活跃后端的真实版本，确保 embedding_version 与实际向量匹配。
        """
        self._ensure_backend()
        if self._active_backend == "local":
            return self._local.version
        return self._remote.version

    @property
    def active_backend(self) -> str:
        self._ensure_backend()
        return self._active_backend

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        将文本列表转换为向量列表。

        Args:
            texts: 文本列表

        Returns:
            向量列表（每个向量是 float 列表）
        """
        if not texts:
            return []
        self._ensure_backend()
        if self._active_backend == "local":
            return self._local.embed(texts)
        return self._remote.embed(texts)

    def embed_chunks(self, chunks: List[CodeChunk]) -> List[CodeChunk]:
        """
        批量为 CodeChunk 填充 embedding 与 embedding_version 字段。

        Args:
            chunks: 待向量化的切片列表

        Returns:
            填充了 embedding 的切片列表（原地修改并返回）
        """
        if not chunks:
            return chunks

        texts = [c.content for c in chunks]
        vectors = self.embed(texts)
        ver = self.version

        for chunk, vec in zip(chunks, vectors):
            chunk.embedding = vec
            chunk.embedding_version = ver

        return chunks


# 单例
_embedding_client: Optional[EmbeddingClient] = None


def get_embedding_client() -> EmbeddingClient:
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = EmbeddingClient.from_settings()
    return _embedding_client
