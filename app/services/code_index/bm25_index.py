"""
BM25 关键词检索器（S5 第 41-42 天）

基于 rank-bm25 的 BM25Okapi 实现，与 S4 的 LanceDB 向量库配合使用，
为后续三路混合检索（向量 + BM25 + 符号）提供关键词召回能力。

核心能力：
  - 多语言分词：中文用 jieba，英文拆驼峰/下划线，解决 rank-bm25 默认空格分词对中文无效的坑
  - 索引构建：遍历 LanceDB 中所有 Chunk 的 content 字段，构建 BM25 词袋
  - 持久化：序列化到 .ai_cache/bm25_index.pkl，避免每次启动重建
  - 检索：bm25_search(query, top_k) 返回 Chunk ID 列表及 BM25 原始分数

风险应对（S5 关键技术预研）：
  - BM25 索引磁盘占用：大型仓库词袋矩阵可能膨胀到数 GB。
    通过 BM25_SKIP_TEST_FILES（跳过测试文件）和 BM25_RECENT_MONTHS
    （仅索引最近 N 个月修改的文件）控制索引规模。
"""

import logging
import os
import pickle
import re
import threading
import time
from typing import Any, Dict, List, Optional

from rank_bm25 import BM25Okapi

try:
    import jieba
except ImportError:  # pragma: no cover - jieba 在 requirements 中已声明
    jieba = None

from .vector_store import get_vector_store

logger = logging.getLogger(__name__)

# ============================================================
# 多语言分词器
# ============================================================

# 匹配非单词字符（保留中英文、数字、下划线）
_TOKEN_SPLIT_RE = re.compile(r"[^\w\u4e00-\u9fff]+", re.UNICODE)
# 驼峰拆分：QuickSort -> Quick Sort（在小写字母和大写字母之间插空格）
_CAMEL_CASE_RE = re.compile(r"([a-z0-9])([A-Z])")
# 连续大写后跟小写：HTTPServer -> HTTP Server（如 URLParser -> URL Parser）
_UPPER_SEQ_RE = re.compile(r"([A-Z]+)([A-Z][a-z])")
# 下划线拆分：quick_sort -> quick sort
_SNAKE_CASE_RE = re.compile(r"_")
# 检测是否含中文字符
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")

# 代码常见停用词（BM25 的 IDF 会自然降低高频词权重，
# 但显式过滤可减少词袋体积、提升检索精度）
_CODE_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "not", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "must", "shall",
    "can", "to", "of", "in", "on", "at", "by", "for", "with", "about",
    "from", "as", "into", "through", "during", "before", "after", "above",
    "below", "up", "down", "out", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "no", "nor", "only", "own", "same", "so", "than",
    "too", "very", "just", "because", "but", "if", "while", "although",
    "though", "since", "until", "unless", "however", "therefore", "thus",
    "def", "return", "class", "import", "from", "self", "none", "true",
    "false", "null", "undefined", "this", "that", "these", "those",
})


def tokenize(text: str) -> List[str]:
    """
    多语言分词：中文用 jieba，英文拆驼峰/下划线。

    处理流程：
    1. 按非单词字符切分原始文本
    2. 对每个 token：拆分驼峰命名（QuickSort -> Quick Sort）、下划线（quick_sort -> quick sort）
    3. 含中文字符的 token 用 jieba 进一步切分
    4. 英文统一转小写，过滤停用词和单字符 token

    Args:
        text: 原始文本（代码切片内容或查询）

    Returns:
        分词后的 token 列表
    """
    if not text:
        return []

    raw_tokens = _TOKEN_SPLIT_RE.split(text)
    tokens: List[str] = []

    for raw in raw_tokens:
        if not raw:
            continue
        # 拆分驼峰：先处理连续大写序列（HTTPServer），再处理普通驼峰（QuickSort）
        processed = _UPPER_SEQ_RE.sub(r"\1 \2", raw)
        processed = _CAMEL_CASE_RE.sub(r"\1 \2", processed)
        # 拆分下划线
        processed = _SNAKE_CASE_RE.sub(" ", processed)

        for part in processed.split():
            if not part:
                continue
            if _CJK_RE.search(part):
                # 中文分词：jieba 可能切出含英文的混合片段，统一再拆一次
                if jieba is not None:
                    for seg in jieba.lcut(part):
                        seg = seg.strip()
                        if not seg:
                            continue
                        # jieba 切出的片段可能仍含驼峰/下划线（如 "QuickSort函数"）
                        sub = _CAMEL_CASE_RE.sub(r"\1 \2", seg)
                        sub = _SNAKE_CASE_RE.sub(" ", sub)
                        for s in sub.split():
                            if s and not _CJK_RE.search(s):
                                s = s.lower()
                            if s and s not in _CODE_STOPWORDS and len(s) > 1:
                                tokens.append(s)
                else:  # pragma: no cover - jieba 不可用时退化为逐字符
                    tokens.extend(c for c in part if c.strip())
            else:
                lower = part.lower()
                if lower and lower not in _CODE_STOPWORDS and len(lower) > 1:
                    tokens.append(lower)

    return tokens


# ============================================================
# BM25 索引
# ============================================================

class BM25Index:
    """
    BM25 关键词索引。

    数据源：S4 存入 LanceDB 的所有 Chunk（content 字段）。
    持久化：pickle 序列化到 BM25_INDEX_PATH，启动时优先加载避免重建。

    使用方式：
        bm25 = get_bm25_index()
        bm25.build()                          # 从 LanceDB 全量构建
        results = bm25.search("排序算法", top_k=20)
        # results: [{"id": "...", "score": 2.31}, ...]
    """

    def __init__(
        self,
        index_path: str = ".ai_cache/bm25_index.pkl",
        k1: float = 1.5,
        b: float = 0.75,
        skip_test_files: bool = True,
        recent_months: int = 0,
        vector_store=None,
    ):
        self.index_path = index_path
        self.k1 = k1
        self.b = b
        self.skip_test_files = skip_test_files
        self.recent_months = recent_months
        # 允许注入 VectorStore 实例（便于测试）；未提供时使用全局单例
        self._vector_store = vector_store

        self._bm25: Optional[BM25Okapi] = None
        # BM25 内部按位置索引文档，需维护 position -> chunk_id 的映射
        self._chunk_ids: List[str] = []
        # 记录每个 chunk 的 file_path（供调试与后续融合使用）
        self._file_paths: List[str] = []
        self._loaded = False
        # 使用可重入锁（RLock），避免 _ensure_index -> load 嵌套加锁导致死锁
        self._lock = threading.RLock()

    def _get_store(self):
        """获取 VectorStore 实例（优先使用注入的，否则用全局单例）"""
        if self._vector_store is not None:
            return self._vector_store
        return get_vector_store()

    # ============================================================
    # 索引构建
    # ============================================================

    def build(self) -> int:
        """
        从 LanceDB 全量构建 BM25 索引并持久化。

        流程：
        1. 从 LanceDB 读取所有 Chunk 的 id / file_path / content
        2. 按配置过滤（跳过测试文件、仅保留最近 N 个月修改的文件）
        3. 对每个 Chunk 的 content 做分词，构建词袋语料
        4. 训练 BM25Okapi 模型
        5. 序列化到 index_path

        Returns:
            成功索引的 Chunk 数量
        """
        store = self._get_store()

        if not store.is_table_exists():
            logger.warning("[BM25Index] 向量表不存在，无法构建 BM25 索引")
            return 0

        table = store._get_table()
        # 只读需要的列，减少内存占用
        try:
            arrow_table = table.to_arrow().select(["id", "file_path", "content"])
        except Exception as e:
            logger.error(f"[BM25Index] 读取 LanceDB 数据失败: {e}", exc_info=True)
            raise

        # 获取工作区根路径（用于 mtime 过滤）
        workspace_root = self._get_workspace_root()
        cutoff_time = self._get_cutoff_time()

        corpus: List[List[str]] = []
        chunk_ids: List[str] = []
        file_paths: List[str] = []

        for row in arrow_table.to_pylist():
            chunk_id = row.get("id")
            file_path = row.get("file_path", "")
            content = row.get("content", "")

            if not chunk_id or not content:
                continue

            # 过滤：跳过测试文件
            if self.skip_test_files and self._is_test_file(file_path):
                continue

            # 过滤：仅保留最近 N 个月修改的文件
            if cutoff_time and workspace_root:
                full_path = os.path.join(workspace_root, file_path)
                if not self._is_recent_file(full_path, cutoff_time):
                    continue

            tokens = tokenize(content)
            if not tokens:
                continue

            corpus.append(tokens)
            chunk_ids.append(chunk_id)
            file_paths.append(file_path)

        if not corpus:
            logger.warning("[BM25Index] 无有效 Chunk，BM25 索引为空")
            self._bm25 = None
            self._chunk_ids = []
            self._file_paths = []
            self._loaded = True
            return 0

        with self._lock:
            self._bm25 = BM25Okapi(corpus, k1=self.k1, b=self.b)
            self._chunk_ids = chunk_ids
            self._file_paths = file_paths
            self._loaded = True

        self._save()
        logger.info(
            f"[BM25Index] 构建完成：{len(chunk_ids)} 个 Chunk，"
            f"词袋大小约 {sum(len(d) for d in corpus)} tokens"
        )
        return len(chunk_ids)

    # ============================================================
    # 持久化
    # ============================================================

    def _save(self) -> None:
        """序列化 BM25 模型与 chunk_id 映射到磁盘"""
        try:
            os.makedirs(os.path.dirname(self.index_path) or ".", exist_ok=True)
            with open(self.index_path, "wb") as f:
                pickle.dump(
                    {
                        "bm25": self._bm25,
                        "chunk_ids": self._chunk_ids,
                        "file_paths": self._file_paths,
                        "k1": self.k1,
                        "b": self.b,
                    },
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            size_kb = os.path.getsize(self.index_path) / 1024
            logger.debug(f"[BM25Index] 索引已保存到 {self.index_path}（{size_kb:.1f} KB）")
        except Exception as e:
            logger.error(f"[BM25Index] 保存索引失败: {e}", exc_info=True)

    def load(self) -> bool:
        """
        从磁盘加载 BM25 索引。

        Returns:
            True 表示加载成功，False 表示索引文件不存在或损坏
        """
        if not os.path.exists(self.index_path):
            return False
        try:
            with open(self.index_path, "rb") as f:
                data = pickle.load(f)
            with self._lock:
                self._bm25 = data.get("bm25")
                self._chunk_ids = data.get("chunk_ids", [])
                self._file_paths = data.get("file_paths", [])
                self._loaded = True
            logger.info(
                f"[BM25Index] 从磁盘加载索引成功：{len(self._chunk_ids)} 个 Chunk"
            )
            return True
        except Exception as e:
            logger.warning(f"[BM25Index] 加载索引失败，将重新构建: {e}")
            return False

    # ============================================================
    # 检索
    # ============================================================

    def search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        BM25 关键词检索。

        Args:
            query:  查询文本（中文或英文）
            top_k:  返回前 K 个结果

        Returns:
            结果列表，按 BM25 分数降序排列，每项为：
              {"id": chunk_id, "file_path": ..., "score": bm25_score}
            若索引未构建或无结果，返回空列表。
        """
        self._ensure_index()
        if self._bm25 is None or not self._chunk_ids:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        # 取 top_k 个最高分的索引位置
        if len(scores) <= top_k:
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        else:
            # argpartition 比全排序更快，但为简单起见用 sorted
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results: List[Dict[str, Any]] = []
        for idx in top_indices:
            score = float(scores[idx])
            # 仅过滤负分（BM25Okapi 对出现于恰好一半文档的词 IDF=0，
            # 零分仍为有效匹配，不应丢弃）
            if score < 0:
                continue
            results.append({
                "id": self._chunk_ids[idx],
                "file_path": self._file_paths[idx] if idx < len(self._file_paths) else "",
                "score": score,
            })
        return results

    def _ensure_index(self) -> None:
        """
        确保索引可用：优先从内存 → 磁盘加载 → 从 LanceDB 构建。

        线程安全：用锁避免并发构建。
        """
        if self._loaded and self._bm25 is not None:
            return
        with self._lock:
            if self._loaded and self._bm25 is not None:
                return
            # 先尝试从磁盘加载
            if self.load():
                return
            # 磁盘无索引，从 LanceDB 构建
            logger.info("[BM25Index] 磁盘无 BM25 索引，从 LanceDB 构建")
            self.build()

    # ============================================================
    # 状态管理
    # ============================================================

    def mark_dirty(self) -> None:
        """
        标记索引为脏数据。

        在索引更新（文件增删改）后调用，下次 search 时会重新从 LanceDB 构建。
        注意：为避免频繁重建，增量更新后仅标记，实际重建延迟到首次检索时执行。
        """
        with self._lock:
            self._loaded = False
            self._bm25 = None
        logger.debug("[BM25Index] 索引已标记为脏数据，下次检索时重建")

    def clear(self) -> None:
        """清空内存索引并删除磁盘持久化文件"""
        with self._lock:
            self._bm25 = None
            self._chunk_ids = []
            self._file_paths = []
            self._loaded = False
        if os.path.exists(self.index_path):
            try:
                os.remove(self.index_path)
                logger.info(f"[BM25Index] 已删除索引文件 {self.index_path}")
            except OSError as e:
                logger.warning(f"[BM25Index] 删除索引文件失败: {e}")

    def count(self) -> int:
        """返回当前索引中的 Chunk 数量"""
        return len(self._chunk_ids)

    # ============================================================
    # 过滤辅助
    # ============================================================

    @staticmethod
    def _is_test_file(file_path: str) -> bool:
        """
        判断是否为测试文件。

        规则（任一满足即视为测试文件）：
        - 路径中包含 /test/ 或 \\test\\ 目录
        - 路径中包含 /tests/ 或 \\tests\\ 目录
        - 文件名以 test_ 开头（如 test_sort.py）
        - 文件名以 _test. 结尾（如 sort_test.py）
        - 文件名以 .test. 结尾（如 utils.test.ts）
        """
        norm = file_path.replace("\\", "/").lower()
        parts = norm.split("/")
        basename = parts[-1]
        # 目录名检测
        if "test" in parts or "tests" in parts:
            return True
        # 文件名模式检测
        if basename.startswith("test_"):
            return True
        if basename.startswith("test."):
            return True
        # sort_test.py / utils.test.ts 等
        if "_test." in basename or ".test." in basename:
            return True
        return False

    @staticmethod
    def _is_recent_file(full_path: str, cutoff_time: float) -> bool:
        """判断文件修改时间是否在 cutoff_time 之后"""
        try:
            return os.path.getmtime(full_path) >= cutoff_time
        except OSError:
            # 文件不存在或无法读取 mtime，默认放行
            return True

    @staticmethod
    def _get_workspace_root() -> Optional[str]:
        """从 IndexService 获取当前工作区根路径"""
        try:
            from .index_service import get_index_service
            return get_index_service().workspace_root
        except Exception:
            return None

    def _get_cutoff_time(self) -> Optional[float]:
        """根据 BM25_RECENT_MONTHS 计算时间截断点（0 表示不限制）"""
        if self.recent_months <= 0:
            return None
        return time.time() - self.recent_months * 30 * 24 * 3600


# ============================================================
# 单例工厂
# ============================================================

_bm25_index: Optional[BM25Index] = None


def get_bm25_index() -> BM25Index:
    """
    获取 BM25Index 单例。

    配置从 app.config.settings 读取，支持：
    - BM25_INDEX_PATH / BM25_DEFAULT_TOP_K / BM25_SKIP_TEST_FILES
    - BM25_RECENT_MONTHS / BM25_K1 / BM25_B
    """
    global _bm25_index
    if _bm25_index is None:
        try:
            from app.config import settings
            _bm25_index = BM25Index(
                index_path=settings.BM25_INDEX_PATH,
                k1=settings.BM25_K1,
                b=settings.BM25_B,
                skip_test_files=settings.BM25_SKIP_TEST_FILES,
                recent_months=settings.BM25_RECENT_MONTHS,
            )
        except Exception as e:
            logger.warning(f"[BM25Index] 读取配置失败，使用默认参数: {e}")
            _bm25_index = BM25Index()
    return _bm25_index


def bm25_search(query: str, top_k: int = 20) -> List[Dict[str, Any]]:
    """
    BM25 关键词检索便捷函数（S5 第 41-42 天指定接口）。

    内部调用 BM25Index 单例的 search 方法，自动处理索引加载/构建。

    Args:
        query:  查询文本（中文或英文）
        top_k:  返回前 K 个结果，默认 20

    Returns:
        结果列表，按 BM25 分数降序排列，每项为：
          {"id": chunk_id, "file_path": ..., "score": bm25_score}
    """
    return get_bm25_index().search(query, top_k=top_k)
