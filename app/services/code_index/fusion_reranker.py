"""
三路检索融合 + RRF 排序（S5 第 43-44 天）

将向量（语义）、BM25（关键词）、符号（精确）三路检索结果用 RRF
（Reciprocal Rank Fusion，倒数排名融合）算法融合为统一排序，
再用 Cross-Encoder 重排序模型对 Top-30 候选逐对（Query, Chunk）
计算相关性，最终输出 Top-5。

核心算法：
  RRF_score(doc) = Σ  1 / (k + rank_i(doc))
  其中 k=60（经典值），rank_i(doc) 是 doc 在第 i 路结果中的排名（从 1 开始）。
  越多路命中、排名越靠前，RRF 分数越高。

输入：
  - 向量检索 Top-20（来自 VectorStore.search）
  - BM25 Top-20（来自 bm25_index.bm25_search）
  - 符号精确检索 Top-10（来自 SymbolSearcher.search，Day 45-44 的轻量版）
输出：
  - RRF 融合后的 Top-30 候选 Chunk 列表
  - 重排序后的 Top-5

风险应对（S5 关键技术预研）：
  - 重排序模型显存不足：Cross-Encoder 90MB，但 CPU 上 30 候选逐对推理可能 1s+。
    方案 A：限定重排序候选数 ≤ 10（RERANK_MAX_CANDIDATES）。
    方案 B：超时熔断——重排序耗时 > 300ms 时直接返回 RRF 结果。
  - 信息过载：System Prompt 中明确"Context 仅供参考"（由上层 Chat 路径处理）。
  - torch 模型加载失败：降级为只走 RRF，不重排序（记录警告日志）。
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .bm25_index import get_bm25_index
from .models import _compute_chunk_id
from .vector_store import get_vector_store

logger = logging.getLogger(__name__)

# ============================================================
# 候选 Chunk 数据结构
# ============================================================


@dataclass
class Candidate:
    """
    统一候选 Chunk 表示（三路融合的内部数据结构）。

    Attributes:
        chunk_id:     chunk 唯一 ID（向量库主键 / 哈希）
        file_path:    相对路径
        symbol_name:  符号名
        chunk_type:   function / class / import / block
        content:      切片文本（rerank 时使用）
        start_line:   起始行号（1-based）
        end_line:     结束行号（1-based）
        vector_score: 向量路径原始分数（distance/score，调试用）
        bm25_score:   BM25 路径原始分数
        symbol_score: 符号路径原始分数（精确=1.0 / 前缀=0.5）
        rrf_score:    RRF 融合分数
        rerank_score: Cross-Encoder 重排序分数（None 表示未做重排序）
        sources:      命中的检索路径列表（["vector","bm25","symbol"]）
    """

    chunk_id: str
    file_path: str = ""
    symbol_name: str = ""
    chunk_type: str = ""
    content: str = ""
    start_line: int = 0
    end_line: int = 0
    vector_score: float = 0.0
    bm25_score: float = 0.0
    symbol_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: Optional[float] = None
    sources: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.chunk_id,
            "file_path": self.file_path,
            "symbol_name": self.symbol_name,
            "chunk_type": self.chunk_type,
            "content": self.content,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "score": self.rerank_score if self.rerank_score is not None else self.rrf_score,
            "rrf_score": self.rrf_score,
            "rerank_score": self.rerank_score,
            "sources": self.sources,
        }


# ============================================================
# 符号精确检索（S5 第 45-46 天：委托给 symbol_exact_matcher.py）
#
# 本类保留作为 RRF 三路融合的"符号精确召回"输入源接口，
# 实际匹配逻辑已迁移到 symbol_exact_matcher.SymbolExactMatcher。
# Day 45-46 的完整实现额外支持：
#   - 子串包含匹配（token in name）
#   - 反向依赖查询（find_callers：谁调用了某符号）
#   - 路径跨平台归一化（POSIX 正斜杠）
# ============================================================


class SymbolSearcher:
    """
    符号精确检索（委托给 SymbolExactMatcher）。

    作为 RRF 三路融合的"符号精确召回"输入源，search() 委托给
    symbol_exact_matcher.SymbolExactMatcher.search()，返回格式保持一致。

    匹配规则：
      - 用户输入 #Tag（如 #DataProcessor）：精确匹配符号名 → score=1.0
      - 驼峰/下划线命名前缀匹配 → score=0.5
      - 子串包含匹配（token 长度 >= 3）→ score=0.3
    """

    def __init__(self, vector_store=None):
        self._vector_store = vector_store
        self._matcher = None

    def _get_matcher(self):
        """惰性获取 SymbolExactMatcher 实例（注入 vector_store）"""
        if self._matcher is None:
            from .symbol_exact_matcher import SymbolExactMatcher
            self._matcher = SymbolExactMatcher(vector_store=self._vector_store)
        return self._matcher

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        从用户 query 中识别符号 token，对 IndexService 内存符号表做精确 / 前缀匹配。

        委托给 SymbolExactMatcher.search()，返回格式：
          [{id, file_path, symbol_name, chunk_type, content,
            start_line, end_line, score, source, match_type}, ...]
        """
        return self._get_matcher().search(query, top_k=top_k)

    @staticmethod
    def _extract_symbol_tokens(query: str) -> List[str]:
        """
        从 query 中提取符号候选 token（委托给 symbol_exact_matcher）。
        保留此静态方法供 app/api/symbols.py 等模块复用。
        """
        from .symbol_exact_matcher import extract_symbol_tokens
        return extract_symbol_tokens(query)


# ============================================================
# Cross-Encoder 重排序器
# ============================================================


class CrossEncoderReranker:
    """
    基于 sentence-transformers Cross-Encoder 的重排序器。

    模型：cross-encoder/ms-marco-MiniLM-L-6-v2（~90MB，CPU 可跑）。
    输入 (Query, Chunk) 对，输出相关性分数（logit），按分数重排。

    风险应对：
      - CPU 慢：限定重排序候选数 ≤ RERANK_MAX_CANDIDATES（默认 10）
      - 超时熔断：单次重排序 > RERANK_TIMEOUT_MS（默认 300ms）时跳过，
        返回 RRF 结果
      - 模型加载失败：降级为不重排序，仅记录警告
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        timeout_ms: int = 300,
        max_candidates: int = 10,
        enabled: bool = True,
    ):
        self.model_name = model_name
        self.timeout_ms = timeout_ms
        self.max_candidates = max_candidates
        self.enabled = enabled
        self._model = None
        self._load_failed = False
        self._lock = threading.Lock()

    def _ensure_model(self) -> bool:
        """惰性加载 Cross-Encoder 模型，加载失败时永久标记不再重试"""
        if not self.enabled:
            return False
        if self._model is not None:
            return True
        if self._load_failed:
            return False

        with self._lock:
            if self._model is not None:
                return True
            if self._load_failed:
                return False
            try:
                from sentence_transformers import CrossEncoder
                # 强制 CPU，避免占用 GPU 显存
                self._model = CrossEncoder(self.model_name, device="cpu")
                # 切到 eval 模式，关闭 dropout，降低推理延迟
                try:
                    self._model.model.eval()
                except Exception:
                    pass
                logger.info(
                    f"[Reranker] Cross-Encoder 加载成功: {self.model_name}"
                )
                return True
            except Exception as e:
                self._load_failed = True
                logger.warning(
                    f"[Reranker] Cross-Encoder 加载失败，将仅用 RRF 结果: {e}"
                )
                return False

    def rerank(
        self,
        query: str,
        candidates: List[Candidate],
        top_k: int = 5,
    ) -> Tuple[List[Candidate], str]:
        """
        对 candidates 逐对 (query, content) 计算相关性，按新分数重排。

        Args:
            query:      用户查询
            candidates: RRF 输出的候选列表（已含 content）
            top_k:      最终返回数量

        Returns:
            (重排后的 Top-K 候选列表, 状态标记)
            状态标记：
              "reranked"   - 成功重排序
              "timeout"    - 超时熔断，返回 RRF 结果
              "no_model"   - 模型不可用，返回 RRF 结果
              "skipped"    - 候选数过少或被禁用，返回 RRF 结果
        """
        # 候选数过少时直接返回 RRF 结果（重排序收益低于开销）
        if not candidates or len(candidates) <= 1:
            return candidates[:top_k], "skipped"

        if not self._ensure_model():
            # 模型不可用：按 RRF 排序截断返回
            sorted_cands = sorted(
                candidates, key=lambda c: c.rrf_score, reverse=True
            )
            return sorted_cands[:top_k], "no_model"

        # 限定重排序候选数（性能保护：Top-30 太多会慢，截到 max_candidates）
        to_rerank = candidates[: self.max_candidates]
        pairs = [(query, c.content or "") for c in to_rerank]

        # 超时熔断：在子线程中跑 rerank，主线程等待 timeout_ms
        result_box: Dict[str, Any] = {"scores": None, "error": None}

        def _do_predict():
            try:
                # CrossEncoder.predict 返回 numpy array
                scores = self._model.predict(pairs, show_progress_bar=False)
                result_box["scores"] = list(scores)
            except Exception as e:
                result_box["error"] = e

        t = threading.Thread(target=_do_predict, daemon=True)
        start = time.time()
        t.start()
        t.join(timeout=self.timeout_ms / 1000.0)
        elapsed_ms = (time.time() - start) * 1000

        if t.is_alive():
            # 超时熔断：后台线程仍在跑，直接返回 RRF 结果
            logger.warning(
                f"[Reranker] 重排序超时（{elapsed_ms:.0f}ms > {self.timeout_ms}ms），"
                f"跳过重排序，返回 RRF 结果"
            )
            sorted_cands = sorted(
                candidates, key=lambda c: c.rrf_score, reverse=True
            )
            return sorted_cands[:top_k], "timeout"

        if result_box["error"] is not None or result_box["scores"] is None:
            logger.warning(
                f"[Reranker] 重排序推理失败: {result_box['error']}，返回 RRF 结果"
            )
            sorted_cands = sorted(
                candidates, key=lambda c: c.rrf_score, reverse=True
            )
            return sorted_cands[:top_k], "no_model"

        scores = result_box["scores"]
        # 将 rerank 分数填回 candidates（仅前 len(to_rerank) 个）
        for c, s in zip(to_rerank, scores):
            try:
                c.rerank_score = float(s)
            except (TypeError, ValueError):
                c.rerank_score = 0.0

        # 按 rerank_score 降序排，取 Top-K
        reranked = sorted(
            to_rerank, key=lambda c: c.rerank_score or 0.0, reverse=True
        )
        # 未参与重排序的候选（candidates[max_candidates:]）丢弃，因为未获得 rerank 分数
        logger.info(
            f"[Reranker] 重排序完成（{elapsed_ms:.0f}ms, 候选={len(to_rerank)}），"
            f"返回 Top-{min(top_k, len(reranked))}"
        )
        return reranked[:top_k], "reranked"


# ============================================================
# 主类：FusionReranker
# ============================================================


class FusionReranker:
    """
    三路检索融合 + 重排序编排器。

    使用方式：
        fr = get_fusion_reranker()
        results = fr.hybrid_search("怎么处理数据流", top_k=5)
        # results: [{"id":..., "file_path":..., "score":..., "sources":[...]}]
    """

    def __init__(
        self,
        rrf_k: int = 60,
        rerank_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        rerank_timeout_ms: int = 300,
        rerank_max_candidates: int = 10,
        rerank_enabled: bool = True,
        hybrid_top_k: int = 5,
        vector_top_k: int = 20,
        bm25_top_k: int = 20,
        symbol_top_k: int = 10,
        rrf_candidate_k: int = 30,
        bm25_index=None,
        vector_store=None,
        embedding_client=None,
        symbol_searcher=None,
        reranker=None,
    ):
        self.rrf_k = rrf_k
        self.hybrid_top_k = hybrid_top_k
        self.vector_top_k = vector_top_k
        self.bm25_top_k = bm25_top_k
        self.symbol_top_k = symbol_top_k
        self.rrf_candidate_k = rrf_candidate_k

        # 可注入依赖（测试用）；生产路径用单例
        self._bm25_index = bm25_index
        self._vector_store = vector_store
        self._embedding_client = embedding_client
        self._symbol_searcher = symbol_searcher or SymbolSearcher(
            vector_store=vector_store
        )
        self._reranker = reranker or CrossEncoderReranker(
            model_name=rerank_model_name,
            timeout_ms=rerank_timeout_ms,
            max_candidates=rerank_max_candidates,
            enabled=rerank_enabled,
        )

    # ------------------------------------------------------------
    # 依赖懒加载
    # ------------------------------------------------------------

    def _get_bm25(self):
        if self._bm25_index is not None:
            return self._bm25_index
        return get_bm25_index()

    def _get_store(self):
        if self._vector_store is not None:
            return self._vector_store
        return get_vector_store()

    def _get_embed_client(self):
        if self._embedding_client is not None:
            return self._embedding_client
        from .embedding_client import get_embedding_client
        return get_embedding_client()

    def _get_symbol_searcher(self) -> SymbolSearcher:
        if self._symbol_searcher is None:
            self._symbol_searcher = SymbolSearcher(
                vector_store=self._vector_store
            )
        return self._symbol_searcher

    # ------------------------------------------------------------
    # RRF 融合
    # ------------------------------------------------------------

    def fuse(
        self,
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        symbol_results: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Candidate]:
        """
        RRF 融合三路结果，返回 Top-K 候选 Candidate 列表（不做 rerank）。

        公式：RRF_score(doc) = Σ 1 / (k + rank_i(doc))，rank 从 1 开始。

        Args:
            vector_results: 向量检索结果（每项含 id/score/file_path/...）
            bm25_results:    BM25 检索结果（每项含 id/score/file_path/...）
            symbol_results:  符号精确检索结果（每项含 id/score/...）
            top_k:           融合后保留的候选数（默认 rrf_candidate_k=30）

        Returns:
            按 rrf_score 降序排列的 Candidate 列表
        """
        if top_k is None:
            top_k = self.rrf_candidate_k

        # chunk_id -> Candidate 累积器
        merged: Dict[str, Candidate] = {}

        def _add(results: List[Dict[str, Any]], source: str, score_field: str):
            for rank, r in enumerate(results, start=1):
                cid = r.get("id")
                if not cid:
                    continue
                cand = merged.get(cid)
                if cand is None:
                    cand = Candidate(
                        chunk_id=cid,
                        file_path=r.get("file_path", ""),
                        symbol_name=r.get("symbol_name", ""),
                        chunk_type=r.get("chunk_type", ""),
                        content=r.get("content", ""),
                        start_line=int(r.get("start_line") or 0),
                        end_line=int(r.get("end_line") or 0),
                    )
                    merged[cid] = cand
                # 累加 RRF 分数
                cand.rrf_score += 1.0 / (self.rrf_k + rank)
                if source not in cand.sources:
                    cand.sources.append(source)
                # 保留各路原始分数（调试/排序辅助）
                if source == "vector":
                    cand.vector_score = float(r.get(score_field, 0.0))
                elif source == "bm25":
                    cand.bm25_score = float(r.get(score_field, 0.0))
                elif source == "symbol":
                    cand.symbol_score = float(r.get(score_field, 0.0))

        _add(vector_results, "vector", "score")
        _add(bm25_results, "bm25", "score")
        _add(symbol_results, "symbol", "score")

        # 按 RRF 分数降序
        ranked = sorted(
            merged.values(), key=lambda c: c.rrf_score, reverse=True
        )
        return ranked[:top_k]

    # ------------------------------------------------------------
    # 候选内容补全
    # ------------------------------------------------------------

    def _enrich_candidates(self, candidates: List[Candidate]) -> None:
        """
        为缺少 content 的候选从 LanceDB 补全元信息。

        BM25 与符号路径通常已带 content（BM25 直接读 LanceDB content 字段，
        SymbolSearcher 通过 fetch_chunks_by_ids 反查）。但为防止某些路径
        返回缺字段的情况，统一对 content 为空的 candidate 用 chunk_id 反查。
        """
        need = [c for c in candidates if not c.content]
        if not need:
            return
        try:
            store = self._get_store()
            chunks = store.fetch_chunks_by_ids([c.chunk_id for c in need])
        except Exception as e:
            logger.debug(f"[FusionReranker] 补全 content 失败: {e}")
            return
        by_id = {c["id"]: c for c in chunks}
        for cand in need:
            chunk = by_id.get(cand.chunk_id)
            if chunk:
                cand.file_path = cand.file_path or chunk.get("file_path", "")
                cand.symbol_name = cand.symbol_name or chunk.get("symbol_name", "")
                cand.chunk_type = cand.chunk_type or chunk.get("chunk_type", "")
                cand.content = chunk.get("content", "")
                if not cand.start_line:
                    cand.start_line = int(chunk.get("start_line") or 0)
                if not cand.end_line:
                    cand.end_line = int(chunk.get("end_line") or 0)

    # ------------------------------------------------------------
    # 三路召回
    # ------------------------------------------------------------

    def _vector_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """向量路径：query -> embedding -> LanceDB 向量检索"""
        try:
            store = self._get_store()
            if not store.is_table_exists():
                return []
            embed_client = self._get_embed_client()
            query_vec = embed_client.embed([query])[0]
            return store.search(query_vec, top_k=top_k)
        except Exception as e:
            logger.warning(f"[FusionReranker] 向量检索失败: {e}")
            return []

    def _bm25_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """BM25 路径"""
        try:
            bm25 = self._get_bm25()
            return bm25.search(query, top_k=top_k)
        except Exception as e:
            logger.warning(f"[FusionReranker] BM25 检索失败: {e}")
            return []

    def _symbol_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """符号精确检索路径"""
        try:
            return self._get_symbol_searcher().search(query, top_k=top_k)
        except Exception as e:
            logger.warning(f"[FusionReranker] 符号检索失败: {e}")
            return []

    # ------------------------------------------------------------
    # 完整混合检索流程
    # ------------------------------------------------------------

    def hybrid_search(
        self,
        query: str,
        top_k: Optional[int] = None,
        include_meta: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        完整流程：三路召回 -> RRF 融合 -> Cross-Encoder 重排序 -> Top-K。

        Args:
            query:        用户查询
            top_k:        最终返回数（默认 hybrid_top_k=5）
            include_meta: 是否在结果中带 rrf_score/rerank_score/sources 等元信息

        Returns:
            候选 chunk 列表，每项含 id/file_path/symbol_name/chunk_type/
            content/start_line/end_line/score（=rerank_score 或 rrf_score），
            include_meta=True 时额外带 sources/rrf_score/rerank_score/rerank_status
        """
        if top_k is None:
            top_k = self.hybrid_top_k

        # S5 第 49-50 天：检索结果缓存 + Singleflight 防击穿
        # 仅对 include_meta=False 的常规检索缓存（chat.py 路径）。
        # get_or_compute 内部：查缓存 → 命中直接返回；未命中时获取 per-key 锁，
        # 双重检查后执行 compute_fn 并写入缓存。高并发下同一 Query 只计算一次，
        # 其余并发请求等待结果，避免 Embedding + Rerank 重复计算导致延迟飙升。
        if not include_meta:
            try:
                from .retrieval_cache import get_retrieval_cache, make_cache_key
                cache = get_retrieval_cache()
                cache_key = make_cache_key(query, top_k)

                def _do_search():
                    return self._hybrid_search_impl(query, top_k, include_meta)

                return cache.get_or_compute(cache_key, _do_search)
            except Exception as e:
                logger.debug(
                    f"[FusionReranker] 检索缓存层异常（降级为直接检索）: {e}"
                )

        return self._hybrid_search_impl(query, top_k, include_meta)

    def _hybrid_search_impl(
        self,
        query: str,
        top_k: int,
        include_meta: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        hybrid_search 的实际检索实现（不含缓存逻辑）。

        流程：三路召回 -> RRF 融合 -> 补全 content -> Cross-Encoder 重排序 -> 输出。
        被 get_or_compute 作为 compute_fn 调用，或在缓存禁用/异常时直接调用。
        """
        # 1. 三路召回
        vector_results = self._vector_search(query, self.vector_top_k)
        bm25_results = self._bm25_search(query, self.bm25_top_k)
        symbol_results = self._symbol_search(query, self.symbol_top_k)

        # 2. RRF 融合
        candidates = self.fuse(
            vector_results, bm25_results, symbol_results,
            top_k=self.rrf_candidate_k,
        )
        if not candidates:
            return []

        # 3. 补全 content（Cross-Encoder 需要 (query, content) 对）
        self._enrich_candidates(candidates)

        # 4. Cross-Encoder 重排序（带超时熔断）
        reranked, status = self._reranker.rerank(query, candidates, top_k=top_k)

        # 5. 输出
        results: List[Dict[str, Any]] = []
        for c in reranked:
            d = {
                "id": c.chunk_id,
                "file_path": c.file_path,
                "symbol_name": c.symbol_name,
                "chunk_type": c.chunk_type,
                "content": c.content,
                "start_line": c.start_line,
                "end_line": c.end_line,
                "score": c.rerank_score if c.rerank_score is not None else c.rrf_score,
            }
            if include_meta:
                d["sources"] = list(c.sources)
                d["rrf_score"] = c.rrf_score
                d["rerank_score"] = c.rerank_score
                d["rerank_status"] = status
            results.append(d)
        return results


# ============================================================
# 单例工厂
# ============================================================

_fusion_reranker: Optional[FusionReranker] = None


def get_fusion_reranker() -> FusionReranker:
    """
    获取 FusionReranker 单例。

    配置从 app.config.settings 读取：
      - RRF_K / HYBRID_SEARCH_TOP_K
      - VECTOR_SEARCH_TOP_K（向量召回数）
      - BM25_DEFAULT_TOP_K（BM25 召回数）
      - SYMBOL_SEARCH_TOP_K
      - RRF_CANDIDATE_K
      - RERANK_MODEL / RERANK_TIMEOUT_MS / RERANK_MAX_CANDIDATES / RERANK_ENABLED
    """
    global _fusion_reranker
    if _fusion_reranker is None:
        try:
            from app.config import settings
            _fusion_reranker = FusionReranker(
                rrf_k=settings.RRF_K,
                rerank_model_name=settings.RERANK_MODEL,
                rerank_timeout_ms=settings.RERANK_TIMEOUT_MS,
                rerank_max_candidates=settings.RERANK_MAX_CANDIDATES,
                rerank_enabled=settings.RERANK_ENABLED,
                hybrid_top_k=settings.HYBRID_SEARCH_TOP_K,
                vector_top_k=settings.VECTOR_SEARCH_TOP_K,
                bm25_top_k=settings.BM25_DEFAULT_TOP_K,
                symbol_top_k=settings.SYMBOL_SEARCH_TOP_K,
                rrf_candidate_k=settings.RRF_CANDIDATE_K,
            )
        except Exception as e:
            logger.warning(f"[FusionReranker] 读取配置失败，使用默认值: {e}")
            _fusion_reranker = FusionReranker()
    return _fusion_reranker


def hybrid_search(
    query: str,
    top_k: int = 5,
    include_meta: bool = False,
) -> List[Dict[str, Any]]:
    """
    便捷函数：执行三路混合检索 + RRF + Rerank。

    S5 第 43-44 天指定接口。供 chat 接口的 retrieval_config.auto_context
    调用，自动检索相关代码注入 Prompt。

    Args:
        query:        用户查询
        top_k:        最终返回数（默认 5）
        include_meta: 是否带 RRF/Rerank 元信息（调试与测试用）

    Returns:
        候选 chunk 列表，详见 FusionReranker.hybrid_search
    """
    return get_fusion_reranker().hybrid_search(
        query, top_k=top_k, include_meta=include_meta
    )
