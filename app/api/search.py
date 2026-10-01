# app/api/search.py
# S4 第 35-36 天：代码语义搜索接口
# - GET /v1/search?q=<关键词>&top_k=<数量>
#
# 通过 LanceDB 向量检索，返回与查询语义最相关的代码切片（函数/类/import 块）。

import logging

from fastapi import APIRouter, HTTPException, Query

from app.config import settings
from app.models.schemas import SearchResponse, SearchResultItem
from app.services.code_index.embedding_client import get_embedding_client
from app.services.code_index.vector_store import get_vector_store

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("", response_model=SearchResponse)
async def search_code(
    q: str = Query(..., min_length=1, description="搜索关键词（支持语义匹配）"),
    top_k: int = Query(default=settings.VECTOR_SEARCH_TOP_K, ge=1, le=100, description="返回结果数量"),
):
    """
    代码语义搜索

    将查询文本向量化后，在 LanceDB 向量库中检索最相似的代码切片。
    即使关键词与函数名不完全匹配（如搜 "排序" 能返回 sort 函数），也能通过语义召回。
    """
    store = get_vector_store()
    if not store.is_table_exists():
        raise HTTPException(
            status_code=409,
            detail="代码库尚未建立向量索引，请先调用 POST /v1/index/start 触发索引",
        )

    try:
        embed_client = get_embedding_client()
        query_vector = embed_client.embed([q])[0]
    except Exception as e:
        logger.error(f"[SearchAPI] 查询向量化失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"查询向量化失败: {e}")

    try:
        results = store.search(query_vector, top_k=top_k)
    except Exception as e:
        logger.error(f"[SearchAPI] 向量检索失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"向量检索失败: {e}")

    items = [SearchResultItem(**r) for r in results]
    return SearchResponse(
        query=q,
        top_k=top_k,
        total=len(items),
        results=items,
    )
