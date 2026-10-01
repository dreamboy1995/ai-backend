# app/api/symbols.py
# S5 第 43-44 天：符号实时补全接口
# - GET /v1/symbols/search?q=<前缀>&limit=<数量>
#
# 用户在 Chat UI 中输入 # 后调用本接口，实时弹出匹配的符号名
# （如输入"Da"返回 DataProcessor / DatabaseConnector）。
# 验收要求：300ms 内弹出候选列表。
#
# 复用 fusion_reranker.SymbolSearcher 的精确/前缀匹配能力，
# Day 45-46 编写完整的 symbol_exact_matcher.py 后可改为委托。

import logging

from fastapi import APIRouter, HTTPException, Query

from app.config import settings
from app.models.schemas import SymbolItem, SymbolSearchResponse
from app.services.code_index.fusion_reranker import SymbolSearcher
from app.services.code_index.index_service import get_index_service

logger = logging.getLogger(__name__)

router = APIRouter()

# 复用 SymbolSearcher 的 token 提取与前缀匹配能力，
# 但本接口不需要从 LanceDB 反查 chunk content，直接走 IndexService 符号表
_symbol_searcher = SymbolSearcher()


@router.get("/search", response_model=SymbolSearchResponse)
async def search_symbols(
    q: str = Query(..., min_length=1, description="符号名前缀或完整名（如 Da / DataProcessor）"),
    limit: int = Query(
        default=settings.SYMBOL_SEARCH_TOP_K,
        ge=1,
        le=50,
        description="返回候选数量上限",
    ),
):
    """
    符号实时补全。

    流程：
      1. 从 q 中提取符号 token（#Tag 直接取后续标识符；普通文本识别驼峰/下划线）
      2. 遍历 IndexService 内存符号表，做精确 + 前缀匹配
      3. 排序：精确 > 前缀；同类型按符号名长度升序
      4. 返回 [{name, type, file_path, line}, ...]

    验收：输入"#Da"应在 300ms 内返回 DataProcessor 和 DatabaseConnector。
    """
    svc = get_index_service()

    tokens = SymbolSearcher._extract_symbol_tokens(q)
    if not tokens:
        # 没有可识别的 token（如纯数字/标点），直接返回空
        return SymbolSearchResponse(query=q, limit=limit, total=0, symbols=[])

    matches = []  # [(file_path, Symbol, "exact"|"prefix")]
    seen_keys = set()

    with svc._lock:
        file_tables = list(svc._index.items())

    for file_path, table in file_tables:
        for sym in table.symbols:
            name = sym.name
            if not name:
                continue
            # 精确匹配（任一 token 命中即记一次，不重复）
            for tok in tokens:
                key = (file_path, name, tok)
                if key in seen_keys:
                    continue
                if name == tok or name.lower() == tok.lower():
                    matches.append((file_path, sym, "exact"))
                    seen_keys.add(key)
                    break
            else:
                # 前缀匹配（仅当未精确命中时）
                for tok in tokens:
                    key = (file_path, name, tok, "prefix")
                    if key in seen_keys:
                        continue
                    if (
                        name.lower().startswith(tok.lower())
                        and len(tok) >= 1
                    ):
                        matches.append((file_path, sym, "prefix"))
                        seen_keys.add(key)
                        break

    if not matches:
        return SymbolSearchResponse(query=q, limit=limit, total=0, symbols=[])

    priority = {"exact": 0, "prefix": 1}
    matches.sort(key=lambda x: (priority[x[2]], len(x[1].name)))
    matches = matches[:limit]

    items = [
        SymbolItem(
            name=sym.name,
            type=(
                sym.symbol_type.value
                if hasattr(sym.symbol_type, "value")
                else str(sym.symbol_type)
            ),
            file_path=file_path,
            line=sym.start_line,
        )
        for file_path, sym, _ in matches
    ]
    return SymbolSearchResponse(query=q, limit=limit, total=len(items), symbols=items)
