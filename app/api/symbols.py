# app/api/symbols.py
# S5 第 45-46 天：符号精确检索 & 依赖图增强接口
# - GET /v1/symbols/search?q=<前缀>&limit=<数量>
#     符号实时补全（用户输入 # 后调用，300ms 内弹出候选）
# - GET /v1/symbols/callers?name=<符号名>&limit=<数量>
#     反向依赖查询：谁调用了指定符号（利用 S4 Call Graph）
# - GET /v1/symbols/definition?name=<符号名>
#     精确定位符号定义（file_path + 行号范围）
#
# 委托给 symbol_exact_matcher.SymbolExactMatcher 实现。

import logging

from fastapi import APIRouter, HTTPException, Query

from app.config import settings
from app.models.schemas import (
    SymbolCallerItem,
    SymbolCallersResponse,
    SymbolDefinitionResponse,
    SymbolItem,
    SymbolSearchResponse,
)
from app.services.code_index.symbol_exact_matcher import get_symbol_exact_matcher

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/search", response_model=SymbolSearchResponse)
async def search_symbols(
    q: str = Query("", description="符号名前缀或完整名（如 Da / DataProcessor / #DataProcessor）。为空时返回空列表。"),
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
    注意：q 为空时返回空列表（HTTP 200），避免前端清空输入框时触发 422。
    """
    matcher = get_symbol_exact_matcher()
    if not q:
        return SymbolSearchResponse(query=q, limit=limit, total=0, symbols=[])
    symbols = matcher.suggest_symbols(q, limit=limit)
    return SymbolSearchResponse(
        query=q,
        limit=limit,
        total=len(symbols),
        symbols=[SymbolItem(**s) for s in symbols],
    )


@router.get("/definition", response_model=SymbolDefinitionResponse)
async def get_symbol_definition(
    name: str = Query(..., min_length=1, description="符号名（如 DataProcessor / save）"),
):
    """
    精确定位符号定义。

    用于验收场景："输入 #DataProcessor，后端能直接定位到定义该类的文件路径和行号范围"。

    仅做精确匹配（name 完全相等，大小写不敏感），返回第一个命中的符号。
    """
    matcher = get_symbol_exact_matcher()
    result = matcher.get_symbol_definition(name)
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"未找到符号 '{name}' 的定义，请先调用 POST /v1/index/start 触发索引",
        )
    return SymbolDefinitionResponse(
        name=result["name"],
        type=result["type"],
        file_path=result["file_path"],
        start_line=result["start_line"],
        end_line=result["end_line"],
    )


@router.get("/callers", response_model=SymbolCallersResponse)
async def get_symbol_callers(
    name: str = Query(..., min_length=1, description="被调用的符号名（如 save / format_data）"),
    limit: int = Query(
        default=20,
        ge=1,
        le=100,
        description="返回调用记录数量上限",
    ),
):
    """
    反向依赖查询：谁调用了指定符号。

    利用 S4 构建的 Call Graph（DependencyGraph），遍历所有 call 边，
    找出 target 匹配 symbol_name 的调用记录。

    匹配规则：
      - target == name（精确，如 "save"）
      - target 以 "." + name 结尾（如 "obj.save"、"utils.save" 匹配 "save"）

    验收场景：输入 "谁调用了 save()"，返回 main.py 第 15 行和 utils.py 第 88 行。
    """
    matcher = get_symbol_exact_matcher()
    callers = matcher.find_callers(name, top_k=limit)
    return SymbolCallersResponse(
        symbol_name=name,
        total=len(callers),
        callers=[SymbolCallerItem(**c) for c in callers],
    )
