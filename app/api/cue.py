# app/api/cue.py
# S6 第 57-58 天：Cue 编辑位置预测接口
# - POST /v1/cue/suggest
#     插件将最近的编辑上下文（文件路径 + 改动行 + 动作 + 符号名）发给后端，
#     后端利用 S4 构建的 AST 依赖图（Call Graph）与符号表返回
#     "可能受影响的文件列表"，作为插件端 Cue 灰色箭头提示的数据来源。
#
# 委托给 symbol_exact_matcher.SymbolExactMatcher 实现：
#   - rename / delete → find_callers（Call Graph 反向查询）
#   - add_method     → find_related_symbols（符号表同名查找）

import logging
from typing import List

from fastapi import APIRouter

from app.config import settings
from app.models.schemas import (
    CueSuggestRequest,
    CueSuggestResponse,
    CueSuggestionItem,
)
from app.services.code_index.symbol_exact_matcher import (
    get_symbol_exact_matcher,
    normalize_path,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/suggest", response_model=CueSuggestResponse)
async def suggest_cue_positions(req: CueSuggestRequest):
    """
    Cue 编辑位置预测（S6 第 57-58 天）。

    插件将最近的编辑上下文（文件路径 + 改动行 + 动作类型 + 符号名）
    发给后端，后端利用 S4 构建的 AST 依赖图（Call Graph）和符号表，
    返回"可能受影响的文件列表"——即用户可能下一步需要编辑的位置。

    三种 action 对应插件端三条启发式规则（详见 Sprint_6.md 第 57-58 天）：
      - rename:    规则 A（依赖跟随）。用户改了一个函数名，后端返回所有
                    调用该符号的跨文件位置，提示用户同步更新调用方。
      - delete:    类似 rename，但符号是被删除的，调用方需要替换或移除。
      - add_method: 规则 C（相似结构补全）。用户新增了一个方法，后端返回
                    其他文件中已存在的同名方法，提示"是否也要修改这里？"。
                    （新方法尚未被调用，find_callers 召回为空，故改用
                    符号表查找同名定义。）

    风险预警应对（S6 关键技术预研 - Cue 规则误报）：
      启发式规则必然有误报。后端通过以下策略缓解：
        1. 限定返回数量上限（settings.CUE_SUGGEST_MAX，默认 20），
           避免编辑器渲染过多灰色箭头干扰用户。
        2. 排除用户当前编辑的文件（同文件由插件 Rule A 自行处理，
           避免跨进程重复提示）。
        3. 排除正在编辑的那一行（req.modified_line），防止"原地提示"。
      弱视觉设计（灰色小点，而非亮色大按钮）与"一键关闭 Cue 提示"
      开关由插件端在设置面板实现；后端仅提供数据，不强制视觉呈现。

    验收场景（Sprint_6.md）：
      在 UserService 中重命名 getUser 为 fetchUser 后，后端返回的候选列表
      中包含 AdminService.js 第 22 行（AdminService 中调用了 getUser）。
    """
    matcher = get_symbol_exact_matcher()

    # 排除用户当前编辑的文件（同文件由插件 Rule A 处理），
    # 同时排除正在编辑的那一行（避免"原地提示"）
    exclude_file = normalize_path(req.file_path) if req.file_path else None

    suggestions: List[CueSuggestionItem] = []

    if req.action in ("rename", "delete"):
        # 规则 A：跨文件调用方跟随（Call Graph 反向查询）
        callers = matcher.find_callers(req.symbol_name, top_k=settings.CUE_SUGGEST_MAX)
        reason = (
            "此函数调用了被改名的符号" if req.action == "rename"
            else "此函数调用了被删除的符号"
        )
        for c in callers:
            fp = c.get("file_path", "")
            line = c.get("line", 0)
            # 排除当前编辑的文件（同文件由插件 Rule A 处理，避免重复提示）
            if exclude_file and fp == exclude_file:
                continue
            # 防御性排除正在编辑的那一行（避免"原地提示"）
            if fp == exclude_file and line == req.modified_line:
                continue
            if line < 1:
                continue
            suggestions.append(CueSuggestionItem(
                file_path=fp,
                line=line,
                reason=reason,
            ))
    elif req.action == "add_method":
        # 规则 C：跨文件同名方法查找（符号表查询）
        # 新方法尚未被调用，find_callers 召回为空，故改用符号表
        related = matcher.find_related_symbols(
            req.symbol_name,
            exclude_file=req.file_path,
            top_k=settings.CUE_SUGGEST_MAX,
        )
        reason = "存在同名方法，可能需要同步修改"
        for r in related:
            suggestions.append(CueSuggestionItem(
                file_path=r["file_path"],
                line=r["line"],
                reason=reason,
            ))

    # 风险预警：误报控制 - 限制返回数量
    suggestions = suggestions[:settings.CUE_SUGGEST_MAX]

    logger.debug(
        f"[Cue] suggest action={req.action} symbol={req.symbol_name} "
        f"file={req.file_path}:{req.modified_line} → {len(suggestions)} 条建议"
    )

    return CueSuggestResponse(
        action=req.action,
        symbol_name=req.symbol_name,
        total=len(suggestions),
        suggestions=suggestions,
    )
