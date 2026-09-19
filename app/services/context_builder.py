"""
上下文拼装器（S2 第 15-16 天：后端上下文拼装 & 系统提示词工程）

核心职责：
1. 将前端传来的 contexts 数组格式化为结构化 XML 标签，插入 System Prompt：
   <context_files>
     <file path="main.py" lang="python">def hello():
    print("hello")</file>
   </context_files>
2. 按优先级裁剪超长上下文：
   - 用户主动 @ 的 file / selection 优先保留
   - 插件自动附带的 implicit 当前文件可截断或丢弃
3. 将拼接后的 System Prompt 纳入 Token 预算计算，保证总 Token 不超限。

Token 预算分配（与 SessionService 协同）：
- 总预算 SESSION_TOKEN_BUDGET = 8000
- 裁剪阈值 = 8000 * (1 - 0.2) = 6400（留 20% 余量，S2 关键技术预研要求）
- 上下文预算 = 6400 * SESSION_CONTEXT_TOKEN_RATIO(0.5) = 3200
- 历史预算 = 6400 - 上下文实际占用（动态），由 SessionService.get_history(reserved_tokens=) 保证
"""

import logging
import html
from typing import List, Optional

from app.config import settings
from app.models.schemas import ContextItem
from app.services.session import count_tokens

logger = logging.getLogger(__name__)

# 基础系统提示词：定义助手角色 + 强调代码块格式（配合第 17-18 天 Apply 功能）
BASE_SYSTEM_PROMPT = (
    "你是一个专业的编程助手，擅长代码理解、生成、调试和重构。\n"
    "\n"
    "【输出规范】\n"
    "请始终使用 ```language 代码围栏格式输出代码块（例如 ```python），"
    "以便前端正确识别语言并支持一键插入编辑器。避免使用无语言标签的代码块。"
)

# 上下文类型优先级：数值越小优先级越高
# file（用户主动 @ 整个文件）和 selection（用户选中的代码）优先级最高，必须保留；
# implicit（插件自动附带的当前激活文件）优先级最低，可被截断或丢弃。
_CONTEXT_PRIORITY = {"file": 0, "selection": 1, "implicit": 2}


class ContextBuilder:
    """
    上下文拼装器。

    用法：
        builder = ContextBuilder()
        system_prompt = builder.build_system_prompt(contexts, existing_system_content)
        system_tokens = builder.last_system_tokens  # 供 SessionService 预留预算
    """

    def __init__(
        self,
        total_token_budget: int = None,
        token_margin: float = None,
        context_ratio: float = None,
    ):
        self._total_budget = (
            total_token_budget
            if total_token_budget is not None
            else settings.SESSION_TOKEN_BUDGET
        )
        self._token_margin = (
            token_margin if token_margin is not None else settings.SESSION_TOKEN_MARGIN
        )
        self._context_ratio = (
            context_ratio
            if context_ratio is not None
            else settings.SESSION_CONTEXT_TOKEN_RATIO
        )

        # 触发裁剪的总阈值（留 20% 余量）
        self._total_threshold = int(self._total_budget * (1 - self._token_margin))
        # 上下文专属预算（占阈值的一部分）
        self._context_budget = int(self._total_threshold * self._context_ratio)

        # 记录最近一次构建的系统提示词 Token 数，供 SessionService 预留预算
        self.last_system_tokens: int = 0

        logger.info(
            f"ContextBuilder 初始化完成：总预算={self._total_budget}, "
            f"阈值={self._total_threshold}, 上下文预算={self._context_budget}, "
            f"上下文比例={self._context_ratio}"
        )

    # ------------------------------------------------------------------
    # 对外主接口
    # ------------------------------------------------------------------
    def build_system_prompt(
        self,
        contexts: Optional[List[ContextItem]],
        existing_system_content: Optional[str] = None,
    ) -> str:
        """
        构建包含上下文的 System Prompt。

        流程：
        1. 对 contexts 按优先级裁剪，使其不超过上下文 Token 预算。
        2. 将裁剪后的上下文格式化为 <context_files> XML 标签。
        3. 与已有系统提示词（若有）拼接，返回最终 System Prompt 内容。

        同时更新 self.last_system_tokens，供调用方（SessionService）预留 Token 预算。
        """
        # 步骤 1：裁剪上下文
        trimmed_contexts = self.truncate_contexts(contexts or [])

        # 步骤 2：格式化 XML
        context_xml = self._format_contexts_xml(trimmed_contexts)

        # 步骤 3：拼接系统提示词
        parts: List[str] = []
        if existing_system_content:
            parts.append(existing_system_content)
        else:
            parts.append(BASE_SYSTEM_PROMPT)

        if context_xml:
            parts.append(context_xml)

        system_prompt = "\n\n".join(parts)

        # 记录 Token 数，供 SessionService 预留预算
        self.last_system_tokens = count_tokens(
            [{"role": "system", "content": system_prompt}]
        )

        logger.info(
            f"[ContextBuilder] 系统提示词构建完成: "
            f"上下文条数={len(trimmed_contexts)}/{len(contexts or [])}, "
            f"系统提示词Token≈{self.last_system_tokens}, "
            f"上下文预算={self._context_budget}"
        )
        if trimmed_contexts:
            for c in trimmed_contexts:
                logger.debug(
                    f"[ContextBuilder] 已纳入上下文: type={c.type}, "
                    f"path={c.file_path}, lang={c.language}, "
                    f"内容长度={len(c.content_snippet)}"
                )

        return system_prompt

    # ------------------------------------------------------------------
    # 上下文裁剪（按优先级）
    # ------------------------------------------------------------------
    def truncate_contexts(self, contexts: List[ContextItem]) -> List[ContextItem]:
        """
        按优先级裁剪上下文列表，使其总 Token 不超过上下文预算。

        策略：
        1. 按优先级排序：file -> selection -> implicit。
        2. 依次尝试将上下文纳入预算：
           - 若完整纳入后未超预算，直接保留。
           - 若超预算：
             * 对 implicit 类型：截断内容以适应剩余预算（保留头部）。
             * 对 file / selection 类型：不截断内容，跳过本条及后续所有
               （因为后续优先级更低，不值得为保留低优先级而截断高优先级）。
        """
        if not contexts:
            return []

        sorted_contexts = sorted(
            contexts, key=lambda c: _CONTEXT_PRIORITY.get(c.type, 99)
        )

        result: List[ContextItem] = []
        used_tokens = 0

        for ctx in sorted_contexts:
            ctx_tokens = self._count_context_tokens(ctx)

            if used_tokens + ctx_tokens <= self._context_budget:
                result.append(ctx)
                used_tokens += ctx_tokens
                continue

            # 超出预算，根据类型决定是否截断
            if ctx.type == "implicit":
                remaining = self._context_budget - used_tokens
                truncated = self._truncate_content(ctx, remaining)
                if truncated is not None:
                    result.append(truncated)
                    used_tokens += self._count_context_tokens(truncated)
                # implicit 截断后无论是否成功都停止（后续优先级更低）
                break
            else:
                # file / selection 不截断，跳过本条及后续
                logger.debug(
                    f"[ContextBuilder] 上下文超预算，跳过: type={ctx.type}, "
                    f"path={ctx.file_path}, 需要≈{ctx_tokens}, 剩余≈{self._context_budget - used_tokens}"
                )
                break

        if len(result) < len(contexts):
            dropped = len(contexts) - len(result)
            logger.info(
                f"[ContextBuilder] 上下文裁剪: 原始 {len(contexts)} 条 -> 保留 {len(result)} 条, "
                f"丢弃 {dropped} 条, 占用 Token≈{used_tokens}/{self._context_budget}"
            )

        return result

    # ------------------------------------------------------------------
    # XML 格式化
    # ------------------------------------------------------------------
    @staticmethod
    def _format_contexts_xml(contexts: List[ContextItem]) -> str:
        """
        将上下文列表格式化为结构化 XML 标签。

        输出示例：
            <context_files>
            <file path="main.py" lang="python">def hello():
                print("hello")</file>
            </context_files>
        """
        if not contexts:
            return ""

        file_tags: List[str] = []
        for ctx in contexts:
            # 转义 XML 特殊字符，防止文件内容中的 < > & 破坏标签结构
            escaped_content = html.escape(ctx.content_snippet, quote=False)
            lang_attr = f' lang="{ctx.language}"' if ctx.language else ""
            file_tags.append(
                f'<file path="{html.escape(ctx.file_path, quote=True)}"{lang_attr}>'
                f"{escaped_content}</file>"
            )

        return "<context_files>\n" + "\n".join(file_tags) + "\n</context_files>"

    # ------------------------------------------------------------------
    # Token 计数与内容截断
    # ------------------------------------------------------------------
    @staticmethod
    def _count_context_tokens(ctx: ContextItem) -> int:
        """估算单条上下文占用的 Token（包含 XML 标签开销）。"""
        return count_tokens([{"role": "user", "content": ctx.content_snippet}])

    @staticmethod
    def _truncate_content(ctx: ContextItem, max_tokens: int) -> Optional[ContextItem]:
        """
        截断上下文内容使其不超过 max_tokens。

        优先使用 tiktoken 精确截断；若不可用则按字符数 / 4 估算截断。
        截断后在末尾添加截断标记，提示模型内容不完整。
        """
        if max_tokens <= 0:
            return None

        content = ctx.content_snippet
        truncated_content = _truncate_text_to_tokens(content, max_tokens)

        if truncated_content == content:
            # 没截断（说明实际 Token 没超限），直接返回原对象
            return ctx

        # 添加截断标记
        truncated_content += "\n... [内容已截断]"
        return ContextItem(
            type=ctx.type,
            file_path=ctx.file_path,
            content_snippet=truncated_content,
            language=ctx.language,
        )


def _truncate_text_to_tokens(text: str, max_tokens: int) -> str:
    """
    将文本截断到指定 Token 数以内。

    使用 tiktoken cl100k_base 编码精确截断；若不可用则降级为字符长度 / 4 估算。
    """
    if max_tokens <= 0:
        return ""

    # 尝试使用 tiktoken 精确截断
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        # 预留几个 token 给截断标记
        truncated_tokens = tokens[: max(0, max_tokens - 5)]
        return encoding.decode(truncated_tokens)
    except Exception:
        # 降级：按字符数 / 4 估算
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text
        return text[: max(0, max_chars - 20)]
