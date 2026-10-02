"""
智能上下文组装器（S5 第 47-48 天）

对 hybrid_search（向量+BM25+符号+RRF+Cross-Encoder）召回的 Top-K Chunk 做：
  1. 位置权重打分：光标所在文件的 Chunk 权重 +30%，其余正常。
  2. 智能压缩（而非简单截断）：
       - 函数/类 Chunk 行数 <= COMPRESS_LINES(100)：保留完整代码。
       - > 100 行：仅保留函数签名（def/class 行）+ 注释 + 前 HEAD_LINES 行 +
         后 TAIL_LINES 行，中间用语言适配的注释 `... 省略 N 行 ...` 替代。
  3. 动态 Token 预算控制：
       预算 = min(model_context_window * FILL_RATIO(0.7), MAX_BUDGET(8000))
       压缩后仍超预算时，按"低分优先丢弃"策略从最低相关度的 Chunk 开始剔除。
  4. 结构化 XML 组装（类似 S2 的 <context_files><file ...>...</file></context_files>）。
  5. 日志打印组装前后的 Token 计数对比。

与 S2 ContextBuilder 的分工：
  - ContextBuilder（S2）：处理用户主动 @ 的 file/selection 上下文，按优先级裁剪。
  - ContextAssembler（S5）：处理自动检索召回的 Chunk，做智能压缩 + 动态预算。
  两者产出的 ContextItem 最终都交给 ContextBuilder 统一拼装 System Prompt。
"""

import html
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.models.schemas import ContextItem, ReferenceItem
from app.services.session import count_tokens

logger = logging.getLogger(__name__)


# ============================================================
# 语言注释符号映射（用于压缩时的省略标记）
# ============================================================
_COMMENT_PREFIX = {
    "python": "#",
    "javascript": "//",
    "typescript": "//",
    "java": "//",
    "go": "//",
    "rust": "//",
    "c": "//",
    "cpp": "//",
    "csharp": "//",
    "php": "//",
    "ruby": "#",
    "swift": "//",
    "kotlin": "//",
    "scala": "//",
}


def _detect_language(file_path: str) -> str:
    """根据扩展名识别语言（小写）。"""
    if not file_path:
        return ""
    lower = file_path.lower()
    ext_map = {
        ".py": "python", ".js": "javascript", ".jsx": "javascript",
        ".ts": "typescript", ".tsx": "typescript",
        ".java": "java", ".go": "go", ".rs": "rust", ".rb": "ruby",
        ".c": "c", ".h": "c", ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp",
        ".cs": "csharp", ".php": "php", ".swift": "swift",
        ".kt": "kotlin", ".scala": "scala",
    }
    for ext, lang in ext_map.items():
        if lower.endswith(ext):
            return lang
    return ""


def _comment_prefix(language: str) -> str:
    """返回语言的单行注释前缀，未知语言回退为 //。"""
    return _COMMENT_PREFIX.get(language or "", "//")


def _is_signature_line(line: str, language: str) -> bool:
    """
    判断一行是否是函数/类签名行。

    用于压缩时定位"签名行"（保留的第一行）。
    覆盖主流语言的 def/function/class 声明。
    """
    stripped = line.strip()
    if not stripped:
        return False
    # Python: def / class / async def
    if language == "python":
        return bool(re.match(r"^(async\s+)?def\s+", stripped)) or \
               stripped.startswith("class ")
    # 花括号语言：function / class / 可见性修饰符 + function/class
    pattern = re.compile(
        r"^("
        r"(public|private|protected|static|final|abstract|async|export|default)\s+)*"
        r"(function|class)\s+"
    )
    if pattern.match(stripped):
        return True
    # 箭头函数 / 方法简写（如 foo() { 或 const foo = () => {）
    if re.search(r"(\)\s*\{|=>\s*\{)", stripped):
        return True
    # Go: func
    if language == "go" and stripped.startswith("func "):
        return True
    # Rust: fn
    if language == "rust" and stripped.startswith("fn "):
        return True
    # Ruby: def
    if language == "ruby" and stripped.startswith("def "):
        return True
    return False


def _is_comment_line(line: str, language: str) -> bool:
    """判断一行是否是注释行（含 docstring 起始行）。"""
    stripped = line.strip()
    if not stripped:
        return False
    prefix = _comment_prefix(language)
    if stripped.startswith(prefix):
        return True
    # Python docstring："""...""" 或 '''...'''（单行或多行起始）
    if language == "python":
        if stripped.startswith('"""') or stripped.startswith("'''"):
            return True
    # 块注释起始 /*
    if stripped.startswith("/*") or stripped.startswith("*"):
        return True
    return False


# ============================================================
# 组装结果数据结构
# ============================================================


@dataclass
class AssemblyStats:
    """组装统计信息（用于日志与验收）。"""
    original_chunks: int = 0          # 输入 Chunk 总数
    kept_chunks: int = 0              # 最终保留的 Chunk 数
    dropped_chunks: int = 0           # 因超预算被丢弃的 Chunk 数
    compressed_chunks: int = 0        # 被智能压缩（省略中间行）的 Chunk 数
    original_tokens: int = 0          # 压缩前所有 Chunk 的 Token 估算
    compressed_tokens: int = 0        # 压缩并组装后实际 Token 数
    token_budget: int = 0             # 本次使用的 Token 预算
    cursor_file: str = ""             # 光标所在文件（用于位置权重）

    def to_dict(self) -> dict:
        return {
            "original_chunks": self.original_chunks,
            "kept_chunks": self.kept_chunks,
            "dropped_chunks": self.dropped_chunks,
            "compressed_chunks": self.compressed_chunks,
            "original_tokens": self.original_tokens,
            "compressed_tokens": self.compressed_tokens,
            "token_budget": self.token_budget,
            "cursor_file": self.cursor_file,
        }


@dataclass
class AssemblyResult:
    """上下文组装结果。"""
    contexts: List[ContextItem] = field(default_factory=list)
    references: List[ReferenceItem] = field(default_factory=list)
    stats: AssemblyStats = field(default_factory=AssemblyStats)


# ============================================================
# 主类：ContextAssembler
# ============================================================


class ContextAssembler:
    """
    智能上下文组装器。

    用法：
        assembler = ContextAssembler()
        result = assembler.assemble(
            chunks=retrieved_chunks,        # hybrid_search 的输出
            cursor_file="src/main.py",      # 当前光标所在文件（隐式上下文）
            model_context_window=128000,    # 模型最大上下文长度
        )
        # result.contexts: 压缩后的 ContextItem 列表（交给 ContextBuilder）
        # result.references: SSE references 元数据
        # result.stats: 组装前后 Token 统计
    """

    def __init__(
        self,
        fill_ratio: float = None,
        max_budget: int = None,
        default_budget: int = None,
        cursor_boost: float = None,
        compress_lines: int = None,
        head_lines: int = None,
        tail_lines: int = None,
    ):
        try:
            from app.config import settings
            self._fill_ratio = fill_ratio if fill_ratio is not None else settings.CONTEXT_ASSEMBLY_FILL_RATIO
            self._max_budget = max_budget if max_budget is not None else settings.CONTEXT_ASSEMBLY_MAX_BUDGET
            self._default_budget = default_budget if default_budget is not None else settings.CONTEXT_ASSEMBLY_DEFAULT_BUDGET
            self._cursor_boost = cursor_boost if cursor_boost is not None else settings.CONTEXT_ASSEMBLY_CURSOR_BOOST
            self._compress_lines = compress_lines if compress_lines is not None else settings.CONTEXT_ASSEMBLY_COMPRESS_LINES
            self._head_lines = head_lines if head_lines is not None else settings.CONTEXT_ASSEMBLY_HEAD_LINES
            self._tail_lines = tail_lines if tail_lines is not None else settings.CONTEXT_ASSEMBLY_TAIL_LINES
        except Exception:
            # 配置读取失败时使用 S5 任务文档默认值
            self._fill_ratio = fill_ratio if fill_ratio is not None else 0.7
            self._max_budget = max_budget if max_budget is not None else 8000
            self._default_budget = default_budget if default_budget is not None else 8000
            self._cursor_boost = cursor_boost if cursor_boost is not None else 0.3
            self._compress_lines = compress_lines if compress_lines is not None else 100
            self._head_lines = head_lines if head_lines is not None else 10
            self._tail_lines = tail_lines if tail_lines is not None else 10

    # ------------------------------------------------------------
    # 动态 Token 预算计算
    # ------------------------------------------------------------

    def compute_budget(self, model_context_window: Optional[int] = None) -> int:
        """
        根据模型上下文窗口计算本次组装的 Token 预算。

        预算 = min(model_context_window * fill_ratio, max_budget)
        若 model_context_window 未知或 <=0，回退到 default_budget。

        S5 风险预警：token 预算不能写死，必须根据模型动态调整；
        按 70% 比例填充，留 30% 给对话历史和模型输出。
        """
        if not model_context_window or model_context_window <= 0:
            return self._default_budget
        budget = int(model_context_window * self._fill_ratio)
        # 硬上限：避免超大模型下上下文膨胀
        return min(budget, self._max_budget)

    # ------------------------------------------------------------
    # 位置权重打分 + 排序
    # ------------------------------------------------------------

    def _rescore_and_sort(
        self,
        chunks: List[Dict[str, Any]],
        cursor_file: Optional[str],
    ) -> List[Dict[str, Any]]:
        """
        按位置权重对 Chunk 重新打分并排序。

        光标所在文件的 Chunk：score *= (1 + cursor_boost)
        其余文件：score 不变。

        排序后返回新的列表（不修改原 dict，复制 score 到 _position_score 字段）。
        """
        if not cursor_file:
            # 无光标信息时，按原始分数降序
            return sorted(
                chunks, key=lambda c: float(c.get("score", 0.0)), reverse=True
            )

        # 归一化光标文件路径（统一为 POSIX 正斜杠，便于跨平台比较）
        cursor_norm = cursor_file.replace("\\", "/").lower()

        rescored = []
        for c in chunks:
            fp = (c.get("file_path") or "").replace("\\", "/").lower()
            base_score = float(c.get("score", 0.0))
            if fp == cursor_norm and fp:
                adjusted = base_score * (1.0 + self._cursor_boost)
            else:
                adjusted = base_score
            # 复制一份，避免污染原数据
            new_c = dict(c)
            new_c["_position_score"] = adjusted
            new_c["_is_cursor_file"] = (fp == cursor_norm and bool(fp))
            rescored.append(new_c)

        rescored.sort(key=lambda c: c["_position_score"], reverse=True)
        return rescored

    # ------------------------------------------------------------
    # 智能压缩
    # ------------------------------------------------------------

    def _compress_chunk(self, chunk: Dict[str, Any]) -> tuple:
        """
        对单个 Chunk 做智能压缩。

        策略：
          - 非函数/类 Chunk（import/block 等）或行数 <= compress_lines：
            保留完整内容，不压缩。
          - 函数/类 Chunk 且行数 > compress_lines：
            保留签名行 + 注释/docstring + 前 head_lines 行 + 后 tail_lines 行，
            中间用 `{prefix} ... 省略 N 行 ...` 替代。

        Returns:
            (compressed_content: str, was_compressed: bool)
        """
        content = chunk.get("content", "") or ""
        chunk_type = (chunk.get("chunk_type") or "").lower()
        file_path = chunk.get("file_path", "") or ""
        language = _detect_language(file_path)

        lines = content.split("\n")
        # 仅对函数/类 Chunk 且行数超阈值时压缩
        if chunk_type not in ("function", "class") or len(lines) <= self._compress_lines:
            return content, False

        # ---- 定位签名行 ----
        sig_idx = 0
        for i, line in enumerate(lines):
            if _is_signature_line(line, language):
                sig_idx = i
                break
        signature = lines[sig_idx]

        # ---- 收集签名之后的连续注释/docstring 行 ----
        comment_end = sig_idx + 1
        # Python docstring 处理：若签名下一行是 """ 或 '''，收集到闭合
        if language == "python" and comment_end < len(lines):
            next_stripped = lines[comment_end].strip()
            if next_stripped.startswith('"""') or next_stripped.startswith("'''"):
                quote = next_stripped[:3]
                # 单行 docstring
                if next_stripped.count(quote) >= 2:
                    comment_end += 1
                else:
                    # 多行 docstring，找到闭合
                    comment_end += 1
                    while comment_end < len(lines) and quote not in lines[comment_end]:
                        comment_end += 1
                    if comment_end < len(lines):
                        comment_end += 1  # 包含闭合行

        # 其余连续注释行
        while comment_end < len(lines) and _is_comment_line(lines[comment_end], language):
            comment_end += 1

        comments_block = lines[sig_idx + 1:comment_end]

        # ---- 剩余 body 行 ----
        body_lines = lines[comment_end:]
        if len(body_lines) <= self._head_lines + self._tail_lines:
            # body 本身不长，无需省略
            return content, False

        head = body_lines[: self._head_lines]
        tail = body_lines[-self._tail_lines:] if self._tail_lines > 0 else []
        omitted = len(body_lines) - len(head) - len(tail)
        prefix = _comment_prefix(language)

        # 拼接：签名 + 注释 + head + 省略标记 + tail
        result_lines = [signature]
        result_lines.extend(comments_block)
        result_lines.extend(head)
        result_lines.append(f"{prefix} ... 省略 {omitted} 行 ...")
        result_lines.extend(tail)

        return "\n".join(result_lines), True

    # ------------------------------------------------------------
    # Token 计数辅助
    # ------------------------------------------------------------

    @staticmethod
    def _estimate_content_tokens(content: str) -> int:
        """估算单段内容的 Token 数（含少量 XML 标签开销）。"""
        return count_tokens([{"role": "user", "content": content}])

    # ------------------------------------------------------------
    # 路径归一化与存在性校验（S5 修复：references 路径统一为工作区相对路径）
    # ------------------------------------------------------------

    @staticmethod
    def _normalize_and_validate_path(
        file_path: str, workspace_root: Optional[str]
    ) -> Optional[str]:
        """
        将 chunk 的 file_path 归一化为工作区相对路径，并校验文件真实存在。

        处理规则：
          1. 若 workspace_root 未知（后端尚未索引过任何工作区），无法判断路径
             是否在工作区内，此时不做过滤（保留原路径，由前端处理）。
          2. 使用 index_service.to_workspace_relative 将绝对/相对路径统一转为
             工作区相对路径；路径不在工作区内时返回 None。
          3. 校验文件是否真实存在于 workspace_root 下，不存在则返回 None。

        用于过滤掉向量库中残留的跨工作区脏数据（如系统临时目录文件、
        其他工作区的相对路径），避免返回给前端的 references 指向不存在的文件。

        Args:
            file_path:      chunk 中的文件路径（可能是绝对路径或其他工作区的相对路径）
            workspace_root: 当前工作区根路径

        Returns:
            工作区相对路径（POSIX 正斜杠）；若无效则返回 None。
        """
        if not file_path:
            return None
        if not workspace_root:
            # 工作区未知时不做过滤，保留原路径（降级策略）
            return file_path.replace("\\", "/")

        # 延迟导入避免循环依赖
        from .index_service import to_workspace_relative

        rel = to_workspace_relative(file_path, workspace_root)
        if rel is None:
            return None

        # 校验文件真实存在于工作区
        full_path = os.path.join(workspace_root, rel)
        if not os.path.isfile(full_path):
            return None
        return rel

    @staticmethod
    def _get_workspace_root() -> Optional[str]:
        """
        获取当前工作区根路径（从 IndexService 单例读取）。

        IndexService 在首次索引时会缓存 workspace_root，
        增量更新与检索时复用该值。若尚未索引过任何工作区则返回 None。
        """
        try:
            from .index_service import get_index_service
            return get_index_service().workspace_root
        except Exception:
            return None

    # ------------------------------------------------------------
    # 主入口：assemble
    # ------------------------------------------------------------

    def assemble(
        self,
        chunks: List[Dict[str, Any]],
        cursor_file: Optional[str] = None,
        model_context_window: Optional[int] = None,
    ) -> AssemblyResult:
        """
        执行完整的上下文组装流程。

        流程：
          0. 路径归一化 + 存在性校验：过滤掉不在工作区内或不存在的文件的 chunk
          1. 位置权重打分 + 排序（光标文件 +30%）
          2. 逐 Chunk 智能压缩
          3. 动态 Token 预算控制（低分优先丢弃）
          4. 转为 ContextItem + ReferenceItem
          5. 打印组装前后 Token 计数日志

        Args:
            chunks: hybrid_search 输出的 chunk 列表，每项需含
                    id/file_path/symbol_name/chunk_type/content/
                    start_line/end_line/score
            cursor_file: 当前光标所在文件路径（相对路径），用于位置权重
            model_context_window: 模型最大上下文长度，用于动态预算

        Returns:
            AssemblyResult(contexts, references, stats)
        """
        stats = AssemblyStats(
            original_chunks=len(chunks),
            cursor_file=cursor_file or "",
        )

        if not chunks:
            return AssemblyResult(stats=stats)

        # 步骤 0：路径归一化 + 存在性校验
        # 向量库为全局共享，可能残留其他工作区/系统临时目录的脏数据。
        # 此处统一将 file_path 转为工作区相对路径，并过滤掉不存在的文件，
        # 确保返回给前端的 references 均可正确跳转。
        workspace_root = self._get_workspace_root()
        valid_chunks: List[Dict[str, Any]] = []
        skipped_invalid = 0
        for ch in chunks:
            fp = ch.get("file_path", "") or ""
            norm_fp = self._normalize_and_validate_path(fp, workspace_root)
            if norm_fp is None:
                skipped_invalid += 1
                logger.debug(
                    f"[ContextAssembler] 跳过无效路径 Chunk: "
                    f"file_path={fp}, symbol={ch.get('symbol_name')}"
                )
                continue
            # 用归一化后的工作区相对路径替换原路径
            ch = dict(ch)
            ch["file_path"] = norm_fp
            valid_chunks.append(ch)
        if skipped_invalid:
            logger.info(
                f"[ContextAssembler] 路径校验过滤: 输入 {len(chunks)} 个 Chunk, "
                f"跳过 {skipped_invalid} 个无效路径, 保留 {len(valid_chunks)} 个"
            )
        chunks = valid_chunks
        if not chunks:
            stats.kept_chunks = 0
            return AssemblyResult(stats=stats)

        budget = self.compute_budget(model_context_window)
        stats.token_budget = budget

        # 步骤 1：位置权重打分 + 排序
        sorted_chunks = self._rescore_and_sort(chunks, cursor_file)

        # 步骤 2：逐 Chunk 智能压缩（此时不丢弃，先全部压缩）
        compressed: List[tuple] = []  # (chunk_dict, compressed_content, was_compressed)
        original_tokens = 0
        for ch in sorted_chunks:
            original_tokens += self._estimate_content_tokens(ch.get("content", "") or "")
            content, was_compressed = self._compress_chunk(ch)
            compressed.append((ch, content, was_compressed))
            if was_compressed:
                stats.compressed_chunks += 1
        stats.original_tokens = original_tokens

        # 步骤 3：Token 预算控制——低分优先丢弃
        # 已按位置权重分数降序排列，从后往前（低分）逐个丢弃直到 fit 预算
        kept: List[tuple] = []
        kept_tokens = 0
        # 先估算每个压缩后 chunk 的 token（含 XML 标签开销）
        for ch, content, was_compressed in compressed:
            kept.append((ch, content, was_compressed))
            kept_tokens += self._estimate_chunk_total_tokens(ch, content)

        # 若超预算，从最低分（列表末尾）开始丢弃
        while kept_tokens > budget and len(kept) > 1:
            dropped = kept.pop()
            dropped_tokens = self._estimate_chunk_total_tokens(dropped[0], dropped[1])
            kept_tokens -= dropped_tokens
            stats.dropped_chunks += 1
            logger.debug(
                f"[ContextAssembler] 超预算丢弃 Chunk: "
                f"file={dropped[0].get('file_path')}, "
                f"symbol={dropped[0].get('symbol_name')}, "
                f"score={dropped[0].get('_position_score', dropped[0].get('score')):.4f}"
            )

        stats.kept_chunks = len(kept)

        # 步骤 4：转为 ContextItem + ReferenceItem
        contexts: List[ContextItem] = []
        references: List[ReferenceItem] = []
        final_tokens = 0
        for ch, content, was_compressed in kept:
            fp = ch.get("file_path", "") or ""
            language = _detect_language(fp) or None
            contexts.append(ContextItem(
                type="implicit",
                file_path=fp,
                content_snippet=content,
                language=language,
            ))
            start = ch.get("start_line", 0) or 0
            end = ch.get("end_line", 0) or 0
            lines_str = f"{start}-{end}" if start and end and start != end else str(start or end)
            references.append(ReferenceItem(
                file=fp,
                lines=lines_str,
                score=float(ch.get("_position_score", ch.get("score", 0.0))),
                symbol=ch.get("symbol_name", "") or "",
            ))
            final_tokens += self._estimate_chunk_total_tokens(ch, content)

        stats.compressed_tokens = final_tokens

        # 步骤 5：日志打印组装前后 Token 计数对比（S5 验收要求）
        logger.info(
            f"[ContextAssembler] 组装完成: "
            f"原始 {stats.original_chunks} 个 Chunk / {stats.original_tokens} tokens → "
            f"保留 {stats.kept_chunks} 个 / {stats.compressed_tokens} tokens "
            f"(丢弃 {stats.dropped_chunks} 个, 压缩 {stats.compressed_chunks} 个), "
            f"预算={budget}, 光标文件={cursor_file or '无'}"
        )

        return AssemblyResult(
            contexts=contexts,
            references=references,
            stats=stats,
        )

    # ------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------

    @staticmethod
    def _estimate_chunk_total_tokens(chunk: Dict[str, Any], content: str) -> int:
        """
        估算单个 Chunk 组装后占用的 Token（含 XML 标签 + 属性开销）。
        用于预算控制时的累加计算。
        """
        # XML 标签固定开销：<file path="..." lang="...">...</file>
        # 路径 + 语言属性 + 标签本身约 30~50 token，这里取 40 作为估算
        overhead = 40
        return count_tokens([{"role": "user", "content": content}]) + overhead


# ============================================================
# 单例工厂
# ============================================================

_assembler: Optional[ContextAssembler] = None


def get_context_assembler() -> ContextAssembler:
    """获取 ContextAssembler 单例（配置从 settings 读取）。"""
    global _assembler
    if _assembler is None:
        _assembler = ContextAssembler()
    return _assembler
