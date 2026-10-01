"""
语义代码切片（S4 第 33-34 天）

将 AST 解析器输出的符号表（SymbolTable）切分为有语义边界的代码切片（CodeChunk），
供后续向量化与向量检索使用。

切片策略：
  1. 文件头聚合：将文件顶部的 import 语句与全局变量合并为一个 import 类型的 Chunk。
  2. 最小单元：单个函数/方法/类作为一个 Chunk（行数 <= 200）。
  3. 大函数拆分：函数超过 200 行时，按逻辑块（if / for / while / try / with / match）
     进一步切分，每段保留函数签名作为"标题"，提升符号搜索准确率。
  4. 注释与 docstring 保留：切片内容包含原注释和文档字符串，不做剥离。

关键技术点（来自 Sprint 提示）：
  - 函数签名（def foo(x: int) -> bool）单独提取作为"标题"，与函数体拼接后一起向量化，
    能大幅提升符号搜索的准确率。
  - 切片时务必保留注释和 docstring。
"""

import logging
import os
import re
from typing import List, Optional

from .models import (
    ChunkType,
    CodeChunk,
    Symbol,
    SymbolTable,
    SymbolType,
)

logger = logging.getLogger(__name__)

# 行数阈值
SMALL_FUNCTION_LINES = 50     # < 50 行：整个函数作为一个 Chunk（最小单元）
LARGE_FUNCTION_LINES = 200    # > 200 行：按逻辑块进一步切分
MIN_BLOCK_LINES = 40          # 大函数拆分时，单个子块最少行数（小于则与相邻块合并）

# 大函数拆分时视为"逻辑块"的语句类型（各语言通用）
BLOCK_STATEMENT_TYPES = {
    # Python
    "if_statement", "for_statement", "while_statement", "try_statement",
    "with_statement", "match_statement",
    # JavaScript / TypeScript
    "if_statement", "for_statement", "for_in_statement", "while_statement",
    "do_statement", "switch_statement", "try_statement",
    # Java
    "if_statement", "for_statement", "while_statement", "do_statement",
    "switch_expression", "switch_block", "try_statement", "synchronized_statement",
    # Go
    "if_statement", "for_statement", "switch_statement", "type_switch_statement",
    "select_statement",
}


# ============================================================
# 工具函数
# ============================================================

def _read_source(file_path: str) -> str:
    """读取源码文本（UTF-8，容错）"""
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception as e:
        logger.warning(f"[Chunker] 读取文件失败 {file_path}: {e}")
        return ""


def _extract_import_lines(source_text: str) -> str:
    """
    提取文件顶部的 import 语句文本。

    策略：从文件开头逐行扫描，直到遇到第一个非 import / 非空行 / 非注释行
    （即第一个类/函数/变量定义）为止，收集所有 import 行。
    """
    lines = source_text.splitlines()
    import_lines: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            # 空行：如果已经收集到 import，则停止；否则跳过开头空行
            if import_lines:
                break
            continue
        # 跳过模块级注释/字符串（可能是模块 docstring）
        if stripped.startswith("#") or stripped.startswith("//"):
            if import_lines:
                break
            continue
        # 匹配各语言 import 语句
        if _is_import_line(stripped):
            import_lines.append(line)
        else:
            # 遇到非 import 行，停止收集
            if import_lines:
                break
    return "\n".join(import_lines)


def _is_import_line(stripped: str) -> bool:
    """判断一行是否为 import 语句（覆盖 Python/JS/TS/Java/Go）"""
    patterns = [
        r'^import\s',           # Python: import x / from x import y
        r'^from\s+\S+\s+import',# Python: from x import y
        r'^import\s+\{',        # JS/TS: import { x } from 'y'
        r'^import\s+\*',        # JS/TS: import * as x from 'y'
        r'^import\s+["\']',     # JS/TS: import 'module'
        r'^import\s+\w+',       # JS/TS/Java: import x ...
        r'^export\s+',          # TS: export ... (再导出)
        r'^static\s+import',    # Java: static import
        r'^package\s+',         # Java/Go: package 声明
        r'^require\s*\(',       # CommonJS: require('x')
    ]
    return any(re.match(p, stripped) for p in patterns)


def _extract_signature(content: str, symbol_type: SymbolType) -> str:
    """
    从符号内容中提取签名行（作为切片标题）。

    - 函数/方法：取第一行（def / function / func 声明），若跨行则取到第一个冒号/大括号。
    - 类：取类声明行。
    """
    lines = content.splitlines()
    if not lines:
        return ""
    # 跳过装饰器行（Python @decorator），取真正的声明行
    start = 0
    while start < len(lines) and lines[start].strip().startswith("@"):
        start += 1
    if start >= len(lines):
        return ""
    first = lines[start].strip()
    # 如果声明跨行（如 C 风格大括号在下一行），只取第一行即可
    return first


def _build_title(symbol_name: str, signature: str) -> str:
    """构建切片标题：符号名 + 签名，用于向量化时提升语义辨识度"""
    parts = [symbol_name]
    if signature and signature != symbol_name:
        parts.append(signature)
    return " | ".join(parts)


# ============================================================
# 文件头切片（import + 全局变量）
# ============================================================

def _build_header_chunk(
    file_path: str,
    source_text: str,
    symbol_table: SymbolTable,
) -> Optional[CodeChunk]:
    """
    构建文件头切片：import 语句 + 全局变量。

    返回 None 表示该文件无 import 也无全局变量。
    """
    import_text = _extract_import_lines(source_text)

    # 收集全局变量的内容
    var_parts: List[str] = []
    var_start = None
    var_end = None
    for sym in symbol_table.variables:
        if sym.content:
            var_parts.append(sym.content)
        if var_start is None or sym.start_line < var_start:
            var_start = sym.start_line
        if var_end is None or sym.end_line > var_end:
            var_end = sym.end_line

    # 无 import 且无全局变量 → 不生成 header chunk
    if not import_text and not var_parts:
        return None

    # 组装内容：import 在前，全局变量在后
    content_parts = []
    if import_text:
        content_parts.append(import_text)
    if var_parts:
        content_parts.append("\n".join(var_parts))
    content = "\n".join(content_parts)

    # 行号范围：取 import 块和变量块的并集
    lines = source_text.splitlines()
    if import_text:
        import_line_count = len(import_text.splitlines())
        start_line = 1
        end_line = import_line_count
    else:
        start_line = var_start if var_start else 1
        end_line = var_end if var_end else 1
    if var_end and var_end > end_line:
        end_line = var_end

    return CodeChunk(
        file_path=file_path,
        symbol_name="__header__",
        chunk_type=ChunkType.IMPORT,
        content=content,
        start_line=start_line,
        end_line=end_line,
    )


# ============================================================
# 函数/类切片
# ============================================================

def _chunk_single_symbol(file_path: str, symbol: Symbol) -> List[CodeChunk]:
    """
    将单个函数/类符号切分为一个或多个 Chunk。

    - 行数 <= LARGE_FUNCTION_LINES：整个符号作为一个 Chunk
    - 行数 > LARGE_FUNCTION_LINES：调用 _split_large_function 按逻辑块拆分
    """
    line_count = symbol.end_line - symbol.start_line + 1
    chunk_type = ChunkType.FUNCTION if symbol.symbol_type == SymbolType.FUNCTION else ChunkType.CLASS

    if line_count <= LARGE_FUNCTION_LINES:
        signature = _extract_signature(symbol.content or "", symbol.symbol_type)
        title = _build_title(symbol.name, signature)
        content = f"{title}\n{symbol.content}"
        return [CodeChunk(
            file_path=file_path,
            symbol_name=symbol.name,
            chunk_type=chunk_type,
            content=content,
            start_line=symbol.start_line,
            end_line=symbol.end_line,
        )]

    # 大函数：按逻辑块拆分
    return _split_large_symbol(file_path, symbol)


def _split_large_symbol(file_path: str, symbol: Symbol) -> List[CodeChunk]:
    """
    将超过 200 行的函数/类按逻辑块（if/for/while/try 等）拆分为多个 Chunk。

    每个子 Chunk 都以函数签名作为标题，保证检索时的语义连贯性。
    拆分策略：
      - 解析符号内容的 AST
      - 遍历函数体（block）的直接子语句
      - 控制流语句（if/for/while/...）单独成块
      - 连续的简单语句合并为一块
    """
    content = symbol.content or ""
    signature = _extract_signature(content, symbol.symbol_type)
    title = _build_title(symbol.name, signature)
    chunk_type = ChunkType.FUNCTION if symbol.symbol_type == SymbolType.FUNCTION else ChunkType.CLASS

    # 尝试用 tree-sitter 解析符号内容以定位逻辑块
    blocks = _split_by_ast(content, file_path)
    if blocks is None:
        # AST 解析失败：降级为按固定行数（每 150 行）切分
        blocks = _split_by_lines(content, chunk_size=150)
    else:
        # 合并过小的相邻块，避免切得过细
        blocks = _merge_small_blocks(blocks, min_lines=MIN_BLOCK_LINES)

    chunks: List[CodeChunk] = []
    current_line = symbol.start_line
    for block_text in blocks:
        block_lines = len(block_text.splitlines())
        end_line = min(current_line + block_lines - 1, symbol.end_line)
        chunk_content = f"{title}\n{block_text}"
        chunks.append(CodeChunk(
            file_path=file_path,
            symbol_name=symbol.name,
            chunk_type=chunk_type,
            content=chunk_content,
            start_line=current_line,
            end_line=end_line,
        ))
        current_line = end_line + 1

    # 确保最后一个 chunk 的 end_line 不超过符号结束行
    if chunks:
        chunks[-1].end_line = symbol.end_line

    return chunks


def _split_by_ast(content: str, file_path: str) -> Optional[List[str]]:
    """
    用 tree-sitter 解析符号内容，按顶层逻辑块拆分。

    返回拆分后的文本块列表；解析失败返回 None。
    """
    from .parser_factory import get_parser_for_file
    parser, language = get_parser_for_file(file_path)
    if parser is None:
        return None

    try:
        source_bytes = content.encode("utf-8")
        tree = parser.parse(source_bytes)
        root = tree.root_node

        # 找到第一个 function/class 定义节点
        def_node = None
        for child in root.children:
            if child.type in (
                "function_definition", "class_definition",
                "function_declaration", "method_declaration", "method_definition",
                "class_declaration",
            ):
                def_node = child
                break
        if def_node is None:
            return None

        # 找到 body 节点（函数体/类体）
        body = def_node.child_by_field_name("body")
        if body is None:
            # 尝试取最后一个子节点作为 body
            body = def_node.children[-1] if def_node.children else None
        if body is None or not body.children:
            return None

        blocks: List[str] = []
        current_buffer: List[str] = []

        for stmt in body.children:
            stmt_text = source_bytes[stmt.start_byte:stmt.end_byte].decode("utf-8", errors="replace")
            if stmt.type in BLOCK_STATEMENT_TYPES:
                # 先把累积的简单语句作为一块
                if current_buffer:
                    blocks.append("\n".join(current_buffer))
                    current_buffer = []
                blocks.append(stmt_text)
            else:
                current_buffer.append(stmt_text)

        if current_buffer:
            blocks.append("\n".join(current_buffer))

        return blocks if blocks else None
    except Exception as e:
        logger.debug(f"[Chunker] AST 拆分失败，降级按行切分: {e}")
        return None


def _split_by_lines(content: str, chunk_size: int = 150) -> List[str]:
    """降级策略：按固定行数切分"""
    lines = content.splitlines()
    return [
        "\n".join(lines[i:i + chunk_size])
        for i in range(0, len(lines), chunk_size)
    ] or [content]


def _merge_small_blocks(blocks: List[str], min_lines: int = 40) -> List[str]:
    """
    合并过小的相邻块，避免大函数被切得过细。

    策略：
      - 若单个块行数 >= min_lines，独立输出（不与相邻块合并）。
      - 连续的小块（< min_lines）累积合并，直到达到 min_lines 后输出。
      - 末尾剩余的小块合并到前一个块。
    """
    if len(blocks) <= 1:
        return blocks

    merged: List[str] = []
    buffer: List[str] = []
    buffer_lines = 0

    for block in blocks:
        block_lines = len(block.splitlines())
        if block_lines >= min_lines:
            # 大块：先输出累积的小块，再独立输出本块
            if buffer:
                merged.append("\n".join(buffer))
                buffer = []
                buffer_lines = 0
            merged.append(block)
        else:
            # 小块：累积
            buffer.append(block)
            buffer_lines += block_lines
            if buffer_lines >= min_lines:
                merged.append("\n".join(buffer))
                buffer = []
                buffer_lines = 0

    # 处理末尾剩余的小块
    if buffer:
        if merged:
            merged[-1] = merged[-1] + "\n" + "\n".join(buffer)
        else:
            merged.append("\n".join(buffer))

    return merged


# ============================================================
# 主入口
# ============================================================

def chunk_file(file_path: str, symbol_table: SymbolTable, source_text: Optional[str] = None) -> List[CodeChunk]:
    """
    将单个文件解析后的符号表切分为语义 Chunk 列表。

    Args:
        file_path:     文件路径（相对路径，用于存储）
        symbol_table:  AST 解析器输出的符号表
        source_text:   可选，文件源码文本；若不传则从 file_path 读取

    Returns:
        CodeChunk 列表（包含一个 import 块和若干 function/class 块）
    """
    if source_text is None:
        source_text = _read_source(file_path)

    chunks: List[CodeChunk] = []

    # 1. 文件头切片（import + 全局变量）
    header = _build_header_chunk(file_path, source_text, symbol_table)
    if header is not None:
        chunks.append(header)

    # 2. 函数/类切片
    for symbol in symbol_table.symbols:
        if symbol.symbol_type in (SymbolType.FUNCTION, SymbolType.CLASS):
            chunks.extend(_chunk_single_symbol(file_path, symbol))

    return chunks


def chunk_symbol_table(symbol_table: SymbolTable, source_text: Optional[str] = None) -> List[CodeChunk]:
    """
    便捷方法：直接对 SymbolTable 做切片（file_path 取自 symbol_table）。
    """
    return chunk_file(symbol_table.file_path, symbol_table, source_text)
