"""
AST 解析器（S4 第 31-32 天）

基于 tree-sitter 解析源文件，提取符号表：
- 类名（Class）
- 方法/函数名（Function）
- 全局变量（Variable）

每个符号记录起始行号、结束行号（1-based）及对应源码文本。

风险应对（S4 关键技术预研）：
- tree-sitter 解析失败时（残缺语法/不规范代码），降级为纯文本按行切块，
  保证索引流程不中断。
"""

import logging
import re
from typing import List, Optional

from .models import Symbol, SymbolTable, SymbolType
from .parser_factory import get_parser_for_file, detect_language

logger = logging.getLogger(__name__)


def _node_text(source: bytes, node) -> str:
    """从源码中提取节点对应的文本"""
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _node_name(node) -> Optional[str]:
    """
    提取类/函数的名称节点文本。
    优先使用 field name，降级为第一个 identifier 子节点。
    """
    name_node = node.child_by_field_name("name")
    if name_node is not None:
        return name_node.text.decode("utf-8", errors="replace")
    # 降级：遍历子节点找 identifier
    for child in node.children:
        if child.type == "identifier":
            return child.text.decode("utf-8", errors="replace")
    return None


def _line_range(node) -> tuple:
    """返回节点的 (start_line, end_line)，1-based"""
    return node.start_point[0] + 1, node.end_point[0] + 1


def _has_error_node(node) -> bool:
    """递归检测 AST 中是否存在 ERROR 节点（语法错误标记）"""
    if node.type == "ERROR" or node.is_missing:
        return True
    for child in node.children:
        if _has_error_node(child):
            return True
    return False


# ============================================================
# 各语言符号提取
# ============================================================

def _extract_python(source: bytes, root_node, file_path: str) -> List[Symbol]:
    """提取 Python 符号：类、函数、全局变量"""
    symbols: List[Symbol] = []

    def walk(node):
        for child in node.children:
            node_type = child.type

            # 带装饰器的定义：decorated_definition 包裹 class/function
            if node_type == "decorated_definition":
                # 找到内部真正的 class/function definition
                inner = None
                for sub in child.children:
                    if sub.type in ("class_definition", "function_definition"):
                        inner = sub
                        break
                if inner is not None:
                    name = _node_name(inner)
                    if name:
                        stype = SymbolType.CLASS if inner.type == "class_definition" else SymbolType.FUNCTION
                        sl, el = _line_range(child)  # 装饰器也算入范围
                        symbols.append(Symbol(
                            name=name,
                            symbol_type=stype,
                            file_path=file_path,
                            start_line=sl,
                            end_line=el,
                            content=_node_text(source, child),
                        ))
                    # 继续遍历内部（嵌套定义）
                    walk(inner)
                continue

            if node_type == "class_definition":
                name = _node_name(child)
                if name:
                    sl, el = _line_range(child)
                    symbols.append(Symbol(
                        name=name, symbol_type=SymbolType.CLASS,
                        file_path=file_path, start_line=sl, end_line=el,
                        content=_node_text(source, child),
                    ))
                walk(child)
                continue

            if node_type == "function_definition":
                name = _node_name(child)
                if name:
                    sl, el = _line_range(child)
                    symbols.append(Symbol(
                        name=name, symbol_type=SymbolType.FUNCTION,
                        file_path=file_path, start_line=sl, end_line=el,
                        content=_node_text(source, child),
                    ))
                walk(child)
                continue

            # 全局变量：顶层 assignment，左值为 identifier
            if node_type == "assignment" and node.type == "module":
                name_node = child.child_by_field_name("left")
                if name_node is not None and name_node.type == "identifier":
                    name = name_node.text.decode("utf-8", errors="replace")
                    sl, el = _line_range(child)
                    symbols.append(Symbol(
                        name=name, symbol_type=SymbolType.VARIABLE,
                        file_path=file_path, start_line=sl, end_line=el,
                        content=_node_text(source, child),
                    ))
                continue

            # 其他节点继续递归，以捕获嵌套的 class/function
            walk(child)

    walk(root_node)
    return symbols


def _extract_js_ts(source: bytes, root_node, file_path: str, is_ts: bool) -> List[Symbol]:
    """提取 JavaScript / TypeScript 符号"""
    symbols: List[Symbol] = []

    def walk(node, is_top_level: bool = False):
        for child in node.children:
            t = child.type

            if t == "class_declaration":
                name = _node_name(child)
                if name:
                    sl, el = _line_range(child)
                    symbols.append(Symbol(
                        name=name, symbol_type=SymbolType.CLASS,
                        file_path=file_path, start_line=sl, end_line=el,
                        content=_node_text(source, child),
                    ))
                walk(child)
                continue

            if t in ("function_declaration", "generator_function_declaration"):
                name = _node_name(child)
                if name:
                    sl, el = _line_range(child)
                    symbols.append(Symbol(
                        name=name, symbol_type=SymbolType.FUNCTION,
                        file_path=file_path, start_line=sl, end_line=el,
                        content=_node_text(source, child),
                    ))
                walk(child)
                continue

            # 类中的方法
            if t in ("method_definition", "function_expression") and node.type == "class_body":
                name = _node_name(child)
                if name:
                    sl, el = _line_range(child)
                    symbols.append(Symbol(
                        name=name, symbol_type=SymbolType.FUNCTION,
                        file_path=file_path, start_line=sl, end_line=el,
                        content=_node_text(source, child),
                    ))
                continue

            # 顶层变量声明（const/let → lexical_declaration, var → variable_declaration）
            if t in ("lexical_declaration", "variable_declaration") and is_top_level:
                # variable_declarator 是子节点，取其 identifier 作为变量名
                for sub in child.children:
                    if sub.type == "variable_declarator":
                        name_node = sub.child_by_field_name("name")
                        if name_node is None:
                            name_node = sub.children[0] if sub.children else None
                        if name_node is not None and name_node.type == "identifier":
                            name = name_node.text.decode("utf-8", errors="replace")
                            sl, el = _line_range(child)
                            symbols.append(Symbol(
                                name=name, symbol_type=SymbolType.VARIABLE,
                                file_path=file_path, start_line=sl, end_line=el,
                                content=_node_text(source, child),
                            ))
                continue

            walk(child)

    walk(root_node, is_top_level=True)
    return symbols


def _extract_java(source: bytes, root_node, file_path: str) -> List[Symbol]:
    """提取 Java 符号：类、方法、字段"""
    symbols: List[Symbol] = []

    def walk(node):
        for child in node.children:
            t = child.type

            if t == "class_declaration":
                name = _node_name(child)
                if name:
                    sl, el = _line_range(child)
                    symbols.append(Symbol(
                        name=name, symbol_type=SymbolType.CLASS,
                        file_path=file_path, start_line=sl, end_line=el,
                        content=_node_text(source, child),
                    ))
                walk(child)
                continue

            if t in ("method_declaration", "constructor_declaration"):
                name = _node_name(child)
                if name:
                    sl, el = _line_range(child)
                    symbols.append(Symbol(
                        name=name, symbol_type=SymbolType.FUNCTION,
                        file_path=file_path, start_line=sl, end_line=el,
                        content=_node_text(source, child),
                    ))
                walk(child)
                continue

            # 字段声明（类成员变量）
            if t == "field_declaration":
                decl_node = child.child_by_field_name("declarator")
                if decl_node is not None:
                    name = decl_node.text.decode("utf-8", errors="replace").split("=")[0].strip()
                    if name:
                        sl, el = _line_range(child)
                        symbols.append(Symbol(
                            name=name, symbol_type=SymbolType.VARIABLE,
                            file_path=file_path, start_line=sl, end_line=el,
                            content=_node_text(source, child),
                        ))
                continue

            walk(child)

    walk(root_node)
    return symbols


def _extract_go(source: bytes, root_node, file_path: str) -> List[Symbol]:
    """提取 Go 符号：函数、方法、结构体（视为类）、变量"""
    symbols: List[Symbol] = []

    def walk(node, is_top_level: bool = False):
        for child in node.children:
            t = child.type

            # 普通函数
            if t == "function_declaration":
                name = _node_name(child)
                if name:
                    sl, el = _line_range(child)
                    symbols.append(Symbol(
                        name=name, symbol_type=SymbolType.FUNCTION,
                        file_path=file_path, start_line=sl, end_line=el,
                        content=_node_text(source, child),
                    ))
                walk(child)
                continue

            # 方法（带 receiver）
            if t == "method_declaration":
                name = _node_name(child)
                if name:
                    sl, el = _line_range(child)
                    symbols.append(Symbol(
                        name=name, symbol_type=SymbolType.FUNCTION,
                        file_path=file_path, start_line=sl, end_line=el,
                        content=_node_text(source, child),
                    ))
                walk(child)
                continue

            # 类型声明中的 struct 视为类
            if t == "type_declaration":
                for sub in child.children:
                    if sub.type == "type_spec":
                        name_node = sub.child_by_field_name("name")
                        type_node = sub.child_by_field_name("type")
                        if name_node is not None and type_node is not None and type_node.type == "struct_type":
                            name = name_node.text.decode("utf-8", errors="replace")
                            sl, el = _line_range(child)
                            symbols.append(Symbol(
                                name=name, symbol_type=SymbolType.CLASS,
                                file_path=file_path, start_line=sl, end_line=el,
                                content=_node_text(source, child),
                            ))
                walk(child)
                continue

            # 顶层变量声明
            if t in ("var_declaration", "short_var_declaration") and is_top_level:
                # 提取变量名
                text = _node_text(source, child)
                # 匹配 var x = ... 或 x := ...
                m = re.match(r'(?:var\s+)?(\w+)', text)
                if m:
                    name = m.group(1)
                    sl, el = _line_range(child)
                    symbols.append(Symbol(
                        name=name, symbol_type=SymbolType.VARIABLE,
                        file_path=file_path, start_line=sl, end_line=el,
                        content=_node_text(source, child),
                    ))
                continue

            walk(child)

    walk(root_node, is_top_level=True)
    return symbols


# ============================================================
# 降级策略：纯文本按行切块
# ============================================================

def _fallback_extract(source_text: str, file_path: str) -> List[Symbol]:
    """
    降级策略：当 AST 解析失败时，按空行分割文本块作为符号。

    风险应对（S4）：保证索引不中断，即使是残缺语法的文件也能产出可检索的切片。
    """
    symbols: List[Symbol] = []
    lines = source_text.splitlines()

    # 用正则匹配简单的函数/类/变量定义（粗略降级）
    func_pattern = re.compile(r'^\s*(?:async\s+)?def\s+(\w+)|^\s*function\s+(\w+)|^\s*func\s+(\w+)')
    class_pattern = re.compile(r'^\s*class\s+(\w+)')
    var_pattern = re.compile(r'^\s*(?:const|let|var)\s+(\w+)\s*=')

    i = 0
    while i < len(lines):
        line = lines[i]
        func_m = func_pattern.match(line)
        class_m = class_pattern.match(line)
        var_m = var_pattern.match(line)

        if func_m:
            name = next(g for g in func_m.groups() if g)
            # 找到下一个同级定义或空行作为结束
            j = i + 1
            while j < len(lines) and not (
                func_pattern.match(lines[j]) or class_pattern.match(lines[j])
            ):
                j += 1
            symbols.append(Symbol(
                name=name, symbol_type=SymbolType.FUNCTION,
                file_path=file_path, start_line=i + 1, end_line=j,
                content="\n".join(lines[i:j]),
            ))
            i = j
        elif class_m:
            name = class_m.group(1)
            j = i + 1
            while j < len(lines) and not (
                func_pattern.match(lines[j]) or class_pattern.match(lines[j])
            ):
                j += 1
            symbols.append(Symbol(
                name=name, symbol_type=SymbolType.CLASS,
                file_path=file_path, start_line=i + 1, end_line=j,
                content="\n".join(lines[i:j]),
            ))
            i = j
        elif var_m:
            name = var_m.group(1)
            symbols.append(Symbol(
                name=name, symbol_type=SymbolType.VARIABLE,
                file_path=file_path, start_line=i + 1, end_line=i + 1,
                content=line,
            ))
            i += 1
        else:
            i += 1

    return symbols


# ============================================================
# 主入口
# ============================================================

def parse_file(file_path: str, use_fallback_on_error: bool = True) -> SymbolTable:
    """
    解析单个文件，返回符号表。

    Args:
        file_path: 文件路径
        use_fallback_on_error: AST 解析失败时是否降级为纯文本切块

    Returns:
        SymbolTable 对象
    """
    language = detect_language(file_path)
    table = SymbolTable(file_path=file_path, language=language)

    # 不支持的语言
    if language == "unknown":
        logger.debug(f"[ASTParser] 不支持的文件类型，跳过: {file_path}")
        return table

    # 读取文件
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            source_text = f.read()
        source_bytes = source_text.encode("utf-8")
    except Exception as e:
        logger.warning(f"[ASTParser] 读取文件失败 {file_path}: {e}")
        return table

    # 获取 Parser
    parser, _ = get_parser_for_file(file_path)
    if parser is None:
        if use_fallback_on_error:
            logger.debug(f"[ASTParser] 无可用 Parser，降级为文本切块: {file_path}")
            table.symbols = _fallback_extract(source_text, file_path)
        return table

    # 解析 AST
    has_error = False
    try:
        tree = parser.parse(source_bytes)
        root_node = tree.root_node
        # tree-sitter 是容错解析器，语法错误时产生 ERROR 节点而非抛异常。
        # 检测是否存在 ERROR 节点，用于决定是否合并降级结果。
        has_error = _has_error_node(root_node)
    except Exception as e:
        logger.warning(f"[ASTParser] tree-sitter 解析失败 {file_path}: {e}")
        if use_fallback_on_error:
            table.symbols = _fallback_extract(source_text, file_path)
        return table

    # 按语言提取符号
    try:
        if language == "python":
            table.symbols = _extract_python(source_bytes, root_node, file_path)
        elif language in ("javascript", "typescript"):
            table.symbols = _extract_js_ts(source_bytes, root_node, file_path, is_ts=(language == "typescript"))
        elif language == "java":
            table.symbols = _extract_java(source_bytes, root_node, file_path)
        elif language == "go":
            table.symbols = _extract_go(source_bytes, root_node, file_path)
    except Exception as e:
        logger.warning(f"[ASTParser] 符号提取异常 {file_path}: {e}", exc_info=True)
        if use_fallback_on_error:
            table.symbols = _fallback_extract(source_text, file_path)
        return table

    # 若 AST 中存在 ERROR 节点，合并降级提取的结果（去重），保证残缺语法的符号不被遗漏
    if has_error and use_fallback_on_error:
        fallback_symbols = _fallback_extract(source_text, file_path)
        existing = {(s.symbol_type, s.name) for s in table.symbols}
        for fs in fallback_symbols:
            if (fs.symbol_type, fs.name) not in existing:
                table.symbols.append(fs)

    return table


def parse_file_to_dict(file_path: str) -> dict:
    """解析文件并返回字典（便于序列化/调试）"""
    return parse_file(file_path).to_dict()
