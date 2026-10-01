"""
依赖关系图（S4 第 39-40 天）

在 AST 遍历过程中，额外提取符号引用关系，构建代码库的依赖图（Call Graph）：
  - 文件级边：a.py 中 `import b` → 边 a.py -> b.py
  - 符号级边：main() 中调用 utils.parse() → 边 main -> utils.parse

存储选型（S4 MVP 建议）：JSON 文件 + 内存缓存
  - 避免引入 Neo4j 等中间件，符合 P2 阶段"免安装、嵌入式"要求
  - 内存缓存提供 O(1) 邻接查询，JSON 文件用于跨进程/重启后恢复
  - 全量索引结束后整体持久化；增量更新时局部修改后重新持久化

风险应对（S4 关键技术预研）：
  - tree-sitter 解析失败：用 try-except 包裹，降级为不提取该文件的引用关系，
    保证整体索引流程不中断（与 ast_parser 的降级策略一致）
  - 模块名→文件路径解析不确定：仅对能解析为本地文件的 import 建立文件级边；
    无法解析的仍记录原始模块名，便于后续符号搜索时辅助定位
  - 大型 Monorepo：图查询使用 BFS + depth 限制，避免全图遍历
"""

import json
import logging
import os
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set, Tuple

from .ast_parser import parse_file
from .models import SymbolTable
from .parser_factory import detect_language, get_parser_for_file

logger = logging.getLogger(__name__)


# ============================================================
# 数据模型
# ============================================================

@dataclass
class GraphEdge:
    """
    图的一条边。

    Attributes:
        source:      源节点（文件相对路径 或 符号全名 file_path::symbol_name）
        target:      目标节点（文件相对路径 或 模块名/符号名）
        edge_type:   "import"（文件级导入）或 "call"（符号级调用）
        source_type: "file"（源为文件）或 "symbol"（源为符号）
        line:        引用所在行号（1-based，0 表示未知）
        raw:         原始引用文本（如 "utils.parse"），便于调试
    """
    source: str
    target: str
    edge_type: str          # "import" | "call"
    source_type: str        # "file" | "symbol"
    line: int = 0
    raw: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FileNode:
    """
    图中的文件节点信息（用于接口返回）。

    Attributes:
        file_path: 文件相对路径
        direction: "upstream"（被查询文件依赖）或 "downstream"（依赖被查询文件）
        depth:     与查询起点的跳数（1=直接依赖，2=间接依赖，0=自身）
    """
    file_path: str
    direction: str          # "upstream" | "downstream" | "self"
    depth: int = 0
    edges: List[dict] = field(default_factory=list)


# ============================================================
# 模块名 → 文件路径 解析
# ============================================================

def _candidate_paths_for_module(module: str, language: str, workspace_files: Set[str]) -> List[str]:
    """
    将导入的模块名解析为候选本地文件相对路径。

    Args:
        module:          模块名（如 "utils" / "a.b" / "./helper"）
        language:        当前文件语言（python / javascript / ...）
        workspace_files: 工作区所有已索引文件的相对路径集合

    Returns:
        候选相对路径列表（已过滤掉不存在的）
    """
    candidates: List[str] = []

    if language == "python":
        # import a.b.c → a/b/c.py 或 a/b/c/__init__.py
        parts = module.replace(".", "/")
        candidates.append(f"{parts}.py")
        candidates.append(f"{parts}/__init__.py")
        # 单层：a → a.py
        if "." not in module:
            candidates.append(f"{module}.py")
    elif language in ("javascript", "typescript"):
        # import './helper' → ./helper.js / ./helper.ts / ./helper/index.js
        # import 'utils' → utils.js（非相对路径时通常为 npm 包，跳过）
        if module.startswith(".") or module.startswith("/"):
            clean = module.lstrip("./")
            exts = [".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx"]
            for ext in exts:
                candidates.append(f"{clean}{ext}")
            candidates.append(f"{clean}/index.js")
            candidates.append(f"{clean}/index.ts")
    elif language == "java":
        # import com.example.Utils → com/example/Utils.java
        parts = module.replace(".", "/")
        candidates.append(f"{parts}.java")
    elif language == "go":
        # Go 的 import 为完整路径，难以匹配本地相对路径
        # 仅当 module 看起来像本地路径时尝试
        if not module.startswith("/"):
            parts = module.split("/")[-1]
            candidates.append(f"{parts}.go")

    return [c for c in candidates if c in workspace_files]


def _resolve_import_to_file(
    module: str,
    language: str,
    workspace_files: Set[str],
) -> Optional[str]:
    """将模块名解析为本地文件路径，无法解析返回 None"""
    for cand in _candidate_paths_for_module(module, language, workspace_files):
        return cand
    return None


# ============================================================
# AST 引用提取
# ============================================================

def _node_text(source: bytes, node) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _walk_imports_python(root_node, source: bytes) -> List[Tuple[str, int]]:
    """
    提取 Python 文件的 import 模块名。

    Returns:
        [(module_name, line_number), ...]
    """
    imports: List[Tuple[str, int]] = []

    def walk(node):
        for child in node.children:
            if child.type == "import_statement":
                # import a, b.c, d as e
                text = _node_text(source, child)
                line = child.start_point[0] + 1
                # 跳过 "import" 关键字，按逗号分割
                body = text.replace("import", "", 1).strip()
                for part in body.split(","):
                    part = part.strip()
                    if not part:
                        continue
                    # 处理 "a as b" → 取 a
                    name = part.split(" as ")[0].strip()
                    if name:
                        imports.append((name, line))
            elif child.type == "import_from_statement":
                # from a.b import c, d
                text = _node_text(source, child)
                line = child.start_point[0] + 1
                # 提取 from 后的模块名
                if text.startswith("from"):
                    rest = text[4:].strip()
                    # "a.b import ..." → 取 a.b
                    name = rest.split(" import ")[0].strip()
                    if name and name != ".":
                        imports.append((name, line))
            walk(child)

    walk(root_node)
    return imports


def _walk_imports_js_ts(root_node, source: bytes) -> List[Tuple[str, int]]:
    """提取 JS/TS 的 import 源路径"""
    imports: List[Tuple[str, int]] = []

    def walk(node):
        for child in node.children:
            if child.type == "import_statement":
                text = _node_text(source, child)
                line = child.start_point[0] + 1
                # 提取引号内的路径：from '...' 或 import '...'
                import re
                m = re.search(r"from\s+['\"]([^'\"]+)['\"]", text)
                if m:
                    imports.append((m.group(1), line))
                else:
                    m = re.search(r"import\s+['\"]([^'\"]+)['\"]", text)
                    if m:
                        imports.append((m.group(1), line))
            walk(child)

    walk(root_node)
    return imports


def _walk_imports_java(root_node, source: bytes) -> List[Tuple[str, int]]:
    """提取 Java 的 import 包名"""
    imports: List[Tuple[str, int]] = []

    def walk(node):
        for child in node.children:
            if child.type == "import_declaration":
                text = _node_text(source, child)
                line = child.start_point[0] + 1
                # import static com.example.Utils.method;
                # import com.example.Utils;
                body = text.replace("import", "", 1).strip().rstrip(";").strip()
                if body.startswith("static "):
                    body = body[7:].strip()
                # 去掉末尾的 .method（static import）
                parts = body.split(".")
                # 简单策略：去掉最后一项如果是小写开头（方法名），保留包.类
                if len(parts) > 1 and parts[-1][0].islower():
                    body = ".".join(parts[:-1])
                if body:
                    imports.append((body, line))
            walk(child)

    walk(root_node)
    return imports


def _walk_imports_go(root_node, source: bytes) -> List[Tuple[str, int]]:
    """提取 Go 的 import 路径"""
    imports: List[Tuple[str, int]] = []

    def walk(node):
        for child in node.children:
            if child.type == "import_declaration":
                text = _node_text(source, child)
                line = child.start_point[0] + 1
                import re
                # 匹配 "path/to/pkg"
                for m in re.finditer(r'"([^"]+)"', text):
                    imports.append((m.group(1), line))
            walk(child)

    walk(root_node)
    return imports


def _walk_calls_python(root_node, source: bytes, file_path: str) -> List[Tuple[str, int]]:
    """
    提取 Python 函数调用。

    仅记录在函数体内部的 call（与最近的外层函数关联），
    返回 [(qualified_name, line), ...]，其中 qualified_name 可能是：
      - foo
      - obj.method
      - module.func
    """
    calls: List[Tuple[str, int]] = []

    def extract_call_name(call_node) -> str:
        """从 call 节点提取被调用函数名"""
        func_node = call_node.child_by_field_name("function")
        if func_node is None:
            return ""
        text = _node_text(source, func_node)
        return text

    def walk(node):
        for child in node.children:
            if child.type == "call":
                name = extract_call_name(child)
                if name:
                    line = child.start_point[0] + 1
                    calls.append((name, line))
            walk(child)

    walk(root_node)
    return calls


def _walk_calls_js_ts(root_node, source: bytes) -> List[Tuple[str, int]]:
    """提取 JS/TS 的 call_expression"""
    calls: List[Tuple[str, int]] = []

    def walk(node):
        for child in node.children:
            if child.type == "call_expression":
                func_node = child.child_by_field_name("function")
                if func_node is not None:
                    name = _node_text(source, func_node)
                    if name:
                        calls.append((name, child.start_point[0] + 1))
            walk(child)

    walk(root_node)
    return calls


def _walk_calls_java(root_node, source: bytes) -> List[Tuple[str, int]]:
    """提取 Java 的 method_invocation"""
    calls: List[Tuple[str, int]] = []

    def walk(node):
        for child in node.children:
            if child.type == "method_invocation":
                # 取方法名（最后一个 identifier）
                name_node = child.child_by_field_name("name")
                if name_node is not None:
                    name = name_node.text.decode("utf-8", errors="replace")
                    # 尝试获取对象限定符
                    obj_node = child.child_by_field_name("object")
                    if obj_node is not None:
                        obj = _node_text(source, obj_node)
                        name = f"{obj}.{name}"
                    calls.append((name, child.start_point[0] + 1))
            walk(child)

    walk(root_node)
    return calls


def _walk_calls_go(root_node, source: bytes) -> List[Tuple[str, int]]:
    """提取 Go 的 call_expression"""
    calls: List[Tuple[str, int]] = []

    def walk(node):
        for child in node.children:
            if child.type == "call_expression":
                func_node = child.child_by_field_name("function")
                if func_node is not None:
                    name = _node_text(source, func_node)
                    if name:
                        calls.append((name, child.start_point[0] + 1))
            walk(child)

    walk(root_node)
    return calls


def _extract_references(
    file_path: str,
    language: str,
    source_bytes: bytes,
    root_node,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """
    从 AST 提取 import 模块名与函数调用名。

    Returns:
        (imports, calls) 两个列表
    """
    imports: List[Tuple[str, int]] = []
    calls: List[Tuple[str, int]] = []

    try:
        if language == "python":
            imports = _walk_imports_python(root_node, source_bytes)
            calls = _walk_calls_python(root_node, source_bytes, file_path)
        elif language in ("javascript", "typescript"):
            imports = _walk_imports_js_ts(root_node, source_bytes)
            calls = _walk_calls_js_ts(root_node, source_bytes)
        elif language == "java":
            imports = _walk_imports_java(root_node, source_bytes)
            calls = _walk_calls_java(root_node, source_bytes)
        elif language == "go":
            imports = _walk_imports_go(root_node, source_bytes)
            calls = _walk_calls_go(root_node, source_bytes)
    except Exception as e:
        logger.warning(f"[DepGraph] 提取引用关系失败 {file_path}: {e}")

    return imports, calls


# ============================================================
# 依赖图主类
# ============================================================

class DependencyGraph:
    """
    代码依赖关系图。

    维护两类边：
      - 文件级 import 边：file_path -> file_path（仅本地文件）
      - 符号级 call 边：symbol_full_name -> called_name

    查询：
      - get_related_files(file_path, depth): BFS 返回上下游关联文件
      - get_imports(file_path): 返回该文件直接 import 的本地模块文件列表

    使用方式：
        graph = DependencyGraph()
        graph.build_from_file(file_path, symbol_table, workspace_files)
        related = graph.get_related_files("main.py", depth=2)
    """

    def __init__(self):
        self._lock = threading.Lock()
        # 文件级邻接：file_path -> {target_file_path: edge_info}
        # _forward: A 依赖 B（A import B）
        # _reverse: B 被 A 依赖
        self._forward: Dict[str, Dict[str, GraphEdge]] = defaultdict(dict)
        self._reverse: Dict[str, Dict[str, GraphEdge]] = defaultdict(dict)
        # 符号级 call 边：symbol_full_name -> {called_name: edge_info}
        self._calls: Dict[str, Dict[str, GraphEdge]] = defaultdict(dict)
        # 本地文件集合（用于 import 解析）
        self._local_files: Set[str] = set()
        # 所有已索引文件的语言映射（用于 import 解析）
        self._file_languages: Dict[str, str] = {}

    # ============================================================
    # 图构建
    # ============================================================

    def register_file(self, file_path: str, language: str) -> None:
        """注册文件到本地文件集合（用于 import 解析）"""
        with self._lock:
            self._local_files.add(file_path)
            self._file_languages[file_path] = language

    def build_from_file(
        self,
        file_path: str,
        symbol_table: SymbolTable,
        source_text: Optional[str] = None,
        full_path: Optional[str] = None,
    ) -> List[GraphEdge]:
        """
        从单个文件提取引用关系，构建该文件的边。

        流程：
        1. 用 tree-sitter 重新解析（与 ast_parser 复用同一 Parser）
        2. 提取 import 模块名 → 尝试解析为本地文件 → 建立文件级边
        3. 提取函数调用 → 记录符号级边（caller = 当前函数）
        4. 旧文件已有的边会被本次结果覆盖（增量更新时先 clear_file）

        Args:
            file_path:     文件相对路径（作为图节点 key）
            symbol_table:  AST 解析器输出的符号表（用于确定调用所属函数）
            source_text:   可选源码文本；不传则从 full_path 或 file_path 读取
            full_path:     可选文件绝对路径，仅用于读取源码（file_path 是相对路径时需传入）

        Returns:
            本次提取到的边列表（便于调试）
        """
        language = symbol_table.language or detect_language(file_path)
        if language == "unknown":
            return []

        # 读取源码
        if source_text is None:
            read_path = full_path or file_path
            try:
                with open(read_path, "r", encoding="utf-8", errors="replace") as f:
                    source_text = f.read()
            except Exception as e:
                logger.warning(f"[DepGraph] 读取文件失败 {read_path}: {e}")
                return []
        source_bytes = source_text.encode("utf-8")

        # 解析 AST
        parser, _ = get_parser_for_file(file_path)
        if parser is None:
            logger.debug(f"[DepGraph] 无可用 Parser，跳过引用提取: {file_path}")
            return []
        try:
            tree = parser.parse(source_bytes)
            root_node = tree.root_node
        except Exception as e:
            logger.warning(f"[DepGraph] tree-sitter 解析失败 {file_path}: {e}")
            return []

        # 提取 import 与 call
        imports, calls = _extract_references(file_path, language, source_bytes, root_node)

        # 注册当前文件（确保在 local_files 中）
        self.register_file(file_path, language)

        edges: List[GraphEdge] = []

        with self._lock:
            # 清空当前文件的旧边（增量更新时由调用方负责，这里也防御性清理）
            self._clear_file_edges_locked(file_path)

            # 1. 构建文件级 import 边
            for module_name, line in imports:
                target_file = _resolve_import_to_file(
                    module_name, language, self._local_files
                )
                if target_file and target_file != file_path:
                    edge = GraphEdge(
                        source=file_path,
                        target=target_file,
                        edge_type="import",
                        source_type="file",
                        line=line,
                        raw=module_name,
                    )
                    self._forward[file_path][target_file] = edge
                    self._reverse[target_file][file_path] = edge
                    edges.append(edge)
                else:
                    # 无法解析为本地文件：记录为"原始引用"边（target = 模块名）
                    # 这类边不参与 BFS 文件级查询，但保留供符号搜索辅助
                    edge = GraphEdge(
                        source=file_path,
                        target=module_name,
                        edge_type="import",
                        source_type="file",
                        line=line,
                        raw=module_name,
                    )
                    # 仅当模块名不像标准库/三方包时记录
                    # 简单启发：模块名短且全小写时可能是本地模块
                    if self._looks_like_local_module(module_name, language):
                        self._forward[file_path][module_name] = edge
                        edges.append(edge)

            # 2. 构建符号级 call 边
            # caller 的确定：通过符号表中的函数范围定位调用所在函数
            # 简化策略：将文件级 caller 记录为 file_path，符号级边存于 _calls
            for call_name, line in calls:
                caller_symbol = self._find_enclosing_symbol(symbol_table, line)
                if caller_symbol:
                    source_key = f"{file_path}::{caller_symbol}"
                else:
                    source_key = file_path  # 模块级调用

                edge = GraphEdge(
                    source=source_key,
                    target=call_name,
                    edge_type="call",
                    source_type="symbol" if caller_symbol else "file",
                    line=line,
                    raw=call_name,
                )
                self._calls[source_key][call_name] = edge
                edges.append(edge)

        return edges

    @staticmethod
    def _looks_like_local_module(module: str, language: str) -> bool:
        """
        启发式判断模块名是否可能是本地模块（用于记录未解析的 import）。

        - Python：标准库/三方包通常较短且全小写；本地模块可能含点号
        - JS/TS：相对路径以 . 开头；npm 包通常含 - 或多个 /
        """
        if language in ("javascript", "typescript"):
            return module.startswith(".") or module.startswith("/")
        if language == "python":
            # 简单过滤：长度 > 2 且不含下划线开头的私有名
            return len(module) > 0 and not module.startswith("_")
        if language == "java":
            # Java 包名通常为全小写，本地包难以从包名判断
            return False
        if language == "go":
            return False
        return False

    @staticmethod
    def _find_enclosing_symbol(
        symbol_table: SymbolTable, line: int
    ) -> Optional[str]:
        """根据行号找到包含该行的符号（函数/类）名"""
        for sym in symbol_table.symbols:
            if sym.start_line <= line <= sym.end_line:
                return sym.name
        return None

    def _clear_file_edges_locked(self, file_path: str) -> None:
        """清空指定文件的所有出边与入边（调用方需持有锁）"""
        # 出边
        if file_path in self._forward:
            for target in list(self._forward[file_path].keys()):
                # 同步清理反向索引
                if target in self._reverse:
                    self._reverse[target].pop(file_path, None)
                    if not self._reverse[target]:
                        del self._reverse[target]
            del self._forward[file_path]

        # 入边
        if file_path in self._reverse:
            del self._reverse[file_path]

        # 符号级 call 边（source 以 file_path:: 开头）
        prefix = f"{file_path}::"
        for source_key in list(self._calls.keys()):
            if source_key == file_path or source_key.startswith(prefix):
                del self._calls[source_key]

    def remove_file(self, file_path: str) -> None:
        """从图中移除文件及其所有边（文件删除/重命名时调用）"""
        with self._lock:
            self._clear_file_edges_locked(file_path)
            self._local_files.discard(file_path)
            self._file_languages.pop(file_path, None)

    # ============================================================
    # 查询
    # ============================================================

    def get_imports(self, file_path: str) -> List[dict]:
        """
        返回该文件直接 import 的本地模块信息。

        用于验收："调用依赖图接口，能返回它 import 的所有本地模块名"
        """
        with self._lock:
            forward = self._forward.get(file_path, {})
            return [
                {
                    "file_path": tgt if tgt in self._local_files else None,
                    "module_name": edge.raw,
                    "line": edge.line,
                    "resolved": tgt in self._local_files,
                }
                for tgt, edge in forward.items()
            ]

    def get_related_files(
        self, file_path: str, depth: int = 2
    ) -> Dict[str, List[dict]]:
        """
        BFS 查询与指定文件强关联的上下游文件。

        Args:
            file_path: 查询起点（相对路径）
            depth:     遍历深度（1=直接依赖，2=间接依赖，...）

        Returns:
            {
                "upstream":   [被查询文件依赖的文件，按 depth 升序],
                "downstream": [依赖被查询文件的文件，按 depth 升序],
            }
            每项为 {file_path, depth, edges: [...]}
        """
        upstream: List[dict] = []
        downstream: List[dict] = []

        with self._lock:
            # downstream：谁依赖 file_path？（reverse 索引）
            downstream = self._bfs(file_path, self._reverse, self._forward, depth, "downstream")
            # upstream：file_path 依赖谁？（forward 索引）
            upstream = self._bfs(file_path, self._forward, self._reverse, depth, "upstream")

        return {
            "file_path": file_path,
            "upstream": upstream,
            "downstream": downstream,
        }

    @staticmethod
    def _bfs(
        start: str,
        adj: Dict[str, Dict[str, GraphEdge]],
        reverse_adj: Dict[str, Dict[str, GraphEdge]],
        max_depth: int,
        direction: str,
    ) -> List[dict]:
        """
        从 start 出发，沿 adj 的邻接边做 BFS，返回所有可达的本地文件节点。

        Args:
            start:       起点文件路径
            adj:         邻接表（决定遍历方向）
            reverse_adj: 反向邻接表（用于补充边信息）
            max_depth:   最大跳数
            direction:   "upstream" 或 "downstream"
        """
        visited: Set[str] = {start}
        queue: deque = deque([(start, 0)])
        results: List[dict] = []

        while queue:
            current, d = queue.popleft()
            if d >= max_depth:
                continue
            neighbors = adj.get(current, {})
            for target, edge in neighbors.items():
                # 仅追踪能解析为本地文件的边（target 在 _local_files 中）
                # 但 adj 中可能含未解析的模块名（target 不在 local_files），
                # 这些不在 BFS 中继续扩展
                if target in visited:
                    continue
                visited.add(target)
                # 判断 target 是否为本地文件（通过 adj 中是否有以 target 为键的记录，
                # 或通过 _local_files 判断；这里无法访问 _local_files，简化为：
                # 只要 target 出现在任何邻接表的键中，就视为图中的文件节点）
                next_d = d + 1
                results.append({
                    "file_path": target,
                    "depth": next_d,
                    "direction": direction,
                    "edge": {
                        "source": edge.source,
                        "target": edge.target,
                        "edge_type": edge.edge_type,
                        "line": edge.line,
                        "raw": edge.raw,
                    },
                })
                queue.append((target, next_d))

        return results

    def get_symbols(self, file_path: Optional[str] = None) -> List[dict]:
        """返回符号级 call 边（调试用）"""
        with self._lock:
            if file_path:
                prefix = f"{file_path}::"
                result = []
                for source_key, calls in self._calls.items():
                    if source_key == file_path or source_key.startswith(prefix):
                        for call_name, edge in calls.items():
                            result.append(edge.to_dict())
                return result
            result = []
            for source_key, calls in self._calls.items():
                for call_name, edge in calls.items():
                    result.append(edge.to_dict())
            return result

    def stats(self) -> dict:
        """返回图统计信息"""
        with self._lock:
            file_edges = sum(len(v) for v in self._forward.values())
            call_edges = sum(len(v) for v in self._calls.values())
            return {
                "local_files": len(self._local_files),
                "file_edges": file_edges,
                "call_edges": call_edges,
            }

    # ============================================================
    # 持久化（JSON 文件 + 内存缓存）
    # ============================================================

    def save(self, file_path: str) -> None:
        """
        将图持久化到 JSON 文件。

        格式：
        {
            "local_files": [...],
            "file_languages": {...},
            "forward_edges": [{source, target, line, raw}, ...],
            "call_edges":    [{source, target, line, raw}, ...],
        }
        """
        with self._lock:
            data = {
                "local_files": sorted(self._local_files),
                "file_languages": dict(self._file_languages),
                "forward_edges": [
                    {
                        "source": src,
                        "target": tgt,
                        "line": edge.line,
                        "raw": edge.raw,
                    }
                    for src, neighbors in self._forward.items()
                    for tgt, edge in neighbors.items()
                ],
                "call_edges": [
                    {
                        "source": src,
                        "target": tgt,
                        "line": edge.line,
                        "raw": edge.raw,
                    }
                    for src, neighbors in self._calls.items()
                    for tgt, edge in neighbors.items()
                ],
            }

        try:
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.debug(f"[DepGraph] 图已持久化到 {file_path}")
        except Exception as e:
            logger.warning(f"[DepGraph] 持久化失败: {e}")

    def load(self, file_path: str) -> bool:
        """从 JSON 文件恢复图到内存。返回是否成功"""
        if not os.path.exists(file_path):
            return False
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"[DepGraph] 读取持久化文件失败: {e}")
            return False

        with self._lock:
            self._forward.clear()
            self._reverse.clear()
            self._calls.clear()
            self._local_files = set(data.get("local_files", []))
            self._file_languages = dict(data.get("file_languages", {}))

            for edge_data in data.get("forward_edges", []):
                src = edge_data["source"]
                tgt = edge_data["target"]
                edge = GraphEdge(
                    source=src,
                    target=tgt,
                    edge_type="import",
                    source_type="file",
                    line=edge_data.get("line", 0),
                    raw=edge_data.get("raw", ""),
                )
                self._forward[src][tgt] = edge
                self._reverse[tgt][src] = edge

            for edge_data in data.get("call_edges", []):
                src = edge_data["source"]
                tgt = edge_data["target"]
                edge = GraphEdge(
                    source=src,
                    target=tgt,
                    edge_type="call",
                    source_type="symbol" if "::" in src else "file",
                    line=edge_data.get("line", 0),
                    raw=edge_data.get("raw", ""),
                )
                self._calls[src][tgt] = edge

        logger.info(
            f"[DepGraph] 已从 {file_path} 恢复图："
            f"{len(self._local_files)} 文件, "
            f"{sum(len(v) for v in self._forward.values())} 文件边, "
            f"{sum(len(v) for v in self._calls.values())} 调用边"
        )
        return True

    def clear(self) -> None:
        """清空整个图"""
        with self._lock:
            self._forward.clear()
            self._reverse.clear()
            self._calls.clear()
            self._local_files.clear()
            self._file_languages.clear()


# ============================================================
# 单例
# ============================================================

_dependency_graph: Optional[DependencyGraph] = None


def get_dependency_graph() -> DependencyGraph:
    global _dependency_graph
    if _dependency_graph is None:
        _dependency_graph = DependencyGraph()
    return _dependency_graph
