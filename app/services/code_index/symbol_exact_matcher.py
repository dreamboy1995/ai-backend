"""
符号精确检索 & 依赖图增强（S5 第 45-46 天）

利用 S4 提取的符号表（内存 SymbolTable）与依赖关系图（Call Graph），
对用户 Query 中的 # 引用或明显的驼峰/下划线命名做精确字符串匹配
（含模糊前缀匹配），并支持反向依赖查询（谁调用了某符号）。

核心能力：
  1. 符号精确/前缀匹配：
     - # 标签（如 #DataProcessor）→ 精确匹配符号名
     - 驼峰/下划线命名 → 前缀匹配 + 子串包含
     - 匹配结果直接从符号表定位 file_path / 行号范围，绕过向量检索
  2. 反向依赖查询（find_callers）：
     - 利用 S4 构建的 Call Graph，返回所有调用指定符号的函数列表
     - 支持 obj.method / module.func 形式的限定调用匹配
  3. 路径跨平台归一化：
     - 统一使用 pathlib 将文件路径转为 POSIX 正斜杠格式（src/main.py）
     - 避免 Windows 反斜杠与 Linux 正斜杠不一致的问题

与 fusion_reranker.SymbolSearcher 的关系：
  - SymbolSearcher 是 Day 43-44 的轻量版，仅做精确/前缀匹配并返回 chunk
  - 本模块是 Day 45-46 的完整版，额外支持反向依赖查询与路径归一化
  - fusion_reranker.SymbolSearcher.search() 可委托给本模块的 search()

风险应对（S5 关键技术预研）：
  - 符号表为空（索引未完成）：返回空列表，不影响向量/BM25 其他检索路径
  - Call Graph 未构建：find_callers 返回空列表，记录 debug 日志
  - 路径跨平台：normalize_path() 统一转为 POSIX 正斜杠
"""

import logging
import re
import threading
from pathlib import PurePosixPath
from typing import Any, Dict, List, Optional, Tuple

from .models import _compute_chunk_id

logger = logging.getLogger(__name__)


# ============================================================
# 路径归一化
# ============================================================

def normalize_path(path: str) -> str:
    """
    将文件路径统一转为 POSIX 正斜杠格式的相对路径。

    解决 S5 风险预警中的跨平台兼容性问题：
      - Windows 下 os.path.relpath 返回反斜杠（src\\main.py）
      - Linux/macOS 返回正斜杠（src/main.py）
    统一输出为正斜杠，便于前端跳转与路径比较。

    Args:
        path: 任意格式的相对路径

    Returns:
        POSIX 风格的相对路径（正斜杠分隔）
    """
    if not path:
        return path
    # 先用 PurePosixPath 规范化分隔符，再 .as_posix() 确保正斜杠
    # 注意：PurePosixPath 不会解析 .. 或 .，仅做分隔符转换
    return PurePosixPath(path.replace("\\", "/")).as_posix()


# ============================================================
# Token 提取
# ============================================================

# 识别驼峰/下划线/全大写标识符（长度 >= 2）
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,}")

# 符号检索时过滤的极简停用词（避免误把 "how"/"the" 当符号前缀）
_STOPWORDS_FOR_SYMBOL = frozenset({
    "the", "a", "an", "and", "or", "not", "is", "are", "was", "were",
    "be", "been", "to", "of", "in", "on", "at", "by", "for", "with",
    "how", "what", "where", "when", "why", "who", "which", "that",
    "this", "these", "those", "do", "does", "did", "has", "have",
    "can", "could", "should", "would", "will", "may", "might", "must",
    "def", "class", "import", "from", "self", "true", "false", "none",
    "null", "undefined", "function", "return", "var", "let", "const",
    "public", "private", "protected", "static", "void", "int", "string",
    # 反向依赖查询常见疑问词
    "call", "calls", "called", "caller", "callers", "use", "used",
    "uses", "usage", "where", "find", "who", "谁", "调用", "哪里",
    "哪个", "哪个函数", "哪里用", "谁用",
})


def extract_symbol_tokens(query: str) -> List[str]:
    """
    从用户 query 中提取符号候选 token。

    规则：
      1. # 标签：取 # 之后到下一个空白/非标识符字符之前的整段
         （#DataProcessor → ["DataProcessor"]）
      2. 驼峰/下划线命名：用正则识别（长度 >= 2）
      3. 过滤停用词（避免 "how"/"the" 被误匹配）

    Args:
        query: 用户查询文本

    Returns:
        去重保序的符号 token 列表
    """
    if not query:
        return []

    tokens: List[str] = []

    # 1. #Tag 提取
    i = 0
    while i < len(query):
        if query[i] == "#":
            j = i + 1
            while j < len(query) and (query[j].isalnum() or query[j] == "_"):
                j += 1
            if j > i + 1:
                tokens.append(query[i + 1:j])
            i = j
        else:
            i += 1

    # 2. 普通标识符 token
    for m in _IDENT_RE.findall(query):
        if m and len(m) >= 2 and m.lower() not in _STOPWORDS_FOR_SYMBOL:
            tokens.append(m)

    # 3. 去重保序
    seen = set()
    unique: List[str] = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


# ============================================================
# 反向依赖查询（谁调用了某符号）
# ============================================================

def _parse_call_source(source_key: str) -> Tuple[str, str]:
    """
    解析 Call Graph 中 call 边的 source 节点。

    source_key 格式：
      - "file_path::caller_symbol"：符号级调用（caller_symbol 中调用了目标）
      - "file_path"：模块级调用

    Args:
        source_key: Call Graph 的 source 字段

    Returns:
        (file_path, caller_symbol)，模块级调用时 caller_symbol 为 ""
    """
    if "::" in source_key:
        file_path, caller = source_key.rsplit("::", 1)
        return file_path, caller
    return source_key, ""


def _match_call_target(target: str, symbol_name: str) -> bool:
    """
    判断 call 边的 target 是否匹配查询的符号名。

    匹配规则（target 可能是 "save" / "obj.save" / "utils.save"）：
      - 精确匹配：target == symbol_name
      - 限定调用匹配：target 以 "." + symbol_name 结尾
        （obj.save 匹配 save，utils.save 匹配 save）
      - 大小写不敏感

    Args:
        target:       call 边的目标（如 "save", "obj.save"）
        symbol_name:  查询的符号名（如 "save"）

    Returns:
        是否匹配
    """
    if not target or not symbol_name:
        return False
    t = target.lower()
    s = symbol_name.lower()
    if t == s:
        return True
    if t.endswith("." + s):
        return True
    return False


# ============================================================
# 符号精确检索主类
# ============================================================

class SymbolExactMatcher:
    """
    基于 S4 符号表与依赖图的精确检索器。

    核心方法：
      - search(query, top_k): 精确 + 前缀匹配符号，返回候选 chunk 列表
      - get_symbol_definition(name): 定位符号定义（file_path + 行号范围）
      - find_callers(symbol_name, top_k): 反向依赖查询（谁调用了该符号）

    使用方式：
        matcher = SymbolExactMatcher()
        results = matcher.search("#DataProcessor", top_k=10)
        callers = matcher.find_callers("save")
    """

    def __init__(self, vector_store=None, index_service=None, dependency_graph=None):
        """
        Args:
            vector_store:     可选，向量库实例（用于反查 chunk content）
            index_service:    可选，索引服务实例（默认用 get_index_service 单例）
            dependency_graph: 可选，依赖图实例（默认用 get_dependency_graph 单例）
        """
        self._vector_store = vector_store
        self._index_service = index_service
        self._dependency_graph = dependency_graph
        self._lock = threading.Lock()

    # ------------------------------------------------------------
    # 依赖获取
    # ------------------------------------------------------------

    def _get_index_service(self):
        if self._index_service is not None:
            return self._index_service
        from .index_service import get_index_service
        return get_index_service()

    def _get_dependency_graph(self):
        if self._dependency_graph is not None:
            return self._dependency_graph
        from .dependency_graph import get_dependency_graph
        return get_dependency_graph()

    def _get_store(self):
        if self._vector_store is not None:
            return self._vector_store
        from .vector_store import get_vector_store
        return get_vector_store()

    # ------------------------------------------------------------
    # 符号搜索（精确 + 前缀匹配）
    # ------------------------------------------------------------

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        从用户 query 中识别符号 token，对 IndexService 内存符号表做精确 / 前缀匹配。

        匹配优先级：
          1. 精确匹配（name == token，大小写不敏感）→ score=1.0
          2. 前缀匹配（name.startswith(token)）→ score=0.5
          3. 子串包含（token in name，仅当 token 长度 >= 3）→ score=0.3

        Args:
            query:  用户查询文本（可能含 #Tag 或驼峰命名）
            top_k:  返回前 K 个匹配

        Returns:
            候选列表，每项含 id/file_path/symbol_name/chunk_type/
            content/start_line/end_line/score/source。
            content 通过 chunk_id 从 LanceDB 反查补全。
            未命中或符号表为空时返回空列表。
        """
        if not query:
            return []

        tokens = extract_symbol_tokens(query)
        if not tokens:
            return []

        # 1. 遍历 IndexService 内存符号表，做匹配
        try:
            svc = self._get_index_service()
        except Exception as e:
            logger.debug(f"[SymbolExactMatcher] IndexService 不可用: {e}")
            return []

        # 拷贝符号表引用，避免长时间持锁
        with svc._lock:
            file_tables = list(svc._index.items())

        # 收集 (file_path, symbol, match_type, matched_token)
        matches: List[Tuple[str, Any, str, str]] = []
        seen_keys = set()

        for file_path, table in file_tables:
            norm_path = normalize_path(file_path)
            for sym in table.symbols:
                name = sym.name
                if not name:
                    continue
                # 按优先级匹配：exact → prefix → contains
                matched_type = None
                matched_token = None
                for tok in tokens:
                    key = (norm_path, name, tok)
                    if key in seen_keys:
                        continue
                    if name == tok or name.lower() == tok.lower():
                        matched_type = "exact"
                        matched_token = tok
                        seen_keys.add(key)
                        break
                if matched_type is None:
                    for tok in tokens:
                        key = (norm_path, name, tok, "prefix")
                        if key in seen_keys:
                            continue
                        if name.lower().startswith(tok.lower()) and len(tok) >= 1:
                            matched_type = "prefix"
                            matched_token = tok
                            seen_keys.add(key)
                            break
                if matched_type is None:
                    for tok in tokens:
                        key = (norm_path, name, tok, "contains")
                        if key in seen_keys:
                            continue
                        if tok.lower() in name.lower() and len(tok) >= 3:
                            matched_type = "contains"
                            matched_token = tok
                            seen_keys.add(key)
                            break
                if matched_type is not None:
                    matches.append((norm_path, sym, matched_type, matched_token))

        if not matches:
            return []

        # 2. 排序：exact(0) > prefix(1) > contains(2)；同类型按符号名长度升序
        priority = {"exact": 0, "prefix": 1, "contains": 2}
        matches.sort(key=lambda x: (priority[x[2]], len(x[1].name)))
        matches = matches[:top_k]

        # 3. 计算 chunk_id，从 LanceDB 反查 chunk 元信息
        chunk_ids = [
            _compute_chunk_id(fp, sym.name, sym.start_line, sym.end_line)
            for fp, sym, _, _ in matches
        ]
        chunks_by_id: Dict[str, dict] = {}
        try:
            store = self._get_store()
            if store.is_table_exists():
                chunks = store.fetch_chunks_by_ids(chunk_ids)
                chunks_by_id = {c["id"]: c for c in chunks}
        except Exception as e:
            logger.debug(f"[SymbolExactMatcher] LanceDB 反查 chunk 失败: {e}")

        # 4. 组装结果（带 score：exact=1.0, prefix=0.5, contains=0.3）
        score_map = {"exact": 1.0, "prefix": 0.5, "contains": 0.3}
        results: List[Dict[str, Any]] = []
        for (fp, sym, mtype, _), cid in zip(matches, chunk_ids):
            chunk = chunks_by_id.get(cid)
            results.append({
                "id": cid,
                "file_path": fp,
                "symbol_name": sym.name,
                "chunk_type": chunk.get("chunk_type", "") if chunk else "",
                "content": chunk.get("content", "") if chunk else "",
                "start_line": int(chunk.get("start_line", sym.start_line)) if chunk else sym.start_line,
                "end_line": int(chunk.get("end_line", sym.end_line)) if chunk else sym.end_line,
                "score": score_map.get(mtype, 0.0),
                "source": "symbol",
                "match_type": mtype,
            })
        return results

    # ------------------------------------------------------------
    # 符号定义定位
    # ------------------------------------------------------------

    def get_symbol_definition(self, name: str) -> Optional[Dict[str, Any]]:
        """
        精确定位符号定义（用于 #DataProcessor 直接跳转）。

        与 search() 的区别：
          - 仅做精确匹配（name 完全相等，大小写不敏感）
          - 返回第一个命中的符号定义，不返回 chunk content
          - 用于"输入 #DataProcessor，直接定位到定义该类的文件路径和行号范围"

        Args:
            name: 符号名（如 "DataProcessor"）

        Returns:
            {name, type, file_path, start_line, end_line} 或 None
        """
        if not name:
            return None

        try:
            svc = self._get_index_service()
        except Exception as e:
            logger.debug(f"[SymbolExactMatcher] IndexService 不可用: {e}")
            return None

        with svc._lock:
            file_tables = list(svc._index.items())

        name_lower = name.lower()
        for file_path, table in file_tables:
            for sym in table.symbols:
                if sym.name and sym.name.lower() == name_lower:
                    return {
                        "name": sym.name,
                        "type": (
                            sym.symbol_type.value
                            if hasattr(sym.symbol_type, "value")
                            else str(sym.symbol_type)
                        ),
                        "file_path": normalize_path(file_path),
                        "start_line": sym.start_line,
                        "end_line": sym.end_line,
                    }
        return None

    # ------------------------------------------------------------
    # 反向依赖查询（谁调用了某符号）
    # ------------------------------------------------------------

    def find_callers(
        self, symbol_name: str, top_k: int = 20
    ) -> List[Dict[str, Any]]:
        """
        反向依赖查询：找出所有调用了指定符号的函数。

        利用 S4 构建的 Call Graph（DependencyGraph._calls），
        遍历所有 call 边，找出 target 匹配 symbol_name 的调用记录。

        匹配规则（见 _match_call_target）：
          - target == symbol_name（精确）
          - target 以 "." + symbol_name 结尾（obj.save 匹配 save）

        Args:
            symbol_name: 被调用的符号名（如 "save"、"format_data"）
            top_k:       返回前 K 条调用记录

        Returns:
            调用记录列表，每项含：
              - file_path:     调用所在文件（POSIX 相对路径）
              - caller_symbol: 调用者函数名（模块级调用为 ""）
              - line:          调用所在行号
              - raw:           原始调用文本（如 "obj.save"）

        验收场景：输入 "谁调用了 save()"，返回 main.py 第 15 行和 utils.py 第 88 行。
        """
        if not symbol_name:
            return []

        try:
            graph = self._get_dependency_graph()
        except Exception as e:
            logger.debug(f"[SymbolExactMatcher] DependencyGraph 不可用: {e}")
            return []

        # 访问依赖图的 call 边（_calls 为 Dict[source_key, Dict[target, GraphEdge]]）
        with graph._lock:
            call_edges = list(graph._calls.items())

        results: List[Dict[str, Any]] = []
        for source_key, targets in call_edges:
            for target, edge in targets.items():
                if _match_call_target(target, symbol_name):
                    file_path, caller_symbol = _parse_call_source(source_key)
                    results.append({
                        "file_path": normalize_path(file_path),
                        "caller_symbol": caller_symbol,
                        "line": edge.line,
                        "raw": edge.raw,
                    })

        # 按文件路径 + 行号排序，便于阅读
        results.sort(key=lambda x: (x["file_path"], x["line"]))
        return results[:top_k]

    # ------------------------------------------------------------
    # 符号补全（用于 /v1/symbols/search 实时下拉）
    # ------------------------------------------------------------

    def suggest_symbols(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        符号实时补全（供 /v1/symbols/search 接口使用）。

        与 search() 的区别：
          - 不返回 chunk content，仅返回符号元信息（name/type/file_path/line）
          - 排序：精确 > 前缀；同类型按符号名长度升序
          - 用于插件输入 # 后的下拉框（300ms 内返回）

        Args:
            query: 用户输入（如 "Da" 或 "#DataProcessor"）
            limit: 返回数量上限

        Returns:
            [{name, type, file_path, line}, ...]
        """
        if not query:
            return []

        tokens = extract_symbol_tokens(query)
        if not tokens:
            return []

        try:
            svc = self._get_index_service()
        except Exception as e:
            logger.debug(f"[SymbolExactMatcher] IndexService 不可用: {e}")
            return []

        with svc._lock:
            file_tables = list(svc._index.items())

        matches: List[Tuple[str, Any, str]] = []
        seen_keys = set()

        for file_path, table in file_tables:
            norm_path = normalize_path(file_path)
            for sym in table.symbols:
                name = sym.name
                if not name:
                    continue
                for tok in tokens:
                    key = (norm_path, name, tok)
                    if key in seen_keys:
                        continue
                    if name == tok or name.lower() == tok.lower():
                        matches.append((norm_path, sym, "exact"))
                        seen_keys.add(key)
                        break
                else:
                    for tok in tokens:
                        key = (norm_path, name, tok, "prefix")
                        if key in seen_keys:
                            continue
                        if name.lower().startswith(tok.lower()) and len(tok) >= 1:
                            matches.append((norm_path, sym, "prefix"))
                            seen_keys.add(key)
                            break

        if not matches:
            return []

        priority = {"exact": 0, "prefix": 1}
        matches.sort(key=lambda x: (priority[x[2]], len(x[1].name)))
        matches = matches[:limit]

        return [
            {
                "name": sym.name,
                "type": (
                    sym.symbol_type.value
                    if hasattr(sym.symbol_type, "value")
                    else str(sym.symbol_type)
                ),
                "file_path": fp,
                "line": sym.start_line,
            }
            for fp, sym, _ in matches
        ]


# ============================================================
# 单例工厂
# ============================================================

_symbol_exact_matcher: Optional[SymbolExactMatcher] = None


def get_symbol_exact_matcher() -> SymbolExactMatcher:
    """获取 SymbolExactMatcher 单例"""
    global _symbol_exact_matcher
    if _symbol_exact_matcher is None:
        _symbol_exact_matcher = SymbolExactMatcher()
    return _symbol_exact_matcher


# ============================================================
# 便捷函数
# ============================================================

def symbol_search(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """便捷函数：符号精确检索"""
    return get_symbol_exact_matcher().search(query, top_k=top_k)


def find_callers(symbol_name: str, top_k: int = 20) -> List[Dict[str, Any]]:
    """便捷函数：反向依赖查询（谁调用了该符号）"""
    return get_symbol_exact_matcher().find_callers(symbol_name, top_k=top_k)
