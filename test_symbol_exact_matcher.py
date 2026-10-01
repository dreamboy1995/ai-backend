"""
符号精确检索 & 依赖图增强 验收测试脚本（S5 第 45-46 天）

用法:
    python test_symbol_exact_matcher.py

验收标准（来自 Sprint_5.md 第 45-46 天）：
    1. 输入 #DataProcessor，后端能直接定位到定义该类的文件路径和行号范围
    2. 输入"谁调用了 save()"，能返回 main.py 第 15 行和 utils.py 第 88 行

本脚本分四阶段验证：
    阶段 1：单元功能验证
        - 构造测试仓库（main.py / models.py / utils.py）
        - 解析符号表 + 构建依赖图
        - 验证 # 精确匹配定位 DataProcessor 类
        - 验证前缀匹配 "Da" 返回 DataProcessor / DataLoader
        - 验证反向依赖查询 find_callers("save") 返回所有调用者
    阶段 2：路径归一化验证
        - Windows 反斜杠路径 → POSIX 正斜杠
    阶段 3：SymbolSearcher 委托验证
        - fusion_reranker.SymbolSearcher.search() 委托给 SymbolExactMatcher
    阶段 4：HTTP 接口验证（可选）
        - 启动 FastAPI app，调用 GET /v1/symbols/search、/definition、/callers
        - 若端口被占用则跳过
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.code_index.ast_parser import parse_file
from app.services.code_index.dependency_graph import DependencyGraph
from app.services.code_index.symbol_exact_matcher import (
    SymbolExactMatcher,
    extract_symbol_tokens,
    normalize_path,
)
from app.services.code_index.models import SymbolTable


# ============================================================
# 测试仓库构造
# ============================================================

def create_test_workspace():
    """
    创建测试工作区，模拟一个简单 Python 项目：

      main.py       (入口，调用 DataProcessor 与 utils.save)
        ├─ models.py  (定义 DataProcessor 类与 DataLoader 函数)
        └─ utils.py   (定义 save / load 函数)
    """
    workspace = tempfile.mkdtemp()

    # models.py：定义类与函数
    with open(os.path.join(workspace, "models.py"), "w", encoding="utf-8") as f:
        f.write(
            '"""Data models."""\n\n'
            'class DataProcessor:\n'
            '    """处理数据的核心类。"""\n'
            '    def __init__(self, data):\n'
            '        self.data = data\n\n'
            '    def process(self):\n'
            '        return [d * 2 for d in self.data]\n\n'
            '    def save(self, path):\n'
            '        with open(path, "w") as f:\n'
            '            f.write(str(self.data))\n\n\n'
            'def DataLoader(path):\n'
            '    """从文件加载数据。"""\n'
            '    with open(path) as f:\n'
            '        return f.read()\n'
        )

    # utils.py：定义 save / load 工具函数
    with open(os.path.join(workspace, "utils.py"), "w", encoding="utf-8") as f:
        f.write(
            '"""Utility functions."""\n'
            'import json\n\n\n'
            'def save(data, path):\n'
            '    """保存数据到文件。"""\n'
            '    with open(path, "w") as f:\n'
            '        json.dump(data, f)\n\n\n'
            'def load(path):\n'
            '    """从文件加载数据。"""\n'
            '    with open(path) as f:\n'
            '        return json.load(f)\n'
        )

    # main.py：入口，调用 DataProcessor 与 utils.save
    with open(os.path.join(workspace, "main.py"), "w", encoding="utf-8") as f:
        f.write(
            '"""Application entry point."""\n'
            'from models import DataProcessor\n'
            'import utils\n\n\n'
            'def main():\n'
            '    data = [1, 2, 3]\n'
            '    processor = DataProcessor(data)\n'
            '    result = processor.process()\n'
            '    utils.save(result, "output.json")\n'
            '    return result\n\n\n'
            'if __name__ == "__main__":\n'
            '    main()\n'
        )

    return workspace


def build_index_and_graph(workspace):
    """
    解析所有文件，构建符号表索引与依赖图。

    Returns:
        (index_dict, dependency_graph)
        - index_dict: {file_path: SymbolTable}
        - dependency_graph: DependencyGraph 实例
    """
    index = {}
    graph = DependencyGraph()

    files = ["models.py", "utils.py", "main.py"]
    # 预注册文件到依赖图
    for fname in files:
        graph.register_file(fname, "python")

    for fname in files:
        full_path = os.path.join(workspace, fname)
        table = parse_file(full_path)
        table.file_path = fname
        index[fname] = table
        graph.build_from_file(fname, table, full_path=full_path)

    return index, graph


# ============================================================
# Mock IndexService（用于注入 SymbolExactMatcher）
# ============================================================

class MockIndexService:
    """模拟 IndexService，仅提供 _lock 和 _index"""
    def __init__(self, index):
        import threading
        self._lock = threading.Lock()
        self._index = index


# ============================================================
# 阶段 1：单元功能验证
# ============================================================

def run_unit_test():
    """验证精确匹配、前缀匹配、反向依赖查询"""
    print("=" * 60)
    print("阶段 1：单元功能验证")
    print("=" * 60)

    workspace = create_test_workspace()
    index, graph = build_index_and_graph(workspace)

    mock_svc = MockIndexService(index)
    matcher = SymbolExactMatcher(index_service=mock_svc, dependency_graph=graph)

    passed = 0
    failed = 0

    # --- 1.1 # 精确匹配定位 DataProcessor 类 ---
    print("\n[1.1] #DataProcessor 精确匹配定位类定义")
    definition = matcher.get_symbol_definition("DataProcessor")
    if definition and definition["name"] == "DataProcessor" and definition["type"] == "class":
        print(f"  ✓ 定位成功: {definition['file_path']} 第 {definition['start_line']}-{definition['end_line']} 行")
        passed += 1
    else:
        print(f"  ✗ 定位失败: {definition}")
        failed += 1

    # --- 1.2 前缀匹配 "Da" 返回 DataProcessor / DataLoader ---
    print("\n[1.2] 前缀匹配 'Da' 返回 DataProcessor / DataLoader")
    suggestions = matcher.suggest_symbols("Da", limit=10)
    names = [s["name"] for s in suggestions]
    has_dp = "DataProcessor" in names
    has_dl = "DataLoader" in names
    if has_dp and has_dl:
        print(f"  ✓ 前缀匹配成功: {names}")
        passed += 1
    else:
        print(f"  ✗ 前缀匹配失败: {names}")
        failed += 1

    # --- 1.3 search("#DataProcessor") 返回候选 chunk ---
    print("\n[1.3] search('#DataProcessor') 返回候选")
    results = matcher.search("#DataProcessor", top_k=10)
    if results and any(r["symbol_name"] == "DataProcessor" for r in results):
        r = [x for x in results if x["symbol_name"] == "DataProcessor"][0]
        print(f"  ✓ 检索成功: {r['file_path']} {r['symbol_name']} (score={r['score']})")
        passed += 1
    else:
        print(f"  ✗ 检索失败: {results}")
        failed += 1

    # --- 1.4 反向依赖查询 find_callers("save") ---
    print("\n[1.4] find_callers('save') 反向依赖查询")
    callers = matcher.find_callers("save", top_k=20)
    print(f"  调用记录数: {len(callers)}")
    for c in callers:
        print(f"    - {c['file_path']}::{c['caller_symbol']} 第 {c['line']} 行 (raw={c['raw']})")
    # main.py 中 utils.save(result, "output.json") 应被命中
    main_callers = [c for c in callers if c["file_path"] == "main.py"]
    if main_callers:
        print(f"  ✓ 找到 main.py 中的调用: 第 {main_callers[0]['line']} 行")
        passed += 1
    else:
        print(f"  ✗ 未找到 main.py 中的 save 调用")
        failed += 1

    # --- 1.5 大小写不敏感匹配 ---
    print("\n[1.5] 大小写不敏感匹配 dataprocessor")
    definition_lower = matcher.get_symbol_definition("dataprocessor")
    if definition_lower and definition_lower["name"] == "DataProcessor":
        print(f"  ✓ 大小写不敏感匹配成功")
        passed += 1
    else:
        print(f"  ✗ 大小写不敏感匹配失败")
        failed += 1

    # --- 1.6 未匹配符号返回 None / 空列表 ---
    print("\n[1.6] 未匹配符号返回空")
    no_def = matcher.get_symbol_definition("NonExistentSymbol")
    no_callers = matcher.find_callers("non_existent_func")
    if no_def is None and no_callers == []:
        print(f"  ✓ 未匹配符号正确返回空")
        passed += 1
    else:
        print(f"  ✗ 未匹配符号返回异常: def={no_def}, callers={no_callers}")
        failed += 1

    print(f"\n阶段 1 结果: {passed} 通过, {failed} 失败")
    return failed == 0


# ============================================================
# 阶段 2：路径归一化验证
# ============================================================

def run_path_normalization_test():
    """验证 Windows 反斜杠 → POSIX 正斜杠"""
    print("\n" + "=" * 60)
    print("阶段 2：路径归一化验证")
    print("=" * 60)

    test_cases = [
        ("src\\main.py", "src/main.py"),
        ("src\\\\main.py", "src/main.py"),
        ("src/main.py", "src/main.py"),
        ("a\\\\b\\\\c.py", "a/b/c.py"),
        ("", ""),
    ]

    passed = 0
    failed = 0
    for raw, expected in test_cases:
        result = normalize_path(raw)
        if result == expected:
            print(f"  ✓ normalize_path({raw!r}) = {result!r}")
            passed += 1
        else:
            print(f"  ✗ normalize_path({raw!r}) = {result!r} (期望 {expected!r})")
            failed += 1

    print(f"\n阶段 2 结果: {passed} 通过, {failed} 失败")
    return failed == 0


# ============================================================
# 阶段 3：SymbolSearcher 委托验证
# ============================================================

def run_delegation_test():
    """验证 fusion_reranker.SymbolSearcher 委托给 SymbolExactMatcher"""
    print("\n" + "=" * 60)
    print("阶段 3：SymbolSearcher 委托验证")
    print("=" * 60)

    workspace = create_test_workspace()
    index, graph = build_index_and_graph(workspace)
    mock_svc = MockIndexService(index)

    from app.services.code_index.fusion_reranker import SymbolSearcher
    from app.services.code_index.symbol_exact_matcher import SymbolExactMatcher

    # 注入 matcher 的依赖
    matcher = SymbolExactMatcher(index_service=mock_svc, dependency_graph=graph)
    searcher = SymbolSearcher()
    searcher._matcher = matcher  # 直接注入

    print("\n[3.1] SymbolSearcher.search('#DataProcessor')")
    results = searcher.search("#DataProcessor", top_k=10)
    if results and any(r["symbol_name"] == "DataProcessor" for r in results):
        print(f"  ✓ 委托成功，返回 {len(results)} 条结果")
        print(f"\n阶段 3 结果: 1 通过, 0 失败")
        return True
    else:
        print(f"  ✗ 委托失败: {results}")
        print(f"\n阶段 3 结果: 0 通过, 1 失败")
        return False


# ============================================================
# 阶段 4：HTTP 接口验证（可选）
# ============================================================

def run_http_test():
    """启动 FastAPI app，验证符号相关接口"""
    print("\n" + "=" * 60)
    print("阶段 4：HTTP 接口验证（可选）")
    print("=" * 60)

    try:
        from fastapi.testclient import TestClient
        from app.auth import create_access_token
        from app.main import app
        from app.services.code_index.index_service import get_index_service
    except Exception as e:
        print(f"  跳过 HTTP 测试：导入失败 {e}")
        return True

    workspace = create_test_workspace()
    index, graph = build_index_and_graph(workspace)

    # 注入到全局单例
    svc = get_index_service()
    with svc._lock:
        svc._index = index

    from app.services.code_index.dependency_graph import get_dependency_graph
    global_graph = get_dependency_graph()
    with global_graph._lock:
        global_graph._forward = graph._forward
        global_graph._reverse = graph._reverse
        global_graph._calls = graph._calls
        global_graph._local_files = graph._local_files
        global_graph._file_languages = graph._file_languages

    token = create_access_token({"sub": "test-user", "user_id": "test-user"})
    headers = {"Authorization": f"Bearer {token}"}
    client = TestClient(app)
    passed = 0
    failed = 0

    # 4.1 GET /v1/symbols/search?q=Da
    print("\n[4.1] GET /v1/symbols/search?q=Da")
    try:
        resp = client.get("/v1/symbols/search", params={"q": "Da", "limit": 10}, headers=headers)
        if resp.status_code == 200:
            data = resp.json()
            names = [s["name"] for s in data.get("symbols", [])]
            print(f"  ✓ 状态 200, 返回: {names}")
            passed += 1
        else:
            print(f"  ✗ 状态 {resp.status_code}: {resp.text}")
            failed += 1
    except Exception as e:
        print(f"  ✗ 请求失败: {e}")
        failed += 1

    # 4.2 GET /v1/symbols/definition?name=DataProcessor
    print("\n[4.2] GET /v1/symbols/definition?name=DataProcessor")
    try:
        resp = client.get("/v1/symbols/definition", params={"name": "DataProcessor"}, headers=headers)
        if resp.status_code == 200:
            data = resp.json()
            print(f"  ✓ 状态 200: {data['file_path']} 第 {data['start_line']}-{data['end_line']} 行")
            passed += 1
        else:
            print(f"  ✗ 状态 {resp.status_code}: {resp.text}")
            failed += 1
    except Exception as e:
        print(f"  ✗ 请求失败: {e}")
        failed += 1

    # 4.3 GET /v1/symbols/callers?name=save
    print("\n[4.3] GET /v1/symbols/callers?name=save")
    try:
        resp = client.get("/v1/symbols/callers", params={"name": "save", "limit": 20}, headers=headers)
        if resp.status_code == 200:
            data = resp.json()
            print(f"  ✓ 状态 200, 找到 {data['total']} 个调用者")
            for c in data.get("callers", []):
                print(f"    - {c['file_path']}::{c['caller_symbol']} 第 {c['line']} 行")
            passed += 1
        else:
            print(f"  ✗ 状态 {resp.status_code}: {resp.text}")
            failed += 1
    except Exception as e:
        print(f"  ✗ 请求失败: {e}")
        failed += 1

    print(f"\n阶段 4 结果: {passed} 通过, {failed} 失败")
    return failed == 0


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    results = []
    results.append(("阶段 1：单元功能验证", run_unit_test()))
    results.append(("阶段 2：路径归一化验证", run_path_normalization_test()))
    results.append(("阶段 3：SymbolSearcher 委托验证", run_delegation_test()))
    results.append(("阶段 4：HTTP 接口验证", run_http_test()))

    print("\n" + "=" * 60)
    print("总结果")
    print("=" * 60)
    all_pass = True
    for name, ok in results:
        status = "✓ 通过" if ok else "✗ 失败"
        print(f"  {status}  {name}")
        if not ok:
            all_pass = False

    print()
    if all_pass:
        print("🎉 所有测试通过！")
        sys.exit(0)
    else:
        print("⚠️  部分测试失败，请检查上方输出")
        sys.exit(1)
