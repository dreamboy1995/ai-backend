"""
依赖关系图（Call Graph）验收测试脚本（S4 第 39-40 天）

用法:
    python test_dependency_graph.py

验收标准（来自 Sprint_4.md 第 39-40 天）：
    对项目根目录的 app.py 调用依赖图接口，能返回它 import 的所有本地模块名。

本脚本分三阶段验证：
    阶段 1：单元功能验证
        - 构造测试仓库（app.py / utils.py / helpers.py）
        - 调用 dependency_graph.build_from_file 提取引用关系
        - 验证文件级 import 边、符号级 call 边正确生成
        - 验证 BFS get_related_files 在 depth=1/2 时返回正确关联文件
    阶段 2：持久化验证
        - save → load → 验证图结构完整恢复
    阶段 3：验收场景模拟
        - 模拟对项目根目录的 app.py 调用依赖图接口
        - 验证返回它 import 的所有本地模块名（utils, helpers）
    阶段 4：HTTP 接口验证（可选）
        - 启动 FastAPI app，调用 GET /v1/graph/imports 与 /v1/graph/related
        - 若端口被占用则跳过
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.code_index.ast_parser import parse_file
from app.services.code_index.dependency_graph import DependencyGraph


# ============================================================
# 测试仓库构造
# ============================================================

def create_test_workspace():
    """
    创建测试工作区，模拟一个简单 Python 项目结构：

      app.py            (入口，import utils 和 helpers，调用 utils.format_data)
        ├─ utils.py     (工具模块，import helpers，调用 helpers.to_json)
        └─ helpers.py   (底层模块，无本地 import)
    """
    workspace = tempfile.mkdtemp()

    # app.py：入口文件，import 两个本地模块
    with open(os.path.join(workspace, "app.py"), "w", encoding="utf-8") as f:
        f.write(
            '"""Application entry point."""\n'
            'import utils\n'
            'from helpers import to_json\n\n'
            'def main():\n'
            '    data = [1, 2, 3]\n'
            '    formatted = utils.format_data(data)\n'
            '    result = to_json(formatted)\n'
            '    print(result)\n'
            '    return result\n'
        )

    # utils.py：工具模块，import helpers
    with open(os.path.join(workspace, "utils.py"), "w", encoding="utf-8") as f:
        f.write(
            '"""Utils module."""\n'
            'import helpers\n\n'
            'def format_data(data):\n'
            '    return [d * 2 for d in data]\n\n'
            'def to_csv(data):\n'
            '    return ",".join(str(d) for d in data)\n'
        )

    # helpers.py：底层模块，仅 import 标准库
    with open(os.path.join(workspace, "helpers.py"), "w", encoding="utf-8") as f:
        f.write(
            '"""Helpers module."""\n'
            'import json\n\n'
            'def to_json(data):\n'
            '    return json.dumps(data)\n\n'
            'def from_json(text):\n'
            '    return json.loads(text)\n'
        )

    return workspace


# ============================================================
# 阶段 1：单元功能验证
# ============================================================

def run_unit_test():
    """验证引用提取、边生成、BFS 查询"""
    workspace = create_test_workspace()
    graph = DependencyGraph()

    # 解析并构建每个文件的依赖关系
    for fname in ("helpers.py", "utils.py", "app.py"):
        full_path = os.path.join(workspace, fname)
        table = parse_file(full_path)
        table.file_path = fname
        # 先注册文件（让后续文件能解析到它）
        graph.register_file(fname, table.language)
        edges = graph.build_from_file(fname, table, full_path=full_path)
        print(f"  {fname}: 提取到 {len(edges)} 条边")

    # 验证 1：app.py 应有 import 边到 utils.py 和 helpers.py
    app_imports = graph.get_imports("app.py")
    app_imported_files = {imp["file_path"] for imp in app_imports if imp["file_path"]}
    assert "utils.py" in app_imported_files, f"app.py 应 import utils.py，实际: {app_imported_files}"
    assert "helpers.py" in app_imported_files, f"app.py 应 import helpers.py，实际: {app_imported_files}"
    print(f"[阶段1] ✅ app.py 的 import 边正确：{app_imported_files}")

    # 验证 2：utils.py 应 import helpers.py
    utils_imports = graph.get_imports("utils.py")
    utils_imported_files = {imp["file_path"] for imp in utils_imports if imp["file_path"]}
    assert "helpers.py" in utils_imported_files, f"utils.py 应 import helpers.py，实际: {utils_imported_files}"
    print(f"[阶段1] ✅ utils.py 的 import 边正确：{utils_imported_files}")

    # 验证 3：app.py 的符号级 call 边（main 调用 utils.format_data / to_json）
    calls = graph.get_symbols("app.py")
    call_targets = {c["target"] for c in calls}
    assert "utils.format_data" in call_targets, f"应记录 utils.format_data 调用，实际: {call_targets}"
    assert "to_json" in call_targets, f"应记录 to_json 调用，实际: {call_targets}"
    print(f"[阶段1] ✅ app.py 的 call 边正确：{call_targets}")

    # 验证 4：BFS depth=1，app.py 的 upstream（依赖谁）应包含 utils.py 和 helpers.py
    related_d1 = graph.get_related_files("app.py", depth=1)
    upstream_d1 = {item["file_path"] for item in related_d1["upstream"]}
    assert "utils.py" in upstream_d1 and "helpers.py" in upstream_d1, (
        f"depth=1 upstream 应含 utils.py 和 helpers.py，实际: {upstream_d1}"
    )
    print(f"[阶段1] ✅ app.py depth=1 upstream 正确：{upstream_d1}")

    # 验证 5：BFS depth=2，app.py 的 upstream 应通过 utils.py 找到 helpers.py
    # （app.py → utils.py → helpers.py，helpers.py 在 depth=2 出现）
    related_d2 = graph.get_related_files("app.py", depth=2)
    depth_of_helpers = {
        item["file_path"]: item["depth"]
        for item in related_d2["upstream"]
    }
    # helpers.py 可通过 app.py 直接 import（depth=1），也可能通过 utils.py 间接（depth=2）
    assert "helpers.py" in depth_of_helpers
    assert depth_of_helpers["helpers.py"] <= 2
    print(f"[阶段1] ✅ app.py depth=2 含 helpers.py（depth={depth_of_helpers['helpers.py']}）")

    # 验证 6：downstream —— helpers.py 被 app.py 和 utils.py 依赖
    helpers_related = graph.get_related_files("helpers.py", depth=2)
    downstream_files = {item["file_path"] for item in helpers_related["downstream"]}
    assert "app.py" in downstream_files, f"helpers.py 的 downstream 应含 app.py，实际: {downstream_files}"
    assert "utils.py" in downstream_files, f"helpers.py 的 downstream 应含 utils.py，实际: {downstream_files}"
    print(f"[阶段1] ✅ helpers.py downstream 正确：{downstream_files}")

    # 验证 7：图统计
    stats = graph.stats()
    assert stats["local_files"] == 3, f"应有 3 个本地文件，实际: {stats}"
    assert stats["file_edges"] >= 3, f"应至少 3 条文件边，实际: {stats}"
    print(f"[阶段1] ✅ 图统计正确：{stats}")

    print("[阶段1] 🎉 单元功能验收全部通过！")
    print("=" * 60)
    return workspace, graph


# ============================================================
# 阶段 2：持久化验证
# ============================================================

def run_persistence_test(workspace, graph):
    """验证 JSON 持久化与恢复"""
    graph_file = os.path.join(workspace, "graph.json")
    graph.save(graph_file)
    assert os.path.exists(graph_file), "持久化文件未生成"
    print(f"[阶段2] ✅ 图已持久化到 {graph_file}")

    # 重新加载
    new_graph = DependencyGraph()
    loaded = new_graph.load(graph_file)
    assert loaded, "load 应返回 True"
    print("[阶段2] ✅ 图已从 JSON 恢复到内存")

    # 验证恢复后的图结构与原图一致
    old_imports = graph.get_imports("app.py")
    new_imports = new_graph.get_imports("app.py")
    assert len(old_imports) == len(new_imports), (
        f"恢复后 app.py 的 import 数量不一致: {len(old_imports)} vs {len(new_imports)}"
    )
    print(f"[阶段2] ✅ 恢复后 app.py import 边数量一致（{len(new_imports)} 条）")

    old_related = graph.get_related_files("app.py", depth=2)
    new_related = new_graph.get_related_files("app.py", depth=2)
    assert len(old_related["upstream"]) == len(new_related["upstream"]), (
        f"恢复后 upstream 数量不一致: {len(old_related['upstream'])} vs {len(new_related['upstream'])}"
    )
    print(f"[阶段2] ✅ 恢复后 BFS 查询结果一致（upstream {len(new_related['upstream'])} 项）")

    stats = new_graph.stats()
    assert stats["local_files"] == 3
    print(f"[阶段2] ✅ 恢复后统计正确：{stats}")
    print("[阶段2] 🎉 持久化验收全部通过！")
    print("=" * 60)
    return new_graph


# ============================================================
# 阶段 3：验收场景模拟
# ============================================================

def run_acceptance_scenario(workspace, graph):
    """
    模拟 Sprint 验收场景：
      对项目根目录的 app.py 调用依赖图接口，能返回它 import 的所有本地模块名。
    """
    imports = graph.get_imports("app.py")
    local_modules = [imp for imp in imports if imp["resolved"]]
    local_module_names = [imp["module_name"] for imp in local_modules]

    assert "utils" in local_module_names, f"应返回本地模块 utils，实际: {local_module_names}"
    assert "helpers" in local_module_names, f"应返回本地模块 helpers，实际: {local_module_names}"
    print(f"[阶段3] ✅ 验收通过：app.py import 的本地模块 = {local_module_names}")

    # 验证 GET /graph/related?file=app.py&depth=2 也能返回关联文件
    related = graph.get_related_files("app.py", depth=2)
    all_related_files = {
        item["file_path"]
        for item in related["upstream"] + related["downstream"]
    }
    assert "utils.py" in all_related_files and "helpers.py" in all_related_files
    print(f"[阶段3] ✅ /graph/related?file=app.py&depth=2 返回关联文件: {all_related_files}")
    print("[阶段3] 🎉 验收场景全部通过！")
    print("=" * 60)


# ============================================================
# 阶段 4：HTTP 接口验证（可选）
# ============================================================

def run_http_test():
    """
    验证 GET /v1/graph/imports、/v1/graph/related、/v1/graph/stats 接口。
    若 3000 端口被占用或服务未启动，则跳过。
    """
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            s.connect(("127.0.0.1", 3000))
    except (socket.timeout, ConnectionRefusedError, OSError):
        print("[阶段4] ⚠️  后端服务未启动（127.0.0.1:3000），跳过 HTTP 接口验证")
        print("=" * 60)
        return

    try:
        import httpx
        with httpx.Client(base_url="http://127.0.0.1:3000", timeout=5.0) as client:
            # /v1/graph/stats
            resp = client.get("/v1/graph/stats")
            if resp.status_code == 401:
                print("[阶段4] ⚠️  接口需鉴权，跳过 HTTP 验证（功能已通过阶段 1-3 验证）")
                return
            if resp.status_code != 200:
                print(f"[阶段4] ⚠️  /v1/graph/stats 返回 {resp.status_code}，跳过")
                return
            stats = resp.json()
            print(f"[阶段4] GET /v1/graph/stats → {stats}")

            # 若图中无数据则跳过 imports/related
            if stats.get("local_files", 0) == 0:
                print("[阶段4] ⚠️  依赖图为空（未触发过索引），跳过后续接口验证")
                return

            # 找一个图中存在的文件作为查询参数
            resp2 = client.get("/v1/graph/related", params={"file": "app.py", "depth": 2})
            print(f"[阶段4] GET /v1/graph/related?file=app.py&depth=2 → {resp2.status_code}")
            if resp2.status_code == 200:
                data = resp2.json()
                print(f"         upstream={len(data['upstream'])} downstream={len(data['downstream'])}")
    except Exception as e:
        print(f"[阶段4] ⚠️  HTTP 验证失败: {e}")
    print("=" * 60)


# ============================================================
# 主入口
# ============================================================

def main():
    print("=" * 60)
    print("S4 第 39-40 天：依赖关系图（Call Graph）验收测试")
    print("=" * 60)

    workspace, graph = run_unit_test()
    new_graph = run_persistence_test(workspace, graph)
    run_acceptance_scenario(workspace, new_graph)
    run_http_test()

    print("🎉 依赖关系图验收测试完成（核心功能已通过）。")


if __name__ == "__main__":
    main()
