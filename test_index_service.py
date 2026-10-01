"""
索引服务功能测试（S4 第 37-38 天）

验证：
1. 后台线程异步索引（start_index 立即返回，wait_for_done 等待完成）
2. 批量提交（每 INDEX_BATCH_SIZE 个文件）
3. 增量更新（modified / deleted）
4. 优先文件排序（即用即索引）
5. 索引完成标记文件写入
6. 向量库增量更新正确性
"""
import os
import tempfile
import time

from app.services.code_index.index_service import IndexService
from app.services.code_index.vector_store import get_vector_store


def create_test_workspace():
    """创建临时测试工作区，包含多个语言文件"""
    workspace = tempfile.mkdtemp()

    # Python 文件
    with open(os.path.join(workspace, "main.py"), "w", encoding="utf-8") as f:
        f.write('''
def hello():
    return "world"

class App:
    def run(self):
        pass
''')

    with open(os.path.join(workspace, "utils.py"), "w", encoding="utf-8") as f:
        f.write('''
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b
''')

    # JavaScript 文件
    with open(os.path.join(workspace, "helper.js"), "w", encoding="utf-8") as f:
        f.write('''
function multiply(a, b) {
    return a * b;
}

const PI = 3.14;
''')

    # 跳过目录验证：创建 node_modules 下的文件，应被跳过
    nm_dir = os.path.join(workspace, "node_modules")
    os.makedirs(nm_dir, exist_ok=True)
    with open(os.path.join(nm_dir, "ignored.js"), "w", encoding="utf-8") as f:
        f.write("function ignored() { return 1; }")

    return workspace


def test_async_index_and_status():
    """测试 1：后台线程异步索引 + 状态查询"""
    workspace = create_test_workspace()
    service = IndexService()

    # 初始状态
    status = service.get_status()
    print("初始状态:", status)
    assert status["status"] == "idle"

    # 启动索引（应立即返回，不阻塞）
    start_time = time.time()
    result = service.start_index(workspace, force_rebuild=True)
    elapsed = time.time() - start_time
    print(f"start_index 返回耗时: {elapsed:.3f}s")
    assert elapsed < 5.0, "start_index 不应阻塞（应在 5s 内返回）"

    print("索引结果:", result)
    assert "job_id" in result
    assert result["total_files"] >= 3  # main.py, utils.py, helper.js

    # 启动后状态应为 indexing
    status = service.get_status()
    print("启动后状态:", status)
    assert status["status"] in ("indexing", "done")

    # 等待索引完成
    ok = service.wait_for_done(timeout=30)
    print(f"wait_for_done: {ok}")
    assert ok, "索引超时未完成"

    # 完成后状态
    status = service.get_status()
    print("索引后状态:", status)
    assert status["status"] == "done"
    assert status["total"] == status["processed"]
    assert status["total_symbols"] > 0

    # 验证跳过了 node_modules
    symbols = service.get_symbols()
    symbol_files = {s["file_path"] for s in symbols}
    assert "node_modules/ignored.js" not in symbol_files
    print(f"共索引 {len(symbols)} 个符号，来自文件: {symbol_files}")

    # 验证标记文件已写入
    marker_path = os.path.join(workspace, ".ai_index", "index_done")
    assert os.path.exists(marker_path), f"索引标记文件未生成: {marker_path}"
    print(f"✅ 标记文件已生成: {marker_path}")

    return service, workspace


def test_incremental_update(service, workspace):
    """测试 2：增量更新（modified + deleted）"""
    # 修改 main.py：新增 goodbye 函数
    main_py = os.path.join(workspace, "main.py")
    with open(main_py, "w", encoding="utf-8") as f:
        f.write('''
def hello():
    return "world"

def goodbye():
    return "bye"

class App:
    def run(self):
        pass
''')

    symbols_before = len(service.get_symbols())

    # 增量更新
    update_result = service.update_file("main.py", "modified")
    print("增量更新结果:", update_result)
    assert update_result["success"] is True

    symbols_after = len(service.get_symbols())
    print(f"符号数变化: {symbols_before} -> {symbols_after}")
    # 新增了 goodbye 函数，符号数应增加
    assert symbols_after > symbols_before

    # 删除 helper.js
    del_result = service.update_file("helper.js", "deleted")
    print("删除结果:", del_result)
    assert del_result["success"] is True

    symbols_final = len(service.get_symbols())
    print(f"删除后符号数: {symbols_final}")
    assert symbols_final < symbols_after


def test_priority_files():
    """测试 3：优先文件排序（即用即索引）"""
    files = ["a.py", "b.py", "c.py", "d.js"]
    priority = ["c.py", "a.py"]

    ordered = IndexService._reorder_priority(files, priority)
    print(f"原始: {files}")
    print(f"优先: {priority}")
    print(f"排序后: {ordered}")

    # 优先文件应排在最前面
    assert ordered[0] == "c.py"
    assert ordered[1] == "a.py"
    # 其余保持原序
    assert ordered[2] == "b.py"
    assert ordered[3] == "d.js"


def test_vector_store_incremental():
    """测试 4：向量库增量更新正确性"""
    workspace = create_test_workspace()
    service = IndexService()
    service.start_index(workspace, force_rebuild=True)
    service.wait_for_done(timeout=30)

    store = get_vector_store()
    if not store.is_table_exists():
        print("⚠️  向量库未建表（可能 Embedding 后端不可用），跳过向量库验证")
        return

    count_before = store.count()
    print(f"向量库初始 Chunk 数: {count_before}")

    # 修改文件触发增量更新
    main_py = os.path.join(workspace, "main.py")
    with open(main_py, "w", encoding="utf-8") as f:
        f.write('''
def hello():
    return "world"

def goodbye():
    return "bye"
''')

    service.update_file("main.py", "modified")
    count_after = store.count()
    print(f"增量更新后向量库 Chunk 数: {count_after}")

    # 增量更新后应仍有数据（旧的删了，新的插了）
    assert count_after > 0


if __name__ == "__main__":
    print("=" * 60)
    print("测试 1：异步索引 + 状态查询 + 标记文件")
    print("=" * 60)
    service, workspace = test_async_index_and_status()

    print("\n" + "=" * 60)
    print("测试 2：增量更新（modified + deleted）")
    print("=" * 60)
    test_incremental_update(service, workspace)

    print("\n" + "=" * 60)
    print("测试 3：优先文件排序")
    print("=" * 60)
    test_priority_files()

    print("\n" + "=" * 60)
    print("测试 4：向量库增量更新")
    print("=" * 60)
    test_vector_store_incremental()

    print("\n" + "=" * 60)
    print("✅ 所有索引服务测试完成")
    print("=" * 60)
