"""索引服务功能测试"""
import os
import tempfile
from app.services.code_index.index_service import IndexService

# 创建临时工作区
with tempfile.TemporaryDirectory() as workspace:
    # 创建几个测试文件
    py_file = os.path.join(workspace, "main.py")
    with open(py_file, "w", encoding="utf-8") as f:
        f.write("""
def hello():
    return "world"

class App:
    def run(self):
        pass
""")

    js_file = os.path.join(workspace, "utils.js")
    with open(js_file, "w", encoding="utf-8") as f:
        f.write("""
function add(a, b) {
    return a + b;
}
""")

    service = IndexService()

    # 1. 初始状态
    status = service.get_status()
    print("初始状态:", status)

    # 2. 启动全量索引
    result = service.start_index(workspace)
    print("索引结果:", result)

    # 3. 查询状态
    status = service.get_status()
    print("索引后状态:", status)

    # 4. 查看已索引符号
    symbols = service.get_symbols()
    print(f"共索引 {len(symbols)} 个符号:")
    for s in symbols:
        print(f"  {s['symbol_type']}: {s['name']} ({s['file_path']} L{s['start_line']}-{s['end_line']})")

    # 5. 增量更新：修改 main.py
    with open(py_file, "w", encoding="utf-8") as f:
        f.write("""
def hello():
    return "world"

def goodbye():
    return "bye"

class App:
    def run(self):
        pass
""")
    update_result = service.update_file(workspace, "main.py", "modified")
    print("\n增量更新结果:", update_result)

    status = service.get_status()
    print("更新后符号总数:", status["total_symbols"])

    # 6. 增量更新：删除文件
    del_result = service.update_file(workspace, "utils.js", "deleted")
    print("删除结果:", del_result)

    status = service.get_status()
    print("删除后符号总数:", status["total_symbols"])

    print("\n✅ 索引服务功能测试完成")
