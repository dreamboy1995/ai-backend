"""
代码切片 & 向量化测试脚本（S4 第 33-34 天验收用）

用法:
    python test_chunker.py

验收标准：
    1. 对一个 ~300 行的 Python 文件跑切片，产出 5-8 个 Chunk。
    2. 每个 Chunk 的 embedding 数组长度为 384。
    3. 向量数值有变化（非全零）。
    4. import 块、函数块、类块均被正确识别。
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.code_index.ast_parser import parse_file
from app.services.code_index.code_chunker import chunk_file
from app.services.code_index.embedding_client import EmbeddingClient
from app.services.code_index.models import ChunkType


def build_large_python_file() -> str:
    """构造一个约 300 行的 Python 测试文件"""
    lines = [
        '"""A large sample module for chunker testing."""',
        "import os",
        "import sys",
        "from typing import List, Dict",
        "",
        "CONFIG = {'debug': True, 'version': '1.0'}",
        "MAX_RETRIES = 3",
        "",
        "",
    ]
    # 一个中等函数（< 50 行）
    lines.append("def calculate_sum(a: int, b: int) -> int:")
    lines.append('    """Calculate the sum of two numbers."""')
    lines.append("    result = a + b")
    lines.append("    if result > 100:")
    lines.append('        print(f"Large sum: {result}")')
    lines.append("    else:")
    lines.append('        print(f"Small sum: {result}")')
    lines.append("    return result")
    lines.append("")
    lines.append("")

    # 一个大函数（> 200 行），包含多个逻辑块
    lines.append("def process_data(items: List[int]) -> Dict[str, int]:")
    lines.append('    """Process a large list of items with multiple branches."""')
    lines.append("    stats = {'total': 0, 'even': 0, 'odd': 0, 'positive': 0, 'negative': 0}")
    lines.append("    results = []")
    lines.append("    for item in items:")
    lines.append("        if item > 0:")
    lines.append("            stats['positive'] += 1")
    lines.append("            results.append(item * 2)")
    lines.append("        elif item < 0:")
    lines.append("            stats['negative'] += 1")
    lines.append("            results.append(abs(item))")
    lines.append("        else:")
    lines.append("            stats['positive'] += 0")
    lines.append("    for i in range(10):")
    lines.append("        if i % 2 == 0:")
    lines.append("            stats['even'] += i")
    lines.append("        else:")
    lines.append("            stats['odd'] += i")
    lines.append("    while stats['total'] < 50:")
    lines.append("        stats['total'] += 1")
    lines.append("    try:")
    lines.append("        risky = 1 / 0")
    lines.append("    except ZeroDivisionError:")
    lines.append("        risky = 0")
    lines.append("    with open('/dev/null', 'w') as f:")
    lines.append("        f.write(str(stats))")
    # 填充到 > 200 行
    for i in range(250):
        lines.append(f"    stats['total'] += {i % 5}")
    lines.append("    return stats")
    lines.append("")
    lines.append("")

    # 一个类
    lines.append("class DataProcessor:")
    lines.append('    """Process data with various transformation methods."""')
    lines.append("")
    lines.append("    def __init__(self, data):")
    lines.append("        self.data = data")
    lines.append("")
    lines.append("    def process(self):")
    lines.append('        """Process the data by doubling each element."""')
    lines.append("        return [d * 2 for d in self.data]")
    lines.append("")
    lines.append("    def filter_even(self):")
    lines.append('        """Filter and return only even numbers."""')
    lines.append("        return [d for d in self.data if d % 2 == 0]")
    lines.append("")

    return "\n".join(lines)


def main():
    # 1. 创建测试文件
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(build_large_python_file())
        test_file = f.name

    try:
        print(f"测试文件: {test_file}")
        with open(test_file, "r", encoding="utf-8") as f:
            total_lines = len(f.readlines())
        print(f"文件总行数: {total_lines}")
        print("=" * 60)

        # 2. AST 解析
        table = parse_file(test_file)
        print(f"AST 解析出 {len(table.symbols)} 个符号:")
        for s in sorted(table.symbols, key=lambda x: x.start_line):
            print(f"  {s.symbol_type.value.capitalize()}: {s.name} "
                  f"(Lines {s.start_line}-{s.end_line}, {s.end_line - s.start_line + 1} 行)")
        print("=" * 60)

        # 3. 语义切片
        chunks = chunk_file(test_file, table)
        print(f"\n切片产出 {len(chunks)} 个 Chunk:")
        for i, c in enumerate(chunks):
            print(f"  [{i}] type={c.chunk_type.value:10s} symbol={c.symbol_name:20s} "
                  f"lines={c.start_line}-{c.end_line} ({c.line_count} 行)")
        print("=" * 60)

        # 验收 1：Chunk 数量在 5-8 之间
        chunk_count = len(chunks)
        if 5 <= chunk_count <= 8:
            print(f"✅ 验收通过：Chunk 数量 {chunk_count} 在 5-8 范围内")
        else:
            print(f"❌ 验收失败：Chunk 数量 {chunk_count} 不在 5-8 范围内")
            sys.exit(1)

        # 验收：包含 import 块
        has_import = any(c.chunk_type == ChunkType.IMPORT for c in chunks)
        if has_import:
            print("✅ 验收通过：包含 import 类型 Chunk")
        else:
            print("❌ 验收失败：未找到 import 类型 Chunk")
            sys.exit(1)

        # 验收：包含 function 和 class 块
        has_function = any(c.chunk_type == ChunkType.FUNCTION for c in chunks)
        has_class = any(c.chunk_type == ChunkType.CLASS for c in chunks)
        if has_function and has_class:
            print("✅ 验收通过：包含 function 和 class 类型 Chunk")
        else:
            print(f"❌ 验收失败：function={has_function}, class={has_class}")
            sys.exit(1)

        # 4. 向量化
        print("\n" + "=" * 60)
        print("开始向量化...")
        client = EmbeddingClient.from_settings()
        print(f"使用后端: {client.active_backend}")
        print(f"向量维度: {client.dimension}")
        print(f"模型版本: {client.version}")

        import time
        t0 = time.time()
        client.embed_chunks(chunks)
        elapsed = time.time() - t0
        print(f"向量化耗时: {elapsed:.2f}s")
        print("=" * 60)

        # 验收 2：每个 Chunk 的向量维度正确
        expected_dim = client.dimension
        dim_ok = all(
            c.embedding is not None and len(c.embedding) == expected_dim
            for c in chunks
        )
        if dim_ok:
            print(f"✅ 验收通过：所有 Chunk 的向量维度均为 {expected_dim}")
        else:
            print(f"❌ 验收失败：部分 Chunk 向量维度不是 {expected_dim}")
            for c in chunks:
                dim = len(c.embedding) if c.embedding else 0
                print(f"    {c.symbol_name}: dim={dim}")
            sys.exit(1)

        # 验收 3：向量非全零
        nonzero_ok = all(
            c.embedding is not None and any(v != 0.0 for v in c.embedding)
            for c in chunks
        )
        if nonzero_ok:
            print("✅ 验收通过：所有向量均非全零")
        else:
            print("❌ 验收失败：存在全零向量")
            sys.exit(1)

        # 验收 4：embedding_version 已填充
        version_ok = all(c.embedding_version for c in chunks)
        if version_ok:
            print("✅ 验收通过：所有 Chunk 均填充了 embedding_version")
        else:
            print("❌ 验收失败：存在未填充 embedding_version 的 Chunk")
            sys.exit(1)

        # 打印前两个向量的前 5 维作为示例
        print("\n向量示例（前 2 个 Chunk 的前 5 维）:")
        for c in chunks[:2]:
            print(f"  {c.symbol_name}: {c.embedding[:5]}")

        print("\n" + "=" * 60)
        print("🎉 所有验收通过！")

    finally:
        os.unlink(test_file)


if __name__ == "__main__":
    main()
