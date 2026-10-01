"""
BM25 关键词检索器测试脚本（S5 第 41-42 天验收用）

用法:
    python test_bm25_index.py

验收标准（来自 Sprint_5.md 第 41-42 天）：
    1. 对"排序算法"提问时，BM25 能优先召回包含"sort"、"quick"、"merge"等关键词的 Chunk
    2. 中文分词正确（jieba），英文驼峰拆分正确（QuickSort -> Quick Sort）
    3. BM25 索引可序列化到磁盘，重启后能加载
    4. 测试文件过滤生效（test_ 前缀文件不被索引）
"""

import os
import sys
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.code_index.models import ChunkType, CodeChunk
from app.services.code_index.vector_store import VectorStore
from app.services.code_index.bm25_index import BM25Index, tokenize


def build_test_chunks() -> list:
    """构造测试 Chunk（含排序/数据/IO 等不同语义，含中文注释）"""
    return [
        CodeChunk(
            file_path="math_utils.py",
            symbol_name="quick_sort",
            chunk_type=ChunkType.FUNCTION,
            content="quick_sort | 快速排序算法实现\n"
                    "def quick_sort(arr):\n"
                    "    if len(arr) <= 1: return arr\n"
                    "    pivot = arr[0]\n"
                    "    return quick_sort([x for x in arr[1:] if x < pivot]) + [pivot] + quick_sort([x for x in arr[1:] if x >= pivot])",
            start_line=1, end_line=5,
        ),
        CodeChunk(
            file_path="math_utils.py",
            symbol_name="merge_sort",
            chunk_type=ChunkType.FUNCTION,
            content="merge_sort | 归并排序算法\n"
                    "def merge_sort(arr):\n"
                    "    if len(arr) <= 1: return arr\n"
                    "    mid = len(arr) // 2\n"
                    "    return merge(merge_sort(arr[:mid]), merge_sort(arr[mid:]))",
            start_line=7, end_line=11,
        ),
        CodeChunk(
            file_path="math_utils.py",
            symbol_name="calculate_sum",
            chunk_type=ChunkType.FUNCTION,
            content="calculate_sum | 计算两数之和\n"
                    "def calculate_sum(a, b):\n"
                    "    return a + b",
            start_line=13, end_line=15,
        ),
        CodeChunk(
            file_path="data_processor.py",
            symbol_name="DataProcessor",
            chunk_type=ChunkType.CLASS,
            content="DataProcessor | 数据处理类\n"
                    "class DataProcessor:\n"
                    "    def process(self, data):\n"
                    "        return [d * 2 for d in data]",
            start_line=1, end_line=4,
        ),
        CodeChunk(
            file_path="data_processor.py",
            symbol_name="save_data",
            chunk_type=ChunkType.FUNCTION,
            content="save_data | 保存数据到磁盘\n"
                    "def save_data(data, path):\n"
                    "    with open(path, 'w') as f:\n"
                    "        f.write(str(data))",
            start_line=6, end_line=9,
        ),
        CodeChunk(
            file_path="io_utils.py",
            symbol_name="read_file",
            chunk_type=ChunkType.FUNCTION,
            content="read_file | 读取文件内容\n"
                    "def read_file(path):\n"
                    "    with open(path) as f:\n"
                    "        return f.read()",
            start_line=1, end_line=4,
        ),
        CodeChunk(
            file_path="test_sort.py",
            symbol_name="test_quick_sort",
            chunk_type=ChunkType.FUNCTION,
            content="test_quick_sort | 测试快速排序\n"
                    "def test_quick_sort():\n"
                    "    assert quick_sort([3, 1, 2]) == [1, 2, 3]",
            start_line=1, end_line=3,
        ),
    ]


def setup_vector_store(tmpdir: str) -> VectorStore:
    """创建临时向量库并写入测试 Chunk"""
    store = VectorStore(db_path=os.path.join(tmpdir, "lancedb"))
    chunks = build_test_chunks()
    for c in chunks:
        c.embedding = [0.1] * 8  # 合成向量，仅用于写入 BM25 不依赖向量
        c.embedding_version = "test-8"
    store.create_table(vector_dim=8)
    store.insert_chunks(chunks)
    return store


def test_tokenizer():
    """测试多语言分词器"""
    print("=== 测试多语言分词器 ===")
    assert "quick" in tokenize("QuickSort")
    assert "sort" in tokenize("QuickSort")
    assert "quick" in tokenize("quick_sort")
    assert "sort" in tokenize("quick_sort")
    assert "http" in tokenize("HTTPServer")
    assert "server" in tokenize("HTTPServer")
    # 中文分词
    tokens = tokenize("快速排序算法")
    assert "快速" in tokens
    assert "排序" in tokens
    assert "算法" in tokens
    # 中英混合
    tokens = tokenize("sortData 函数实现了快速排序")
    assert "sort" in tokens
    assert "data" in tokens
    assert "排序" in tokens
    print("分词器测试通过")


def test_bm25_build_and_search(tmpdir: str):
    """测试 BM25 索引构建与检索"""
    print("\n=== 测试 BM25 构建与检索 ===")
    store = setup_vector_store(tmpdir)

    bm25 = BM25Index(
        index_path=os.path.join(tmpdir, "bm25.pkl"),
        skip_test_files=True,
        vector_store=store,
    )

    # 构建索引
    count = bm25.build()
    print(f"索引构建完成，共 {count} 个 Chunk")
    # 7 个 chunk 中，test_sort.py 应被过滤，所以剩 6 个
    assert count == 6, f"期望 6 个 Chunk（过滤测试文件后），实际 {count}"

    # 测试中文查询：排序
    results = bm25.search("排序", top_k=5)
    print(f"查询 '排序' -> {len(results)} 条结果")
    assert len(results) > 0
    top_file = results[0]["file_path"]
    assert top_file == "math_utils.py", f"排序查询应优先返回 math_utils.py，实际 {top_file}"
    print(f"  Top1: {results[0]['file_path']} (score={results[0]['score']:.4f})")

    # 测试英文驼峰查询
    results = bm25.search("quick sort", top_k=5)
    print(f"查询 'quick sort' -> {len(results)} 条结果")
    assert len(results) > 0
    top_file = results[0]["file_path"]
    assert top_file == "math_utils.py"
    print(f"  Top1: {results[0]['file_path']} (score={results[0]['score']:.4f})")

    # 测试数据处理查询
    results = bm25.search("数据处理", top_k=5)
    print(f"查询 '数据处理' -> {len(results)} 条结果")
    assert len(results) > 0
    top_file = results[0]["file_path"]
    assert top_file == "data_processor.py", f"数据处理查询应返回 data_processor.py，实际 {top_file}"
    print(f"  Top1: {results[0]['file_path']} (score={results[0]['score']:.4f})")

    print("BM25 构建与检索测试通过")


def test_persistence(tmpdir: str):
    """测试 BM25 索引持久化（保存/加载）"""
    print("\n=== 测试 BM25 持久化 ===")
    store = setup_vector_store(tmpdir)

    bm25_path = os.path.join(tmpdir, "bm25.pkl")
    bm25 = BM25Index(index_path=bm25_path, vector_store=store)
    count = bm25.build()
    assert count > 0
    assert os.path.exists(bm25_path), "BM25 索引文件应已保存"

    # 重新创建实例并加载（需传入同一 vector_store 以便后续检索）
    bm25_loaded = BM25Index(index_path=bm25_path, vector_store=store)
    assert bm25_loaded.load(), "应能从磁盘加载 BM25 索引"
    assert bm25_loaded.count() == count, "加载后的 Chunk 数应与构建时一致"

    # 加载后检索应正常工作
    results = bm25_loaded.search("排序", top_k=5)
    assert len(results) > 0
    print(f"加载后查询 '排序' -> {len(results)} 条结果，Top1: {results[0]['file_path']}")
    print("持久化测试通过")


def test_dirty_rebuild(tmpdir: str):
    """测试 mark_dirty 后懒加载重建"""
    print("\n=== 测试脏标记重建 ===")
    store = setup_vector_store(tmpdir)

    bm25 = BM25Index(index_path=os.path.join(tmpdir, "bm25.pkl"), vector_store=store)
    bm25.build()
    assert bm25.count() > 0

    # 标记为脏数据
    bm25.mark_dirty()
    # 此时内存中索引应被清空
    assert bm25._bm25 is None

    # 再次搜索应触发重建
    results = bm25.search("排序", top_k=5)
    assert len(results) > 0
    assert bm25._bm25 is not None, "搜索后应自动重建索引"
    print(f"脏标记重建后查询 '排序' -> {len(results)} 条结果")
    print("脏标记重建测试通过")


def test_test_file_filter():
    """测试测试文件过滤逻辑"""
    print("\n=== 测试测试文件过滤 ===")
    assert BM25Index._is_test_file("test_sort.py")
    assert BM25Index._is_test_file("test_sort.py")
    assert BM25Index._is_test_file("tests/test_utils.py")
    assert BM25Index._is_test_file("app/tests/test_api.py")
    assert BM25Index._is_test_file("sort_test.py")
    assert not BM25Index._is_test_file("math_utils.py")
    assert not BM25Index._is_test_file("app/models/data.py")
    assert not BM25Index._is_test_file("contest_123.py")
    print("测试文件过滤测试通过")


def test_empty_query(tmpdir: str):
    """测试空查询处理"""
    print("\n=== 测试空查询 ===")
    store = setup_vector_store(tmpdir)
    bm25 = BM25Index(index_path=os.path.join(tmpdir, "bm25.pkl"), vector_store=store)
    bm25.build()

    results = bm25.search("", top_k=5)
    assert results == [], "空查询应返回空列表"

    results = bm25.search("   ", top_k=5)
    assert results == [], "空白查询应返回空列表"
    print("空查询测试通过")


def main():
    tmpdir = tempfile.mkdtemp(prefix="bm25_test_")
    try:
        test_tokenizer()
        test_bm25_build_and_search(tmpdir)
        test_persistence(tmpdir)
        test_dirty_rebuild(tmpdir)
        test_test_file_filter()
        test_empty_query(tmpdir)
        print("\n" + "=" * 50)
        print("所有 BM25 测试通过！")
        print("=" * 50)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
