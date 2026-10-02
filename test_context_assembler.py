"""
ContextAssembler 验证脚本（S5 第 47-48 天：智能上下文组装器 & Token 预算控制）

验证：
1. 小函数（<=100 行）保留完整代码，不压缩
2. 大函数（>100 行）智能压缩：签名 + 注释 + 前 10 行 + 省略标记 + 后 10 行
3. 非函数/类 Chunk（import/block）保留完整，不压缩
4. 位置权重打分：光标所在文件的 Chunk 权重 +30%，排在前面
5. 动态 Token 预算：根据模型上下文窗口计算，超预算时低分优先丢弃
6. references 正确生成（file/lines/score/symbol）
7. 组装前后 Token 计数统计正确
"""

from app.services.code_index.context_assembler import (
    ContextAssembler,
    _is_signature_line,
    _is_comment_line,
    _detect_language,
)


def _make_chunk(**overrides) -> dict:
    """构造一个模拟 hybrid_search 输出的 chunk"""
    base = {
        "id": "chunk-1",
        "file_path": "src/main.py",
        "symbol_name": "foo",
        "chunk_type": "function",
        "content": "def foo():\n    return 1\n",
        "start_line": 1,
        "end_line": 2,
        "score": 0.8,
    }
    base.update(overrides)
    return base


def _make_large_function(lines: int = 150, name: str = "big_func") -> str:
    """生成一个指定行数的 Python 函数"""
    code_lines = [f"def {name}():"]
    code_lines.append('    """这是一个大函数的文档字符串。"""')
    for i in range(lines - 2):
        code_lines.append(f"    x_{i} = {i}")
    return "\n".join(code_lines)


# ============================================================
# 1. 小函数保留完整
# ============================================================

def test_small_function_kept_complete():
    """行数 <= 100 的函数 Chunk 不压缩，保留完整代码"""
    assembler = ContextAssembler()
    content = "def small():\n    return 42\n"
    chunk = _make_chunk(content=content, chunk_type="function")
    result = assembler.assemble([chunk])

    assert len(result.contexts) == 1
    assert result.contexts[0].content_snippet == content
    assert result.stats.compressed_chunks == 0
    assert result.stats.kept_chunks == 1
    print("✅ 小函数保留完整验证通过")


# ============================================================
# 2. 大函数智能压缩
# ============================================================

def test_large_function_compressed():
    """行数 > 100 的函数 Chunk 被智能压缩：签名 + 注释 + 头10行 + 省略 + 尾10行"""
    assembler = ContextAssembler(compress_lines=100, head_lines=10, tail_lines=10)
    big_content = _make_large_function(lines=150, name="big_func")
    chunk = _make_chunk(
        content=big_content,
        chunk_type="function",
        file_path="src/big.py",
        symbol_name="big_func",
        start_line=1,
        end_line=150,
    )
    result = assembler.assemble([chunk])

    compressed = result.contexts[0].content_snippet
    # 必须保留签名行
    assert "def big_func():" in compressed
    # 必须保留 docstring 注释
    assert '"""这是一个大函数的文档字符串。"""' in compressed
    # 必须包含省略标记
    assert "省略" in compressed
    # 统计：被压缩了 1 个
    assert result.stats.compressed_chunks == 1
    # 压缩后行数应远小于原始 150 行
    compressed_lines = len(compressed.split("\n"))
    assert compressed_lines < 150
    # 压缩后应包含前 10 行中的某些内容和后 10 行中的某些内容
    # 前 10 行 body（i=0~9）
    assert "x_0" in compressed
    assert "x_9" in compressed
    # 后 10 行 body（i=138~147，因为总 150-2=148 body 行，后 10 行是 i=138~147）
    assert "x_147" in compressed
    # 中间行应被省略
    assert "x_50" not in compressed
    print(f"✅ 大函数智能压缩验证通过（150 行 → {compressed_lines} 行）")


def test_compression_omission_count_correct():
    """省略标记中的行数应等于实际省略的行数"""
    assembler = ContextAssembler(compress_lines=100, head_lines=10, tail_lines=10)
    total = 200
    big_content = _make_large_function(lines=total, name="huge")
    chunk = _make_chunk(content=big_content, chunk_type="function", file_path="a.py")
    result = assembler.assemble([chunk])

    compressed = result.contexts[0].content_snippet
    # 签名 1 行 + docstring 1 行 = 2 行，body = 198 行
    # 保留 head 10 + tail 10 = 20 行，省略 198 - 20 = 178 行
    assert "省略 178 行" in compressed, f"省略行数不正确: {compressed}"
    print("✅ 省略行数计算正确验证通过")


# ============================================================
# 3. 非函数/类 Chunk 不压缩
# ============================================================

def test_non_function_chunk_not_compressed():
    """import/block 类型的 Chunk 即使超过 100 行也不压缩"""
    assembler = ContextAssembler(compress_lines=100)
    big_import = "\n".join([f"import module_{i}" for i in range(150)])
    chunk = _make_chunk(
        content=big_import,
        chunk_type="import",
        file_path="src/imports.py",
    )
    result = assembler.assemble([chunk])

    assert result.contexts[0].content_snippet == big_import
    assert result.stats.compressed_chunks == 0
    print("✅ 非函数/类 Chunk 不压缩验证通过")


# ============================================================
# 4. 位置权重打分
# ============================================================

def test_cursor_file_boosted():
    """光标所在文件的 Chunk 权重 +30%，应排在其他文件前面"""
    assembler = ContextAssembler(cursor_boost=0.3)
    # 两个 Chunk 分数相同，但一个在光标文件，一个不在
    cursor_chunk = _make_chunk(
        file_path="src/cursor.py", score=0.5, symbol_name="cursor_func",
        chunk_type="function",
    )
    other_chunk = _make_chunk(
        file_path="src/other.py", score=0.5, symbol_name="other_func",
        chunk_type="function", id="chunk-2",
    )
    result = assembler.assemble(
        [other_chunk, cursor_chunk],  # 故意把 other 放前面
        cursor_file="src/cursor.py",
    )

    # 光标文件的 Chunk 应排在前面
    assert result.contexts[0].file_path == "src/cursor.py"
    assert result.contexts[1].file_path == "src/other.py"
    # references 中的 score 应反映加权后的分数
    assert result.references[0].score > result.references[1].score
    print("✅ 光标文件位置权重加成验证通过")


def test_cursor_boost_magnitude():
    """光标文件 Chunk 的分数应精确等于原分数 * (1 + 0.3)"""
    assembler = ContextAssembler(cursor_boost=0.3)
    chunk = _make_chunk(file_path="src/cursor.py", score=0.5, chunk_type="function")
    result = assembler.assemble([chunk], cursor_file="src/cursor.py")

    expected = 0.5 * 1.3
    assert abs(result.references[0].score - expected) < 1e-6
    print(f"✅ 位置权重加成幅度验证通过（0.5 → {result.references[0].score:.4f}）")


# ============================================================
# 5. 动态 Token 预算 & 低分丢弃
# ============================================================

def test_dynamic_budget_from_model_window():
    """预算 = min(model_window * fill_ratio, max_budget)"""
    assembler = ContextAssembler(fill_ratio=0.7, max_budget=8000)
    # DeepSeek 64K: 64000 * 0.7 = 44800 > 8000 → 取 8000
    assert assembler.compute_budget(64000) == 8000
    # 小模型 8K: 8000 * 0.7 = 5600
    assert assembler.compute_budget(8000) == 5600
    # 未知模型 → default
    assert assembler.compute_budget(None) == assembler._default_budget
    print("✅ 动态 Token 预算计算验证通过")


def test_over_budget_drops_lowest_score():
    """超预算时从最低分 Chunk 开始丢弃"""
    # 预算设很小，迫使丢弃
    assembler = ContextAssembler(default_budget=100, max_budget=100)
    chunks = [
        _make_chunk(id="c1", file_path="a.py", score=0.9, symbol_name="high",
                    content="x" * 2000, chunk_type="function"),
        _make_chunk(id="c2", file_path="b.py", score=0.5, symbol_name="mid",
                    content="y" * 2000, chunk_type="function"),
        _make_chunk(id="c3", file_path="c.py", score=0.1, symbol_name="low",
                    content="z" * 2000, chunk_type="function"),
    ]
    result = assembler.assemble(chunks, model_context_window=100)

    # 预算很小，应只保留最高分的 1 个
    assert result.stats.dropped_chunks >= 2
    assert result.stats.kept_chunks == 1
    # 保留的应是最高分的 a.py
    assert result.contexts[0].file_path == "a.py"
    print(
        f"✅ 超预算低分丢弃验证通过 "
        f"(原始 3 个 → 保留 {result.stats.kept_chunks} 个, 丢弃 {result.stats.dropped_chunks} 个)"
    )


def test_compressed_tokens_within_budget():
    """压缩后的 Token 数应不超过预算"""
    assembler = ContextAssembler(default_budget=5000, max_budget=5000)
    big_content = _make_large_function(lines=300, name="huge_func")
    chunks = [
        _make_chunk(id="c1", file_path="a.py", score=0.9, content=big_content,
                    chunk_type="function"),
        _make_chunk(id="c2", file_path="b.py", score=0.8, content=big_content,
                    chunk_type="function"),
    ]
    result = assembler.assemble(chunks, model_context_window=5000)

    # 压缩后的总 Token 应 <= 预算
    assert result.stats.compressed_tokens <= result.stats.token_budget + 50, (
        f"压缩后 Token {result.stats.compressed_tokens} 超过预算 {result.stats.token_budget}"
    )
    print(
        f"✅ 压缩后 Token 在预算内验证通过 "
        f"({result.stats.original_tokens} → {result.stats.compressed_tokens} / 预算 {result.stats.token_budget})"
    )


# ============================================================
# 6. references 生成
# ============================================================

def test_references_generated_correctly():
    """references 应正确包含 file/lines/score/symbol"""
    assembler = ContextAssembler()
    chunk = _make_chunk(
        file_path="src/main.py",
        symbol_name="my_func",
        start_line=10,
        end_line=50,
        score=0.75,
        chunk_type="function",
    )
    result = assembler.assemble([chunk])

    assert len(result.references) == 1
    ref = result.references[0]
    assert ref.file == "src/main.py"
    assert ref.lines == "10-50"
    assert abs(ref.score - 0.75) < 1e-6
    assert ref.symbol == "my_func"
    print("✅ references 生成正确验证通过")


def test_references_single_line():
    """单行 Chunk 的 lines 应为单个数字"""
    assembler = ContextAssembler()
    chunk = _make_chunk(start_line=42, end_line=42, file_path="a.py", chunk_type="function")
    result = assembler.assemble([chunk])
    assert result.references[0].lines == "42"
    print("✅ 单行引用 lines 格式验证通过")


# ============================================================
# 7. 统计信息
# ============================================================

def test_stats_counts():
    """stats 中 original/kept/dropped/compressed 计数应正确"""
    assembler = ContextAssembler(default_budget=5000, max_budget=5000)
    big = _make_large_function(lines=200, name="big")
    small = "def small():\n    return 1\n"
    chunks = [
        _make_chunk(id="c1", file_path="a.py", score=0.9, content=big, chunk_type="function"),
        _make_chunk(id="c2", file_path="b.py", score=0.8, content=small, chunk_type="function"),
    ]
    result = assembler.assemble(chunks, model_context_window=5000)

    assert result.stats.original_chunks == 2
    assert result.stats.kept_chunks == 2
    assert result.stats.dropped_chunks == 0
    assert result.stats.compressed_chunks == 1  # 只有大函数被压缩
    assert result.stats.original_tokens > result.stats.compressed_tokens
    print(
        f"✅ 统计信息验证通过 "
        f"(原始 {result.stats.original_chunks}, 保留 {result.stats.kept_chunks}, "
        f"压缩 {result.stats.compressed_chunks}, 丢弃 {result.stats.dropped_chunks})"
    )


def test_empty_chunks():
    """空 chunks 列表应返回空结果"""
    assembler = ContextAssembler()
    result = assembler.assemble([])
    assert result.contexts == []
    assert result.references == []
    assert result.stats.original_chunks == 0
    print("✅ 空 chunks 处理验证通过")


# ============================================================
# 8. 辅助函数单元测试
# ============================================================

def test_is_signature_line():
    """签名行识别应覆盖主流语言"""
    assert _is_signature_line("def foo():", "python")
    assert _is_signature_line("async def foo():", "python")
    assert _is_signature_line("class Bar:", "python")
    assert _is_signature_line("function foo() {", "javascript")
    assert _is_signature_line("class Bar {", "javascript")
    assert _is_signature_line("public void foo() {", "java")
    assert _is_signature_line("func foo() {", "go")
    assert _is_signature_line("fn foo() {", "rust")
    assert _is_signature_line("def foo", "ruby")
    # 非签名行
    assert not _is_signature_line("    x = 1", "python")
    assert not _is_signature_line("# comment", "python")
    print("✅ 签名行识别验证通过")


def test_is_comment_line():
    """注释行识别"""
    assert _is_comment_line("# comment", "python")
    assert _is_comment_line("// comment", "javascript")
    assert _is_comment_line('"""docstring"""', "python")
    assert _is_comment_line("/* block */", "c")
    assert not _is_comment_line("x = 1", "python")
    assert not _is_comment_line("", "python")
    print("✅ 注释行识别验证通过")


def test_detect_language():
    """语言识别"""
    assert _detect_language("a.py") == "python"
    assert _detect_language("a.js") == "javascript"
    assert _detect_language("a.ts") == "typescript"
    assert _detect_language("a.go") == "go"
    assert _detect_language("a.rs") == "rust"
    assert _detect_language("a.unknown") == ""
    print("✅ 语言识别验证通过")


# ============================================================
# 9. 路径跨平台归一化
# ============================================================

def test_cursor_file_path_normalization():
    """Windows 反斜杠路径应被归一化后正确匹配光标文件"""
    assembler = ContextAssembler(cursor_boost=0.3)
    chunk = _make_chunk(
        file_path="src\\cursor.py",  # Windows 风格
        score=0.5,
        chunk_type="function",
    )
    # 光标文件用正斜杠
    result = assembler.assemble([chunk], cursor_file="src/cursor.py")
    # 应识别为光标文件，分数加成
    assert abs(result.references[0].score - 0.5 * 1.3) < 1e-6
    print("✅ 路径跨平台归一化验证通过")


if __name__ == "__main__":
    test_small_function_kept_complete()
    test_large_function_compressed()
    test_compression_omission_count_correct()
    test_non_function_chunk_not_compressed()
    test_cursor_file_boosted()
    test_cursor_boost_magnitude()
    test_dynamic_budget_from_model_window()
    test_over_budget_drops_lowest_score()
    test_compressed_tokens_within_budget()
    test_references_generated_correctly()
    test_references_single_line()
    test_stats_counts()
    test_empty_chunks()
    test_is_signature_line()
    test_is_comment_line()
    test_detect_language()
    test_cursor_file_path_normalization()
    print("\n" + "=" * 50)
    print("全部 ContextAssembler 测试通过！")
    print("=" * 50)
