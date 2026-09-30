"""测试降级策略：残缺语法的文件"""
import tempfile
import os
from app.services.code_index.ast_parser import parse_file

# 残缺语法的 Python 文件（缺少冒号、缩进错误）
broken_code = '''
def broken_func(a, b)
    return a +

class BrokenClass
    def method(self):
        return 1

valid_var = 42
'''

with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
    f.write(broken_code)
    path = f.name

try:
    table = parse_file(path)
    print(f"语言: {table.language}")
    print(f"符号数量: {len(table.symbols)}")
    print("即使语法残缺，降级策略仍产出以下符号：")
    for s in sorted(table.symbols, key=lambda x: x.start_line):
        print(f"  {s.symbol_type.value.capitalize()}: {s.name} (Lines {s.start_line}-{s.end_line})")
finally:
    os.unlink(path)
