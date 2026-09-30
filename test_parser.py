"""
AST 解析器测试脚本（S4 第 31-32 天验收用）

用法:
    python test_parser.py --file sample.py

验收标准：
    Function: calculate_sum (Lines 5-12)
    Function: main (Lines 15-22)
    Class: DataProcessor (Lines 25-45)
"""

import argparse
import sys
import os

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.code_index.ast_parser import parse_file
from app.services.code_index.models import SymbolType


def format_symbol(symbol) -> str:
    """格式化符号输出，匹配验收标准"""
    type_label = {
        SymbolType.CLASS: "Class",
        SymbolType.FUNCTION: "Function",
        SymbolType.VARIABLE: "Variable",
    }.get(symbol.symbol_type, symbol.symbol_type.value)
    return f"{type_label}: {symbol.name} (Lines {symbol.start_line}-{symbol.end_line})"


def main():
    parser = argparse.ArgumentParser(description="AST 解析器测试")
    parser.add_argument("--file", required=True, help="待解析的源文件路径")
    parser.add_argument("--verbose", action="store_true", help="显示详细信息（含变量）")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"错误: 文件不存在: {args.file}")
        sys.exit(1)

    table = parse_file(args.file)
    print(f"文件: {args.file}")
    print(f"语言: {table.language}")
    print(f"共解析出 {len(table.symbols)} 个符号")
    print("-" * 50)

    # 按行号排序输出
    symbols_sorted = sorted(table.symbols, key=lambda s: s.start_line)

    for symbol in symbols_sorted:
        # 验收标准默认只展示 Function 和 Class
        if symbol.symbol_type in (SymbolType.FUNCTION, SymbolType.CLASS):
            print(format_symbol(symbol))
        elif args.verbose and symbol.symbol_type == SymbolType.VARIABLE:
            print(format_symbol(symbol))

    # 断言验收标准
    expected = [
        ("function", "calculate_sum"),
        ("function", "main"),
        ("class", "DataProcessor"),
    ]
    found = {(s.symbol_type.value, s.name) for s in table.symbols}

    all_pass = True
    for stype, name in expected:
        if (stype, name) not in found:
            print(f"\n❌ 未找到预期符号: {stype} {name}")
            all_pass = False

    if all_pass:
        print("\n✅ 验收通过：所有预期符号均已正确解析")
    else:
        print("\n❌ 验收未通过")
        sys.exit(1)


if __name__ == "__main__":
    main()
