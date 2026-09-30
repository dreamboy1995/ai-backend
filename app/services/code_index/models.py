"""
代码索引数据模型（S4 第 31-32 天）

定义符号（Symbol）与符号表（SymbolTable）的数据结构，
供 AST 解析器输出与后续切片/向量化复用。
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Optional


class SymbolType(str, Enum):
    """符号类型"""
    CLASS = "class"
    FUNCTION = "function"
    VARIABLE = "variable"


@dataclass
class Symbol:
    """
    单个代码符号

    Attributes:
        name:        符号名称（类名 / 函数名 / 变量名）
        symbol_type: 符号类型（class / function / variable）
        file_path:   符号所在文件路径（相对路径）
        start_line:  起始行号（1-based）
        end_line:    结束行号（1-based，闭区间）
        content:     符号对应的源码文本（可选，用于后续切片向量化）
    """
    name: str
    symbol_type: SymbolType
    file_path: str
    start_line: int
    end_line: int
    content: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["symbol_type"] = self.symbol_type.value
        return d


@dataclass
class SymbolTable:
    """
    单个文件解析出的符号表

    Attributes:
        file_path: 文件路径
        language:  识别出的语言（python / javascript / typescript / java / go / unknown）
        symbols:   符号列表
    """
    file_path: str
    language: str
    symbols: List[Symbol] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "language": self.language,
            "symbols": [s.to_dict() for s in self.symbols],
        }

    @property
    def functions(self) -> List[Symbol]:
        return [s for s in self.symbols if s.symbol_type == SymbolType.FUNCTION]

    @property
    def classes(self) -> List[Symbol]:
        return [s for s in self.symbols if s.symbol_type == SymbolType.CLASS]

    @property
    def variables(self) -> List[Symbol]:
        return [s for s in self.symbols if s.symbol_type == SymbolType.VARIABLE]
