"""
代码索引数据模型（S4 第 31-34 天）

定义符号（Symbol）、符号表（SymbolTable）与代码切片（CodeChunk）的数据结构，
供 AST 解析器输出、语义切片与向量化复用。
"""

import hashlib
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Optional


class SymbolType(str, Enum):
    """符号类型"""
    CLASS = "class"
    FUNCTION = "function"
    VARIABLE = "variable"


class ChunkType(str, Enum):
    """切片类型（S4 第 33-34 天）"""
    IMPORT = "import"      # 文件头：import 语句 + 全局变量
    FUNCTION = "function"  # 函数/方法切片
    CLASS = "class"        # 类切片
    BLOCK = "block"        # 大函数拆分出的逻辑块


def _compute_chunk_id(file_path: str, symbol_name: str, start_line: int, end_line: int) -> str:
    """
    计算 Chunk 的唯一 ID：file_path + symbol_name + 行号范围 的哈希。

    用于向量数据库主键与增量更新时的精确匹配。
    """
    raw = f"{file_path}:{symbol_name}:{start_line}-{end_line}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


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


@dataclass
class CodeChunk:
    """
    语义代码切片（S4 第 33-34 天）

    是向量化与向量数据库存储的基本单元。每个 Chunk 对应一段有语义边界的代码
    （一个函数、一个类、文件头的 import 块，或大函数拆分出的逻辑块）。

    Attributes:
        id:               唯一 ID（file_path + symbol_name + 行号的哈希）
        file_path:        所在文件路径（相对路径）
        symbol_name:      所属符号名（函数名/类名；import 块为 "__header__"）
        chunk_type:       切片类型（import / function / class / block）
        content:          切片文本（含注释与 docstring；函数签名作为标题拼在前面）
        start_line:       起始行号（1-based）
        end_line:         结束行号（1-based，闭区间）
        embedding:        向量（384 维，由 EmbeddingClient 填充；未填充时为 None）
        embedding_version: 向量模型版本标识，用于维度迁移（S4 风险预警：换模型时重建）
    """
    file_path: str
    symbol_name: str
    chunk_type: ChunkType
    content: str
    start_line: int
    end_line: int
    embedding: Optional[List[float]] = None
    embedding_version: str = ""
    id: str = field(default="")

    def __post_init__(self):
        if not self.id:
            self.id = _compute_chunk_id(
                self.file_path, self.symbol_name, self.start_line, self.end_line
            )

    @property
    def line_count(self) -> int:
        return self.end_line - self.start_line + 1

    def to_dict(self) -> dict:
        d = asdict(self)
        d["chunk_type"] = self.chunk_type.value
        return d

    @staticmethod
    def from_dict(data: dict) -> "CodeChunk":
        return CodeChunk(
            id=data.get("id", ""),
            file_path=data["file_path"],
            symbol_name=data["symbol_name"],
            chunk_type=ChunkType(data["chunk_type"]),
            content=data["content"],
            start_line=data["start_line"],
            end_line=data["end_line"],
            embedding=data.get("embedding"),
            embedding_version=data.get("embedding_version", ""),
        )
