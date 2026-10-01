"""
索引服务（S4 第 31-36 天）

管理代码库索引的状态与流程。

第 31-32 天：tree-sitter 解析基建，符号表暂存于内存。
第 33-34 天：语义切片 + 向量化（EmbeddingClient）。
第 35-36 天：接入 LanceDB 向量数据库，索引时将切片写入向量库，
            支持按向量检索与按文件删除（增量更新）。
"""

import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from .ast_parser import parse_file
from .code_chunker import chunk_file
from .embedding_client import get_embedding_client
from .models import SymbolTable
from .parser_factory import is_supported
from .vector_store import get_vector_store

logger = logging.getLogger(__name__)

# 索引时跳过的目录
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", ".idea", ".vscode"}
# 单文件大小上限（1MB）
MAX_FILE_SIZE = 1024 * 1024


class IndexService:
    """
    索引服务单例，管理全量/增量索引状态。

    状态字段：
    - status: idle / indexing / done / error
    - total: 待处理文件总数
    - processed: 已处理文件数
    - total_symbols: 已索引的符号总数
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.status: str = "idle"
        self.total: int = 0
        self.processed: int = 0
        self.total_symbols: int = 0
        self.message: Optional[str] = None
        self.job_id: Optional[str] = None
        # file_path(相对路径) -> SymbolTable 的内存索引
        self._index: Dict[str, SymbolTable] = {}

    def get_status(self) -> dict:
        with self._lock:
            percentage = (self.processed / self.total * 100) if self.total > 0 else 0.0
            return {
                "status": self.status,
                "total": self.total,
                "processed": self.processed,
                "percentage": round(percentage, 1),
                "total_symbols": self.total_symbols,
                "message": self.message,
            }

    def _scan_files(self, workspace_root: str) -> List[str]:
        """扫描工作区，返回待索引的文件相对路径列表"""
        files: List[str] = []
        root = Path(workspace_root)
        if not root.exists() or not root.is_dir():
            raise ValueError(f"工作区路径不存在: {workspace_root}")

        for dirpath, dirnames, filenames in os.walk(root):
            # 原地修改 dirnames 以跳过指定目录
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fname in filenames:
                full_path = os.path.join(dirpath, fname)
                # 跳过过大文件
                try:
                    if os.path.getsize(full_path) > MAX_FILE_SIZE:
                        continue
                except OSError:
                    continue
                # 仅处理支持的语言
                if not is_supported(full_path):
                    continue
                rel_path = os.path.relpath(full_path, workspace_root)
                files.append(rel_path)

        return files

    def start_index(self, workspace_root: str, force_rebuild: bool = False) -> dict:
        """
        启动全量索引（同步执行，便于第 31-36 天验收）。

        第 37-38 天将改为后台线程 + 进度写入 Redis。
        """
        with self._lock:
            if self.status == "indexing":
                return {"job_id": self.job_id, "total_files": self.total, "message": "索引正在进行中"}

            self.job_id = str(uuid.uuid4())
            self.status = "indexing"
            self.processed = 0
            self.total_symbols = 0
            self.message = None
            if force_rebuild:
                self._index.clear()

        # S4 第 35-36 天：初始化向量库（按 Embedding 维度建表；force_rebuild 时清空）
        vector_store = get_vector_store()
        try:
            embed_client = get_embedding_client()
            vector_store.create_table(vector_dim=embed_client.dimension)
            if force_rebuild:
                vector_store.clear_table()
        except Exception as e:
            logger.warning(f"[IndexService] 向量库初始化失败，索引将跳过向量化: {e}")

        try:
            files = self._scan_files(workspace_root)
            with self._lock:
                self.total = len(files)

            for rel_path in files:
                full_path = os.path.join(workspace_root, rel_path)
                try:
                    table = parse_file(full_path)
                    # 统一使用相对路径存储
                    table.file_path = rel_path
                    with self._lock:
                        self._index[rel_path] = table
                        self.total_symbols += len(table.symbols)
                    # S4 第 35-36 天：切片 → 向量化 → 写入向量库
                    self._index_file_to_vector_store(full_path, rel_path, table)
                except Exception as e:
                    logger.warning(f"[IndexService] 索引文件失败 {rel_path}: {e}")
                finally:
                    with self._lock:
                        self.processed += 1

            with self._lock:
                self.status = "done"
                self.message = f"索引完成，共 {self.total_symbols} 个符号"

            return {"job_id": self.job_id, "total_files": self.total}

        except Exception as e:
            with self._lock:
                self.status = "error"
                self.message = str(e)
            raise

    def _index_file_to_vector_store(
        self, full_path: str, rel_path: str, table: SymbolTable
    ) -> None:
        """
        对单个文件执行：切片 → 向量化 → 删旧向量 → 插新向量。

        任何环节失败仅记录日志，不中断整体索引流程（保证索引不中断，S4 风险应对）。
        """
        try:
            chunks = chunk_file(full_path, table)
            if not chunks:
                return
            embed_client = get_embedding_client()
            embed_client.embed_chunks(chunks)
            store = get_vector_store()
            # 先删旧向量（该文件可能此前已索引过），再写入新向量
            store.delete_by_file(rel_path)
            store.insert_chunks(chunks)
        except Exception as e:
            logger.warning(f"[IndexService] 文件向量化入库失败 {rel_path}: {e}")

    def update_file(self, workspace_root: str, file_path: str, action: str) -> dict:
        """
        增量更新：处理单个文件的变更。

        action: modified / deleted / renamed
        """
        if action == "deleted":
            with self._lock:
                removed = self._index.pop(file_path, None)
                count = len(removed.symbols) if removed else 0
                self.total_symbols = max(0, self.total_symbols - count)
            # S4 第 35-36 天：同步删除向量库中的旧向量
            try:
                get_vector_store().delete_by_file(file_path)
            except Exception as e:
                logger.warning(f"[IndexService] 删除向量库记录失败 {file_path}: {e}")
            return {"success": True, "message": f"已删除 {file_path} 的索引", "symbols_count": 0}

        # modified / renamed: 重新解析
        full_path = os.path.join(workspace_root, file_path)
        if not os.path.exists(full_path):
            return {"success": False, "message": f"文件不存在: {file_path}", "symbols_count": 0}

        table = parse_file(full_path)
        table.file_path = file_path
        with self._lock:
            old = self._index.get(file_path)
            old_count = len(old.symbols) if old else 0
            self._index[file_path] = table
            self.total_symbols = self.total_symbols - old_count + len(table.symbols)

        # S4 第 35-36 天：重新切片向量化并写入向量库
        self._index_file_to_vector_store(full_path, file_path, table)

        return {
            "success": True,
            "message": f"已更新 {file_path} 的索引",
            "symbols_count": len(table.symbols),
        }

    def get_symbols(self, file_path: Optional[str] = None) -> List[dict]:
        """获取已索引的符号（调试/搜索用）"""
        with self._lock:
            if file_path:
                table = self._index.get(file_path)
                return [s.to_dict() for s in table.symbols] if table else []
            result = []
            for table in self._index.values():
                result.extend(s.to_dict() for s in table.symbols)
            return result


# 单例
_index_service: Optional[IndexService] = None


def get_index_service() -> IndexService:
    global _index_service
    if _index_service is None:
        _index_service = IndexService()
    return _index_service
