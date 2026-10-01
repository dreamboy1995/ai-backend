"""
向量数据库封装（S4 第 35-36 天）

基于 LanceDB（纯文件、免安装、嵌入式）的代码切片向量存储，
提供表创建、批量写入、向量检索、按文件删除等 CRUD 能力。

核心方法：
  - create_table(vector_dim) : 创建表（按向量维度动态生成 Schema；维度不匹配时自动重建）
  - insert_chunks(chunks)    : 批量写入已向量化的 CodeChunk
  - search(query_vector, top_k) : 向量相似度检索，返回相似 Chunk 列表（含距离分数）
  - delete_by_file(file_path): 删除指定文件的所有 Chunk（文件删除/重命名时清理旧向量）

维护：
  - compact(older_than, delete_unverified): 合并分片并清理旧版本快照，回收磁盘空间

风险应对（S4 关键技术预研）：
  - 向量维度不兼容：LanceDB 不支持维度自动迁移。
    切换 Embedding 模型时，create_table 会检测维度变化，drop 旧表后重建，
    并通过 embedding_version 字段标识数据来源版本。
"""

import logging
import os
from datetime import timedelta
from typing import Any, Dict, List, Optional

import lancedb
from lancedb.pydantic import LanceModel, Vector

from .models import CodeChunk

logger = logging.getLogger(__name__)


def _make_chunk_schema(vector_dim: int):
    """
    根据向量维度动态生成 LanceDB Schema（Pydantic 风格）。

    LanceDB 的向量列必须在创建表时指定固定维度，因此不能在类定义时写死。
    通过工厂函数在运行时构造，保证维度与当前 Embedding 后端一致。

    Args:
        vector_dim: 向量维度（如 384 / 1536）

    Returns:
        LanceModel 子类，可直接传给 db.create_table(schema=...)
    """

    class CodeChunkSchema(LanceModel):
        """
        代码切片向量表 Schema。

        字段与 CodeChunk 数据模型一一对应，embedding 为固定维度向量列。
        """
        id: str                       # 文件路径 + 符号名 + 行号的哈希
        file_path: str                # 相对路径
        symbol_name: str              # 符号名（函数名/类名；import 块为 "__header__"）
        chunk_type: str               # function / class / import / block
        content: str                  # 切片文本
        start_line: int               # 起始行号（1-based）
        end_line: int                 # 结束行号（1-based，闭区间）
        embedding_version: str        # 向量模型版本标识，用于维度迁移
        embedding: Vector(vector_dim) # 向量列（固定维度）

    return CodeChunkSchema


class VectorStore:
    """
    LanceDB 向量存储封装。

    使用方式：
        store = VectorStore.from_settings()
        store.create_table(vector_dim=384)
        store.insert_chunks(chunks)            # chunks 已通过 EmbeddingClient 填充 embedding
        results = store.search(query_vec, top_k=5)
        store.delete_by_file("src/main.py")
    """

    def __init__(
        self,
        db_path: str = ".ai_index/lancedb",
        table_name: str = "code_chunks",
    ):
        self.db_path = db_path
        self.table_name = table_name
        self._db: Optional[lancedb.DBConnection] = None
        self._table: Optional[lancedb.table.Table] = None
        self._vector_dim: Optional[int] = None

    @classmethod
    def from_settings(cls) -> "VectorStore":
        from app.config import settings
        return cls(
            db_path=settings.VECTOR_STORE_DB_PATH,
            table_name=settings.VECTOR_STORE_TABLE_NAME,
        )

    # ============================================================
    # 连接管理
    # ============================================================

    def _connect(self) -> lancedb.DBConnection:
        """获取或创建 LanceDB 连接（目录不存在时自动创建）"""
        if self._db is None:
            os.makedirs(self.db_path, exist_ok=True)
            self._db = lancedb.connect(self.db_path)
        return self._db

    @property
    def vector_dim(self) -> Optional[int]:
        """当前表的向量维度（未建表时为 None）"""
        return self._vector_dim

    def is_table_exists(self) -> bool:
        db = self._connect()
        return self.table_name in db.table_names()

    # ============================================================
    # 建表 / 维度迁移
    # ============================================================

    def create_table(self, vector_dim: int) -> lancedb.table.Table:
        """
        创建向量表。

        - 若表不存在：以指定维度创建。
        - 若表已存在且维度匹配：直接打开复用。
        - 若表已存在但维度不匹配（切换 Embedding 模型）：drop 旧表后重建，
          保证向量列维度与新模型一致（LanceDB 不支持维度自动迁移）。

        Args:
            vector_dim: 向量维度

        Returns:
            LanceDB Table 对象
        """
        db = self._connect()
        schema = _make_chunk_schema(vector_dim)

        if self.table_name in db.table_names():
            existing = db.open_table(self.table_name)
            existing_dim = self._read_vector_dim(existing)
            if existing_dim == vector_dim:
                logger.debug(f"[VectorStore] 表已存在且维度匹配（{vector_dim}），复用")
                self._table = existing
                self._vector_dim = vector_dim
                return existing
            # 维度不匹配：重建表
            logger.warning(
                f"[VectorStore] 向量维度变化 {existing_dim} -> {vector_dim}，"
                f"重建表 '{self.table_name}'（旧数据将被清除）"
            )
            db.drop_table(self.table_name)

        self._table = db.create_table(self.table_name, schema=schema)
        self._vector_dim = vector_dim
        logger.info(f"[VectorStore] 创建表 '{self.table_name}'，向量维度={vector_dim}")
        return self._table

    @staticmethod
    def _read_vector_dim(table) -> Optional[int]:
        """从已有表的 schema 中读取向量列维度"""
        try:
            field = table.schema.field("embedding")
            return field.type.list_size
        except Exception:
            return None

    def _get_table(self) -> lancedb.table.Table:
        """获取已打开的表；未建表时抛异常"""
        if self._table is None:
            raise RuntimeError(
                "向量表尚未创建，请先调用 create_table(vector_dim)"
            )
        return self._table

    # ============================================================
    # 写入
    # ============================================================

    def insert_chunks(self, chunks: List[CodeChunk]) -> int:
        """
        批量写入已向量化的 CodeChunk。

        Args:
            chunks: CodeChunk 列表，每个 chunk 的 embedding 字段必须已填充

        Returns:
            成功写入的 Chunk 数量

        Raises:
            ValueError: 若存在未填充 embedding 的 Chunk，或向量维度与表不匹配
        """
        if not chunks:
            return 0

        table = self._get_table()
        expected_dim = self._vector_dim

        records: List[Dict[str, Any]] = []
        for c in chunks:
            if c.embedding is None:
                raise ValueError(
                    f"Chunk '{c.symbol_name}'（{c.file_path}）未填充 embedding，"
                    f"请先调用 EmbeddingClient.embed_chunks()"
                )
            if expected_dim is not None and len(c.embedding) != expected_dim:
                raise ValueError(
                    f"Chunk '{c.symbol_name}' 向量维度 {len(c.embedding)} 与表维度 "
                    f"{expected_dim} 不匹配"
                )
            records.append({
                "id": c.id,
                "file_path": c.file_path,
                "symbol_name": c.symbol_name,
                "chunk_type": c.chunk_type.value if hasattr(c.chunk_type, "value") else c.chunk_type,
                "content": c.content,
                "start_line": c.start_line,
                "end_line": c.end_line,
                "embedding_version": c.embedding_version,
                "embedding": c.embedding,
            })

        try:
            table.add(records)
        except Exception as e:
            logger.error(f"[VectorStore] 批量写入失败: {e}", exc_info=True)
            raise

        logger.debug(f"[VectorStore] 写入 {len(records)} 个 Chunk")
        return len(records)

    # ============================================================
    # 检索
    # ============================================================

    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        向量相似度检索。

        Args:
            query_vector: 查询向量（维度必须与表一致）
            top_k:        返回最相似的前 K 个结果

        Returns:
            结果列表，每项为 dict，包含：
              - id, file_path, symbol_name, chunk_type, content
              - start_line, end_line, embedding_version
              - distance: LanceDB L2 距离（越小越相似）
              - score:    归一化相似度分数（0~1，越大越相似，= 1/(1+distance)）
        """
        table = self._get_table()
        try:
            results = (
                table.search(query_vector)
                .limit(top_k)
                .to_list()
            )
        except Exception as e:
            logger.error(f"[VectorStore] 向量检索失败: {e}", exc_info=True)
            raise

        # LanceDB 检索结果含 _distance 字段；转换为统一格式
        output: List[Dict[str, Any]] = []
        for row in results:
            distance = float(row.get("_distance", 0.0))
            output.append({
                "id": row.get("id"),
                "file_path": row.get("file_path"),
                "symbol_name": row.get("symbol_name"),
                "chunk_type": row.get("chunk_type"),
                "content": row.get("content"),
                "start_line": row.get("start_line"),
                "end_line": row.get("end_line"),
                "embedding_version": row.get("embedding_version"),
                "distance": distance,
                "score": 1.0 / (1.0 + distance),
            })
        return output

    # ============================================================
    # 删除
    # ============================================================

    def delete_by_file(self, file_path: str) -> int:
        """
        删除指定文件的所有 Chunk。

        用于文件删除/重命名时清理旧向量，避免增量更新后残留过时数据。

        Args:
            file_path: 文件相对路径

        Returns:
            删除前该文件的 Chunk 数量（LanceDB delete 不返回删除行数，
            这里通过删除前的 count_rows 差值近似）
        """
        table = self._get_table()
        # 统计删除前数量
        try:
            before = table.count_rows(filter=f"file_path = '{self._escape(file_path)}'")
        except Exception:
            before = -1

        try:
            table.delete(f"file_path = '{self._escape(file_path)}'")
        except Exception as e:
            logger.error(f"[VectorStore] 删除文件 Chunk 失败 {file_path}: {e}", exc_info=True)
            raise

        logger.debug(f"[VectorStore] 已删除文件 {file_path} 的 Chunk（约 {before} 条）")
        return before

    @staticmethod
    def _escape(value: str) -> str:
        """转义 SQL filter 字符串中的单引号"""
        return value.replace("'", "''")

    # ============================================================
    # 维护：合并分片 + 清理旧版本
    # ============================================================

    def compact(
        self,
        older_than: timedelta = timedelta(seconds=0),
        delete_unverified: bool = True,
    ) -> Dict[str, Any]:
        """
        压缩表并清理旧版本快照，回收磁盘空间。

        LanceDB 每次写入都会生成一份 manifest + txn，长期累积会显著占用磁盘
        且降低查询性能。本方法对表执行：
          - 合并 data/ 下小分片为紧凑分片（compaction）
          - 清理 older_than 之前的旧版本 manifest
          - 清理未验证的事务文件（可选）

        注意：清理后无法再回滚到旧版本（time travel 失效）。生产/线上若需
        保留时间旅行能力，请传 older_than=timedelta(days=7) 之类保留窗口。

        Args:
            older_than:        保留该时长内的版本；早于该时长的版本被清理。
                               默认 timedelta(seconds=0) 表示仅保留当前最新版本。
            delete_unverified: 是否清理未验证的事务文件，默认 True。

        Returns:
            报告 dict，含清理前后 {manifests, txns, data_files, size_bytes}。
        """
        table = self._get_table()
        before = self._stats()
        try:
            table.optimize(
                cleanup_older_than=older_than,
                delete_unverified=delete_unverified,
            )
        except Exception as e:
            logger.error(f"[VectorStore] compact 失败: {e}", exc_info=True)
            raise
        after = self._stats()
        logger.info(
            f"[VectorStore] compact 完成："
            f"manifests {before['manifests']} -> {after['manifests']}，"
            f"data_files {before['data_files']} -> {after['data_files']}，"
            f"size {before['size_bytes']}B -> {after['size_bytes']}B"
        )
        return {"before": before, "after": after}

    def _stats(self) -> Dict[str, Any]:
        """统计当前表目录下的 manifest/txn/data 文件数与总占用大小（字节）"""
        base = os.path.join(self.db_path, f"{self.table_name}.lance")
        manifests = txns = data_files = 0
        size_bytes = 0
        if os.path.isdir(base):
            for root, _, files in os.walk(base):
                for f in files:
                    try:
                        size_bytes += os.path.getsize(os.path.join(root, f))
                    except OSError:
                        pass
            vdir = os.path.join(base, "_versions")
            if os.path.isdir(vdir):
                manifests = len([f for f in os.listdir(vdir) if f.endswith(".manifest")])
            tdir = os.path.join(base, "_transactions")
            if os.path.isdir(tdir):
                txns = len([f for f in os.listdir(tdir) if f.endswith(".txn")])
            ddir = os.path.join(base, "data")
            if os.path.isdir(ddir):
                data_files = len(os.listdir(ddir))
        return {
            "manifests": manifests,
            "txns": txns,
            "data_files": data_files,
            "size_bytes": size_bytes,
        }

    # ============================================================
    # 辅助
    # ============================================================

    def count(self) -> int:
        """返回表中 Chunk 总数"""
        table = self._get_table()
        try:
            return table.count_rows()
        except Exception:
            return 0

    def clear_table(self) -> None:
        """清空表中所有数据（保留表结构），用于强制重建索引"""
        if not self.is_table_exists():
            return
        table = self._get_table()
        try:
            # 用一个恒真条件删除全部
            table.delete("id IS NOT NULL")
            logger.info(f"[VectorStore] 已清空表 '{self.table_name}'")
        except Exception as e:
            logger.warning(f"[VectorStore] 清空表失败（将尝试 drop 重建）: {e}")
            db = self._connect()
            db.drop_table(self.table_name)
            if self._vector_dim is not None:
                self.create_table(self._vector_dim)

    def drop_table(self) -> None:
        """删除表（用于维度迁移或完全重置）"""
        db = self._connect()
        if self.table_name in db.table_names():
            db.drop_table(self.table_name)
            logger.info(f"[VectorStore] 已删除表 '{self.table_name}'")
        self._table = None
        self._vector_dim = None


# 单例
_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore.from_settings()
    return _vector_store
