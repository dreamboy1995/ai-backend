"""
索引服务（S4 第 31-40 天）

管理代码库索引的状态与流程。

第 31-32 天：tree-sitter 解析基建，符号表暂存于内存。
第 33-34 天：语义切片 + 向量化（EmbeddingClient）。
第 35-36 天：接入 LanceDB 向量数据库，索引时将切片写入向量库，
            支持按向量检索与按文件删除（增量更新）。
第 37-38 天：全仓库首次索引 & 增量更新机制
            - 后台线程异步执行全量索引，HTTP 接口立即返回 job_id
            - 每 INDEX_BATCH_SIZE 个文件批量提交向量（切片→向量化→删旧→插新）
            - 索引进度写入 Redis（供插件轮询展示），Redis 不可用时降级为内存
            - "即用即索引"策略：priority_files 优先处理，后台静默索引剩余文件
            - 索引完成后写入 .ai_index 标记文件，供插件检测仓库是否已索引
第 39-40 天：依赖关系图构建
            - 在 AST 解析后，额外提取 import / call 引用关系，构建 Call Graph
            - 图存储为 JSON 文件 + 内存缓存（MVP 阶段不引入 Neo4j）
            - 全量索引结束后整体持久化；增量更新时局部修改后重新持久化
            - 提供 GET /v1/graph/related?file=xxx&depth=2 接口供下游检索使用
"""

import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional

from .ast_parser import parse_file
from .code_chunker import chunk_file
from .dependency_graph import get_dependency_graph
from .embedding_client import get_embedding_client
from .models import CodeChunk, SymbolTable
from .parser_factory import is_supported
from .vector_store import get_vector_store

logger = logging.getLogger(__name__)


# ============================================================
# S5 第 49-50 天：跨平台路径归一化
#
# os.path.relpath 在 Windows 下返回反斜杠路径（src\\main.py），
# 在 Linux/macOS 下返回正斜杠（src/main.py）。为保证 LanceDB / BM25 /
# 依赖图 / 符号表中存储的路径格式一致，统一在源头（索引入口）转为
# POSIX 正斜杠相对路径，下游所有模块无需再做分隔符适配。
# ============================================================

def to_posix_relpath(path: str) -> str:
    """
    将任意相对路径统一转为 POSIX 正斜杠格式。

    使用 pathlib.PurePosixPath 做分隔符转换（不解析 .. / .），
    确保 Windows 反斜杠路径也能正确归一化为 src/main.py。

    Args:
        path: 任意格式的相对路径（可能含反斜杠）

    Returns:
        POSIX 风格的相对路径（正斜杠分隔）
    """
    if not path:
        return path
    return PurePosixPath(path.replace("\\", "/")).as_posix()

# 索引时跳过的目录
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", ".idea", ".vscode"}
# 单文件大小上限（1MB）
MAX_FILE_SIZE = 1024 * 1024


def _get_batch_size() -> int:
    """从配置读取批量提交大小（延迟导入避免循环依赖）"""
    try:
        from app.config import settings
        return settings.INDEX_BATCH_SIZE
    except Exception:
        return 10


def _get_redis_key_prefix() -> str:
    try:
        from app.config import settings
        return settings.INDEX_REDIS_KEY_PREFIX
    except Exception:
        return "index:status"


def _get_redis_ttl() -> int:
    try:
        from app.config import settings
        return settings.INDEX_REDIS_TTL
    except Exception:
        return 3600


def _get_marker_file() -> str:
    try:
        from app.config import settings
        return settings.INDEX_MARKER_FILE
    except Exception:
        return ".ai_index/index_done"


def _get_graph_file() -> Optional[str]:
    """依赖图 JSON 持久化文件路径（相对工作区根目录）"""
    try:
        from app.config import settings
        return settings.DEPENDENCY_GRAPH_FILE
    except Exception:
        return ".ai_index/dependency_graph.json"


def _get_graph_persist_path(workspace_root: str) -> str:
    """依赖图持久化文件的绝对路径"""
    rel = _get_graph_file() or ".ai_index/dependency_graph.json"
    return os.path.join(workspace_root, rel)


class IndexService:
    """
    索引服务单例，管理全量/增量索引状态与流程。

    状态字段：
    - status: idle / indexing / done / error
    - total: 待处理文件总数
    - processed: 已处理文件数
    - total_symbols: 已索引的符号总数
    - workspace_root: 当前索引的工作区根路径（供增量更新复用）
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.status: str = "idle"
        self.total: int = 0
        self.processed: int = 0
        self.total_symbols: int = 0
        self.message: Optional[str] = None
        self.job_id: Optional[str] = None
        self.workspace_root: Optional[str] = None
        # 后台索引线程
        self._index_thread: Optional[threading.Thread] = None
        # file_path(相对路径) -> SymbolTable 的内存索引
        self._index: Dict[str, SymbolTable] = {}

    # ============================================================
    # 状态查询
    # ============================================================

    def get_status(self) -> dict:
        with self._lock:
            # 数据契约：percentage 为 0~1 的小数（与 quota.py 的 percentage 语义一致），
            # 前端展示时自行 * 100 转为百分比。
            percentage = (self.processed / self.total) if self.total > 0 else 0.0
            return {
                "status": self.status,
                "total": self.total,
                "processed": self.processed,
                "percentage": round(percentage, 4),
                "total_symbols": self.total_symbols,
                "message": self.message,
                "workspace_root": self.workspace_root,
            }

    def get_status_from_redis(self) -> Optional[dict]:
        """
        从 Redis 读取最新索引进度。

        优先返回 Redis 中的状态（多进程/多实例场景下更准确），
        Redis 不可用时返回 None，由调用方降级到内存 get_status()。
        """
        job_id = self.job_id
        if not job_id:
            return None
        try:
            from app.services.redis_client import get_redis_sync
            r = get_redis_sync()
            if r is None:
                return None
            raw = r.get(f"{_get_redis_key_prefix()}:{job_id}")
            if raw:
                return json.loads(raw)
        except Exception as e:
            logger.debug(f"[IndexService] 从 Redis 读取进度失败: {e}")
        return None

    # ============================================================
    # 文件扫描
    # ============================================================

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
                # S5 第 49-50 天：统一转为 POSIX 正斜杠相对路径，
                # 避免 Windows 反斜杠与 Linux 正斜杠不一致导致下游
                #（LanceDB/BM25/依赖图/符号表）存储格式混乱。
                rel_path = to_posix_relpath(os.path.relpath(full_path, workspace_root))
                files.append(rel_path)

        return files

    # ============================================================
    # 全量索引（后台线程）
    # ============================================================

    def start_index(
        self,
        workspace_root: str,
        force_rebuild: bool = False,
        priority_files: Optional[List[str]] = None,
    ) -> dict:
        """
        启动全量索引（后台线程异步执行）。

        第 37-38 天改造：
        - 立即返回 job_id 与 total_files，不阻塞 HTTP 请求
        - 后台线程执行扫描→解析→切片→批量向量化→批量写入
        - priority_files 优先处理（即用即索引策略）

        Args:
            workspace_root: 工作区根路径
            force_rebuild: 是否强制重建（清空已有索引与向量库）
            priority_files: 优先索引的文件列表（相对路径），如用户当前打开的文件

        Returns:
            {"job_id": ..., "total_files": ...}
        """
        with self._lock:
            if self.status == "indexing":
                return {
                    "job_id": self.job_id,
                    "total_files": self.total,
                    "message": "索引正在进行中",
                }

            self.job_id = str(uuid.uuid4())
            self.status = "indexing"
            self.processed = 0
            self.total_symbols = 0
            self.message = None
            self.workspace_root = workspace_root
            if force_rebuild:
                self._index.clear()

        # 前台快速扫描文件列表，用于立即返回 total_files
        # （扫描本身不做解析/向量化，对 100 文件仓库通常 < 1s）
        try:
            files = self._scan_files(workspace_root)
        except ValueError:
            with self._lock:
                self.status = "error"
                self.message = f"工作区路径不存在: {workspace_root}"
            raise

        with self._lock:
            self.total = len(files)

        # 启动后台线程执行索引
        self._index_thread = threading.Thread(
            target=self._run_index,
            args=(workspace_root, files, force_rebuild, priority_files),
            daemon=True,
            name=f"index-worker-{self.job_id[:8]}",
        )
        self._index_thread.start()

        logger.info(
            f"[IndexService] 索引任务已启动 job_id={self.job_id} "
            f"total_files={self.total} priority={len(priority_files or [])}"
        )
        return {"job_id": self.job_id, "total_files": self.total}

    def _run_index(
        self,
        workspace_root: str,
        files: List[str],
        force_rebuild: bool,
        priority_files: Optional[List[str]],
    ) -> None:
        """
        后台线程：执行全量索引流程。

        流程：
        1. 初始化向量库（按维度建表；force_rebuild 时清空）
        2. 重排文件：priority_files 优先（即用即索引）
        3. 逐文件解析→切片，收集到批次缓冲区
        4. 每 INDEX_BATCH_SIZE 个文件批量向量化→删旧→批量写入
        5. 更新进度到 Redis
        6. 完成后写入标记文件
        """
        # 1. 初始化向量库与依赖图
        vector_store = get_vector_store()
        try:
            embed_client = get_embedding_client()
            vector_store.create_table(vector_dim=embed_client.dimension)
            if force_rebuild:
                vector_store.clear_table()
                # S5 第 41-42 天：强制重建时同时清空 BM25 索引
                try:
                    from .bm25_index import get_bm25_index
                    get_bm25_index().clear()
                except Exception as e:
                    logger.warning(f"[IndexService] 清空 BM25 索引失败: {e}")
                # S5 第 49-50 天：强制重建时清空检索缓存
                self._invalidate_retrieval_cache()
        except Exception as e:
            logger.warning(f"[IndexService] 向量库初始化失败，索引将跳过向量化: {e}")

        # 依赖图初始化：force_rebuild 时清空；否则尝试从 JSON 恢复
        dep_graph = get_dependency_graph()
        if force_rebuild:
            dep_graph.clear()
        else:
            graph_file = _get_graph_persist_path(workspace_root)
            if not dep_graph.load(graph_file):
                dep_graph.clear()

        # 2. 重排文件：priority_files 优先
        # S5 第 49-50 天：priority_files 由插件传入，可能含 Windows 反斜杠，
        # 统一归一化为 POSIX 正斜杠，确保与 _scan_files 输出的 files 能正确匹配。
        if priority_files:
            priority_files = [to_posix_relpath(f) for f in priority_files]
        ordered_files = self._reorder_priority(files, priority_files)
        logger.info(f"[IndexService] 开始索引 {len(ordered_files)} 个文件")

        # 预注册所有文件到依赖图（确保 import 解析能匹配到本地文件，
        # 即使被 import 的文件在遍历顺序中靠后）
        from .parser_factory import detect_language
        for rel_path in ordered_files:
            dep_graph.register_file(rel_path, detect_language(rel_path))

        # 3-4. 逐文件处理 + 批量提交
        batch_chunks: List[CodeChunk] = []
        batch_file_paths: List[str] = []
        batch_size = _get_batch_size()

        for rel_path in ordered_files:
            full_path = os.path.join(workspace_root, rel_path)
            try:
                table = parse_file(full_path)
                table.file_path = rel_path
                with self._lock:
                    self._index[rel_path] = table
                    self.total_symbols += len(table.symbols)

                # 切片并加入批次缓冲区
                # 注意：chunk_file(full_path, ...) 用 full_path 读取源码，
                # 但生成的 CodeChunk.file_path 也会被设为 full_path（绝对路径）。
                # 检索结果与插件跳转都需要相对路径，故此处统一修正为 rel_path。
                chunks = chunk_file(full_path, table)
                for _c in chunks:
                    _c.file_path = rel_path
                batch_chunks.extend(chunks)
                batch_file_paths.append(rel_path)

                # 第 39-40 天：构建依赖图（提取 import / call 边）
                try:
                    dep_graph.build_from_file(rel_path, table, full_path=full_path)
                except Exception as e:
                    logger.debug(f"[IndexService] 依赖图构建失败 {rel_path}: {e}")
            except Exception as e:
                logger.warning(f"[IndexService] 索引文件失败 {rel_path}: {e}")
            finally:
                with self._lock:
                    self.processed += 1
                self._write_progress_to_redis()

            # 每 batch_size 个文件批量提交一次
            if len(batch_file_paths) >= batch_size:
                self._flush_batch(batch_chunks, batch_file_paths)
                batch_chunks = []
                batch_file_paths = []

        # 处理剩余的批次
        if batch_chunks:
            self._flush_batch(batch_chunks, batch_file_paths)

        # 6. 完成：持久化依赖图并写标记
        try:
            dep_graph.save(_get_graph_persist_path(workspace_root))
            graph_stats = dep_graph.stats()
            logger.info(
                f"[IndexService] 依赖图已持久化: "
                f"{graph_stats['local_files']} 文件, "
                f"{graph_stats['file_edges']} 文件边, "
                f"{graph_stats['call_edges']} 调用边"
            )
        except Exception as e:
            logger.warning(f"[IndexService] 依赖图持久化失败: {e}")

        # S5 第 41-42 天：全量索引完成后构建 BM25 关键词索引
        try:
            from .bm25_index import get_bm25_index
            bm25_count = get_bm25_index().build()
            logger.info(f"[IndexService] BM25 索引构建完成：{bm25_count} 个 Chunk")
        except Exception as e:
            logger.warning(f"[IndexService] BM25 索引构建失败（不影响索引流程）: {e}")

        with self._lock:
            self.status = "done"
            self.message = f"索引完成，共 {self.total_symbols} 个符号"
        self._write_progress_to_redis()
        self._write_index_marker(workspace_root)
        logger.info(f"[IndexService] 索引完成 job_id={self.job_id} symbols={self.total_symbols}")

    @staticmethod
    def _reorder_priority(
        files: List[str], priority_files: Optional[List[str]]
    ) -> List[str]:
        """
        将 priority_files 排在列表前面，其余保持原序。

        实现"即用即索引"策略：优先索引用户当前打开的文件，
        其余文件后台静默处理。
        """
        if not priority_files:
            return files
        files_set = set(files)
        # 去重且保持顺序：先按 priority_files 的顺序，再按 files 的顺序
        seen = set()
        result: List[str] = []
        # 1. 按 priority_files 指定的顺序优先排列
        for f in priority_files:
            if f in files_set and f not in seen:
                result.append(f)
                seen.add(f)
        # 2. 剩余文件按原始顺序追加
        for f in files:
            if f not in seen:
                result.append(f)
                seen.add(f)
        return result

    def _flush_batch(
        self, chunks: List[CodeChunk], file_paths: List[str]
    ) -> None:
        """
        批量提交：向量化 → 删除旧向量 → 批量写入。

        任何环节失败仅记录日志，不中断整体索引流程
        （保证索引不中断，S4 风险应对）。
        """
        if not chunks:
            return

        try:
            embed_client = get_embedding_client()
            embed_client.embed_chunks(chunks)
            store = get_vector_store()
            # 先批量删除旧向量（这些文件此前可能已索引过）
            for fp in file_paths:
                try:
                    store.delete_by_file(fp)
                except Exception as e:
                    logger.warning(f"[IndexService] 删除旧向量失败 {fp}: {e}")
            # 批量写入新向量
            store.insert_chunks(chunks)
            logger.debug(
                f"[IndexService] 批次提交完成: {len(file_paths)} 个文件, "
                f"{len(chunks)} 个 Chunk"
            )
        except Exception as e:
            logger.warning(f"[IndexService] 批次向量化入库失败: {e}", exc_info=True)

    def _write_progress_to_redis(self) -> None:
        """
        将当前索引进度写入 Redis。

        Redis 不可用时静默降级（不影响索引），插件轮询 GET /status
        时会回退到内存状态。
        """
        job_id = self.job_id
        if not job_id:
            return
        status = self.get_status()
        try:
            from app.services.redis_client import get_redis_sync
            r = get_redis_sync()
            if r is not None:
                r.setex(
                    f"{_get_redis_key_prefix()}:{job_id}",
                    _get_redis_ttl(),
                    json.dumps(status, ensure_ascii=False),
                )
        except Exception as e:
            logger.debug(f"[IndexService] Redis 进度写入失败（不影响索引）: {e}")

    def _write_index_marker(self, workspace_root: str) -> None:
        """写入索引完成标记文件，供插件检测当前仓库是否已索引。"""
        try:
            marker_path = os.path.join(workspace_root, _get_marker_file())
            os.makedirs(os.path.dirname(marker_path), exist_ok=True)
            with open(marker_path, "w", encoding="utf-8") as f:
                f.write(json.dumps({
                    "job_id": self.job_id,
                    "completed_at": time.time(),
                    "total_symbols": self.total_symbols,
                    "workspace_root": workspace_root,
                }, ensure_ascii=False, indent=2))
            logger.debug(f"[IndexService] 索引标记文件已写入: {marker_path}")
        except Exception as e:
            logger.warning(f"[IndexService] 写入索引标记文件失败: {e}")

    def wait_for_done(self, timeout: float = 60.0) -> bool:
        """
        等待后台索引线程完成（测试用）。

        Args:
            timeout: 超时秒数

        Returns:
            True 表示索引完成（done 或 error），False 表示超时
        """
        if self._index_thread is None:
            return True
        self._index_thread.join(timeout=timeout)
        return not self._index_thread.is_alive()

    # ============================================================
    # 增量更新
    # ============================================================

    def update_file(self, file_path: str, action: str, workspace_root: Optional[str] = None) -> dict:
        """
        增量更新：处理单个文件的变更。

        action: modified / deleted / renamed

        第 37-38 天改进：
        - 使用缓存的 workspace_root（首次索引时传入），无需每次由插件传入
        - 支持外部传入 workspace_root 覆盖（如插件切换项目时）

        第 39-40 天改进：
        - 同步更新依赖图：删除旧边、提取新引用关系、重新持久化
        """
        root = workspace_root or self.workspace_root or os.getcwd()
        dep_graph = get_dependency_graph()

        # S5 第 49-50 天：插件传入的 file_path 可能含 Windows 反斜杠，
        # 统一归一化为 POSIX 正斜杠，与 _scan_files 的全量索引保持一致，
        # 避免 delete_by_file / 符号表查找因分隔符不同而匹配失败。
        file_path = to_posix_relpath(file_path)

        if action == "deleted":
            with self._lock:
                removed = self._index.pop(file_path, None)
                count = len(removed.symbols) if removed else 0
                self.total_symbols = max(0, self.total_symbols - count)
            # 同步删除向量库与依赖图中的旧记录
            try:
                get_vector_store().delete_by_file(file_path)
            except Exception as e:
                logger.warning(f"[IndexService] 删除向量库记录失败 {file_path}: {e}")
            try:
                dep_graph.remove_file(file_path)
                dep_graph.save(_get_graph_persist_path(root))
            except Exception as e:
                logger.warning(f"[IndexService] 删除依赖图记录失败 {file_path}: {e}")
            # S5 第 41-42 天：标记 BM25 索引为脏数据（下次检索时重建）
            try:
                from .bm25_index import get_bm25_index
                get_bm25_index().mark_dirty()
            except Exception as e:
                logger.warning(f"[IndexService] 标记 BM25 索引失败: {e}")
            # S5 第 49-50 天：索引变更后清空检索缓存，避免返回过期结果
            self._invalidate_retrieval_cache()
            return {
                "success": True,
                "message": f"已删除 {file_path} 的索引",
                "symbols_count": 0,
            }

        # modified / renamed: 重新解析
        full_path = os.path.join(root, file_path)
        if not os.path.exists(full_path):
            return {"success": False, "message": f"文件不存在: {file_path}", "symbols_count": 0}

        table = parse_file(full_path)
        table.file_path = file_path
        with self._lock:
            old = self._index.get(file_path)
            old_count = len(old.symbols) if old else 0
            self._index[file_path] = table
            self.total_symbols = self.total_symbols - old_count + len(table.symbols)

        # 重新切片向量化并写入向量库（单文件无需批量，直接处理）
        self._index_single_file_to_store(full_path, file_path, table)

        # 同步更新依赖图：重新提取引用关系并持久化
        try:
            dep_graph.build_from_file(file_path, table, full_path=full_path)
            dep_graph.save(_get_graph_persist_path(root))
        except Exception as e:
            logger.warning(f"[IndexService] 依赖图增量更新失败 {file_path}: {e}")

        # S5 第 41-42 天：标记 BM25 索引为脏数据（下次检索时重建）
        try:
            from .bm25_index import get_bm25_index
            get_bm25_index().mark_dirty()
        except Exception as e:
            logger.warning(f"[IndexService] 标记 BM25 索引失败: {e}")

        # S5 第 49-50 天：索引变更后清空检索缓存，避免返回过期结果
        self._invalidate_retrieval_cache()

        return {
            "success": True,
            "message": f"已更新 {file_path} 的索引",
            "symbols_count": len(table.symbols),
        }

    def _invalidate_retrieval_cache(self) -> None:
        """
        S5 第 49-50 天：清空检索结果缓存。

        索引变更（文件增删改）后调用，避免 hybrid_search 返回过期的检索结果。
        失败时仅记录日志，不中断索引流程。
        """
        try:
            from .retrieval_cache import get_retrieval_cache
            get_retrieval_cache().invalidate()
        except Exception as e:
            logger.warning(f"[IndexService] 清空检索缓存失败: {e}")

    def _index_single_file_to_store(
        self, full_path: str, rel_path: str, table: SymbolTable
    ) -> None:
        """
        对单个文件执行：切片 → 向量化 → 删旧向量 → 插新向量。

        用于增量更新（单文件，无需批量）。
        任何环节失败仅记录日志，不中断流程。
        """
        try:
            chunks = chunk_file(full_path, table)
            if not chunks:
                return
            embed_client = get_embedding_client()
            embed_client.embed_chunks(chunks)
            store = get_vector_store()
            store.delete_by_file(rel_path)
            store.insert_chunks(chunks)
        except Exception as e:
            logger.warning(f"[IndexService] 文件向量化入库失败 {rel_path}: {e}")

    # ============================================================
    # 符号查询
    # ============================================================

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
