# app/api/index.py
# S4 第 31-38 天：代码索引控制接口
# - POST /v1/index/start   触发全量索引（后台线程异步执行，立即返回 job_id）
# - GET  /v1/index/status  查询索引进度（优先 Redis，降级内存）
# - POST /v1/index/update  增量更新通知（复用首次索引的 workspace_root）

import logging

from fastapi import APIRouter, HTTPException

from app.models.schemas import (
    IndexStartRequest,
    IndexStartResponse,
    IndexStatusResponse,
    IndexUpdateRequest,
    IndexUpdateResponse,
)
from app.services.code_index.index_service import get_index_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/start", response_model=IndexStartResponse)
async def start_index(request: IndexStartRequest):
    """
    触发全量索引

    输入工作区根路径，后端扫描并解析所有支持的源文件，建立符号表索引。
    第 37-38 天改造：后台线程异步执行，立即返回 job_id，不阻塞请求。

    - workspace_root: 工作区根路径
    - force_rebuild: 是否强制重建索引
    - priority_files: 优先索引的文件列表（相对路径），实现"即用即索引"
    """
    service = get_index_service()
    try:
        result = service.start_index(
            workspace_root=request.workspace_root,
            force_rebuild=request.force_rebuild,
            priority_files=request.priority_files,
        )
        return IndexStartResponse(
            job_id=result["job_id"],
            total_files=result["total_files"],
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"[IndexAPI] 启动索引失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"索引启动失败: {e}")


@router.get("/status", response_model=IndexStatusResponse)
async def get_index_status():
    """
    查询索引进度

    返回当前索引状态、已处理文件数、进度百分比、符号总数等。
    第 37-38 天：优先从 Redis 读取（多实例场景下更准确），
    Redis 不可用时降级到内存状态。
    """
    service = get_index_service()
    # 优先 Redis，降级内存
    status = service.get_status_from_redis() or service.get_status()
    # IndexStatusResponse 不含 workspace_root，过滤掉
    return IndexStatusResponse(
        status=status["status"],
        total=status["total"],
        processed=status["processed"],
        percentage=status["percentage"],
        total_symbols=status["total_symbols"],
        message=status["message"],
    )


@router.post("/update", response_model=IndexUpdateResponse)
async def update_index(request: IndexUpdateRequest):
    """
    增量更新通知

    由插件在文件保存/删除/重命名时调用，后端仅重新处理该文件。
    第 37-38 天：复用首次索引时缓存的 workspace_root，无需插件每次传入。
    """
    service = get_index_service()
    result = service.update_file(
        file_path=request.file_path,
        action=request.action,
    )
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    return IndexUpdateResponse(**result)
