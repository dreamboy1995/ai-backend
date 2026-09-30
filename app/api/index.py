# app/api/index.py
# S4 第 31-32 天：代码索引控制接口
# - POST /v1/index/start   触发全量索引
# - GET  /v1/index/status  查询索引进度
# - POST /v1/index/update  增量更新通知

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
    """
    service = get_index_service()
    try:
        result = service.start_index(
            workspace_root=request.workspace_root,
            force_rebuild=request.force_rebuild,
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
    """
    service = get_index_service()
    status = service.get_status()
    return IndexStatusResponse(**status)


@router.post("/update", response_model=IndexUpdateResponse)
async def update_index(request: IndexUpdateRequest):
    """
    增量更新通知

    由插件在文件保存/删除/重命名时调用，后端仅重新处理该文件。
    """
    # 注意：workspace_root 应由插件在首次索引时传入并缓存，
    # 第 31-32 天暂用当前工作目录，第 37-38 天完善工作区管理。
    import os
    workspace_root = os.getcwd()
    service = get_index_service()
    result = service.update_file(
        workspace_root=workspace_root,
        file_path=request.file_path,
        action=request.action,
    )
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    return IndexUpdateResponse(**result)
