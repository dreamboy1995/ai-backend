# app/api/graph.py
# S4 第 39-40 天：依赖关系图查询接口
# - GET /v1/graph/related?file=<path>&depth=<n>
#     返回与指定文件强关联的上下游文件（BFS 遍历）
# - GET /v1/graph/imports?file=<path>
#     返回该文件直接 import 的本地模块名（验收用）
# - GET /v1/graph/stats
#     返回依赖图统计信息（节点数、边数等，调试用）

import logging

from fastapi import APIRouter, HTTPException, Query

from app.config import settings
from app.models.schemas import (
    GraphEdgeInfo,
    GraphRelatedItem,
    GraphRelatedResponse,
    GraphImportsResponse,
)
from app.services.code_index.dependency_graph import get_dependency_graph

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/related", response_model=GraphRelatedResponse)
async def get_related_files(
    file: str = Query(..., min_length=1, description="查询起点文件（相对路径）"),
    depth: int = Query(
        default=settings.GRAPH_DEFAULT_DEPTH,
        ge=1,
        le=settings.GRAPH_MAX_DEPTH,
        description="BFS 遍历深度（1=直接依赖，2=间接依赖）",
    ),
):
    """
    查询与指定文件强关联的上下游文件列表。

    - upstream：被查询文件依赖的文件（即 file 所 import 的本地模块及其传递依赖）
    - downstream：依赖被查询文件的文件（即谁 import 了 file）

    实现说明：
      - BFS 从 file 出发，沿依赖图的 forward / reverse 邻接表遍历
      - 仅追踪能解析为本地文件的边，未解析的模块名（如标准库）不会扩展
      - 结果按 depth 升序，便于前端按"距离"分组展示
    """
    graph = get_dependency_graph()
    result = graph.get_related_files(file, depth=depth)

    upstream_items = [
        GraphRelatedItem(
            file_path=item["file_path"],
            direction="upstream",
            depth=item["depth"],
            edge=GraphEdgeInfo(**item["edge"]) if item.get("edge") else None,
        )
        for item in result.get("upstream", [])
    ]
    downstream_items = [
        GraphRelatedItem(
            file_path=item["file_path"],
            direction="downstream",
            depth=item["depth"],
            edge=GraphEdgeInfo(**item["edge"]) if item.get("edge") else None,
        )
        for item in result.get("downstream", [])
    ]

    return GraphRelatedResponse(
        file_path=file,
        depth=depth,
        upstream=upstream_items,
        downstream=downstream_items,
        total=len(upstream_items) + len(downstream_items),
    )


@router.get("/imports", response_model=GraphImportsResponse)
async def get_file_imports(
    file: str = Query(..., min_length=1, description="查询文件（相对路径）"),
):
    """
    返回该文件直接 import 的所有本地模块名。

    验收接口（来自 Sprint_4.md 第 39-40 天）：
      "对项目根目录的 app.py 调用依赖图接口，能返回它 import 的所有本地模块名"

    返回结构：
      {
        "file_path": "...",
        "imports": [
          {"file_path": "utils.py", "module_name": "utils", "line": 1, "resolved": true},
          {"file_path": null, "module_name": "os", "line": 2, "resolved": false},
          ...
        ]
      }
      - resolved=true 表示已解析为本地文件（file_path 非空）
      - resolved=false 表示未解析为本地文件（如标准库 / 三方包），file_path 为 null
    """
    graph = get_dependency_graph()
    imports = graph.get_imports(file)
    if not imports:
        # 文件未被索引过时给出明确提示
        raise HTTPException(
            status_code=404,
            detail=f"文件 '{file}' 未在依赖图中找到，请先调用 POST /v1/index/start 触发索引",
        )
    return GraphImportsResponse(file_path=file, imports=imports)


@router.get("/stats")
async def get_graph_stats():
    """依赖图统计信息（调试用）"""
    graph = get_dependency_graph()
    return graph.stats()
