"""
检索延迟压测脚本（S5 第 49-50 天：联调、性能压测与 S5 Demo）

模拟 50 个并发用户的 RAG 请求，统计 P95 延迟。
压测范围：检索 + 重排序 + 上下文组装（不含 LLM 推理）。
目标：P95 延迟 < 800ms，CPU 峰值 < 80%。

用法：
    python test_retrieval_latency.py
    python test_retrieval_latency.py --concurrency 50 --rounds 3

输出：
    - 每个请求的延迟明细
    - P50 / P95 / P99 / 平均 / 最大 延迟
    - 缓存命中率（若启用了检索缓存）
    - 是否达标（P95 < 800ms）

说明：
    - 若 LanceDB 索引未构建，hybrid_search 返回空结果，脚本仍会测量
      "空检索"的延迟（含 Cross-Encoder 加载等开销），并给出提示。
    - 脚本不依赖外部 LLM API，仅测试检索侧性能。
"""

import argparse
import logging
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("retrieval_stress_test")

# 压测目标（S5 验收标准）
TARGET_P95_MS = 800.0
TARGET_CPU_PEAK = 80.0  # CPU 峰值百分比（参考值，需外部工具配合测量）

# 示例查询集（覆盖语义/关键词/符号三类场景）
SAMPLE_QUERIES = [
    "排序算法的实现",
    "怎么处理数据流",
    "用户登录逻辑在哪里",
    "DataProcessor 类",
    "save 函数被谁调用了",
    "数据库连接池配置",
    "错误处理中间件",
    "token 预算控制",
    "BM25 关键词检索",
    "Cross-Encoder 重排序",
]


def run_single_request(query: str, top_k: int = 5) -> Dict:
    """
    执行单次 RAG 检索请求（检索 + 重排序 + 上下文组装）。

    模拟 chat.py 中 auto_context 路径的完整检索侧流程：
      1. hybrid_search（向量 + BM25 + 符号 + RRF + Cross-Encoder）
      2. context_assembler.assemble（智能压缩 + Token 预算控制）

    Args:
        query: 用户查询
        top_k: 注入 Prompt 的片段数

    Returns:
        {"query": ..., "latency_ms": ..., "chunks": ..., "error": ...}
    """
    start = time.perf_counter()
    try:
        from app.services.code_index.fusion_reranker import hybrid_search
        from app.services.code_index.context_assembler import get_context_assembler

        # 1. 混合检索
        chunks = hybrid_search(query, top_k=top_k, include_meta=False)

        # 2. 上下文组装（cursor_file=None，model_context_window=None 走默认预算）
        assembler = get_context_assembler()
        asm_result = assembler.assemble(
            chunks=chunks,
            cursor_file=None,
            model_context_window=None,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            "query": query,
            "latency_ms": elapsed_ms,
            "chunks": len(chunks),
            "kept": asm_result.stats.kept_chunks,
            "error": None,
        }
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            "query": query,
            "latency_ms": elapsed_ms,
            "chunks": 0,
            "kept": 0,
            "error": str(e),
        }


def percentile(data: List[float], p: float) -> float:
    """计算百分位数（p 为 0-100）。"""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[f]
    return sorted_data[f] + (sorted_data[c] - sorted_data[f]) * (k - f)


def print_report(results: List[Dict], concurrency: int, total_requests: int) -> bool:
    """
    打印压测报告并判断是否达标。

    Returns:
        True 表示达标（P95 < 800ms），False 表示超标。
    """
    latencies = [r["latency_ms"] for r in results]
    errors = [r for r in results if r["error"]]
    success = [r for r in results if not r["error"]]

    print("\n" + "=" * 70)
    print("📊 检索延迟压测报告")
    print("=" * 70)
    print(f"  并发用户数: {concurrency}")
    print(f"  总请求数:   {total_requests}")
    print(f"  成功请求:   {len(success)}")
    print(f"  失败请求:   {len(errors)}")
    print("-" * 70)

    if not latencies:
        print("  ⚠️  无有效请求数据")
        return False

    print(f"  平均延迟:   {statistics.mean(latencies):.1f} ms")
    print(f"  中位数:     {percentile(latencies, 50):.1f} ms (P50)")
    print(f"  P95 延迟:   {percentile(latencies, 95):.1f} ms")
    print(f"  P99 延迟:   {percentile(latencies, 99):.1f} ms")
    print(f"  最大延迟:   {max(latencies):.1f} ms")
    print(f"  最小延迟:   {min(latencies):.1f} ms")
    print("-" * 70)

    # 缓存命中率统计（从日志中推断较难，这里直接提示）
    print("  💡 提示：检索缓存命中率请查看后端日志中")
    print("     '[FusionReranker] 检索缓存命中' 的出现频率。")

    if errors:
        print("-" * 70)
        print("  ❌ 失败请求详情（前 5 条）:")
        for r in errors[:5]:
            print(f"     - query='{r['query'][:30]}', error={r['error'][:80]}")

    print("=" * 70)

    p95 = percentile(latencies, 95)
    passed = p95 < TARGET_P95_MS and len(errors) == 0
    if passed:
        print(f"  ✅ 达标：P95 = {p95:.1f}ms < 目标 {TARGET_P95_MS}ms")
    else:
        print(f"  ❌ 超标：P95 = {p95:.1f}ms >= 目标 {TARGET_P95_MS}ms")
        if errors:
            print(f"     （且有 {len(errors)} 个请求失败）")
        print("  建议：")
        print("    1. 启用 Redis 缓存（REDIS_ENABLED=true）缓存重复查询结果")
        print("    2. 降低 RERANK_MAX_CANDIDATES（当前默认 10）以减少重排序开销")
        print("    3. 考虑关闭 RERANK_ENABLED 仅走 RRF（牺牲精度换延迟）")
    print("=" * 70 + "\n")
    return passed


def main():
    parser = argparse.ArgumentParser(description="S5 检索延迟压测脚本")
    parser.add_argument(
        "--concurrency", type=int, default=50,
        help="并发用户数（默认 50，S5 验收标准）"
    )
    parser.add_argument(
        "--rounds", type=int, default=2,
        help="每个用户的请求轮数（总请求数 = concurrency * rounds）"
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="检索返回的 Top-K 片段数（默认 5）"
    )
    parser.add_argument(
        "--warmup", type=int, default=2,
        help="预热请求数（触发模型加载，不计入统计）"
    )
    args = parser.parse_args()

    concurrency = args.concurrency
    rounds = args.rounds
    top_k = args.top_k
    total_requests = concurrency * rounds

    print(f"🚀 开始检索延迟压测")
    print(f"   并发={concurrency}, 每用户轮数={rounds}, 总请求={total_requests}, top_k={top_k}")

    # 1. 预热：触发 Cross-Encoder 模型加载等一次性开销
    print(f"\n🔥 预热中（{args.warmup} 个请求）...")
    warmup_results = []
    with ThreadPoolExecutor(max_workers=min(args.warmup, 8)) as executor:
        futures = [
            executor.submit(run_single_request, SAMPLE_QUERIES[i % len(SAMPLE_QUERIES)], top_k)
            for i in range(args.warmup)
        ]
        for f in as_completed(futures):
            warmup_results.append(f.result())
    print(f"   预热完成，平均延迟 {statistics.mean([r['latency_ms'] for r in warmup_results]):.1f}ms")

    # 2. 正式压测：concurrency 个并发用户，每个用户发 rounds 个请求
    print(f"\n⚡ 正式压测中（{total_requests} 个请求，{concurrency} 并发）...")
    results: List[Dict] = []
    results_lock = threading.Lock()

    def user_worker(user_id: int):
        """单个并发用户的请求循环。"""
        user_results = []
        for r in range(rounds):
            query = SAMPLE_QUERIES[(user_id * rounds + r) % len(SAMPLE_QUERIES)]
            res = run_single_request(query, top_k)
            user_results.append(res)
        with results_lock:
            results.extend(user_results)

    start_time = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(user_worker, uid) for uid in range(concurrency)]
        for f in as_completed(futures):
            f.result()  # 传播异常
    total_time = time.perf_counter() - start_time

    print(f"   压测完成，总耗时 {total_time:.2f}s，吞吐量 {total_requests / total_time:.1f} req/s")

    # 3. 输出报告
    passed = print_report(results, concurrency, total_requests)

    # 4. 返回退出码
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
