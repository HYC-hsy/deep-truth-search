"""
Deep Truth Search — MVP 端到端冒烟测试

用 3 个示例观点跑通完整流程，检查：
1. 闭环是否通（从输入到输出无异常）
2. 证据数量 > 0
3. 输出格式符合「论点 + 论据」结构
4. 每条证据有 source_url
"""

import asyncio
import sys
from pathlib import Path

# 确保项目根目录在 Python path 中
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── 测试用例 ──────────────────────────────────────────────────

TEST_QUERIES = [
    "AI在2024取得突破",
    "多模态模型应用",
    "医疗AI发展",
]


async def run_single_test(query: str, idx: int) -> dict:
    """运行单个观点的端到端测试，返回测试结果摘要。"""
    from agents.main_agent import run_research

    print(f"\n{'='*60}")
    print(f"测试 {idx}/3: {query}")
    print(f"{'='*60}")

    errors = []

    try:
        output = await run_research(query)
    except Exception as e:
        return {"query": query, "pass": False, "errors": [f"运行异常: {e}"], "claims": 0, "evidences": 0}

    # 检查 1: 有论点
    if not output.claims or len(output.claims) == 0:
        errors.append("无论点返回 (claims 为空)")

    # 检查 2: 有证据
    if output.total_evidences == 0:
        errors.append("无证据返回 (total_evidences == 0)")

    # 检查 3: 每个论点有标题和证据
    for i, claim in enumerate(output.claims):
        if not claim.claim_title:
            errors.append(f"论点 {i+1} 缺少标题")
        if not claim.evidences:
            errors.append(f"论点 {i+1} '{claim.claim_title[:30]}' 无证据")

    # 检查 4: 每条证据有 source_url
    urls_missing = 0
    for claim in output.claims:
        for ev in claim.evidences:
            if not ev.source_url:
                urls_missing += 1
    if urls_missing > 0:
        errors.append(f"{urls_missing} 条证据缺少 source_url")

    # 检查 5: 输出可序列化为 JSON
    try:
        output.model_dump(mode="json")
    except Exception as e:
        errors.append(f"JSON 序列化失败: {e}")

    passed = len(errors) == 0

    # 打印结果
    status = "PASS" if passed else "FAIL"
    print(f"\n[{status}] {query}")
    print(f"  论点数: {len(output.claims)}")
    print(f"  证据数: {output.total_evidences}")
    for i, claim in enumerate(output.claims, 1):
        print(f"  论点 {i}: {claim.claim_title[:50]} ({len(claim.evidences)} 条证据)")
    if errors:
        for err in errors:
            print(f"  ERROR: {err}")

    return {
        "query": query,
        "pass": passed,
        "errors": errors,
        "claims": len(output.claims),
        "evidences": output.total_evidences,
    }


async def main():
    print("Deep Truth Search — MVP 端到端冒烟测试")
    print(f"测试 {len(TEST_QUERIES)} 个观点\n")

    results = []
    for idx, query in enumerate(TEST_QUERIES, 1):
        r = await run_single_test(query, idx)
        results.append(r)

    # 汇总
    print(f"\n{'='*60}")
    print("测试汇总")
    print(f"{'='*60}")

    total_pass = sum(1 for r in results if r["pass"])
    total_fail = len(results) - total_pass

    for r in results:
        status = "PASS" if r["pass"] else "FAIL"
        print(f"  [{status}] {r['query']} — {r['claims']} 论点, {r['evidences']} 证据")
        if r["errors"]:
            for err in r["errors"]:
                print(f"         ERROR: {err}")

    print(f"\n通过: {total_pass}/{len(results)}, 失败: {total_fail}/{len(results)}")

    if total_fail > 0:
        sys.exit(1)
    print("\nMVP 冒烟测试全部通过!")


if __name__ == "__main__":
    asyncio.run(main())
