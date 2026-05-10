"""
Main Agent 子观点拆解质量测试

走真实的 run_agent_loop 代码路径，保证与生产环境 100% 一致。
唯一差异：拦截 do_batch_search，拿到拆解结果后立即退出，不执行实际搜索。

用法：
    # 测试所有观点
    python tests/test_decomposition.py

    # 测试单个观点（按编号）
    python tests/test_decomposition.py 1

    # 测试多个观点
    python tests/test_decomposition.py 1 3 5
"""

import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# 项目根目录加入 path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import cfg
from agents.agent_loop import StepOutcome, run_agent_loop
from agents.main_handler import MainAgentHandler


# ── 测试观点集 ──────────────────────────────────────────────────

TEST_CLAIMS = [
    {
        "id": 1,
        "domain": "政治/治理",
        "user_type": "普通市民",
        "claim": "陈吉宁是近十年来上海最好的市长",
        "notes": "涉及政治人物评价，需要多维度拆解（经济、民生、城市建设等）",
    },
    {
        "id": 2,
        "domain": "计算机科学",
        "user_type": "研究生/学者",
        "claim": "知识图谱作为一种技术体系，是大数据时代知识工程的代表性进展",
        "notes": "学术性较强，需要技术演进、应用、学术影响等维度",
    },
    {
        "id": 3,
        "domain": "医学/健康",
        "user_type": "关注健康的普通人",
        "claim": "间歇性断食是目前科学证据最充分的长寿饮食方式",
        "notes": "健康类声明，容易有偏见，需要区分人类实验和动物实验",
    },
    {
        "id": 4,
        "domain": "经济/房地产",
        "user_type": "财经媒体从业者",
        "claim": "房地产税是解决中国房价过高问题的最有效政策工具",
        "notes": "经济政策辩论，需要国内外对比、经济学理论、实际案例等",
    },
    {
        "id": 5,
        "domain": "能源/环境",
        "user_type": "环保组织研究员",
        "claim": "核电是实现碳中和目标中最不可替代的基荷能源技术",
        "notes": "涉及能源政策、安全性、经济性、技术发展等多维度争议",
    },
    {
        "id": 6,
        "domain": "历史",
        "user_type": "历史爱好者/学生",
        "claim": "郑和下西洋对世界航海史的实际贡献被严重低估了",
        "notes": "历史评价类，需要史料、国际比较、学术争论等维度",
    },
    {
        "id": 7,
        "domain": "社会/劳动",
        "user_type": "企业HR/管理者",
        "claim": "远程办公的大规模普及正在不可逆地改变城市空间结构和劳动力市场",
        "notes": "社会趋势类，涉及城市规划、房产、劳动经济学等",
    },
    {
        "id": 8,
        "domain": "生物医学",
        "user_type": "医学院学生",
        "claim": "肠道微生物组研究是近二十年来理解人类慢性疾病最重要的突破口",
        "notes": "前沿科学，需要区分已验证结论和初步发现",
    },
    {
        "id": 9,
        "domain": "文化/教育",
        "user_type": "中学教师/家长",
        "claim": "短视频平台正在系统性地降低青少年的深度阅读能力和注意力持续时间",
        "notes": "文化争议，需要实证研究、反面论据、跨文化比较",
    },
    {
        "id": 10,
        "domain": "国际关系",
        "user_type": "政策研究人员",
        "claim": "一带一路倡议从根本上重塑了发展中国家基础设施融资的全球格局",
        "notes": "地缘政治，需要经济数据、区域案例、多方评价",
    },
    {
        "id": 11,
        "domain": "法律/科技伦理",
        "user_type": "法学院学生/互联网从业者",
        "claim": "算法推荐系统应当承担与传统媒体编辑同等的内容审查法律责任",
        "notes": "法律与科技交叉，需要法理、判例、各国立法比较",
    },
    {
        "id": 12,
        "domain": "人工智能",
        "user_type": "AI创业者/投资人",
        "claim": "大语言模型的涌现能力证明了规模定律是通向通用人工智能的可行路径",
        "notes": "AI前沿争论，需要技术论据、批评声音、哲学层面讨论",
    },
]


# ── 拦截式 Handler ──────────────────────────────────────────────


class DecompositionCapture(MainAgentHandler):
    """继承真实 MainAgentHandler，只拦截 do_batch_search。

    走完真实的 system prompt 构建、披露窗口加载、agent loop 消息流，
    LLM 返回 batch_search 调用时拦截参数并立即退出。
    """

    def __init__(self, query: str):
        super().__init__(query=query, on_status=None)
        self.captured_subclaims: list[dict] = []
        self.captured_thinking: str = ""

    async def do_batch_search(self, args: dict) -> StepOutcome:
        """拦截：只记录拆解结果，不执行搜索，立即退出。"""
        self.captured_subclaims = args.get("subclaims", [])
        # 返回 should_exit=True 终止循环
        return StepOutcome(
            data={"intercepted": True, "subclaims": self.captured_subclaims},
            should_exit=True,
        )


# ── 核心测试逻辑 ──────────────────────────────────────────────────


async def test_one_claim(claim_info: dict) -> dict:
    """用真实 agent loop 测试单个观点的拆解。"""

    claim_text = claim_info["claim"]
    handler = DecompositionCapture(query=claim_text)

    # 捕获 LLM 思考过程
    thinking_parts: list[str] = []

    def on_step(event_type: str, content: str) -> None:
        if event_type == "think":
            thinking_parts.append(content)

    start_time = time.time()

    try:
        # 走真实的 run_agent_loop，和生产环境完全一致
        await run_agent_loop(
            handler=handler,
            user_message=claim_text,
            max_turns=3,  # 拆解只需 1 轮，留 3 轮余量防止异常
            on_step=on_step,
        )
    except Exception as e:
        return {
            **claim_info,
            "error": str(e),
            "thinking": None,
            "subclaims": [],
            "subclaim_count": 0,
            "latency_s": round(time.time() - start_time, 2),
        }

    latency = round(time.time() - start_time, 2)
    thinking = "\n".join(thinking_parts) if thinking_parts else "(无思考文本)"

    return {
        **claim_info,
        "error": None,
        "thinking": thinking,
        "subclaims": handler.captured_subclaims,
        "subclaim_count": len(handler.captured_subclaims),
        "latency_s": latency,
    }


# ── 输出格式化 ──────────────────────────────────────────────────


def print_result(result: dict) -> None:
    """格式化打印单个测试结果。"""
    print("\n" + "=" * 80)
    print(f"【{result['id']}】{result['domain']}  |  模拟用户：{result['user_type']}")
    print(f"观点：{result['claim']}")
    print("-" * 80)

    if result.get("error"):
        print(f"错误：{result['error']}")
        return

    print(f"响应耗时：{result['latency_s']}s")
    print(f"\nLLM 思考过程：")
    print(result["thinking"])
    print(f"\n拆解结果（{result['subclaim_count']} 个子观点）：")

    for i, sc in enumerate(result["subclaims"], 1):
        subclaim = sc.get("subclaim", "")
        topic = sc.get("topic_context", "")
        print(f"  {i}. {subclaim}")
        if topic:
            print(f"     [topic: {topic}]")

    print("=" * 80)


def save_results(results: list[dict], output_path: Path) -> None:
    """保存结果到 JSON 文件。"""
    summary = []
    for r in results:
        summary.append({
            "id": r["id"],
            "domain": r["domain"],
            "claim": r["claim"],
            "subclaim_count": r.get("subclaim_count", 0),
            "subclaims": [sc.get("subclaim", "") for sc in r.get("subclaims", [])],
            "thinking": r.get("thinking"),
            "latency_s": r.get("latency_s"),
            "error": r.get("error"),
        })

    output = {
        "test_time": datetime.now().isoformat(),
        "model": cfg.llm.model,
        "total_claims": len(results),
        "summary": summary,
        "full_results": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n完整结果已保存到：{output_path}")


def print_summary(results: list[dict]) -> None:
    """打印汇总分析。"""
    print("\n" + "=" * 80)
    print("汇总分析")
    print("=" * 80)

    successful = [r for r in results if not r.get("error")]
    failed = [r for r in results if r.get("error")]

    print(f"测试模型：{cfg.llm.model}")
    print(f"成功：{len(successful)}/{len(results)}")

    if failed:
        print(f"失败：{', '.join(str(r['id']) for r in failed)}")

    if successful:
        counts = [r["subclaim_count"] for r in successful]
        latencies = [r["latency_s"] for r in successful]
        print(f"子观点数量：min={min(counts)}, max={max(counts)}, avg={sum(counts)/len(counts):.1f}")
        print(f"响应耗时：min={min(latencies)}s, max={max(latencies)}s, avg={sum(latencies)/len(latencies):.1f}s")

    print(f"\n{'ID':>3} | {'领域':<14} | {'子观点数':>5} | {'耗时':>6} | 观点摘要")
    print("-" * 80)
    for r in results:
        claim_short = r["claim"][:30] + ("..." if len(r["claim"]) > 30 else "")
        count = r.get("subclaim_count", "ERR")
        latency = f"{r.get('latency_s', 0)}s"
        print(f"{r['id']:>3} | {r['domain']:<14} | {count:>5} | {latency:>6} | {claim_short}")


# ── 入口 ──────────────────────────────────────────────────────────


async def main():
    if len(sys.argv) > 1:
        ids = [int(x) for x in sys.argv[1:]]
        claims = [c for c in TEST_CLAIMS if c["id"] in ids]
        if not claims:
            print(f"未找到 ID {ids} 对应的观点")
            return
    else:
        claims = TEST_CLAIMS

    print(f"Main Agent 拆解质量测试")
    print(f"模型：{cfg.llm.model}")
    print(f"待测观点：{len(claims)} 个")
    print(f"代码路径：run_agent_loop -> MainAgentHandler (拦截 do_batch_search)")

    results = []
    for claim_info in claims:
        print(f"\n正在测试 [{claim_info['id']}] {claim_info['claim'][:40]}...")
        result = await test_one_claim(claim_info)
        results.append(result)
        print_result(result)

    print_summary(results)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = ROOT / "tests" / "results" / f"decomposition_{timestamp}.json"
    save_results(results, output_path)


if __name__ == "__main__":
    asyncio.run(main())
