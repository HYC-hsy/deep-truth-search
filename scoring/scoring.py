"""
Deep Truth Search — 五维评分模块

核心理念：让 LLM 像专业评审人员一样，从内容本身判断质量。
不依赖正则提取的信号（它们不可靠且会误导 LLM），
只提供 LLM 从正文中无法获得的少量客观信息（正文总长度、作者、日期）。

规则层极简：只在 LLM 物理上无法感知的极端情况下介入（通用主页、正文极短）。

五维度：
  - 权威性 authority (30分): 作者资质/机构背书/专业性
  - 准确性 accuracy  (30分): 引用/数据出处/可验证性
  - 目的性 purpose   (20分): 教育/研究/新闻 vs 营销/广告
  - 时效性 timeliness (10分): 发布日期/内容新鲜度
  - 覆盖度 coverage   (10分): 内容深度/完整性

用法：
    from scoring.scoring import score_page

    result = await score_page(page_content, subclaim)
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from config import cfg
from llm.llm_client import call_scoring_llm_json
from models import DIMENSION_LABELS, DimensionScore, PageContent, ScoreResult

logger = logging.getLogger(__name__)


# ── 通用主页检测 ───────────────────────────────────────────────
_GENERIC_TITLE_KEYWORDS = {"首页", "主页", "home", "index", "welcome", "about"}


# ── LLM 评分输出模型 ───────────────────────────────────────────


class DimensionDetail(BaseModel):
    """LLM 对单个维度的评分输出"""
    score: int = Field(description="该维度得分")
    reason: str = Field(description="评分理由（简短）")


class LLMScoreOutput(BaseModel):
    """LLM 五维评分的完整输出"""
    authority: DimensionDetail = Field(description="权威性 (0-30)")
    accuracy: DimensionDetail = Field(description="准确性 (0-30)")
    purpose: DimensionDetail = Field(description="目的性 (0-20)")
    timeliness: DimensionDetail = Field(description="时效性 (0-10)")
    coverage: DimensionDetail = Field(description="覆盖度 (0-10)")
    domain_tags: list[str] = Field(
        default_factory=list,
        description="该来源擅长的领域标签，2-4个短词（如 ['医学', 'mRNA疫苗', '临床试验']）",
    )


# ── 评分 Prompt ─────────────────────────────────────────────────

_SCORING_SYSTEM_PROMPT_TEMPLATE = """\
你是一位资深研究员，擅长评估信息源质量。请像一位专业人士那样，从内容本身出发判断这个网页作为证据来源的质量。

当前日期：{today}

## 核心评估原则

你要像一个真正的专业评审那样阅读这段内容，从文字本身感受它的质量：
- 作者是否有资质？论述是否专业？
- 说的话有没有依据？能不能追溯？数据是否具体？
- 是为了传播知识还是为了卖东西？
- 内容是否足够新、足够深入？

不要依赖我给你的元数据来做判断——元数据可能有误。请以你读到的正文内容为准。

## 评分维度

### 1. 权威性 authority（满分 30 分）

评估"这个人/机构有没有资格谈这个话题"。

从正文中判断：
- 作者是否署名？是否展现了该领域的专业知识？
- 是否有机构背书？该机构是否有公信力？
- 使用专业术语是否恰当？是否引用了同行研究？
- 是否经过同行评审或编辑把关？

分数档位：
- **25-30**：领域公认权威（署名专家+权威机构+同行认可）
- **18-24**：可信专业来源（有作者和机构，或虽无署名但展现深厚专业知识）
- **10-17**：一般来源（部分信息缺失，但有一定专业性）
- **0-9**：缺乏可信度（匿名+无机构+浅薄或带偏见）

### 2. 准确性 accuracy（满分 30 分）

评估"内容说的是不是真的，能不能验证"。

从正文中判断：
- 主张是否有出处？是否提到了具体的研究、报告、机构？
- 是否有具体数据（数字、百分比、统计量）？还是全是笼统描述？
- 信息是否可追溯验证？
- 是否区分了事实和观点？

不同类型来源侧重不同：
- 学术论文：看方法论、参考文献、实验数据
- 新闻报道：看信息来源标注、多方采访、事实核查
- 行业分析：看数据来源、统计方法、结论是否有数据支撑

分数档位：
- **25-30**：信息高度可靠（出处充分+数据丰富+可交叉验证）
- **20-24**：基本可靠（有部分出处和数据，主要事实可验证）
- **14-19**：可靠性一般（出处和数据较少，部分无法验证）
- **8-13**：可靠性偏低（引用极少，多为概括性描述）
- **0-7**：不可靠（无出处无数据，或存在明显偏差）

### 3. 目的性 purpose（满分 20 分）

评估"这篇内容的写作目的"。

从正文中判断：
- 是为了教育、研究、客观报道？还是推销产品、博取流量？
- 作者或机构是否在话题中有商业利益？
- 语气是客观分析还是煽动推销？
- 内容围绕论点还是围绕产品卖点？

分数档位：
- **18-20**：纯粹信息传播（学术研究、独立报道、政府信息发布）
- **14-17**：基本客观但有可识别立场
- **8-13**：有商业推广掺入（企业博客、带SEO导向）
- **3-7**：营销成分显著（软文、带货、流量导向）
- **0-2**：纯营销或严重偏见

### 4. 时效性 timeliness（满分 10 分）

评估"内容是否足够新"。

判断方法：
- 优先看正文中提到的时间线索（年份、事件、"最近"等表述）
- 元数据中的发布日期仅供参考，可能不准确
- 科技/政策类时效要求高，基础科学/历史类可以宽松

分数档位：
- **9-10**：6个月内
- **7-8**：6个月-1年
- **5-6**：1-2年
- **3-4**：2-3年
- **1-2**：3-5年
- **0**：5年以上
- 如果无法判断时间，给 4 分（不确定）

### 5. 覆盖度 coverage（满分 10 分）

评估"内容对主题的覆盖深度和广度"。

注意：你看到的正文是从原文首部、中部、尾部三段采样的。"正文总长度"反映了完整篇幅。
- 采样文本的信息密度反映写作质量
- 总长度反映覆盖潜力（但长≠好）
- 首、中、尾主题不同说明覆盖面广

分数档位：
- **8-10**：深入全面，多角度分析
- **5-7**：覆盖合理但不够深入
- **2-4**：浅层覆盖，只有概要
- **0-1**：极少内容或与主题无关

## 评分要求

1. 像专业人士一样，从你读到的内容本身做判断
2. 评分要有区分度
3. reason 字段用简短中文说明关键依据
4. 分数必须是整数，在对应满分范围内

## 领域标签

根据页面内容，输出 2-4 个短领域标签到 domain_tags，描述这个来源擅长什么领域。
- 用中文短词（如"医学"、"人工智能"、"气候科学"）
- 尽量具体（"多模态模型"优于"大模型"）
- 反映页面内容本身的领域
"""


# ── 元数据提取（仅保留 LLM 从正文中无法获得的信息）──────────────


def _parse_date_age(date_str: str) -> Optional[int]:
    """尝试解析日期字符串，返回距今天数。"""
    if not date_str or not date_str.strip():
        return None

    formats = [
        "%Y-%m-%d", "%Y/%m/%d", "%Y年%m月%d日",
        "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ",
        "%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%d %b %Y",
        "%Y.%m.%d", "%m/%d/%Y",
    ]
    date_clean = date_str.strip()
    for fmt in formats:
        try:
            parsed = datetime.strptime(date_clean, fmt)
            return max(0, (datetime.now() - parsed).days)
        except ValueError:
            continue

    year_match = re.search(r'20[12]\d', date_clean)
    if year_match:
        year = int(year_match.group())
        return max(0, (datetime.now().year - year) * 365)
    return None


def _is_generic_homepage(title: str, content_length: int) -> bool:
    """检测是否为通用主页（标题为通用词且内容极短）。"""
    if not title or content_length > 1500:
        return False
    return title.strip().lower() in _GENERIC_TITLE_KEYWORDS


# ── 构造 LLM 评分输入 ──────────────────────────────────────────


def _sample_body_text(body_text: str, budget: int = 2000) -> str:
    """对正文进行采样：首 40% + 中间 20% + 尾 40%。"""
    if not body_text or len(body_text) <= budget:
        return body_text or ""
    head_size = int(budget * 0.4)
    mid_size = int(budget * 0.2)
    tail_size = budget - head_size - mid_size
    head = body_text[:head_size]
    mid_start = (len(body_text) - mid_size) // 2
    mid = body_text[mid_start:mid_start + mid_size]
    tail = body_text[-tail_size:]
    return f"{head}\n\n[...正文中部节选...]\n\n{mid}\n\n[...正文尾部节选...]\n\n{tail}"


def _build_scoring_user_message(page: PageContent, subclaim: str) -> str:
    """构造评分的 user message。

    只给 LLM 它从正文中无法获得的客观信息，不给正则提取的信号。
    """
    body_preview = _sample_body_text(page.body_text or "")

    # 日期：标注为自动提取、仅供参考
    if page.date and page.date.strip():
        date_line = f"- 发布日期（自动提取，仅供参考）: {page.date}"
    else:
        date_line = "- 发布日期: 未提取到（请从正文内容推断时效性）"

    meta_lines = [
        f"- 作者: {page.author if page.author and page.author.strip() else '未提取到'}",
        date_line,
        f"- 机构/站点: {page.institution if page.institution and page.institution.strip() else '未提取到'}",
        f"- 正文总长度: {len(page.body_text) if page.body_text else 0} 字符",
    ]

    return (
        f"## 待评分页面\n\n"
        f"**子观点**: {subclaim}\n"
        f"**URL**: {page.url}\n"
        f"**域名**: {page.domain}\n"
        f"**标题**: {page.title}\n\n"
        f"## 页面元数据\n\n"
        + "\n".join(meta_lines)
        + f"\n\n## 页面正文（采样约 2000 字符）\n\n{body_preview}"
    )


# ── 最小规则层（只处理 LLM 物理上无法感知的情况）─────────────────


def _apply_content_guards(
    dimensions: dict[str, int],
    content_length: int,
    is_homepage: bool,
) -> dict[str, str]:
    """极简规则修正。只在 LLM 无法感知的极端情况下介入。"""
    corrections: dict[str, str] = {}

    if is_homepage:
        if dimensions.get("coverage", 0) > 3:
            corrections["coverage"] = f"通用主页（内容仅{content_length}字符），覆盖度上限3分"
            dimensions["coverage"] = 3
        if dimensions.get("accuracy", 0) > 15:
            corrections["accuracy"] = "通用主页，准确性上限15分"
            dimensions["accuracy"] = 15
    elif content_length < 500 and dimensions.get("coverage", 0) > 4:
        corrections["coverage"] = f"正文仅{content_length}字符，覆盖度上限4分"
        dimensions["coverage"] = 4

    return corrections


# ── LLM 评分主函数 ─────────────────────────────────────────────


def _clamp(value: int, min_val: int, max_val: int) -> int:
    return max(min_val, min(max_val, value))


async def score_page(
    page: PageContent,
    subclaim: str,
    quality_threshold: Optional[int] = None,
) -> ScoreResult:
    """对页面进行五维质量评分。

    核心：让 LLM 像专业评审一样从内容判断，不用正则信号干预。
    规则层只在通用主页、正文极短时介入。
    """
    threshold = quality_threshold or cfg.agent.quality_threshold
    content_length = len(page.body_text) if page.body_text else 0
    is_homepage = _is_generic_homepage(page.title, content_length)

    import time as _time
    scoring_prompt = _SCORING_SYSTEM_PROMPT_TEMPLATE.format(today=_time.strftime('%Y-%m-%d %a'))
    user_message = _build_scoring_user_message(page, subclaim)

    try:
        llm_result: LLMScoreOutput = await call_scoring_llm_json(
            system_prompt=scoring_prompt,
            user_message=user_message,
            response_model=LLMScoreOutput,
            temperature=0.1,
            max_tokens=1024,
        )
    except Exception as e:
        logger.warning("LLM 评分调用失败 (%s): %s，使用兜底评分", page.url[:60], e)
        return _fallback_score(page, threshold)

    raw_scores = {
        "authority": _clamp(llm_result.authority.score, 0, 30),
        "accuracy": _clamp(llm_result.accuracy.score, 0, 30),
        "purpose": _clamp(llm_result.purpose.score, 0, 20),
        "timeliness": _clamp(llm_result.timeliness.score, 0, 10),
        "coverage": _clamp(llm_result.coverage.score, 0, 10),
    }
    reasons = {
        "authority": llm_result.authority.reason,
        "accuracy": llm_result.accuracy.reason,
        "purpose": llm_result.purpose.reason,
        "timeliness": llm_result.timeliness.reason,
        "coverage": llm_result.coverage.reason,
    }
    max_scores = {"authority": 30, "accuracy": 30, "purpose": 20, "timeliness": 10, "coverage": 10}

    # 极简规则修正
    corrections = _apply_content_guards(raw_scores, content_length, is_homepage)

    dimensions: list[DimensionScore] = []
    total = 0.0
    explanation_per_dim: dict[str, str] = {}

    for name in ["authority", "accuracy", "purpose", "timeliness", "coverage"]:
        score_val = raw_scores[name]
        total += score_val
        reason = reasons[name]
        if name in corrections and corrections[name]:
            reason = f"{reason}（规则修正：{corrections[name]}）"
        label = DIMENSION_LABELS.get(name, name)
        dimensions.append(DimensionScore(
            name=name, label=label,
            score=float(score_val), max_score=float(max_scores[name]),
            deduction_reason=reason,
        ))
        explanation_per_dim[name] = f"{label} {int(score_val)}/{int(max_scores[name])}：{reason}"

    passes = total >= threshold
    explanation_parts = [f"{d.label}: {int(d.score)}/{int(d.max_score)}" for d in dimensions]
    explanation = f"总分 {total:.0f}/{100} | " + " | ".join(explanation_parts)

    result = ScoreResult(
        total_score=total,
        dimensions=dimensions,
        passes_threshold=passes,
        explanation=explanation,
        explanation_per_dimension=explanation_per_dim,
        domain_tags=llm_result.domain_tags or [],
    )

    correction_log = ""
    if corrections:
        correction_log = " | 规则修正: " + "; ".join(f"{k}={v}" for k, v in corrections.items() if v)

    logger.info(
        "页面评分: %s → %.0f/100 %s | %s%s",
        page.domain, total, "PASS" if passes else "FAIL",
        page.title[:40], correction_log,
    )
    return result


def _fallback_score(page: PageContent, threshold: int) -> ScoreResult:
    """LLM 评分失败时的保守兜底评分。"""
    content_length = len(page.body_text) if page.body_text else 0
    has_author = bool(page.author and page.author.strip())

    authority = 16 if has_author else 10
    accuracy = 14
    purpose = 12
    timeliness = 4  # 不确定
    coverage = 5 if content_length >= 1000 else (3 if content_length >= 200 else 1)

    if _is_generic_homepage(page.title, content_length):
        coverage = 2
        accuracy = 10

    total = float(authority + accuracy + purpose + timeliness + coverage)
    reason = "规则兜底评分（LLM 不可用）"

    scores = {"authority": (authority, 30), "accuracy": (accuracy, 30),
              "purpose": (purpose, 20), "timeliness": (timeliness, 10), "coverage": (coverage, 10)}
    dimensions = []
    explanation_per_dim: dict[str, str] = {}
    for name, (sv, mv) in scores.items():
        label = DIMENSION_LABELS.get(name, name)
        dimensions.append(DimensionScore(
            name=name, label=label, score=float(sv), max_score=float(mv), deduction_reason=reason,
        ))
        explanation_per_dim[name] = f"{label} {sv}/{mv}：{reason}"

    return ScoreResult(
        total_score=total, dimensions=dimensions,
        passes_threshold=total >= threshold,
        explanation=f"兜底评分 {total:.0f}/100", explanation_per_dimension=explanation_per_dim,
    )


# ── 兼容旧接口（其他模块可能 import 这些）──────────────────────


class QualitySignals(BaseModel):
    """保留用于向后兼容，不再用于评分。"""
    has_author: bool = False
    has_date: bool = False
    has_institution: bool = False
    domain_category: str = "general"
    reference_count: int = 0
    content_length: int = 0
    date_age_days: Optional[int] = None
    is_generic_homepage: bool = False
    data_density: int = 0
    attribution_count: int = 0
    promotional_signal_count: int = 0
    date_suspicious: bool = False


def extract_quality_signals(page: PageContent) -> QualitySignals:
    """保留用于向后兼容。"""
    return QualitySignals(
        has_author=bool(page.author and page.author.strip()),
        has_date=bool(page.date and page.date.strip()),
        content_length=len(page.body_text) if page.body_text else 0,
    )
