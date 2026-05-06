"""
Deep Truth Search — 核心数据模型

所有 agent、tool、scoring 模块共用的类型定义。
基于 masterplan.md 概念数据模型设计。
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ── 枚举类型 ──────────────────────────────────────────────────


class TaskStatus(str, Enum):
    """查询任务状态"""
    PENDING = "pending"
    SEARCHING = "searching"
    ANALYZING = "analyzing"
    DONE = "done"
    ERROR = "error"


class SourceLevel(str, Enum):
    """信息源分层等级"""
    ELITE = "elite"
    TRUSTED = "trusted"
    VERIFIED = "verified"
    TRIAL = "trial"
    DEPRECATED = "deprecated"


class PageType(str, Enum):
    """页面内容类型"""
    NEWS = "news"
    ACADEMIC = "academic"
    BLOG = "blog"
    OFFICIAL = "official"
    WIKI = "wiki"
    OTHER = "other"


# ── 搜索相关模型 ──────────────────────────────────────────────


class SearchResult(BaseModel):
    """搜索工具返回的单条候选结果"""
    title: str = Field(default="", description="搜索结果标题")
    snippet: str = Field(default="", description="搜索结果摘要")
    url: str = Field(description="搜索结果 URL")
    relevance_score: Optional[float] = Field(default=None, description="轻评估相关性分数（P2 阶段使用）")


class PageContent(BaseModel):
    """Visit 工具提取的页面内容"""
    url: str = Field(description="页面 URL")
    title: str = Field(default="", description="页面标题")
    body_text: str = Field(default="", description="页面正文")
    author: str = Field(default="", description="作者")
    date: str = Field(default="", description="发布日期")
    domain: str = Field(default="", description="域名")
    institution: str = Field(default="", description="机构")
    page_type: PageType = Field(default=PageType.OTHER, description="页面类型")
    references: list[str] = Field(default_factory=list, description="引用和数据来源")


# ── 评分相关模型 ──────────────────────────────────────────────


class DimensionScore(BaseModel):
    """单个评分维度"""
    name: str = Field(description="维度名称：authority / accuracy / purpose / timeliness / coverage")
    label: str = Field(default="", description="维度中文标签（前端展示用）")
    score: float = Field(default=0.0, description="得分")
    max_score: float = Field(default=0.0, description="该维度满分")
    deduction_reason: str = Field(default="", description="评分理由")


# 维度名称到中文标签的映射
DIMENSION_LABELS: dict[str, str] = {
    "authority": "权威性",
    "accuracy": "准确性",
    "purpose": "目的性",
    "timeliness": "时效性",
    "coverage": "覆盖度",
}


class ScoreResult(BaseModel):
    """五维评分结果"""
    total_score: float = Field(default=0.0, description="总分")
    dimensions: list[DimensionScore] = Field(default_factory=list, description="各维度评分")
    passes_threshold: bool = Field(default=False, description="是否达到准入阈值")
    explanation: str = Field(default="", description="评分总览（如 '总分 85/100 | authority: 28/30 | ...'）")
    explanation_per_dimension: dict[str, str] = Field(
        default_factory=dict,
        description="逐维度评分解释，key 为维度名（如 'authority'），value 为中文解释文本",
    )
    domain_tags: list[str] = Field(
        default_factory=list,
        description="LLM 从页面内容和子观点推导的领域标签（如 ['医学', 'mRNA疫苗', '临床试验']）",
    )


# ── 证据相关模型 ──────────────────────────────────────────────


class EvidenceItem(BaseModel):
    """一条可展示的论据"""
    claim: str = Field(description="所属论点")
    evidence_text: str = Field(description="证据摘要")
    source_url: str = Field(description="来源 URL")
    source_title: str = Field(default="", description="来源标题")
    source_domain: str = Field(default="", description="来源域名")
    score: Optional[ScoreResult] = Field(default=None, description="五维评分结果")
    source_level: SourceLevel = Field(default=SourceLevel.TRIAL, description="来源等级")
    extracted_at: datetime = Field(default_factory=datetime.now, description="提取时间")


# ── 子观点模型 ────────────────────────────────────────────────


class SubClaim(BaseModel):
    """拆解后的子观点"""
    text: str = Field(description="子观点文本")
    topic: str = Field(default="", description="主题标签")
    evidences: list[EvidenceItem] = Field(default_factory=list, description="该子观点下累积的证据")


# ── Search Agent 接口模型 ─────────────────────────────────────


class SearchEvidenceInput(BaseModel):
    """Search Agent Facade 输入参数（Main Agent → Search Agent）"""
    subclaim: str = Field(description="当前子观点")
    topic_context: str = Field(default="", description="主题上下文")
    preferred_sources: list[str] = Field(default_factory=list, description="来自披露窗口的优先来源")
    search_budget: int = Field(default=5, description="本轮搜索预算")
    quality_threshold: int = Field(default=60, description="质量准入阈值")


class SearchEvidenceResult(BaseModel):
    """Search Agent 返回给 Main Agent 的完整结果"""
    subclaim: str = Field(description="对应的子观点")
    evidence_items: list[EvidenceItem] = Field(default_factory=list, description="通过筛选的证据列表")
    accepted_sources: list[str] = Field(default_factory=list, description="通过阈值的来源 URL")
    rejected_sources: list[str] = Field(default_factory=list, description="未通过阈值的来源 URL")
    search_trace: list[str] = Field(default_factory=list, description="搜索行为记录")
    coverage_notes: str = Field(default="", description="本轮覆盖到的内容说明")
    execution_summary: str = Field(default="", description="执行摘要")


# ── 查询任务模型 ──────────────────────────────────────────────


class ClaimResult(BaseModel):
    """单个论点及其证据列表（最终输出用）"""
    claim_title: str = Field(description="论点标题")
    evidences: list[EvidenceItem] = Field(default_factory=list, description="论据列表")


class QueryTask(BaseModel):
    """一次完整的用户查询任务"""
    query: str = Field(description="用户原始观点")
    task_id: str = Field(default="", description="任务 ID")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="任务状态")
    subclaims: list[SubClaim] = Field(default_factory=list, description="拆解后的子观点")
    results: list[ClaimResult] = Field(default_factory=list, description="最终论点+论据结果")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    completed_at: Optional[datetime] = Field(default=None, description="完成时间")


# ── 最终输出模型（面向 API / 前端）─────────────────────────────


class ResearchOutput(BaseModel):
    """面向前端渲染的最终输出"""
    query: str = Field(description="用户原始观点")
    claims: list[ClaimResult] = Field(default_factory=list, description="论点+论据列表")
    total_evidences: int = Field(default=0, description="证据总数")
    search_rounds: int = Field(default=0, description="搜索轮次")
    completed_at: Optional[datetime] = Field(default=None, description="完成时间")


# ── 信息源长期记录模型（P4 阶段使用，此处预定义）─────────────


class ExpertiseTag(BaseModel):
    """来源在某个领域的专长记录（随使用逐步积累、细化、衰退）"""
    avg_score: float = Field(default=0.0, description="该标签关联的平均评分")
    count: int = Field(default=0, description="该标签被关联的次数")
    last_seen: Optional[datetime] = Field(default=None, description="最近一次关联时间")


class SourceProfile(BaseModel):
    """信息源的长期表现记录（永久保存，不删除）"""
    domain: str = Field(description="域名（主键）")
    url_pattern: str = Field(default="", description="URL 匹配模式")
    topics: list[str] = Field(default_factory=list, description="相关主题列表（向后兼容）")
    expertise_tags: dict[str, ExpertiseTag] = Field(
        default_factory=dict,
        description="领域专长标签：tag_name → ExpertiseTag（如 {'医学': {avg_score: 85, count: 5}}）",
    )
    current_level: SourceLevel = Field(default=SourceLevel.TRIAL, description="当前层级")
    historical_scores: list[float] = Field(default_factory=list, description="历史评分记录")
    accepted_count: int = Field(default=0, description="被接受次数（通过质量阈值）")
    rejected_count: int = Field(default=0, description="被拒绝次数（未通过质量阈值）")
    first_seen_at: Optional[datetime] = Field(default=None, description="首次发现时间")
    last_seen_at: Optional[datetime] = Field(default=None, description="最近出现时间")
    failure_count: int = Field(default=0, description="连续失效次数（404/超时）")
    failure_flags: list[str] = Field(default_factory=list, description="失效记录（如 '2026-04-20: 404'）")
    avg_score: float = Field(default=0.0, description="历史评分均值（缓存，随 historical_scores 更新）")


class DisclosureWindow(BaseModel):
    """披露窗口：当前主题下可向 Agent 披露的来源列表及分层统计"""
    topic: str = Field(description="查询主题")
    window_size: int = Field(description="窗口容量上限")
    sources: list["SourceProfile"] = Field(default_factory=list, description="按优先级排序的来源列表")
    visible_elite: int = Field(default=0, description="窗口中 Elite 来源数量")
    visible_trusted: int = Field(default=0, description="窗口中 Trusted 来源数量")
    visible_verified: int = Field(default=0, description="窗口中 Verified 来源数量")
    visible_trial: int = Field(default=0, description="窗口中 Trial 来源数量")

    @property
    def total(self) -> int:
        return len(self.sources)

    @property
    def domains(self) -> list[str]:
        """返回窗口中所有来源的域名列表（传给 Search Agent 用）"""
        return [s.domain for s in self.sources]


# ── 会话模型（P5-K1）────────────────────────────────────────────


class SessionMessage(BaseModel):
    """会话中的一条消息（用户查询或 Agent 结果）"""
    role: Literal["user", "agent"] = Field(description="消息角色")
    content: str = Field(description="user: 查询文本; agent: ResearchOutput JSON")
    timestamp: datetime = Field(default_factory=datetime.now)
    task_id: str = Field(default="", description="agent 消息关联的 task_id")


class Session(BaseModel):
    """一个会话，包含多轮查询/结果"""
    session_id: str = Field(description="会话唯一 ID")
    title: str = Field(default="", description="会话标题，首条查询自动截取")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    messages: list[SessionMessage] = Field(default_factory=list)


# ── LLM Tool Calling 模型 ──────────────────────────────────────


class ToolCallInfo(BaseModel):
    """LLM 返回的单个工具调用"""
    id: str = Field(description="工具调用 ID")
    function_name: str = Field(description="要调用的函数名")
    arguments: dict = Field(default_factory=dict, description="解析后的参数")


class LLMToolResponse(BaseModel):
    """LLM 带工具调用的完整响应"""
    content: Optional[str] = Field(default=None, description="文本内容")
    tool_calls: list[ToolCallInfo] = Field(default_factory=list, description="工具调用列表")
