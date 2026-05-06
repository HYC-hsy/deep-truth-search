"""
Deep Truth Search — 信息源记忆持久化

Repository 模式：抽象接口隔离存储细节，MVP 阶段用 JSON 文件实现。
设计原则：
- 永不删除来源记录，只做分层管理（delete 仅供测试/维护）
- 只记录实际被访问和评分过的来源
- 存储格式人类可读（JSON）

合并两版实现的优点：
- 内存缓存 + 原子写入（D 盘）
- Pydantic 原生序列化 model_dump_json / model_validate（D 盘）
- 单条记录容错加载（D 盘）
- 真正的懒加载单例（D 盘）
- threading.Lock 线程安全（E 盘）
- batch_save / delete 扩展接口（E 盘）
- metadata 层：任务计数器、最后更新时间（E 盘）
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Optional

from models import DisclosureWindow, ExpertiseTag, SourceLevel, SourceProfile

logger = logging.getLogger(__name__)


# ── 抽象接口 ──────────────────────────────────────────────────


class SourceRepository(ABC):
    """信息源存储抽象接口，后续可替换为 SQLite 等实现"""

    @abstractmethod
    def save(self, profile: SourceProfile) -> None:
        """保存或更新一个信息源记录（以 domain 为主键）"""

    @abstractmethod
    def find_by_domain(self, domain: str) -> Optional[SourceProfile]:
        """根据域名查找信息源"""

    @abstractmethod
    def find_by_topic(self, topic: str) -> list[SourceProfile]:
        """根据主题查找相关信息源（模糊匹配）"""

    @abstractmethod
    def find_by_level(self, level: SourceLevel) -> list[SourceProfile]:
        """根据层级查找信息源"""

    @abstractmethod
    def find_all(self) -> list[SourceProfile]:
        """返回所有信息源"""

    @abstractmethod
    def delete(self, domain: str) -> bool:
        """删除指定域名的信息源（仅用于测试/维护，正常流程不删除）"""

    @abstractmethod
    def count(self) -> int:
        """返回信息源总数"""


# ── JSON 文件实现 ─────────────────────────────────────────────


class JsonSourceRepository(SourceRepository):
    """基于 JSON 文件的信息源存储实现

    存储结构：
    {
        "sources": { "domain.com": { ...SourceProfile fields... }, ... },
        "metadata": { "total_tasks": 0, "last_updated": "..." }
    }

    架构特点：
    - 内存缓存：启动时一次性加载，读操作零 IO
    - 原子写入：先写 .tmp 文件再 rename，防止进程中断写出半截文件
    - 线程安全：Lock 保护所有缓存和磁盘操作
    - 单条容错：加载时损坏的记录被跳过，不影响其他记录
    - Pydantic 原生序列化：model_dump_json / model_validate
    """

    def __init__(self, storage_dir: Path) -> None:
        self._dir = storage_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._file = self._dir / "sources.json"
        self._lock = Lock()
        self._cache: dict[str, SourceProfile] = {}
        self._topic_index: dict[str, set[str]] = {}  # topic_lower → {domain, ...}
        self._tag_index: dict[str, set[str]] = {}    # expertise_tag_lower → {domain, ...}
        self._metadata: dict = {"total_tasks": 0, "last_updated": None}
        self._load()

    def _load(self) -> None:
        """从 JSON 文件加载全部信息源到内存缓存"""
        if not self._file.exists():
            self._cache = {}
            self._metadata = {"total_tasks": 0, "last_updated": None}
            return

        try:
            raw = json.loads(self._file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.error("加载信息源文件失败 %s: %s", self._file, e)
            self._cache = {}
            return

        # 加载 metadata
        self._metadata = raw.get("metadata", {"total_tasks": 0, "last_updated": None})

        # 加载 sources，单条容错
        sources = raw.get("sources", {})
        self._cache = {}
        for domain, data in sources.items():
            try:
                self._cache[domain] = SourceProfile.model_validate(data)
            except Exception as e:
                logger.warning("跳过损坏的信息源记录 %s: %s", domain, e)

        self._rebuild_topic_index()
        logger.info("加载 %d 个信息源记录", len(self._cache))

    def _rebuild_topic_index(self) -> None:
        """从 _cache 全量重建主题和标签倒排索引"""
        self._topic_index = {}
        self._tag_index = {}
        for domain, profile in self._cache.items():
            for topic in profile.topics:
                key = topic.lower()
                if key not in self._topic_index:
                    self._topic_index[key] = set()
                self._topic_index[key].add(domain)
            for tag in profile.expertise_tags:
                key = tag.lower()
                if key not in self._tag_index:
                    self._tag_index[key] = set()
                self._tag_index[key].add(domain)

    def _update_topic_index(self, profile: SourceProfile) -> None:
        """增量更新单个来源的主题和标签索引"""
        for topic in profile.topics:
            key = topic.lower()
            if key not in self._topic_index:
                self._topic_index[key] = set()
            self._topic_index[key].add(profile.domain)
        for tag in profile.expertise_tags:
            key = tag.lower()
            if key not in self._tag_index:
                self._tag_index[key] = set()
            self._tag_index[key].add(profile.domain)

    def _persist(self) -> None:
        """原子写入：先写临时文件，成功后重命名覆盖"""
        self._metadata["last_updated"] = datetime.now().isoformat()

        serialized = {
            "sources": {},
            "metadata": self._metadata,
        }
        for domain, profile in self._cache.items():
            serialized["sources"][domain] = json.loads(profile.model_dump_json())

        tmp = self._file.with_suffix(".tmp")
        try:
            tmp.write_text(
                json.dumps(serialized, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp.replace(self._file)
        except OSError as e:
            logger.error("持久化信息源失败: %s", e)
            if tmp.exists():
                tmp.unlink(missing_ok=True)

    # ── 接口实现 ──────────────────────────────────────────────────

    def save(self, profile: SourceProfile) -> None:
        with self._lock:
            self._cache[profile.domain] = profile
            self._update_topic_index(profile)
            self._persist()
        logger.debug("保存信息源: %s (level=%s)", profile.domain, profile.current_level.value)

    def find_by_domain(self, domain: str) -> Optional[SourceProfile]:
        with self._lock:
            return self._cache.get(domain)

    @staticmethod
    def _topic_match(query: str, indexed: str, min_gram: int = 3) -> bool:
        """主题匹配：双向子串 + n-gram 重叠。

        除了经典的双向子串匹配外，还检查两个字符串之间是否有
        长度 >= min_gram 的公共子串。这解决中文长句 topic 之间
        因为措辞不同而无法子串匹配的问题。
        例如 "mRNA疫苗在癌症治疗中的最新进展" 和 "mRNA技术在肿瘤免疫治疗中的突破"
        虽然互不是子串，但共享 "mRNA"、"治疗中的" 等片段。
        """
        if query in indexed or indexed in query:
            return True
        # n-gram 重叠：提取 query 的所有 min_gram 长度子串，检查是否出现在 indexed 中
        if len(query) >= min_gram and len(indexed) >= min_gram:
            for i in range(len(query) - min_gram + 1):
                gram = query[i:i + min_gram]
                if gram in indexed:
                    return True
        return False

    def find_by_topic(self, topic: str) -> list[SourceProfile]:
        """按主题查找来源，使用倒排索引 + 子串/n-gram 混合匹配"""
        topic_lower = topic.lower()
        matched_domains: set[str] = set()
        with self._lock:
            for indexed_topic, domains in self._topic_index.items():
                if self._topic_match(topic_lower, indexed_topic):
                    matched_domains.update(domains)
            return [self._cache[d] for d in matched_domains if d in self._cache]

    def find_by_topic_and_level(
        self, topic: str, levels: list[SourceLevel],
    ) -> list[SourceProfile]:
        """按主题 + 层级组合查询（披露窗口用），按 avg_score 降序"""
        topic_matches = self.find_by_topic(topic)
        level_set = set(levels)
        filtered = [p for p in topic_matches if p.current_level in level_set]
        filtered.sort(key=lambda p: p.avg_score, reverse=True)
        return filtered

    def find_by_expertise(
        self,
        query: str,
        exclude_deprecated: bool = True,
    ) -> list[SourceProfile]:
        """按领域专长标签匹配来源（披露窗口主入口）。

        从 query 中提取片段，在 expertise_tags 的短标签中查找匹配。
        比 find_by_topic 更准确：短标签（"医学"）比长句子更容易被 query 命中。
        按匹配标签的加权分数排序（匹配越多、标签分数越高的来源排越前）。

        Fallback：如果标签匹配结果为空，降级到 find_by_topic。
        """
        query_lower = query.lower()
        # domain → 匹配到的标签加权分数之和
        domain_relevance: dict[str, float] = {}

        with self._lock:
            for tag_key, domains in self._tag_index.items():
                # 双向子串 + n-gram 匹配
                if self._topic_match(query_lower, tag_key):
                    for d in domains:
                        if d not in self._cache:
                            continue
                        profile = self._cache[d]
                        if exclude_deprecated and profile.current_level == SourceLevel.DEPRECATED:
                            continue
                        # 用匹配到的标签的 avg_score 作为权重
                        tag_obj = profile.expertise_tags.get(tag_key)
                        tag_score = tag_obj.avg_score if tag_obj else profile.avg_score
                        domain_relevance[d] = domain_relevance.get(d, 0) + tag_score

        if domain_relevance:
            # 按加权分数降序
            sorted_domains = sorted(domain_relevance.keys(), key=lambda d: -domain_relevance[d])
            results = [self._cache[d] for d in sorted_domains if d in self._cache]
            logger.debug("标签匹配: query='%s' → %d 个来源", query[:40], len(results))
            return results

        # Fallback: 标签匹配无结果，降级到旧的 topic 匹配
        fallback = self.find_by_topic(query)
        if exclude_deprecated:
            fallback = [s for s in fallback if s.current_level != SourceLevel.DEPRECATED]
        if fallback:
            logger.debug("标签匹配无结果，降级 topic 匹配: query='%s' → %d 个来源", query[:40], len(fallback))
        return fallback

    def find_by_level(self, level: SourceLevel) -> list[SourceProfile]:
        with self._lock:
            return [p for p in self._cache.values() if p.current_level == level]

    def find_all(self) -> list[SourceProfile]:
        with self._lock:
            return list(self._cache.values())

    def delete(self, domain: str) -> bool:
        with self._lock:
            if domain in self._cache:
                # 清理倒排索引
                profile = self._cache[domain]
                for topic in profile.topics:
                    key = topic.lower()
                    if key in self._topic_index:
                        self._topic_index[key].discard(domain)
                        if not self._topic_index[key]:
                            del self._topic_index[key]
                del self._cache[domain]
                self._persist()
                logger.debug("删除信息源: %s", domain)
                return True
        return False

    def count(self) -> int:
        with self._lock:
            return len(self._cache)

    # ── 批量操作 ──────────────────────────────────────────────────

    def batch_save(self, profiles: list[SourceProfile]) -> None:
        """批量保存信息源（单次磁盘写入）"""
        with self._lock:
            for profile in profiles:
                self._cache[profile.domain] = profile
                self._update_topic_index(profile)
            self._persist()
        logger.debug("批量保存 %d 个信息源", len(profiles))

    # ── 元数据操作（供周期整合使用） ─────────────────────────────

    def get_total_tasks(self) -> int:
        """获取已完成的总任务数"""
        with self._lock:
            return self._metadata.get("total_tasks", 0)

    def increment_task_count(self) -> int:
        """任务完成时递增计数器，返回新值"""
        with self._lock:
            count = self._metadata.get("total_tasks", 0) + 1
            self._metadata["total_tasks"] = count
            self._persist()
        return count


# ── 主题化来源索引 ───────────────────────────────────────────

# 层级排序权重：Elite 最优先，Deprecated 最低
_LEVEL_PRIORITY = {
    SourceLevel.ELITE: 0,
    SourceLevel.TRUSTED: 1,
    SourceLevel.VERIFIED: 2,
    SourceLevel.TRIAL: 3,
    SourceLevel.DEPRECATED: 4,
}


def get_sources_by_topic(
    topic: str,
    repo: Optional["SourceRepository"] = None,
    exclude_deprecated: bool = True,
) -> list[SourceProfile]:
    """按主题检索来源，按层级优先、评分降序排列（P4-H7）。

    排序规则：
    1. 层级越高越靠前（Elite > Trusted > Verified > Trial）
    2. 同层级内按 avg_score 降序

    Args:
        topic: 主题关键词（模糊匹配）
        repo: 仓库实例，默认使用全局单例
        exclude_deprecated: 是否排除 Deprecated 来源（默认 True）

    Returns:
        按优先级排序的来源列表
    """
    if repo is None:
        repo = get_source_repository()

    sources = repo.find_by_topic(topic)

    if exclude_deprecated:
        sources = [s for s in sources if s.current_level != SourceLevel.DEPRECATED]

    sources.sort(key=lambda s: (_LEVEL_PRIORITY.get(s.current_level, 99), -s.avg_score))
    return sources


def get_top_sources(
    topic: str,
    limit: int = 20,
    repo: Optional["SourceRepository"] = None,
) -> list[SourceProfile]:
    """获取指定主题下的 Top N 高价值来源（P4-H7，为 Phase 5 披露窗口准备）。

    按 Elite → Trusted → Verified → Trial 优先级填充，直到达到 limit。
    Deprecated 来源永不返回。

    Args:
        topic: 主题关键词
        limit: 返回数量上限
        repo: 仓库实例

    Returns:
        最多 limit 个高价值来源，按优先级排序
    """
    sources = get_sources_by_topic(topic, repo=repo, exclude_deprecated=True)
    return sources[:limit]


# ── 披露窗口（Phase 5） ─────────────────────────────────────────

# 披露窗口的填充层级顺序（Deprecated 永不进入）
_DISCLOSURE_TIERS = [
    SourceLevel.ELITE,
    SourceLevel.TRUSTED,
    SourceLevel.VERIFIED,
    SourceLevel.TRIAL,
]

# 披露窗口缓存：topic_lower → DisclosureWindow（层级变动时自动失效，P5-I4）
_disclosure_cache: dict[str, DisclosureWindow] = {}


def invalidate_disclosure_cache(topic: Optional[str] = None) -> None:
    """使披露窗口缓存失效（P5-I4）。

    当来源层级变动时调用，确保下次获取窗口时反映最新层级。

    Args:
        topic: 指定主题失效；None 则清空全部缓存
    """
    if topic is None:
        _disclosure_cache.clear()
        logger.debug("披露窗口缓存已全部清空")
    else:
        # 该 topic 可能匹配多个缓存 key，使用同样的 n-gram 匹配逻辑
        topic_lower = topic.lower()
        keys_to_remove = [
            k for k in _disclosure_cache
            if JsonSourceRepository._topic_match(topic_lower, k)
        ]
        for k in keys_to_remove:
            del _disclosure_cache[k]
        if keys_to_remove:
            logger.debug("披露窗口缓存已失效: %s", keys_to_remove)


def get_disclosure_window(
    topic: str,
    window_size: Optional[int] = None,
    max_trial_ratio: float = 0.15,
    repo: Optional["SourceRepository"] = None,
) -> DisclosureWindow:
    """计算指定主题的披露窗口（P5-I1）。

    披露窗口是 Main Agent 每次任务可见的信息源子集。
    按 Elite → Trusted → Verified → Trial 优先级逐层填充，
    直到达到 window_size 或所有可用来源已用完。
    Deprecated 来源永不进入窗口。

    填充规则：
    1. 优先用 Elite 填满窗口
    2. Elite 不够时补 Trusted
    3. Trusted 还不够时补 Verified
    4. 仍不足时少量补充 Trial（不超过窗口的 max_trial_ratio）
    5. Deprecated 与低质量来源永不披露

    每层内部按 avg_score 降序排列，确保同层中最优来源优先入窗。

    早期（高层来源少）：窗口混合 Trusted/Verified/Trial，系统仍可正常工作。
    后期（Elite 积累充足）：窗口主要被 Elite 占满，搜索更快更准。

    Args:
        topic: 主题关键词（支持模糊匹配）
        window_size: 窗口容量上限，默认从 config 读取（DISCLOSURE_WINDOW_SIZE）
        max_trial_ratio: Trial 来源最多占窗口的比例，默认 0.15
        repo: 仓库实例，默认使用全局单例

    Returns:
        DisclosureWindow 包含排序后的来源列表和各层统计
    """
    if repo is None:
        repo = get_source_repository()

    if window_size is None:
        from config import cfg
        window_size = cfg.memory.disclosure_window

    # P5-I4：缓存命中检查（仅默认参数时使用缓存）
    cache_key = topic.lower()
    if max_trial_ratio == 0.15 and cache_key in _disclosure_cache:
        cached = _disclosure_cache[cache_key]
        if cached.window_size == window_size:
            logger.debug("披露窗口命中缓存: topic='%s'", topic)
            return cached

    max_trial = max(1, int(window_size * max_trial_ratio))

    # 用标签匹配获取候选来源（已按相关度排序，已排除 Deprecated）
    all_candidates = repo.find_by_expertise(topic, exclude_deprecated=True)

    # 按层级分组，保留标签匹配的顺序
    tier_buckets: dict[SourceLevel, list[SourceProfile]] = {level: [] for level in _DISCLOSURE_TIERS}
    for src in all_candidates:
        if src.current_level in tier_buckets:
            tier_buckets[src.current_level].append(src)

    # 逐层填充
    window_sources: list[SourceProfile] = []
    tier_counts = {level: 0 for level in _DISCLOSURE_TIERS}
    already_in: set[str] = set()
    remaining = window_size

    for level in _DISCLOSURE_TIERS:
        if remaining <= 0:
            break

        # Trial 层受额外配额限制
        if level == SourceLevel.TRIAL:
            remaining = min(remaining, max_trial)

        candidates = [c for c in tier_buckets[level] if c.domain not in already_in]

        selected = candidates[:remaining]
        window_sources.extend(selected)
        already_in.update(s.domain for s in selected)
        tier_counts[level] = len(selected)
        remaining -= len(selected)

    window = DisclosureWindow(
        topic=topic,
        window_size=window_size,
        sources=window_sources,
        visible_elite=tier_counts[SourceLevel.ELITE],
        visible_trusted=tier_counts[SourceLevel.TRUSTED],
        visible_verified=tier_counts[SourceLevel.VERIFIED],
        visible_trial=tier_counts[SourceLevel.TRIAL],
    )

    # P5-I4：写入缓存
    _disclosure_cache[cache_key] = window

    logger.info(
        "披露窗口生成: topic='%s' | 总数=%d/%d | Elite=%d Trusted=%d Verified=%d Trial=%d",
        topic, window.total, window_size,
        window.visible_elite, window.visible_trusted,
        window.visible_verified, window.visible_trial,
    )

    return window


# ── 五层分级逻辑 ─────────────────────────────────────────────


def compute_level(profile: SourceProfile) -> SourceLevel:
    """根据来源的历史表现计算应有的层级（P4-H3）。

    升级规则（从低到高）：
      Trial   → 默认，新来源起点
      Verified → accepted >= 3 且 avg_score >= 60
      Trusted  → accepted >= 5 且 avg_score >= 70
      Elite    → accepted >= 10 且 avg_score >= 80

    降级规则：
      Deprecated → rejected > accepted 且总评次数 >= 5
                   或 failure_count >= 3（连续失效）

    注意：
    - 此函数只计算应有层级，不修改 profile
    - Deprecated 来源不会自动升级回来（需人工或周期整合干预）
    """
    # 当前已是 Deprecated 的不自动恢复
    if profile.current_level == SourceLevel.DEPRECATED:
        return SourceLevel.DEPRECATED

    total = profile.accepted_count + profile.rejected_count

    # 降级判定优先
    if profile.failure_count >= 3:
        return SourceLevel.DEPRECATED
    if total >= 5 and profile.rejected_count > profile.accepted_count:
        return SourceLevel.DEPRECATED

    # 升级判定（从高到低匹配，取最高可达层级）
    avg = profile.avg_score

    if profile.accepted_count >= 10 and avg >= 80:
        return SourceLevel.ELITE
    if profile.accepted_count >= 5 and avg >= 70:
        return SourceLevel.TRUSTED
    if profile.accepted_count >= 3 and avg >= 60:
        return SourceLevel.VERIFIED

    return SourceLevel.TRIAL


# ── 来源记录写入逻辑 ─────────────────────────────────────────


def _update_expertise_tags(
    profile: SourceProfile,
    domain_tags: list[str],
    score: float,
    now: datetime,
) -> None:
    """更新来源的领域专长标签（累积/衰退，标签永不删除）。"""
    for tag in domain_tags:
        tag = tag.strip()
        if not tag:
            continue
        if tag in profile.expertise_tags:
            existing = profile.expertise_tags[tag]
            existing.avg_score = (existing.avg_score * existing.count + score) / (existing.count + 1)
            existing.count += 1
            existing.last_seen = now
        else:
            profile.expertise_tags[tag] = ExpertiseTag(avg_score=score, count=1, last_seen=now)


def record_source(
    domain: str,
    score: float,
    passed: bool,
    topics: Optional[list[str]] = None,
    domain_tags: Optional[list[str]] = None,
    repo: Optional[SourceRepository] = None,
) -> Optional[SourceProfile]:
    """记录一个来源的本次表现（P4-H2）。

    写入策略：
    - 通过阈值的来源（passed=True）：新来源创建 Trial 记录，已有来源更新统计
    - 未通过阈值的来源（passed=False）：仅更新已在记忆中的来源（为降级做准备），
      不为新的低质量来源创建记录
    - 访问失败的来源：不记录（短期回避由 page_cache 处理）

    Args:
        domain: 来源域名
        score: 本次评分总分
        passed: 是否通过质量阈值
        topics: 本次相关的主题标签（向后兼容）
        domain_tags: LLM 生成的领域专长标签（如 ['医学', 'mRNA疫苗']）
        repo: 使用的仓库实例，默认使用全局单例

    Returns:
        更新后的 SourceProfile，或 None（新来源未通过阈值时不记录）
    """
    if repo is None:
        repo = get_source_repository()

    now = datetime.now()
    existing = repo.find_by_domain(domain)

    if existing is not None:
        # 已有来源：无论 passed 与否都更新统计（为升降级提供数据）
        profile = existing
        profile.historical_scores.append(score)
        profile.avg_score = sum(profile.historical_scores) / len(profile.historical_scores)
        if passed:
            profile.accepted_count += 1
            profile.failure_count = 0  # 成功访问重置连续失效计数
        else:
            profile.rejected_count += 1
        profile.last_seen_at = now
        # 合并 topics（去重，向后兼容）
        if topics:
            merged = set(profile.topics)
            merged.update(topics)
            profile.topics = sorted(merged)
        # 更新领域专长标签
        if domain_tags:
            _update_expertise_tags(profile, domain_tags, score, now)
        # 补充 first_seen_at（兼容旧数据）
        if profile.first_seen_at is None:
            profile.first_seen_at = now
    elif passed:
        # 新来源且通过阈值：创建 Trial 记录
        profile = SourceProfile(
            domain=domain,
            topics=sorted(set(topics)) if topics else [],
            current_level=SourceLevel.TRIAL,
            historical_scores=[score],
            accepted_count=1,
            rejected_count=0,
            first_seen_at=now,
            last_seen_at=now,
            avg_score=score,
        )
        # 初始化领域专长标签
        if domain_tags:
            _update_expertise_tags(profile, domain_tags, score, now)
    else:
        # 新来源但未通过阈值：不值得记录
        return None

    # 升降级判定（P4-H4）
    old_level = profile.current_level
    new_level = compute_level(profile)
    if new_level != old_level:
        profile.current_level = new_level
        logger.info(
            "来源层级变更: %s %s -> %s (avg=%.1f, accepted=%d, rejected=%d)",
            domain, old_level.value, new_level.value,
            profile.avg_score, profile.accepted_count, profile.rejected_count,
        )
        # P5-I4：层级变动时使相关主题的披露窗口缓存失效
        for t in profile.topics:
            invalidate_disclosure_cache(t)

    repo.save(profile)
    logger.debug(
        "来源记录已更新: %s (avg=%.1f, level=%s, accepted=%d, rejected=%d)",
        domain, profile.avg_score, profile.current_level.value,
        profile.accepted_count, profile.rejected_count,
    )
    return profile


# ── 失效惩罚逻辑 ─────────────────────────────────────────────


def record_source_failure(
    domain: str,
    reason: str = "unknown",
    repo: Optional[SourceRepository] = None,
) -> Optional[SourceProfile]:
    """记录一次来源访问失效（P4-H5）。

    仅对已在记忆中的来源生效：递增 failure_count，追加 failure_flags，
    触发升降级判定（failure_count >= 3 → Deprecated）。
    不在记忆中的来源直接忽略。

    Args:
        domain: 来源域名
        reason: 失效原因（如 "404"、"timeout"、"empty"）
        repo: 使用的仓库实例，默认使用全局单例

    Returns:
        更新后的 SourceProfile，或 None（来源不在记忆中）
    """
    if repo is None:
        repo = get_source_repository()

    existing = repo.find_by_domain(domain)
    if existing is None:
        return None

    profile = existing
    profile.failure_count += 1
    profile.failure_flags.append(f"{datetime.now().strftime('%Y-%m-%d')}: {reason}")
    profile.last_seen_at = datetime.now()

    # 升降级判定
    old_level = profile.current_level
    new_level = compute_level(profile)
    if new_level != old_level:
        profile.current_level = new_level
        logger.info(
            "来源因连续失效降级: %s %s -> %s (failure_count=%d)",
            domain, old_level.value, new_level.value, profile.failure_count,
        )
        # P5-I4：层级变动时使相关主题的披露窗口缓存失效
        for t in profile.topics:
            invalidate_disclosure_cache(t)

    repo.save(profile)
    logger.debug("来源失效记录: %s (reason=%s, failure_count=%d)", domain, reason, profile.failure_count)
    return profile


# ── 模块级便捷入口（懒加载单例） ───────────────────────────────


_default_repo: Optional[JsonSourceRepository] = None


def get_source_repository() -> JsonSourceRepository:
    """获取默认的信息源仓库实例（懒加载单例）"""
    global _default_repo
    if _default_repo is None:
        from config import cfg
        storage_dir = Path(cfg.memory.dir)
        if not storage_dir.is_absolute():
            storage_dir = cfg.root / storage_dir
        _default_repo = JsonSourceRepository(storage_dir)
    return _default_repo
