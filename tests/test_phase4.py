"""Phase 4 全面单元测试 — 信息源记忆系统"""

import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

# 确保项目根目录在 path 中
sys.path.insert(0, str(Path(__file__).parent.parent))

from memory.source_memory import (
    JsonSourceRepository,
    compute_level,
    get_source_repository,
    record_source,
    record_source_failure,
)
from models import (
    DimensionScore,
    EvidenceItem,
    PageContent,
    ScoreResult,
    SourceLevel,
    SourceProfile,
)

import memory.source_memory as sm

passed = 0
failed = 0


def ok(name):
    global passed
    passed += 1
    print(f"  OK {name}")


def check(name, condition, msg=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  OK {name}")
    else:
        failed += 1
        print(f"  FAIL {name}: {msg}")


def fresh_repo():
    d = Path(tempfile.mkdtemp()) / "mem"
    sm._default_repo = JsonSourceRepository(d)
    return get_source_repository(), d


def mkp(**kw):
    defaults = dict(
        domain="t.com",
        accepted_count=0,
        rejected_count=0,
        avg_score=0.0,
        failure_count=0,
        current_level=SourceLevel.TRIAL,
    )
    defaults.update(kw)
    return SourceProfile(**defaults)


def make_score(total):
    return ScoreResult(total_score=total, passes_threshold=total >= 60)


def make_evidence(domain, url, score_total):
    return EvidenceItem(
        claim="test",
        evidence_text="test evidence",
        source_url=url,
        source_domain=domain,
        score=make_score(score_total),
    )


# ============================================================
print("=== 1. Repository edge cases ===")
# ============================================================

repo, d = fresh_repo()

check("empty find_by_domain", repo.find_by_domain("x.com") is None)
check("empty find_by_topic", repo.find_by_topic("AI") == [])
check("empty find_by_level", repo.find_by_level(SourceLevel.TRIAL) == [])
check("empty find_all", repo.find_all() == [])
check("empty count", repo.count() == 0)
check("empty delete", repo.delete("x.com") is False)

# Corrupted JSON file
(d / "sources.json").write_text("NOT VALID JSON", encoding="utf-8")
sm._default_repo = None
sm._default_repo = JsonSourceRepository(d)
repo2 = get_source_repository()
check("corrupted json loads empty", repo2.count() == 0)

# Partially corrupted sources (single record fault tolerance)
good_bad = {
    "sources": {
        "good.com": {"domain": "good.com", "current_level": "trial"},
        "bad.com": {"domain": "bad.com", "current_level": "INVALID"},
    },
    "metadata": {"total_tasks": 5, "last_updated": None},
}
(d / "sources.json").write_text(json.dumps(good_bad), encoding="utf-8")
sm._default_repo = None
sm._default_repo = JsonSourceRepository(d)
repo3 = get_source_repository()
check("partial corrupt: good survives", repo3.find_by_domain("good.com") is not None)
check("partial corrupt: bad skipped", repo3.find_by_domain("bad.com") is None)
check("partial corrupt: count=1", repo3.count() == 1)
check("metadata preserved", repo3.get_total_tasks() == 5)

# batch_save
repo, d = fresh_repo()
profiles = [
    SourceProfile(domain=f"s{i}.com", topics=["t"], avg_score=float(i * 10))
    for i in range(5)
]
repo.batch_save(profiles)
check("batch_save count", repo.count() == 5)
check("batch_save content", repo.find_by_domain("s3.com").avg_score == 30.0)

# save overwrites
repo.save(SourceProfile(domain="s3.com", avg_score=99.0))
check("save overwrites", repo.find_by_domain("s3.com").avg_score == 99.0)
check("save overwrites count unchanged", repo.count() == 5)

# find_by_topic fuzzy match
repo, d = fresh_repo()
repo.save(SourceProfile(domain="a.com", topics=["AI research", "machine learning"]))
repo.save(SourceProfile(domain="b.com", topics=["climate science"]))
repo.save(SourceProfile(domain="c.com", topics=["AI ethics"]))
check("find_by_topic: AI matches 2", len(repo.find_by_topic("AI")) == 2)
check("find_by_topic: case insensitive", len(repo.find_by_topic("ai")) == 2)
check("find_by_topic: science matches 1", len(repo.find_by_topic("science")) == 1)
check("find_by_topic: no match", len(repo.find_by_topic("quantum")) == 0)

# task count
repo, d = fresh_repo()
check("task count initial=0", repo.get_total_tasks() == 0)
check("task count +1", repo.increment_task_count() == 1)
check("task count +2", repo.increment_task_count() == 2)


# ============================================================
print("=== 2. compute_level boundary values ===")
# ============================================================

check("Verified: 3/60", compute_level(mkp(accepted_count=3, avg_score=60)) == SourceLevel.VERIFIED)
check("Trial: 3/59.9", compute_level(mkp(accepted_count=3, avg_score=59.9)) == SourceLevel.TRIAL)
check("Trial: 2/60", compute_level(mkp(accepted_count=2, avg_score=60)) == SourceLevel.TRIAL)
check("Trusted: 5/70", compute_level(mkp(accepted_count=5, avg_score=70)) == SourceLevel.TRUSTED)
check("Verified: 5/69.9", compute_level(mkp(accepted_count=5, avg_score=69.9)) == SourceLevel.VERIFIED)
check("Elite: 10/80", compute_level(mkp(accepted_count=10, avg_score=80)) == SourceLevel.ELITE)
check("Trusted: 10/79.9", compute_level(mkp(accepted_count=10, avg_score=79.9)) == SourceLevel.TRUSTED)
check("Deprecated: fail=3", compute_level(mkp(failure_count=3)) == SourceLevel.DEPRECATED)
check("Not deprecated: fail=2", compute_level(mkp(failure_count=2)) != SourceLevel.DEPRECATED)
check(
    "Deprecated: rej>acc total=5",
    compute_level(mkp(accepted_count=2, rejected_count=3)) == SourceLevel.DEPRECATED,
)
check(
    "Not deprecated: rej=acc total=6",
    compute_level(mkp(accepted_count=3, rejected_count=3)) != SourceLevel.DEPRECATED,
)
check(
    "Not deprecated: rej>acc total=4",
    compute_level(mkp(accepted_count=1, rejected_count=3)) != SourceLevel.DEPRECATED,
)
check(
    "Deprecated sticky",
    compute_level(mkp(accepted_count=100, avg_score=99, current_level=SourceLevel.DEPRECATED))
    == SourceLevel.DEPRECATED,
)
check(
    "Deprecated via fail overrides Elite",
    compute_level(mkp(accepted_count=10, avg_score=85, failure_count=3)) == SourceLevel.DEPRECATED,
)


# ============================================================
print("=== 3. record_source edge cases ===")
# ============================================================

repo, d = fresh_repo()

# New rejected = None
check("new rejected returns None", record_source("z.com", 40, False) is None)
check("new rejected not stored", repo.find_by_domain("z.com") is None)

# Topics dedup and sort
record_source("t.com", 70, True, topics=["B", "A", "B"])
p = repo.find_by_domain("t.com")
check("topics deduped and sorted on create", p.topics == ["A", "B"])

record_source("t.com", 75, True, topics=["C", "A"])
p = repo.find_by_domain("t.com")
check("topics merged on update", p.topics == ["A", "B", "C"])

# Empty / None topics
record_source("e.com", 65, True, topics=[])
check("empty topics ok", repo.find_by_domain("e.com").topics == [])
record_source("e.com", 70, True, topics=None)
check("None topics ok", repo.find_by_domain("e.com").topics == [])

# first_seen_at backfill
repo.save(SourceProfile(domain="old.com", first_seen_at=None, accepted_count=0))
record_source("old.com", 70, True)
check("first_seen_at backfilled", repo.find_by_domain("old.com").first_seen_at is not None)

# avg_score accuracy
repo, d = fresh_repo()
record_source("avg.com", 60, True)
record_source("avg.com", 80, True)
record_source("avg.com", 70, True)
p = repo.find_by_domain("avg.com")
expected = (60 + 80 + 70) / 3
check("avg_score correct", abs(p.avg_score - expected) < 0.01, f"{p.avg_score} vs {expected}")

# failure_count reset only on passed=True
repo, d = fresh_repo()
record_source("f.com", 70, True)
record_source_failure("f.com", "404")
record_source_failure("f.com", "404")
check("failure count=2", repo.find_by_domain("f.com").failure_count == 2)
record_source("f.com", 50, False)  # rejected
check("rejected NOT reset", repo.find_by_domain("f.com").failure_count == 2)
record_source("f.com", 70, True)  # passed
check("passed DOES reset", repo.find_by_domain("f.com").failure_count == 0)

# Auto-promotion
repo, d = fresh_repo()
for _ in range(3):
    record_source("up.com", 75, True)
check("auto-promoted to Verified", repo.find_by_domain("up.com").current_level == SourceLevel.VERIFIED)


# ============================================================
print("=== 4. record_source_failure edge cases ===")
# ============================================================

repo, d = fresh_repo()

check("failure on nonexistent = None", record_source_failure("ghost.com") is None)

record_source("fl.com", 70, True)
record_source_failure("fl.com", "404")
record_source_failure("fl.com", "timeout")
record_source_failure("fl.com", "empty")
p = repo.find_by_domain("fl.com")
check("failure_flags count=3", len(p.failure_flags) == 3)
check(
    "failure_flags contain reasons",
    "404" in p.failure_flags[0]
    and "timeout" in p.failure_flags[1]
    and "empty" in p.failure_flags[2],
)
today = datetime.now().strftime("%Y-%m-%d")
check("failure_flags have dates", all(today in f for f in p.failure_flags))
check("auto-deprecated after 3 failures", p.current_level == SourceLevel.DEPRECATED)


# ============================================================
print("=== 5. main_handler._record_sources logic ===")
# ============================================================

from agents.main_handler import MainAgentHandler

repo, d = fresh_repo()

handler = MainAgentHandler(query="test")
handler._page_cache["https://bad.com/page1"] = (
    PageContent(url="https://bad.com/page1", domain="bad.com"),
    make_score(45),
)

evidence_items = [
    make_evidence("good.com", "https://good.com/a", 85),
    make_evidence("good.com", "https://good.com/b", 90),
    make_evidence("other.com", "https://other.com/x", 80),
]

handler._record_sources(
    evidence_items=evidence_items,
    accepted_sources=["https://good.com/a", "https://good.com/b", "https://other.com/x"],
    rejected_sources=["https://bad.com/page1"],
    topic_context="test topic",
)

check("accepted: good.com recorded", repo.find_by_domain("good.com") is not None)
check("accepted: other.com recorded (score>=75)", repo.find_by_domain("other.com") is not None)

p = repo.find_by_domain("good.com")
check("good.com accepted=1 (deduped by domain)", p.accepted_count == 1, f"got {p.accepted_count}")
check("good.com uses max score=90", p.avg_score == 90.0, f"got {p.avg_score}")

check("new rejected bad.com NOT created", repo.find_by_domain("bad.com") is None)

# Put bad.com in memory, then reject
record_source("bad.com", 65, True, topics=["t"])
handler._record_sources(
    evidence_items=[],
    accepted_sources=[],
    rejected_sources=["https://bad.com/page1"],
    topic_context="test topic",
)
p = repo.find_by_domain("bad.com")
check("existing bad.com rejected: rejected_count=1", p.rejected_count == 1)
check(
    "bad.com uses page_cache score",
    abs(p.avg_score - (65 + 45) / 2) < 0.1,
    f"got {p.avg_score}",
)

# Empty inputs - no crash
handler._record_sources([], [], [], "")
check("empty inputs no crash", True)

# Malformed URLs - no crash
handler._record_sources(
    evidence_items=[],
    accepted_sources=["not-a-url", "", "https://ok.com/page"],
    rejected_sources=["also-bad"],
    topic_context="t",
)
check("malformed URLs handled gracefully", True)

# Duplicate rejected domain only processed once
repo, d = fresh_repo()
handler2 = MainAgentHandler(query="test")
record_source("dup.com", 70, True)
handler2._page_cache["https://dup.com/a"] = (
    PageContent(url="https://dup.com/a", domain="dup.com"),
    make_score(45),
)
handler2._page_cache["https://dup.com/b"] = (
    PageContent(url="https://dup.com/b", domain="dup.com"),
    make_score(40),
)
handler2._record_sources(
    evidence_items=[],
    accepted_sources=[],
    rejected_sources=["https://dup.com/a", "https://dup.com/b"],
    topic_context="t",
)
p = repo.find_by_domain("dup.com")
check("duplicate rejected domain processed once", p.rejected_count == 1)


# ============================================================
print("=== 6. search_handler._record_failure logic ===")
# ============================================================

from agents.search_handler import SearchAgentHandler

repo, d = fresh_repo()
sh = SearchAgentHandler("test subclaim", "test topic")

# Unknown domain
sh._record_failure("https://unknown.com/page")
check("failure unknown: no record", repo.find_by_domain("unknown.com") is None)

# Known domain
record_source("known.com", 75, True)
sh._record_failure("https://known.com/page")
p = repo.find_by_domain("known.com")
check("failure known: count=1", p.failure_count == 1)
check("failure known: reason", "empty_or_inaccessible" in p.failure_flags[0])

# Empty/malformed URL - no crash
sh._record_failure("")
sh._record_failure("not-a-url")
check("empty/malformed URL no crash", True)


# ============================================================
print("=== 7. Persistence roundtrip ===")
# ============================================================

repo, d = fresh_repo()
record_source("persist.com", 80, True, topics=["AI"])
record_source("persist.com", 85, True, topics=["ML"])
record_source_failure("persist.com", "404")

sm._default_repo = None
sm._default_repo = JsonSourceRepository(d)
repo2 = get_source_repository()
p = repo2.find_by_domain("persist.com")
check("persist: domain", p.domain == "persist.com")
check("persist: scores", p.historical_scores == [80, 85])
check("persist: avg", abs(p.avg_score - 82.5) < 0.01)
check("persist: accepted=2", p.accepted_count == 2)
check("persist: failure_count=1", p.failure_count == 1)
check("persist: failure_flags=1", len(p.failure_flags) == 1)
check("persist: topics", sorted(p.topics) == ["AI", "ML"])
check("persist: first_seen_at set", p.first_seen_at is not None)
check("persist: last_seen_at set", p.last_seen_at is not None)
check("persist: level=trial", p.current_level == SourceLevel.TRIAL)

# Cleanup
sm._default_repo = None

print(f"\n========================================")
print(f"RESULTS: {passed} passed, {failed} failed")
print(f"========================================")
if failed > 0:
    sys.exit(1)
