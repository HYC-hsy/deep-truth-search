"""
Deep Truth Search — Visit Tool

访问 URL 并提取页面结构化内容。
支持 HTML（trafilatura）和 PDF（pymupdf4llm）两种提取路径。

用法：
    from tools.visit_tool import visit

    page = await visit("https://example.com/article")
    page = await visit("https://example.com/report.pdf")  # PDF 也支持
"""

from __future__ import annotations

import logging
import re
from urllib.parse import urlparse

from models import PageContent, PageType

logger = logging.getLogger(__name__)

# trafilatura 提取配置
_TRAFILATURA_OPTS = dict(
    include_comments=False,
    include_tables=True,
    deduplicate=True,
    favor_recall=True,
)

# 不可提取的文件类型（PDF 已支持，从黑名单中移除）
_SKIP_EXTENSIONS = {
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".zip", ".rar", ".gz", ".tar",
    ".mp3", ".mp4", ".avi", ".mov", ".wav", ".flac",
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg",
}

# PDF 文件后缀
_PDF_EXTENSIONS = {".pdf"}


def _get_url_extension(url: str) -> str:
    """从 URL 路径中提取文件扩展名（小写）。"""
    path = urlparse(url).path.lower().split("?")[0]
    for ext in _SKIP_EXTENSIONS | _PDF_EXTENSIONS:
        if path.endswith(ext):
            return ext
    return ""


def _is_skip_url(url: str) -> bool:
    """检查 URL 是否指向不可提取的资源。"""
    ext = _get_url_extension(url)
    return ext in _SKIP_EXTENSIONS


def _is_pdf_url(url: str) -> bool:
    """检查 URL 是否指向 PDF 文件。"""
    return _get_url_extension(url) in _PDF_EXTENSIONS


async def visit(url: str) -> PageContent | None:
    """访问 URL，提取页面结构化内容。

    根据内容类型自动选择提取策略：
    - HTML → trafilatura 提取
    - PDF → pymupdf4llm 提取
    - 其他不可提取类型 → 返回 None

    Returns:
        PageContent 或 None（访问失败时）
    """
    from tools.http_client import fetch_url

    # 不可提取的资源直接跳过
    if _is_skip_url(url):
        logger.info("跳过不可提取资源: %s", url[:120])
        return None

    is_pdf = _is_pdf_url(url)

    logger.info("访问页面: %s%s", url[:120], " [PDF]" if is_pdf else "")

    try:
        resp = await fetch_url(url)
    except Exception as exc:
        logger.warning("访问失败 %s: %s", url[:80], exc)
        return None

    if resp.status_code >= 400:
        logger.warning("页面访问失败 %s: HTTP %d", url[:80], resp.status_code)
        return None

    # 通过 Content-Type 头二次判断是否为 PDF
    content_type = resp.headers.get("content-type", "").lower()
    if "application/pdf" in content_type:
        is_pdf = True

    if is_pdf:
        return _extract_pdf(url, resp.content)
    else:
        html = resp.text
        if not html:
            return None
        return _extract_html(url, html)


# ── HTML 提取 ────────────────────────────────────────────────


def _extract_html(url: str, html: str) -> PageContent:
    """从 HTML 中提取结构化内容（增强版）。"""
    import trafilatura

    domain = urlparse(url).netloc

    # 提取正文
    body_text = trafilatura.extract(html, **_TRAFILATURA_OPTS) or ""

    # 提取元数据
    metadata = trafilatura.extract_metadata(html, default_url=url)

    title = ""
    author = ""
    date = ""
    institution = ""
    if metadata:
        title = metadata.title or ""
        author = metadata.author or ""
        date = metadata.date or ""
        # trafilatura 的 sitename 通常对应机构/站点名
        institution = metadata.sitename or ""

    # fallback 标题
    if not title:
        title = _extract_title_from_html(html)

    # 页面类型推断
    page_type = _infer_page_type(domain, url, html, body_text)

    # 引用提取
    references = _extract_references_from_html(html, body_text)

    page = PageContent(
        url=url,
        title=title,
        body_text=body_text[:50000],
        author=author,
        date=date,
        domain=domain,
        institution=institution,
        page_type=page_type,
        references=references,
    )

    logger.info(
        "HTML 提取完成: %s (标题=%s, 类型=%s, 正文=%d, 引用=%d)",
        url[:60], title[:30], page_type.value, len(body_text), len(references),
    )
    return page


# ── PDF 提取 ─────────────────────────────────────────────────


def _extract_pdf(url: str, content: bytes) -> PageContent | None:
    """从 PDF 二进制内容中提取结构化文本。"""
    domain = urlparse(url).netloc

    try:
        import pymupdf4llm
        import pymupdf
    except ImportError:
        logger.warning("pymupdf4llm 未安装，无法提取 PDF: %s", url[:80])
        return None

    try:
        doc = pymupdf.open(stream=content, filetype="pdf")
    except Exception as e:
        logger.warning("PDF 打开失败 %s: %s", url[:80], e)
        return None

    # 提取元数据
    meta = doc.metadata or {}
    title = meta.get("title", "") or ""
    author = meta.get("author", "") or ""
    date = meta.get("creationDate", "") or ""
    # PDF creationDate 格式通常为 D:20240101120000，取前8位
    if date.startswith("D:"):
        date = date[2:10]  # 20240101
        if len(date) == 8:
            date = f"{date[:4]}-{date[4:6]}-{date[6:8]}"

    page_count = len(doc)

    # 用 pymupdf4llm 提取 markdown 格式文本（更适合 LLM 消费）
    try:
        body_text = pymupdf4llm.to_markdown(doc)
    except Exception as e:
        logger.warning("pymupdf4llm 提取失败，降级为纯文本: %s", e)
        # 降级：直接提取纯文本
        body_text = ""
        for page in doc:
            body_text += page.get_text() + "\n"

    doc.close()

    if not body_text.strip():
        logger.info("PDF 内容为空: %s", url[:80])
        return None

    # 从 PDF 文本中提取引用
    references = _extract_references_from_text(body_text)

    page = PageContent(
        url=url,
        title=title,
        body_text=body_text[:50000],
        author=author,
        date=date,
        domain=domain,
        institution="",  # PDF 元数据中通常没有机构信息
        page_type=PageType.ACADEMIC if _looks_academic(body_text) else PageType.OTHER,
        references=references,
    )

    logger.info(
        "PDF 提取完成: %s (标题=%s, 页数=%d, 正文=%d, 引用=%d)",
        url[:60], title[:30], page_count, len(body_text), len(references),
    )
    return page


# ── 页面类型推断 ─────────────────────────────────────────────

# 域名 → 页面类型映射
_DOMAIN_TYPE_MAP: dict[str, PageType] = {}
_ACADEMIC_DOMAINS = {
    "arxiv.org", "scholar.google.com", "pubmed.ncbi.nlm.nih.gov",
    "sciencedirect.com", "nature.com", "science.org", "ieee.org",
    "acm.org", "springer.com", "wiley.com", "researchgate.net",
}
_NEWS_DOMAINS = {
    "reuters.com", "apnews.com", "bbc.com", "nytimes.com",
    "washingtonpost.com", "theguardian.com", "cnn.com",
    "xinhuanet.com", "people.com.cn", "chinadaily.com.cn",
}
_WIKI_DOMAINS = {"wikipedia.org", "britannica.com"}
_GOV_TLDS = {".gov", ".gov.cn", ".gov.uk"}


def _infer_page_type(domain: str, url: str, html: str, body_text: str) -> PageType:
    """根据域名、URL 模式和内容信号推断页面类型。"""
    domain_lower = domain.lower().lstrip("www.")

    # 域名匹配
    for d in _ACADEMIC_DOMAINS:
        if domain_lower == d or domain_lower.endswith("." + d):
            return PageType.ACADEMIC
    for d in _NEWS_DOMAINS:
        if domain_lower == d or domain_lower.endswith("." + d):
            return PageType.NEWS
    for d in _WIKI_DOMAINS:
        if domain_lower == d or domain_lower.endswith("." + d):
            return PageType.WIKI

    # TLD 匹配
    if any(domain_lower.endswith(tld) for tld in _GOV_TLDS):
        return PageType.OFFICIAL
    if domain_lower.endswith(".edu") or domain_lower.endswith(".edu.cn") or domain_lower.endswith(".ac.uk"):
        return PageType.ACADEMIC

    # 内容信号
    html_lower = html[:5000].lower()

    # 学术信号
    if _looks_academic(body_text):
        return PageType.ACADEMIC

    # 新闻信号
    news_signals = ['<article', 'class="article"', '"datePublished"', '"newsarticle"', "journalism"]
    if sum(1 for s in news_signals if s in html_lower) >= 2:
        return PageType.NEWS

    # 博客信号
    blog_signals = ["blog", "/blog/", "class=\"post\"", "class=\"entry\""]
    if sum(1 for s in blog_signals if s in html_lower or s in url.lower()) >= 2:
        return PageType.BLOG

    return PageType.OTHER


def _looks_academic(text: str) -> bool:
    """检测文本是否具有学术特征。"""
    text_lower = text[:5000].lower()
    academic_signals = ["abstract", "references", "doi:", "et al.", "methodology", "peer-reviewed", "journal"]
    return sum(1 for s in academic_signals if s in text_lower) >= 2


# ── 引用提取 ─────────────────────────────────────────────────


def _extract_references_from_html(html: str, body_text: str) -> list[str]:
    """从 HTML 和正文中提取引用链接和参考文献。"""
    refs: list[str] = []

    # 1. 提取 <a> 标签中带 DOI 或引用标记的链接
    doi_pattern = re.compile(r'href="(https?://doi\.org/[^"]+)"', re.IGNORECASE)
    for match in doi_pattern.finditer(html[:100000]):
        ref = match.group(1)
        if ref not in refs:
            refs.append(ref)

    # 2. 从正文中提取 DOI
    doi_text_pattern = re.compile(r'(10\.\d{4,}/[^\s,;)\]]+)')
    for match in doi_text_pattern.finditer(body_text[:20000]):
        doi = f"https://doi.org/{match.group(1)}"
        if doi not in refs:
            refs.append(doi)

    # 限制数量
    return refs[:20]


def _extract_references_from_text(text: str) -> list[str]:
    """从纯文本（PDF 提取结果）中提取引用。"""
    refs: list[str] = []

    # DOI
    doi_pattern = re.compile(r'(10\.\d{4,}/[^\s,;)\]]+)')
    for match in doi_pattern.finditer(text):
        doi = f"https://doi.org/{match.group(1)}"
        if doi not in refs:
            refs.append(doi)

    # URL
    url_pattern = re.compile(r'https?://[^\s<>")\]]+')
    for match in url_pattern.finditer(text):
        url = match.group(0).rstrip(".,;)")
        if url not in refs and "doi.org" not in url:
            refs.append(url)

    return refs[:20]


# ── 辅助函数 ─────────────────────────────────────────────────


def _extract_title_from_html(html: str) -> str:
    """简单提取 <title> 标签内容作为 fallback。"""
    match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()[:200]
    return ""
