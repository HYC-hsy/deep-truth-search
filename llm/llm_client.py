"""
Deep Truth Search — 统一 LLM 调用模块

支持 OpenAI 兼容 API 和 Anthropic 原生 API。
所有配置从 config.py 读取（源头为 .env）。

用法：
    from llm import call_llm, call_llm_json

    # 纯文本
    text = await call_llm("你是研究助手", "请分析这个观点")

    # 结构化 JSON + Pydantic 校验
    result = await call_llm_json("你是研究助手", "请返回子观点列表", MyModel)
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional, Type

import httpx
from pydantic import BaseModel

from config import cfg

logger = logging.getLogger(__name__)


# ── Claude 模型检测 ────────────────────────────────────────────


def _is_claude_model() -> bool:
    """检测当前模型是否为 Claude 系列。

    通过 OpenAI 兼容代理使用 Claude 时，需跳过 Claude 不支持的
    OpenAI 特有功能（如 response_format）。
    """
    return "claude" in cfg.llm.model.lower()


# ── 异常定义（放在顶部，避免前向引用问题）────────────────────


class LLMError(Exception):
    """LLM 调用相关错误的基类"""
    pass


class LLMConfigError(LLMError):
    """LLM 配置错误（缺少 API key 等）"""
    pass


class LLMParseError(LLMError):
    """LLM 返回的 JSON 格式无法解析"""
    pass


# ── 可重试的 HTTP 状态码 ──────────────────────────────────────

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


# ── 退避策略 ──────────────────────────────────────────────────


def _backoff_delay(response: Optional[httpx.Response], attempt: int) -> float:
    """计算重试等待时间，优先使用服务器返回的 Retry-After 头"""
    if response is not None:
        retry_after = response.headers.get("retry-after")
        if retry_after:
            try:
                return max(0.5, float(retry_after))
            except ValueError:
                pass
    return max(0.5, min(30.0, 1.5 * (2 ** attempt)))


# ── 通用 HTTP 请求 + 分层重试 ─────────────────────────────────


async def _request_with_retry(
    url: str,
    headers: dict,
    payload: dict,
    extract_fn: Any = None,
    max_retries_override: Optional[int] = None,
) -> Any:
    """带分层重试和指数退避的 HTTP POST 请求。

    区分处理：超时、连接错误、HTTP 4xx/5xx、未知异常。
    只有可重试的状态码才会重试，401/403 等直接抛出。
    """
    if extract_fn is None:
        extract_fn = _extract_openai

    max_retries = max_retries_override if max_retries_override is not None else cfg.llm.max_retries
    timeout = httpx.Timeout(
        connect=10.0,
        read=float(cfg.llm.timeout),
        write=10.0,
        pool=10.0,
    )

    last_error: Optional[Exception] = None
    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(max_retries + 1):
            try:
                resp = await client.post(url, headers=headers, json=payload)

                if resp.status_code >= 400:
                    body = resp.text[:500]
                    logger.warning(
                        "LLM API 错误: HTTP %d | 第 %d/%d 次 | body: %s",
                        resp.status_code, attempt + 1, max_retries + 1, body,
                    )
                    if resp.status_code in RETRYABLE_STATUS and attempt < max_retries:
                        delay = _backoff_delay(resp, attempt)
                        await asyncio.sleep(delay)
                        continue
                    raise LLMError(f"HTTP {resp.status_code}: {body}")

                data = resp.json()
                return extract_fn(data)

            except httpx.TimeoutException as e:
                last_error = e
                logger.warning("LLM 超时: 第 %d/%d 次 | %s", attempt + 1, max_retries + 1, e)
                if attempt < max_retries:
                    await asyncio.sleep(_backoff_delay(None, attempt))
                    continue

            except httpx.ConnectError as e:
                last_error = e
                logger.warning("LLM 连接错误: 第 %d/%d 次 | %s", attempt + 1, max_retries + 1, e)
                if attempt < max_retries:
                    await asyncio.sleep(_backoff_delay(None, attempt))
                    continue

            except LLMError:
                raise

            except Exception as e:
                last_error = e
                logger.error("LLM 未知错误: %s", e)
                if attempt < max_retries:
                    await asyncio.sleep(_backoff_delay(None, attempt))
                    continue

    raise LLMError(f"全部 {max_retries + 1} 次尝试均失败，最后错误: {last_error}")


# ── 响应提取函数 ──────────────────────────────────────────────


def _extract_openai(data: dict) -> str:
    """从 OpenAI 响应中提取文本"""
    finish_reason = data.get("choices", [{}])[0].get("finish_reason", "unknown")
    usage = data.get("usage", {})
    content = data["choices"][0]["message"]["content"]
    if finish_reason != "stop":
        logger.warning("LLM finish_reason=%s (非stop) | usage=%s | output_len=%d",
                        finish_reason, usage, len(content))
    return content


def _extract_anthropic(data: dict) -> str:
    """从 Anthropic 响应中提取文本"""
    for block in data.get("content", []):
        if block.get("type") == "text":
            return block["text"]
    return ""


# ── Provider 实现 ─────────────────────────────────────────────


async def _call_openai(
    system_prompt: str,
    user_message: str,
    max_tokens: int,
    temperature: float,
    json_mode: bool = False,
) -> str:
    """调用 OpenAI 兼容 API"""
    url = f"{cfg.llm.base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {cfg.llm.api_key}",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {
        "model": cfg.llm.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    # OpenAI 原生 JSON mode — Claude 不支持此参数，跳过
    if json_mode and not _is_claude_model():
        payload["response_format"] = {"type": "json_object"}

    return await _request_with_retry(url, headers, payload)


async def _call_anthropic(
    system_prompt: str,
    user_message: str,
    max_tokens: int,
    temperature: float,
    json_mode: bool = False,
) -> str:
    """调用 Anthropic Messages API（直连或通过代理）"""
    url = f"{cfg.llm.base_url.rstrip('/')}/messages"
    headers = {
        "Authorization": f"Bearer {cfg.llm.api_key}",
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }
    payload = {
        "model": cfg.llm.model,
        "system": system_prompt,
        "messages": [
            {"role": "user", "content": user_message},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    return await _request_with_retry(url, headers, payload, extract_fn=_extract_anthropic)


# ── Provider 路由表（扩展时只需加一行）────────────────────────

_PROVIDERS = {
    "openai": _call_openai,
    "anthropic": _call_anthropic,
}


# ── JSON 解析工具 ─────────────────────────────────────────────


def _try_parse_json(text: str) -> Any:
    """多策略 JSON 解析，容忍 LLM 常见格式问题。

    依次尝试：
    1. 直接解析整个文本
    2. 去掉 markdown 代码块后解析
    3. 提取第一个 { ... } 对象
    4. 提取第一个 [ ... ] 数组
    """
    # 策略 1：直接解析
    try:
        return json.loads(text.strip())
    except (json.JSONDecodeError, ValueError) as e:
        _s1_err = e

    # 策略 2：去掉 markdown 代码块
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
        try:
            return json.loads(cleaned)
        except (json.JSONDecodeError, ValueError) as e:
            _s2_err = e
    else:
        _s2_err = "不以```开头，跳过"

    # 策略 3：提取第一个 JSON 对象
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(cleaned[start:end + 1])
        except (json.JSONDecodeError, ValueError) as e:
            _s3_err = e
    else:
        _s3_err = f"未找到 {{}} 对 (start={start}, end={end})"

    # 策略 4：提取第一个 JSON 数组
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start != -1 and end > start:
        try:
            return json.loads(cleaned[start:end + 1])
        except (json.JSONDecodeError, ValueError) as e:
            _s4_err = e
    else:
        _s4_err = f"未找到 [] 对 (start={start}, end={end})"

    logger.debug("4 个策略全部失败: S1=%s | S2=%s | S3=%s | S4=%s", _s1_err, _s2_err, _s3_err, _s4_err)

    # 策略 5：截断 JSON 自动补全闭合
    # Claude 不支持 response_format，输出被 max_tokens 截断时 JSON 不完整
    # 尝试截断到最后一个完整的值，然后补全缺失的 ] 和 }
    logger.debug("JSON 解析前 4 个策略均失败，尝试截断修复")
    repaired = _try_repair_truncated_json(cleaned)
    if repaired is not None:
        return repaired

    raise LLMParseError(f"无法从 LLM 返回中解析 JSON: {text[:200]}...")


def _try_repair_truncated_json(text: str) -> Any:
    """尝试修复被截断的 JSON。

    思路：找到 JSON 起始位置，截断到最后一个完整的字符串值或数字值结尾，
    然后用括号栈计算缺失的闭合符号并补全。
    """
    # 找到 JSON 的起始 { 或 [
    obj_start = text.find("{")
    arr_start = text.find("[")
    if obj_start == -1 and arr_start == -1:
        return None
    if obj_start == -1:
        start = arr_start
    elif arr_start == -1:
        start = obj_start
    else:
        start = min(obj_start, arr_start)

    fragment = text[start:]
    if not fragment:
        return None

    # 尝试截断到最后一个完整的值边界，然后补全
    # 逐步从末尾回退，找到可以闭合的位置
    for cutoff in _find_truncation_points(fragment):
        candidate = fragment[:cutoff]
        closed = _close_json(candidate)
        if closed is not None:
            try:
                result = json.loads(closed)
                logger.warning("截断 JSON 自动修复成功（截断位置 %d/%d）", cutoff, len(fragment))
                return result
            except (json.JSONDecodeError, ValueError):
                continue

    return None


def _find_truncation_points(text: str) -> list[int]:
    """找到可能的截断点（从后往前），优先在完整值边界截断。"""
    points = []
    # 从末尾往前找这些边界字符：", }, ], 数字, true, false, null
    i = len(text)
    while i > 0:
        i -= 1
        ch = text[i]
        if ch in ('"', '}', ']'):
            points.append(i + 1)
        elif ch == ',' or ch == ':':
            # 逗号/冒号前截断（丢弃不完整的下一个键值对）
            points.append(i)
    return points


def _close_json(text: str) -> Optional[str]:
    """分析括号栈，补全缺失的闭合符号。

    返回补全后的字符串，如果无法补全返回 None。
    """
    # 去掉末尾不完整的键值对（以逗号或冒号结尾）
    stripped = text.rstrip()
    if stripped.endswith(","):
        stripped = stripped[:-1]
    if stripped.endswith(":"):
        # 冒号后面缺值，回退到冒号前的逗号或括号
        last_comma = stripped.rfind(",")
        last_brace = max(stripped.rfind("{"), stripped.rfind("["))
        cut = max(last_comma, last_brace)
        if cut > 0:
            stripped = stripped[:cut + 1] if stripped[cut] in ('{', '[') else stripped[:cut]
        else:
            return None

    # 计算括号栈
    stack = []
    in_string = False
    escape = False
    for ch in stripped:
        if escape:
            escape = False
            continue
        if ch == '\\' and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in ('{', '['):
            stack.append(ch)
        elif ch == '}':
            if stack and stack[-1] == '{':
                stack.pop()
        elif ch == ']':
            if stack and stack[-1] == '[':
                stack.pop()

    if not stack:
        # 已经闭合了
        return stripped

    # 按栈逆序补全闭合符号
    closers = []
    for opener in reversed(stack):
        closers.append('}' if opener == '{' else ']')

    return stripped + ''.join(closers)


# ── 配置校验 ──────────────────────────────────────────────────


def _validate_config() -> None:
    """前置校验 LLM 配置，提供清晰的错误提示"""
    if not cfg.llm.api_key:
        raise LLMConfigError("LLM_API_KEY 未配置，请在 .env 文件中填写")
    provider = cfg.llm.provider.lower()
    if provider not in _PROVIDERS:
        raise LLMConfigError(f"不支持的 LLM 提供商: {provider}，可选: {', '.join(_PROVIDERS)}")


# ── 公开接口：纯文本 ─────────────────────────────────────────


async def call_llm(
    system_prompt: str,
    user_message: str,
    *,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> str:
    """调用 LLM 返回纯文本。

    自动根据 cfg.llm.provider 选择 OpenAI 或 Anthropic。

    Raises:
        LLMConfigError: 配置缺失
        LLMError: API 调用失败（重试耗尽后）
    """
    _validate_config()

    _max_tokens = max_tokens or cfg.llm.max_tokens
    _temperature = temperature if temperature is not None else cfg.llm.temperature
    provider = cfg.llm.provider.lower()

    return await _PROVIDERS[provider](
        system_prompt=system_prompt,
        user_message=user_message,
        max_tokens=_max_tokens,
        temperature=_temperature,
    )


# ── 公开接口：结构化 JSON ────────────────────────────────────


async def call_llm_json(
    system_prompt: str,
    user_message: str,
    response_model: Optional[Type[BaseModel]] = None,
    *,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    max_parse_retries: int = 1,
) -> dict | list | BaseModel:
    """调用 LLM 并解析为结构化 JSON。

    如果提供 response_model（Pydantic BaseModel），会：
    1. 自动从 model 生成 JSON Schema 注入 prompt，引导 LLM 输出格式
    2. 解析后用 model_validate() 校验字段完整性和类型
    3. 解析失败时把错误反馈给 LLM，让它自我纠正后重试

    Args:
        system_prompt: 系统提示词
        user_message: 用户消息
        response_model: Pydantic 模型类，用于 schema 注入和输出校验
        max_tokens: 最大输出 token
        temperature: 温度参数
        max_parse_retries: JSON 解析失败后的最大重试次数

    Returns:
        - 有 response_model 时返回校验后的 Pydantic 实例
        - 无 response_model 时返回 dict 或 list

    Raises:
        LLMConfigError: 配置缺失
        LLMError: API 调用失败
        LLMParseError: JSON 解析/校验失败（重试耗尽后）
    """
    _validate_config()

    # 构造 JSON 引导 prompt
    json_hint = "\n\n请严格以 JSON 格式返回结果，不要包含其他文字。"
    if response_model:
        schema = json.dumps(response_model.model_json_schema(), ensure_ascii=False, indent=2)
        json_hint += f"\n\nJSON Schema:\n```json\n{schema}\n```"

    full_system_prompt = system_prompt + json_hint
    current_user_message = user_message

    _max_tokens = max_tokens or cfg.llm.max_tokens
    _temperature = temperature if temperature is not None else cfg.llm.temperature
    provider = cfg.llm.provider.lower()

    for attempt in range(max_parse_retries + 1):
        # 调用 LLM（OpenAI 启用原生 JSON mode）
        raw_text = await _PROVIDERS[provider](
            system_prompt=full_system_prompt,
            user_message=current_user_message,
            max_tokens=_max_tokens,
            temperature=_temperature,
            json_mode=(provider == "openai"),
        )

        try:
            parsed = _try_parse_json(raw_text)
            if response_model:
                return response_model.model_validate(parsed)
            return parsed
        except (LLMParseError, Exception) as e:
            if attempt < max_parse_retries:
                logger.warning("JSON 解析失败 (第 %d 次)，将反馈错误给 LLM 重试: %s", attempt + 1, e)
                # 把错误信息反馈给 LLM，让它自我纠正
                current_user_message = (
                    f"你上次的回答格式不正确，请修正。错误: {e}\n"
                    f"原始问题: {user_message}"
                )
                continue
            raise LLMParseError(f"JSON 解析/校验在 {max_parse_retries + 1} 次尝试后仍失败: {e}") from e

    # 理论上不会到这里，但保险起见
    raise LLMParseError("JSON 解析重试逻辑异常")


def _fix_unescaped_quotes_in_json_array(text: str) -> str:
    """修复 JSON 字符串数组中未转义的内嵌引号。

    113 代理把 tool input 序列化为字符串时，会把中文引号 "" 转为 ASCII "，
    导致 JSON 字符串值内出现未转义的 "。

    策略：逐字符扫描，跟踪是否在 JSON 字符串值内部。
    如果在字符串值内部遇到 " 且后面不是 , ] } : 等 JSON 分隔符，
    说明它是内容中的引号，替换为中文引号。
    """
    result = []
    in_string = False
    i = 0
    while i < len(text):
        ch = text[i]

        if ch == '\\' and in_string:
            # 转义字符，原样保留两个字符
            result.append(text[i:i+2])
            i += 2
            continue

        if ch == '"':
            if not in_string:
                # 开始一个字符串
                in_string = True
                result.append(ch)
            else:
                # 在字符串内遇到 "，判断它是字符串结束符还是内嵌引号
                # 向后看：跳过空白后，如果是 , ] } 或字符串结尾，则是合法的字符串结束符
                rest = text[i+1:].lstrip()
                if not rest or rest[0] in (',', ']', '}', ':'):
                    # 合法的字符串结束
                    in_string = False
                    result.append(ch)
                else:
                    # 内嵌的引号，替换为中文左/右引号
                    result.append('\u201c')
            i += 1
            continue

        result.append(ch)
        i += 1

    return ''.join(result)


def _deep_deserialize_json_strings(obj: Any) -> Any:
    """递归反序列化被代理错误序列化为字符串的 JSON 值。

    某些代理（如 113 中转）会把 tool_use 的 input 中的数组/对象
    序列化为 JSON 字符串而非原生类型。此函数递归修复。
    """
    if isinstance(obj, str):
        stripped = obj.strip()
        if stripped.startswith(("[", "{")):
            try:
                return json.loads(stripped)
            except (json.JSONDecodeError, ValueError):
                pass
            # 113 代理有时把中文引号 "" 转为 ASCII "，导致 JSON 字符串值内出现未转义的 "
            # 修复策略：用正则找到 JSON 字符串值内的裸 "，替换为转义形式
            try:
                fixed = _fix_unescaped_quotes_in_json_array(stripped)
                return json.loads(fixed)
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning("JSON 反序列化最终失败: %s | 前100字符: %r", e, stripped[:100])
                return obj
        return obj
    elif isinstance(obj, dict):
        return {k: _deep_deserialize_json_strings(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_deep_deserialize_json_strings(item) for item in obj]
    return obj


# ── 公开接口：tool_use 结构化提取 ────────────────────────────


async def call_llm_structured(
    system_prompt: str,
    user_message: str,
    tool_name: str,
    tool_description: str,
    tool_schema: dict,
    *,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    response_model: Optional[Type[BaseModel]] = None,
) -> dict | BaseModel:
    """通过 Anthropic tool_use 强制模型返回结构化 JSON。

    原理：定义一个工具并用 tool_choice 强制模型调用，
    模型返回的 tool input 天然是合法 JSON，不需要文本解析。

    适用场景：Claude 模型的结构化数据提取（替代 call_llm_json 的文本解析方式）。

    Args:
        system_prompt: 系统提示词
        user_message: 用户消息
        tool_name: 工具名称
        tool_description: 工具描述
        tool_schema: 工具输入的 JSON Schema（Pydantic model.model_json_schema()）
        max_tokens: 最大输出 token
        temperature: 温度参数
        response_model: 可选 Pydantic model，用于校验返回结果

    Returns:
        dict 或 Pydantic BaseModel 实例
    """
    _validate_config()

    _max_tokens = max_tokens or cfg.llm.max_tokens
    _temperature = temperature if temperature is not None else cfg.llm.temperature

    url = f"{cfg.llm.base_url.rstrip('/')}/messages"
    headers = {
        "Authorization": f"Bearer {cfg.llm.api_key}",
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }
    payload: dict[str, Any] = {
        "model": cfg.llm.model,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_message}],
        "max_tokens": _max_tokens,
        "temperature": _temperature,
        "tools": [{
            "name": tool_name,
            "description": tool_description,
            "input_schema": tool_schema,
        }],
        "tool_choice": {"type": "tool", "name": tool_name},
    }

    def _extract_tool_input(data: dict) -> Any:
        """从 Anthropic tool_use 响应中提取工具输入。"""
        for block in data.get("content", []):
            if block.get("type") == "tool_use" and block.get("name") == tool_name:
                inp = block["input"]
                # 113 代理有时把 input 整体序列化为 JSON 字符串
                if isinstance(inp, str):
                    try:
                        inp = json.loads(inp)
                    except (json.JSONDecodeError, ValueError):
                        pass
                return inp
        raise LLMParseError(f"响应中未找到 tool_use block (tool={tool_name})")

    raw_result = await _request_with_retry(url, headers, payload, extract_fn=_extract_tool_input)

    # 某些代理会把 tool input 整体或嵌套字段序列化为字符串，
    # 或把中文引号转为 ASCII 引号导致 JSON 无法解析
    raw_result = _deep_deserialize_json_strings(raw_result)

    if response_model:
        return response_model.model_validate(raw_result)
    return raw_result


# ── 公开接口：评分专用 LLM ────────────────────────────────────


async def call_scoring_llm_json(
    system_prompt: str,
    user_message: str,
    response_model: Optional[Type[BaseModel]] = None,
    *,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    max_parse_retries: int = 1,
) -> dict | list | BaseModel:
    """评分专用 LLM 调用。

    如果 SCORING_LLM_* 已配置，使用独立的评分模型（如 GPT-4o）；
    否则 fallback 到主 LLM，行为与 call_llm_json 一致。
    """
    if not cfg.scoring_llm.is_configured():
        return await call_llm_json(
            system_prompt, user_message, response_model,
            max_tokens=max_tokens, temperature=temperature,
            max_parse_retries=max_parse_retries,
        )

    # 构造 JSON 引导 prompt
    json_hint = "\n\n请严格以 JSON 格式返回结果，不要包含其他文字。"
    if response_model:
        schema = json.dumps(response_model.model_json_schema(), ensure_ascii=False, indent=2)
        json_hint += f"\n\nJSON Schema:\n```json\n{schema}\n```"

    full_system_prompt = system_prompt + json_hint
    current_user_message = user_message

    _max_tokens = max_tokens or cfg.llm.max_tokens
    _temperature = temperature if temperature is not None else cfg.llm.temperature

    # 评分模型配置覆盖
    s = cfg.scoring_llm
    url = f"{s.base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {s.api_key}",
        "Content-Type": "application/json",
    }

    for attempt in range(max_parse_retries + 1):
        payload: dict[str, Any] = {
            "model": s.model,
            "messages": [
                {"role": "system", "content": full_system_prompt},
                {"role": "user", "content": current_user_message},
            ],
            "max_tokens": _max_tokens,
            "temperature": _temperature,
        }
        # 评分模型非 Claude 时启用 JSON mode
        if "claude" not in s.model.lower():
            payload["response_format"] = {"type": "json_object"}

        _scoring_retries = s.max_retries if s.max_retries > 0 else None
        raw_text = await _request_with_retry(
            url, headers, payload,
            extract_fn=_extract_openai,
            max_retries_override=_scoring_retries,
        )

        try:
            parsed = _try_parse_json(raw_text)
            if response_model:
                return response_model.model_validate(parsed)
            return parsed
        except (LLMParseError, Exception) as e:
            if attempt < max_parse_retries:
                logger.warning("评分 JSON 解析失败 (第 %d 次): %s", attempt + 1, e)
                current_user_message = (
                    f"你上次的回答格式不正确，请修正。错误: {e}\n"
                    f"原始问题: {user_message}"
                )
                continue
            raise LLMParseError(f"评分 JSON 解析在 {max_parse_retries + 1} 次尝试后仍失败: {e}") from e

    raise LLMParseError("评分 JSON 解析重试逻辑异常")


# ── 响应提取函数（Tool Calling）───────────────────────────────


def _extract_openai_full(data: dict) -> Any:
    """从 OpenAI 响应中提取完整信息（含 tool_calls）。"""
    from models import LLMToolResponse, ToolCallInfo

    choice = data["choices"][0]["message"]
    content = choice.get("content")
    tool_calls_raw = choice.get("tool_calls", [])

    tool_calls = []
    for tc in tool_calls_raw:
        func = tc.get("function", {})
        args_str = func.get("arguments", "{}")
        try:
            args = json.loads(args_str)
        except (json.JSONDecodeError, ValueError):
            try:
                args = _try_parse_json(args_str)
                if not isinstance(args, dict):
                    args = {}
            except LLMParseError:
                args = {}

        tool_calls.append(ToolCallInfo(
            id=tc.get("id", ""),
            function_name=func.get("name", ""),
            arguments=args,
        ))

    return LLMToolResponse(content=content, tool_calls=tool_calls)


def _extract_claude_full(data: dict) -> Any:
    """从 Anthropic Messages API 响应中提取完整信息（含 tool_use）。"""
    from models import LLMToolResponse, ToolCallInfo

    # DEBUG: 打印 Claude 返回的所有 block 类型
    block_types = [b.get("type", "?") for b in data.get("content", [])]
    logger.debug("Claude 响应 block 类型: %s", block_types)

    text_parts = []
    tool_calls = []
    for block in data.get("content", []):
        btype = block.get("type", "")
        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "thinking":
            # Claude extended thinking block — 也作为思考内容保留
            thinking_text = block.get("thinking", "")
            if thinking_text:
                logger.debug("发现 thinking block: %d 字符", len(thinking_text))
                text_parts.append(thinking_text)
        elif btype == "tool_use":
            tool_calls.append(ToolCallInfo(
                id=block.get("id", ""),
                function_name=block.get("name", ""),
                arguments=block.get("input", {}),
            ))
        else:
            logger.warning("Claude 响应中发现未知 block 类型: %s", btype)

    content = "\n".join(text_parts).strip() or None
    if content:
        logger.debug("Claude content 长度: %d 字符", len(content))
    else:
        logger.debug("Claude content 为空 (block_types=%s)", block_types)
    return LLMToolResponse(content=content, tool_calls=tool_calls)


# ── Claude Tool Calling 格式转换 ─────────────────────────────


def _openai_tools_to_claude(tools: list[dict]) -> list[dict]:
    """OpenAI function calling 工具 schema → Claude 工具 schema。

    OpenAI: {type: "function", function: {name, description, parameters}}
    Claude: {name, description, input_schema}
    """
    result = []
    for t in tools:
        if "input_schema" in t:
            result.append(t)
            continue
        fn = t.get("function", t)
        result.append({
            "name": fn["name"],
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
        })
    return result


def _openai_messages_to_claude(messages: list[dict]) -> tuple[str, list[dict]]:
    """OpenAI messages 格式 → Claude messages 格式。

    提取 system 消息到单独字段，将 tool role 消息转为 tool_result block，
    合并连续的 tool_result 到同一个 user 消息中。

    Returns:
        (system_prompt, claude_messages)
    """
    system_prompt = ""
    claude_msgs: list[dict] = []

    for msg in messages:
        role = msg.get("role", "")

        if role == "system":
            system_prompt = msg.get("content", "")

        elif role == "user":
            claude_msgs.append({"role": "user", "content": msg["content"]})

        elif role == "assistant":
            # 转换 assistant 消息：可能含 tool_calls
            blocks: list[dict] = []
            if msg.get("content"):
                blocks.append({"type": "text", "text": msg["content"]})
            for tc in msg.get("tool_calls", []):
                fn = tc.get("function", {})
                args_str = fn.get("arguments", "{}")
                try:
                    args = json.loads(args_str) if isinstance(args_str, str) else args_str
                except (json.JSONDecodeError, ValueError):
                    args = {}
                blocks.append({
                    "type": "tool_use",
                    "id": tc.get("id", ""),
                    "name": fn.get("name", ""),
                    "input": args,
                })
            if blocks:
                claude_msgs.append({"role": "assistant", "content": blocks})

        elif role == "tool":
            # tool result → 追加到 user 消息的 tool_result block
            tool_result_block = {
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id", ""),
                "content": msg.get("content", ""),
            }
            # 如果上一条已经是 user（含 tool_result），合并进去
            if claude_msgs and claude_msgs[-1]["role"] == "user" and isinstance(claude_msgs[-1]["content"], list):
                claude_msgs[-1]["content"].append(tool_result_block)
            else:
                claude_msgs.append({"role": "user", "content": [tool_result_block]})

    return system_prompt, claude_msgs


# ── 公开接口：Tool Calling ────────────────────────────────────


async def call_llm_with_tools(
    messages: list[dict],
    tools: list[dict],
    *,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> Any:
    """调用 LLM，支持多轮消息和函数调用。

    Claude 模型自动走 Anthropic Messages API（/v1/messages），
    其他模型走 OpenAI chat/completions API。

    Args:
        messages: 完整对话历史（system/user/assistant/tool 消息）
        tools: OpenAI function calling 工具 schema 列表
        max_tokens: 最大输出 token
        temperature: 温度参数

    Returns:
        LLMToolResponse，含文本内容和/或工具调用
    """
    _validate_config()

    _max_tokens = max_tokens or cfg.llm.max_tokens
    _temperature = temperature if temperature is not None else cfg.llm.temperature

    if _is_claude_model():
        return await _call_claude_with_tools(messages, tools, _max_tokens, _temperature)
    else:
        return await _call_openai_with_tools(messages, tools, _max_tokens, _temperature)


async def _call_openai_with_tools(
    messages: list[dict],
    tools: list[dict],
    max_tokens: int,
    temperature: float,
) -> Any:
    """OpenAI chat/completions 路径（非 Claude 模型）。"""
    url = f"{cfg.llm.base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {cfg.llm.api_key}",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {
        "model": cfg.llm.model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if tools:
        payload["tools"] = tools

    return await _request_with_retry(url, headers, payload, extract_fn=_extract_openai_full)


async def _call_claude_with_tools(
    messages: list[dict],
    tools: list[dict],
    max_tokens: int,
    temperature: float,
) -> Any:
    """Anthropic Messages API 路径（Claude 模型）。

    通过代理的 /v1/messages 端点发送，使用 Claude 原生格式。
    """
    url = f"{cfg.llm.base_url.rstrip('/')}/messages"
    headers = {
        "Authorization": f"Bearer {cfg.llm.api_key}",
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }

    system_prompt, claude_messages = _openai_messages_to_claude(messages)

    payload: dict[str, Any] = {
        "model": cfg.llm.model,
        "messages": claude_messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if system_prompt:
        payload["system"] = system_prompt
    if tools:
        payload["tools"] = _openai_tools_to_claude(tools)

    return await _request_with_retry(url, headers, payload, extract_fn=_extract_claude_full)
