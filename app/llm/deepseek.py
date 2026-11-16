"""DeepSeek 流式客户端：只负责从上游取增量文本与用量，不关心业务事件"""

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import httpx

from app.config import DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, DEEPSEEK_THINKING, require_api_key


class UpstreamError(RuntimeError):
    """上游返回非 200 或流中出现异常时抛出，由调用方转成 error 事件"""


@dataclass
class StreamResult:
    """一次流式调用的最终统计，文本吐完之后由调用方读取

    用可变对象而不是 yield 一个元组：生成器结束后调用方才能拿到最终值，
    这样 delta 循环里不需要为「最后一个包」写特殊分支
    """

    # 上游实际使用的模型名，可能与请求里传的不同，以响应为准
    model: str = ""
    # 输出 token 数，取 completion_tokens；缺失时为 0
    completion_tokens: int = 0
    # 输入 token 数，取 prompt_tokens；缺失时为 0
    prompt_tokens: int = 0
    # 是否收到过 usage 包，用于区分「上游没给」与「给了 0」
    usage_received: bool = False
    # 累计收到的文本片数，便于日志与排障
    chunk_count: int = 0
    # 上游返回的原始 usage，排障时打印用
    raw_usage: dict = field(default_factory=dict)


async def stream_completion(prompt: str, result: StreamResult) -> AsyncIterator[str]:
    """按增量吐上游返回的文本片段，同时把最终统计写进 result

    只处理「一个 token 一个 delta」这一种情况：
    - 非 data: 行（空行、注释）直接跳过
    - [DONE] 是上游结束哨兵，收到即停止，它不等于我们的 done 事件
    - delta.content 可能为空（首包只有 role、尾包只有 finish_reason），空就跳过
    - usage 默认不在流里返回，靠 stream_options.include_usage 让它多发一包
    """
    api_key = require_api_key()
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        # 流式下 usage 默认不返回，加这个才会在 [DONE] 之前多一包带 usage 的数据
        "stream_options": {"include_usage": True},
    }
    # 只对推理型模型传这个字段；传错值会被上游 400
    if DEEPSEEK_THINKING in {"enabled", "disabled"}:
        payload["thinking"] = {"type": DEEPSEEK_THINKING}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }

    # 读超时设 None：默认 5 秒会把长回答掐断；连接超时保留 10 秒以便快速失败
    timeout = httpx.Timeout(None, connect=10.0)
    async with (
        httpx.AsyncClient(timeout=timeout) as client,
        client.stream(
            "POST", f"{DEEPSEEK_BASE_URL}/chat/completions", json=payload, headers=headers
        ) as resp,
    ):
        if resp.status_code != 200:
            # 上游错误体最多截 300 字，避免把整页 HTML 塞进日志
            detail = (await resp.aread()).decode("utf-8", "ignore")[:300]
            raise UpstreamError(f"上游返回 {resp.status_code}：{detail}")

        async for line in resp.aiter_lines():
            # 空行是事件分隔，不是数据；非 data: 前缀的行（注释、event:）一律跳过
            if not line or not line.startswith("data:"):
                continue

            raw = line[5:].strip()
            if raw == "[DONE]":
                return

            try:
                chunk = json.loads(raw)
            except json.JSONDecodeError:
                # 上游偶发半包时容错跳过，不因一行坏数据中断整条流
                # 少一个字不影响理解，整条流挂掉用户会看到错误页，所以降级优于失败
                continue

            # 模型名以响应为准：请求 deepseek-chat 时上游可能回别的实际模型
            if chunk.get("model"):
                result.model = str(chunk["model"])

            # usage 可能在流末尾单独一包里，choices 为空，不能只看 choices
            usage = chunk.get("usage")
            if usage:
                # 只取最后一次：上游可能重复下发，累加会导致 token 翻倍
                result.usage_received = True
                result.raw_usage = usage
                result.completion_tokens = int(usage.get("completion_tokens") or 0)
                result.prompt_tokens = int(usage.get("prompt_tokens") or 0)

            choices = chunk.get("choices") or []
            if not choices:
                continue

            piece = choices[0].get("delta", {}).get("content") or ""
            if piece:
                result.chunk_count += 1
                yield piece
