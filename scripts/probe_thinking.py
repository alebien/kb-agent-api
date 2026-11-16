"""探测思考模式参数：对比关闭/低/默认三种设置的首字延迟"""

import asyncio
import json
import sys
import time
from pathlib import Path

import httpx

# 让脚本能 import app 包（从 kb-agent-api 根目录跑时可以省掉这段）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, require_api_key

QUESTION = "用两句话说明什么是 RAG"

VARIANTS = {
    "默认": {},
    "thinking disabled": {"thinking": {"type": "disabled"}},
    "reasoning_effort low": {"reasoning_effort": "low"},
    "reasoning none": {"reasoning": {"effort": "none"}},
}


async def probe(name: str, extra: dict) -> None:
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [{"role": "user", "content": QUESTION}],
        "stream": True,
        "stream_options": {"include_usage": True},
        **extra,
    }
    headers = {"Authorization": f"Bearer {require_api_key()}"}
    started = time.monotonic()
    first_at = 0.0
    usage = {}

    try:
        async with (
            httpx.AsyncClient(timeout=httpx.Timeout(None, connect=10.0)) as client,
            client.stream(
                "POST", f"{DEEPSEEK_BASE_URL}/chat/completions", json=payload, headers=headers
            ) as resp,
        ):
            if resp.status_code != 200:
                body = (await resp.aread()).decode("utf-8", "ignore")[:200]
                print(f"{name:22} 失败 {resp.status_code}: {body}")
                return
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if raw == "[DONE]":
                    break
                chunk = json.loads(raw)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                delta = (chunk.get("choices") or [{}])[0].get("delta", {})
                if delta.get("content") and not first_at:
                    first_at = time.monotonic() - started
    except Exception as exc:  # noqa: BLE001
        print(f"{name:22} 异常 {type(exc).__name__}: {exc}")
        return

    total = time.monotonic() - started
    reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", "?")
    print(
        f"{name:22} 首字 {first_at * 1000:6.0f}ms  总 {total * 1000:6.0f}ms  思考token {reasoning}"
    )


async def main() -> None:
    for name, extra in VARIANTS.items():
        await probe(name, extra)
        await asyncio.sleep(0.5)


asyncio.run(main())
