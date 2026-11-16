"""DeepSeek 非流式连通性自检：验 key、网络、模型名，与流式解析无关"""

import sys
import time
from pathlib import Path

import httpx

# 让脚本能 import app 包（从 kb-agent-api 根目录跑时可以省掉这段）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, require_api_key  # noqa: E402


def main() -> None:
    api_key = require_api_key()
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [{"role": "user", "content": "用一句话说明什么是 SSE"}],
        "stream": False,
    }
    started = time.monotonic()

    resp = httpx.post(
        f"{DEEPSEEK_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
        timeout=30.0,
    )
    elapsed_ms = int((time.monotonic() - started) * 1000)

    print(f"状态码 {resp.status_code} | 耗时 {elapsed_ms}ms")
    if resp.status_code != 200:
        print("响应体:", resp.text[:500])
        return

    data = resp.json()
    print("模型:", data.get("model"))
    print("回答:", data["choices"][0]["message"]["content"])
    print("用量:", data.get("usage"))


if __name__ == "__main__":
    main()
