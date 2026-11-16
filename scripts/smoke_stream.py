"""上游流式自检：不经过 FastAPI，直接看增量、首字延迟与 usage"""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.llm.deepseek import StreamResult, stream_completion


async def main() -> None:
    # 承接上游统计：模型名与 token 数在流结束后才可读
    result = StreamResult()
    started = time.monotonic()
    # 首字到达时刻，0 表示还没收到第一个片
    first_at = 0.0

    async for piece in stream_completion("用两句话说明什么是 RAG", result):
        if not first_at:
            first_at = time.monotonic() - started
            print(f"[首字延迟 {first_at * 1000:.0f}ms]")
        print(piece, end="", flush=True)

    total = time.monotonic() - started
    print(f"\n\n[共 {result.chunk_count} 片，总耗时 {total * 1000:.0f}ms]")
    print(f"[模型 {result.model or '未知'}]")
    print(
        f"[tokens prompt={result.prompt_tokens} completion={result.completion_tokens}"
        f" usage收到={result.usage_received}]"
    )


asyncio.run(main())
