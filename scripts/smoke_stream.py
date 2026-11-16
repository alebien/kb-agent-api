"""上游流式自检：不经过 FastAPI，直接看增量与首字延迟"""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.llm.deepseek import stream_completion  # noqa: E402


async def main() -> None:
    started = time.monotonic()
    first_at = 0.0
    total = 0

    async for piece in stream_completion("用两句话说明什么是 RAG"):
        if not first_at:
            first_at = time.monotonic() - started
            print(f"[首字延迟 {first_at * 1000:.0f}ms]")
        print(piece, end="", flush=True)
        total += 1

    print(f"\n\n[共 {total} 片，总耗时 {(time.monotonic() - started) * 1000:.0f}ms]")


asyncio.run(main())
