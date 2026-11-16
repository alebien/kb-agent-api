"""模拟客户端中途断开：收够 N 条事件后主动关闭连接"""

import asyncio

import httpx

URL = "http://127.0.0.1:8000/api/chat"
EVENT_LIMIT = 1  # 收够多少条事件后断开
SLEEP_PER_LINE_MS = 20  # 每行之间停一下，模拟人眼看的速度


async def main() -> None:
    count = 0
    async with (
        httpx.AsyncClient(timeout=httpx.Timeout(None, connect=5.0)) as client,
        client.stream("POST", URL, json={"prompt": "test"}) as resp,
    ):
        async for line in resp.aiter_lines():
            print(line)
            # 空行是一条事件的结束标记，用它计数才等于「收了几条事件」
            if line == "":
                count += 1
                if count >= EVENT_LIMIT:
                    print(f"—— 已收 {count} 条事件，模拟客户端断开 ——")
                    break
            await asyncio.sleep(SLEEP_PER_LINE_MS / 1000)


asyncio.run(main())
