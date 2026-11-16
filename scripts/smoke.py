"""流式接口自检：发一条中文提问，把收到的事件按顺序打印出来"""

import asyncio

import httpx

URL = "http://127.0.0.1:8000/api/chat"


async def main() -> None:
    payload = {"prompt": "发票怎么换开", "answerId": "smoke"}
    async with (
        httpx.AsyncClient(timeout=httpx.Timeout(None, connect=5.0)) as client,
        client.stream("POST", URL, json=payload) as resp,
    ):
        print("状态码:", resp.status_code, "| Content-Type:", resp.headers.get("content-type"))
        async for line in resp.aiter_lines():
            if line:
                print(line)


asyncio.run(main())
