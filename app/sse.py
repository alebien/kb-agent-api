"""SSE 事件序列化，等价于前端 src/api/sse.ts 的反向操作"""

import json


def sse_event(name: str, payload: dict, seq: int) -> str:
    """把一条事件序列化成 SSE 文本块

    末尾的空行是事件边界，少一个前端会把这条并进下一条
    seq 写进 id 字段，断线重连时用 Last-Event-ID 回放
    """
    data = json.dumps(payload, ensure_ascii=False)
    return f"id: {seq}\nevent: {name}\ndata: {data}\n\n"


def sse_comment(text: str) -> str:
    """心跳注释行，前端解析器按规范跳过 `:` 开头的行

    只用于保活：代理和网关常在连接空闲时掐断，定期吐一个注释行即可避免，
    但它不产生任何业务事件，前端不会回调
    """
    return f": {text}\n\n"
