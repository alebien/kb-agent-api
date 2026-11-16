"""对话流式接口：上午用固定文案验证事件契约与字节流，下午换成真模型"""

import asyncio
import logging
import secrets
import time
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.sse import sse_comment, sse_event

logger = logging.getLogger("knoagent.chat")

router = APIRouter()

FIXED_ANSWER = """## 发票处理建议

已为你检索到发票相关规则，可以先按下面步骤确认：

1. 核对开票主体与税号是否正确
2. 电子普票在开具后 **30 天内** 可申请换开
3. 跨月换开需要先红冲原发票

> 如果订单已报销，请先确认财务是否接受红冲后的新票
"""

CHUNK_SIZE = 3
CHUNK_INTERVAL_MS = 30
HEARTBEAT_INTERVAL_MS = 1200


def normalize_chunk_size(raw: str | None) -> int:
    """解析 ?chunkSize=N，非数字回落到默认值；调试面板靠它切换颗粒度"""
    try:
        value = int(raw) if raw is not None else CHUNK_SIZE
    except ValueError:
        return CHUNK_SIZE
    return max(1, min(40, value))


@router.post("/api/chat")
async def chat(request: Request) -> StreamingResponse:
    """接收提问并流式返回回答，事件顺序 start → delta×N → done"""
    body = await request.json()
    prompt = str(body.get("prompt", "")).strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt 不能为空")

    chunk_size = normalize_chunk_size(request.query_params.get("chunkSize"))
    started_at = datetime.now(UTC)
    answer_id = f"srv-{int(started_at.timestamp())}-{secrets.token_hex(2)}"

    async def stream():
        """异步生成器：逐条吐事件；结束时用日志区分正常、断开、失败三种收尾"""
        seq = 1
        # 累计已发出的字符数，断开时用来说明「说到哪儿了」
        sent_chars = 0
        # 上一次发心跳的时间点，毫秒
        last_ping_at = time.monotonic() * 1000
        # 是否已把 done 交出去：前端收到 done 会立刻 cancel 连接
        # 靠它区分「正常收尾」与「中途断开」
        sent_done = False

        try:
            yield sse_event(
                "start",
                {
                    "answerId": answer_id,
                    "model": "kb-agent-fixed",
                    "createdAt": started_at.isoformat(),
                },
                seq,
            )

            # 按 chunk_size 切片，模拟模型一次 flush 携带的字符数
            for index in range(0, len(FIXED_ANSWER), chunk_size):
                seq += 1
                piece = FIXED_ANSWER[index : index + chunk_size]
                sent_chars += len(piece)
                yield sse_event("delta", {"text": piece}, seq)

                # 空闲超过阈值就插一条注释行保活，不占用事件序号
                now_ms = time.monotonic() * 1000
                if now_ms - last_ping_at >= HEARTBEAT_INTERVAL_MS:
                    last_ping_at = now_ms
                    yield sse_comment(f"ping {int(now_ms)}")

                await asyncio.sleep(CHUNK_INTERVAL_MS / 1000)

            seq += 1
            duration_ms = int((datetime.now(UTC) - started_at).total_seconds() * 1000)
            # 先置标记再吐 done：客户端很可能在收到 done 的同一次读取里就断开
            sent_done = True
            yield sse_event(
                "done",
                {
                    "confidence": 88,
                    "model": "kb-agent-fixed",
                    "durationMs": duration_ms,
                    "tokens": len(FIXED_ANSWER),
                },
                seq,
            )
            logger.info(
                "回答正常结束 answerId=%s 事件=%d 字符=%d 耗时=%dms",
                answer_id,
                seq,
                sent_chars,
                duration_ms,
            )
        except asyncio.CancelledError:
            # 客户端关连接时，Starlette 会取消这个生成器任务，抛的就是它
            if sent_done:
                logger.info("正常结束（done 后客户端关闭连接）answerId=%s 事件=%d", answer_id, seq)
            else:
                logger.warning(
                    "客户端断开 answerId=%s 已发事件=%d 已发字符=%d/%d",
                    answer_id,
                    seq,
                    sent_chars,
                    len(FIXED_ANSWER),
                )
            # 必须重新抛出，否则取消信号被吞，uvicorn 会认为任务正常结束
            raise

        except Exception:
            logger.exception("生成失败 answerId=%s 事件=%d", answer_id, seq)
            yield sse_event("error", {"message": "生成失败，请稍后重试"}, seq + 1)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )
