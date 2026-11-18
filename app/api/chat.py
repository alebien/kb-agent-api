"""对话流式接口：把上游模型的增量转成契约里的 6 个事件"""

import asyncio
import logging
import secrets
import time
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.config import DEEPSEEK_MODEL
from app.intent import (
    ANSWER_CONFIDENCE,
    INTENT_THRESHOLD,
    STRONG_CITATION_CONFIDENCE,
    intent_confidence,
)
from app.llm.deepseek import StreamResult, UpstreamError, stream_completion
from app.retrieval.bm25 import search
from app.sse import sse_comment, sse_event
from app.tools.order import find_order_no, lookup_order

logger = logging.getLogger("knoagent.chat")

router = APIRouter()


class ChatRequest(BaseModel):
    """对话请求体：字段校验交给 Pydantic，非法输入自动 422 且带字段级说明"""

    # 对应 PRD FR-3-1「单条提问不超过 2000 字」，此前只在输入框限制，直连接口可绕过
    prompt: str = Field(min_length=1, max_length=2000, description="用户提问")
    # 前端上屏用的临时 id，仅用于日志回显与幂等键，真实后端不依赖它
    answerId: str | None = Field(default=None, max_length=128, description="前端临时消息 id")


# 心跳间隔：空闲超过它就补一个注释行，避免网关把长连接当死连接掐掉
HEARTBEAT_INTERVAL_MS = 1200

# 断流模拟位置：scenario=broken 时收到这么多片后中断，用来验 error 事件
# 用固定片数而不是比例：调试开关的行为要可预测，不随上游实际片数漂移
BROKEN_AFTER_CHUNKS = 20


@router.post("/api/chat")
async def chat(payload: ChatRequest, request: Request) -> StreamingResponse:
    """接收提问并流式返回回答，事件顺序 start → tool → citation → tool → delta×N → done"""
    # 校验已由 Pydantic 完成，这里只去掉首尾空白；纯空白按空提问处理
    prompt = payload.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt 不能为空")

    # 调试开关：前端 SSE 调试面板用它模拟断流，生产环境不会传
    broken = request.query_params.get("scenario") == "broken"
    started_at = datetime.now(UTC)
    # 每次请求独立 id：时间戳只到秒，同秒并发会撞，补 4 位随机后缀
    answer_id = f"srv-{int(started_at.timestamp())}-{secrets.token_hex(2)}"

    async def stream():
        """异步生成器：逐条吐事件；结束时用日志区分正常、断开、失败三种收尾"""
        seq = 1
        # 累计已发出的字符数，断开时用来说明「说到哪儿了」
        sent_chars = 0
        # 上一次发心跳的时间点，毫秒
        last_ping_at = time.monotonic() * 1000
        # 是否已把 done 交出去：前端收到 done 会立刻 cancel 连接
        # 靠它区分「正常收尾」与「中途断开」，否则每次成功请求都会打一条断开日志
        sent_done = False
        # 承接上游统计：模型名、token 数、片数，生成器结束后才可读
        result = StreamResult()
        # 意图分与检索无关，先算好：后面订单工具的门槛用它，done 事件也要带出去
        intent_score = intent_confidence(prompt)

        try:
            yield sse_event(
                "start",
                {
                    "answerId": answer_id,
                    "model": DEEPSEEK_MODEL,
                    "createdAt": started_at.isoformat(),
                },
                seq,
            )

            # 检索占位：先发 running，把首字到达前的一两秒空档变成可见的「正在检索」
            tool_id = f"{answer_id}-search"
            search_started = time.monotonic()
            seq += 1
            yield sse_event(
                "tool",
                {
                    "id": tool_id,
                    "name": "search_knowledge",
                    "status": "running",
                    "arguments": {"query": prompt, "top_k": 5},
                    "startAt": datetime.now(UTC).isoformat(),
                },
                seq,
            )

            # 真实检索：BM25 召回，命中几条发几条 citation
            # 契约硬口径：citation 必须全部早于第一个 delta，否则前端没法先把来源卡片挂上去
            citations = search(prompt)
            # 检索耗时只算 search 本身，单位毫秒，D9 前端第四行质量行要用
            # 语料只有 16 块，检索常在 1ms 内完成，实测为 0 时兜底成 1，避免前端把 0 读成「未测量」
            retrieval_ms = max(1, int((time.monotonic() - search_started) * 1000))
            for item in citations:
                # 块 id 只保证块内唯一，加上 answerId 前缀避免跨请求撞 id
                chunk_id = item["id"]
                payload = {key: value for key, value in item.items() if key != "id"}
                seq += 1
                yield sse_event("citation", {"id": f"{answer_id}-{chunk_id}", **payload}, seq)

            # 工具结束必须与 running 用同一个 id，前端按 id 覆盖同一张卡片
            seq += 1
            yield sse_event(
                "tool",
                {
                    "id": tool_id,
                    "name": "search_knowledge",
                    "status": "success",
                    "arguments": {"query": prompt, "top_k": 5},
                    "result": {
                        "hits": len(citations),
                        # 零命中时没有最高分，给 0 而不是 null，前端按数字处理
                        "topScore": citations[0]["score"] if citations else 0,
                    },
                    "durationMs": retrieval_ms,
                },
                seq,
            )

            # 订单查询：命中订单号才发，位置固定在检索之后、首个 delta 之前
            order_no = find_order_no(prompt)
            if order_no is not None and intent_score >= INTENT_THRESHOLD:
                order_tool_id = f"{answer_id}-order"
                order_started = time.monotonic()
                seq += 1
                yield sse_event(
                    "tool",
                    {
                        "id": order_tool_id,
                        "name": "get_order",
                        "status": "running",
                        "arguments": {"orderNo": order_no},
                        "startAt": datetime.now(UTC).isoformat(),
                    },
                    seq,
                )

                # 查不到也发 success：工具没查到是正常结果，不是故障
                order_result = lookup_order(order_no)
                order_ms = max(0, int((time.monotonic() - order_started) * 1000))
                seq += 1
                yield sse_event(
                    "tool",
                    {
                        "id": order_tool_id,
                        "name": "get_order",
                        "status": "success",
                        "arguments": {"orderNo": order_no},
                        "result": order_result,
                        "durationMs": order_ms,
                    },
                    seq,
                )

            # 断流阈值：-1 表示不中断
            broken_at = -1
            if broken:
                broken_at = BROKEN_AFTER_CHUNKS

            async for piece in stream_completion(prompt, result):
                if result.chunk_count == broken_at:
                    # 模拟上游中途失败：走 error 事件而不是断连，前端才能显示原因
                    seq += 1
                    yield sse_event("error", {"message": "模型服务响应超时"}, seq)
                    return

                seq += 1
                sent_chars += len(piece)
                yield sse_event("delta", {"text": piece}, seq)

                # 空闲超过阈值就插一条注释行保活，不占用事件序号
                now_ms = time.monotonic() * 1000
                if now_ms - last_ping_at >= HEARTBEAT_INTERVAL_MS:
                    last_ping_at = now_ms
                    yield sse_comment(f"ping {int(now_ms)}")

            seq += 1
            duration_ms = int((datetime.now(UTC) - started_at).total_seconds() * 1000)
            # 先置标记再吐 done：客户端很可能在收到 done 的同一次读取里就断开
            sent_done = True
            answer_confidence = ANSWER_CONFIDENCE["none"]
            if citations:
                answer_confidence = (
                    ANSWER_CONFIDENCE["strong"]
                    if citations[0].get("confidence", 0.0) >= STRONG_CITATION_CONFIDENCE
                    else ANSWER_CONFIDENCE["weak"]
                )
            yield sse_event(
                "done",
                {
                    "confidence": answer_confidence,
                    "intentConfidence": intent_score,
                    "model": result.model or DEEPSEEK_MODEL,
                    "durationMs": duration_ms,
                    "tokens": result.completion_tokens,
                    # 质量行四步链路的第二段：检索耗时，D9 前端加「检索」那一行时要读
                    "breakdown": {"retrievalMs": retrieval_ms},
                },
                seq,
            )
            logger.info(
                "回答正常结束 answerId=%s 事件=%d 字符=%d 片数=%d tokens=%d usage=%s 耗时=%dms",
                answer_id,
                seq,
                sent_chars,
                result.chunk_count,
                result.completion_tokens,
                result.usage_received,
                duration_ms,
            )

        except asyncio.CancelledError:
            # 客户端关连接时，Starlette 会取消这个生成器任务，抛的就是它
            if sent_done:
                logger.info("正常结束（done 后客户端关闭连接）answerId=%s 事件=%d", answer_id, seq)
            else:
                logger.warning(
                    "客户端断开 answerId=%s 已发事件=%d 已发字符=%d",
                    answer_id,
                    seq,
                    sent_chars,
                )
            # 必须重新抛出，否则取消信号被吞，uvicorn 会认为任务正常结束
            raise

        except UpstreamError as exc:
            # 上游错误（401、余额不足、超时）属于流内失败，只能发事件，改不了状态码
            logger.warning("上游失败 answerId=%s 原因=%s", answer_id, exc)
            yield sse_event("error", {"message": f"模型服务暂不可用：{exc}"}, seq + 1)

        except Exception:
            # 兜底：任何未预期异常都要变成 error 事件，否则前端只看到断连、拿不到原因
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
