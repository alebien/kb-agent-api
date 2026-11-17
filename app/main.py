"""KnoAgent 后端入口：只负责建应用，后续再挂路由与中间件"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.chat import router as chat_router
from app.retrieval.bm25 import warmup

# 没有这句，自己的 logger.info 会被丢弃：uvicorn 只配置它自己的 logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """启动时预热词典与索引：jieba 首次加载词典约 0.5 秒，不预热会算进首次提问的首字延迟"""
    warmup()
    yield


app = FastAPI(title="KnoAgent API", lifespan=lifespan)
app.include_router(chat_router)


@app.get("/health")  # 装饰器 将下面的函数“注册”到目标路由上
async def health() -> dict[str, str]:  # 注册函数 async 同ts异步
    """存活探针：确认服务与端口正常，D10 部署时的健康检查也用它"""  # docstring 会出现在docs文档里
    return {"status": "ok"}  # dict字面量，等价 TS 的对象字面量
