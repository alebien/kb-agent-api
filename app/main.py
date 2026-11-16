"""KnoAgent 后端入口：只负责建应用，后续再挂路由与中间件"""

import logging

from fastapi import FastAPI  # 导入

from app.api.chat import router as chat_router  # 新增：导入路由

# 没有这句，自己的 logger.info 会被丢弃：uvicorn 只配置它自己的 logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

app = FastAPI(title="KnoAgent API")  # 调用
app.include_router(chat_router)  # 新增：挂载


@app.get("/health")  # 装饰器 将下面的函数“注册”到目标路由上
async def health() -> dict[str, str]:  # 注册函数 async 同ts异步
    """存活探针：确认服务与端口正常，D10 部署时的健康检查也用它"""  # docstring 会出现在docs文档里
    return {"status": "ok"}  # dict字面量，等价 TS 的对象字面量
