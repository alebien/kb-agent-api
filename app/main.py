"""KnoAgent 后端入口：只负责建应用，后续再挂路由与中间件"""

from fastapi import FastAPI

app = FastAPI(title="KnoAgent API")


@app.get("/health")
async def health() -> dict[str, str]:
    """存活探针：确认服务与端口正常，D10 部署时的健康检查也用它"""
    return {"status": "ok"}