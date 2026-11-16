"""环境变量集中读取，其他地方不直接碰 os.environ"""

import os

from dotenv import load_dotenv

# 读取同目录上层的 .env；不存在时静默跳过，方便生产环境直接用系统环境变量
load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")


def require_api_key() -> str:
    """取 key，缺失时给出可操作的错误而不是 401 玄学"""
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("缺少 DEEPSEEK_API_KEY，请在 kb-agent-api/.env 中配置")
    return DEEPSEEK_API_KEY
