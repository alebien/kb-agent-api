"""订单查询：识别订单号、查本地订单表，出口只给掩码号码"""

import json
import re
from pathlib import Path

# 订单号形态 SO-八位日期-四位序号，只做识别、不判断真伪
ORDER_NO_PATTERN = re.compile(r"SO-\d{8}-\d{4}")

# 按本文件位置推算订单表路径，不依赖启动时的工作目录
ORDERS_PATH = Path(__file__).parents[2] / "data" / "orders.json"

# 进程内订单表缓存，键为订单号；语料量小，不做热更新
_orders_cache: dict[str, dict] | None = None


def find_order_no(prompt: str) -> str | None:
    """返回提问里第一个订单号，没有则 None"""
    match = ORDER_NO_PATTERN.search(prompt)
    return match.group(0) if match else None


def load_orders() -> dict[str, dict]:
    """读订单表并按订单号建索引，首次读盘后驻留内存"""
    global _orders_cache
    if _orders_cache is None:
        raw = json.loads(ORDERS_PATH.read_text(encoding="utf-8"))
        _orders_cache = {item["orderNo"]: item for item in raw}
    return _orders_cache


def mask_phone(phone: str | None) -> str:
    """号码掩码：留首 3 尾 4；形态不符就整体打码，避免漏出半截"""
    if not phone or len(phone) != 11:
        return "***"
    return f"{phone[:3]}****{phone[-4:]}"


def lookup_order(order_no: str) -> dict:
    """查单：查不到也返回同结构结果，调用方据此发 success 而非 error"""
    order = load_orders().get(order_no)
    if order is None:
        return {"orderNo": order_no, "found": False}
    return {
        "orderNo": order["orderNo"],
        "status": order["status"],
        "maskedPhone": mask_phone(order.get("phone")),
        "found": True,
    }
