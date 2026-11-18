"""工单落盘与复用：低置信度回答自动转人工时用它建单"""

import json
import re
from datetime import UTC, datetime
from pathlib import Path

# 工单库文件，与 orders.json 同层；文件不存在时按空库处理
TICKETS_PATH = Path(__file__).parents[2] / "data" / "tickets.json"

# 工单号前缀与位宽：A + 4 位自增，与前端 mock 里的 A1026 形态一致
TICKET_PREFIX = "A"
TICKET_WIDTH = 4

# 归一化只保留文字与数字：中文在 Python 里算 \w，所以这个模式只吃掉标点、空白与下划线
NOISE_PATTERN = re.compile(r"[\s\W_]+", re.UNICODE)

# 工单号形如 A0001，用于从既有工单里取最大序号
TICKET_NO_PATTERN = re.compile(rf"{TICKET_PREFIX}(\d+)")


def normalize_prompt(prompt: str) -> str:
    """把提问压成复用键：去标点空白并转小写，避免同一问题复述时标点不同就当成新问题"""
    return NOISE_PATTERN.sub("", prompt).lower()


def _load_tickets() -> list[dict]:
    """读取工单库；文件缺失或内容损坏时返回空列表，不让读盘问题中断对话"""
    if not TICKETS_PATH.exists():
        return []

    try:
        raw = json.loads(TICKETS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []

    return raw if isinstance(raw, list) else []


def _save_tickets(tickets: list[dict]) -> None:
    """原子写盘：先写临时文件再替换，避免进程中断留下半个 JSON 把工单库读废"""
    TICKETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = TICKETS_PATH.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(tickets, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(TICKETS_PATH)


def _next_ticket_no(tickets: list[dict]) -> str:
    """取现有最大序号 +1；每次从文件重算而不是内存自增，重启后既不跳号也不重号"""
    biggest = 0
    for ticket in tickets:
        matched = TICKET_NO_PATTERN.fullmatch(str(ticket.get("ticketNo", "")))
        if matched:
            biggest = max(biggest, int(matched.group(1)))

    return f"{TICKET_PREFIX}{biggest + 1:0{TICKET_WIDTH}d}"


def _public_view(ticket: dict, reused: bool) -> dict:
    """对外只给这三个字段：复用键是内部去重用的，不必发到事件流里"""
    return {"ticketNo": ticket.get("ticketNo"), "status": ticket.get("status"), "reused": reused}


def create_ticket(prompt: str, conversation_id: str | None) -> dict:
    """按「会话 + 归一化提问」复用建单：同一问题重复求助返回既有工单号，不新增记录"""
    tickets = _load_tickets()
    prompt_key = normalize_prompt(prompt)

    for ticket in tickets:
        if (
            ticket.get("promptKey") == prompt_key
            and ticket.get("conversationId") == conversation_id
        ):
            return _public_view(ticket, reused=True)

    record = {
        "ticketNo": _next_ticket_no(tickets),
        "conversationId": conversation_id,
        "promptKey": prompt_key,
        "prompt": prompt,
        "status": "待人工处理",
        "createdAt": datetime.now(UTC).isoformat(),
    }
    tickets.append(record)
    _save_tickets(tickets)

    return _public_view(record, reused=False)
