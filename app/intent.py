# 意图与回答置信度的全部阈值，集中一处。
# 集中理由：卡片 8 判据要求 confidence 只出现 92/78/62 三个值，
# 阈值一旦散进 if 分支，改一处漏一处就会冒出第四个值。
# 本文件只放常量与词表、不放逻辑，避免 chat.py 与本文件循环 import。
from app.tools.order import find_order_no

# ---- 意图判定：0–1 小数，衡量「这句话是不是在问订单/要建单」----
INTENT_THRESHOLD = 0.6  # 及格线。≥ 它才允许调 get_order / create_ticket，< 它只走知识检索
INTENT_ORDER_HIT = 0.9  # 命中订单号 SO-八位日期-四位序号 的得分。单这一项即过线，因为它是最强信号
INTENT_ACTION_HIT = 0.6  # 命中售后动作词（退/换/投诉/建单/转人工）的得分，正好等于及格线
INTENT_TOPIC_HIT = 0.15  # 每个物流话题词的加分，用来拉开「随口一提」与「真在问」
INTENT_TOPIC_MAX = 0.45  # 话题词累计上限：光凑话题词最高 0.45，过不了 0.6，防关键词误判

# ---- 整条回答的置信度：0–100 整数，前端 ConfidenceBadge 按 85/75 分三档 ----
ANSWER_CONFIDENCE = {"strong": 92, "weak": 78, "none": 62}
# strong = 有引用，且顶部那一条 citation.confidence ≥ 0.5
# weak   = 有引用，但顶部相关度低
# none   = 零引用，「你们竞品是谁」落 62，卡片 9 的分流靠它自动建单

# 判 strong 的那条线。比的是单条引用的 confidence，不是 score——score 第一名恒为 1.000
STRONG_CITATION_CONFIDENCE = 0.5

ACTION_WORDS = ("退", "换", "投诉", "建单", "转人工")  # 子串匹配，"退"已覆盖退货/退款
TOPIC_WORDS = ("订单", "物流", "快递", "发货", "到哪", "签收", "送达")  # 注意：不要放「发票」「开」


def intent_confidence(prompt: str) -> float:
    """返回 0–1 的意图置信度：问订单/要建单偏高，闲聊偏低"""
    score = 0.0
    # 订单号是唯一强信号，命中即基本过线，不必再靠词表凑
    if find_order_no(prompt) is not None:
        score += INTENT_ORDER_HIT
    # any() 只看是否命中任意一个动作词，命中一次就够，不按次数累加
    if any(word in prompt for word in ACTION_WORDS):
        score += INTENT_ACTION_HIT
    # 话题词按个数累加但封顶，光堆「订单物流快递」也到不了 0.6
    topic_hits = sum(1 for word in TOPIC_WORDS if word in prompt)
    score += min(INTENT_TOPIC_MAX, topic_hits * INTENT_TOPIC_HIT)
    # 收口：不让浮点尾巴上屏，也保证不超过 1.0
    return min(1.0, round(score, 2))
