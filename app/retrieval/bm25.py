"""BM25 检索层：切词、打分、召回知识块，并过滤已下线文档"""

import json
import logging
import re
import threading
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi

logger = logging.getLogger("knoagent.retrieval")

# 仓库根目录：本文件位于 app/retrieval/bm25.py，向上三级即 kb-agent-api/
BASE_DIR = Path(__file__).resolve().parents[2]
# 语料与清单的固定位置，写死绝对路径会让换机器、换启动目录就失效
DATA_DIR = BASE_DIR / "data"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
DOCUMENTS_PATH = DATA_DIR / "documents.json"
USERDICT_PATH = DATA_DIR / "userdict.txt"

# 引用片段长度：前端卡片只展示开头，过长会把布局挤变形
SNIPPET_LEN = 120
# 清洗后低于这个长度说明整条基本都是 Markdown 标记，改用未清洗文本兜底，避免给出半句或空串
MIN_SNIPPET_LEN = 40
# 一行表格必然带三个以上竖线，超过这条线就认为 snippet 已经走进表格区域
MAX_TABLE_PIPES = 2
# 行首标题标记：把「## 六、常见失败原因」清成「六、常见失败原因」
HEADING_PATTERN = re.compile(r"#{1,6}\s+")
# 相对门槛：低于最高分这个比例的命中一律丢弃
# 语料只有十几块，BM25 原始分无上界且随查询长度漂移，绝对阈值（如 0.60）调不出来
RELATIVE_THRESHOLD = 0.4
# 默认召回条数，与前端契约里的 top_k 对齐
DEFAULT_TOP_K = 3
MIN_COVERAGE = 0.5
STOPWORDS = frozenset(
    [
        "的",
        "了",
        "吗",
        "呢",
        "吧",
        "啊",
        "呀",
        "是",
        "能",
        "可以",
        "怎么",
        "怎样",
        "什么",
        "你们",
        "我们",
        "他们",
        "请问",
        "一下",
        "这个",
        "那个",
        "在",
        "我",
        "你",
        "他",
        "谁",
        "还",
        "就",
        "也",
    ]
)


class _Index:
    """一次构建、反复查询的索引，持有可用块列表与 BM25 模型"""

    def __init__(self, chunks: list[dict], bm25: BM25Okapi | None, vocab: set[str]) -> None:
        # 已剔除下线文档的块，顺序与 bm25 语料顺序严格一致
        self.chunks = chunks
        # 语料为空时为 None，调用方要判空
        self.bm25 = bm25
        # 语料里出现过的全部词：查覆盖率时先把查询词过滤一遍
        # 否则「开错、旧、名字」这类语料里根本没有的词会撑大分母，正常提问被误拦
        self.vocab = vocab


# 模块级单例：jieba 加载词典约 0.5 秒，每次查询重建索引会让首个提问慢一秒
_index: _Index | None = None
# 并发首访时只允许一个线程建索引，其余等待
_index_lock = threading.Lock()


def _tokenize(text: str) -> list[str]:
    """切词并丢掉空白与纯标点，统一小写以免 Invoice 与 invoice 被当成两个词"""
    tokens: list[str] = []
    for word in jieba.lcut(text):
        word = word.strip().lower()
        # 纯标点对 BM25 无意义，留下只会拉低平均分
        if not word or word in STOPWORDS or not any(ch.isalnum() for ch in word):
            continue
        tokens.append(word)
    return tokens


def _load_documents() -> dict[str, dict]:
    """读文档清单，返回 doc_id → 元数据；清单缺失时降级为空，不让整个检索挂掉"""
    if not DOCUMENTS_PATH.exists():
        logger.warning("文档清单不存在，跳过下线过滤 path=%s", DOCUMENTS_PATH)
        return {}
    raw = json.loads(DOCUMENTS_PATH.read_text(encoding="utf-8"))
    return {item["doc_id"]: item for item in raw}


def _load_chunks(documents: dict[str, dict]) -> list[dict]:
    """读块文件并剔除已下线文档的块

    过滤必须发生在建索引阶段：查完再滤会让 top_k 因为被滤掉的条数而凑不满
    """
    chunk_list: list[dict] = []
    skipped = 0
    # 清单里查不到的 doc_id 集合：块照旧保留，但必须留下痕迹
    # 丢了语料是无声故障，搜不到也不报错；多一条引用反而容易发现，所以默认保留
    orphan_doc_ids: set[str] = set()
    with CHUNKS_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            chunk = json.loads(line)
            doc = documents.get(chunk["doc_id"])
            # 清单缺失时不做下线判断，避免清单漏写把整篇语料静默丢掉
            if doc is None:
                orphan_doc_ids.add(chunk["doc_id"])
            # deleted_at 非空即整篇下线
            elif doc.get("deleted_at"):
                skipped += 1
                continue
            chunk_list.append(chunk)
    logger.info("块加载完成 可用=%d 剔除=%d", len(chunk_list), skipped)
    # 对账：清单与块文件不同步时要能立刻看出来，否则只能等某个问题莫名答不上来才发现
    if orphan_doc_ids:
        logger.warning(
            "清单缺少这些文档，块已保留但无法判断是否下线 doc_id=%s",
            sorted(orphan_doc_ids),
        )
    return chunk_list


def _build_index() -> _Index:
    """加载词典、读语料、建 BM25；只在首次调用时执行一次"""
    # 自定义词典必须先于切词加载，否则「按摩仪」「抬头换开」照样被切成碎词
    if USERDICT_PATH.exists():
        jieba.load_userdict(str(USERDICT_PATH))
    jieba.initialize()

    documents = _load_documents()
    chunks = _load_chunks(documents)
    # 标题并入正文一起打分：标题里的政策名是强信号，纯正文可能通篇不重复文档名
    corpus = [_tokenize(f"{chunk['heading']} {chunk['text']}") for chunk in chunks]
    bm25 = BM25Okapi(corpus) if corpus else None
    vocab = {token for tokens_of_chunk in corpus for token in tokens_of_chunk}
    return _Index(chunks, bm25, vocab)


def _get_index() -> _Index:
    """取单例索引，懒加载；双检锁避免并发首访重复建索引"""
    global _index
    if _index is None:
        with _index_lock:
            if _index is None:
                _index = _build_index()
    return _index


def warmup() -> None:
    """服务启动时调一次，把词典与索引提前建好，避免第一次提问多等半秒"""
    _get_index()


def _confidence(matched: set[str], query_terms: set[str], idf: dict[str, float]) -> float:
    """按 IDF 加权的命中比例，作为给用户看的可信度

    单纯数命中词数会失真：命中「发票」「抬头」和命中两个虚词，分量完全不同
    所以用词的信息量加权——稀有词命中才真正抬分，常见词几乎不动
    """
    # BM25Okapi 会把出现在过半文档里的词的 IDF 压成极小负值，直接求和会抵消，取绝对值只看量级
    total = sum(abs(idf.get(term, 0.0)) for term in query_terms)
    if total <= 0:
        # 语料里一个带权重的词都没有，退回纯覆盖率，避免除零
        return len(matched & query_terms) / len(query_terms)
    hit = sum(abs(idf.get(term, 0.0)) for term in query_terms & matched)
    return hit / total


# 单词查询的可信度上限
# 只有一个查询词时，覆盖率非 0 即 1，指标退化成布尔值；直接给 1.0 会让用户以为满分命中
# 封顶到 0.7 表达「知道你要这个主题，但不确定哪一段最贴题」
SINGLE_TERM_CONFIDENCE_CAP = 0.7
# 可信度总上限：任何情况下都不报 1.000
# 检索是概率判断，给用户看「100% 可信」是不诚实，留一线表达不确定性
CONFIDENCE_CEILING = 0.95

# 数量单位与量词：单独出现时不代表业务实体，命中它等于没命中
# 「写一篇 1000 字作文」正是靠「字」这种词在发票语料里凑出了覆盖率
UNIT_WORDS = frozenset("字元天个次页条块人年月日号份张台件")
# 数词与量词组合：语料里常出现在「一个容器」「一篇说明」这类无关上下文
NOISE_WORDS = frozenset({"一个", "一篇", "这种", "那种", "一下", "一种"})
# 单字虚词与方位词：几乎每块都有，留下只会抬覆盖率、不抬贴题度
FUNCTION_CHARS = frozenset("到内外前后上中里间等和与或也都再还很太更最给被把对从向让使算来去下")


def _is_effective_term(term: str) -> bool:
    """判断查询词算不算「有效实词」

    有效实词表是第四道闸：只有它能进覆盖率的分子分母。语料里存在不等于有意义——
    纯数字（1000、15、SO 号里的 20261118）、单位量词、单字虚词都能在文档里找到，
    放进来会让「写一篇 1000 字作文」这种语料外提问凑满覆盖率、落成强引用
    """
    if term.isdigit():
        return False
    return term not in UNIT_WORDS and term not in NOISE_WORDS and term not in FUNCTION_CHARS


# 句子终止符：开窗起点必须落在这些字符之后，否则会给用户看半句
SENTENCE_ENDS = "。！？；!?;"
# 块首若是这些结构标记，说明它本身就是干净起点（标题、列表、表格），不必再挪
STRUCTURAL_STARTS = "#-|*"


def _find_anchor(text: str, terms: set[str]) -> tuple[int, str]:
    """挑一个命中词当开窗锚点：最长优先，其次最靠前，最末按字典序

    不能直接迭代 set 取第一个命中的词：字符串哈希按进程随机化，两个进程跑同一条
    查询可能挑到不同的词，snippet 起点就会抖，卡片 5 的「连问两次一致」直接过不了
    """
    anchors = [(text.find(term), term) for term in terms if term and term in text]
    if not anchors:
        return -1, ""
    return min(anchors, key=lambda item: (-len(item[1]), item[0], item[1]))


def _snap_start(text: str, raw_start: int) -> int:
    """把开窗起点右移到最近的句子起点，避开重叠切块留下的半句

    块首那几十字常是上一块的尾巴（如「子发票不可重复开具」），所以 raw_start 为 0
    也要挪；只有块本身以标题／列表／表格标记开头时，才认定它已是干净起点
    """
    if not text:
        return 0
    if raw_start <= 0 and text[0] in STRUCTURAL_STARTS:
        return 0
    index = max(1, raw_start)
    while index < len(text):
        previous = text[index - 1]
        if previous in SENTENCE_ENDS:
            break
        # 段间空行在开窗时已被压成两个空格，同样是句子边界
        if previous == " " and index >= 2 and text[index - 2] == " ":
            break
        index += 1
    if index >= len(text):
        # 整块找不到可用边界，退回原起点，至少保住命中词
        return raw_start if raw_start > 0 else 0
    while index < len(text) and text[index] == " ":
        index += 1
    return index


def _display_text(text: str) -> str:
    """把块原文转成详情里可读的全文

    引用抽屉要展示「这块到底写了什么」，直接把 Markdown 原文丢过去会满屏 `##`、`**` 和空行。
    这里只做最轻的三件事：去掉加粗标记、去掉行首标题井号、把连续空行压成一个，
    分段与表格原样保留——详情里需要的是结构，不是压缩成一行
    """
    if not text:
        return text
    cleaned = text.replace("**", "")
    # 换行在原文里是真的（这里是块原文，不是开窗后压平的 snippet），所以要按行处理标题标记
    lines = [
        HEADING_PATTERN.sub("", line) if line.startswith("#") else line
        for line in cleaned.split("\n")
    ]
    collapsed: list[str] = []
    for line in lines:
        # 连续空行只留一个，避免详情里出现大片空白
        if not line.strip() and collapsed and not collapsed[-1].strip():
            continue
        collapsed.append(line)
    return "\n".join(collapsed).strip()


def _flatten_table(text: str) -> str:
    """把落在表格里的片段压成一行可读文本

    截断到表格之前只剩半句时不能硬截，可原样返回又会让卡片显示一排竖线，
    折中办法是把单元格当成并列项：竖线变分隔符，`---` 分隔行与空单元格直接丢掉
    """
    cells = [cell.strip() for cell in text.split("|")]
    kept = [cell for cell in cells if cell and set(cell) != {"-"}]
    return " · ".join(kept)


def _clean_snippet(text: str) -> str:
    """去掉 snippet 里会当纯文本显示的 Markdown 源标记

    引用卡片把 snippet 直接插进 <p>/<blockquote>，星号、井号、竖线不会被渲染成格式，
    只会以「**」「##」「| --- |」的样子糊在用户脸上，所以在开窗之后做一次清洗。
    只清洗展示用的 snippet，块原文与打分排序都不受影响
    """
    if not text:
        return text
    # 先削掉成对加粗标记，再削行首标题标记；换行在开窗时已压成空格，标题标记退化成「 ## 」形式
    cleaned = text.replace("**", "")
    cleaned = HEADING_PATTERN.sub("", cleaned)
    # 表格区域不算有效信息：竖线超过两个就截断到第一个竖线之前
    if cleaned.count("|") > MAX_TABLE_PIPES:
        cut = cleaned.index("|")
        candidate = cleaned[:cut].rstrip()
        # 截完还剩够长才认，否则整段都在表格里，改走压平而不是硬截半句
        if len(candidate) >= MIN_SNIPPET_LEN:
            cleaned = candidate if candidate.endswith("…") else f"{candidate}…"
        else:
            cleaned = _flatten_table(cleaned)
    # 清洗后掉到很短说明整条基本都是标记，退回未清洗文本的开头，不能返回空串
    if len(cleaned) < MIN_SNIPPET_LEN:
        return text[:SNIPPET_LEN]
    return cleaned


def _to_citation(chunk: dict, ratio: float, query_terms: set[str], confidence: float) -> dict:
    """把块转成契约形状的引用项

    score 是相对最高分的比例，只用于排序比较，第一名恒为 1.000；
    confidence 是按 IDF 加权的命中比例，表示「这个块到底有多贴题」，给前端展示用
    """
    text = chunk["text"].replace("\n", " ").strip()
    # 块首常是 80 字重叠切下来的半句，直接截前 120 字会给用户看断句
    # 所以以命中词为中心开窗，起点一律 snap 到句子起点；命中词全无时退化成从首个句子起点截
    pos, term = _find_anchor(text, query_terms)
    if pos < 0:
        start = _snap_start(text, 0)
        end = min(len(text), start + SNIPPET_LEN)
    else:
        start = _snap_start(text, max(0, pos - SNIPPET_LEN // 2))
        if start > pos:
            # 句子起点越过了命中词，宁可从句中起也不能让命中词消失
            start = pos
        end = min(len(text), max(start + SNIPPET_LEN, pos + len(term)))
    head = "…" if start > 0 else ""
    tail = "…" if end < len(text) else ""
    # 开窗拿到的原文带 Markdown 标记，先清洗再进契约；块原文与检索排序一律不动
    snippet = _clean_snippet(f"{head}{text[start:end]}{tail}")
    return {
        "id": chunk["chunk_id"],
        "documentId": chunk["doc_id"],
        "knowledgeBaseId": chunk["knowledge_base_id"],
        "name": chunk["doc_name"],
        "description": chunk["description"],
        "snippet": snippet,
        # 详情抽屉要展示的全文：块原文去标记后的可读版本，卡片只显示 snippet 的前几十字
        "content": _display_text(chunk["text"]),
        "page": chunk["page"],
        "score": round(ratio, 3),
        # 展示用的可信度，与 score 语义不同：score 是相对排序值，confidence 是贴题程度
        "confidence": round(confidence, 3),
    }


def search(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """按 BM25 召回知识块，返回契约形状的引用列表，顺序在全序键上稳定"""
    index = _get_index()
    if index.bm25 is None or not query.strip():
        return []

    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    # 只保留语料里真实存在、且本身算实词的查询词
    # 语料外的词算进分母会把覆盖率压到门槛以下；数字、量词、单字虚词则相反，会把噪音词抬成命中
    effective_terms = [
        token for token in query_tokens if token in index.vocab and _is_effective_term(token)
    ]
    # 一个有效词都没有，说明问的东西语料里压根没有，直接零命中而不是凑一条最高的
    if not effective_terms:
        return []

    query_terms = set(effective_terms)
    scores = index.bm25.get_scores(query_tokens)
    # 全序键：分数降序 → 页码升序 → 块 id 升序
    # 少了后两级，同分块在两次查询里顺序会抖，验收「连问两次顺序一致」就过不了
    ranked = sorted(
        zip(index.chunks, scores, strict=False),
        key=lambda pair: (-pair[1], pair[0]["page"], pair[0]["chunk_id"]),
    )

    top_score = float(ranked[0][1])
    # 最高分不为正说明没有任何词命中，直接零命中，不要凑数返回
    if top_score <= 0:
        return []

    floor = top_score * RELATIVE_THRESHOLD
    # BM25Okapi 内部的词到 IDF 映射，算可信度直接用，不必自己再统计一遍
    idf: dict[str, float] = getattr(index.bm25, "idf", {})
    hits: list[dict] = []
    for chunk, score in ranked:
        # ranked 已按分数降序，一旦跌破门槛，后面的只会更低
        if float(score) < floor:
            break
        # 块的分词只算一次，覆盖率和可信度共用，别在循环里切两遍
        chunk_terms = set(_tokenize(chunk["heading"] + " " + chunk["text"]))
        if len(query_terms & chunk_terms) / len(query_terms) < MIN_COVERAGE:
            continue
        hits.append(
            _to_citation(
                chunk,
                float(score) / top_score,
                query_terms,
                _confidence(query_terms & chunk_terms, query_terms, idf)
                # 单词查询封顶并按分数比例拉开：全是 1.0 无意义，全归 0 又等于说不可信
                * (
                    SINGLE_TERM_CONFIDENCE_CAP * float(score) / top_score
                    if len(query_terms) == 1
                    else CONFIDENCE_CEILING
                ),
            )
        )
        if len(hits) >= top_k:
            break
    return hits


if __name__ == "__main__":
    # 手工自测入口：python -m app.retrieval.bm25
    built = _get_index()
    print(f"可用块数: {len(built.chunks)}")
    for probe in [
        "按摩仪拆封了能退吗",
        "电子发票抬头开错公司旧名字能改吗",
        "15 天内可以无理由退货吗",
        "你们竞品是谁",
    ]:
        print(f"\n查询: {probe}")
        for item in search(probe):
            print(f"  {item['score']:.3f} p{item['page']} {item['name']} :: {item['snippet'][:40]}")
