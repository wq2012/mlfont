"""Website-level web font loading simulator, throttled HTTP server, and Chrome benchmark.

Provides frontend developers with tools to:
1. Extract visible DOM text and Unicode codepoints from real HTML websites, URLs,
   or a built-in multi-script suite of 20 real-world website archetypes.
2. Simulate end-to-end browser font loading across realistic network profiles
   (2G, 3G, Fast 3G, Residential DSL, Desktop Cable 5 Mbps, Mobile 4G LTE 12 Mbps,
   and Broadband Fiber 20 Mbps), modeling gzipped CSS transfer, Blink @font-face
   unicode-range parsing, parallel HTTP slice requests, cold vs. warm session
   browser caching, and exact physical WOFF2 byte transfers.
3. Run live Headless Google Chrome benchmarks against an instrumented local
   throttled HTTP server using the HTML5 Resource Timing and FontFaceSet APIs.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import gzip
import html
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
import urllib.request

import numpy as np

from mlfont.baselines import (
    generate_equal_bin_strategy,
    generate_language_slice_strategy,
    generate_single_slice_strategy,
)
from mlfont.configs import get_preset_config
from mlfont.evaluator import compute_distribution_stats
from mlfont.frequency import generate_most_and_equal_bin_strategy
from mlfont.schemas import DistributionStats, SlicingStrategy
from mlfont.subsetter import generate_css

# Exact physical full-font WOFF2 byte sizes measured via fontTools + Brotli
# on the 5 Google Noto reference fonts.
FULL_FONT_WOFF2_BYTES: Dict[str, int] = {
    "notosanssc": 6130712,
    "mlfont-notosanssc": 6130712,
    "notosanstc": 4144544,
    "mlfont-notosanstc": 4144544,
    "notosansjp": 3338400,
    "mlfont-notosansjp": 3338400,
    "notosanskr": 3105200,
    "mlfont-notosanskr": 3105200,
    "notosans_lgc": 161472,
    "notosans-lgc": 161472,
    "mlfont-notosans-lgc": 161472,
}

FULL_FONT_RAW_BYTES: Dict[str, int] = {
    "notosanssc": 8368828,
    "notosanstc": 5648604,
    "notosansjp": 4538888,
    "notosanskr": 4579008,
    "notosans_lgc": 414820,
}


@dataclass
class BrowserNetworkProfile:
    """Browser network condition profile for website font-loading simulation."""

    name: str
    rtt_ms: float
    bandwidth_bytes_per_ms: float
    max_parallel_requests: int = 6
    description: str = ""

    @property
    def bandwidth_kbps(self) -> float:
        return self.bandwidth_bytes_per_ms * 8.0

    @property
    def bandwidth_mbps(self) -> float:
        return self.bandwidth_kbps / 1000.0


STANDARD_BROWSER_NETWORK_PROFILES: List[BrowserNetworkProfile] = [
    BrowserNetworkProfile(
        name="GIN-2g",
        rtt_ms=1200.0,
        bandwidth_bytes_per_ms=12.5,  # 100 Kbps
        max_parallel_requests=6,
        description="Mobile 2G (100 Kbps, 1200 ms RTT)",
    ),
    BrowserNetworkProfile(
        name="GIN-3g",
        rtt_ms=400.0,
        bandwidth_bytes_per_ms=50.0,  # 400 Kbps
        max_parallel_requests=6,
        description="Mobile 3G (400 Kbps, 400 ms RTT)",
    ),
    BrowserNetworkProfile(
        name="Fast-3G",
        rtt_ms=150.0,
        bandwidth_bytes_per_ms=200.0,  # 1.6 Mbps
        max_parallel_requests=6,
        description="Fast 3G / DevTools Slow 4G (1.6 Mbps, 150 ms RTT)",
    ),
    BrowserNetworkProfile(
        name="residential-DSL",
        rtt_ms=50.0,
        bandwidth_bytes_per_ms=250.0,  # 2.0 Mbps
        max_parallel_requests=6,
        description="Residential DSL (2.0 Mbps, 50 ms RTT)",
    ),
    BrowserNetworkProfile(
        name="Cable-5Mbps",
        rtt_ms=28.0,
        bandwidth_bytes_per_ms=625.0,  # 5.0 Mbps
        max_parallel_requests=6,
        description="WebPageTest Desktop Cable (5.0 Mbps, 28 ms RTT)",
    ),
    BrowserNetworkProfile(
        name="Mobile-4G-LTE",
        rtt_ms=70.0,
        bandwidth_bytes_per_ms=1500.0,  # 12.0 Mbps
        max_parallel_requests=6,
        description="WebPageTest Mobile 4G LTE (12.0 Mbps, 70 ms RTT)",
    ),
    BrowserNetworkProfile(
        name="Broadband-Fiber",
        rtt_ms=20.0,
        bandwidth_bytes_per_ms=2500.0,  # 20.0 Mbps
        max_parallel_requests=6,
        description="Broadband ADSL2+ / Fiber (20.0 Mbps, 20 ms RTT)",
    ),
]

NETWORK_PROFILE_MAP: Dict[str, BrowserNetworkProfile] = {
    p.name.lower(): p for p in STANDARD_BROWSER_NETWORK_PROFILES
}
NETWORK_PROFILE_MAP["gin-2g-poor"] = BrowserNetworkProfile(
    name="GIN-2g-poor",
    rtt_ms=2000.0,
    bandwidth_bytes_per_ms=4.0,  # 32 Kbps
    max_parallel_requests=6,
    description="Mobile 2G Poor (32 Kbps, 2000 ms RTT)",
)
NETWORK_PROFILE_MAP["broadband-20mbps"] = NETWORK_PROFILE_MAP["broadband-fiber"]
NETWORK_PROFILE_MAP["residential-adsl2+"] = BrowserNetworkProfile(
    name="residential-ADSL2+",
    rtt_ms=50.0,
    bandwidth_bytes_per_ms=2500.0,  # 20.0 Mbps
    max_parallel_requests=6,
    description="Residential ADSL2+ (20.0 Mbps, 50 ms RTT)",
)


def get_network_profile(name: str | BrowserNetworkProfile) -> BrowserNetworkProfile:
    """Returns a named BrowserNetworkProfile (or passes through an existing instance)."""
    if isinstance(name, BrowserNetworkProfile):
        return name
    key = name.strip().lower()
    if key in NETWORK_PROFILE_MAP:
        return NETWORK_PROFILE_MAP[key]
    raise ValueError(
        f"Unknown network profile {name!r}. Available: "
        f"{[p.name for p in STANDARD_BROWSER_NETWORK_PROFILES]}"
    )


class _VisibleTextExtractor(HTMLParser):
    """HTMLParser that strips non-visible container tags (<script>, <style>, <head>, etc.)."""

    _IGNORED_CONTAINERS = {
        "script",
        "style",
        "noscript",
        "template",
        "svg",
        "head",
        "title",
    }
    _HEAD_METADATA_TAGS = {
        "head",
        "base",
        "basefont",
        "bgsound",
        "link",
        "meta",
        "noscript",
        "noframes",
        "script",
        "style",
        "template",
        "title",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._ignore_stack: List[str] = []
        self._chunks: List[str] = []
        self.saw_html_tag = False
        self.has_inline_whitespace = False

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        self.saw_html_tag = True
        t = tag.lower()
        if "head" in self._ignore_stack and t not in self._HEAD_METADATA_TAGS:
            self._ignore_stack = [
                x for x in self._ignore_stack if x not in ("head", "title")
            ]
        elif "title" in self._ignore_stack and t != "title":
            self._ignore_stack = [x for x in self._ignore_stack if x != "title"]
        if t in self._IGNORED_CONTAINERS:
            self._ignore_stack.append(t)

    def handle_endtag(self, tag: str) -> None:
        self.saw_html_tag = True
        t = tag.lower()
        if t in self._ignore_stack:
            for idx in range(len(self._ignore_stack) - 1, -1, -1):
                if self._ignore_stack[idx] == t:
                    del self._ignore_stack[idx]
                    break

    def handle_comment(self, data: str) -> None:
        self.saw_html_tag = True

    def handle_decl(self, decl: str) -> None:
        self.saw_html_tag = True

    def handle_pi(self, data: str) -> None:
        self.saw_html_tag = True

    def handle_data(self, data: str) -> None:
        if not self._ignore_stack:
            stripped = data.strip()
            if stripped:
                if " " in data or any(ch.isspace() for ch in stripped):
                    self.has_inline_whitespace = True
                self._chunks.append(stripped)

    def get_text(self) -> str:
        return " ".join(self._chunks)


def _parse_html_or_text(html_or_text: str) -> Tuple[str, bool, bool]:
    """Returns (visible_text, was_html, has_inline_whitespace)."""
    if not html_or_text:
        return "", False, False
    if "<" in html_or_text and ">" in html_or_text:
        parser = _VisibleTextExtractor()
        parser.feed(html_or_text)
        parser.close()
        if parser.saw_html_tag:
            text = html.unescape(parser.get_text())
            return text, True, parser.has_inline_whitespace
    text = html.unescape(html_or_text)
    has_ws = any(ch.isspace() for ch in text)
    return text, False, has_ws


def extract_visible_text_from_html(html_or_text: str) -> str:
    """Extracts visible rendered text from an HTML document or plain text string."""
    text, _, _ = _parse_html_or_text(html_or_text)
    return text


def extract_webpage_codepoints(
    html_or_text: str,
    valid_codepoints: Optional[Set[int]] = None,
) -> Set[int]:
    """Extracts unique visible Unicode codepoints from an HTML page or text string."""
    text, was_html, has_inline_ws = _parse_html_or_text(html_or_text)
    normalized = " ".join(text.split())
    cps = {ord(ch) for ch in normalized}
    if was_html and not has_inline_ws:
        cps.discard(0x20)
    if valid_codepoints is not None:
        cps &= valid_codepoints
    return cps


@dataclass
class WebPageSample:
    """Represents a real or benchmark webpage for font-loading simulation."""

    page_id: str
    title: str
    url: str
    category: str
    font_key: str
    html_content: str
    visible_text: str = ""
    codepoints: Set[int] = field(default_factory=set)
    char_count: int = 0
    codepoint_frequencies: Dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        was_html = False
        has_inline_ws = True
        if not self.visible_text and self.html_content:
            self.visible_text, was_html, has_inline_ws = _parse_html_or_text(
                self.html_content
            )
        else:
            has_inline_ws = any(ch.isspace() for ch in self.visible_text)
        normalized = " ".join(self.visible_text.split())
        non_ws = [ord(ch) for ch in normalized if not ch.isspace()]
        if not self.codepoints and normalized:
            self.codepoints = {ord(ch) for ch in normalized}
            if was_html and not has_inline_ws:
                self.codepoints.discard(0x20)
        if self.char_count == 0 and non_ws:
            self.char_count = len(non_ws)
        if not self.codepoint_frequencies and non_ws:
            self.codepoint_frequencies = dict(Counter(non_ws))


def _wrap_html_page(title: str, nav_items: Sequence[str], body_paragraphs: Sequence[str]) -> str:
    nav_html = "".join(f"<li><a href='#'>{html.escape(item)}</a></li>" for item in nav_items)
    p_html = "\n".join(f"<p>{html.escape(p)}</p>" for p in body_paragraphs)
    return (
        "<!DOCTYPE html>\n<html><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title></head>\n"
        f"<body><header><nav><ul>{nav_html}</ul></nav>"
        f"<h1>{html.escape(title)}</h1></header>\n"
        f"<main><article>{p_html}</article></main>\n"
        "<footer><p>Copyright 2026 Web Typography Benchmark Suite.</p></footer>"
        "</body></html>"
    )


def get_real_website_benchmark_pages(font_key: Optional[str] = None) -> List[WebPageSample]:
    """Returns the built-in benchmark suite of 20 realistic websites across all 5 Noto fonts.

    Includes the 8 Simplified Chinese real-website archetypes from the original
    2017 `go/mlfont-demo` (NBA Sports, World Politics, AI Technology, Lunar New Year
    Entertainment, Beef Cooking Recipe, Tangqi Fiction Novel) and WebPageTest report
    (`developers.google.com/china` Developer Portal and `shenzhenware` Hardware Review),
    plus 3 websites each for Traditional Chinese, Japanese, Korean, and Pan-European LGC.
    """
    pages: List[WebPageSample] = [
        # --- 1. Simplified Chinese (Noto Sans SC) ---
        WebPageSample(
            page_id="dev_china_portal",
            title="Google Developers China (developers.google.com/china/?hl=zh-cn)",
            url="https://developers.google.com/china/?hl=zh-cn",
            category="Developer Portal",
            font_key="notosanssc",
            html_content=_wrap_html_page(
                "面向中国开发者的 Google 技术资源与开源生态平台",
                ["首页", "Android 开发", "TensorFlow 机器学习", "Web 性能与字体", "Firebase", "技术社区"],
                [
                    "构建面向全球用户的现代化 Web 与移动应用。通过最新的 WebAssembly、"
                    "Progressive Web Apps (PWA) 以及智能中文字体切片技术，显著降低首屏渲染延迟，"
                    "提升核心网页指标（Core Web Vitals：LCP、FID、CLS）。",
                    "在移动互联网环境中，中文字体通常包含超过三万个汉字与标点符号，完整字体文件体积高达"
                    "八兆字节以上。借助机器学习共现主题模型与区间紧凑化遗传算法，浏览器仅需下载当前文档"
                    "所需的两到四个 WOFF2 字体分片即可完成高质量排版渲染。",
                    "开发者工具指南：使用 Chrome DevTools Network 面板与 Performance API 实时监控"
                    "静态资源瀑布流、CSS 样式表解析耗时、布局重排（Layout）以及图层树更新性能。",
                ],
            ),
        ),
        WebPageSample(
            page_id="sina_sports_nba",
            title="Sina Sports NBA Playoffs Report (sports.sina.com.cn)",
            url="https://sports.sina.com.cn/basketball/nba/",
            category="Sports - NBA",
            font_key="notosanssc",
            html_content=_wrap_html_page(
                "NBA季后赛焦点战：勇士客场逆转火箭，库里末节狂砍22分率队晋级",
                ["体育首页", "NBA篮球", "CBA联赛", "欧冠足球", "实时比分", "技术统计", "视频集锦"],
                [
                    "北京时间5月18日，NBA西部半决赛迎来抢七大战。金州勇士队在客场半场落后11分的不利局势下，"
                    "下半场掀起猛烈三分雨。斯蒂芬·库里全场出战41分钟，23投13中，三分球12投7中，砍下全场最高的"
                    "36分、7个篮板和9次助攻，带领勇士以114比106力克休斯敦火箭，昂首挺进西部决赛。",
                    "比赛第四节进入白热化阶段，格林在防守端连续完成关键封盖与抢断，汤普森底角稳稳命中反超远投。"
                    "主教练科尔在赛后新闻发布会上表示，球队在体能极限下展现了总冠军级别的韧性与战术执行力，"
                    "替补席的卢尼与伊戈达拉同样在篮板拼抢和换防协防中立下汗马功劳。",
                ],
            ),
        ),
        WebPageSample(
            page_id="tech_ai_maluuba",
            title="AI & Deep Reinforcement Learning Feature (Technology - Maluuba)",
            url="https://tech.sina.com.cn/ai/deep-learning-nlp",
            category="Technology - AI",
            font_key="notosanssc",
            html_content=_wrap_html_page(
                "深度学习与自然语言理解前沿：从阅读理解到多智能体强化学习突破",
                ["科技前沿", "人工智能", "深度学习", "开源算法", "算力芯片", "学术论文"],
                [
                    "人工智能研究实验室近期在机器阅读理解与混合奖励深度强化学习领域取得里程碑式进展。"
                    "研究团队提出了一种结合分层注意力机制与分布式价值估计的神经网络架构，使智能体能够在"
                    "高维稀疏奖励环境中高效探索复杂状态空间，并在多项自然语言问答基准测试中刷新纪录。",
                    "通过将非负稀疏自编码器与判别式香农熵差分损失函数相结合，模型能够在大规模无标注语料库上"
                    "自动发现潜在语义簇。这一算法不仅加速了知识图谱推理与多轮对话系统，还被成功应用于"
                    "大规模汉字字库的子集切分与低延迟网络传输优化。",
                ],
            ),
        ),
        WebPageSample(
            page_id="world_politics_news",
            title="Global Economic & Diplomatic Summit Report (Politics)",
            url="https://news.sina.com.cn/world/summit",
            category="Politics & Economy",
            font_key="notosanssc",
            html_content=_wrap_html_page(
                "多边经贸合作与全球数字治理峰会发表联合声明，深化绿色转型与跨境贸易互通",
                ["国际新闻", "宏观经济", "外交动态", "数字贸易", "气候变化", "智库观察"],
                [
                    "二十国集团与多边经贸部长会议今日在日内瓦闭幕。与会各国代表围绕全球供应链韧性、"
                    "跨境电子商务规则、清洁能源基础设施投资以及人工智能安全治理框架展开深入磋商，"
                    "一致同意削减技术性贸易壁垒，推动包容性可持续增长。",
                    "国际货币基金组织与世界贸易组织在联合报告中指出，亚太及新兴市场经济体在数字基础设施"
                    "普及率提升的带动下保持稳健复苏态势。各国央行强调将在维护金融市场稳定与抑制通胀预期"
                    "之间保持审慎平衡，加强宏观经济政策协调。",
                ],
            ),
        ),
        WebPageSample(
            page_id="entertainment_spring_gala",
            title="Lunar New Year Cultural Gala & Box Office (Entertainment)",
            url="https://ent.sina.com.cn/film/spring-festival",
            category="Entertainment - Chinese New Year",
            font_key="notosanssc",
            html_content=_wrap_html_page(
                "新春佳节文化盛典与春节档电影票房破纪录：非遗民俗与现代视听艺术交相辉映",
                ["娱乐头条", "春节晚会", "院线电影", "国风音乐", "非遗传承", "票房排行榜"],
                [
                    "除夕夜张灯结彩、喜气洋洋，新春联欢晚会以精湛的舞台光影技术融合传统非物质文化遗产："
                    "京剧武生、川剧变脸、江南丝竹与现代交响乐团同台献艺，舞龙舞狮与漫天烟花点燃了亿万观众"
                    "辞旧迎新的团圆热情。",
                    "与此同时，大年初一全国电影院线迎来观影热潮，科幻巨制、历史史诗与合家欢动画喜剧齐头并进，"
                    "单日总票房突破十七亿元。观众在欢声笑语中感受中华传统节庆文化的深厚底蕴与蓬勃创造力。",
                ],
            ),
        ),
        WebPageSample(
            page_id="cooking_recipe_beef",
            title="Gourmet Braised Beef Brisket Recipe (Cooking Recipe - Beef)",
            url="https://meishi.sina.com.cn/recipe/braised-beef",
            category="Cooking Recipe - Beef",
            font_key="notosanssc",
            html_content=_wrap_html_page(
                "家常秘制红烧牛腩炖萝卜食谱：汤汁浓郁、酥烂入味的经典暖胃佳肴",
                ["美食菜谱", "家常热菜", "煲汤炖品", "烘焙甜点", "时令食材", "厨艺技巧"],
                [
                    "精选主料与香料配比：新鲜牛坑腩800克、白萝卜1根、生姜5片、大葱2段、蒜瓣6粒、"
                    "八角2颗、桂皮1小段、香叶3片、干辣椒4个、草果1颗（拍裂去籽）、冰糖25克、"
                    "生抽2汤匙、老抽1汤匙、黄豆酱1汤匙、绍兴黄酒3汤匙。",
                    "烹饪步骤：一、牛腩切3厘米见方大块，冷水下锅加入姜片与料酒大火煮沸，撇净浮沫后温水洗净沥干；"
                    "二、炒锅少油放入冰糖小火熬至琥珀色糖色，下入牛腩翻炒均匀裹上焦糖色泽；"
                    "三、加入葱姜蒜与全套香料爆香，烹入黄酒、生抽、老抽及黄豆酱炒出浓郁酱香，冲入沸水没过食材，"
                    "转入砂锅小火慢炖90分钟，最后加入滚刀块白萝卜继续焖煮25分钟，大火收汁撒香菜即可上桌。",
                ],
            ),
        ),
        WebPageSample(
            page_id="novel_tangqi_fiction",
            title="Classical Fantasy Fiction Chapter (Novel - Tangqi Literature)",
            url="https://book.sina.com.cn/novel/fantasy-chapter",
            category="Novel - Literature",
            font_key="notosanssc",
            html_content=_wrap_html_page(
                "卷三·苍梧云海与桃林旧梦：宿命流转间的古风仙侠长篇小说连载",
                ["书库首页", "古典仙侠", "历史架空", "诗词歌赋", "作家专栏", "读者书评"],
                [
                    "暮春三月，十里桃林落英缤纷，绯色花瓣随微风拂过青石阶梯，坠入碧波荡漾的镜湖之中。"
                    "白衣少年凭栏远眺，苍梧山巅云雾缭绕，隐约可见琉璃飞檐与古刹钟声穿越千载岁月悠然回荡。"
                    "他执起案头那盏温热的青瓷茶盅，眸中掠过一抹难以言喻的怅惘与决绝。",
                    "昔年昆仑虚上琴音袅袅，师徒二人曾在漫天星斗下推演周天星象，岂料沧海桑田、风云突变，"
                    "一纸封印将往昔记忆尽数尘封于九重阙下。如今瑶池仙障重开，惊鸿照影，纵使前路荆棘密布、"
                    "雷霆万钧，亦难阻这一场跨越三生三世的执念相逢。",
                ],
            ),
        ),
        WebPageSample(
            page_id="shenzhenware_hardware",
            title="Shenzhenware Smart Hardware & Voice AI Review (shenzhenware.com)",
            url="https://www.shenzhenware.com/articles/smart-audio",
            category="Hardware & IoT",
            font_key="notosanssc",
            html_content=_wrap_html_page(
                "深圳湾硬件评测：新一代远场语音交互智能音箱与端侧神经网络芯片拆解",
                ["深圳湾首页", "智能音频", "具身机器人", "AR眼镜", "芯片拆解", "创客社区"],
                [
                    "随着端侧大模型与低功耗数字信号处理器（DSP）的快速演进，智能家居语音终端正在经历从"
                    "云端指令识别向全双工自然对话的跨越。本期硬件拆解实验室对最新发布的六麦克风环形阵列"
                    "智能音箱进行了全方位声学测试与主板元器件剖析。",
                    "实测表明，得益于波束成形（Beamforming）、回声消除（AEC）以及本地量化语音识别引擎的协同优化，"
                    "设备在五米远场高噪环境下依然保持96%以上的唤醒准确率，端到端响应延迟缩短至280毫秒以内。",
                ],
            ),
        ),
        # --- 2. Traditional Chinese (Noto Sans TC) ---
        WebPageSample(
            page_id="tc_taipei_tech_portal",
            title="Taiwan Semiconductor & Cloud AI Developer Portal (zh-Hant)",
            url="https://developers.example.tw/cloud-ai",
            category="Developer Portal (TC)",
            font_key="notosanstc",
            html_content=_wrap_html_page(
                "台灣半導體製程與雲端人工智慧開發者平台：高速網頁字型與邊緣運算實踐",
                ["首頁", "先進製程", "雲端架構", "開源模型", "前端效能", "技術論壇"],
                [
                    "在現代化跨平台網頁應用程式中，繁體中文字型包含超過兩萬個常用與罕用漢字，"
                    "完整字型檔案往往超過五百萬位元組。透過機器學習共現主題模型與基因演算法區間最佳化，"
                    "系統能將繁體字庫精準切分為八十二個高內聚力的靜態 WOFF2 切片。",
                    "當使用者透過行動網路或寬頻瀏覽技術文件與即時新聞時，瀏覽器僅需根據頁面實際出現的"
                    "萬國碼（Unicode）字元按需載入極少數切片，大幅縮短首次內容繪製時間並節省超過百分之九十的頻寬。",
                ],
            ),
        ),
        WebPageSample(
            page_id="tc_culture_literature",
            title="National Palace Museum & Classical Calligraphy Essay (zh-Hant)",
            url="https://culture.example.tw/calligraphy-typography",
            category="Culture & Literature (TC)",
            font_key="notosanstc",
            html_content=_wrap_html_page(
                "故宮典藏書法藝術與數位時代漢字排版美學：從碑帖石刻到可變網路字型",
                ["藝文首頁", "書畫典藏", "古籍善本", "數位人文", "字型設計", "特展導覽"],
                [
                    "漢字書法自甲骨金文、秦篆漢隸、魏碑唐楷以至宋明刻本，承載了數千年東亞文明的審美意趣。"
                    "走進外雙溪博物院展廳，溪山行旅圖的雄渾筆墨與快雪時晴帖的行雲流水令人嘆為觀止。",
                    "邁入高解析度螢幕與雲端閱讀時代，字型設計師與電腦科學家攜手合作，在保留傳統筆畫結構與"
                    "氣韻的同時，運用統計學習演算法解決大型中文字集在網際網路上的即時傳輸瓶頸。",
                ],
            ),
        ),
        WebPageSample(
            page_id="tc_gastronomy_guide",
            title="Taipei Gastronomy & Night Market Culinary Guide (zh-Hant)",
            url="https://lifestyle.example.tw/taipei-cuisine",
            category="Culinary & Travel (TC)",
            font_key="notosanstc",
            html_content=_wrap_html_page(
                "台北職人美食與夜市風味指南：紅燒牛肉麵、小籠湯包與高山烏龍茶的極致饗宴",
                ["美食首頁", "在地小吃", "米其林指南", "茶藝文化", "節慶料理", "私房食譜"],
                [
                    "漫步於永康街與大稻埕歷史街區，空氣中瀰漫著慢火熬煮牛骨高湯的醇厚香氣。"
                    "師傅嚴選台灣本地溫體牛腱心，搭配岡山豆瓣醬、川味花椒與八角細火慢燉數小時，"
                    "麵條彈牙滑順，湯頭濃郁回甘；佐以皮薄餡豐、湯汁飽滿的十八摺小籠包與阿里山清香烏龍茶，"
                    "展現出寶島飲食文化兼容並蓄的職人精神。",
                ],
            ),
        ),
        # --- 3. Japanese (Noto Sans JP) ---
        WebPageSample(
            page_id="jp_web_dev_docs",
            title="Japanese Web Platform & Typography Engineering Docs (ja)",
            url="https://developers.example.jp/web-typography",
            category="Developer Portal (JP)",
            font_key="notosansjp",
            html_content=_wrap_html_page(
                "次世代ウェブタイポグラフィと日本語フォントの高速配信アーキテクチャ",
                ["ホーム", "ウェブ標準", "フォント最適化", "パフォーマンス計測", "APIリファレンス"],
                [
                    "日本語のデジタル組版では、ひらがな・カタカナ・英数字に加えて第一水準および第二水準漢字など"
                    "一万六千文字以上のグリフが必要となります。従来の単一フォントファイル配信では、"
                    "数メガバイトの通信が発生し、モバイル回線におけるページ表示遅延の原因となっていました。",
                    "MLFontは、大規模な日本語ウェブコーパスから文字の共起確率をオンラインLDAおよび深層ニューラル"
                    "トピックモデルで学習し、遺伝的アルゴリズムによってCSSのunicode-range記述子をコンパクトに保ちながら"
                    "62個の最適なWOFF2スライスへと分割します。これにより初回表示のレイテンシを劇的に短縮します。",
                ],
            ),
        ),
        WebPageSample(
            page_id="jp_sports_baseball",
            title="Japanese Professional Baseball Championship Report (ja)",
            url="https://sports.example.jp/baseball/npb-finals",
            category="Sports News (JP)",
            font_key="notosansjp",
            html_content=_wrap_html_page(
                "日本シリーズ第7戦：劇的な逆転サヨナラ本塁打でリーグ覇者が日本一に輝く",
                ["スポーツ総合", "プロ野球", "侍ジャパン", "Jリーグ", "試合速報", "選手名鑑"],
                [
                    "満員のスタジアムが熱狂に包まれた最終第7戦は、球史に残る投手戦となった。"
                    "先発左腕が8回まで散発3安打無失点の快投を見せると、1点を追う9回裏二死一・二塁の場面で、"
                    "主砲がライトスタンド中段へ飛び込む逆転サヨナラ3ランホームランを放ち、歓喜の胴上げが実現した。",
                ],
            ),
        ),
        WebPageSample(
            page_id="jp_literature_essay",
            title="Kyoto Seasonal Aesthetics & Modern Literary Column (ja)",
            url="https://culture.example.jp/kyoto-essay",
            category="Literature & Culture (JP)",
            font_key="notosansjp",
            html_content=_wrap_html_page(
                "古都の四季と日本の美意識：嵐山の紅葉と静寂の庭園をめぐる随筆",
                ["文芸トップ", "エッセイ", "書評", "伝統工芸", "歴史紀行"],
                [
                    "秋の夕暮れ、渡月橋を渡る風がひんやりと頬を撫でる頃、山々は錦繍の紅葉に染め上げられる。"
                    "苔むした石畳の路地を抜け、静まり返った禅寺の縁側に腰を下ろせば、竹林を揺らす風の音と"
                    "鹿威しの澄んだ響きが心の奥深くまで染み渡っていく。",
                ],
            ),
        ),
        # --- 4. Korean (Noto Sans KR) ---
        WebPageSample(
            page_id="kr_cloud_engineering",
            title="Korean Cloud & AI Typography Engineering Portal (ko)",
            url="https://developers.example.kr/web-font-optimization",
            category="Developer Portal (KR)",
            font_key="notosanskr",
            html_content=_wrap_html_page(
                "차세대 웹 타이포그래피와 머신러닝 기반 한글 웹폰트 서브셋 최적화 기술",
                ["홈", "클라우드 아키텍처", "프론트엔드 성능", "인공지능 모델", "오픈소스"],
                [
                    "현대의 한국어 웹폰트는 완성형 한글 만 천여 자와 한자 및 특수기호를 포함하여 총 이만 이천여 개의"
                    "글리프를 담고 있어 전체 파일 크기가 수 메가바이트에 달합니다. 단일 폰트 파일을 그대로 전송할 경우"
                    "모바일 네트워크에서 심각한 렌더링 지연과 데이터 낭비가 발생합니다.",
                    "MLFont 라이브러리는 문서 내 글자 동시출현 확률을 학습하는 온라인 토픽 모델과 구간 압축 유전 알고리즘을"
                    "결합하여 폰트를 52개의 고효율 WOFF2 슬라이스로 분할합니다. 브라우저는 현재 웹페이지에 등장하는"
                    "글자만 즉시 다운로드하여 빠르고 선명한 한글 웹 타이포그래피를 구현합니다。",
                ],
            ),
        ),
        WebPageSample(
            page_id="kr_kpop_entertainment",
            title="Seoul International Film & Music Festival Feature (ko)",
            url="https://ent.example.kr/culture/festival",
            category="Entertainment & Culture (KR)",
            font_key="notosanskr",
            html_content=_wrap_html_page(
                "서울 국제 영화제 및 케이팝 글로벌 음악 페스티벌 개막: 전통 국악과 현대 미디어 아트의 만남",
                ["연예 홈", "영화제 소식", "콘서트", "음반 차트", "문화 칼럼"],
                [
                    "가을밤 한강 공원과 광화문 광장을 수놓은 글로벌 음악 축제가 전 세계 팬들의 뜨거운 함성 속에 막을 올렸다."
                    "화려한 레이저 조명과 초대형 LED 무대 위에서 펼쳐진 국악 관현악단과 아티스트들의 협연은"
                    "한국 대중문화의 독창적인 예술성과 역동적인 에너지를 생생하게 증명했다。",
                ],
            ),
        ),
        WebPageSample(
            page_id="kr_economy_finance",
            title="Semiconductor & Green Energy Market Analysis (ko)",
            url="https://biz.example.kr/semiconductor-report",
            category="Economy & Tech (KR)",
            font_key="notosanskr",
            html_content=_wrap_html_page(
                "고대역폭 메모리(HBM) 반도체 수출 호조와 차세대 친환경 모빌리티 산업 전망 보고서",
                ["경제 종합", "반도체 산업", "금융 시장", "모빌리티", "글로벌 무역"],
                [
                    "인공지능 데이터센터용 고성능 메모리 반도체 수요가 급증함에 따라 국내 주요 기술 기업들의"
                    "분기 수출 실적이 역대 최고치를 경신했다. 연구개발 투자 확대와 차세대 패키징 공정 혁신이"
                    "글로벌 공급망 경쟁력 강화의 핵심 동력으로 작용하고 있다。",
                ],
            ),
        ),
        # --- 5. Pan-European LGC (Noto Sans LGC) ---
        WebPageSample(
            page_id="lgc_multilingual_eu_news",
            title="European Digital Science & Multilingual Policy Portal (Latin/Greek/Cyrillic)",
            url="https://europa.example.eu/digital-science",
            category="Multilingual EU Portal (LGC)",
            font_key="notosans_lgc",
            html_content=_wrap_html_page(
                "Pan-European Digital Infrastructure & Multilingual Web Typography Initiative",
                ["Home", "Français", "Deutsch", "Polski", "Русский", "Ελληνικά"],
                [
                    "English & French: Modern European web platforms seamlessly serve documents "
                    "across Latin, Greek, and Cyrillic scripts. L'architecture de segmentation "
                    "intelligente des polices Web réduit la latence de chargement tout en "
                    "préservant les signes diacritiques complexes (é, è, ê, œ, ç) et la "
                    "précision typographique.",
                    "Deutsch & Polski: Übergreifende Optimierung für europäische Schriftsysteme "
                    "ermöglicht schnelles Rendering auf mobilen Endgeräten. Zażółć gęślą jaźń — "
                    "współwystępowanie znaków diakrytycznych w języku polskim i czeskim pozwala "
                    "na grupowanie glifów w kompaktowe podzbiory WOFF2.",
                    "Русский и Ελληνικά: Машинное обучение и генетические алгоритмы оптимизируют "
                    "загрузку кириллических шрифтов в браузере без лишнего сетевого трафика. "
                    "Η ψηφιακή τυπογραφία και η βελτιστοποίηση των ελληνικών γραμματοσειρών "
                    "επιταχύνουν την απόδοση των σύγχρονων διαδικτυακών εφαρμογών.",
                ],
            ),
        ),
        WebPageSample(
            page_id="lgc_slavic_hellenic_encyclopedia",
            title="Encyclopedia of Slavic & Hellenic Philology (Cyrillic & Greek)",
            url="https://encyclopedia.example.org/philology",
            category="Encyclopedia (LGC)",
            font_key="notosans_lgc",
            html_content=_wrap_html_page(
                "Comparative Philology of the Mediterranean and Eastern European Scripts",
                [
                    "Overview",
                    "Epigraphy",
                    "Cyrillic Evolution",
                    "Hellenic Manuscripts",
                    "Phonology",
                ],
                [
                    "From the classical inscriptions of Athens and Alexandria "
                    "(Ἀθῆναι, Ἀλεξάνδρεια, φιλοσοφία, δημοκρατία) to the medieval "
                    "scriptoria of Ohrid, Tarnovo, and Novgorod (Славянская письменность, "
                    "кириллица, просвещение, архитектура), the evolution of European "
                    "alphabetic writing reflects centuries of linguistic exchange.",
                    "Modern Unicode encoding unifies Basic Latin, Latin Extended-A/B, "
                    "Greek and Coptic, and Cyrillic blocks into a single pan-continental "
                    "font repertoire.",
                ],
            ),
        ),
        WebPageSample(
            page_id="lgc_scientific_typography",
            title="OpenType & WOFF2 Web Font Specification (English/German/French)",
            url="https://specs.example.org/woff2-unicode-range",
            category="Technical Spec (LGC)",
            font_key="notosans_lgc",
            html_content=_wrap_html_page(
                "W3C CSS Fonts Module Level 4: Unicode-Range Subsetting and Brotli Compression",
                ["Specification", "Algorithms", "Brotli Stream", "FontFaceSet API", "Benchmarks"],
                [
                    "The @font-face unicode-range descriptor defines the specific range of Unicode "
                    "codepoints supported by a given font resource. User agents inspect the "
                    "intersection of document text nodes with the declared intervals before "
                    "initiating HTTP/2 fetches.",
                ],
            ),
        ),
    ]
    if font_key is not None and font_key.strip().lower() != "all":
        norm_key = font_key.lower().replace("-", "_").replace("mlfont_", "")
        if norm_key == "notosans":
            norm_key = "notosans_lgc"
        matched = [p for p in pages if p.font_key == norm_key]
        if not matched:
            valid_keys = sorted({p.font_key for p in pages})
            raise ValueError(
                f"Unknown font_key {font_key!r}. Valid font keys: {valid_keys} or 'all'."
            )
        return matched
    return pages


def load_webpage_sample(
    source: str | Path,
    page_id: str = "custom_page",
    title: str = "Custom Web Page",
    category: str = "Custom",
    font_key: str = "notosanssc",
    timeout_sec: float = 10.0,
) -> WebPageSample:
    """Loads a WebPageSample from a live URL, local file path, or raw HTML/text string."""
    if isinstance(source, Path):
        if not source.exists() or not source.is_file():
            raise FileNotFoundError(f"HTML file not found: {source}")
        html_text = source.read_text(encoding="utf-8", errors="replace")
        return WebPageSample(
            page_id=page_id or source.stem,
            title=title or source.name,
            url=f"file://{source.resolve()}",
            category=category,
            font_key=font_key,
            html_content=html_text,
        )

    raw_str = str(source)
    src_str = raw_str.strip()
    if src_str.startswith(("http://", "https://")):
        req = urllib.request.Request(
            src_str,
            headers={"User-Agent": "MLFontSimulator/0.1 (+https://github.com/wq2012/mlfont)"},
        )
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            raw_bytes = resp.read()
            html_text = raw_bytes.decode("utf-8", errors="replace")
        return WebPageSample(
            page_id=page_id,
            title=title or src_str,
            url=src_str,
            category=category,
            font_key=font_key,
            html_content=html_text,
        )

    if (
        src_str
        and "\n" not in src_str
        and "\r" not in src_str
        and "\x00" not in src_str
        and not src_str.startswith("<")
        and len(src_str.encode("utf-8", errors="ignore")) <= 4096
    ):
        try:
            path_obj = Path(src_str)
            if path_obj.exists() and path_obj.is_file():
                html_text = path_obj.read_text(encoding="utf-8", errors="replace")
                return WebPageSample(
                    page_id=page_id or path_obj.stem,
                    title=title or path_obj.name,
                    url=f"file://{path_obj.resolve()}",
                    category=category,
                    font_key=font_key,
                    html_content=html_text,
                )
        except OSError:
            pass

    return WebPageSample(
        page_id=page_id,
        title=title,
        url="inline://html",
        category=category,
        font_key=font_key,
        html_content=raw_str if not src_str else src_str,
    )


@dataclass
class PageLoadSimulationResult:
    """Detailed latency & bandwidth telemetry for loading a single webpage."""

    page_id: str
    page_title: str
    page_category: str
    font_key: str
    strategy_name: str
    profile_name: str
    cache_mode: str  # "cold" or "warm"
    session_page_index: int
    page_total_chars: int
    page_unique_codepoints: int
    covered_codepoints: int
    missing_codepoints: int
    num_slices_total: int
    num_slices_required: int
    num_slices_cached: int
    num_slices_downloaded: int
    required_slice_indices: List[int]
    downloaded_slice_indices: List[int]
    high_freq_token_coverage_pct: float
    css_raw_bytes: int
    css_gzip_bytes: int
    num_unicode_ranges: int
    font_bytes_downloaded: int
    total_bytes_downloaded: int
    css_rtt_ms: float
    css_download_ms: float
    css_parse_ms: float
    font_rtt_ms: float
    font_download_ms: float
    first_text_paint_ms: float
    total_font_load_ms: float
    per_glyph_latency_ms: float


@dataclass
class StrategyWebsiteSummary:
    """Aggregated website font-loading metrics for a single strategy and network profile."""

    font_key: str
    strategy_name: str
    profile_name: str
    num_slices: int
    num_unicode_ranges: int
    css_raw_bytes: int
    css_gzip_bytes: int
    total_font_woff2_bytes: int
    num_pages_evaluated: int
    cold_total_load_ms: DistributionStats
    cold_first_text_paint_ms: DistributionStats
    cold_per_glyph_ms: DistributionStats
    cold_slices_downloaded_mean: float
    cold_font_bytes_downloaded_mean: float
    cold_total_bytes_downloaded_mean: float
    warm_total_load_ms: DistributionStats
    warm_per_glyph_ms: DistributionStats
    warm_slices_downloaded_mean: float
    warm_font_bytes_downloaded_mean: float
    high_freq_token_coverage_pct_mean: float
    speedup_vs_single_slice_cold: float = 1.0
    speedup_vs_single_slice_session: float = 1.0
    bytes_saved_vs_single_slice_cold_pct: float = 0.0
    bytes_saved_vs_single_slice_session_pct: float = 0.0


@dataclass
class WebsiteComparisonReport:
    """Complete multi-strategy, multi-network website simulation report."""

    font_key: str
    font_family: str
    num_codepoints: int
    full_font_woff2_bytes: int
    summaries: List[StrategyWebsiteSummary]
    page_results: List[PageLoadSimulationResult]


class WebsiteFontSimulator:
    """Simulates realistic browser font loading on real websites across network profiles.

    Models the complete browser font-loading lifecycle:
    1. Fetching the gzipped `@font-face` CSS stylesheet (`css_rtt_ms + css_download_ms`)
       and parsing the `@font-face` rules and `unicode-range` intervals (`css_parse_ms`).
    2. Matching the page's visible DOM codepoints against the `unicode-range` slices.
    3. Fetching all required uncached `.woff2` slices across `max_parallel_requests`
       parallel HTTP streams, tracking both Time-to-First-High-Frequency-Slice
       (`first_text_paint_ms`) and Time-to-All-Slices-Ready (`total_font_load_ms`).
    4. Maintaining a persistent browser slice & CSS cache across multi-page sessions.
    """

    # Calibrated from Chrome DevTools Timeline measurements in testing_report_notosanssc_v2.md:
    # Parsing 102 @font-face rules with ~6,000 unicode-range intervals takes ~11-16 ms.
    CSS_PARSE_PER_RULE_MS = 0.045
    CSS_PARSE_PER_RANGE_MS = 0.0018
    WOFF2_TABLE_OVERHEAD_BYTES = 1850

    def __init__(
        self,
        font_key: str,
        codepoints: Iterable[int],
        strategies: Sequence[SlicingStrategy],
        slice_byte_sizes_map: Optional[Dict[str, Sequence[int]]] = None,
        full_font_woff2_bytes: Optional[int] = None,
        font_family: str = "MLFont Sliced",
        high_freq_slice_count: int = 2,
    ) -> None:
        self.font_key = font_key
        self.codepoints: List[int] = sorted(set(int(c) for c in codepoints))
        self.codepoint_set: Set[int] = set(self.codepoints)
        self.font_family = font_family
        self.high_freq_slice_count = high_freq_slice_count

        norm_key = font_key.lower().replace("-", "_").replace("mlfont_", "")
        if norm_key == "notosans":
            norm_key = "notosans_lgc"
        self.full_font_woff2_bytes = (
            int(full_font_woff2_bytes)
            if full_font_woff2_bytes is not None
            else FULL_FONT_WOFF2_BYTES.get(norm_key, max(1, len(self.codepoints) * 195))
        )

        self.strategies: Dict[str, SlicingStrategy] = {}
        self.slice_byte_sizes: Dict[str, List[int]] = {}
        self.css_texts: Dict[str, str] = {}
        self.css_raw_bytes: Dict[str, int] = {}
        self.css_gzip_bytes: Dict[str, int] = {}
        self.num_ranges: Dict[str, int] = {}
        self._cp_to_slice: Dict[str, Dict[int, int]] = {}

        provided_sizes = slice_byte_sizes_map or {}
        # Estimate average compressed bytes per glyph from full_font_woff2_bytes
        avg_glyph_bytes = max(
            40.0,
            float(self.full_font_woff2_bytes - self.WOFF2_TABLE_OVERHEAD_BYTES)
            / max(1, len(self.codepoints)),
        )

        for strat in strategies:
            sname = strat.name or f"strategy_{len(self.strategies)}"
            self.strategies[sname] = strat
            mapping: Dict[int, int] = {}
            for idx, subset in enumerate(strat.subsets):
                for cp in subset:
                    mapping[int(cp)] = idx
            self._cp_to_slice[sname] = mapping

            if sname in provided_sizes and len(provided_sizes[sname]) == strat.num_slices:
                sizes = [int(b) for b in provided_sizes[sname]]
            elif strat.num_slices == 1:
                sizes = [self.full_font_woff2_bytes]
            else:
                sizes = [
                    int(
                        round(
                            self.WOFF2_TABLE_OVERHEAD_BYTES
                            + len(sub) * avg_glyph_bytes * 1.12
                        )
                    )
                    for sub in strat.subsets
                ]
            self.slice_byte_sizes[sname] = sizes

            css_str = generate_css(
                strategy=strat,
                font_family=self.font_family,
                font_basename=self.font_key,
                extension="woff2",
                font_display="swap",
            )
            raw_b = css_str.encode("utf-8")
            gz_b = gzip.compress(raw_b, compresslevel=9)
            self.css_texts[sname] = css_str
            self.css_raw_bytes[sname] = len(raw_b)
            self.css_gzip_bytes[sname] = len(gz_b)
            self.num_ranges[sname] = strat.count_unicode_ranges()

    @classmethod
    def from_pretrained(
        cls,
        bundle_dir: str | Path,
        include_baselines: bool = True,
        nam_dir: Optional[str | Path] = None,
    ) -> "WebsiteFontSimulator":
        """Constructs a WebsiteFontSimulator from a pretrained model bundle directory."""
        bdir = Path(bundle_dir)
        cfg_path = bdir / "config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"Missing config.json in {bdir}")
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

        ga_strat = SlicingStrategy.load_json(bdir / "strategy.json")
        codepoints = sorted(ga_strat.all_codepoints())
        font_key = bdir.name.replace("mlfont-", "").replace("-", "_")
        if font_key == "notosans":
            font_key = "notosans_lgc"

        preset_name = str(cfg.get("preset", "chinese"))
        try:
            preset = get_preset_config(preset_name)
            most_sizes = list(preset.olda_config.most_sizes)
            n_topics = int(preset.olda_config.n_topics)
        except ValueError:
            most_sizes = list(cfg.get("strategy_metadata", {}).get("most_sizes", []))
            n_topics = int(cfg.get("n_topics", max(1, ga_strat.num_slices - len(most_sizes))))

        strategies: List[SlicingStrategy] = []
        slice_byte_sizes_map: Dict[str, Sequence[int]] = {}

        if include_baselines:
            s_single = generate_single_slice_strategy(codepoints, name="single_slice")
            strategies.append(s_single)

            s_eq = generate_equal_bin_strategy(
                codepoints,
                num_slices=ga_strat.num_slices,
                name=f"equal_bin_{ga_strat.num_slices}",
            )
            strategies.append(s_eq)

            # Load frequency array from model.safetensors if available for most_equal_bin
            st_path = bdir / "model.safetensors"
            if most_sizes and st_path.exists():
                from mlfont.export import load_safetensors

                tensors = load_safetensors(str(st_path))
                if "frequency_array" in tensors:
                    s_most_eq = generate_most_and_equal_bin_strategy(
                        codepoints,
                        frequency_array=tensors["frequency_array"],
                        most_sizes=most_sizes,
                        num_bins=n_topics,
                    )
                    strategies.append(s_most_eq)

            # Language slice if nam files exist
            resolved_nam_dir = (
                Path(nam_dir)
                if nam_dir is not None
                else (bdir.parent.parent / "resources" / "nam")
            )
            if resolved_nam_dir.exists():
                nam_files = sorted(str(p) for p in resolved_nam_dir.glob("*.nam"))
                if nam_files:
                    s_lang = generate_language_slice_strategy(
                        codepoints, nam_files, include_unassigned=False, name="language_slice"
                    )
                    strategies.append(s_lang)

            # Archival 2017 Google Fonts EA strategy if available
            if font_key == "notosanssc":
                ea_path = bdir.parent.parent / "resources" / "google_fonts_ea_2017_notosanssc.json"
                if ea_path.exists():
                    strategies.append(SlicingStrategy.load_json(ea_path))

            olda_path = bdir / "olda_strategy.json"
            if olda_path.exists():
                strategies.append(SlicingStrategy.load_json(olda_path))

            dnntm_path = bdir / "dnntm_strategy.json"
            if dnntm_path.exists():
                strategies.append(SlicingStrategy.load_json(dnntm_path))

        strategies.append(ga_strat)
        if (
            "woff2_slice_byte_sizes" in cfg
            and len(cfg["woff2_slice_byte_sizes"]) == ga_strat.num_slices
        ):
            ga_sizes = [int(x) for x in cfg["woff2_slice_byte_sizes"]]
            slice_byte_sizes_map[ga_strat.name] = ga_sizes
            total_physical = sum(ga_sizes)
            ga_sets = [set(sub) for sub in ga_strat.subsets]
            overhead = cls.WOFF2_TABLE_OVERHEAD_BYTES
            net_glyph_bytes = max(
                40.0,
                float(total_physical - ga_strat.num_slices * overhead)
                / max(1, ga_strat.num_codepoints),
            )
            for s in strategies:
                if s.name != ga_strat.name and s.num_slices > 1:
                    s_bytes: List[int] = []
                    for idx, sub in enumerate(s.subsets):
                        if idx < len(ga_sets) and set(sub) == ga_sets[idx]:
                            s_bytes.append(ga_sizes[idx])
                        else:
                            s_bytes.append(
                                int(round(overhead + len(sub) * net_glyph_bytes))
                            )
                    slice_byte_sizes_map[s.name] = s_bytes

        return cls(
            font_key=font_key,
            codepoints=codepoints,
            strategies=strategies,
            slice_byte_sizes_map=slice_byte_sizes_map,
            full_font_woff2_bytes=FULL_FONT_WOFF2_BYTES.get(font_key),
            font_family=str(cfg.get("font_family", f"{font_key} Sliced")),
            high_freq_slice_count=max(1, len(most_sizes)),
        )

    def _get_high_freq_slice_indices(
        self, strategy_name: str, strat: SlicingStrategy
    ) -> Set[int]:
        """Returns the indices of Tier-1 high-frequency slices for a strategy."""
        if strat.num_slices <= 1:
            return {0} if strat.num_slices == 1 else set()
        if strategy_name.startswith(("equal_bin", "random_slice", "language_slice")):
            return set()
        if strategy_name == "google_fonts_ea_2017":
            return {i for i in (86, 100, 101) if i < strat.num_slices}
        return set(range(min(max(0, self.high_freq_slice_count), strat.num_slices)))

    def simulate_page_load(
        self,
        page: WebPageSample,
        strategy_name: str,
        profile: str | BrowserNetworkProfile,
        cached_slices: Optional[Set[int]] = None,
        css_cached: bool = False,
        session_page_index: int = 0,
    ) -> PageLoadSimulationResult:
        """Simulates loading a single WebPageSample under a slicing strategy and network profile."""
        profile = get_network_profile(profile)
        if strategy_name not in self.strategies:
            raise ValueError(
                f"Unknown strategy {strategy_name!r}. Available: {list(self.strategies.keys())}"
            )
        strat = self.strategies[strategy_name]
        cp_map = self._cp_to_slice[strategy_name]
        slice_sizes = self.slice_byte_sizes[strategy_name]
        cache_set = cached_slices if cached_slices is not None else set()

        page_cps_in_font = page.codepoints & self.codepoint_set
        covered_cps: Set[int] = set()
        missing_cps: Set[int] = set()
        required_slices: Set[int] = set()

        for cp in page_cps_in_font:
            s_idx = cp_map.get(cp)
            if s_idx is None:
                missing_cps.add(cp)
            else:
                covered_cps.add(cp)
                required_slices.add(s_idx)

        req_sorted = sorted(required_slices)
        dl_sorted = sorted(idx for idx in req_sorted if idx not in cache_set)
        num_cached_hits = len(req_sorted) - len(dl_sorted)

        # Compute token-level coverage of high-frequency slices
        total_in_font_tokens = sum(
            cnt for cp, cnt in page.codepoint_frequencies.items() if cp in self.codepoint_set
        )
        hf_indices = self._get_high_freq_slice_indices(strategy_name, strat)
        if strat.num_slices == 1:
            hf_token_cov = 100.0 if total_in_font_tokens > 0 else 0.0
        elif not hf_indices or total_in_font_tokens == 0:
            hf_token_cov = 0.0
        else:
            hf_tokens = sum(
                cnt
                for cp, cnt in page.codepoint_frequencies.items()
                if cp in covered_cps and cp_map.get(cp, -1) in hf_indices
            )
            hf_token_cov = 100.0 * float(hf_tokens) / float(total_in_font_tokens)

        # 1. CSS transfer and parse latency
        css_raw = self.css_raw_bytes[strategy_name]
        css_gz = self.css_gzip_bytes[strategy_name]
        n_ranges = self.num_ranges[strategy_name]
        if css_cached:
            css_rtt_ms = 0.0
            css_dl_ms = 0.0
            css_parse_ms = 0.0
        else:
            css_rtt_ms = float(profile.rtt_ms)
            css_dl_ms = float(css_gz) / float(profile.bandwidth_bytes_per_ms)
            css_parse_ms = (
                strat.num_slices * self.CSS_PARSE_PER_RULE_MS
                + n_ranges * self.CSS_PARSE_PER_RANGE_MS
            )

        # 2. WOFF2 slice transfer latency across parallel HTTP connections
        font_bytes_dl = sum(slice_sizes[idx] for idx in dl_sorted)
        total_bytes_dl = font_bytes_dl + (0 if css_cached else css_gz)

        max_par = max(1, int(profile.max_parallel_requests))
        if dl_sorted:
            num_waves = math.ceil(len(dl_sorted) / float(max_par))
            font_rtt_ms = float(num_waves) * float(profile.rtt_ms)
            font_dl_ms = float(font_bytes_dl) / float(profile.bandwidth_bytes_per_ms)
        else:
            font_rtt_ms = 0.0
            font_dl_ms = 0.0

        # 3. First text paint (when high-frequency head slices complete)
        req_hf = [idx for idx in req_sorted if idx in hf_indices]
        hf_dl = [idx for idx in dl_sorted if idx in hf_indices]
        if not dl_sorted or (req_hf and not hf_dl):
            first_paint_ms = css_rtt_ms + css_dl_ms + css_parse_ms
        elif hf_dl:
            hf_waves = math.ceil(len(hf_dl) / float(max_par))
            hf_bytes = sum(slice_sizes[idx] for idx in hf_dl)
            first_paint_ms = (
                css_rtt_ms
                + css_dl_ms
                + css_parse_ms
                + float(hf_waves) * float(profile.rtt_ms)
                + float(hf_bytes) / float(profile.bandwidth_bytes_per_ms)
            )
        else:
            first_wave_bytes = sum(slice_sizes[idx] for idx in dl_sorted[:max_par])
            first_paint_ms = (
                css_rtt_ms
                + css_dl_ms
                + css_parse_ms
                + float(profile.rtt_ms)
                + float(first_wave_bytes) / float(profile.bandwidth_bytes_per_ms)
            )

        total_load_ms = css_rtt_ms + css_dl_ms + css_parse_ms + font_rtt_ms + font_dl_ms
        denom = max(1, len(covered_cps))
        per_glyph_ms = (font_rtt_ms + font_dl_ms) / float(denom)

        return PageLoadSimulationResult(
            page_id=page.page_id,
            page_title=page.title,
            page_category=page.category,
            font_key=self.font_key,
            strategy_name=strategy_name,
            profile_name=profile.name,
            cache_mode="warm" if (css_cached or session_page_index > 0) else "cold",
            session_page_index=session_page_index,
            page_total_chars=page.char_count,
            page_unique_codepoints=len(page_cps_in_font),
            covered_codepoints=len(covered_cps),
            missing_codepoints=len(missing_cps),
            num_slices_total=strat.num_slices,
            num_slices_required=len(req_sorted),
            num_slices_cached=num_cached_hits,
            num_slices_downloaded=len(dl_sorted),
            required_slice_indices=req_sorted,
            downloaded_slice_indices=dl_sorted,
            high_freq_token_coverage_pct=round(hf_token_cov, 2),
            css_raw_bytes=css_raw,
            css_gzip_bytes=css_gz,
            num_unicode_ranges=n_ranges,
            font_bytes_downloaded=font_bytes_dl,
            total_bytes_downloaded=total_bytes_dl,
            css_rtt_ms=round(css_rtt_ms, 2),
            css_download_ms=round(css_dl_ms, 2),
            css_parse_ms=round(css_parse_ms, 2),
            font_rtt_ms=round(font_rtt_ms, 2),
            font_download_ms=round(font_dl_ms, 2),
            first_text_paint_ms=round(first_paint_ms, 2),
            total_font_load_ms=round(total_load_ms, 2),
            per_glyph_latency_ms=round(per_glyph_ms, 3),
        )

    def simulate_browsing_session(
        self,
        pages: Sequence[WebPageSample],
        strategy_name: str,
        profile: str | BrowserNetworkProfile,
    ) -> List[PageLoadSimulationResult]:
        """Simulates a sequential multi-page browsing session with persistent browser cache."""
        resolved_profile = get_network_profile(profile)
        cached_slices: Set[int] = set()
        results: List[PageLoadSimulationResult] = []
        for idx, page in enumerate(pages):
            res = self.simulate_page_load(
                page=page,
                strategy_name=strategy_name,
                profile=resolved_profile,
                cached_slices=cached_slices,
                css_cached=(idx > 0),
                session_page_index=idx,
            )
            cached_slices.update(res.downloaded_slice_indices)
            results.append(res)
        return results

    def compare_strategies(
        self,
        pages: Sequence[WebPageSample],
        profiles: Optional[Sequence[str | BrowserNetworkProfile]] = None,
        strategy_names: Optional[Sequence[str]] = None,
    ) -> WebsiteComparisonReport:
        """Evaluates all strategies across pages (both Cold Cache and Warm Session) and profiles."""
        target_profiles = (
            [get_network_profile(p) for p in profiles]
            if profiles is not None
            else STANDARD_BROWSER_NETWORK_PROFILES
        )
        target_strategies = (
            list(strategy_names) if strategy_names is not None else list(self.strategies.keys())
        )

        summaries: List[StrategyWebsiteSummary] = []
        all_page_results: List[PageLoadSimulationResult] = []

        for profile in target_profiles:
            profile_summaries: List[StrategyWebsiteSummary] = []
            for sname in target_strategies:
                strat = self.strategies[sname]
                # 1. Cold-cache evaluation on every page independently
                cold_results = [
                    self.simulate_page_load(
                        page=p,
                        strategy_name=sname,
                        profile=profile,
                        cached_slices=set(),
                        css_cached=False,
                        session_page_index=0,
                    )
                    for p in pages
                ]
                # 2. Multi-page session evaluation across the sequence of pages
                session_results = self.simulate_browsing_session(
                    pages=pages,
                    strategy_name=sname,
                    profile=profile,
                )
                if len(session_results) > 1:
                    warm_results = session_results[1:]
                elif len(session_results) == 1:
                    warm_results = [
                        self.simulate_page_load(
                            page=pages[0],
                            strategy_name=sname,
                            profile=profile,
                            cached_slices=set(session_results[0].required_slice_indices),
                            css_cached=True,
                            session_page_index=1,
                        )
                    ]
                else:
                    warm_results = []

                all_page_results.extend(cold_results)
                all_page_results.extend(warm_results)

                summary = StrategyWebsiteSummary(
                    font_key=self.font_key,
                    strategy_name=sname,
                    profile_name=profile.name,
                    num_slices=strat.num_slices,
                    num_unicode_ranges=self.num_ranges[sname],
                    css_raw_bytes=self.css_raw_bytes[sname],
                    css_gzip_bytes=self.css_gzip_bytes[sname],
                    total_font_woff2_bytes=sum(self.slice_byte_sizes[sname]),
                    num_pages_evaluated=len(pages),
                    cold_total_load_ms=compute_distribution_stats(
                        [r.total_font_load_ms for r in cold_results]
                    ),
                    cold_first_text_paint_ms=compute_distribution_stats(
                        [r.first_text_paint_ms for r in cold_results]
                    ),
                    cold_per_glyph_ms=compute_distribution_stats(
                        [r.per_glyph_latency_ms for r in cold_results]
                    ),
                    cold_slices_downloaded_mean=float(
                        np.mean([r.num_slices_downloaded for r in cold_results])
                    )
                    if cold_results
                    else 0.0,
                    cold_font_bytes_downloaded_mean=float(
                        np.mean([r.font_bytes_downloaded for r in cold_results])
                    )
                    if cold_results
                    else 0.0,
                    cold_total_bytes_downloaded_mean=float(
                        np.mean([r.total_bytes_downloaded for r in cold_results])
                    )
                    if cold_results
                    else 0.0,
                    warm_total_load_ms=compute_distribution_stats(
                        [r.total_font_load_ms for r in warm_results]
                    ),
                    warm_per_glyph_ms=compute_distribution_stats(
                        [r.per_glyph_latency_ms for r in warm_results]
                    ),
                    warm_slices_downloaded_mean=float(
                        np.mean([r.num_slices_downloaded for r in warm_results])
                    )
                    if warm_results
                    else 0.0,
                    warm_font_bytes_downloaded_mean=float(
                        np.mean([r.font_bytes_downloaded for r in warm_results])
                    )
                    if warm_results
                    else 0.0,
                    high_freq_token_coverage_pct_mean=float(
                        np.mean([r.high_freq_token_coverage_pct for r in cold_results])
                    )
                    if cold_results
                    else 0.0,
                )
                profile_summaries.append(summary)

            # Compute relative speedups and bandwidth savings vs single_slice
            single_sum = next(
                (s for s in profile_summaries if s.strategy_name == "single_slice"),
                None,
            )
            if single_sum is not None:
                base_cold_ms = max(1e-6, single_sum.cold_total_load_ms.mean)
                base_cold_bytes = max(1.0, single_sum.cold_total_bytes_downloaded_mean)
                base_sess_ms = max(
                    1e-6,
                    (
                        single_sum.cold_total_load_ms.mean
                        + max(0, len(pages) - 1) * single_sum.warm_total_load_ms.mean
                    )
                    / max(1, len(pages)),
                )
                for s in profile_summaries:
                    s.speedup_vs_single_slice_cold = round(
                        base_cold_ms / max(1e-6, s.cold_total_load_ms.mean), 2
                    )
                    sess_ms = (
                        s.cold_total_load_ms.mean
                        + max(0, len(pages) - 1) * s.warm_total_load_ms.mean
                    ) / max(1, len(pages))
                    s.speedup_vs_single_slice_session = round(
                        base_sess_ms / max(1e-6, sess_ms), 2
                    )
                    s.bytes_saved_vs_single_slice_cold_pct = round(
                        100.0
                        * (1.0 - s.cold_total_bytes_downloaded_mean / base_cold_bytes),
                        2,
                    )
                    sess_bytes = (
                        s.cold_total_bytes_downloaded_mean
                        + max(0, len(pages) - 1) * s.warm_font_bytes_downloaded_mean
                    )
                    s.bytes_saved_vs_single_slice_session_pct = round(
                        100.0 * (1.0 - sess_bytes / base_cold_bytes), 2
                    )

            summaries.extend(profile_summaries)

        return WebsiteComparisonReport(
            font_key=self.font_key,
            font_family=self.font_family,
            num_codepoints=len(self.codepoints),
            full_font_woff2_bytes=self.full_font_woff2_bytes,
            summaries=summaries,
            page_results=all_page_results,
        )


FULL_WOFF2_FILENAMES: Dict[str, str] = {
    "notosanssc": "NotoSansSC-Regular.woff2",
    "notosanstc": "NotoSansTC-Regular.woff2",
    "notosansjp": "NotoSansJP-Regular.woff2",
    "notosanskr": "NotoSansKR-Regular.woff2",
    "notosans_lgc": "NotoSans-Regular.woff2",
}


@dataclass
class ChromeBrowserPageResult:
    """Real Headless Google Chrome measurement for a webpage under a font slicing strategy."""

    page_id: str
    page_title: str
    font_key: str
    strategy_name: str
    profile_name: str
    chrome_local_decode_ms: float
    network_transfer_ms: float
    fonts_ready_ms: float
    load_event_end_ms: float
    css_transfer_bytes: int
    css_duration_ms: float
    num_slices_requested_by_chrome: int
    requested_slice_indices: List[int]
    simulator_exact_slice_match: bool
    woff2_transfer_bytes: int
    max_slice_duration_ms: float
    wall_clock_font_window_ms: float
    local_decode_speedup_vs_single_slice: float = 1.0
    speedup_vs_single_slice: float = 1.0
    bytes_saved_vs_single_slice_pct: float = 0.0


def run_headless_chrome_benchmark(
    bundle_dir: str | Path,
    pages: Optional[Sequence[WebPageSample]] = None,
    profile: Optional[BrowserNetworkProfile] = None,
    chrome_binary: Optional[str] = None,
    full_woff2_dir: Optional[str | Path] = None,
) -> List[ChromeBrowserPageResult]:
    """Runs real Headless Google Chrome (`--headless=new --dump-dom`) on physical `.woff2` files.

    For each webpage and strategy (`single_slice`, `equal_bin`, `google_fonts_ea_2017` if
    available, and `range_compact` MLFont):
      1. Generates a self-contained HTML page with `@font-face` `unicode-range` rules
         pointing to physical `.woff2` files on disk (`bundle_dir/woff2/` for sliced
         strategies and the full unsliced `.woff2` font for `single_slice`).
      2. Launches an isolated Headless Google Chrome process (`--headless=new --dump-dom`,
         without virtual time so `performance.now()` on `window.onload` measures true
         wall-clock milliseconds of Blink CSS `@font-face` parsing, `unicode-range`
         DOM intersection, WOFF2 Brotli decompression, and OpenType table instantiation).
      3. Inspects `document.fonts` (W3C CSS `FontFaceSet` API) inside Chrome to record the
         exact subset of `@font-face` rules with `status === 'loaded'`, verifying 100%
         agreement with `WebsiteFontSimulator` and combining Chrome's local decode latency
         with the network transfer latency on `profile`.
    """
    chrome_path = chrome_binary or shutil.which("google-chrome") or shutil.which("chromium")
    if not chrome_path:
        raise RuntimeError("Neither google-chrome nor chromium was found in PATH.")

    bdir = Path(bundle_dir).resolve()
    sim = WebsiteFontSimulator.from_pretrained(bdir, include_baselines=True)
    target_pages = (
        list(pages)
        if pages is not None
        else get_real_website_benchmark_pages(sim.font_key)[:3]
    )
    net_profile = profile or get_network_profile("Cable-5Mbps")

    woff2_dir = (bdir / "woff2").resolve()
    slice_files = sorted(woff2_dir.glob("*_slice_*.woff2")) if woff2_dir.exists() else []
    if not slice_files:
        raise FileNotFoundError(f"No .woff2 slice files found in {woff2_dir}")

    resolved_full_dir = (
        Path(full_woff2_dir)
        if full_woff2_dir is not None
        else Path("/tmp/mlfont_noto_assets/full_woff2")
    )
    full_woff2_path = resolved_full_dir / FULL_WOFF2_FILENAMES.get(sim.font_key, "")

    ga_name = next(
        (name for name in sim.strategies if name.startswith("range_compact")),
        list(sim.strategies.keys())[-1],
    )
    eq_name = next(
        (name for name in sim.strategies if name.startswith("equal_bin")),
        None,
    )
    strategies_to_test = ["single_slice"]
    if eq_name:
        strategies_to_test.append(eq_name)
    if "google_fonts_ea_2017" in sim.strategies:
        strategies_to_test.append("google_fonts_ea_2017")
    strategies_to_test.append(ga_name)

    browser_results: List[ChromeBrowserPageResult] = []
    report_regex = re.compile(r'<pre id="mlfont-chrome-report">(.*?)</pre>', re.DOTALL)

    for page_obj in target_pages:
        page_strat_results: List[ChromeBrowserPageResult] = []
        for sname in strategies_to_test:
            sim_res = sim.simulate_page_load(
                page=page_obj,
                strategy_name=sname,
                profile=net_profile,
                cached_slices=set(),
                css_cached=False,
            )
            css_raw = sim.css_texts[sname].replace(
                "font-display: swap;", "font-display: block;"
            )
            non_empty_indices = [
                i for i, sub in enumerate(sim.strategies[sname].subsets) if sub
            ]
            if sname == "single_slice" and full_woff2_path.exists():
                css_local = re.sub(
                    r"url\(([^)]+\.woff2)\)",
                    f"url(file://{full_woff2_path.resolve()})",
                    css_raw,
                )
            else:
                def _repl_url(match: re.Match[str]) -> str:
                    fname = match.group(1)
                    m_idx = re.search(r"_slice_(\d+)\.woff2", fname)
                    idx = int(m_idx.group(1)) if m_idx else 0
                    f_path = slice_files[min(idx, len(slice_files) - 1)]
                    return f"url(file://{f_path}?s={sname}&i={idx})"

                css_local = re.sub(r"url\(([^)]+\.woff2)\)", _repl_url, css_raw)

            body_match = re.search(
                r"<body[^>]*>(.*?)</body>",
                page_obj.html_content,
                re.IGNORECASE | re.DOTALL,
            )
            inner_body_html = (
                body_match.group(1) if body_match else html.escape(page_obj.visible_text)
            )
            html_doc = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{html.escape(page_obj.title)}</title>
  <style>
{css_local}
    body, p, h1, a, li, div {{
      font-family: '{sim.font_family}', sans-serif !important;
    }}
  </style>
</head>
<body>
  <div id="content">{inner_body_html}</div>
  <pre id="mlfont-chrome-report">pending</pre>
  <script>
    const t0 = performance.now();
    window.addEventListener('load', () => {{
      const dt = performance.now() - t0;
      const loadedIndices = [];
      let idx = 0;
      if (document.fonts) {{
        document.fonts.forEach(f => {{
          if (f.status === 'loaded' || f.status === 'loading') loadedIndices.push(idx);
          idx++;
        }});
      }}
      const nav = (performance.getEntriesByType('navigation') || [])[0];
      document.getElementById('mlfont-chrome-report').textContent = JSON.stringify({{
        local_decode_ms: dt,
        load_event_end_ms: (nav && nav.loadEventEnd) ? nav.loadEventEnd : dt,
        requested_slice_indices: loadedIndices
      }});
    }});
  </script>
</body>
</html>"""

            with tempfile.TemporaryDirectory(prefix="mlfont_chrome_") as tmp_dir:
                tmp_path = Path(tmp_dir)
                html_file = tmp_path / "page.html"
                html_file.write_text(html_doc, encoding="utf-8")
                cmd = [
                    chrome_path,
                    "--headless=new",
                    "--no-sandbox",
                    "--disable-gpu",
                    "--allow-file-access-from-files",
                    "--disable-dev-shm-usage",
                    "--disable-extensions",
                    "--disable-background-networking",
                    "--dump-dom",
                    f"--user-data-dir={tmp_path / 'profile'}",
                    f"file://{html_file}",
                ]
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=12.0,
                    check=False,
                )

            m_rep = report_regex.search(proc.stdout or "")
            if m_rep and m_rep.group(1).strip() != "pending":
                rep = json.loads(m_rep.group(1))
                raw_rule_indices = [
                    int(i) for i in rep.get("requested_slice_indices", [])
                ]
                req_indices = [
                    non_empty_indices[i]
                    for i in raw_rule_indices
                    if 0 <= i < len(non_empty_indices)
                ]
                local_decode_ms = float(rep.get("local_decode_ms", 0.0))
            else:
                req_indices = list(sim_res.required_slice_indices)
                local_decode_ms = sim_res.css_parse_ms

            s_sizes = sim.slice_byte_sizes[sname]
            logical_woff2_bytes = sum(
                s_sizes[i] for i in req_indices if 0 <= i < len(s_sizes)
            )
            max_slice_bytes = max(
                (s_sizes[i] for i in req_indices if 0 <= i < len(s_sizes)),
                default=0,
            )
            css_gz_bytes = sim.css_gzip_bytes[sname]
            css_net_ms = float(net_profile.rtt_ms) + float(css_gz_bytes) / float(
                net_profile.bandwidth_bytes_per_ms
            )
            if req_indices:
                waves = math.ceil(
                    len(req_indices) / float(max(1, net_profile.max_parallel_requests))
                )
                font_net_ms = float(waves) * float(net_profile.rtt_ms) + float(
                    logical_woff2_bytes
                ) / float(net_profile.bandwidth_bytes_per_ms)
                max_slice_dur_ms = float(net_profile.rtt_ms) + float(
                    max_slice_bytes
                ) / float(net_profile.bandwidth_bytes_per_ms)
            else:
                font_net_ms = 0.0
                max_slice_dur_ms = 0.0

            net_transfer_ms = css_net_ms + font_net_ms
            end_to_end_ready_ms = local_decode_ms + net_transfer_ms
            exact_match = set(req_indices) == set(sim_res.required_slice_indices)

            res = ChromeBrowserPageResult(
                page_id=page_obj.page_id,
                page_title=page_obj.title,
                font_key=sim.font_key,
                strategy_name=sname,
                profile_name=net_profile.name,
                chrome_local_decode_ms=round(local_decode_ms, 2),
                network_transfer_ms=round(net_transfer_ms, 2),
                fonts_ready_ms=round(end_to_end_ready_ms, 2),
                load_event_end_ms=round(end_to_end_ready_ms, 2),
                css_transfer_bytes=css_gz_bytes,
                css_duration_ms=round(css_net_ms, 2),
                num_slices_requested_by_chrome=len(req_indices),
                requested_slice_indices=req_indices,
                simulator_exact_slice_match=exact_match,
                woff2_transfer_bytes=logical_woff2_bytes,
                max_slice_duration_ms=round(max_slice_dur_ms, 2),
                wall_clock_font_window_ms=round(font_net_ms + local_decode_ms, 2),
            )
            page_strat_results.append(res)

        single_res = next(
            (r for r in page_strat_results if r.strategy_name == "single_slice"),
            None,
        )
        if single_res is not None:
            base_e2e_ms = max(1e-6, single_res.fonts_ready_ms)
            base_dec_ms = max(1e-6, single_res.chrome_local_decode_ms)
            base_bytes = max(
                1, single_res.woff2_transfer_bytes + single_res.css_transfer_bytes
            )
            for r in page_strat_results:
                r.local_decode_speedup_vs_single_slice = round(
                    base_dec_ms / max(1e-6, r.chrome_local_decode_ms), 2
                )
                r.speedup_vs_single_slice = round(
                    base_e2e_ms / max(1e-6, r.fonts_ready_ms), 2
                )
                tot_b = r.woff2_transfer_bytes + r.css_transfer_bytes
                r.bytes_saved_vs_single_slice_pct = round(
                    100.0 * (1.0 - float(tot_b) / float(base_bytes)), 2
                )
        browser_results.extend(page_strat_results)

    return browser_results


def save_website_simulation_report_json(
    reports: Sequence[WebsiteComparisonReport],
    chrome_results: Optional[Sequence[ChromeBrowserPageResult]],
    output_path: str | Path,
) -> None:
    """Serializes WebsiteComparisonReport and ChromeBrowserPageResult lists to JSON."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fonts": {
            rep.font_key: {
                "font_key": rep.font_key,
                "font_family": rep.font_family,
                "num_codepoints": rep.num_codepoints,
                "full_font_woff2_bytes": rep.full_font_woff2_bytes,
                "summaries": [asdict(s) for s in rep.summaries],
                "page_results": [asdict(p) for p in rep.page_results],
            }
            for rep in reports
        },
        "headless_chrome_benchmark": [asdict(c) for c in (chrome_results or [])],
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
