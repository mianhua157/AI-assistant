import re
from dataclasses import dataclass


@dataclass
class IntentResult:
    intent: str
    confidence: float
    reason: str


INTENT_PATTERNS = {
    "diagnosis": [
        r"资料里有(没有|没讲|没提到)",
        r"有没有覆盖",
        r"课程里有(没有|没讲|没提到)",
        r"为什么.*答不出来",
        r"覆盖",
        r"covered",
        r"coverage",
        r"material",
    ],
    "quiz": [
        r"出题",
        r"出\d+道题",
        r"出几道题",
        r"练习题",
        r"测验",
        r"测试我",
        r"帮我生成.*题",
        r"帮我出.*题",
        r"quiz",
        r"test me",
        r"practice questions?",
    ],
    "study_plan": [
        r"学习计划",
        r"复习计划",
        r"怎么复习",
        r"怎么学",
        r"学习路线",
        r"study plan",
        r"roadmap",
        r"review plan",
    ],
    "summary": [
        r"总结",
        r"概括",
        r"归纳",
        r"这一章",
        r"这一节",
        r"summary",
        r"summarize",
        r"overview",
    ],
    "comparison": [
        r"区别",
        r"不同",
        r"对比",
        r"比较",
        r"\bvs\b",
        r"versus",
        r"difference",
        r"compare",
    ],
    "definition": [
        r"是什么",
        r"什么是",
        r"定义",
        r"解释一下",
        r"概念",
        r"what is",
        r"define",
        r"meaning of",
    ],
}


def route_intent(question: str) -> IntentResult:
    normalized = question.lower().strip()

    for intent, patterns in INTENT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, normalized):
                return IntentResult(
                    intent=intent,
                    confidence=0.9,
                    reason=f"matched pattern: {pattern}",
                )

    return IntentResult(
        intent="general_qa",
        confidence=0.6,
        reason="no specific pattern matched; default to general_qa",
    )
