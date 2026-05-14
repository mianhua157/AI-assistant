INTENT_CONFIG = {
    "definition": {
        "top_k": 4,
        "fetch_k": 10,
        "answer_style": "concept_explanation",
    },
    "comparison": {
        "top_k": 6,
        "fetch_k": 16,
        "answer_style": "comparison_table",
    },
    "summary": {
        "top_k": 8,
        "fetch_k": 18,
        "answer_style": "structured_summary",
    },
    "quiz": {
        "top_k": 8,
        "fetch_k": 18,
        "answer_style": "quiz_generation",
    },
    "diagnosis": {
        "top_k": 8,
        "fetch_k": 18,
        "answer_style": "coverage_check",
    },
    "study_plan": {
        "top_k": 8,
        "fetch_k": 18,
        "answer_style": "study_plan",
    },
    "general_qa": {
        "top_k": 5,
        "fetch_k": 12,
        "answer_style": "normal_qa",
    },
}
