from __future__ import annotations

import os
from http import HTTPStatus

import dashscope
from dotenv import load_dotenv


load_dotenv()


def call_llm(prompt: str, *, temperature: float = 0, max_tokens: int = 1000) -> str:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("Please set DASHSCOPE_API_KEY before running the evaluation agent.")

    model = os.getenv("QWEN_MODEL", "qwen-plus")
    response = dashscope.Generation.call(
        model=model,
        prompt=prompt,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    if response.status_code == HTTPStatus.OK:
        return response.output.text

    raise RuntimeError(f"LLM judge request failed: {response.code} - {response.message}")
