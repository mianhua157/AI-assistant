"""
测试阿里云百炼 Coding Plan 的两种调用方式
"""
import os

# 你的 API Key 配置
API_KEY = "sk-sp-095dca95f43b44249197132cdab21216"
MODEL = "qwen3.5-plus"

print("=" * 60)
print("阿里云百炼 Coding Plan 调用测试")
print("=" * 60)

# ============================================================
# 方式一：使用 OpenAI 兼容接口（推荐，最通用）
# ============================================================
print("\n【方式一】使用 OpenAI 兼容接口测试...")
print("-" * 60)

try:
    from openai import OpenAI

    # Coding Plan 的 Base URL
    client = OpenAI(
        api_key=API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": "请回复'测试成功'"}
        ]
    )

    print(f"状态：成功")
    print(f"模型：{response.model}")
    print(f"回复：{response.choices[0].message.content}")

except ImportError:
    print(f"错误：需要安装 openai 库")
    print(f"运行：pip install openai")
except Exception as e:
    print(f"错误：{e}")

# ============================================================
# 方式二：使用 DashScope SDK
# ============================================================
print("\n【方式二】使用 DashScope SDK 测试...")
print("-" * 60)

try:
    import dashscope
    from http import HTTPStatus
    from dashscope import Generation

    # 设置 API Key
    dashscope.api_key = API_KEY

    # 调用（使用 messages 格式，不是 prompt）
    response = Generation.call(
        model=MODEL,
        messages=[{"role": "user", "content": "请回复'测试成功'"}],
        # 注意：DashScope 原生调用不需要设置 base_url，它会自动路由
    )

    print(f"状态码：{response.status_code}")
    if response.status_code == HTTPStatus.OK:
        print(f"状态：成功")
        print(f"回复：{response.output.choices[0].message.content}")
    else:
        print(f"错误：{response.message}")
        print(f"错误码：{response.code}")
        print(f"请求 ID: {response.request_id}")

except ImportError:
    print(f"错误：需要安装 dashscope 库")
    print(f"运行：pip install dashscope")
except Exception as e:
    print(f"错误：{e}")

print("\n" + "=" * 60)
