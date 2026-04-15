"""
使用原生 requests 直接调用阿里云百炼 API
跳过 SDK 可能的问题
"""
import requests
import json

API_KEY = "sk-sp-095dca95f43b44249197132cdab21216"
MODEL = "qwen3.5-plus"

# 测试两个不同的 Base URL
urls_to_test = [
    "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    "https://dashscope-beijing.aliyuncs.com/compatible-mode/v1/chat/completions",
    "https://coding.dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
]

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

payload = {
    "model": MODEL,
    "messages": [
        {"role": "user", "content": "请回复测试成功"}
    ]
}

print("测试原生 API 调用...")
print("=" * 60)

for url in urls_to_test:
    print(f"\n测试 URL: {url}")
    print("-" * 60)
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        print(f"状态码：{response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print("SUCCESS!")
            print(f"回复：{data['choices'][0]['message']['content']}")
            break
        else:
            print(f"错误：{response.text}")
    except Exception as e:
        print(f"异常：{e}")

print("=" * 60)
