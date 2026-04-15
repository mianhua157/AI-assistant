"""
使用 OpenAI 兼容接口调用阿里云百炼
这是官方推荐的方式，最稳定
"""
from openai import OpenAI

# 配置
API_KEY = "sk-b0f15faccf3f46cdb915d8d472e405a5"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen3.5-plus"

# 创建客户端
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

# 发起请求
response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "user", "content": "请回复'测试成功'"}
    ]
)

# 输出结果
print(f"状态码：{response.choices[0].message.content}")
