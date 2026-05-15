import dashscope
from http import HTTPStatus
import os

# 设置控制台编码为 UTF-8
os.system("chcp 65001 >nul")

# 使用正确的 Coding Plan 配置
API_KEY = "sk-sp-095dca95f43b44249197132cdab21216"
# 正确的 Base URL 格式 - 必须包含 coding.dashscope.aliyuncs.com
BASE_URL = "https://coding.dashscope.aliyuncs.com/compatible-mode/v1"

dashscope.api_key = API_KEY
dashscope.base_url = BASE_URL

resp = dashscope.Generation.call(
    model="qwen3.5-plus",
    prompt="请回复测试成功"
)

print(f"Status code: {resp.status_code}")
print(f"Response: {resp}")
if resp.status_code == HTTPStatus.OK:
    print("Test PASSED!")
    print("Model response:", resp.output.text)
else:
    print("Test FAILED")
    print("Error message:", resp.message)
    print("Request ID:", resp.request_id)
    print("Error code:", resp.code if hasattr(resp, 'code') else 'N/A')
