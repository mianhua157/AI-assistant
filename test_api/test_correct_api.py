"""
阿里云百炼 API 调用测试
使用正确的 API Key 配置
"""
import dashscope
from http import HTTPStatus

# ======== 请修改这里！ ========
# 使用从"API-KEY 管理"页面获取的 Key（sk-开头，不是 sk-sp-）
API_KEY = "sk-在此填入你的 API Key"
# ===========================

# 设置 API Key
dashscope.api_key = API_KEY

# 发起测试请求（使用 messages 格式）
response = dashscope.Generation.call(
    model="qwen3.5-plus",
    messages=[{"role": "user", "content": "请回复'测试成功'"}],
    result_format="message"
)

# 查看结果
print(f"状态码：{response.status_code}")
if response.status_code == HTTPStatus.OK:
    print("测试成功！")
    print("模型回复:", response.output.choices[0].message.content)
else:
    print("测试失败")
    print("错误信息:", response.message)
    print("错误码:", response.code)
    print("请求 ID:", response.request_id)
