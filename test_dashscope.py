"""测试 DashScope OpenAI 兼容模式配置."""

import os
from pathlib import Path

import pandas as pd
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from timecopilot import TimeCopilot

# 加载 .env 文件（如果存在）
def load_dotenv():
    """从项目根目录加载 .env 文件."""
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and value:
                        os.environ[key] = value

# 加载环境变量
load_dotenv()

# 从环境变量读取配置
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
# 根据阿里云文档，base_url 可以是：
# - 新加坡地域：https://dashscope-intl.aliyuncs.com/compatible-mode/v1
# - 华北2（北京）地域：https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_BASE_URL = os.getenv(
    "OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 模型名称（根据阿里云文档，可以是 qwen-turbo, qwen-plus, qwen-long 等）
MODEL_NAME = os.getenv("DASHSCOPE_MODEL", "qwen-turbo")

print(f"配置信息:")
print(f"  API Key: {DASHSCOPE_API_KEY[:10] if DASHSCOPE_API_KEY else 'None'}...")
print(f"  Base URL: {DASHSCOPE_BASE_URL}")
print(f"  Model: {MODEL_NAME}")
print()

if not DASHSCOPE_API_KEY:
    print("❌ 错误: 未找到 DASHSCOPE_API_KEY 或 OPENAI_API_KEY 环境变量")
    print("请在 .env 文件中设置:")
    print("  DASHSCOPE_API_KEY=your-dashscope-api-key")
    print("  或")
    print("  OPENAI_API_KEY=your-dashscope-api-key")
    print("\n获取 API Key:")
    print("  1. 访问 https://dashscope.console.aliyun.com/")
    print("  2. 创建 API Key")
    print("  3. 复制 Key 到 .env 文件")
    exit(1)

# 创建 OpenAI 兼容的模型配置
print("创建 OpenAI 兼容模型配置...")
model = OpenAIChatModel(
    MODEL_NAME,
    provider=OpenAIProvider(
        base_url=DASHSCOPE_BASE_URL,
        api_key=DASHSCOPE_API_KEY,
    ),
)

# 初始化 TimeCopilot
print("初始化 TimeCopilot...")
tc = TimeCopilot(
    llm=model,
    retries=3,  # 增加重试次数
)

# 测试数据（简单的 Air Passengers 数据集）
print("\n准备测试数据...")
df = pd.read_csv("https://timecopilot.s3.amazonaws.com/public/data/air_passengers.csv")

print(f"数据形状: {df.shape}")
print(f"数据预览:\n{df.head()}")
print()

# 运行预测
print("开始运行 TimeCopilot 预测...")
print("注意: 这可能需要一些时间，因为需要调用 DashScope API...")
print()

try:
    result = tc.forecast(df=df, freq="MS", h=12)

    print("✅ 预测成功!")
    print(f"\n预测结果:")
    print(result.fcst_df.head(10))
    print(f"\n选择的模型: {result.output.selected_model}")
    print(f"模型比较: {result.output.model_comparison[:200]}...")

except Exception as e:
    print(f"❌ 预测失败: {type(e).__name__}: {e}")
    
    # 提供针对性的错误提示
    error_str = str(e).lower()
    if "401" in error_str or "invalid_api_key" in error_str or "authentication" in error_str:
        print("\n💡 这是 API Key 认证错误，请检查:")
        print("  1. API Key 是否正确（从 DashScope 控制台复制）")
        print("  2. API Key 是否已激活")
        print("  3. .env 文件中的 Key 格式是否正确（不要有多余空格）")
        print("  4. 确保使用的是 DashScope 的 API Key，不是 OpenAI 的")
    elif "429" in error_str or "quota" in error_str:
        print("\n💡 这是配额错误，请检查:")
        print("  1. DashScope 账户是否有足够的 API 调用配额")
        print("  2. 是否超过了调用频率限制")
    elif "unexpectedmodelbehavior" in error_str.lower() or "retries" in error_str.lower():
        print("\n💡 这是模型输出格式不兼容的问题:")
        print("  1. DashScope 的工具调用格式可能与 OpenAI 有细微差异")
        print("  2. 可以尝试增加重试次数或使用不同的模型")
        print("  3. 或者考虑使用本地预测模式（不依赖 LLM）")
    
    import traceback
    traceback.print_exc()

