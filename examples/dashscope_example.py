"""
DashScope OpenAI 兼容模式使用示例

根据阿里云文档：https://help.aliyun.com/zh/model-studio/compatibility-of-openai-with-dashscope
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from timecopilot import TimeCopilot


def load_env_file():
    """从项目根目录加载 .env 文件."""
    env_file = Path(__file__).parent.parent / ".env"
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


def create_dashscope_model(
    api_key: str | None = None,
    base_url: str | None = None,
    model_name: str = "qwen-turbo",
):
    """
    创建 DashScope OpenAI 兼容模型配置.

    Args:
        api_key: DashScope API Key，如果为 None 则从环境变量读取
        base_url: DashScope base URL，如果为 None 则使用默认值
        model_name: 模型名称，默认 qwen-turbo

    Returns:
        OpenAIChatModel 实例
    """
    # 从环境变量读取配置
    api_key = api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = base_url or os.getenv(
        "OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    if not api_key:
        raise ValueError(
            "未找到 API Key。请设置 DASHSCOPE_API_KEY 或 OPENAI_API_KEY 环境变量，"
            "或在调用时传入 api_key 参数。"
        )

    print(f"配置 DashScope 模型:")
    print(f"  Model: {model_name}")
    print(f"  Base URL: {base_url}")
    print(f"  API Key: {api_key[:10]}...{api_key[-4:] if len(api_key) > 14 else ''}")
    print()

    return OpenAIChatModel(
        model_name,
        provider=OpenAIProvider(
            base_url=base_url,
            api_key=api_key,
        ),
    )


def main():
    """主函数：演示如何使用 DashScope 配置 TimeCopilot."""
    # 加载 .env 文件
    load_env_file()

    # 创建 DashScope 模型
    try:
        model = create_dashscope_model(model_name="qwen-turbo")
    except ValueError as e:
        print(f"❌ 配置错误: {e}")
        print("\n请确保:")
        print("  1. 在 .env 文件中设置了 DASHSCOPE_API_KEY 或 OPENAI_API_KEY")
        print("  2. API Key 是从 DashScope 控制台获取的有效 Key")
        return

    # 初始化 TimeCopilot（增加重试次数以提高兼容性）
    print("初始化 TimeCopilot...")
    tc = TimeCopilot(
        llm=model,
        retries=5,  # 增加重试次数，因为 DashScope 可能需要更多容错
    )

    # 准备测试数据
    print("\n准备测试数据...")
    try:
        df = pd.read_csv("https://timecopilot.s3.amazonaws.com/public/data/air_passengers.csv")
        print(f"✅ 数据加载成功: {df.shape[0]} 行, {df.shape[1]} 列")
        print(f"数据预览:\n{df.head()}")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return

    # 运行预测
    print("\n" + "=" * 60)
    print("开始运行 TimeCopilot 预测...")
    print("注意: 这可能需要一些时间，因为需要调用 DashScope API")
    print("=" * 60)
    print()

    try:
        result = tc.forecast(df=df, freq="MS", h=12)

        print("\n" + "=" * 60)
        print("✅ 预测成功!")
        print("=" * 60)
        print(f"\n预测结果 (前 10 行):")
        print(result.fcst_df.head(10))
        print(f"\n选择的模型: {result.output.selected_model}")
        if result.output.model_comparison:
            print(f"\n模型比较 (前 200 字符):")
            print(result.output.model_comparison[:200] + "...")

    except Exception as e:
        print(f"\n❌ 预测失败: {type(e).__name__}: {e}")

        # 提供针对性的错误提示
        error_str = str(e).lower()
        if "401" in error_str or "invalid_api_key" in error_str or "authentication" in error_str:
            print("\n💡 这是 API Key 认证错误，请检查:")
            print("  1. API Key 是否正确（从 DashScope 控制台复制）")
            print("  2. API Key 是否已激活")
            print("  3. .env 文件中的 Key 格式是否正确（不要有多余空格）")
            print("  4. 确保使用的是 DashScope 的 API Key，不是 OpenAI 的")
            print("\n获取 DashScope API Key:")
            print("  https://dashscope.console.aliyun.com/")
        elif "429" in error_str or "quota" in error_str:
            print("\n💡 这是配额错误，请检查:")
            print("  1. DashScope 账户是否有足够的 API 调用配额")
            print("  2. 是否超过了调用频率限制")
        elif "unexpectedmodelbehavior" in error_str.lower() or "retries" in error_str.lower():
            print("\n💡 这是模型输出格式不兼容的问题:")
            print("  1. DashScope 的工具调用格式可能与 OpenAI 有细微差异")
            print("  2. 可以尝试:")
            print("     - 增加重试次数（已在代码中设置为 5）")
            print("     - 使用不同的模型（qwen-plus 或 qwen-long）")
            print("     - 使用本地预测模式（不依赖 LLM）")

        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

