# DashScope OpenAI 兼容模式配置指南

根据 [阿里云 DashScope 文档](https://help.aliyun.com/zh/model-studio/compatibility-of-openai-with-dashscope)，TimeCopilot 可以通过 OpenAI 兼容接口使用 DashScope 的 Qwen 模型。

## 配置步骤

### 1. 获取 DashScope API Key

1. 访问 [阿里云百炼控制台](https://dashscope.console.aliyun.com/)
2. 创建 API Key
3. 复制 API Key（格式类似：`sk-xxxxx...`）

### 2. 配置环境变量

在项目根目录的 `.env` 文件中添加：

```env
# DashScope OpenAI 兼容模式配置
# 方式 1：使用 DASHSCOPE_API_KEY
DASHSCOPE_API_KEY=你的-dashscope-api-key
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_MODEL=qwen-turbo

# 方式 2：使用 OPENAI_API_KEY（TimeCopilot 默认读取这个）
# OPENAI_API_KEY=你的-dashscope-api-key
# OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

**重要提示**：
- API Key 格式：DashScope 的 API Key 通常以 `sk-` 开头
- 确保从 [DashScope 控制台](https://dashscope.console.aliyun.com/) 获取的是正确的 API Key
- 不要使用 OpenAI 的 API Key，必须使用 DashScope 的

### 3. 地域选择

根据你的地理位置，选择合适的 base_url：

- **华北2（北京）**：`https://dashscope.aliyuncs.com/compatible-mode/v1`
- **新加坡**：`https://dashscope-intl.aliyuncs.com/compatible-mode/v1`

### 4. 可用模型

根据阿里云文档，可用的模型包括：

- `qwen-turbo` - 快速响应
- `qwen-plus` - 平衡性能
- `qwen-long` - 长文本支持

### 5. 测试配置

运行示例脚本验证配置（推荐）：

```bash
python examples/dashscope_example.py
```

或者运行测试脚本：

```bash
python test_dashscope.py
```

如果看到 "✅ 预测成功!" 的输出，说明配置正确。

## 在代码中使用

### 方式 1：使用环境变量（推荐）

```python
from timecopilot import TimeCopilot
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
import os

# 从环境变量读取
model = OpenAIChatModel(
    "qwen-turbo",
    provider=OpenAIProvider(
        base_url=os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        api_key=os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY"),
    ),
)

tc = TimeCopilot(llm=model)
```

### 方式 2：直接在代码中配置

```python
from timecopilot import TimeCopilot
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

model = OpenAIChatModel(
    "qwen-turbo",
    provider=OpenAIProvider(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key="你的-dashscope-api-key",
    ),
)

tc = TimeCopilot(llm=model)
```

## 注意事项

1. **工具调用兼容性**：DashScope 的 OpenAI 兼容模式可能在某些工具调用细节上与 OpenAI 官方 API 有细微差异。如果遇到 `UnexpectedModelBehavior` 错误，可能需要：
   - 增加重试次数：`TimeCopilot(llm=model, retries=5)`
   - 使用更稳定的模型：尝试 `qwen-plus` 或 `qwen-long`

2. **API 配额**：确保你的 DashScope 账户有足够的 API 调用配额

3. **网络连接**：确保可以访问阿里云 DashScope 的 API 端点

## 故障排查

### 错误：`ModuleNotFoundError: No module named 'pydantic_ai'`

```bash
pip install pydantic-ai
```

### 错误：`401 Unauthorized`

- 检查 API Key 是否正确
- 确认 API Key 是否已激活

### 错误：`UnexpectedModelBehavior: Exceeded maximum retries`

- 这是工具调用格式不兼容的问题
- 尝试增加重试次数
- 或考虑使用本地预测模式（不依赖 LLM）

## 参考文档

- [阿里云 DashScope OpenAI 兼容模式文档](https://help.aliyun.com/zh/model-studio/compatibility-of-openai-with-dashscope)
- [TimeCopilot LLM Providers 示例](https://timecopilot.dev/examples/llm-providers/)

