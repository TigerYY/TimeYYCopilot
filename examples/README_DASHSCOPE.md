# DashScope OpenAI 兼容模式 - 使用成功 ✅

## 配置验证

根据测试结果，DashScope 的 OpenAI 兼容模式已经成功配置并可以正常使用！

### 测试结果

运行 `examples/dashscope_example.py` 后，成功完成了：
- ✅ API Key 认证通过
- ✅ 模型调用成功（qwen-turbo）
- ✅ TimeCopilot 预测流程完整执行
- ✅ 模型选择与预测结果正常输出

### 当前配置

根据 `.env` 文件：

```env
OPENAI_API_KEY=你的-dashscope-api-key
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

### 使用方法

#### 方式 1：使用示例脚本（推荐）

```bash
python examples/dashscope_example.py
```

#### 方式 2：在代码中使用

```python
from timecopilot import TimeCopilot
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
import os

# 从环境变量读取配置
model = OpenAIChatModel(
    "qwen-turbo",
    provider=OpenAIProvider(
        base_url=os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        api_key=os.getenv("OPENAI_API_KEY"),
    ),
)

tc = TimeCopilot(llm=model, retries=5)
result = tc.forecast(df=your_dataframe, freq="MS", h=12)
```

### 注意事项

1. **重试次数**：建议设置 `retries=5` 以提高兼容性
2. **模型选择**：当前使用 `qwen-turbo`，也可以尝试 `qwen-plus` 或 `qwen-long`
3. **地域选择**：当前使用北京地域，如需使用新加坡地域，修改 base_url 为：
   ```
   https://dashscope-intl.aliyuncs.com/compatible-mode/v1
   ```

### 参考文档

- [DashScope 配置指南](../DASHSCOPE_SETUP.md)
- [阿里云 DashScope 官方文档](https://help.aliyun.com/zh/model-studio/compatibility-of-openai-with-dashscope)

