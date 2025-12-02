# TimeYYCopilot 图形化分析预测系统

基于 TimeCopilot 的加密货币多周期预测与交易模拟系统的图形化界面。

## 功能特性

### 📥 数据获取
- 从 Binance 公共 API 获取历史 K 线数据
- 支持多种交易对：BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT, ADAUSDT
- 支持多种周期：5m, 15m, 1h, 4h, 1d
- 实时 K 线图表可视化

### 🔮 价格预测
- 使用 TimeCopilot 进行多模型预测
- 自动选择最佳模型（基于交叉验证）
- 支持 DashScope (Qwen) 模型
- 可视化历史价格和预测价格
- 显示模型比较和分析报告

### 💹 策略回测
- 趋势跟随策略（基于预测趋势）
- 可配置策略参数：
  - 趋势阈值
  - 最小置信度
  - 初始资金
  - 手续费率
- 模拟交易执行（考虑手续费和滑点）
- 生成资金曲线和交易记录

### 📊 结果分析
- 性能指标计算：
  - 总收益率
  - 夏普比率
  - 最大回撤
  - 交易次数
- 详细的交易记录
- 资金曲线可视化

## 快速开始

### 1. 配置环境变量

在项目根目录的 `.env` 文件中配置：

```env
# DashScope API Key（用于 TimeCopilot）
DASHSCOPE_API_KEY=你的-dashscope-api-key
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_MODEL=qwen-turbo
```

或者使用 OpenAI API Key：

```env
OPENAI_API_KEY=你的-openai-api-key
```

### 2. 启动应用

**方式 1：使用启动脚本**
```bash
./start_crypto_ui.sh
```

**方式 2：直接运行**
```bash
streamlit run crypto_ui_app.py
```

应用会自动在浏览器中打开，默认地址：`http://localhost:8501`

### 3. 使用流程

1. **数据获取**
   - 选择交易对和周期
   - 设置获取天数
   - 点击「获取数据」按钮
   - 查看 K 线图表

2. **价格预测**
   - 设置预测步数和参数
   - 点击「开始预测」按钮
   - 等待预测完成（可能需要几分钟）
   - 查看预测结果和可视化图表

3. **策略回测**
   - 配置策略参数
   - 点击「运行回测」按钮
   - 查看回测结果和资金曲线

4. **结果分析**
   - 查看详细的性能指标
   - 分析交易记录
   - 评估策略表现

## 系统架构

```
crypto_ui_app.py (Streamlit UI)
    ├── crypto_data/binance_fetcher.py (Binance 数据获取)
    ├── crypto_adapter/binance_adapter.py (数据格式转换)
    ├── timecopilot (TimeCopilot 预测引擎)
    ├── crypto_strategy/simple_strategy.py (交易策略)
    └── crypto_backtest/simple_backtest.py (回测引擎)
```

## 技术栈

- **前端**: Streamlit
- **可视化**: Plotly
- **数据获取**: Binance Public API
- **预测引擎**: TimeCopilot (DashScope/Qwen)
- **数据处理**: Pandas

## 注意事项

1. **API 限制**
   - Binance API 有请求频率限制，请勿过于频繁请求
   - DashScope API 有调用配额限制

2. **数据准确性**
   - 本系统使用公开的历史数据
   - 预测结果仅供参考，不构成投资建议

3. **性能考虑**
   - 预测过程可能需要几分钟时间
   - 建议使用较小的数据集进行测试

4. **免责声明**
   - 本项目仅用于技术研究与模拟交易学习
   - 不构成任何投资建议
   - 加密货币交易具有极高风险

## 故障排查

### 问题：无法获取 Binance 数据
- 检查网络连接
- 确认交易对和周期参数正确
- 检查 Binance API 是否可访问

### 问题：预测失败
- 检查 API Key 是否正确配置
- 确认 `.env` 文件中的配置正确
- 查看错误信息中的详细提示

### 问题：回测结果异常
- 检查预测数据是否已生成
- 确认策略参数设置合理
- 查看交易记录中的详细信息

## 后续开发计划

- [ ] 支持更多交易策略
- [ ] 多币种组合回测
- [ ] 实时数据更新
- [ ] 更详细的性能分析
- [ ] 策略参数优化
- [ ] 导出回测报告

## 参考文档

- [TimeCopilot 文档](https://timecopilot.dev/)
- [Binance API 文档](https://binance-docs.github.io/apidocs/spot/cn/)
- [DashScope 文档](https://help.aliyun.com/zh/model-studio/compatibility-of-openai-with-dashscope)

