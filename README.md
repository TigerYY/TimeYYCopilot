<div align="center">
  <h1>TimeYYCopilot · Crypto Trading Simulator</h1>
  <p><em>基于 TimeCopilot 的加密货币多周期预测与交易模拟系统</em></p>
</div>

---

## 项目简介

**TimeYYCopilot** 是在开源项目 [TimeCopilot](https://github.com/TimeCopilot/timecopilot) 之上构建的一个实践项目，
目标是为 **BTC/ETH 等加密货币** 提供：

- 多周期（5m / 15m / 1h / 4h）K 线的 **价格与波动预测**  
- 基于预测结果的 **策略生成与资金曲线回测**  
- 面向未来的 **半自动 / 自动实盘下单架构雏形**

本仓库更多是「应用工程层」，复用 TimeCopilot 的智能体与模型能力，聚焦在：

- 对接 Binance 行情数据
- 将加密货币 K 线数据转换成 TimeCopilot 兼容格式
- 设计策略引擎与模拟交易 / 回测框架

> 注意：当前阶段仅用于 **个人研究与模拟交易**，不直接面向真实资金的自动化交易。

---

## 核心能力

- **多周期加密货币预测**
  - 支持 5m / 15m / 1h / 4h 的 BTCUSDT / ETHUSDT 等主流交易对
  - 利用 TimeCopilot 统一调用传统统计模型 + ML + 神经网络 + TSFM
  - 自动进行模型选择与交叉验证，并输出可解释的分析报告

- **交易策略 & 回测**
  - 基于未来数根 K 线的预测轨迹，生成 BUY/SELL/HOLD 信号
  - 支持手续费、滑点、仓位控制等基础要素
  - 可以在历史区间上做完整回测，输出资金曲线与关键指标

- **面向实盘的架构设计**
  - 数据源、预测服务、策略引擎、执行引擎分层解耦
  - 当前执行引擎仅实现“模拟成交”，未来可替换为 Binance 实盘下单

---

## 项目结构（规划中）

本仓库基于上游 `timecopilot` 代码结构，并将逐步增加与加密货币交易相关的模块，例如：

- `crypto_data/`：Binance 数据拉取与本地存储（REST + WebSocket）
- `crypto_adapter/`：将 K 线数据转换为 `unique_id, ds, y` 形式供 TimeCopilot 使用
- `crypto_strategy/`：策略实现（趋势跟随、均值回复、波动过滤等）
- `crypto_backtest/`：回测与模拟交易引擎
- `examples/`：Jupyter Notebook 示例（单币种 / 多周期组合回测）

> 目前这些目录会在后续迭代中逐步补齐，具体以仓库实际代码为准。

---

## 开发环境与前置依赖

- 硬件环境（推荐）：
  - Apple Silicon（如 **MacBook Pro M4 Max, 36GB RAM**）或同级别 Linux 主机
- 软件环境：
  - Python 3.10–3.13
  - Git
  - 推荐使用 [uv](https://docs.astral.sh/uv/) 或 `venv` 创建虚拟环境

### 必要的 API Key

- **OpenAI / 其他 LLM 提供商**（用于 TimeCopilot 智能体）
  - 环境变量：`OPENAI_API_KEY` 等
- **Binance API Key（未来接入实盘/实时数据时使用，可选）**
  - 环境变量示例：`BINANCE_API_KEY`, `BINANCE_API_SECRET`
  - 模拟回测阶段可以仅使用公共 K 线历史数据

---

## 快速开始（规划）

> 下述命令展示的是目标形态，具体脚本与模块名称将在实现过程中逐步补充。

### 1. 克隆仓库并安装依赖

```bash
git clone https://github.com/TigerYY/TimeYYCopilot.git
cd TimeYYCopilot

# 推荐使用 uv
uv sync  # 或者使用: pip install -e .
```

### 2. 配置环境变量

```bash
export OPENAI_API_KEY="your-openai-key"
# 可选：未来接入 Binance 实盘/实时数据时需要
export BINANCE_API_KEY="your-binance-key"
export BINANCE_API_SECRET="your-binance-secret"
```

### 3. 运行示例（回测）

计划提供类似以下脚本 / Notebook（示意）：

```bash
python -m crypto_backtest.run_btc_eth_example \
  --symbol BTCUSDT \
  --interval 15m \
  --start 2024-01-01 \
  --end 2024-03-01
```

脚本会完成：

- 从本地或 Binance 拉取历史 K 线
- 转换为 TimeCopilot 接受的时间序列格式
- 运行 TimeCopilot 做多模型预测与模型选择
- 根据策略规则生成交易信号并模拟成交
- 输出资金曲线和关键指标（年化收益、最大回撤等）

---

## 与上游 TimeCopilot 的关系

- 本仓库是 **个人实验项目**，基于上游项目：
  - 源项目地址：<https://github.com/TimeCopilot/timecopilot>
  - 如果你只关心「通用时间序列 + LLM 预测智能体」，建议直接使用上游库。
- **TimeYYCopilot** 在此基础上：
  - 聚焦「加密货币交易」这一具体场景；
  - 增加数据接入、交易策略、回测与（未来的）下单执行层。

上游项目的详细说明可参考本仓库中的 `README.TC.md`。

---

## Roadmap（计划）

1. **数据层**
   - [ ] Binance 历史 K 线拉取脚本（多周期）
   - [ ] 本地数据落地（Parquet / SQLite / DuckDB）
2. **适配层**
   - [ ] 将 K 线数据标准化为 TimeCopilot 所需的 `unique_id, ds, y`
   - [ ] 多资产 / 多周期统一管理
3. **预测层**
   - [ ] 基于 TimeCopilot 的多模型预测（5m / 15m / 1h / 4h）
   - [ ] 结果缓存与可视化（价格 + 预测曲线）
4. **策略 & 回测层**
   - [ ] 简单趋势策略回测（单币种）
   - [ ] 多币种 / 多周期组合策略
   - [ ] 资金曲线与指标报表
5. **执行层**
   - [ ] 半自动下单（生成“建议单”，人工确认）
   - [ ] 自动化下单的风险控制与限额机制

---

## 免责声明

- 本项目仅用于 **技术研究与模拟交易学习**，不构成任何投资建议。
- 加密货币/衍生品交易具有极高风险，可能导致本金全部损失；请勿将本项目或其输出用于缺乏充分风控与合规检查的真实资金交易。


