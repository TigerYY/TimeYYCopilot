"""
TimeYYCopilot - 加密货币分析预测系统
基于 TimeCopilot 的图形化交易模拟系统
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from crypto_adapter.binance_adapter import BinanceKlineAdapter
from crypto_backtest.simple_backtest import SimpleBacktestEngine
from crypto_data.binance_fetcher import BinanceDataFetcher
from crypto_strategy.simple_strategy import TrendFollowingStrategy
from timecopilot import TimeCopilot


def load_env_file():
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


def create_dashscope_model():
    """创建 DashScope OpenAI 兼容模型配置."""
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv(
        "OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    model_name = os.getenv("DASHSCOPE_MODEL", "qwen-turbo")

    if not api_key:
        return None

    return OpenAIChatModel(
        model_name,
        provider=OpenAIProvider(
            base_url=base_url,
            api_key=api_key,
        ),
    )


# 加载环境变量
load_env_file()

# 页面配置
st.set_page_config(
    page_title="TimeYYCopilot - 加密货币分析预测系统",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 TimeYYCopilot")
st.markdown("**基于 TimeCopilot 的加密货币多周期预测与交易模拟系统**")

# 侧边栏配置
with st.sidebar:
    st.header("⚙️ 系统配置")

    # API Key 检查
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if api_key:
        st.success(f"✅ API Key 已配置")
        st.caption(f"Key: {api_key[:10]}...{api_key[-4:] if len(api_key) > 14 else ''}")
    else:
        st.error("❌ 未找到 API Key")
        st.info("请在 .env 文件中设置 DASHSCOPE_API_KEY 或 OPENAI_API_KEY")

    base_url = os.getenv(
        "OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    model_name = os.getenv("DASHSCOPE_MODEL", "qwen-turbo")
    st.info(f"**Base URL:** {base_url}\n**Model:** {model_name}")

    st.divider()
    st.markdown("### 📊 数据源")
    st.caption("使用 Binance 公共 API 获取历史 K 线数据")

# 主界面标签页
tab1, tab2, tab3, tab4 = st.tabs(
    ["📥 数据获取", "🔮 价格预测", "💹 策略回测", "📊 结果分析"]
)

# ========== 标签页 1: 数据获取 ==========
with tab1:
    st.header("📥 Binance K线数据获取")

    col1, col2, col3 = st.columns(3)

    with col1:
        symbol = st.selectbox(
            "交易对",
            ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT"],
            index=0,
        )

    with col2:
        interval = st.selectbox(
            "K线周期",
            ["5m", "15m", "1h", "4h", "1d"],
            index=1,
        )

    with col3:
        days_back = st.number_input("获取最近天数", min_value=1, max_value=365, value=30)

    if st.button("📥 获取数据", type="primary", use_container_width=True):
        with st.spinner("正在从 Binance 获取数据..."):
            try:
                fetcher = BinanceDataFetcher()
                end_time = datetime.now()
                start_time = end_time - timedelta(days=days_back)

                kline_data = fetcher.fetch_klines(
                    symbol=symbol,
                    interval=interval,
                    start_time=start_time,
                    end_time=end_time,
                )

                if not kline_data.empty:
                    st.session_state.kline_data = kline_data
                    st.session_state.symbol = symbol
                    st.session_state.interval = interval
                    st.success(f"✅ 成功获取 {len(kline_data)} 条 K 线数据")

                    # 显示数据预览
                    st.subheader("📋 数据预览")
                    st.dataframe(kline_data.head(10), use_container_width=True)

                    # 显示 K 线图
                    st.subheader("📈 K 线图")
                    fig = go.Figure(
                        data=[
                            go.Candlestick(
                                x=kline_data["open_time"],
                                open=kline_data["open"],
                                high=kline_data["high"],
                                low=kline_data["low"],
                                close=kline_data["close"],
                            )
                        ]
                    )
                    fig.update_layout(
                        title=f"{symbol} {interval} K线图",
                        xaxis_title="时间",
                        yaxis_title="价格 (USDT)",
                        height=500,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                else:
                    st.error("❌ 未获取到数据，请检查网络连接和参数设置")

            except Exception as e:
                st.error(f"❌ 获取数据失败: {e}")
                import traceback

                with st.expander("查看详细错误"):
                    st.code(traceback.format_exc())

# ========== 标签页 2: 价格预测 ==========
with tab2:
    st.header("🔮 TimeCopilot 价格预测")

    if "kline_data" not in st.session_state or st.session_state.kline_data.empty:
        st.warning("⚠️ 请先在「数据获取」标签页获取 K 线数据")
    else:
        kline_data = st.session_state.kline_data
        symbol = st.session_state.symbol
        interval = st.session_state.interval

        st.info(f"当前数据: {symbol} | {interval} | {len(kline_data)} 条记录")

        col1, col2, col3 = st.columns(3)

        with col1:
            forecast_horizon = st.number_input(
                "预测步数 (h)", min_value=1, max_value=100, value=12
            )

        with col2:
            retries = st.number_input("重试次数", min_value=1, max_value=10, value=5)

        with col3:
            price_type = st.selectbox(
                "预测价格类型", ["close", "open", "high", "low"], index=0
            )

        if st.button("🚀 开始预测", type="primary", use_container_width=True):
            if not api_key:
                st.error("❌ 请先配置 API Key！")
            else:
                with st.spinner("正在运行 TimeCopilot 预测，这可能需要一些时间..."):
                    try:
                        # 转换数据格式
                        adapter = BinanceKlineAdapter()
                        tc_data = adapter.to_timecopilot_format(
                            kline_data, symbol, price_type
                        )
                        freq = adapter.get_freq(interval)

                        # 创建模型
                        model = create_dashscope_model()
                        if model is None:
                            st.error("❌ 无法创建模型，请检查 API Key 配置")
                        else:
                            # 初始化 TimeCopilot
                            tc = TimeCopilot(llm=model, retries=retries)

                            # 运行预测
                            result = tc.forecast(df=tc_data, freq=freq, h=forecast_horizon)

                            # 保存预测结果
                            st.session_state.forecast_result = result
                            st.session_state.forecast_data = result.fcst_df

                            st.success("✅ 预测完成！")

                            # 显示预测结果
                            st.subheader("📊 预测结果")
                            st.dataframe(result.fcst_df, use_container_width=True)

                            # 可视化预测
                            st.subheader("📈 预测可视化")

                            # 合并历史数据和预测数据
                            historical_prices = kline_data[price_type].values
                            historical_times = kline_data["open_time"].values

                            # 获取预测列名（模型名称，如 'AutoARIMA'）
                            # fcst_df 包含 unique_id, ds, 和模型名称列
                            forecast_cols = [
                                col
                                for col in result.fcst_df.columns
                                if col not in ["unique_id", "ds"]
                            ]
                            
                            forecast_col = None
                            forecast_prices = []
                            forecast_times = []
                            
                            if not forecast_cols:
                                st.warning("⚠️ 预测结果中没有找到预测值列")
                            else:
                                # 使用第一个模型列（通常是选择的模型）
                                forecast_col = forecast_cols[0]
                                forecast_prices = result.fcst_df[forecast_col].values
                                forecast_times = pd.to_datetime(
                                    result.fcst_df["ds"]
                                ).values

                            fig = go.Figure()

                            # 历史数据
                            fig.add_trace(
                                go.Scatter(
                                    x=historical_times,
                                    y=historical_prices,
                                    mode="lines",
                                    name="历史价格",
                                    line=dict(color="blue", width=2),
                                )
                            )

                            # 预测数据
                            if len(forecast_prices) > 0 and forecast_col:
                                fig.add_trace(
                                    go.Scatter(
                                        x=forecast_times,
                                        y=forecast_prices,
                                        mode="lines+markers",
                                        name=f"预测价格 ({forecast_col})",
                                        line=dict(color="red", width=2, dash="dash"),
                                        marker=dict(size=6),
                                    )
                                )

                                # 连接点
                                if len(historical_times) > 0 and len(forecast_times) > 0:
                                    fig.add_trace(
                                        go.Scatter(
                                            x=[historical_times[-1], forecast_times[0]],
                                            y=[historical_prices[-1], forecast_prices[0]],
                                            mode="lines",
                                            name="连接",
                                            line=dict(color="gray", width=1, dash="dot"),
                                            showlegend=False,
                                        )
                                    )

                            fig.update_layout(
                                title=f"{symbol} 价格预测 ({interval})",
                                xaxis_title="时间",
                                yaxis_title="价格 (USDT)",
                                height=600,
                                hovermode="x unified",
                            )

                            st.plotly_chart(fig, use_container_width=True)

                            # 模型信息
                            st.subheader("🤖 模型信息")
                            col1, col2 = st.columns(2)

                            with col1:
                                st.metric(
                                    "选择的模型", result.output.selected_model
                                )
                                st.metric(
                                    "优于基线",
                                    "✅ 是"
                                    if result.output.is_better_than_seasonal_naive
                                    else "❌ 否",
                                )

                            with col2:
                                # 显示交叉验证结果（如果有 eval_df）
                                if hasattr(result, "eval_df") and result.eval_df is not None:
                                    st.write("**交叉验证结果 (MASE):**")
                                    eval_df = result.eval_df
                                    # eval_df 包含 metric 列和各个模型的列
                                    if not eval_df.empty:
                                        for col in eval_df.columns:
                                            if col != "metric":
                                                mase_score = eval_df[col].iloc[0] if len(eval_df) > 0 else None
                                                if pd.notna(mase_score):
                                                    st.write(f"- {col}: {float(mase_score):.4f}")
                                else:
                                    # 如果没有 eval_df，从 model_comparison 中提取信息
                                    if result.output.model_comparison:
                                        st.write("**模型比较:**")
                                        st.caption("详见下方模型比较分析")

                            # 详细分析
                            if result.output.model_comparison:
                                with st.expander("🔍 模型比较分析"):
                                    st.write(result.output.model_comparison)

                            if result.output.forecast_analysis:
                                with st.expander("📉 预测分析"):
                                    st.write(result.output.forecast_analysis)

                    except Exception as e:
                        st.error(f"❌ 预测失败: {type(e).__name__}: {e}")
                        import traceback

                        with st.expander("查看详细错误"):
                            st.code(traceback.format_exc())

# ========== 标签页 3: 策略回测 ==========
with tab3:
    st.header("💹 交易策略回测")

    if (
        "kline_data" not in st.session_state
        or st.session_state.kline_data.empty
        or "forecast_data" not in st.session_state
    ):
        st.warning("⚠️ 请先完成数据获取和价格预测")
    else:
        kline_data = st.session_state.kline_data
        forecast_data = st.session_state.forecast_data
        symbol = st.session_state.symbol

        st.subheader("策略参数配置")

        col1, col2, col3 = st.columns(3)

        with col1:
            trend_threshold = st.slider(
                "趋势阈值 (%)", min_value=0.0, max_value=10.0, value=2.0, step=0.1
            ) / 100

        with col2:
            min_confidence = st.slider(
                "最小置信度", min_value=0.0, max_value=1.0, value=0.6, step=0.05
            )

        with col3:
            initial_capital = st.number_input(
                "初始资金 (USDT)", min_value=100.0, max_value=1000000.0, value=10000.0
            )

        fee_rate = st.slider(
            "手续费率 (%)", min_value=0.0, max_value=1.0, value=0.1, step=0.01
        ) / 100

        if st.button("▶️ 运行回测", type="primary", use_container_width=True):
            with st.spinner("正在运行回测..."):
                try:
                    # 创建策略
                    strategy = TrendFollowingStrategy(
                        trend_threshold=trend_threshold,
                        min_confidence=min_confidence,
                    )

                    # 创建回测引擎
                    backtest_engine = SimpleBacktestEngine(
                        strategy=strategy,
                        initial_capital=initial_capital,
                        fee_rate=fee_rate,
                    )

                    # 运行回测
                    backtest_result = backtest_engine.run(
                        historical_data=kline_data,
                        forecast_data=forecast_data,
                        price_column="close",
                    )

                    # 保存结果
                    st.session_state.backtest_result = backtest_result

                    st.success("✅ 回测完成！")

                    # 显示回测结果
                    st.subheader("📊 回测结果")

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("初始资金", f"${initial_capital:,.2f}")

                    with col2:
                        st.metric("最终资金", f"${backtest_result.final_capital:,.2f}")

                    with col3:
                        total_return_pct = backtest_result.total_return * 100
                        st.metric(
                            "总收益率",
                            f"{total_return_pct:.2f}%",
                            delta=f"{backtest_result.final_capital - initial_capital:,.2f}",
                        )

                    with col4:
                        st.metric("交易次数", len(backtest_result.trades))

                    # 资金曲线
                    st.subheader("📈 资金曲线")
                    equity_df = backtest_result.equity_curve

                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=equity_df["timestamp"],
                            y=equity_df["total_value"],
                            mode="lines",
                            name="账户总值",
                            line=dict(color="green", width=2),
                        )
                    )
                    fig.add_hline(
                        y=initial_capital,
                        line_dash="dash",
                        line_color="gray",
                        annotation_text="初始资金",
                    )

                    fig.update_layout(
                        title="资金曲线",
                        xaxis_title="时间",
                        yaxis_title="账户总值 (USDT)",
                        height=500,
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # 交易记录
                    if backtest_result.trades:
                        st.subheader("📋 交易记录")
                        trades_df = pd.DataFrame(
                            [
                                {
                                    "时间": trade.timestamp,
                                    "操作": trade.action,
                                    "价格": trade.price,
                                    "数量": trade.quantity,
                                    "金额": trade.value,
                                    "手续费": trade.fee,
                                    "余额": trade.balance,
                                }
                                for trade in backtest_result.trades
                            ]
                        )
                        st.dataframe(trades_df, use_container_width=True)

                except Exception as e:
                    st.error(f"❌ 回测失败: {type(e).__name__}: {e}")
                    import traceback

                    with st.expander("查看详细错误"):
                        st.code(traceback.format_exc())

# ========== 标签页 4: 结果分析 ==========
with tab4:
    st.header("📊 综合分析")

    if "backtest_result" not in st.session_state:
        st.warning("⚠️ 请先完成策略回测")
    else:
        backtest_result = st.session_state.backtest_result

        st.subheader("📈 性能指标")

        # 计算更多指标
        equity_curve = backtest_result.equity_curve
        returns = equity_curve["total_value"].pct_change().dropna()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_return = backtest_result.total_return * 100
            st.metric("总收益率", f"{total_return:.2f}%")

        with col2:
            if len(returns) > 0:
                sharpe_ratio = (
                    returns.mean() / returns.std() * (252**0.5)
                    if returns.std() > 0
                    else 0
                )
                st.metric("夏普比率", f"{sharpe_ratio:.2f}")

        with col3:
            max_value = equity_curve["total_value"].max()
            min_value_after_max = equity_curve.loc[
                equity_curve["total_value"].idxmax() :, "total_value"
            ].min()
            max_drawdown = (min_value_after_max - max_value) / max_value * 100
            st.metric("最大回撤", f"{max_drawdown:.2f}%")

        with col4:
            win_rate = (
                len([t for t in backtest_result.trades if t.action == "SELL"])
                / len(backtest_result.trades)
                * 100
                if backtest_result.trades
                else 0
            )
            st.metric("交易次数", len(backtest_result.trades))

        st.divider()

        st.subheader("📝 使用说明")
        st.markdown(
            """
        ### 功能说明

        1. **数据获取**
           - 从 Binance 公共 API 获取历史 K 线数据
           - 支持多种交易对和周期（5m, 15m, 1h, 4h, 1d）

        2. **价格预测**
           - 使用 TimeCopilot 进行多模型预测
           - 自动选择最佳模型
           - 可视化历史价格和预测价格

        3. **策略回测**
           - 基于预测结果生成交易信号
           - 模拟交易执行（考虑手续费）
           - 计算资金曲线和性能指标

        4. **结果分析**
           - 查看详细的回测指标
           - 分析交易记录
           - 评估策略表现

        ### 注意事项

        - 本项目仅用于技术研究与模拟交易学习
        - 不构成任何投资建议
        - 加密货币交易具有极高风险
        """
        )

# 页脚
st.divider()
st.caption(
    "⚠️ 免责声明：本项目仅用于技术研究与模拟交易学习，不构成任何投资建议。加密货币交易具有极高风险，可能导致本金全部损失。"
)

