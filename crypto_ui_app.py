"""
TimeYYCopilot - 加密货币分析预测系统
基于 TimeCopilot 的图形化交易模拟系统
"""

import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 抑制一些不必要的警告（保留重要错误信息）
warnings.filterwarnings("ignore", category=UserWarning, module="streamlit")
# 抑制 cmdstanpy 的 INFO 级别日志（保留 WARNING 和 ERROR）
import logging
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

import pandas as pd
import numpy as np
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
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["📥 数据获取", "🔮 价格预测", "💹 策略回测", "📊 结果分析", "⏱️ 实时回预测", "🔄 实时预测"]
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
        days_back = st.number_input(
            "获取最近天数", 
            min_value=1, 
            max_value=1000, 
            value=30,
            help="最多可获取 1000 天的历史数据。注意：数据量越大，获取和预测所需时间越长。"
        )

    if st.button("📥 获取数据", type="primary", use_container_width=True):
        # 估算数据量并显示提示
        estimated_records = days_back
        if interval == "5m":
            estimated_records = days_back * 288  # 每天 288 个 5 分钟 K 线
        elif interval == "15m":
            estimated_records = days_back * 96  # 每天 96 个 15 分钟 K 线
        elif interval == "1h":
            estimated_records = days_back * 24  # 每天 24 个 1 小时 K 线
        elif interval == "4h":
            estimated_records = days_back * 6   # 每天 6 个 4 小时 K 线
        # 1d 就是 days_back 本身
        
        if estimated_records > 5000:
            st.info(f"ℹ️ 预计将获取约 {estimated_records:,} 条数据，这可能需要 10-30 秒，请耐心等待...")
        
        with st.spinner(f"正在从 Binance 获取数据（预计 {estimated_records:,} 条）..."):
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
                    st.dataframe(kline_data.head(10), width='stretch')

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
                    st.plotly_chart(fig, width='stretch')

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
                # 创建进度容器
                progress_container = st.container()
                with progress_container:
                    status_text = st.empty()
                    progress_bar = st.progress(0)
                
                try:
                    status_text.info("🔄 步骤 1/5: 准备数据...")
                    progress_bar.progress(10)
                    
                    # 验证输入数据
                    if kline_data.empty:
                        raise ValueError("K线数据为空，请先获取数据")
                    
                    if price_type not in kline_data.columns:
                        raise ValueError(f"价格列 '{price_type}' 不存在于数据中")
                    
                    # 转换数据格式
                    adapter = BinanceKlineAdapter()
                    tc_data = adapter.to_timecopilot_format(
                        kline_data, symbol, price_type
                    )
                    
                    # 验证转换后的数据
                    if tc_data.empty:
                        raise ValueError("数据转换后为空，请检查数据格式")
                    
                    if "y" not in tc_data.columns or tc_data["y"].isna().all():
                        raise ValueError("价格数据无效，所有值都是 NaN")
                    
                    # 如果数据量太大，进行采样（保留最近的数据）
                    max_data_points = 1000  # 限制最大数据点数
                    original_length = len(tc_data)
                    if len(tc_data) > max_data_points:
                        st.warning(f"⚠️ 数据量较大（{len(tc_data)} 条），将使用最近 {max_data_points} 条数据进行预测以提高速度")
                        tc_data = tc_data.tail(max_data_points).reset_index(drop=True)
                    
                    # 检查最小数据量要求
                    min_data_points = max(20, forecast_horizon * 2)  # 至少需要预测步数的 2 倍
                    if len(tc_data) < min_data_points:
                        raise ValueError(
                            f"数据量不足（{len(tc_data)} 条），至少需要 {min_data_points} 条数据。"
                            f"请增加历史数据获取天数。"
                        )
                    
                    freq = adapter.get_freq(interval)

                    # 根据周期和数据长度动态映射 seasonality
                    # seasonality 的单位是 freq 对应的 period 数
                    data_length = len(tc_data)
                    seasonality_map = {
                        "5m": 288,   # 24*60/5, 日内季节
                        "15m": 96,   # 24*60/15
                        "1h": 24,    # 24 小时
                        "4h": 6,     # 24/4
                        "1d": 7,     # 简单按一周 7 天
                    }
                    base_seasonality = seasonality_map.get(interval)
                    
                    # 如果数据长度不足，降低 seasonality 以避免过度拟合
                    if base_seasonality and data_length < base_seasonality * 2:
                        # 数据不足时，使用更小的 seasonality 或设为 None
                        if data_length < base_seasonality:
                            seasonality = None  # 数据太短，不设置季节性
                        else:
                            # 使用数据长度的一半作为 seasonality 上限
                            seasonality = min(base_seasonality, data_length // 2)
                    else:
                        seasonality = base_seasonality
                    
                    # 计算历史数据的基本统计信息（用于诊断）
                    historical_values = tc_data["y"].values
                    price_change_pct = ((historical_values[-1] - historical_values[0]) / historical_values[0]) * 100
                    price_std = pd.Series(historical_values).std()
                    price_mean = pd.Series(historical_values).mean()
                    volatility_pct = (price_std / price_mean) * 100 if price_mean > 0 else 0
                    
                    # 简单的趋势检测：计算最近 N 个点的斜率
                    # 使用更大的窗口来检测近期趋势（至少 5% 的数据，最多 100 个点）
                    recent_window = max(10, min(100, len(historical_values) // 20))
                    if recent_window >= 2:
                        recent_prices = historical_values[-recent_window:]
                        x_recent = np.arange(len(recent_prices))
                        slope, intercept = np.polyfit(x_recent, recent_prices, 1)
                        # 计算趋势百分比：斜率相对于起始价格的百分比
                        recent_trend_pct = (slope * len(recent_prices) / recent_prices[0]) * 100 if recent_prices[0] > 0 else 0
                        
                        # 计算趋势的统计显著性（R²）
                        y_pred = slope * x_recent + intercept
                        ss_res = np.sum((recent_prices - y_pred) ** 2)
                        ss_tot = np.sum((recent_prices - np.mean(recent_prices)) ** 2)
                        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    else:
                        recent_trend_pct = 0
                        r_squared = 0
                    
                    # 计算中期趋势（中间 1/3 的数据）
                    mid_start = len(historical_values) // 3
                    mid_end = len(historical_values) * 2 // 3
                    if mid_end > mid_start + 10:
                        mid_prices = historical_values[mid_start:mid_end]
                        mid_trend_pct = ((mid_prices[-1] - mid_prices[0]) / mid_prices[0]) * 100 if mid_prices[0] > 0 else 0
                    else:
                        mid_trend_pct = 0

                    status_text.info("🔄 步骤 2/5: 创建模型...")
                    progress_bar.progress(30)
                    
                    # 创建模型
                    model = create_dashscope_model()
                    if model is None:
                        st.error("❌ 无法创建模型，请检查 API Key 配置")
                    else:
                        status_text.info("🔄 步骤 3/5: 初始化 TimeCopilot...")
                        progress_bar.progress(40)
                        
                        # 初始化 TimeCopilot
                        tc = TimeCopilot(llm=model, retries=retries)

                        # 定义面向交易的中文查询，引导输出简洁的趋势结论
                        query_text = (
                            "你是一名加密货币量化交易分析师，"
                            "根据历史价格和未来预测结果，判断在本次预测区间内价格整体是上涨、下跌还是震荡/横盘。"
                            "请用简短的中文给出结论，可以提到趋势强弱和大致风险提示，"
                            "不要解释模型原理，也不要输出代码。"
                        )

                        status_text.info("🔄 步骤 4/5: 运行预测（这可能需要 1-5 分钟，请耐心等待）...")
                        progress_bar.progress(50)
                        
                        # 显示提示信息
                        with st.expander("ℹ️ 预测过程说明", expanded=False):
                            st.markdown("""
                            **TimeCopilot 预测包含以下步骤：**
                            1. 📊 时间序列特征分析（识别趋势、季节性等）
                            2. 🔍 模型选择和交叉验证（比较多个模型性能）
                            3. 🎯 最终模型选择和预测
                            4. 🚨 异常检测
                            5. 📝 生成分析报告
                            
                            **预计时间：** 1-5 分钟（取决于数据量和模型复杂度）
                            """)
                        
                        # 运行预测（允许系统自动选择最优模型）
                        # 直接运行，Streamlit 会显示 spinner
                        result = tc.forecast(
                            df=tc_data,
                            freq=freq,
                            h=forecast_horizon,
                            seasonality=seasonality,
                            query=query_text,
                        )
                        
                        # 检查结果
                        if result is None:
                            raise Exception("预测未返回结果，请检查数据格式和参数设置")
                        
                        if not hasattr(result, 'fcst_df') or result.fcst_df.empty:
                            raise Exception("预测结果为空，可能是数据格式问题或模型选择失败")
                        
                        status_text.info("🔄 步骤 5/5: 处理结果...")
                        progress_bar.progress(95)

                        # 保存预测结果
                        st.session_state.forecast_result = result
                        st.session_state.forecast_data = result.fcst_df

                        progress_bar.progress(100)
                        status_text.empty()  # 清除状态文本
                        progress_bar.empty()  # 清除进度条
                        
                        st.success("✅ 预测完成！")

                        # 数据质量诊断信息
                        with st.expander("📊 数据质量诊断", expanded=True):
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("数据点数", len(tc_data))
                            with col2:
                                st.metric("整体涨跌", f"{price_change_pct:.2f}%", 
                                         delta="历史整体" if abs(price_change_pct) > 1 else None)
                            with col3:
                                st.metric("波动率", f"{volatility_pct:.2f}%")
                            with col4:
                                st.metric("近期趋势", f"{recent_trend_pct:.2f}%",
                                         delta=f"最近 {recent_window} 个点" if recent_window > 0 else None)
                            
                            st.divider()
                            
                            # 趋势分析（区分整体和近期）
                            trend_threshold = 0.5  # 趋势判断阈值（%）
                            
                            # 整体趋势判断
                            if abs(price_change_pct) < trend_threshold:
                                overall_trend = "横盘"
                                overall_icon = "🟡"
                            elif price_change_pct > 0:
                                overall_trend = "上涨"
                                overall_icon = "🟢"
                            else:
                                overall_trend = "下跌"
                                overall_icon = "🔴"
                            
                            # 近期趋势判断
                            if abs(recent_trend_pct) < trend_threshold:
                                recent_trend = "横盘"
                                recent_icon = "🟡"
                            elif recent_trend_pct > 0:
                                recent_trend = "上涨"
                                recent_icon = "🟢"
                            else:
                                recent_trend = "下跌"
                                recent_icon = "🔴"
                            
                            # 趋势一致性分析
                            trend_consistent = (overall_trend == recent_trend)
                            
                            st.markdown("### 📈 趋势分析")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(f"**整体趋势（全部数据）：** {overall_icon} {overall_trend}")
                                st.caption(f"从 {historical_values[0]:.2f} 到 {historical_values[-1]:.2f}，变化 {price_change_pct:.2f}%")
                            
                            with col2:
                                st.markdown(f"**近期趋势（最近 {recent_window} 个点）：** {recent_icon} {recent_trend}")
                                st.caption(f"线性拟合斜率：{recent_trend_pct:.2f}%，R² = {r_squared:.3f}")
                            
                            # 趋势不一致时的特殊说明
                            if not trend_consistent:
                                st.warning(
                                    f"⚠️ **趋势不一致：** 整体趋势为 {overall_trend}，但近期趋势为 {recent_trend}。\n\n"
                                    f"这说明价格可能在近期发生了**趋势转换**。模型更关注近期数据，"
                                    f"因此预测可能反映近期趋势（{recent_trend}）而非整体趋势（{overall_trend}）。"
                                )
                            else:
                                if recent_trend == "横盘":
                                    st.info(
                                        f"💡 **趋势判断：** 整体和近期都显示为 {recent_trend}，"
                                        f"价格缺乏明确方向。模型预测为水平走势是合理的。"
                                    )
                                else:
                                    st.info(
                                        f"💡 **趋势判断：** 整体和近期都显示为 {recent_trend}趋势，"
                                        f"但模型可能认为趋势强度不足以持续，因此预测为水平或接近水平。"
                                    )
                            
                            # 波动率分析
                            st.divider()
                            st.markdown("### 📊 波动率分析")
                            if volatility_pct < 2:
                                volatility_level = "低波动"
                                volatility_color = "🟢"
                            elif volatility_pct < 5:
                                volatility_level = "中等波动"
                                volatility_color = "🟡"
                            else:
                                volatility_level = "高波动"
                                volatility_color = "🔴"
                            
                            st.markdown(f"{volatility_color} **波动率水平：** {volatility_level} ({volatility_pct:.2f}%)")
                            st.caption("高波动率可能导致模型选择保守的水平预测，以降低预测误差。")
                            
                            # 模型预测合理性说明
                            st.divider()
                            st.markdown("### 🤖 模型预测合理性")
                            
                            # 判断预测为水平的原因
                            reasons = []
                            if abs(recent_trend_pct) < trend_threshold:
                                reasons.append("近期数据缺乏明显趋势（横盘）")
                            if volatility_pct > 5:
                                reasons.append("数据波动率较高，模型选择保守预测")
                            if not trend_consistent:
                                reasons.append("整体趋势与近期趋势不一致，模型更关注近期")
                            if r_squared < 0.3:
                                reasons.append("近期趋势的统计显著性较低（R² < 0.3）")
                            
                            if reasons:
                                st.info(
                                    "**预测为水平可能的原因：**\n\n" +
                                    "\n".join([f"• {reason}" for reason in reasons]) +
                                    "\n\n**建议：** 如果希望获得更明确的趋势预测，可以尝试：\n"
                                    "• 增加历史数据长度（获取更多天数）\n"
                                    "• 使用更长的时间周期（如从 15m 改为 1h 或 4h）\n"
                                    "• 检查数据质量（是否有异常值或缺失值）"
                                )
                            else:
                                st.success("数据质量良好，模型预测应该较为可靠。")
                            
                            # 季节性设置
                            st.divider()
                            if seasonality:
                                # 计算季节性对应的时间长度（小时）
                                interval_hours_map = {
                                    "5m": 5/60,
                                    "15m": 15/60,
                                    "1h": 1,
                                    "4h": 4,
                                    "1d": 24,
                                }
                                interval_hours = interval_hours_map.get(interval, 1)
                                seasonality_hours = seasonality * interval_hours
                                st.caption(f"📅 **季节性设置：** {seasonality} ({interval} 周期，约 {seasonality_hours:.1f} 小时）")
                            else:
                                st.caption(f"📅 **季节性设置：** 未设置（数据长度可能不足，或周期不适合设置季节性）")

                        # 显示预测结果
                        st.subheader("📊 预测结果")
                        st.dataframe(result.fcst_df, width='stretch')

                        # 可视化预测
                        st.subheader("📈 预测可视化")

                        # 合并历史数据和预测数据
                        historical_prices = kline_data[price_type].values
                        historical_times = kline_data["open_time"].values

                        # 获取预测列名（模型名称，如 'AutoARIMA'）
                        # 优先使用 TimeCopilot 选择的最佳模型列，避免误用基线模型（如 seasonal_naive）导致水平线
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
                            selected_model = getattr(result.output, "selected_model", None)
                            if selected_model in forecast_cols:
                                forecast_col = selected_model
                            else:
                                # 兜底：仍然使用第一个模型列
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
                            # 计算预测的趋势
                            if len(forecast_prices) >= 2:
                                forecast_start = forecast_prices[0]
                                forecast_end = forecast_prices[-1]
                                forecast_change_pct = ((forecast_end - forecast_start) / forecast_start) * 100 if forecast_start > 0 else 0
                                
                                # 判断预测是否为水平
                                is_flat = abs(forecast_change_pct) < 0.1  # 变化小于 0.1% 视为水平
                                
                                # 根据预测趋势选择颜色
                                if is_flat:
                                    forecast_color = "orange"  # 橙色表示水平
                                    forecast_name = f"预测价格 ({forecast_col}) - 横盘"
                                elif forecast_change_pct > 0:
                                    forecast_color = "green"  # 绿色表示上涨
                                    forecast_name = f"预测价格 ({forecast_col}) - 上涨 {forecast_change_pct:.2f}%"
                                else:
                                    forecast_color = "red"  # 红色表示下跌
                                    forecast_name = f"预测价格 ({forecast_col}) - 下跌 {abs(forecast_change_pct):.2f}%"
                            else:
                                forecast_color = "red"
                                forecast_name = f"预测价格 ({forecast_col})"
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=forecast_times,
                                    y=forecast_prices,
                                    mode="lines+markers",
                                    name=forecast_name,
                                    line=dict(color=forecast_color, width=2, dash="dash"),
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
                            
                            # 如果预测是水平的，添加说明
                            if len(forecast_prices) >= 2 and abs(forecast_change_pct) < 0.1:
                                st.info(
                                    f"💡 **预测说明：** 模型预测未来 {forecast_horizon} 个周期内价格基本保持稳定（变化 < 0.1%），"
                                    f"这可能是因为：\n"
                                    f"1. 历史数据在近期缺乏明显趋势\n"
                                    f"2. 模型认为当前价格水平是合理的均衡点\n"
                                    f"3. 数据波动较大，模型选择保守预测\n\n"
                                    f"**建议：** 如果历史数据显示有明显趋势但预测为水平，可以尝试：\n"
                                    f"- 增加历史数据长度（获取更多天数）\n"
                                    f"- 检查数据质量（是否有异常值）\n"
                                    f"- 考虑使用其他模型或调整参数"
                                )

                        fig.update_layout(
                            title=f"{symbol} 价格预测 ({interval})",
                            xaxis_title="时间",
                            yaxis_title="价格 (USDT)",
                            height=600,
                            hovermode="x unified",
                        )

                        st.plotly_chart(fig, width='stretch')

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
                        # 清除进度显示
                        if 'progress_container' in locals():
                            status_text.empty()
                            progress_bar.empty()
                        
                        st.error(f"❌ 预测失败: {type(e).__name__}: {e}")
                        import traceback

                        with st.expander("查看详细错误"):
                            st.code(traceback.format_exc())
                        
                        # 提供故障排除建议
                        st.info(
                            "**故障排除建议：**\n\n"
                            "1. **检查数据量**：\n"
                            "   - 如果数据点超过 1000 个，系统会自动采样到最近 1000 条\n"
                            "   - 数据量太少（< 20 条）也会导致预测失败\n\n"
                            "2. **减少预测步数**：\n"
                            "   - 尝试将预测步数（h）设置为较小的值（如 5-10）\n"
                            "   - 预测步数越大，计算时间越长\n\n"
                            "3. **检查 API Key**：\n"
                            "   - 确保 DashScope API Key 有效且有足够余额\n"
                            "   - 检查 .env 文件中的 DASHSCOPE_API_KEY 配置\n\n"
                            "4. **网络连接**：\n"
                            "   - 确保网络连接稳定，预测过程需要多次 API 调用\n"
                            "   - 如果网络不稳定，可以增加重试次数\n\n"
                            "5. **数据质量**：\n"
                            "   - 确保价格数据没有异常值或缺失值\n"
                            "   - 尝试使用不同的价格类型（close/open/high/low）\n\n"
                            "6. **重试**：\n"
                            "   - 如果失败，可以点击按钮重试，有时是临时网络问题\n"
                            "   - 如果多次失败，请检查终端日志获取详细错误信息"
                        )

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
                "趋势阈值 (%)", min_value=0.0, max_value=10.0, value=0.5, step=0.1,
                help="预测价格变化超过此百分比才认为有趋势。降低此值可以更容易触发交易，但可能产生更多假信号。"
            ) / 100

        with col2:
            min_confidence = st.slider(
                "最小置信度", min_value=0.0, max_value=1.0, value=0.3, step=0.05,
                help="信号置信度低于此值将被过滤。降低此值可以更容易触发交易，但可能降低信号质量。"
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
                    
                    # 信号统计和诊断信息
                    if hasattr(backtest_result, "signal_stats") and backtest_result.signal_stats:
                        st.divider()
                        st.subheader("🔍 信号分析")
                        
                        signal_stats = backtest_result.signal_stats
                        total_signals = sum([signal_stats.get(k, 0) for k in ["BUY", "SELL", "HOLD"]])
                        
                        if total_signals > 0:
                            col1, col2, col3, col4, col5 = st.columns(5)
                            with col1:
                                st.metric("买入信号", signal_stats.get("BUY", 0))
                            with col2:
                                st.metric("卖出信号", signal_stats.get("SELL", 0))
                            with col3:
                                st.metric("持有信号", signal_stats.get("HOLD", 0))
                            with col4:
                                st.metric("低置信度", signal_stats.get("LOW_CONFIDENCE", 0))
                            with col5:
                                st.metric("无趋势", signal_stats.get("NO_TREND", 0))
                            
                            # 诊断信息
                            if len(backtest_result.trades) == 0:
                                st.warning("⚠️ **未生成任何交易**")
                                reasons = []
                                if signal_stats.get("BUY", 0) == 0 and signal_stats.get("SELL", 0) == 0:
                                    reasons.append("策略未生成任何买入或卖出信号")
                                if signal_stats.get("LOW_CONFIDENCE", 0) > total_signals * 0.5:
                                    reasons.append(f"大部分信号因置信度不足被过滤（{signal_stats.get('LOW_CONFIDENCE', 0)}/{total_signals}）")
                                if signal_stats.get("NO_TREND", 0) > total_signals * 0.5:
                                    reasons.append(f"大部分信号因趋势不足被过滤（{signal_stats.get('NO_TREND', 0)}/{total_signals}）")
                                
                                if reasons:
                                    st.info(
                                        "**可能的原因：**\n\n" +
                                        "\n".join([f"• {reason}" for reason in reasons]) +
                                        "\n\n**优化建议：**\n"
                                        f"• 降低趋势阈值（当前: {trend_threshold*100:.1f}%）\n"
                                        f"• 降低最小置信度（当前: {min_confidence:.2f}）\n"
                                        "• 检查预测数据是否有明显趋势\n"
                                        "• 考虑使用其他策略或调整参数"
                                    )
                        else:
                            st.warning("⚠️ 未生成任何信号，请检查预测数据和策略参数")

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

                    st.plotly_chart(fig, width='stretch')

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
                        st.dataframe(trades_df, width='stretch')

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

# ========== 标签页 5: 实时回预测 ==========
with tab5:
    st.header("⏱️ 实时回预测（历史预测 vs 实际 & 下一阶段预测）")

    if "kline_data" not in st.session_state or st.session_state.kline_data.empty:
        st.warning("⚠️ 请先在「数据获取」标签页获取 K 线数据")
    else:
        kline_data = st.session_state.kline_data
        symbol = st.session_state.symbol
        interval = st.session_state.interval

        st.info(
            f"当前数据: {symbol} | {interval} | {len(kline_data)} 条记录；"
            "本页会先在历史上做一段“回测预测 vs 实际”对比，再给出未来 10 根 K 线的价格预测。"
        )

        # 历史回测长度允许调整；未来预测固定使用 10 步 close 价格
        backtest_horizon = st.slider(
            "历史回测长度（K 线数）",
            min_value=20,
            max_value=min(300, len(kline_data) - 20),
            value=min(100, max(20, len(kline_data) // 5)),
            step=5,
            help="从最近的历史数据中截取一段（例如最近 100 根 K 线），"
            "用更早的数据训练模型，对这段历史做预测并与真实价格对比。",
        )
        realtime_horizon = 10

        if st.button("⚡ 运行实时回预测", type="primary", use_container_width=True):
            if not api_key:
                st.error("❌ 请先配置 API Key！")
            else:
                try:
                    with st.spinner("正在进行历史回测预测与未来预测，这通常需要 1-3 分钟，请耐心等待..."):
                        # 准备 TimeCopilot 输入数据
                        adapter = BinanceKlineAdapter()
                        tc_data = adapter.to_timecopilot_format(
                            kline_data, symbol, "close"
                        )

                        if tc_data.empty:
                            raise ValueError("数据转换后为空，请检查数据格式")

                        if "y" not in tc_data.columns or tc_data["y"].isna().all():
                            raise ValueError("价格数据无效，所有值都是 NaN")

                        # 为了避免长时间阻塞，限制参与预测的数据量（与价格预测页保持一致）
                        max_data_points = 800  # 比价格预测页稍小，保证本页更快
                        original_length = len(tc_data)
                        if len(tc_data) > max_data_points:
                            st.info(
                                f"ℹ️ 原始数据共有 {original_length} 条，本页为了加快回测速度，"
                                f"仅使用最近 {max_data_points} 条数据进行历史回测与未来预测。"
                            )
                            tc_data = tc_data.tail(max_data_points).reset_index(drop=True)

                        data_length = len(tc_data)

                        if data_length <= backtest_horizon + 10:
                            raise ValueError(
                                f"当前用于预测的数据量为 {data_length} 条，"
                                f"不足以进行长度为 {backtest_horizon} 的历史回测预测。\n\n"
                                f"请尝试：\n"
                                f"- 将历史回测长度从 {backtest_horizon} 调小；或\n"
                                f"- 在「数据获取」页增加历史天数，再重新运行本页。"
                            )

                        # 频率与季节性设置（与价格预测页保持一致）
                        freq = adapter.get_freq(interval)
                        seasonality_map = {
                            "5m": 288,
                            "15m": 96,
                            "1h": 24,
                            "4h": 6,
                            "1d": 7,
                        }
                        base_seasonality = seasonality_map.get(interval)
                        if base_seasonality and data_length < base_seasonality * 2:
                            if data_length < base_seasonality:
                                seasonality = None
                            else:
                                seasonality = min(base_seasonality, data_length // 2)
                        else:
                            seasonality = base_seasonality

                        # 拆分训练集和“历史回测”测试集
                        train_df = tc_data.iloc[:-backtest_horizon].reset_index(
                            drop=True
                        )
                        test_df = tc_data.iloc[-backtest_horizon:].reset_index(
                            drop=True
                        )

                        # 创建模型与 TimeCopilot
                        model = create_dashscope_model()
                        if model is None:
                            st.error("❌ 无法创建模型，请检查 API Key 配置")
                        else:
                            tc = TimeCopilot(llm=model, retries=3)

                            # 1）对历史回测区间做预测（不需要自然语言分析，只要预测值）
                            backtest_result = tc.forecast(
                                df=train_df,
                                freq=freq,
                                h=backtest_horizon,
                                seasonality=seasonality,
                                query=None,
                            )

                            if (
                                backtest_result is None
                                or not hasattr(backtest_result, "fcst_df")
                                or backtest_result.fcst_df.empty
                            ):
                                raise RuntimeError("历史回测预测结果为空")

                            back_fcst_df = backtest_result.fcst_df
                            back_cols = [
                                c
                                for c in back_fcst_df.columns
                                if c not in ["unique_id", "ds"]
                            ]
                            if not back_cols:
                                raise RuntimeError("历史回测预测结果中没有预测值列")

                            # 优先使用 TimeCopilot 选择的最佳模型，避免默认拿到基线模型（水平线）
                            back_selected = getattr(
                                backtest_result.output, "selected_model", None
                            )
                            if back_selected in back_cols:
                                back_col = back_selected
                            else:
                                back_col = back_cols[0]

                            back_pred = back_fcst_df[back_col].values
                            back_time = pd.to_datetime(back_fcst_df["ds"]).values

                            # 对齐真实价格（使用 tc_data 最后 backtest_horizon 段的 y）
                            real_back_prices = test_df["y"].values
                            real_back_time = pd.to_datetime(test_df["ds"]).values

                            # 2）基于全部历史数据做未来 realtime_horizon 步预测
                            live_result = tc.forecast(
                                df=tc_data,
                                freq=freq,
                                h=realtime_horizon,
                                seasonality=seasonality,
                                query=None,
                            )

                            if (
                                live_result is None
                                or not hasattr(live_result, "fcst_df")
                                or live_result.fcst_df.empty
                            ):
                                raise RuntimeError("未来预测结果为空")

                            live_fcst_df = live_result.fcst_df
                            live_cols = [
                                c
                                for c in live_fcst_df.columns
                                if c not in ["unique_id", "ds"]
                            ]
                            if not live_cols:
                                raise RuntimeError("未来预测结果中没有预测值列")

                            live_selected = getattr(
                                live_result.output, "selected_model", None
                            )
                            if live_selected in live_cols:
                                live_col = live_selected
                            else:
                                live_col = live_cols[0]

                            live_pred = live_fcst_df[live_col].values
                            live_time = pd.to_datetime(live_fcst_df["ds"]).values

                        # === 图表 1：历史回测预测 vs 真实 close 双线 ===
                        st.subheader("📉 历史回测：预测 vs 实际")

                        fig_back = go.Figure()
                        fig_back.add_trace(
                            go.Scatter(
                                x=real_back_time,
                                y=real_back_prices,
                                mode="lines",
                                name="真实价格 (close)",
                                line=dict(color="blue", width=2),
                            )
                        )
                        fig_back.add_trace(
                            go.Scatter(
                                x=back_time,
                                y=back_pred,
                                mode="lines+markers",
                                name="模型预测价格 (回测)",
                                line=dict(color="orange", width=2, dash="dash"),
                                marker=dict(size=4),
                            )
                        )

                        fig_back.update_layout(
                            title=f"{symbol} {interval} 历史回测：预测 vs 实际（最近 {backtest_horizon} 根 K 线）",
                            xaxis_title="时间",
                            yaxis_title="价格 (USDT)",
                            height=500,
                            hovermode="x unified",
                        )
                        st.plotly_chart(fig_back, use_container_width=True)

                        # === 图表 2：全历史 close + 未来 10 步预测 ===
                        st.subheader(f"📈 全历史 + 未来 {realtime_horizon} 步预测（close）")

                        hist_close = kline_data["close"].values
                        hist_time = kline_data["open_time"].values

                        fig_live = go.Figure()
                        fig_live.add_trace(
                            go.Scatter(
                                x=hist_time,
                                y=hist_close,
                                mode="lines",
                                name="历史价格 (close)",
                                line=dict(color="blue", width=2),
                            )
                        )
                        fig_live.add_trace(
                            go.Scatter(
                                x=live_time,
                                y=live_pred,
                                mode="lines+markers",
                                name=f"未来 {realtime_horizon} 根 K 线预测价格",
                                line=dict(color="green", width=2, dash="dash"),
                                marker=dict(size=5),
                            )
                        )

                        if len(hist_time) > 0 and len(live_time) > 0:
                            fig_live.add_trace(
                                go.Scatter(
                                    x=[hist_time[-1], live_time[0]],
                                    y=[hist_close[-1], live_pred[0]],
                                    mode="lines",
                                    name="连接",
                                    line=dict(color="gray", width=1, dash="dot"),
                                    showlegend=False,
                                )
                            )

                        fig_live.update_layout(
                            title=f"{symbol} {interval} 历史价格 + 未来 {realtime_horizon} 根 K 线预测",
                            xaxis_title="时间",
                            yaxis_title="价格 (USDT)",
                            height=500,
                            hovermode="x unified",
                        )
                        st.plotly_chart(fig_live, use_container_width=True)

                except Exception as e:
                    st.error(f"❌ 实时回预测失败: {type(e).__name__}: {e}")
                    import traceback

                    with st.expander("查看详细错误"):
                        st.code(traceback.format_exc())

# ========== 标签页 6: 实时预测 ==========
with tab6:
    st.header("🔄 实时预测（基于最新K线数据）")

    st.info(
        "本页面会从 Binance 获取最新的K线数据，并基于这些实时数据对未来10根K线进行预测。"
        "数据会自动获取到当前时刻的最新K线。"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        realtime_symbol = st.selectbox(
            "交易对",
            ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT"],
            index=0,
            key="realtime_symbol",
        )

    with col2:
        realtime_interval = st.selectbox(
            "K线周期",
            ["5m", "15m", "1h", "4h", "1d"],
            index=1,
            key="realtime_interval",
        )

    with col3:
        realtime_data_limit = st.number_input(
            "历史数据量（K线数）",
            min_value=50,
            max_value=1000,
            value=500,
            step=50,
            help="用于预测的历史K线数量。数据量越大，预测可能更准确，但计算时间更长。",
            key="realtime_data_limit",
        )

    # 实时预测的步数固定为10，价格类型固定为close
    realtime_forecast_horizon = 10
    realtime_price_type = "close"

    if st.button("🔄 获取最新数据并预测", type="primary", use_container_width=True):
        if not api_key:
            st.error("❌ 请先配置 API Key！")
        else:
            try:
                with st.spinner("正在获取最新K线数据..."):
                    # 获取最新数据
                    fetcher = BinanceDataFetcher()
                    latest_kline_data = fetcher.fetch_latest_klines(
                        symbol=realtime_symbol,
                        interval=realtime_interval,
                        limit=realtime_data_limit,
                    )

                    if latest_kline_data.empty:
                        st.error("❌ 未能获取到最新数据，请检查网络连接")
                    else:
                        st.success(f"✅ 成功获取 {len(latest_kline_data)} 条最新K线数据")
                        
                        # 显示最新数据信息
                        latest_time = latest_kline_data["open_time"].iloc[-1]
                        latest_price = latest_kline_data["close"].iloc[-1]
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("最新K线时间", latest_time.strftime("%Y-%m-%d %H:%M:%S"))
                        with col2:
                            st.metric("最新收盘价", f"${latest_price:,.2f}")
                        with col3:
                            st.metric("数据条数", len(latest_kline_data))

                # 进行预测
                with st.spinner("正在进行实时预测（这可能需要1-3分钟）..."):
                    # 转换数据格式
                    adapter = BinanceKlineAdapter()
                    tc_data = adapter.to_timecopilot_format(
                        latest_kline_data, realtime_symbol, realtime_price_type
                    )

                    if tc_data.empty:
                        raise ValueError("数据转换后为空，请检查数据格式")

                    if "y" not in tc_data.columns or tc_data["y"].isna().all():
                        raise ValueError("价格数据无效，所有值都是 NaN")

                    # 限制数据量以提高速度
                    max_data_points = 600  # 实时预测使用稍少的数据量以加快速度
                    original_length = len(tc_data)
                    if len(tc_data) > max_data_points:
                        st.info(
                            f"ℹ️ 原始数据共有 {original_length} 条，为了加快实时预测速度，"
                            f"仅使用最近 {max_data_points} 条数据进行预测。"
                        )
                        tc_data = tc_data.tail(max_data_points).reset_index(drop=True)

                    # 检查最小数据量要求
                    min_data_points = max(20, realtime_forecast_horizon * 2)
                    if len(tc_data) < min_data_points:
                        raise ValueError(
                            f"数据量不足（{len(tc_data)} 条），至少需要 {min_data_points} 条数据。"
                            f"请增加历史数据量。"
                        )

                    # 频率与季节性设置
                    freq = adapter.get_freq(realtime_interval)
                    seasonality_map = {
                        "5m": 288,
                        "15m": 96,
                        "1h": 24,
                        "4h": 6,
                        "1d": 7,
                    }
                    base_seasonality = seasonality_map.get(realtime_interval)
                    data_length = len(tc_data)
                    if base_seasonality and data_length < base_seasonality * 2:
                        if data_length < base_seasonality:
                            seasonality = None
                        else:
                            seasonality = min(base_seasonality, data_length // 2)
                    else:
                        seasonality = base_seasonality

                    # 创建模型与 TimeCopilot
                    model = create_dashscope_model()
                    if model is None:
                        st.error("❌ 无法创建模型，请检查 API Key 配置")
                    else:
                        tc = TimeCopilot(llm=model, retries=3)

                        # 定义面向交易的中文查询
                        query_text = (
                            "你是一名加密货币量化交易分析师，"
                            "根据历史价格和未来预测结果，判断在本次预测区间内价格整体是上涨、下跌还是震荡/横盘。"
                            "请用简短的中文给出结论，可以提到趋势强弱和大致风险提示，"
                            "不要解释模型原理，也不要输出代码。"
                        )

                        # 运行预测
                        result = tc.forecast(
                            df=tc_data,
                            freq=freq,
                            h=realtime_forecast_horizon,
                            seasonality=seasonality,
                            query=query_text,
                        )

                        if result is None:
                            raise Exception("预测未返回结果，请检查数据格式和参数设置")

                        if not hasattr(result, 'fcst_df') or result.fcst_df.empty:
                            raise Exception("预测结果为空，可能是数据格式问题或模型选择失败")

                        st.success("✅ 实时预测完成！")

                        # 保存预测结果到 session_state（可选，用于后续分析）
                        st.session_state.realtime_forecast_result = result
                        st.session_state.realtime_forecast_data = result.fcst_df
                        st.session_state.realtime_kline_data = latest_kline_data

                        # 显示预测结果
                        st.subheader("📊 预测结果")
                        st.dataframe(result.fcst_df, width='stretch')

                        # 可视化预测
                        st.subheader("📈 实时预测可视化")

                        # 合并历史数据和预测数据
                        historical_prices = latest_kline_data[realtime_price_type].values
                        historical_times = latest_kline_data["open_time"].values

                        # 获取预测列名（优先使用最佳模型）
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
                            selected_model = getattr(result.output, "selected_model", None)
                            if selected_model in forecast_cols:
                                forecast_col = selected_model
                            else:
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
                                name="历史价格（实时）",
                                line=dict(color="blue", width=2),
                            )
                        )

                        # 预测数据
                        if len(forecast_prices) > 0 and forecast_col:
                            # 计算预测的趋势
                            if len(forecast_prices) >= 2:
                                forecast_start = forecast_prices[0]
                                forecast_end = forecast_prices[-1]
                                forecast_change_pct = ((forecast_end - forecast_start) / forecast_start) * 100 if forecast_start > 0 else 0

                                # 根据预测趋势选择颜色和名称
                                if abs(forecast_change_pct) < 0.1:
                                    forecast_color = "orange"
                                    forecast_name = f"未来 {realtime_forecast_horizon} 根K线预测（{forecast_col}）- 横盘"
                                elif forecast_change_pct > 0:
                                    forecast_color = "green"
                                    forecast_name = f"未来 {realtime_forecast_horizon} 根K线预测（{forecast_col}）- 上涨 {forecast_change_pct:.2f}%"
                                else:
                                    forecast_color = "red"
                                    forecast_name = f"未来 {realtime_forecast_horizon} 根K线预测（{forecast_col}）- 下跌 {abs(forecast_change_pct):.2f}%"
                            else:
                                forecast_color = "green"
                                forecast_name = f"未来 {realtime_forecast_horizon} 根K线预测（{forecast_col}）"

                            fig.add_trace(
                                go.Scatter(
                                    x=forecast_times,
                                    y=forecast_prices,
                                    mode="lines+markers",
                                    name=forecast_name,
                                    line=dict(color=forecast_color, width=2, dash="dash"),
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
                            title=f"{realtime_symbol} {realtime_interval} 实时预测（最新数据 + 未来 {realtime_forecast_horizon} 根K线）",
                            xaxis_title="时间",
                            yaxis_title="价格 (USDT)",
                            height=600,
                            hovermode="x unified",
                        )

                        st.plotly_chart(fig, width='stretch')

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
                                if not eval_df.empty:
                                    for col in eval_df.columns:
                                        if col != "metric":
                                            mase_score = eval_df[col].iloc[0] if len(eval_df) > 0 else None
                                            if pd.notna(mase_score):
                                                st.write(f"- {col}: {float(mase_score):.4f}")
                            else:
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
                st.error(f"❌ 实时预测失败: {type(e).__name__}: {e}")
                import traceback

                with st.expander("查看详细错误"):
                    st.code(traceback.format_exc())

                st.info(
                    "**故障排除建议：**\n\n"
                    "1. **检查网络连接**：确保能够访问 Binance API\n"
                    "2. **检查数据量**：确保历史数据量足够（至少 50 条）\n"
                    "3. **检查 API Key**：确保 DashScope API Key 有效且有足够余额\n"
                    "4. **重试**：如果失败，可以点击按钮重试"
                )

# 页脚
st.divider()
st.caption(
    "⚠️ 免责声明：本项目仅用于技术研究与模拟交易学习，不构成任何投资建议。加密货币交易具有极高风险，可能导致本金全部损失。"
)

