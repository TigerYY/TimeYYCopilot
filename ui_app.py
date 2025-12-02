"""
TimeCopilot Streamlit UI - 使用 DashScope 配置
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import pandas as pd
import streamlit as st
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

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
    page_title="TimeCopilot - DashScope",
    page_icon="📈",
    layout="wide",
)

st.title("📈 TimeCopilot 预测系统")
st.markdown("使用 DashScope (Qwen) 模型进行时间序列预测")

# 侧边栏配置
with st.sidebar:
    st.header("⚙️ 配置")
    
    # 检查 API Key
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if api_key:
        st.success(f"✅ API Key 已配置: {api_key[:10]}...{api_key[-4:] if len(api_key) > 14 else ''}")
    else:
        st.error("❌ 未找到 API Key")
        st.info("请在 .env 文件中设置 DASHSCOPE_API_KEY 或 OPENAI_API_KEY")
    
    base_url = os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    model_name = os.getenv("DASHSCOPE_MODEL", "qwen-turbo")
    
    st.info(f"**Base URL:** {base_url}\n**Model:** {model_name}")

# 主界面
tab1, tab2 = st.tabs(["📊 数据预测", "📝 使用说明"])

with tab1:
    st.header("时间序列预测")
    
    # 数据输入方式选择
    input_method = st.radio(
        "选择数据输入方式",
        ["使用示例数据", "上传 CSV 文件", "输入 URL"],
        horizontal=True,
    )
    
    df = None
    
    if input_method == "使用示例数据":
        if st.button("加载 Air Passengers 示例数据"):
            try:
                df = pd.read_csv("https://timecopilot.s3.amazonaws.com/public/data/air_passengers.csv")
                st.success("✅ 数据加载成功！")
                st.dataframe(df.head(10), use_container_width=True)
            except Exception as e:
                st.error(f"❌ 数据加载失败: {e}")
    
    elif input_method == "上传 CSV 文件":
        uploaded_file = st.file_uploader("上传 CSV 文件", type=["csv"])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success("✅ 文件上传成功！")
                st.dataframe(df.head(10), use_container_width=True)
            except Exception as e:
                st.error(f"❌ 文件读取失败: {e}")
    
    elif input_method == "输入 URL":
        url = st.text_input("输入 CSV 文件 URL")
        if url:
            if st.button("加载数据"):
                try:
                    df = pd.read_csv(url)
                    st.success("✅ 数据加载成功！")
                    st.dataframe(df.head(10), use_container_width=True)
                except Exception as e:
                    st.error(f"❌ 数据加载失败: {e}")
    
    # 预测参数配置
    if df is not None and not df.empty:
        st.divider()
        st.subheader("预测参数")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            freq = st.text_input("频率 (freq)", value="MS", help="例如: D (日), MS (月初), H (小时)")
        
        with col2:
            h = st.number_input("预测步数 (h)", min_value=1, max_value=100, value=12)
        
        with col3:
            retries = st.number_input("重试次数", min_value=1, max_value=10, value=5)
        
        query = st.text_input("可选：自然语言查询", placeholder="例如：未来12个月的总预期是多少？")
        
        # 运行预测
        if st.button("🚀 开始预测", type="primary", use_container_width=True):
            if not api_key:
                st.error("❌ 请先配置 API Key！")
            else:
                with st.spinner("正在运行预测，这可能需要一些时间..."):
                    try:
                        # 创建模型
                        model = create_dashscope_model()
                        if model is None:
                            st.error("❌ 无法创建模型，请检查 API Key 配置")
                        else:
                            # 初始化 TimeCopilot
                            tc = TimeCopilot(llm=model, retries=retries)
                            
                            # 运行预测
                            result = tc.forecast(df=df, freq=freq, h=h, query=query if query else None)
                            
                            # 显示结果
                            st.success("✅ 预测完成！")
                            
                            # 预测结果表格
                            st.subheader("📊 预测结果")
                            st.dataframe(result.fcst_df, use_container_width=True)
                            
                            # 模型信息
                            st.subheader("🤖 模型信息")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.metric("选择的模型", result.output.selected_model)
                                st.metric("优于季节性基线", "是" if result.output.is_better_than_seasonal_naive else "否")
                            
                            with col2:
                                if result.output.cross_validation_results:
                                    st.write("**交叉验证结果:**")
                                    for cv_result in result.output.cross_validation_results:
                                        st.write(f"- {cv_result}")
                            
                            # 详细分析
                            if result.output.tsfeatures_analysis:
                                st.subheader("📈 特征分析")
                                st.write(result.output.tsfeatures_analysis)
                            
                            if result.output.model_comparison:
                                st.subheader("🔍 模型比较")
                                st.write(result.output.model_comparison)
                            
                            if result.output.forecast_analysis:
                                st.subheader("📉 预测分析")
                                st.write(result.output.forecast_analysis)
                            
                            if result.output.user_query_response:
                                st.subheader("💬 查询回答")
                                st.write(result.output.user_query_response)
                            
                            # 可视化（简单图表）
                            if not result.fcst_df.empty:
                                st.subheader("📈 预测可视化")
                                chart_data = result.fcst_df.copy()
                                chart_data['ds'] = pd.to_datetime(chart_data['ds'])
                                
                                # 获取模型列名（排除 unique_id 和 ds）
                                model_cols = [col for col in chart_data.columns if col not in ['unique_id', 'ds']]
                                
                                if model_cols:
                                    st.line_chart(chart_data.set_index('ds')[model_cols[0]])
                    
                    except Exception as e:
                        st.error(f"❌ 预测失败: {type(e).__name__}: {e}")
                        import traceback
                        with st.expander("查看详细错误信息"):
                            st.code(traceback.format_exc())

with tab2:
    st.header("📝 使用说明")
    
    st.markdown("""
    ### 功能说明
    
    1. **数据输入**
       - 可以使用示例数据（Air Passengers）
       - 可以上传本地 CSV 文件
       - 可以输入 CSV 文件的 URL
    
    2. **数据格式要求**
       - CSV 文件必须包含以下列：
         - `unique_id`: 时间序列的唯一标识符（字符串）
         - `ds`: 日期列（日期时间格式）
         - `y`: 目标变量（浮点数）
    
    3. **预测参数**
       - **频率 (freq)**: 数据的频率，例如：
         - `D`: 日
         - `MS`: 月初
         - `H`: 小时
         - `15T`: 15分钟
       - **预测步数 (h)**: 要预测的未来步数
       - **重试次数**: API 调用失败时的重试次数
    
    4. **自然语言查询**
       - 可以输入自然语言问题，例如：
         - "未来12个月的总预期是多少？"
         - "哪个模型表现最好？"
         - "预测的趋势是什么？"
    
    ### 配置说明
    
    - API Key 配置在 `.env` 文件中
    - 支持 DashScope (Qwen) 模型
    - Base URL: `https://dashscope.aliyuncs.com/compatible-mode/v1`
    - 默认模型: `qwen-turbo`
    
    ### 注意事项
    
    - 预测过程可能需要一些时间，请耐心等待
    - 确保网络连接正常，可以访问 DashScope API
    - 如果遇到错误，请检查 API Key 是否正确配置
    """)

