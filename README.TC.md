TimeCopilot is an open-source forecasting agent that combines the power of large language models with state-of-the-art time series foundation models (Amazon Chronos, Salesforce Moirai, Google TimesFM, Nixtla TimeGPT, etc.). It automates and explains complex forecasting workflows, making time series analysis more accessible while maintaining professional-grade accuracy.

Developed with 💙 at timecopilot.dev.

!!! tip "Want the latest on TimeCopilot?" Have ideas or want to test it in real-world use? Join our Discord community and help shape the future.

🚀 Key Capabilities
Unified Forecasting Layer. Combines 30+ time-series foundation models (Chronos, Moirai, TimesFM, TimeGPT…) with LLM reasoning for automated model selection and explanation.

Natural-Language Forecasting. Ask questions in plain English and get forecasts, analysis, validation, and model comparisons. No scripts, pipelines, or dashboards needed.

One-Line Forecasting. Run end-to-end forecasts on any dataset in seconds with a single command (uvx timecopilot forecast <url>).

📰 News
#1 on the GIFT-Eval benchmark. TimeCopilot reached the top position on the global GIFT-Eval benchmark above AWS, Salesforce, Google, IBM, and top universities.
Accepted at NeurIPS (BERTs Workshop). Our work on agentic forecasting was accepted at NeurIPS 2025.
New Manifesto: “Forecasting, the Agentic Way”. Our founding essay on agentic forecasting, TSFMs, and the future of time series.
How It Works
TimeCopilot is a generative agent that applies a systematic forecasting approach using large language models (LLMs) to:

Interpret statistical features and patterns
Guide model selection based on data characteristics
Explain technical decisions in natural language
Answer domain-specific questions about forecasts
Here is an schematic of TimeCopilot's architecture:

Diagram

Quickstart in 30 sec
TimeCopilot can pull a public time series dataset directly from the web and forecast it in one command. No local files, no Python script, just run it with uvx:

# Baseline run (uses default model: openai:gpt-4o-mini)
uvx timecopilot forecast https://otexts.com/fpppy/data/AirPassengers.csv
Want to try a different LL​M?

uvx timecopilot forecast https://otexts.com/fpppy/data/AirPassengers.csv \
  --llm openai:gpt-4o
Have a specific question?

uvx timecopilot forecast https://otexts.com/fpppy/data/AirPassengers.csv \
  --llm openai:gpt-4o \
  --query "How many air passengers are expected in total in the next 12 months?"
Installation and Setup
TimeCopilot is available on PyPI as timecopilot for Python development. Installation and setup is a simple three-step process:

Install TimeCopilot by running:
pip install timecopilot
or

uv add timecopilot
Generate an OpenAI API Key:

Create an openai account.
Visit the API key page.
Generate a new secret key.
Make sure to copy it, as you’ll need it in the next step.
Export your OpenAI API key as an environment variable by running:

On Linux:

export OPENAI_API_KEY="your-new-secret-key"
On Windows (PowerShell):

setx OPENAI_API_KEY "your-new-secret-key"
Remember to restart session after doing so in order to preserve the changes in the environment variables (Windows). You can also do this through python:

import openai
os.environ["OPENAI_API_KEY"] = "your-new-secret-key"
and that's it!

!!! Important - TimeCopilot requires Python 3.10+. Additionally, it currently does not support macOS running on Intel processors (x86_64). If you’re using this setup, you may encounter installation issues with some dependencies like PyTorch. If you need support for this architecture, please create a new issue. - If on Windows, Python 3.10 is recommended due to some of the packages' current architecture.

Hello World Example
Here is an example to test TimeCopilot:

# Import libraries
import pandas as pd
from timecopilot import TimeCopilot

# Load the dataset
# The DataFrame must include at least the following columns:
# - unique_id: Unique identifier for each time series (string)
# - ds: Date column (datetime format)
# - y: Target variable for forecasting (float format)
# The pandas frequency will be inferred from the ds column, if not provided.
# If the seasonality is not provided, it will be inferred based on the frequency. 
# If the horizon is not set, it will default to 2 times the inferred seasonality.
df = pd.read_csv("https://timecopilot.s3.amazonaws.com/public/data/air_passengers.csv")

# Initialize the forecasting agent
# You can use any LLM by specifying the model parameter
tc = TimeCopilot(
    llm="openai:gpt-4o",
    retries=3,
)

# Generate forecast
# You can optionally specify the following parameters:
# - freq: The frequency of your data (e.g., 'D' for daily, 'M' for monthly)
# - h: The forecast horizon, which is the number of periods to predict
# - seasonality: The seasonal period of your data, which can be inferred if not provided
result = tc.forecast(df=df, freq="MS")

# The output contains:
# - tsfeatures_results: List of calculated time series features
# - tsfeatures_analysis: Natural language analysis of the features
# - selected_model: The best performing model chosen
# - model_details: Technical details about the selected model
# - cross_validation_results: Performance comparison of different models
# - model_comparison: Analysis of why certain models performed better/worse
# - is_better_than_seasonal_naive: Boolean indicating if model beats baseline
# - reason_for_selection: Explanation for model choice
# - forecast: List of future predictions with dates
# - forecast_analysis: Interpretation of the forecast results
# - user_query_response: Response to the user prompt, if any
print(result.output)

# You can also access the forecast results in the same shape of the
# provided input dataframe.  
print(result.fcst_df)

"""
        unique_id         ds       Theta
0   AirPassengers 1961-01-01  440.969208
1   AirPassengers 1961-02-01  429.249237
2   AirPassengers 1961-03-01  490.693176
...
21  AirPassengers 1962-10-01  472.164032
22  AirPassengers 1962-11-01  411.458160
23  AirPassengers 1962-12-01  462.795227
"""
Click to expand the full forecast output
Non-OpenAI LLM endpoints
TimeCopilot uses Pydantic to make calls to LLM endpoints, so it should be compatible with all endpoints pydantic supports. Instructions on using other models/endpoints with Pydantic can be found on the matching pydantic docs page, such as this page for Google's models.

For more details go to the LLM Providers example in TimeCopilot's documentation.

Note: models need support for tool use to function properly with TimeCopilot.

Ask about the future
With TimeCopilot, you can ask questions about the forecast in natural language. The agent will analyze the data, generate forecasts, and provide detailed answers to your queries.

Let's for example ask: "how many air passengers are expected in the next 12 months?"

# Ask specific questions about the forecast
result = tc.forecast(
    df=df,
    freq="MS",
    query="how many air passengers are expected in the next 12 months?",
)

# The output will include:
# - All the standard forecast information
# - user_query_response: Detailed answer to your specific question
#   analyzing the forecast in the context of your query
print(result.output.user_query_response)

"""
The total expected air passengers for the next 12 months is approximately 5,919.
"""
You can ask other types of questions:

Trend analysis:
"What's the expected passenger growth rate?"
Seasonal patterns:
"Which months have peak passenger traffic?"
Specific periods:
"What's the passenger forecast for summer months?"
Comparative analysis:
"How does passenger volume compare to last year?"
Business insights:
"Should we increase aircraft capacity next quarter?"
Next Steps
Try TimeCopilot:
Check out the examples above
Join our Discord server for community support
Share your use cases and feedback
Contribute:
We are passionate about contributions!
Submit feature requests and bug reports
Pick an item from the roadmap
Follow our contributing guide
Stay Updated:
Star the repository
Watch for new releases!
How to cite?
Our pre-print paper is available in arxiv.

@misc{garza2025timecopilot,
      title={TimeCopilot}, 
      author={Azul Garza and Renée Rosillo},
      year={2025},
      eprint={2509.00616},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2509.00616}, 
}