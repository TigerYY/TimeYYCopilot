# Project Context

## Purpose
TimeCopilot is an open‑source **GenAI forecasting agent** that combines large language models with state‑of‑the‑art time series foundation models  
to automate end‑to‑end forecasting workflows.  

Core goals:
- Provide a **unified forecasting layer** across 30+ time‑series models (Chronos, Moirai, TimesFM, TimeGPT, classical stats/ML/neural, etc.).
- Let users run **natural‑language forecasting and analysis** instead of writing bespoke pipelines.
- Offer **interpretable outputs**: feature analysis, model comparison, anomaly detection and narrative explanations.
- Serve as a **research & production reference implementation** for “agentic forecasting” (LLMs orchestrating TSFMs + classical models).

## Tech Stack
- **Language & Runtime**
  - Python 3.10–3.13 (library & CLI)

- **Core Libraries**
  - Time series modeling: `statsforecast`, `neuralforecast`, `mlforecast`, `prophet`, `gluonts[torch]`, `tsfeatures`, `utilsforecast[plotting]`
  - Foundation model adapters: `timecopilot-chronos-forecasting`, `timecopilot-granite-tsfm`, `timecopilot-timesfm`, `timecopilot-tirex`, `timecopilot-toto`, `timecopilot-uni2ts`
  - LLM / Agent stack: `openai`, `pydantic-ai`, `pydantic`
  - Infra / tooling: `fire` (CLI), `logfire`, `pandas`, `scipy`, `lightgbm`

- **Build & Tooling**
  - Packaging: `hatchling` via `pyproject.toml`
  - Linting/formatting: `ruff`
  - Testing: `pytest`, `pytest-asyncio`, `pytest-xdist`, `pytest-rerunfailures`, `pytest-cov`, `mktestdocs`
  - Docs site: `mkdocs` + `mkdocs-material` + `mkdocstrings` + `mkdocs-jupyter`

- **Distribution & Interfaces**
  - PyPI package: `timecopilot`
  - CLI entrypoint: `timecopilot` → `timecopilot._cli:main`
  - Python API: `timecopilot.TimeCopilot`, `timecopilot.AsyncTimeCopilot`, `timecopilot.TimeCopilotForecaster`

## Project Conventions

### Code Style
- **Language level**
  - Target Python ≥3.10; use `from __future__` features only when needed.

- **Formatting & Lint**
  - Enforced by `ruff` (see `pyproject.toml`):
    - Line length: 88
    - Selected rules: `B`, `E`, `F`, `I`, `SIM`, `UP`
    - Ignored: `F811` (redefinitions allowed in some patterns, e.g. fasthtml)
  - Imports managed in `isort`‑style via `ruff.lint.isort`, with `timecopilot` as local‑folder.
  - Prefer explicit imports from module paths rather than wildcard imports.

- **Typing & Models**
  - Use standard typing (`list[str]` etc.) and `collections.abc` for callables/iterables.
  - Data structures exchanged with LLMs are modeled with `pydantic.BaseModel` (e.g. `ForecastAgentOutput`) for explicit, validated schemas.

- **Naming**
  - Classes: `PascalCase` (e.g. `TimeCopilotForecaster`, `AsyncTimeCopilot`).
  - Functions & methods: `snake_case`.
  - Private helpers: `_prefix` (e.g. `_transform_time_series_to_text`).
  - Model wrappers expose a consistent `.alias` used as column prefixes in forecast dataframes.

### Architecture Patterns
- **Layered structure**
  - `agent.py`: High‑level LLM‑driven agent (`TimeCopilot` / `AsyncTimeCopilot`) that orchestrates:
    - Feature extraction (`tsfeatures_tool`)
    - Cross‑validation and model selection (`cross_validation_tool`)
    - Forecast generation (`forecast_tool`)
    - Anomaly detection (`detect_anomalies_tool`)
    - Plotting / follow‑up Q&A (`query_agent` + `plot_tool`)
  - `forecaster.py`: Unified `TimeCopilotForecaster` that fans out calls to one or more `Forecaster` implementations and merges their outputs.
  - `models/`: Individual model families (foundation, stats, ML, neural, prophet, ensembles) implementing a shared `Forecaster` interface.
  - `utils/experiment_handler.py`: `ExperimentDataset` abstraction to:
    - Normalize input (DataFrame / file path / URL) into `["unique_id", "ds", "y"]`.
    - Infer `freq` and seasonality when absent.
    - Evaluate forecasts (e.g. MASE) over cross‑validation windows.

- **Agent orchestration**
  - Agents built with `pydantic-ai.Agent`:
    - `forecasting_agent` runs the 4‑step workflow via tools and produces a `ForecastAgentOutput`.
    - `query_agent` reuses cached dataframes (`fcst_df`, `eval_df`, `features_df`, `anomalies_df`) to answer follow‑up questions and generate plots.
  - System prompts encode the required workflow order and constraints (e.g. must beat seasonal naive baseline).
  - An `output_validator` enforces that the selected model is strictly better than `SeasonalNaive` according to MASE, otherwise triggers `ModelRetry`.

- **Dataframe semantics**
  - Standardized long format:
    - `unique_id`: series identifier
    - `ds`: timestamp
    - `y`: observed value
  - Forecast and CV outputs:
    - One column per model alias, optionally additional columns for quantiles / prediction intervals.
    - Cross‑validation outputs include `cutoff` to identify windows.
  - Anomaly detection outputs add `*-anomaly` boolean columns and bounds where appropriate.

- **Extensibility**
  - New models should subclass / conform to `Forecaster` and implement:
    - `forecast(df, h, freq, level=None, quantiles=None) -> pd.DataFrame`
    - `cross_validation(df, h, freq, ...) -> pd.DataFrame`
    - `detect_anomalies(df, freq, level, ...) -> pd.DataFrame`
  - `TimeCopilotForecaster` validates unique `.alias` values to avoid column conflicts.

### Testing Strategy
- **Frameworks & tools**
  - Core: `pytest`, `pytest-asyncio`, `pytest-mock`, `pytest-xdist`, `pytest-rerunfailures`, `pytest-cov`.
  - Docs tests: `mktestdocs` to ensure examples in docs/README stay runnable.

- **Configuration**
  - `pyproject.toml`:
    - `testpaths = ["tests"]`
    - Default markers exclude slow / external tests: `-m 'not docs and not live and not gift_eval'`
  - Coverage:
    - `fail_under = 80`
    - Branch coverage enabled; sources limited to `timecopilot/`.

- **Test layout**
  - `tests/agent.py`, `tests/test_forecaster.py`: agent orchestration and unified forecaster behavior.
  - `tests/models/...`: individual model adapters, foundation models, and utilities.
  - `tests/utils/test_experiment_handler.py`: dataset parsing, evaluation logic.
  - `tests/docs/test_docs.py`: documentation build and example integrity.

- **Guidelines**
  - New behavior SHOULD come with tests; bug fixes SHOULD include regression tests.
  - Prefer deterministic tests that do not depend on live LLM calls:
    - Use fixtures/mocks for OpenAI / pydantic‑ai interactions.
    - Mark live tests with `@pytest.mark.live` or similar and keep them opt‑in.

### Git Workflow
- **Branching**
  - Default branch: `main`.
  - Feature work is generally done on topic branches (e.g. `feature/add-chronos-option`) and merged via PR.

- **Commits**
  - No strict enforced format, but recommended:
    - Clear, imperative messages: `Add AsyncTimeCopilot query_stream`, `Fix TimesFM seasonality detection`.
    - Group related code + tests + docs in the same commit/PR where feasible.

- **PR & Reviews**
  - Run tests (`pytest`) and linters (`ruff`) before opening PRs.
  - Keep PRs small and focused on a single capability or refactor when possible.

## Domain Context
- **Problem domain**
  - Univariate and multivariate time series forecasting across business, operations, and research use cases.
  - Tasks include: point forecasting, model selection, cross‑validated evaluation, anomaly detection, and interpretability.

- **Typical usage patterns**
  - CLI one‑liners for quick experiments:
    - `uvx timecopilot forecast <csv_url> [--llm <provider:model>] [--query "..."]`
  - Python API usage in notebooks and applications:
    - `TimeCopilot(llm="openai:gpt-4o").analyze(df, freq="MS", h=12, query="...")`
    - Follow‑up questions via `tc.query("Which model performed best?")`.

- **Key concepts for assistants**
  - **TSFM (Time Series Foundation Models)** vs. classical models (ARIMA/ETS/Theta) and ML (LightGBM, neural nets).
  - **MASE** as the canonical evaluation metric for model comparison.
  - **Seasonality & frequency inference**: many parameters are inferred; user hints via natural language override defaults.
  - **Seasonal naive baseline** as a mandatory comparison reference.

## Important Constraints
- **Runtime & platform**
  - Requires Python ≥3.10; certain dependencies have tighter constraints (e.g. `tabpfn-time-series` only for `<3.13`, some packages prefer ≤3.12).
  - On macOS Intel (x86_64) there may be installation issues (especially around PyTorch / GPU‑related dependencies).

- **LLM access**
  - Requires valid API keys for LLM providers (OpenAI by default) configured via environment variables (e.g. `OPENAI_API_KEY`).
  - Many core workflows rely on the LLM; offline or key‑less operation is not a primary goal.

- **Performance & cost**
  - Cross‑validation + multiple models + LLM‑driven reasoning can be compute and cost intensive:
    - Use selective model lists and shorter horizons for quick experiments.
    - For large‑scale batch jobs, prefer minimal prompts and limited model sets.

- **Spec‑driven changes**
  - For new capabilities / breaking changes / architecture shifts, follow OpenSpec workflow:
    - Create a `changes/<change-id>/` entry with `proposal.md`, `tasks.md` and spec deltas.
    - Keep `openspec/specs` as the source of truth for “what IS built”.

## External Dependencies
- **LLM providers**
  - Primary: OpenAI models through `openai` and `pydantic-ai`.
  - Compatible with any pydantic‑ai supported LLM endpoint (e.g. other cloud providers) provided they support tool use.

- **Cloud & data**
  - Example datasets and assets hosted on `https://timecopilot.s3.amazonaws.com/` (public read).
  - Documentation and experiments referenced from `https://timecopilot.dev/` and related URLs.

- **Foundation model backends**
  - Time series foundation models may rely on vendor‑specific infrastructure (e.g. AWS for Chronos, Salesforce, Google, IBM, Nixtla APIs, etc.) encapsulated by the `timecopilot-*` adapter packages.
  - When adding or updating these adapters, ensure compatibility with upstream APIs and document any additional environment variables or credentials required.
