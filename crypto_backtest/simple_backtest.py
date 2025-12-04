"""简单的回测引擎."""

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from crypto_strategy.simple_strategy import Signal, TrendFollowingStrategy


@dataclass
class Trade:
    """交易记录."""

    timestamp: pd.Timestamp
    action: str  # 'BUY', 'SELL'
    price: float
    quantity: float
    value: float
    fee: float
    balance: float


@dataclass
class BacktestResult:
    """回测结果."""

    initial_capital: float
    final_capital: float
    total_return: float
    trades: List[Trade]
    equity_curve: pd.DataFrame
    signal_stats: dict = None  # 信号统计信息（可选）


class SimpleBacktestEngine:
    """简单的回测引擎."""

    def __init__(
        self,
        strategy: TrendFollowingStrategy,
        initial_capital: float = 10000.0,
        fee_rate: float = 0.001,
    ):
        """
        初始化回测引擎.

        Args:
            strategy: 交易策略
            initial_capital: 初始资金
            fee_rate: 手续费率（默认 0.1%）
        """
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate

    def run(
        self,
        historical_data: pd.DataFrame,
        forecast_data: pd.DataFrame,
        price_column: str = "close",
    ) -> BacktestResult:
        """
        运行回测.

        Args:
            historical_data: 历史K线数据
            forecast_data: 预测数据（包含 unique_id, ds, y）
            price_column: 价格列名

        Returns:
            回测结果
        """
        trades: List[Trade] = []
        equity_curve_data = []

        capital = self.initial_capital
        position = 0.0  # 持仓数量
        position_value = 0.0  # 持仓价值

        # 合并历史数据和预测数据
        historical_prices = historical_data[price_column]
        
        # 确保预测数据的时间列是 datetime 类型
        forecast_data_copy = forecast_data.copy()
        forecast_data_copy["ds"] = pd.to_datetime(forecast_data_copy["ds"])
        
        # 获取预测列名（模型名称，如 'AutoARIMA'）
        # forecast_data 包含 unique_id, ds, 和模型名称列
        forecast_cols = [
            col
            for col in forecast_data_copy.columns
            if col not in ["unique_id", "ds"]
        ]
        if not forecast_cols:
            raise ValueError("预测数据中没有找到预测值列")
        
        # 使用第一个模型列
        forecast_col = forecast_cols[0]
        forecast_prices = forecast_data_copy.set_index("ds")[forecast_col]

        # 确保历史数据的时间列是 datetime 类型
        historical_data = historical_data.copy()
        historical_data["open_time"] = pd.to_datetime(historical_data["open_time"])
        
        # 信号统计（用于调试）
        signal_stats = {"BUY": 0, "SELL": 0, "HOLD": 0, "LOW_CONFIDENCE": 0, "NO_TREND": 0}
        
        # 遍历历史数据，在每个时间点生成信号并执行
        # 注意：预测数据是未来的，我们在每个历史时间点使用完整的预测数据来生成信号
        for idx, row in historical_data.iterrows():
            current_price = row[price_column]
            current_time = row["open_time"]

            # 使用完整的预测数据来生成信号（预测是未来的，但我们用它来判断趋势）
            # 如果预测数据为空，跳过
            if forecast_prices.empty or len(forecast_prices) < 2:
                signal_stats["HOLD"] += 1
                equity_curve_data.append(
                    {
                        "timestamp": current_time,
                        "capital": capital,
                        "position": position,
                        "total_value": capital + position_value,
                    }
                )
                continue

            # 生成信号（使用完整的预测数据）
            signal = self.strategy.generate_signal(
                historical_prices[: idx + 1],
                forecast_prices,  # 使用完整的预测数据
                current_price,
            )
            
            # 统计信号
            signal_stats[signal.action] += 1
            if signal.action == "HOLD":
                if signal.confidence < self.strategy.min_confidence:
                    signal_stats["LOW_CONFIDENCE"] += 1
                else:
                    # 检查是否因为趋势不足而 HOLD
                    forecast_start = forecast_prices.iloc[0]
                    forecast_end = forecast_prices.iloc[-1]
                    forecast_change = (forecast_end - forecast_start) / forecast_start
                    if abs(forecast_change) < self.strategy.trend_threshold:
                        signal_stats["NO_TREND"] += 1

            # 执行交易
            if signal.action == "BUY" and position == 0:
                # 买入
                trade_value = capital * 0.95  # 使用 95% 的资金
                fee = trade_value * self.fee_rate
                quantity = (trade_value - fee) / current_price

                position = quantity
                position_value = position * current_price
                capital -= (trade_value + fee)

                trades.append(
                    Trade(
                        timestamp=current_time,
                        action="BUY",
                        price=current_price,
                        quantity=quantity,
                        value=trade_value,
                        fee=fee,
                        balance=capital + position_value,
                    )
                )

            elif signal.action == "SELL" and position > 0:
                # 卖出
                trade_value = position * current_price
                fee = trade_value * self.fee_rate
                capital += (trade_value - fee)

                trades.append(
                    Trade(
                        timestamp=current_time,
                        action="SELL",
                        price=current_price,
                        quantity=position,
                        value=trade_value,
                        fee=fee,
                        balance=capital + position_value,
                    )
                )

                position = 0.0
                position_value = 0.0

            # 更新持仓价值
            if position > 0:
                position_value = position * current_price

            # 记录权益曲线
            equity_curve_data.append(
                {
                    "timestamp": current_time,
                    "capital": capital,
                    "position": position,
                    "total_value": capital + position_value,
                }
            )

        # 计算最终结果
        final_capital = capital + position_value
        total_return = (final_capital - self.initial_capital) / self.initial_capital

        equity_curve = pd.DataFrame(equity_curve_data)

        return BacktestResult(
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            trades=trades,
            equity_curve=equity_curve,
            signal_stats=signal_stats,  # 添加信号统计
        )

