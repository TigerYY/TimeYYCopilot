"""简单的交易策略实现."""

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class Signal:
    """交易信号."""

    action: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float  # 0-1
    price: float
    timestamp: pd.Timestamp


class TrendFollowingStrategy:
    """趋势跟随策略 - 基于预测趋势生成交易信号."""

    def __init__(
        self,
        trend_threshold: float = 0.02,
        min_confidence: float = 0.6,
        lookback_periods: int = 3,
    ):
        """
        初始化策略.

        Args:
            trend_threshold: 趋势阈值（百分比），超过此值才认为有趋势
            min_confidence: 最小置信度，低于此值不交易
            lookback_periods: 回看周期数，用于计算趋势
        """
        self.trend_threshold = trend_threshold
        self.min_confidence = min_confidence
        self.lookback_periods = lookback_periods

    def generate_signal(
        self,
        historical_prices: pd.Series,
        forecast_prices: pd.Series,
        current_price: float,
    ) -> Signal:
        """
        生成交易信号.

        Args:
            historical_prices: 历史价格序列
            forecast_prices: 预测价格序列
            current_price: 当前价格

        Returns:
            交易信号
        """
        if forecast_prices.empty:
            return Signal("HOLD", 0.0, current_price, pd.Timestamp.now())

        # 计算预测趋势
        forecast_start = forecast_prices.iloc[0]
        forecast_end = forecast_prices.iloc[-1]
        forecast_change = (forecast_end - forecast_start) / forecast_start

        # 计算置信度（基于预测的一致性）
        if len(forecast_prices) > 1:
            price_changes = forecast_prices.diff().dropna()
            consistency = 1.0 - (price_changes.std() / price_changes.abs().mean())
            confidence = max(0.0, min(1.0, consistency))
        else:
            confidence = 0.5

        # 如果置信度太低，不交易
        if confidence < self.min_confidence:
            return Signal("HOLD", confidence, current_price, pd.Timestamp.now())

        # 根据趋势生成信号
        if forecast_change > self.trend_threshold:
            return Signal("BUY", confidence, current_price, pd.Timestamp.now())
        elif forecast_change < -self.trend_threshold:
            return Signal("SELL", confidence, current_price, pd.Timestamp.now())
        else:
            return Signal("HOLD", confidence, current_price, pd.Timestamp.now())

