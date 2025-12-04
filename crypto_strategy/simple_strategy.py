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

        # 计算置信度
        # 目标：只要预测的整体趋势幅度相对于阈值足够大，就给出较高置信度，
        # 避免绝大多数情况下置信度都接近 0 导致没有任何交易。
        if len(forecast_prices) > 1:
            # 基于整体趋势幅度的置信度：当 |forecast_change| == trend_threshold 时约为 0.5，
            # 当 |forecast_change| >= 2 * trend_threshold 时接近 1.0。
            if self.trend_threshold > 0:
                trend_based_conf = min(
                    1.0, abs(forecast_change) / (self.trend_threshold * 2)
                )
            else:
                trend_based_conf = 1.0

            # 可选：结合预测变化的一致性（平滑度）
            price_changes = forecast_prices.diff().dropna()
            if not price_changes.empty and price_changes.abs().mean() > 0:
                consistency_raw = 1.0 - (
                    price_changes.std() / price_changes.abs().mean()
                )
                consistency = max(0.0, min(1.0, consistency_raw))
            else:
                # 完全水平或无法计算波动时，认为一致性较高
                consistency = 0.8

            # 综合置信度：趋势强度 + 一致性，各占 50%
            confidence = 0.5 * trend_based_conf + 0.5 * consistency
        else:
            # 仅一个点时，无法判断趋势，给中等置信度
            confidence = 0.5

        # 如果置信度太低，不交易
        if confidence < self.min_confidence:
            return Signal("HOLD", confidence, current_price, pd.Timestamp.now())

        # 根据趋势生成信号
        # 对于水平预测（变化很小），可以考虑基于历史趋势或当前价格位置来判断
        if abs(forecast_change) < self.trend_threshold:
            # 预测为水平，检查历史趋势
            if len(historical_prices) >= 2:
                historical_trend = (historical_prices.iloc[-1] - historical_prices.iloc[-min(10, len(historical_prices))]) / historical_prices.iloc[-min(10, len(historical_prices))]
                # 如果历史有轻微趋势，且预测为水平，可以跟随历史趋势
                if abs(historical_trend) > self.trend_threshold * 0.5:  # 历史趋势阈值降低
                    if historical_trend > 0:
                        return Signal("BUY", confidence * 0.8, current_price, pd.Timestamp.now())  # 降低置信度
                    else:
                        return Signal("SELL", confidence * 0.8, current_price, pd.Timestamp.now())
            # 否则保持 HOLD
            return Signal("HOLD", confidence, current_price, pd.Timestamp.now())
        elif forecast_change > self.trend_threshold:
            return Signal("BUY", confidence, current_price, pd.Timestamp.now())
        elif forecast_change < -self.trend_threshold:
            return Signal("SELL", confidence, current_price, pd.Timestamp.now())
        else:
            return Signal("HOLD", confidence, current_price, pd.Timestamp.now())

