"""Binance K线数据获取模块."""

import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests


class BinanceDataFetcher:
    """从 Binance API 获取 K线数据."""

    BASE_URL = "https://api.binance.com/api/v3/klines"

    def __init__(self, rate_limit_delay: float = 0.1):
        """
        初始化数据获取器.

        Args:
            rate_limit_delay: 请求之间的延迟（秒），用于避免触发限流
        """
        self.rate_limit_delay = rate_limit_delay

    def fetch_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        获取 K线数据.

        Args:
            symbol: 交易对，如 'BTCUSDT'
            interval: K线周期，如 '5m', '15m', '1h', '4h', '1d'
            start_time: 开始时间
            end_time: 结束时间
            limit: 单次请求的最大K线数量（最大1000）

        Returns:
            DataFrame with columns: open_time, open, high, low, close, volume, ...
        """
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": min(limit, 1000),
        }

        if start_time:
            params["startTime"] = int(start_time.timestamp() * 1000)
        if end_time:
            params["endTime"] = int(end_time.timestamp() * 1000)

        all_klines = []
        while True:
            try:
                response = requests.get(self.BASE_URL, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()

                if not data:
                    break

                all_klines.extend(data)

                # 如果返回的数据少于 limit，说明已经获取完所有数据
                if len(data) < limit:
                    break

                # 更新 startTime 为最后一条数据的时间 + 1
                last_time = data[-1][0]
                params["startTime"] = last_time + 1

                # 如果设置了 end_time 且已经超过，则停止
                if end_time and last_time >= int(end_time.timestamp() * 1000):
                    break

                # 延迟以避免限流
                time.sleep(self.rate_limit_delay)

            except requests.exceptions.RequestException as e:
                raise Exception(f"获取 Binance 数据失败: {e}")

        if not all_klines:
            return pd.DataFrame()

        # 转换为 DataFrame
        df = pd.DataFrame(
            all_klines,
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_volume",
                "trades",
                "taker_buy_base",
                "taker_buy_quote",
                "ignore",
            ],
        )

        # 转换数据类型
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
        for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
            df[col] = df[col].astype(float)

        return df

