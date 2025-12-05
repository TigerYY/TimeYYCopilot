"""Binance K线数据适配器 - 转换为 TimeCopilot 格式."""

import pandas as pd


class BinanceKlineAdapter:
    """将 Binance K线数据转换为 TimeCopilot 所需的格式."""

    @staticmethod
    def to_timecopilot_format(
        kline_df: pd.DataFrame, symbol: str, price_column: str = "close"
    ) -> pd.DataFrame:
        """
        将 Binance K线数据转换为 TimeCopilot 格式.

        Args:
            kline_df: Binance K线 DataFrame，必须包含 open_time 和价格列
            symbol: 交易对标识符
            price_column: 使用的价格列（'open', 'high', 'low', 'close'）

        Returns:
            DataFrame with columns: unique_id, ds, y
        """
        if kline_df.empty:
            return pd.DataFrame(columns=["unique_id", "ds", "y"])

        if price_column not in kline_df.columns:
            raise ValueError(f"价格列 '{price_column}' 不存在于 DataFrame 中")

        result = pd.DataFrame(
            {
                "unique_id": symbol,
                "ds": kline_df["open_time"],
                "y": kline_df[price_column],
            }
        )

        # 确保 ds 列是 datetime 类型
        result["ds"] = pd.to_datetime(result["ds"])
        
        # 按时间排序
        result = result.sort_values(by="ds")
        
        # 去除重复的时间戳（保留最后一个）
        result = result.drop_duplicates(subset=["ds"], keep="last")
        
        # 重置索引
        result = result.reset_index(drop=True)

        return result

    @staticmethod
    def get_freq(interval: str) -> str:
        """
        将 Binance 周期转换为 pandas 频率.

        Args:
            interval: Binance 周期，如 '5m', '15m', '1h', '4h', '1d'

        Returns:
            pandas 频率字符串
        """
        mapping = {
            "1m": "1T",
            "3m": "3T",
            "5m": "5T",
            "15m": "15T",
            "30m": "30T",
            "1h": "H",  # 使用 "H" 而不是 "1H"，确保与模型兼容
            "2h": "2H",
            "4h": "4H",
            "6h": "6H",
            "8h": "8H",
            "12h": "12H",
            "1d": "1D",
            "3d": "3D",
            "1w": "1W",
            "1M": "1M",
        }
        return mapping.get(interval, interval)

