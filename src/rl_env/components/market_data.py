import numpy as np
import pandas as pd

from rl_env.components.config import TradingEnvConfig


class MarketData:
    def __init__(self, df: pd.DataFrame, cfg: TradingEnvConfig) -> None:
        self.cfg = cfg

        if df.empty:
            raise ValueError("dataframe cannot be empty")

        if "close" not in df.columns:
            raise ValueError("close column is required.")

        if cfg.window_size >= len(df):
            raise ValueError(
                f"window_size : {cfg.window_size} musst be a smaller number than",
                f"the lenght of the dataset : {len(df)}",
            )

        if not pd.api.types.is_numeric_dtype(df["close"]):
            raise ValueError(f"'close' column must be numeric, got dtype {df['close'].dtype!r}")

        if df["close"].isna().any():
            raise ValueError("'close' column contains NaN values")

        if not np.isfinite(df["close"].to_numpy()).all():
            raise ValueError("'close' column contains non-finite values")

        if (df["close"] <= 0).any():
            raise ValueError("'close' column must contain strictly positive prices")

        self.df = df.reset_index(drop=True)
        self.prices = df["close"].to_numpy(dtype=np.float32)

        self.length = len(self.df)

    def get_current_price(self, current_step: int) -> float:
        """
        Get price at step.
        """

        return self.prices[current_step]

    def get_window(self, step: int) -> np.ndarray:
        """
        Calculate price window for current step.
        """

        return self.prices[step - self.cfg.window_size : step]

    def finished(self, step: int) -> bool:
        """
        Check if dataset is finished.
        """

        return step >= self.length
