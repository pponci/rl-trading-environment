import datetime

import numpy as np
import pandas as pd

from rl_env.components.config import TradingEnvConfig


class MarketData:
    def __init__(self, df: pd.DataFrame, cfg: TradingEnvConfig) -> None:
        self.cfg = cfg

        if df.empty:
            raise ValueError("dataframe cannot be empty")

        if cfg.close_col not in df.columns:
            raise ValueError("close column is required.")

        if cfg.time_col not in df.columns:
            raise ValueError("time column is required.")

        if cfg.window_size >= len(df):
            raise ValueError(
                f"window_size : {cfg.window_size} musst be a smaller number than",
                f"the lenght of the dataset : {len(df)}",
            )

        df_close = df[cfg.close_col]

        if not pd.api.types.is_numeric_dtype(df[cfg.close_col]):
            raise ValueError(f"'close' column must be numeric, got dtype {df['close'].dtype!r}")

        if df_close.isna().any():
            raise ValueError("'close' column contains NaN values")

        if not np.isfinite(df_close.to_numpy()).all():
            raise ValueError("'close' column contains non-finite values")

        if (df_close <= 0).any():
            raise ValueError("'close' column must contain strictly positive prices")

        df_time = df[cfg.time_col]

        if not pd.api.types.is_datetime64_any_dtype(df_time):
            raise ValueError(
                f"'time' column must be datetime64, got dtype {df_time.dtype!r}. "
                "Cast it with pd.to_datetime() first."
            )

        if df_time.isna().any():
            raise ValueError("'time' column contains NaT values")

        if df_time.duplicated().any():
            raise ValueError("'time' column contains duplicate timestamps")

        self.df = df.sort_values(by=cfg.time_col).reset_index(drop=True)
        self.prices = self.df[cfg.close_col].to_numpy(dtype=np.float32)
        self.times = self.df[cfg.time_col].to_numpy()

        self.length = len(self.df)

    def get_current_price(self, current_step: int) -> float:
        """
        Get price at step.
        """

        return self.prices[current_step]

    def get_time(self, step: int) -> datetime.datetime:
        """
        Get time at current step.
        """

        return self.times[step]

    def get_window(self, step: int) -> np.ndarray:
        """
        Calculate price window for current step.
        """

        return self.prices[step - self.cfg.window_size : step]

    def get_scaled_window(self, step: int) -> np.ndarray:
        """
        Calculate normalized price window for current step.
        """

        window = self.get_window(step=step)

        if window[-1] == 0:
            return np.zeros(self.cfg.window_size)

        return window / window[-1] - 1

    def finished(self, step: int) -> bool:
        """
        Check if dataset is finished.
        """

        return step >= self.length
