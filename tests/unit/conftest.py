import pandas as pd
import pytest

from rl_env.trading_env import TradingEnv


@pytest.fixture
def simple_df() -> pd.DataFrame:
    """
    Simple dataframe.
    """

    prices = [100, 101, 100, 102, 100, 95, 96, 94, 101]

    return pd.DataFrame({"close": prices})


@pytest.fixture
def simple_env(simple_df: pd.DataFrame) -> TradingEnv:
    """
    Simple Enviroment.
    """

    env = TradingEnv(df=simple_df, window_size=2)

    return env
