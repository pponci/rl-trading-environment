import pandas as pd
import pytest

from rl_env.components.config import TradingEnvConfig
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

    cfg = TradingEnvConfig(window_size=2)

    env = TradingEnv(df=simple_df, cfg=cfg)

    return env
