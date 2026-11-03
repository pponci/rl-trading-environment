import datetime
from typing import cast

import numpy as np
import pandas as pd
import pytest
from gymnasium.spaces import Box, Discrete

from rl_env.components.config import TradingEnvConfig
from rl_env.trading_env import TradingEnv


def test_action_observation_shape(simple_env: TradingEnv) -> None:
    action_space = cast(Discrete, simple_env.action_space)
    obs_space = cast(Box, simple_env.observation_space)

    assert action_space.n == 3
    assert obs_space.shape == (5,)


def test_balance_and_share_values(simple_env: TradingEnv) -> None:
    obs, info = simple_env.reset()

    assert simple_env.portfolio.balance == 10_000.0
    assert simple_env.cfg.initial_balance == 10_000.0
    assert simple_env.portfolio.prev_balance == 10_000.0
    assert simple_env.portfolio.shares_held == 0.0


def test_commission_and_slippage_values(simple_env: TradingEnv) -> None:
    obs, info = simple_env.reset()

    assert simple_env.commission == 0.001
    assert simple_env.slippage == 0.0005


def test_position_open_value(simple_env: TradingEnv) -> None:
    obs, info = simple_env.reset()

    assert simple_env.postion_open_value == 2000.0


def test_invalid_action_penalty_value(simple_env: TradingEnv) -> None:
    obs, info = simple_env.reset()

    assert simple_env.invalid_action_penalty == -10.0


def test_prices_type_and_dtype(simple_env: TradingEnv) -> None:
    obs, info = simple_env.reset()

    assert isinstance(simple_env.market_data.prices, np.ndarray)
    assert simple_env.market_data.prices.dtype == np.float32


def test_no_close_column() -> None:
    data = [100, 101, 102, 99]
    times = [
        datetime.datetime(2026, 1, 1, 9, 30),
        datetime.datetime(2026, 1, 1, 9, 31),
        datetime.datetime(2026, 1, 1, 9, 32),
        datetime.datetime(2026, 1, 1, 9, 33),
    ]
    df = pd.DataFrame({"ref_close": data, "time": times})
    cfg = TradingEnvConfig(window_size=1)

    with pytest.raises(ValueError, match="close column"):
        TradingEnv(df=df, cfg=cfg)


def test_NaN_price() -> None:
    data = [100, 99, np.nan]
    times = [
        datetime.datetime(2026, 1, 1, 9, 30),
        datetime.datetime(2026, 1, 1, 9, 31),
        datetime.datetime(2026, 1, 1, 9, 32),
    ]
    df = pd.DataFrame({"close": data, "time": times})
    cfg = TradingEnvConfig(window_size=1)

    with pytest.raises(ValueError, match="'close' column contains NaN values"):
        TradingEnv(df=df, cfg=cfg)


def test_inf_price() -> None:
    data = [100, 99, np.inf]
    times = [
        datetime.datetime(2026, 1, 1, 9, 30),
        datetime.datetime(2026, 1, 1, 9, 31),
        datetime.datetime(2026, 1, 1, 9, 32),
    ]
    df = pd.DataFrame({"close": data, "time": times})
    cfg = TradingEnvConfig(window_size=1)

    with pytest.raises(ValueError, match="'close' column contains non-finite values"):
        TradingEnv(df=df, cfg=cfg)


def test_negative_price() -> None:
    data = [100, 99, -11]
    times = [
        datetime.datetime(2026, 1, 1, 9, 30),
        datetime.datetime(2026, 1, 1, 9, 31),
        datetime.datetime(2026, 1, 1, 9, 32),
    ]
    df = pd.DataFrame({"close": data, "time": times})
    cfg = TradingEnvConfig(window_size=1)

    with pytest.raises(ValueError, match="'close' column must contain strictly positive prices"):
        TradingEnv(df=df, cfg=cfg)


def test_zero_price() -> None:
    data = [100, 99, 0]
    times = [
        datetime.datetime(2026, 1, 1, 9, 30),
        datetime.datetime(2026, 1, 1, 9, 31),
        datetime.datetime(2026, 1, 1, 9, 32),
    ]
    df = pd.DataFrame({"close": data, "time": times})
    cfg = TradingEnvConfig(window_size=1)

    with pytest.raises(ValueError, match="'close' column must contain strictly positive prices"):
        TradingEnv(df=df, cfg=cfg)


def test_empty_dataframe() -> None:
    df = pd.DataFrame()
    cfg = TradingEnvConfig(window_size=1)

    with pytest.raises(ValueError, match="dataframe cannot be empty"):
        TradingEnv(df=df, cfg=cfg)
