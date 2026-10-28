from typing import cast

import numpy as np
import pandas as pd
import pytest
from gymnasium.spaces import Box, Discrete

from rl_env.trading_env import TradingEnv


def test_action_observation_shape(simple_env: TradingEnv) -> None:
    action_space = cast(Discrete, simple_env.action_space)
    obs_space = cast(Box, simple_env.observation_space)

    assert action_space.n == 3
    assert obs_space.shape == (5,)


def test_balance_and_share_values(simple_env: TradingEnv) -> None:
    obs, info = simple_env.reset()

    assert simple_env.balance == 10_000.0
    assert simple_env.initial_balance == 10_000.0
    assert simple_env.prev_balance == 10_000.0
    assert simple_env.shares_held == 0.0


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

    assert isinstance(simple_env.prices, np.ndarray)
    assert simple_env.prices.dtype == np.float32


def test_no_close_column() -> None:
    df = pd.DataFrame({"ref_close": [100, 101, 102, 99]})

    with pytest.raises(ValueError, match="close column"):
        TradingEnv(df=df, window_size=1)


def test_NaN_price() -> None:
    df = pd.DataFrame({"close": [100, 99, np.nan]})

    with pytest.raises(ValueError, match="invalid prices"):
        TradingEnv(df=df, window_size=1)


def test_inf_price() -> None:
    df = pd.DataFrame({"close": [100, 99, np.inf]})

    with pytest.raises(ValueError, match="invalid prices"):
        TradingEnv(df=df, window_size=1)


def test_negative_price() -> None:
    df = pd.DataFrame({"close": [100, 99, -11]})

    with pytest.raises(ValueError, match="invalid prices"):
        TradingEnv(df=df, window_size=1)


def test_zero_price() -> None:
    df = pd.DataFrame({"close": [100, 99, 0]})

    with pytest.raises(ValueError, match="invalid prices"):
        TradingEnv(df=df, window_size=1)


def test_empty_dataframe() -> None:
    df = pd.DataFrame()

    with pytest.raises(ValueError, match="dataframe cannot be empty"):
        TradingEnv(df=df, window_size=1)
