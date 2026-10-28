import numpy as np
import pandas as pd
import pytest

from rl_env.trading_env import TradingEnv


def test_obs_info_type(simple_env: TradingEnv) -> None:
    obs, info = simple_env.reset()

    assert isinstance(obs, np.ndarray)
    assert isinstance(info, dict)


def test_obs_dtype(simple_env: TradingEnv) -> None:
    obs, info = simple_env.reset()

    assert obs.dtype == np.float32


def test_too_large_window_size(simple_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="window_size"):
        TradingEnv(df=simple_df, window_size=9)


def test_window_size_plus_one(simple_df: pd.DataFrame) -> None:
    env = TradingEnv(df=simple_df, window_size=8)

    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(0)

    assert terminated is True
    assert truncated is False


def test_window_size_plus_two(simple_df: pd.DataFrame) -> None:
    env = TradingEnv(df=simple_df, window_size=7)

    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(0)

    assert terminated is False
    assert truncated is False


def test_initial_price_window(simple_env: TradingEnv, simple_df: pd.DataFrame) -> None:
    obs, _ = simple_env.reset()
    prices = simple_df["close"].to_numpy().astype(np.float32)

    np.testing.assert_array_equal(obs[:2], prices[:2])


def test_observation_after_buy(simple_env: TradingEnv) -> None:
    simple_env.reset()
    obs, _, _, _, _ = simple_env.step(1)

    np.testing.assert_allclose(obs[:2], [101, 100])
    assert obs[-3] == pytest.approx(8_000.0)
    assert obs[-2] == pytest.approx(19.97003555, rel=1e-4)
    assert obs[-1] == pytest.approx(2_036.94362, rel=1e-4)


def test_observation_window_advances(simple_env: TradingEnv) -> None:
    simple_env.reset()
    obs, _, _, _, _ = simple_env.step(0)

    np.testing.assert_allclose(obs[:2], [101, 100])
    assert simple_env.observation_space.contains(obs)
