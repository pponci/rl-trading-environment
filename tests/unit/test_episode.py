import numpy as np
import pandas as pd
import pytest

from rl_env.trading_env import TradingEnv


def test_reset_starts_at_window_size(simple_env: TradingEnv) -> None:
    obs, _ = simple_env.reset()

    assert simple_env.current_step == simple_env.window_size
    np.testing.assert_allclose(obs[:2], [100, 101])
    assert obs[2] == pytest.approx(10_000.0)
    assert obs[3] == pytest.approx(0.0)
    assert obs[4] == pytest.approx(0.0)


def test_each_step_advances_one_timestep(simple_env: TradingEnv) -> None:
    simple_env.reset()

    initial_step = simple_env.current_step

    simple_env.step(0)

    assert simple_env.current_step == initial_step + 1


def test_minimum_dataset_length_supports_one_transition() -> None:
    window_size = 3
    data = [90, 95, 98, 100]
    env = TradingEnv(df=pd.DataFrame({"close": data}), window_size=window_size)

    env.reset()
    obs, _, terminated, truncated, _ = env.step(0)

    assert terminated
    assert not truncated
    assert obs.shape == (window_size + 3,)
    assert env.observation_space.contains(obs)


def test_dataset_with_no_transition_is_rejected() -> None:
    data = [90, 95, 98, 100]
    with pytest.raises(ValueError):
        TradingEnv(df=pd.DataFrame({"close": data}), window_size=4)


def test_reset_after_termination(simple_env: TradingEnv) -> None:
    simple_env.reset()

    terminated = False
    while not terminated:
        _, _, terminated, _, _ = simple_env.step(0)

    obs, info = simple_env.reset()

    assert simple_env.current_step == simple_env.window_size
    assert simple_env.balance == pytest.approx(simple_env.initial_balance)
    assert simple_env.shares_held == pytest.approx(0.0)
    assert simple_env.prev_balance == pytest.approx(simple_env.initial_balance)

    np.testing.assert_allclose(obs[:2], [100, 101])
    assert isinstance(info, dict)


def test_terminal_observation_is_valid(simple_env: TradingEnv) -> None:
    simple_env.reset()

    terminated = False
    truncated = False
    final_obs = None

    while not terminated:
        final_obs, _, terminated, truncated, _ = simple_env.step(0)

    assert not truncated
    assert final_obs is not None
    assert final_obs.shape == (simple_env.window_size + 3,)
    assert simple_env.observation_space.contains(final_obs)


def test_reset_is_reproducible(simple_env: TradingEnv) -> None:
    obs1, _ = simple_env.reset(seed=123)
    obs2, _ = simple_env.reset(seed=123)

    np.testing.assert_array_equal(obs1, obs2)
