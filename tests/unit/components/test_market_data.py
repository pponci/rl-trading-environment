import datetime

import numpy as np
import pandas as pd
import pytest

from rl_env.components.config import TradingEnvConfig
from rl_env.components.market_data import MarketData


def make_config(
    window_size: int = 3, close_col: str = "close", time_col: str = "time"
) -> TradingEnvConfig:
    return TradingEnvConfig(window_size=window_size, close_col=close_col, time_col=time_col)


def make_df(n: int = 10, start: str = "2024-01-01", freq: str = "D") -> pd.DataFrame:
    times = pd.date_range(start=start, periods=n, freq=freq)
    prices = np.linspace(100.0, 100.0 + n, n)

    return pd.DataFrame({"time": times, "close": prices})


@pytest.fixture
def cfg() -> TradingEnvConfig:
    return make_config()


@pytest.fixture
def df() -> pd.DataFrame:
    return make_df()


def test_valid_data_builds(df: pd.DataFrame, cfg: TradingEnvConfig) -> None:
    md = MarketData(df, cfg)

    assert md.length == len(df)
    assert md.cfg is cfg
    assert md.prices.dtype == np.float32
    assert len(md.prices) == len(df)
    assert len(md.times) == len(df)


def test_dataframe_is_sorted_by_time(cfg: TradingEnvConfig) -> None:
    df = make_df(n=5)
    shuffled = df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    md = MarketData(shuffled, cfg)

    assert md.times.tolist() == sorted(md.times.tolist())


def test_internal_df_is_reset_index(cfg: TradingEnvConfig) -> None:
    df = make_df(n=5).iloc[::-1]
    md = MarketData(df, cfg)

    assert list(md.df.index) == list(range(len(df)))


def test_returns_float32_prices(df: pd.DataFrame, cfg: TradingEnvConfig) -> None:
    md = MarketData(df, cfg)

    assert md.prices.dtype == np.float32


def test_empty_dataframe_raises(cfg: TradingEnvConfig) -> None:
    with pytest.raises(ValueError, match="dataframe cannot be empty"):
        MarketData(pd.DataFrame(), cfg)


def test_missing_close_column_raises(cfg: TradingEnvConfig) -> None:
    df = make_df().drop(columns=["close"])

    with pytest.raises(ValueError, match="close column is required"):
        MarketData(df, cfg)


def test_missing_time_column_raises(cfg: TradingEnvConfig) -> None:
    df = make_df().drop(columns=["time"])

    with pytest.raises(ValueError, match="time column is required"):
        MarketData(df, cfg)


def test_window_size_too_large_raises() -> None:
    df = make_df(n=5)
    cfg = make_config(window_size=5)

    with pytest.raises(ValueError, match="window_size"):
        MarketData(df, cfg)


def test_non_numeric_close_raises(cfg: TradingEnvConfig) -> None:
    df = make_df()
    df["close"] = ["a"] * len(df)

    with pytest.raises(ValueError, match="must be numeric"):
        MarketData(df, cfg)


def test_nan_close_raises(cfg: TradingEnvConfig) -> None:
    df = make_df()
    df.loc[2, "close"] = np.nan

    with pytest.raises(ValueError, match="NaN"):
        MarketData(df, cfg)


def test_non_finite_close_raises(cfg: TradingEnvConfig) -> None:
    df = make_df()
    df.loc[3, "close"] = np.inf

    with pytest.raises(ValueError, match="non-finite"):
        MarketData(df, cfg)


def test_non_positive_close_raises(cfg: TradingEnvConfig) -> None:
    df = make_df()
    df.loc[1, "close"] = 0.0

    with pytest.raises(ValueError, match="strictly positive"):
        MarketData(df, cfg)


def test_negative_close_raises(cfg: TradingEnvConfig) -> None:
    df = make_df()
    df.loc[1, "close"] = -5.0

    with pytest.raises(ValueError, match="strictly positive"):
        MarketData(df, cfg)


def test_non_datetime_time_raises(cfg: TradingEnvConfig) -> None:
    df = make_df()
    df["time"] = df["time"].astype(str)

    with pytest.raises(ValueError, match="datetime64"):
        MarketData(df, cfg)


def test_nat_time_raises(cfg: TradingEnvConfig) -> None:
    df = make_df()
    df.loc[2, "time"] = pd.NaT

    with pytest.raises(ValueError, match="NaT"):
        MarketData(df, cfg)


def test_duplicate_timestamps_raise(cfg: TradingEnvConfig) -> None:
    df = make_df()
    df.loc[3, "time"] = df.loc[2, "time"]

    with pytest.raises(ValueError, match="duplicate"):
        MarketData(df, cfg)


def test_get_current_price_returns_correct_price(df: pd.DataFrame, cfg: TradingEnvConfig) -> None:
    md = MarketData(df, cfg)

    for step in range(len(df)):
        expected = np.float32(df["close"].iloc[step])

        assert md.get_current_price(step) == expected


def test_get_current_price_is_float(df: pd.DataFrame, cfg: TradingEnvConfig) -> None:
    md = MarketData(df, cfg)

    assert isinstance(md.get_current_price(0), (float, np.floating))


def test_get_time_returns_correct_time(df: pd.DataFrame, cfg: TradingEnvConfig) -> None:
    md = MarketData(df, cfg)
    expected = pd.date_range(start="2024-01-01", periods=len(df), freq="D")

    for step in range(len(df)):
        got = md.get_time(step)

        assert isinstance(got, (pd.Timestamp, datetime.datetime, np.datetime64))
        assert pd.Timestamp(got) == expected[step]


def test_get_time_is_monotonic(df: pd.DataFrame, cfg: TradingEnvConfig) -> None:
    md = MarketData(df, cfg)
    times = [md.get_time(i) for i in range(md.length)]

    assert times == sorted(times)


def test_get_window_shape(df: pd.DataFrame, cfg: TradingEnvConfig) -> None:
    md = MarketData(df, cfg)
    w = md.get_window(step=5)

    assert w.shape == (cfg.window_size,)


def test_get_window_contains_previous_prices(df: pd.DataFrame, cfg: TradingEnvConfig) -> None:
    md = MarketData(df, cfg)
    step = 5
    w = md.get_window(step=step)
    expected = md.prices[step - cfg.window_size : step]

    np.testing.assert_array_equal(w, expected)


def test_get_window_at_first_valid_step(df: pd.DataFrame, cfg: TradingEnvConfig) -> None:
    md = MarketData(df, cfg)
    w = md.get_window(step=cfg.window_size)

    assert w.shape == (cfg.window_size,)
    np.testing.assert_array_equal(w, md.prices[: cfg.window_size])


def test_get_scaled_window_shape(df: pd.DataFrame, cfg: TradingEnvConfig) -> None:
    md = MarketData(df, cfg)
    sw = md.get_scaled_window(step=5)

    assert sw.shape == (cfg.window_size,)


def test_get_scaled_window_last_value_is_zero(df: pd.DataFrame, cfg: TradingEnvConfig) -> None:
    md = MarketData(df, cfg)
    sw = md.get_scaled_window(step=5)

    assert sw[-1] == pytest.approx(0.0, abs=1e-6)


def test_get_scaled_window_matches_manual_formula(df: pd.DataFrame, cfg: TradingEnvConfig) -> None:
    md = MarketData(df, cfg)
    step = 6
    window = md.get_window(step=step)
    expected = window / window[-1] - 1

    np.testing.assert_allclose(md.get_scaled_window(step=step), expected, rtol=1e-6)


def test_get_scaled_window_equals_relative_change(cfg: TradingEnvConfig) -> None:
    prices = np.array([100.0, 110.0, 121.0, 133.1], dtype=np.float64)
    df = pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=len(prices), freq="D"),
            "close": prices,
        }
    )
    cfg = make_config(window_size=3)
    md = MarketData(df, cfg)

    step = 3
    sw = md.get_scaled_window(step=step)

    window = prices[step - cfg.window_size : step]
    expected = window / window[-1] - 1

    np.testing.assert_allclose(sw, expected, rtol=1e-6)


def test_get_scaled_window_zero_denominator_returns_zeros(cfg: TradingEnvConfig) -> None:
    df = make_df(n=5)
    cfg = make_config(window_size=2)
    md = MarketData(df, cfg)

    md.get_window = lambda step: np.array([1.0, 0.0])
    result = md.get_scaled_window(step=2)

    np.testing.assert_array_equal(result, np.zeros(cfg.window_size))


def test_finished_false_for_valid_steps(df: pd.DataFrame, cfg: TradingEnvConfig) -> None:
    md = MarketData(df, cfg)

    for step in range(md.length):
        assert md.finished(step) is False


def test_finished_true_at_length(df: pd.DataFrame, cfg: TradingEnvConfig) -> None:
    md = MarketData(df, cfg)

    assert md.finished(md.length) is True


def test_finished_true_beyond_length(df: pd.DataFrame, cfg: TradingEnvConfig) -> None:
    md = MarketData(df, cfg)

    assert md.finished(md.length + 100) is True
