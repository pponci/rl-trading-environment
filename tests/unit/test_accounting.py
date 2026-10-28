import pandas as pd
import pytest

from rl_env.trading_env import TradingEnv


def test_buy_without_costs(simple_df: pd.DataFrame) -> None:
    env = TradingEnv(
        df=simple_df,
        commission=0.0,
        slippage=0.0,
        window_size=2,
    )

    env.reset()
    _, reward, _, _, _ = env.step(1)

    assert env.shares_held == pytest.approx(20.0)
    assert env.balance == pytest.approx(8_000.0)
    assert reward == pytest.approx(0.0)


def test_buy_with_commission(simple_df: pd.DataFrame) -> None:
    env = TradingEnv(df=simple_df, commission=0.001, slippage=0.0, window_size=2)

    env.reset()
    env.step(1)

    expected_shares = 2_000 / (100 * 1.001)
    expected_cost = expected_shares * 100 * 1.001

    assert env.shares_held == pytest.approx(expected_shares)
    assert env.balance == pytest.approx(10_000 - expected_cost)


def test_buy_with_slippage(simple_df: pd.DataFrame) -> None:
    env = TradingEnv(
        df=simple_df,
        commission=0.0,
        slippage=0.001,
        window_size=2,
    )

    env.reset()
    env.step(1)

    fill_price = 100 * 1.001
    expected_shares = 2_000 / fill_price

    assert env.shares_held == pytest.approx(expected_shares)
    assert env.balance == pytest.approx(8_000.0)


def test_buy_with_commission_and_slippage(simple_df: pd.DataFrame) -> None:
    env = TradingEnv(df=simple_df, commission=0.001, slippage=0.001, window_size=2)

    env.reset()
    env.step(1)

    fill_price = 100 * 1.001
    expected_shares = 2_000 / (fill_price * 1.001)

    assert env.shares_held == pytest.approx(expected_shares)
    assert env.balance == pytest.approx(8_000.0)


def test_unaffordable_buy_does_not_change_portfolio(simple_df: pd.DataFrame) -> None:
    env = TradingEnv(
        df=simple_df,
        initial_balance=1000.0,
        position_size=0.20,
        window_size=2,
    )

    env.reset()
    env.balance = 100.0
    balance_before = env.balance
    shares_before = env.shares_held

    _, reward, _, _, _ = env.step(1)

    assert env.balance == pytest.approx(balance_before)
    assert env.shares_held == pytest.approx(shares_before)
    assert reward == pytest.approx(0.0)


def test_buy_while_holding_is_rejected(simple_env: TradingEnv) -> None:
    simple_env.reset()
    simple_env.step(1)

    balance_before = simple_env.balance
    shares_before = simple_env.shares_held

    _, reward, _, _, _ = simple_env.step(1)

    assert simple_env.balance == pytest.approx(balance_before)
    assert simple_env.shares_held == pytest.approx(shares_before)
    assert reward == pytest.approx(simple_env.invalid_action_penalty)


def test_sell_without_costs(simple_df: pd.DataFrame) -> None:
    env = TradingEnv(
        df=simple_df,
        commission=0.0,
        slippage=0.0,
        window_size=2,
    )

    env.reset()
    env.step(1)
    _, reward, _, _, _ = env.step(2)

    assert env.shares_held == pytest.approx(0.0)
    assert env.balance == pytest.approx(10_040.0)
    assert reward == pytest.approx(40.0)


def test_sell_with_commission(simple_df: pd.DataFrame) -> None:
    commission = 0.001
    env = TradingEnv(
        df=simple_df,
        commission=commission,
        slippage=0.0,
        window_size=2,
    )

    env.reset()
    env.step(1)
    _, reward, _, _, _ = env.step(2)

    expected_shares = 2_000 / (100 * (1 + commission))
    expected_final_balance = 10_000 - 2_000 + expected_shares * 102 * (1 - commission)

    assert env.shares_held == pytest.approx(0.0)
    assert env.balance == pytest.approx(expected_final_balance, rel=1e-5)
    assert reward == pytest.approx(expected_final_balance - 10_000, rel=1e-5)


def test_sell_with_slippage(simple_df: pd.DataFrame) -> None:
    slippage = 0.001
    env = TradingEnv(
        df=simple_df,
        commission=0.0,
        slippage=slippage,
        window_size=2,
    )

    env.reset()
    env.step(1)
    _, reward, _, _, _ = env.step(2)

    buy_fill = 100 * (1 + slippage)
    sell_fill = 102 * (1 - slippage)
    expected_shares = 2_000 / buy_fill
    expected_balance = 8_000 + expected_shares * sell_fill

    assert env.shares_held == pytest.approx(0.0)
    assert env.balance == pytest.approx(expected_balance, rel=1e-5)
    assert reward == pytest.approx(expected_balance - 10_000, rel=1e-5)


def test_sell_while_flat_is_rejected(simple_env: TradingEnv) -> None:
    simple_env.reset()

    balance_before = simple_env.balance
    shares_before = simple_env.shares_held

    _, reward, _, _, _ = simple_env.step(2)

    assert simple_env.balance == pytest.approx(balance_before)
    assert simple_env.shares_held == pytest.approx(shares_before)
    assert reward == pytest.approx(simple_env.invalid_action_penalty)


def test_sell_closes_entire_position(simple_env: TradingEnv) -> None:
    simple_env.reset()
    simple_env.step(1)
    simple_env.step(2)

    assert simple_env.shares_held == pytest.approx(0.0)
