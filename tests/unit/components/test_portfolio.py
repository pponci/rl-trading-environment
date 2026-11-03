import pytest

from rl_env.components.config import TradingEnvConfig
from rl_env.components.portfolio import Portfolio


@pytest.fixture
def portfolio() -> Portfolio:
    cfg = TradingEnvConfig(
        initial_balance=10_000.0,
        commission=0.0,
        slippage=0.0,
        position_size=0.20,
        window_size=2,
    )

    p = Portfolio(cfg)
    p.reset()

    return p


def test_reset_sets_initial_state(portfolio: Portfolio) -> None:
    portfolio.balance = 123.0
    portfolio.prev_balance = 45.0
    portfolio.shares_held = 7.0

    portfolio.reset()

    assert portfolio.balance == 10_000.0
    assert portfolio.prev_balance == 10_000.0
    assert portfolio.shares_held == 0.0


def test_get_position_value(portfolio: Portfolio) -> None:
    portfolio.shares_held = 10.0

    assert portfolio.get_position_value(price=15.0) == pytest.approx(150.0)


def test_get_position_value_zero_shares(portfolio: Portfolio) -> None:
    assert portfolio.get_position_value(price=15.0) == pytest.approx(0.0)


def test_open_position_success(portfolio: Portfolio) -> None:
    ok = portfolio.open_position(n_shares=20.0, total_cost=2_000.0)

    assert ok is True
    assert portfolio.shares_held == pytest.approx(20.0)
    assert portfolio.balance == pytest.approx(8_000.0)


def test_open_position_insufficient_balance(portfolio: Portfolio) -> None:
    portfolio.balance = 100.0
    balance_before = portfolio.balance
    shares_before = portfolio.shares_held

    ok = portfolio.open_position(n_shares=20.0, total_cost=2_000.0)

    assert ok is False
    assert portfolio.balance == pytest.approx(balance_before)
    assert portfolio.shares_held == pytest.approx(shares_before)


def test_open_position_overwrites_shares(portfolio: Portfolio) -> None:
    portfolio.open_position(n_shares=10.0, total_cost=1_000.0)
    portfolio.open_position(n_shares=5.0, total_cost=500.0)

    assert portfolio.shares_held == pytest.approx(5.0)
    assert portfolio.balance == pytest.approx(8_500.0)


def test_close_position(portfolio: Portfolio) -> None:
    portfolio.shares_held = 20.0
    portfolio.balance = 8_000.0

    portfolio.close_position(total_proceeds=2_040.0)

    assert portfolio.balance == pytest.approx(10_040.0)
    assert portfolio.shares_held == pytest.approx(0.0)


def test_get_return(portfolio: Portfolio) -> None:
    portfolio.prev_balance = 10_000.0
    portfolio.balance = 10_040.0

    assert portfolio.get_return() == pytest.approx(40.0)


def test_get_percentage_return(portfolio: Portfolio) -> None:
    portfolio.prev_balance = 10_000.0
    portfolio.balance = 10_040.0

    assert portfolio.get_percentage_return() == pytest.approx(0.4)


def test_get_percentage_return_zero_prev_balance(portfolio: Portfolio) -> None:
    portfolio.prev_balance = 0.0
    portfolio.balance = 10_040.0

    assert portfolio.get_percentage_return() == pytest.approx(0.0)


def test_get_percentage_return_negative_prev_balance(portfolio: Portfolio) -> None:
    portfolio.prev_balance = -10.0
    portfolio.balance = 10_040.0

    assert portfolio.get_percentage_return() == pytest.approx(0.0)


def test_update_prev_portfolio_balance(portfolio: Portfolio) -> None:
    portfolio.balance = 8_000.0
    portfolio.update_prev_portfolio_balance()

    assert portfolio.prev_balance == pytest.approx(8_000.0)


def test_in_position_false_when_empty(portfolio: Portfolio) -> None:
    assert portfolio.in_position() is False


def test_in_position_true_when_holding(portfolio: Portfolio) -> None:
    portfolio.shares_held = 1.0

    assert portfolio.in_position() is True
