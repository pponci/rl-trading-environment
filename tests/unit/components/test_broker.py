import datetime

import numpy as np
import pytest

from rl_env.components.broker import Broker
from rl_env.components.config import TradingEnvConfig
from rl_env.components.portfolio import Portfolio


@pytest.fixture
def cfg() -> TradingEnvConfig:
    return TradingEnvConfig(
        initial_balance=10_000.0,
        position_size=0.5,
        slippage=0.01,
        commission=0.001,
        overnight_holding=False,
        market_close=datetime.time(16, 0),
    )


@pytest.fixture
def portfolio(cfg: TradingEnvConfig) -> Portfolio:
    p = Portfolio(cfg)
    p.reset()

    return p


@pytest.fixture
def broker(cfg: TradingEnvConfig, portfolio: Portfolio) -> Broker:
    return Broker(cfg, portfolio)


def test_position_open_value_computed(broker: Broker, cfg: TradingEnvConfig) -> None:
    expected = cfg.initial_balance * cfg.position_size

    assert broker.position_open_value == pytest.approx(expected)


def test_stores_cfg_and_portfolio(
    broker: Broker, cfg: TradingEnvConfig, portfolio: Portfolio
) -> None:
    assert broker.cfg is cfg
    assert broker.portfolio is portfolio


def test_buy_returns_false_when_already_in_position(broker: Broker, portfolio: Portfolio) -> None:
    portfolio.open_position(n_shares=10.0, total_cost=1_000.0)
    balance_before = portfolio.balance
    shares_before = portfolio.shares_held

    assert broker.execute_buy(price=100.0) is False
    assert portfolio.balance == pytest.approx(balance_before)
    assert portfolio.shares_held == pytest.approx(shares_before)


def test_buy_updates_portfolio_state(
    broker: Broker, portfolio: Portfolio, cfg: TradingEnvConfig
) -> None:
    price = 100.0
    result = broker.execute_buy(price=price)

    assert result is True
    assert portfolio.in_position() is True
    assert portfolio.shares_held > 0
    assert portfolio.balance == pytest.approx(cfg.initial_balance - broker.position_open_value)


def test_buy_shares_match_math(broker: Broker, portfolio: Portfolio, cfg: TradingEnvConfig) -> None:
    price = 100.0
    broker.execute_buy(price=price)

    fill_price = price * (1 + cfg.slippage)
    expected_shares = broker.position_open_value / (fill_price * (1 + cfg.commission))

    assert portfolio.shares_held == pytest.approx(expected_shares)


def test_buy_slippage_increases_fill_price(
    broker: Broker, portfolio: Portfolio, cfg: TradingEnvConfig
) -> None:
    price = 100.0
    broker.execute_buy(price=price)

    fill_price = broker.position_open_value / (portfolio.shares_held * (1 + cfg.commission))

    assert fill_price > price
    assert fill_price == pytest.approx(price * (1 + cfg.slippage))


def test_buy_returns_false_when_insufficient_balance(cfg: TradingEnvConfig) -> None:
    cfg.initial_balance = 1_000.0
    cfg.position_size = 2.0
    p = Portfolio(cfg)
    p.reset()
    b = Broker(cfg, p)

    balance_before = p.balance
    result = b.execute_buy(price=100.0)

    assert result is False
    assert p.shares_held == 0.0
    assert p.balance == pytest.approx(balance_before)


def test_buy_zero_slippage_and_commission(cfg: TradingEnvConfig, portfolio: Portfolio) -> None:
    cfg.slippage = 0.0
    cfg.commission = 0.0
    b = Broker(cfg, portfolio)

    b.execute_buy(price=50.0)

    expected_shares = b.position_open_value / 50.0

    assert portfolio.shares_held == pytest.approx(expected_shares)
    assert portfolio.balance == pytest.approx(cfg.initial_balance - b.position_open_value)


def test_sell_returns_false_when_not_in_position(
    broker: Broker, portfolio: Portfolio, cfg: TradingEnvConfig
) -> None:
    balance_before = portfolio.balance

    assert broker.execute_sell(price=100.0) is False
    assert portfolio.balance == pytest.approx(balance_before)
    assert portfolio.shares_held == 0.0


def test_sell_closes_position_and_credits_proceeds(
    broker: Broker, portfolio: Portfolio, cfg: TradingEnvConfig
) -> None:
    entry_price = 100.0
    exit_price = 120.0

    broker.execute_buy(price=entry_price)
    shares = portfolio.shares_held
    balance_after_buy = portfolio.balance

    result = broker.execute_sell(price=exit_price)

    fill_price = exit_price * (1 - cfg.slippage)
    gross = shares * fill_price
    expected_proceeds = gross * (1 - cfg.commission)

    assert result is True
    assert portfolio.in_position() is False
    assert portfolio.shares_held == 0.0
    assert portfolio.balance == pytest.approx(balance_after_buy + expected_proceeds)


def test_sell_slippage_reduces_proceeds(cfg: TradingEnvConfig, portfolio: Portfolio) -> None:
    cfg.slippage = 0.01
    b = Broker(cfg, portfolio)
    b.execute_buy(price=100.0)
    b.execute_sell(price=100.0)
    balance_with_slip = portfolio.balance

    cfg.slippage = 0.0
    p2 = Portfolio(cfg)
    p2.reset()
    b2 = Broker(cfg, p2)
    b2.execute_buy(price=100.0)
    b2.execute_sell(price=100.0)
    balance_no_slip = p2.balance

    assert balance_with_slip < balance_no_slip


def test_sell_commission_reduces_proceeds(cfg: TradingEnvConfig, portfolio: Portfolio) -> None:
    cfg.commission = 0.001
    b = Broker(cfg, portfolio)
    b.execute_buy(price=100.0)
    b.execute_sell(price=100.0)
    balance_with_comm = portfolio.balance

    cfg.commission = 0.0
    p2 = Portfolio(cfg)
    p2.reset()
    b2 = Broker(cfg, p2)
    b2.execute_buy(price=100.0)
    b2.execute_sell(price=100.0)
    balance_no_comm = p2.balance

    assert balance_with_comm < balance_no_comm


def test_can_trade_overnight_holding_always_true(broker: Broker, cfg: TradingEnvConfig) -> None:
    cfg.overnight_holding = True
    late = datetime.datetime(2024, 1, 1, 23, 59)

    assert broker.can_trade(late) is True


def test_can_trade_before_market_close_true(broker: Broker, cfg: TradingEnvConfig) -> None:
    cfg.market_close = datetime.time(16, 0)
    t = datetime.datetime(2024, 1, 1, 15, 59)

    assert broker.can_trade(t) is True


def test_can_trade_exactly_at_market_close_false(broker: Broker, cfg: TradingEnvConfig) -> None:
    cfg.market_close = datetime.time(16, 0)
    t = datetime.datetime(2024, 1, 1, 16, 0)

    assert broker.can_trade(t) is False


def test_can_trade_after_market_close_false(broker: Broker, cfg: TradingEnvConfig) -> None:
    cfg.market_close = datetime.time(16, 0)
    t = datetime.datetime(2024, 1, 1, 16, 1)

    assert broker.can_trade(t) is False


def test_can_trade_accepts_numpy_datetime64(broker: Broker, cfg: TradingEnvConfig) -> None:
    cfg.market_close = datetime.time(16, 0)
    before = np.datetime64("2024-01-01T15:00:00")
    after = np.datetime64("2024-01-01T17:00:00")

    assert broker.can_trade(before) is True
    assert broker.can_trade(after) is False


def test_can_trade_numpy_datetime_conversion_equivalence(
    broker: Broker, cfg: TradingEnvConfig
) -> None:
    cfg.market_close = datetime.time(16, 0)
    dt = datetime.datetime(2024, 6, 15, 12, 30)
    np_dt = np.datetime64(dt)

    assert broker.can_trade(dt) == broker.can_trade(np_dt)
