import pytest

from rl_env.components.config import TradingEnvConfig


def test_defaults_are_valid() -> None:
    cfg = TradingEnvConfig()

    assert cfg.window_size == 30
    assert cfg.initial_balance == 10_000
    assert cfg.commission == 0.001
    assert cfg.slippage == 0.0005
    assert cfg.position_size == 0.20
    assert cfg.invalid_action_penalty == -10.0
    assert cfg.close_col == "close"
    assert cfg.time_col == "time"
    assert cfg.stop_loss == 0.05


def test_costum_valid_values() -> None:
    cfg = TradingEnvConfig(
        window_size=1,
        initial_balance=1.0,
        commission=0.0,
        slippage=0.0,
        position_size=1.0,
        invalid_action_penalty=-2,
        close_col="ref_close",
        time_col="ref_time",
        stop_loss=None,
    )

    assert cfg.window_size == 1
    assert cfg.initial_balance == 1.0
    assert cfg.commission == 0.0
    assert cfg.slippage == 0.0
    assert cfg.position_size == 1.0
    assert cfg.invalid_action_penalty == -2
    assert cfg.close_col == "ref_close"
    assert cfg.time_col == "ref_time"
    assert cfg.stop_loss is None


@pytest.mark.parametrize("bad", [0, -1, -100])
def test_window_size_must_be_positive(bad: int) -> None:
    with pytest.raises(ValueError, match="window_size"):
        TradingEnvConfig(window_size=bad)


def test_window_size_minimum_valid_value() -> None:
    assert TradingEnvConfig(window_size=1).window_size == 1


@pytest.mark.parametrize("bad", [0.0, -0.01, -1000])
def test_initial_balance_must_be_postive(bad: float) -> None:
    with pytest.raises(ValueError, match="initial_balance"):
        TradingEnvConfig(initial_balance=bad)


def test_initial_ballance_small_postive_value() -> None:
    assert TradingEnvConfig(initial_balance=1e-9).initial_balance == 1e-9


@pytest.mark.parametrize("bad", [-0.01, -1.0, 1.01, 2.0, 100.0])
def test_commission_reject_out_of_range(bad: float) -> None:
    with pytest.raises(ValueError, match="commission"):
        TradingEnvConfig(commission=bad)


@pytest.mark.parametrize("boundary", [0.0, 1.0])
def test_commission_accept_inclusive_boundaries(boundary: float) -> None:
    cfg = TradingEnvConfig(commission=boundary)

    assert cfg.commission == boundary


def test_commission_accept_typical_value() -> None:
    cfg = TradingEnvConfig(commission=0.5)

    assert cfg.commission == 0.5


@pytest.mark.parametrize("bad", [-0.01, -1.0, 1.01, 2.0, 100.0])
def test_slippage_reject_out_of_range(bad: float) -> None:
    with pytest.raises(ValueError, match="slippage"):
        TradingEnvConfig(slippage=bad)


@pytest.mark.parametrize("boundary", [0.0, 1.0])
def test_slippage_accept_inclusive_boundaries(boundary: float) -> None:
    cfg = TradingEnvConfig(slippage=boundary)

    assert cfg.slippage == boundary


def test_slippage_accept_typical_value() -> None:
    cfg = TradingEnvConfig(slippage=0.5)

    assert cfg.slippage == 0.5


@pytest.mark.parametrize("bad", [-0.01, -1.0, 1.01, 2.0, 100.0])
def test_position_size_reject_out_of_range(bad: float) -> None:
    with pytest.raises(ValueError, match="position_size"):
        TradingEnvConfig(position_size=bad)


@pytest.mark.parametrize("boundary", [0.0, 1.0])
def test_position_size_accept_inclusive_boundaries(boundary: float) -> None:
    cfg = TradingEnvConfig(position_size=boundary)

    assert cfg.position_size == boundary


def test_position_size_accept_typical_value() -> None:
    cfg = TradingEnvConfig(position_size=0.5)

    assert cfg.position_size == 0.5


@pytest.mark.parametrize("bad", [-0.01, -1.0, 1.01, 2.0, 100.0])
def test_stop_loss_reject_out_of_range(bad: float) -> None:
    with pytest.raises(ValueError, match="stop_loss"):
        TradingEnvConfig(stop_loss=bad)


@pytest.mark.parametrize("boundary", [0.0, 1.0])
def test_stop_loss_accept_inclusive_boundaries(boundary: float) -> None:
    cfg = TradingEnvConfig(stop_loss=boundary)

    assert cfg.stop_loss == boundary


def test_stop_loss_accept_typical_value() -> None:
    cfg = TradingEnvConfig(stop_loss=0.5)

    assert cfg.stop_loss == 0.5
