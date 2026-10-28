import pytest

from rl_env.trading_env import TradingEnv


def test_hold_while_flat(simple_env: TradingEnv) -> None:
    simple_env.reset()

    balance_before = simple_env.balance

    _, reward, terminated, truncated, _ = simple_env.step(0)

    assert reward == pytest.approx(0.0)
    assert simple_env.balance == pytest.approx(balance_before)
    assert simple_env.shares_held == pytest.approx(0.0)
    assert not terminated
    assert not truncated


def test_hold_with_open_position_produces_zero_reward(simple_env: TradingEnv) -> None:
    simple_env.reset()
    simple_env.step(1)

    shares_before = simple_env.shares_held
    balance_before = simple_env.balance

    _, reward, _, _, _ = simple_env.step(0)

    assert reward == pytest.approx(0.0)
    assert simple_env.shares_held == pytest.approx(shares_before)
    assert simple_env.balance == pytest.approx(balance_before)


def test_profitable_sale_reward(simple_env: TradingEnv) -> None:
    simple_env.reset()
    simple_env.step(1)
    _, reward, _, _, _ = simple_env.step(2)

    assert reward == pytest.approx(33.88867, rel=1e-4)
    assert simple_env.prev_balance == pytest.approx(10_033.890, rel=1e-4)


def test_losing_sale_reward(simple_env: TradingEnv) -> None:
    simple_env.reset()
    simple_env.step(1)
    simple_env.step(0)
    simple_env.step(0)
    _, reward, _, _, _ = simple_env.step(2)

    assert reward == pytest.approx(-105.69141, rel=1e-4)
    assert simple_env.balance == pytest.approx(9_894.3086, rel=1e-4)
    assert simple_env.prev_balance == pytest.approx(9_894.3086, rel=1e-4)


def test_rewards_for_two_completed_trades(simple_env: TradingEnv) -> None:
    simple_env.reset()

    simple_env.step(1)
    _, reward_1, _, _, _ = simple_env.step(2)

    assert reward_1 == pytest.approx(33.88867, rel=1e-4)

    simple_env.step(1)
    _, reward_2, _, _, _ = simple_env.step(2)

    assert reward_2 == pytest.approx(-105.69141, rel=1e-4)


@pytest.mark.parametrize("action", [2])
def test_invalid_sell_penalty_when_flat(simple_env: TradingEnv, action: int) -> None:
    simple_env.reset()
    _, reward, _, _, _ = simple_env.step(action)

    assert reward == pytest.approx(simple_env.invalid_action_penalty)


def test_buy_hold_sell_sequence(simple_env: TradingEnv) -> None:
    simple_env.reset()

    _, buy_reward, _, _, _ = simple_env.step(1)
    shares_after_buy = simple_env.shares_held

    _, hold_reward, _, _, _ = simple_env.step(0)

    assert buy_reward == pytest.approx(0.0)
    assert hold_reward == pytest.approx(0.0)
    assert simple_env.shares_held == pytest.approx(shares_after_buy)

    _, sell_reward, _, _, _ = simple_env.step(2)

    assert sell_reward == pytest.approx(-5.99121, rel=1e-3)
    assert simple_env.shares_held == pytest.approx(0.0)


def test_buy_buy_sell_sequence(simple_env: TradingEnv) -> None:
    simple_env.reset()
    simple_env.step(1)

    shares_before = simple_env.shares_held
    balance_before = simple_env.balance

    _, reward, _, _, _ = simple_env.step(1)

    assert reward == pytest.approx(simple_env.invalid_action_penalty)
    assert simple_env.shares_held == pytest.approx(shares_before)
    assert simple_env.balance == pytest.approx(balance_before)

    simple_env.step(2)
    assert simple_env.shares_held == pytest.approx(0.0)


def test_sell_buy_sell_sequence(simple_env: TradingEnv) -> None:
    simple_env.reset()

    _, reward_flat_sell, _, _, _ = simple_env.step(2)
    assert reward_flat_sell == pytest.approx(simple_env.invalid_action_penalty)

    simple_env.step(1)
    _, first_sell_reward, _, _, _ = simple_env.step(2)
    assert first_sell_reward == pytest.approx(-45.08984, rel=1e-3)

    simple_env.step(1)
    _, second_sell_reward, _, _, _ = simple_env.step(2)

    assert second_sell_reward == pytest.approx(14.998, rel=1e-3)
    assert simple_env.shares_held == pytest.approx(0.0)


@pytest.mark.parametrize("action", [-1, 3, 100])
def test_out_of_range_action_is_rejected(simple_env: TradingEnv, action: int) -> None:
    simple_env.reset()

    with pytest.raises((ValueError, AssertionError)):
        simple_env.step(action)
