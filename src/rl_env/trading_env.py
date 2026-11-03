from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd

from rl_env.components.broker import Broker
from rl_env.components.config import TradingEnvConfig
from rl_env.components.market_data import MarketData
from rl_env.components.portfolio import Portfolio


class TradingEnv(gym.Env):
    def __init__(self, df: pd.DataFrame, cfg: TradingEnvConfig) -> None:

        super().__init__()

        self.cfg = cfg
        self.portfolio = Portfolio(cfg=cfg)
        self.market_data = MarketData(df=df, cfg=cfg)
        self.broker = Broker(cfg=cfg, portfolio=self.portfolio)

        n_portfolio_features = 3

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(cfg.window_size + n_portfolio_features,),
            dtype=np.float32,
        )

        self.action_space = gym.spaces.Discrete(3)

        self.invalid_action_penalty = cfg.invalid_action_penalty

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Reset enviroment to starting state, needed before new episode.
        """

        super().reset(seed=seed)

        self.portfolio.reset()

        self.current_step = self.cfg.window_size

        return self._get_obs(self.current_step), {}

    def _get_obs(self, current_step: int) -> np.ndarray:
        """
        Build the observation from the enviroment.
        """

        window = self.market_data.get_scaled_window(step=current_step)

        price = self.market_data.get_current_price(current_step)
        position_value = self.portfolio.get_position_value(price=price)

        portfolio_features = np.array(
            [self.portfolio.balance, self.portfolio.shares_held, position_value]
        )

        return np.concatenate([window, portfolio_features]).astype(np.float32)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Exections required for each step.
        """

        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action!r}, expected 0, 1, 2")

        price = self.market_data.get_current_price(self.current_step)
        time = self.market_data.get_time(step=self.current_step)

        reward = 0.0

        if not self.broker.can_trade(time=time):
            sell_status = self.broker.execute_sell(price=price)

            if sell_status:
                reward = self.portfolio.get_percentage_return()
                self.portfolio.update_prev_portfolio_balance()

        if action == 0:
            reward = 0.0

        elif action == 1:
            buy_status = self.broker.execute_buy(price=price)

            if buy_status:
                reward = 0.0

            else:
                reward = self.invalid_action_penalty

        elif action == 2:
            sell_status = self.broker.execute_sell(price=price)

            if sell_status:
                reward = self.portfolio.get_percentage_return()
                self.portfolio.update_prev_portfolio_balance()

            else:
                reward = self.invalid_action_penalty

        self.current_step += 1

        terminated = self.market_data.finished(step=self.current_step)
        truncated = False

        info: dict[str, Any] = {}

        obs_step = min(self.current_step, len(self.market_data.prices) - 1)

        return self._get_obs(obs_step), reward, terminated, truncated, info
