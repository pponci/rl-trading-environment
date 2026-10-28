from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd


class TradingEnv(gym.Env):
    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = 30,
        initial_balance: float = 10_000.00,
        commission: float = 0.001,
        slippage: float = 0.0005,
        position_size: float = 0.20,
        invalid_action_penalty: float = -10.0,
    ) -> None:

        super().__init__()

        if df.empty:
            raise ValueError("dataframe cannot be empty")

        if "close" not in df.columns:
            raise ValueError("close column is required.")

        if window_size >= len(df):
            raise ValueError(
                f"window_size : {window_size} musst be a smaller number than",
                f"the lenght of the dataset : {len(df)}",
            )

        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.prices = df["close"].to_numpy(dtype=np.float32)

        if not np.all(np.isfinite(self.prices)) or np.any(self.prices <= 0):
            raise ValueError("invalid prices present in the dataset")

        self.commission = commission
        self.slippage = slippage
        self.postion_open_value = initial_balance * position_size

        self.initial_balance = initial_balance
        n_portfolio_features = 3

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(window_size + n_portfolio_features,), dtype=np.float32
        )

        self.action_space = gym.spaces.Discrete(3)

        self.invalid_action_penalty = invalid_action_penalty

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Reset enviroment to starting state, needed before new episode.
        """

        super().reset(seed=seed)

        self.current_step = self.window_size

        self.balance = self.initial_balance
        self.shares_held = 0.0

        self.prev_balance = self.initial_balance

        return self._get_obs(self.current_step), {}

    def _current_price(self, current_step: int) -> float:
        """
        Get price at step.
        """

        return self.prices[current_step]

    def _postion_value(self, price: float) -> float:
        """
        Compute estimated value of open postion.
        """

        return self.shares_held * price

    def _get_obs(self, current_step: int) -> np.ndarray:
        """
        Build the observation from the enviroment.
        """

        window = self.prices[current_step - self.window_size : current_step]

        price = self._current_price(current_step)
        position_value = self._postion_value(price=price)

        portfolio_features = np.array([self.balance, self.shares_held, position_value])

        return np.concatenate([window, portfolio_features]).astype(np.float32)

    def _buy(self, price: float) -> None:
        """
        Open postion.
        """

        fill_price = price * (1 + self.slippage)

        n_shares = self.postion_open_value / (fill_price * (1 + self.commission))

        total_cost = n_shares * fill_price * (1 + self.commission)

        # to do: for now insuffucent balance fails silenttly, same as other
        # failed operation agent should know.
        if self.balance < total_cost:
            return

        self.shares_held = n_shares
        self.balance -= total_cost

    def _sell(self, price: float) -> None:
        """
        Close postion.
        """

        fill_price = price * (1 - self.slippage)
        gross_proceeds = self.shares_held * fill_price
        commission_cost = gross_proceeds * self.commission

        self.balance += gross_proceeds - commission_cost
        self.shares_held = 0.0

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Exections required for each step.
        """

        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action!r}, expected 0, 1, 2")

        price = self._current_price(self.current_step)

        reward = 0.0

        if action == 0:
            reward = 0.0

        elif action == 1:
            if self.shares_held <= 0:
                self._buy(price=price)
                reward = 0.0

            else:
                reward = self.invalid_action_penalty

        elif action == 2:
            if self.shares_held > 0:
                self._sell(price=price)
                reward = self.balance - self.prev_balance
                self.prev_balance = self.balance

            else:
                reward = self.invalid_action_penalty

        self.current_step += 1

        terminated = self.current_step >= len(self.prices)
        truncated = False

        info: dict[str, Any] = {}

        obs_step = min(self.current_step, len(self.prices) - 1)

        return self._get_obs(obs_step), reward, terminated, truncated, info
