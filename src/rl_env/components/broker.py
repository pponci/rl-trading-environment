import datetime
from typing import cast

import numpy as np

from rl_env.components.config import TradingEnvConfig
from rl_env.components.portfolio import Portfolio


class Broker:
    def __init__(self, cfg: TradingEnvConfig, portfolio: Portfolio) -> None:
        self.cfg = cfg
        self.portfolio = portfolio

        self.position_open_value = cfg.initial_balance * cfg.position_size

    def execute_buy(self, price: float) -> bool:
        """
        Open position at broker level.
        """

        if self.portfolio.in_position():
            return False

        fill_price = price * (1 + self.cfg.slippage)
        n_shares = self.position_open_value / (fill_price * (1 + self.cfg.commission))
        total_cost = n_shares * fill_price * (1 + self.cfg.commission)

        buy_status = self.portfolio.open_position(n_shares=n_shares, total_cost=total_cost)

        return buy_status

    def execute_sell(self, price: float) -> bool:
        """
        Close position at broker level.
        """

        if not self.portfolio.in_position():
            return False

        fill_price = price * (1 - self.cfg.slippage)
        gross_proceeds = self.portfolio.shares_held * fill_price
        commission_cost = gross_proceeds * self.cfg.commission

        total_proceeds = gross_proceeds - commission_cost

        self.portfolio.close_position(total_proceeds=total_proceeds)

        return True

    def can_trade(self, time: datetime.datetime | np.datetime64) -> bool:
        """
        Check if agent is still allowed to trade.
        """

        if self.cfg.overnight_holding:
            return True

        if isinstance(time, np.datetime64):
            time = cast(
                datetime.datetime,
                time.astype("datetime64[us]").astype(datetime.datetime),
            )

        return time.time() < self.cfg.market_close
