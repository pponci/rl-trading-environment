from rl_env.components.config import TradingEnvConfig


class Portfolio:
    def __init__(self, cfg: TradingEnvConfig) -> None:
        self.cfg = cfg

        self.shares_held = 0.0

    def reset(self) -> None:
        """
        Retrun portfolio values to initial state.
        """

        self.shares_held = 0.0

        self.balance = self.cfg.initial_balance
        self.prev_balance = self.cfg.initial_balance

    def get_position_value(self, price: float) -> float:
        """
        Compute estimated value of open position.
        """

        return self.shares_held * price

    def open_position(self, n_shares: float, total_cost: float) -> None:
        """
        Open the position at the portfolio level.
        """

        if self.balance < total_cost:
            return

        self.shares_held = n_shares
        self.balance -= total_cost

    def close_position(self, total_proceeds: float) -> None:
        """
        Close the position at portofolio level.
        """

        self.balance += total_proceeds
        self.shares_held = 0.0

    def update_prev_portfolio_balance(self) -> None:
        """
        Updates the previous portfolio balance.
        """

        self.prev_balance = self.balance

    def in_position(self) -> bool:
        """
        Checks if there is an open position.
        """

        return self.shares_held > 0
