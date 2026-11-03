import datetime
from dataclasses import dataclass


@dataclass
class TradingEnvConfig:
    window_size: int = 30

    initial_balance: float = 10_000
    commission: float = 0.001
    slippage: float = 0.0005
    position_size: float = 0.20

    invalid_action_penalty: float = -10.0

    close_col: str = "close"
    time_col: str = "time"

    market_close: datetime.time = datetime.time(15, 55)
    overnight_holding: bool = False

    def __post_init__(self) -> None:
        if self.window_size <= 0:
            raise ValueError("window_size must be > 0")

        if self.initial_balance <= 0:
            raise ValueError("initial_balance must be > 0")

        if not (0.0 <= self.commission <= 1.0):
            raise ValueError("commission must be a percentage")

        if not (0.0 <= self.slippage <= 1.0):
            raise ValueError("slippage must be a percentage")

        if not (0.0 <= self.position_size <= 1.0):
            raise ValueError("position_size must be a percentage")
