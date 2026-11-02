from dataclasses import dataclass


@dataclass
class TradingEnvConfig:
    window_size: int = 30

    initial_balance: float = 10_000
    commission: float = 0.001
    slippage: float = 0.0005
    position_size: float = 0.20

    invalid_action_penalty: float = -10.0
