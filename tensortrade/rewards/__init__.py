from .reward_scheme import RewardScheme
from .simple_profit import SimpleProfit
from .quantstats_rewards import QuantStatsRewards
from .risk_adjusted_returns import RiskAdjustedReturns

_registry = {
    'simple': SimpleProfit,
    'risk-adjusted': RiskAdjustedReturns,
    'quantstats': QuantStatsRewards,
}


def get(identifier: str) -> RewardScheme:
    """Gets the `RewardScheme` that matches with the identifier.

    Arguments:
        identifier: The identifier for the `RewardScheme`

    Raises:
        KeyError: if identifier is not associated with any `RewardScheme`
    """
    if identifier not in _registry.keys():
        raise KeyError(
            'Identifier {} is not associated with any `RewardScheme`.'.format(identifier))
    return _registry[identifier]()
