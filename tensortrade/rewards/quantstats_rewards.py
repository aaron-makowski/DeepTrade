# Copyright 2019 The TensorTrade Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import pandas as pd
import numpy as np
import quantstats as qs

from typing import Callable

from tensortrade.rewards import RewardScheme


class QuantStatsRewards(RewardScheme):
    """A reward scheme that uses a combination of QuantStats metrics."""
    def __init__(self, window_size: int = 1, **kwargs):
        super(QuantStatsRewards, self).__init__(**kwargs)
        self._window_size = self.default('window_size', window_size)

    def get_reward(self, portfolio: 'Portfolio') -> float:
        net_worth = portfolio.performance['net_worth']

        # check for max drawdown
        max_dd = -0.05  # max allowed drawdown is -5%
        dd = qs.stats.to_drawdown_series(net_worth).iat[-1]
        if dd < max_dd:
            return -100  # severely punish model for a big drawdown

        # TODO: we can add BuyAndHold benchmark here.
        #   If net_worth < (buy_and_hold - 10%) then punish model

        # check if we did enough steps to calculate metrics
        # TODO: for sharp we can calculate this with MinTRL (Minimum Track Record Length)
        #   (mlfinlab library has it)
        min_data_len = 20
        if len(net_worth) < min_data_len:
            return 0

        # calc difference between current sharp value and a previous one
        sharpe = qs.stats.sharpe(net_worth)
        sharp_prev = qs.stats.sharpe(net_worth[:-1]) if len(net_worth) > min_data_len else 0
        return sharpe - sharp_prev
