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


from abc import abstractmethod

from tensortrade import Component, TimeIndexed


class RewardScheme(Component, TimeIndexed):

    registered_name = "rewards"

    def __init__(self, minimize_trades: bool = False):
        self.minimize_trades = self.default('minimize_trades', minimize_trades)

    def reset(self):
        """Optionally implementable method for resetting stateful schemes."""
        pass

    @abstractmethod
    def get_reward(self, portfolio: 'Portfolio') -> float:
        """
        Arguments:
            portfolio: The portfolio being used by the environment.

        Returns:
            A float corresponding to the benefit earned by the action taken this timestep.
        """
        raise NotImplementedError()

    def get_processed_reward(self, portfolio: 'Portfolio', trade_count: int):
        reward = self.get_reward(portfolio)

        if self.minimize_trades:
            # blunt positive rewards and magnify negative rewards
            reward = reward / (trade_count+1) if reward > 0 else reward * (trade_count+1)

        return reward
