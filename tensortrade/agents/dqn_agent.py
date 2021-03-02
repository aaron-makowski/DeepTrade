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
# limitations under the License.

"""
References:
    - https://towardsdatascience.com/deep-reinforcement-learning-build-a-deep-q-network-dqn-to-play-cartpole-with-tensorflow-2-and-gym-8e105744b998
    - https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#dqn-algorithm
"""

import random
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
import quantstats as qs
from collections import namedtuple

from tensortrade.agents import Agent, ReplayMemory
from tensortrade.environments.render import PlotlyTradingChart
from tensortrade.tensorboard import TensorBoardCallback
from pathlib import Path

DQNTransition = namedtuple('DQNTransition', ['state', 'action', 'reward', 'next_state', 'done'])


class DQNAgent(Agent):

    def __init__(self,
                 env: 'TradingEnvironment',
                 policy_network: tf.keras.Model = None):
        self.env = env
        self.n_actions = env.action_space.n
        self.observation_shape = env.observation_space.shape

        self.policy_network = policy_network or self._build_policy_network()

        self.target_network = tf.keras.models.clone_model(self.policy_network)
        self.target_network.trainable = False

        self.env.agent_id = self.id

        self.episode_metrics = {}

    def _build_policy_network(self):
        network = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=self.observation_shape),
            tf.keras.layers.Conv1D(filters=64, kernel_size=6, padding="same", activation="tanh"),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding="same", activation="tanh"),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(self.n_actions, activation="sigmoid"),
            tf.keras.layers.Dense(self.n_actions, activation="softmax")
        ])

        return network

    def restore(self, path: str, **kwargs):
        self.policy_network = tf.keras.models.load_model(path)
        self.target_network = tf.keras.models.clone_model(self.policy_network)
        self.target_network.trainable = False

    def save(self, path: str, **kwargs):
        episode: int = kwargs.get('episode', None)

        if episode:
            filename = "policy_network__" + self.id + "__" + str(episode).zfill(3) + ".hdf5"
        else:
            filename = "policy_network__" + self.id + ".hdf5"

        self.policy_network.save(path + filename)

    def get_action(self, state: np.ndarray, **kwargs) -> int:
        threshold: float = kwargs.get('threshold', 0)

        rand = random.random()

        if rand < threshold:
            return np.random.choice(self.n_actions)
        else:
            return np.argmax(self.policy_network(np.expand_dims(state, 0)))

    def _apply_gradient_descent(self, memory: ReplayMemory, batch_size: int, learning_rate: float, discount_factor: float):
        optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
        loss = tf.keras.losses.Huber()

        transitions = memory.sample(batch_size)
        batch = DQNTransition(*zip(*transitions))

        state_batch = tf.convert_to_tensor(batch.state)
        action_batch = tf.convert_to_tensor(batch.action)
        reward_batch = tf.convert_to_tensor(batch.reward, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(batch.next_state)
        done_batch = tf.convert_to_tensor(batch.done)

        with tf.GradientTape() as tape:
            state_action_values = tf.math.reduce_sum(
                self.policy_network(state_batch) * tf.one_hot(action_batch, self.n_actions),
                axis=1
            )

            next_state_values = tf.where(
                done_batch,
                tf.zeros(batch_size),
                tf.math.reduce_max(self.target_network(next_state_batch), axis=1)
            )

            expected_state_action_values = reward_batch + (discount_factor * next_state_values)
            loss_value = loss(expected_state_action_values, state_action_values)

        variables = self.policy_network.trainable_variables
        gradients = tape.gradient(loss_value, variables)
        optimizer.apply_gradients(zip(gradients, variables))

        self.episode_metrics['loss'] = loss_value.numpy()

    def add_episode_metrics(self):
        # Calculate episode metrics for Tensorboard or other loggers.
        # Attention: this code works with simple orders that were executed in 1 step.
        # It does not handle things like slippage or partial fills.
        trades = self.env.broker.trades.values()
        net_worth = self.env.portfolio.performance['net_worth']

        # convert trade commissions to same currency
        commissions = [trade[0].commission if trade[0].is_buy else trade[0].commission * trade[0].price
                       for trade in trades]

        self.episode_metrics.update({
            'net_worth': self.env.portfolio.net_worth,
            'profit_loss': self.env.portfolio.profit_loss,
            'trades_num': len(trades),
            'traded_amount': float(sum(trade[0].size for trade in trades)),
            'max_drawdown': qs.stats.max_drawdown(net_worth),
            'avg_drawdown': np.mean(qs.stats.to_drawdown_series(net_worth)),
            'sharpe_ratio': qs.stats.sharpe(net_worth),
            'commissions': sum(x.as_float() for x in commissions),
        })

        # calculate intra-episode histograms
        sharpe_ratio_hist = []
        for i in range(len(net_worth)):  # sharpe ratio on every episode step
            nw = net_worth[:i + 1]
            sharpe_ratio_hist.append(qs.stats.sharpe(nw))
        self.episode_metrics.update({
            'sharpe_ratio_hist': pd.Series(sharpe_ratio_hist),
            'drawdown_hist': qs.stats.to_drawdown_series(net_worth),
        })

    def best_episode_report(self, episode_data, log_dir: str):
        # quantstats report for the best episode
        log_dir = Path(log_dir, 'best_episode')
        log_dir.mkdir(parents=True, exist_ok=True)

        n_steps = self.env.max_steps

        dates = self.env.price_history['datetime'][:n_steps + 1].apply(lambda x: pd.to_datetime(x))
        returns = pd.concat([episode_data['net_worth'], dates], axis=1).set_index(
            'datetime')  # add date index to net worth values
        qs.reports.html(returns['net_worth'], output=Path(log_dir, 'quantstats.html'))

        chart = PlotlyTradingChart(
            display=False,  # show the chart on screen (default)
            height=800,  # affects both displayed and saved file height. None for 100% height.
            save_format='html',  # save the chart to an HTML file
            auto_open_html=False,  # open the saved HTML chart in a new browser tab,
            path=log_dir
        )

        chart.render(episode=episode_data['episode'],
                     max_episodes=self.env.max_episodes,
                     step=self.env.max_steps,
                     max_steps=self.env.max_steps,
                     price_history=self.env.price_history[self.env.price_history.index < n_steps],
                     net_worth=episode_data['net_worth'],
                     performance=episode_data['performance'],
                     trades=episode_data['trades'])
        chart.save()

    def train(self,
              n_steps: int = None,
              n_episodes: int = None,
              save_every: int = None,
              save_path: str = None,
              callback: callable = None,
              **kwargs) -> float:
        batch_size: int = kwargs.get('batch_size', 128)
        discount_factor: float = kwargs.get('discount_factor', 0.9999)
        learning_rate: float = kwargs.get('learning_rate', 0.0001)
        eps_start: float = kwargs.get('eps_start', 0.9)
        eps_end: float = kwargs.get('eps_end', 0.05)
        eps_decay_steps: int = kwargs.get('eps_decay_steps', n_episodes * n_steps / 4)
        update_target_every: int = kwargs.get('update_target_every', 1000)
        memory_capacity: int = kwargs.get('memory_capacity', 1000)
        render_interval: int = kwargs.get('render_interval', 50)  # in steps, None for episode end render only

        # data of best episode
        best_episode = {
            'reward': float('-inf'),
        }

        memory = ReplayMemory(memory_capacity, transition_type=DQNTransition)
        episode = 0
        total_steps_done = 0

        if n_steps and not n_episodes:
            n_episodes = np.iinfo(np.int32).max

        print('====      AGENT ID: {}      ===='.format(self.id))
        self.env.max_episodes = n_episodes
        self.env.max_steps = n_steps

        #train_date = datetime.now().strftime("%Y-%m-%d %H.%M.%S")
        #tensorboard_callback = TensorBoardCallback(log_dir=Path('logs/tensorboard', train_date), histogram_freq=1)
        #tensorboard_callback.set_model(self.policy_network)

        #tensorboard_callback.on_train_begin()

        while episode < n_episodes:
            state = self.env.reset()
            done = False
            steps_done = 0
            total_reward = 0
            threshold = None

            self.episode_metrics = {}
            #tensorboard_callback.on_epoch_begin(episode)

            while not done:
                #tensorboard_callback.on_train_batch_begin(steps_done)

                threshold = eps_end + (eps_start - eps_end) * np.exp(-total_steps_done / eps_decay_steps)
                action = self.get_action(state, threshold=threshold)
                next_state, reward, done, _ = self.env.step(action)

                memory.push(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward
                steps_done += 1
                total_steps_done += 1

                if len(memory) < batch_size:
                    continue

                self._apply_gradient_descent(memory, batch_size, learning_rate, discount_factor)

                if n_steps and steps_done >= n_steps:
                    done = True

                if render_interval is not None and steps_done % render_interval == 0:
                    self.env.render(episode)

                if total_steps_done % update_target_every == 0:
                    self.target_network = tf.keras.models.clone_model(self.policy_network)
                    self.target_network.trainable = False

                #tensorboard_callback.on_train_batch_end(steps_done-1)

            self.add_episode_metrics()
            self.episode_metrics.update({
                'total_reward': total_reward,
                'epsilon': threshold,
            })

            # check if this is the best episode so far (to use for quantstats report and plotly chart)
            if total_reward > best_episode['reward']:
                net_worth = self.env.portfolio.performance['net_worth']
                best_episode['episode'] = episode
                best_episode['reward'] = total_reward
                best_episode['net_worth'] = net_worth
                best_episode['performance'] = self.env.portfolio.performance
                best_episode['trades'] = self.env.broker.trades

            is_checkpoint = save_every and episode % save_every == 0

            if save_path and (is_checkpoint or episode == n_episodes - 1):
                self.save(save_path, episode=episode)

            if not render_interval or steps_done < n_steps:
                self.env.render(episode)  # render final state at episode end if not rendered earlier

            self.env.save()

            #tensorboard_callback.on_epoch_end(episode, self.episode_metrics)

            episode += 1

        self.best_episode_report(best_episode, f'logs/{train_date}/')

        mean_reward = total_reward / steps_done

        #tensorboard_callback.on_train_end()

        return mean_reward

