from tensortrade.utils import Data
from tensortrade.utils import TA_Features as ta

from tensortrade.actions import SimpleOrders, ManagedRiskOrders 
# Function dictionaries for ease of access in dropdowns
action_funcs = {
    'Simple: Buy/Hold/Sell': SimpleOrders,
    'Managed Risk: Simple + TP & SL': ManagedRiskOrders
}

from tensortrade.rewards import SimpleProfit, RiskAdjustedReturns 
# Reward Function Options
reward_funcs = {
    'Simple Net Worth': SimpleProfit,
    'Sortino of Net Worth': RiskAdjustedReturns,
    'Sharpe of Net Worth': RiskAdjustedReturns,
    #'Max Drawdown %': QuantStatsRewards,
    #'Max Drawdown Days': QuantStatsRewards
}

from tensortrade.agents import A2CAgent, DQNAgent
agent_funcs = {
    'A2C': A2CAgent,
    'DQN': DQNAgent,
    #'A3C': A3CAgent,
    #'Parallel DQN': ParallelDQNAgent,
    #'PPO': PPO,
    #'PPO Continuous': PPO,
    #'PPO-LSTM': PPO,
    #'Vanilla Actor Critic': ActorCritic,
    #'ACER': ActorCritic,
    #'TD3': TD3,
    #'DDPG': MuNet,
    #'SAC': PolicyNet,
}

from tensortrade.exchanges import Exchange
from tensortrade.exchanges.services.execution.simulated import execute_order
def return_exchange_obj(asset_label, close_data, exchange_label='exchange-module'):
    return Exchange(exchange_label, service=execute_order)(
                Stream(list(close_data), asset_label))


from tensortrade.data import Module, Stream, DataFeed
def return_datafeed(dataset, exchange_label='exchange-module'):
    # Init Datafeed with the series of virtual Exchange data Nodes
    with Module(exchange_label) as exchange_ns:
        nodes = [
            Stream(list(dataset[col]), col) for col in dataset.columns
        ]
        #nodes += More stream arrays when using more assets then 1
    return DataFeed([exchange_ns]) #feed


# To make custom instruments with
from tensortrade.instruments import Instrument
def return_instrument(asset, decimals=2):
    return Instrument(asset, decimals)


# RENDERERS/LOGGERS/CHART PLOTTING
from tensortrade.environments.render import PlotlyTradingChart
def return_plotlylogger(save_as_json=False):
    return PlotlyTradingChart(
        display=False,
        height=800,
        save_as_json=save_as_json,
        #save_format='html',
        #path=path,
        #auto_open_html=False
    )


from tensortrade.environments.render import ScreenLogger 
def return_screenlogger(date_format='%Y-%m-%d %H:%M:%S %p'):
    return ScreenLogger(date_format=date_format)


# Feature Engineering
import numpy as np

def close_minus_open(df, close_='close', open_='open'):
    try:
        df['close_minus_open'] = np.log(df[close_]) - np.log(df[open_])
    except:
        pass
    return df 

def high_minus_low(df, high_='high', low_='low'):
    try:
        df['high_minus_low'] = np.log(df['high']) - np.log(df['low'])
    except:
        pass
    return df


# Custom StreamLit Object Hashing Function for Cache
def agent_hashing(agent):
    return [
        agent.n_episodes, 
        agent.learning_rate, 
        agent.save_path,
        [node.name for node in agent.env.feed.inputs],
        type(agent),
        agent.env.window_size,
        type(agent.env.action_scheme),
        agent.env.action_scheme.trade_sizes,
        type(agent.env.reward_scheme),
    ]


# import streamlit as st
# AGENT = None

# @st.cache(allow_output_mutation=True,
#           hash_funcs={A2CAgent: agent_hashing})
# def set_cached_agent_model(AGENT_):
#     AGENT = AGENT_
#     return AGENT

# @st.cache(allow_output_mutation=True,
#           hash_funcs={A2CAgent: agent_hashing})
# def return_cached_agent_model():
#     return AGENT