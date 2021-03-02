from src.pages.training_functions import *

import os
import streamlit as st
import quantstats as qs
import pandas as pd
import numpy as np

import ccxt
from Pytrader_API import Pytrader_API

# Global Vars
mt5_watchlist = {
    'EURUSD':'EURUSD', 'GBPUSD':'GBPUSD',
    'LTCUSD':'LTCUSD', 'BTCUSD':'BTCUSD',
    'ETHUSD':'ETHUSD', 'XAUUSD':'XAUUSD',
    'USDCAD':'USDCAD', 'USDCHF':'USDCHF',
    'USDJPY':'USDJPY',
    'AUDUSD':'AUDUSD', 'AUDCAD':'AUDCAD',
    'AUDCHF':'AUDCHF', 'AUDJPY':'AUDJPY',
    'AUDNZD':'AUDNZD',
    'EURGBP':'EURGBP', 'EURAUD':'EURAUD',
    'EURCHF':'EURCHF', 'EURJPY':'EURJPY',
    'EURNZD':'EURNZD', 'EURCAD':'EURCAD',
    'GBPCHF':'GBPCHF', 'GBPJPY':'GBPJPY',
    'GBPUSD':'GBPUSD', 'GBPAUD':'GBPAUD',
    'GBPCAD':'GBPCAD', 'GBPNZD':'GBPNZD',
    'NZDCAD':'NZDCAD', 'NZDCHF':'NZDCHF',
    'NZDJPY':'NZDJPY', 'NZDUSD':'NZDUSD',
    'CADCHF':'CADCHF', 'CADJPY':'CADJPY',
    'CHFJPY':'CHFJPY'
}

ASSET_CHOICES = []


# @st.cache(allow_output_mutation=True)
# def ADD_ASSET_CHOICES(ASSET_CHOICES, NEW_ASSET):
#     return ASSET_CHOICES.append(NEW_ASSET)

# @st.cache(allow_output_mutation=True)
# def DEL_ASSET_CHOICES(ASSET_CHOICES, NEW_ASSET):
#     return ASSET_CHOICES.remove(NEW_ASSET)

@st.cache(allow_output_mutation=True)
def mt5_asset_check(CHECK, ASSET):
    ASSET = CHECK(ASSET)
    if ASSET != 'none':
        return ASSET

@st.cache(allow_output_mutation=True,
          suppress_st_warning=True)
def mt5_api_object():
    MT = Pytrader_API()
    MT.Connect(server='207.180.211.227', port=1112,
               instrument_lookup=mt5_watchlist)
    # Verify Connection
    if not MT.connected:
        st.write('MT Connection Error')
    else:
        st.write('MT Connected')
    return MT

@st.cache
def mt5_symbol_watchlist(get_mt5_watchlist):
    return get_mt5_watchlist()

@st.cache(hash_funcs={A2CAgent: agent_hashing,
                      DQNAgent: agent_hashing},
          suppress_st_warning=True)
def train_agent(AGENT, KWARGS):
    MEAN_REWARD = AGENT.train(**KWARGS)
    return AGENT, MEAN_REWARD

@st.cache
def ccxt_exchanges():
    """Get/Cache of all exchanges in CCXT lib"""
    return ccxt.exchanges

@st.cache(suppress_st_warning=True)
def ccxt_exchange_object(exchange_id):
    """Return object that lets us access exchange data"""
    exchange_obj = getattr(ccxt, exchange_id)({
        'enableRateLimit': True })
    exchange_obj.load_markets() # Requisite CCXT step
    return exchange_obj

@st.cache
def ccxt_exchange_symbols(symbols):
    """Get/Cache all valid symbols on exchange"""
    return [symbol for symbol in symbols]

@st.cache
def ccxt_exchange_tfs(timeframes):
    """Get/Cache all valid timeframes for symbol"""
    return [tf for tf in timeframes]

@st.cache
def fetch_ohlcv_data(exchange, symbol, timeframe, 
                     candle_amount, save_path=None, save_format='csv'):
    if type(exchange) == str:
        candles = Data.CCXT.fetch_candles(
            exchange = exchange,
            symbol = symbol,
            timeframe = timeframe,
            candle_amount = candle_amount,
            TT_Format = False, #True if multiple assets
            save_path = save_path, 
            save_format = save_format if save_path else None)
    else:
        candles = exchange.Get_last_x_bars_from_now(
            instrument = symbol,
            timeframe = exchange.get_timeframe_value(timeframe),
            nbrofbars = candle_amount)

    if type(candles) != pd.DataFrame:
        candles = pd.DataFrame(candles)

    return candles
    # candles = Data.Yahoo.fetch_candles(
    #     symbol = 'AAPL SPY',
    #     timeframe = '1d',
    #     start='2015-01-01',
    #     #st.date_input()
    #     end='2020-01-01',
    #     #st.date_input()

def write():
    """Method used to write page for streamlit"""

    st.header('**DeepTrade** `Agent Training`')
    st.subheader('Training Data Options')
    st.text('')
    
    # Custom CSV Upload Zone
    if st.checkbox('Upload a Custom CSV', False):
        st.set_option( # Supress warning
            'deprecation.showfileUploaderEncoding', False)
        DATASET = st.file_uploader(
            'Choose a CSV file for training', type='csv')
            
        #DATASET = io.TextIOWrapper(DATASET)
        if DATASET is not None:
            print(f'Uploaded: {DATASET}')
            DATASET = pd.DataFrame(DATASET)
            st.write(DATASET)

        ASSET_CHOICE_BOX = st.text_input(
            'Name this Dataset/Asset', 
            value='')
            
        TIMEFRAME = st.text_input(
            'Input Time Frame of this Series if Applicable', 
            value='')
            
        BASE_ASSET = st.text_input(
            'Base Portfolio Asset. If BTC/USD then this should be USD', value='USD')
            
        QUOTE_ASSET = st.text_input(
            'Dataset Asset/Currency. If BTC/USD then this should be BTC', value='BTC')
            
        # Display column names and ask them to edit them or designate which is TOHLCV
        
    else:
        ASSET_CLASS_CHOICE = st.selectbox(
            'Data Source/Broker', 
            ('MT5','Yahoo','Crypto'))

        if ASSET_CLASS_CHOICE == 'Crypto':
            EXCHANGES = ccxt_exchanges()

            EXCHANGE_CHOICE_BOX = st.selectbox(
                "Search for a Crypto Exchange",
                EXCHANGES, 
                index=EXCHANGES.index('binance'), 
                format_func = lambda x: x[0].upper() + x[1:])

            EXCHANGE = ccxt_exchange_object(EXCHANGE_CHOICE_BOX)
            ALL_SYMBOLS = ccxt_exchange_symbols(EXCHANGE.symbols)
            ALL_TIMEFRAMES = ccxt_exchange_tfs(EXCHANGE.timeframes)

            ASSET_CHOICE_BOX = st.multiselect(
                "Search for an asset to add to the training dataset.",
                ALL_SYMBOLS, ['BTC/USDT'])

            TIMEFRAME = st.selectbox(
                "Candle/Bar Time Frame", 
                ALL_TIMEFRAMES)

            if not st.checkbox('Fetch Maximum Candles/Bars', False):
                CANDLE_AMOUNT = st.number_input(f'Amount of Candles/Bars to Download from {EXCHANGE_CHOICE_BOX}', 1000)
            else:
                CANDLE_AMOUNT = 'all'

            # Set Starting Money/Assets for training agent
            BASE_ASSET = ASSET_CHOICE_BOX[0][ASSET_CHOICE_BOX[0].find('/')+1:]
            
            # Need to loop to create as many QUOTE ASSETS as we need from data
            # But do it not here in the code do it later 
            QUOTE_ASSET = ASSET_CHOICE_BOX[0][:ASSET_CHOICE_BOX[0].find('/')]

        elif ASSET_CLASS_CHOICE == 'MT5':
            MT = mt5_api_object()

            # ASSET_CHOICE_BOX = st.multiselect(
            #     "Add an asset to add to the training dataset.",
            #     mt5_symbol_watchlist(MT.Get_instruments), ASSET_CHOICES)

            ASSET_CHOICE_BOX = st.text_input(
                "Input a symbol to add an asset to add to the training dataset.", value='EURUSD')
            ASSET_CHOICE_BOX = [ASSET_CHOICE_BOX]
            
            if not mt5_asset_check(MT.get_broker_instrument_name, ASSET_CHOICE_BOX[0]):
                st.error(f'MT5 cannot find asset: {ASSET_CHOICE_BOX[0].upper()}')
                
            # if ASSET_CHOICE_BOX != '':
            #     CUSTOM_ASSET = ASSET_CHOICE_BOX.upper().replace('/','').replace('_','').replace('-','')
            #     valid_symbol = MT.Check_instrument(CUSTOM_ASSET)
            #     if valid_symbol and CUSTOM_ASSET not in ASSET_CHOICES:
            #         ASSET_CHOICES.append(CUSTOM_ASSET)

            TIMEFRAME = st.selectbox(
                "Candle/Bar Time Frame", 
                ['MN1', 'W1', 'D1', 'H12', 'H8', 'H6', 
                'H4', 'H3', 'H2', 'H1', 'M30',
                'M20', 'M15', 'M12', 'M10', 'M6', 
                'M5', 'M4', 'M3', 'M2', 'M1'], 
                index=2,
                format_func = lambda x: x[1:] + x[0] if x != 'MN1' else '1MN' )

            CANDLE_AMOUNT = st.number_input(f'Amount of Candles/Bars to Download', 1000)

            # Set Starting Money/Assets for training agent
            BASE_ASSET = ASSET_CHOICE_BOX[0][-3:]
            
            # Need to loop to create as many QUOTE ASSETS as we need from data
            # But do it not here in the code do it later 
            QUOTE_ASSET = ASSET_CHOICE_BOX[0][:3]
    
        elif ASSET_CLASS_CHOICE == 'Yahoo':
            st.write('Yahoo Training not implemented, like 241')
            pass
        
    st.subheader('Starting Asset Amounts')
    QUOTE_CASH = int(st.number_input(f'{QUOTE_ASSET} starting amount', min_value=0, value=1))
    BASE_CASH = int(st.number_input(f'{BASE_ASSET} starting amount', min_value=0, value=100000))
    # Need dynamic amounts of quote cashe vars for multi asset training sessions

    OHLCV = ['Open', 'High', 'Low', 'Close', 'Volume']
    feature_list = OHLCV + ['Close Minus Open', 'High Minus Low'] \
                         + list(ta.pantulipy_indicators.keys()) \
                         + list(ta.unique_pandas_ta_indicators.keys())   
    # "Forex Factory Data", "Day of the Week", 
    # "Open Minus Last Open aka Diff()", "High", "Low", "Close" "Volume"
    # "Log(Close Minus Open)"
    # "Log(High Minus Low)"

    st.subheader('Features')
    FEATURE_CHOICES = st.multiselect(
        'Choose which data to use as features', 
        feature_list,
        # Default Feature Choices
        ('Volume', 'Close Minus Open', 'High Minus Low', 'macd', 'rsi'),
        format_func = lambda x: x[0].upper() + x[1:])

    st.subheader('Agent Options')
    AGENT_ALGO_CHOICE = st.selectbox(
        "Reinforcement Learning Agent",
        ("DQN", "A2C"))

    WINDOW_SIZE = st.number_input(
        "Amount of consecutive previous candles/bars that the Agent sees for each decision.", value=100)

    LEARNING_RATE = st.number_input(
        "Learning Rate", value=0.01)

    st.subheader('Reward Scheme Options')
    REWARD_SCHEME_CHOICE = st.selectbox(
         "Reward Scheme Choice",
        ("Simple Net Worth", 
         "Sortino of Net Worth", 
         "Sharpe of Net Worth",
         #"Max Drawdown %", 
         #"Max Drawdown Days", 
         # Upgrade QuantstatsRewards to do these and not use sharpe too
    ))

    if 'Sharpe' in REWARD_SCHEME_CHOICE:
        RISK_FREE_RATE = st.number_input('Risk Free Rate', value=0.0)
        MINIMIZE_TRADES = st.checkbox('Incentivise Less Trades', True)

    if REWARD_SCHEME_CHOICE == 'Sortino of Net Worth':
        RISK_FREE_RATE = st.number_input('Risk Free Rate', value=0.0)
        TARGET_RETURNS = st.number_input('Target Returns', value=0.0)
        MINIMIZE_TRADES = st.checkbox('Incentivise Less Trades', True)

    st.subheader('Action Scheme Options')
    ACTION_SCHEME_CHOICE = st.selectbox(
         'Action Scheme Choice',
        ('Simple: Buy/Hold/Sell',
         'Managed Risk: Simple + TP & SL'))

    MAX_TRADE_SIZE = 33 # 33% of portfolio

    TRADE_SIZES = st.multiselect(
        f'Input Trade Sizes for the AI to experiment with: 0.05 = 5% of entire {BASE_ASSET} wallet amount:',
        [x/100 for x in range(1, MAX_TRADE_SIZE, 1)],
        [0.05, 0.10, 0.15]
    )

    if ACTION_SCHEME_CHOICE == 'Managed Risk: Simple + TP & SL':
        SL_PERCENTS = st.multiselect(
            f'Input Stop Loss Order Sizes: 0.05 = 5% of entire {BASE_ASSET} wallet amount:',
            [x/100 for x in range(1, MAX_TRADE_SIZE, 1)],
            [x/100 for x in range(1, 6, 1)]
        )
        TP_PERCENTS = st.multiselect(
            f'Input Take Profit Order Sizes: 0.05 = 5% of entire {BASE_ASSET} wallet amount:',
            [x/100 for x in range(1, MAX_TRADE_SIZE, 1)],
            [x/100 for x in range(1, 6, 1)] + [x/100 for x in range(6, 18, 3)]
        )
    else:
        SL_PERCENTS = TP_PERCENTS = None

    N_EPISODES = st.number_input(
        'Episodes: Amount of full passes over the candle/bar history (this multiplies training time)', value=1)

    #Save essential Parameters to model filename
    model_filepath = None
    model_folderpath = 'trained_models'
    model_filename = 'Demo'#f'{N_EPISODES}_{REWARD_SCHEME_CHOICE}_{AGENT_ALGO_CHOICE}_{WINDOW_SIZE}_{ACTION_SCHEME_CHOICE}_{TRADE_SIZES}_{SL_PERCENTS}_{TP_PERCENTS}_{FEATURE_CHOICES}'


    # Name and Save Model files to Disk
    if st.checkbox('Save agent model files to disk', False):

        model_name = st.text_input('Cusom Model Label/Name')
        if model_name: 
            os.system(f'mkdir -p {model_folderpath}/{model_name}/')
            model_filename = f'{model_folderpath}/{model_name}/{model_filename}'
        else:
            model_filename = f'{model_folderpath}/{model_filename}'
        model_filepath = os.path.abspath(model_filename)

        try: 
            os.makedirs(model_filename[:model_filename.rfind('/')])
            st.write(f'Model will be saved at: {model_filename}')
        except OSError:
            pass # Dir exists
        except Exception as e:
            st.write(f'Error Creating Folders: {str(e)}')


    # Train Agent Button and Execution       
    if st.button("Start Training (ง •̀_•́)ง ผ(•̀_•́ผ)"):
        DATASET = fetch_ohlcv_data(
            exchange=EXCHANGE_CHOICE_BOX if ASSET_CLASS_CHOICE == 'Crypto' else MT,
            symbol=ASSET_CHOICE_BOX[0],
            timeframe=TIMEFRAME,
            candle_amount=CANDLE_AMOUNT) 

        # st.write(DATASET)
        # Check if current timestamp is close to now or not
        # if len(DATASET) > WINDOW_SIZE:
        #     DATASET = DATASET[-(WINDOW_SIZE+1):-1].reset_index(drop=True)
        #     st.write(DATASET)

        # Make column names all lowercase
        DATASET.columns = [col.lower() for col in DATASET.columns]
        DATASET.rename(columns={'date': 'datetime'}, inplace=True)
        # Covert dates to datetime objects
        # if Asset not Crypto:
        DATASET['datetime'] = DATASET['datetime'].apply(
            lambda x: pd.to_datetime(x, unit='s'))
    
        #Separate TOHLCV from features for charting
        PRICE_HISTORY = DATASET[['datetime', 'open', 'high', 'low', 'close', 'volume']]
        DATASET = DATASET.set_index('datetime')
      
        feature_funcs = {
            'Close Minus Open': close_minus_open, 
            'High Minus Low': high_minus_low}

        # Apply chosen features
        for feature in FEATURE_CHOICES:
            if feature not in OHLCV:
                if feature in feature_funcs.keys():
                    DATASET = feature_funcs[feature](DATASET)
                elif feature in ta.unique_pandas_ta_indicators.keys():
                    DATASET =   ta.get_one_pandas_ta_indicator(feature, DATASET)
                elif feature in ta.pantulipy_indicators.keys(): 
                    DATASET =   ta.get_one_pantulipy_indicator(feature, DATASET)

        DATASET.index.name = 'datetime'
        # Drop any unwanted OHLCV data
        OHLCV_TO_DROP = [x.lower() for x in OHLCV if x not in FEATURE_CHOICES]
        DATASET.drop(columns=OHLCV_TO_DROP, inplace=True)

        # Show Dataset
        st.header(f"{'_'.join(ASSET_CHOICE_BOX)}_{TIMEFRAME} Dataset")
        st.dataframe(DATASET)

        #Simulated Exchange to make trades on
        EXCHANGE_OBJ = return_exchange_obj(
            asset_label = f'{BASE_ASSET}-{QUOTE_ASSET}',
            close_data = PRICE_HISTORY['close'])

        # Features for observing by the agent
        DATAFEED = return_datafeed(
            dataset=DATASET,
            exchange_label='exchange-module')

        # Include EVERY asset involved in the training DATASET
        # May need to add custom assets in instrument.py (or closeby file), probs need dynamic adding method
        # Need loop to add wallets to portfolio
        # Need CASH amount input box for each asset in training data
        # Initialize TT Portfolio object

        from tensortrade.instruments import Instrument
        BASE_OBJ = Instrument(BASE_ASSET, 2) # need to fetch decimals
        QUOTE_OBJ = Instrument(QUOTE_ASSET, 2) # need to fetch decimals

        from tensortrade.wallets import Portfolio, Wallet
        PORTFOLIO = Portfolio(
            BASE_OBJ, # Set Base Asset in Portfolio
            [Wallet(EXCHANGE_OBJ, BASE_CASH * BASE_OBJ),
             Wallet(EXCHANGE_OBJ, QUOTE_CASH * QUOTE_OBJ)])

        # Init ActionScheme + build options
        action_options = {'trade_sizes': TRADE_SIZES}
        if ACTION_SCHEME_CHOICE == 'Managed Risk: Simple + TP & SL':
            action_options['SL_PERCENTS'] = SL_PERCENTS
            action_options['TP_PERCENTS'] = TP_PERCENTS
        #Pass in action_options dict{} as 'keyword args' via **
        ACTION_SCHEME = action_funcs[ACTION_SCHEME_CHOICE](**action_options)


        # Init RewardScheme + build options
        reward_options = {'window_size': WINDOW_SIZE}

        if REWARD_SCHEME_CHOICE in ['Sharpe', 'Sortino']:
            reward_options['risk_free_rate'] = RISK_FREE_RATE
            reward_options['return_algorithm'] = reward_funcs[REWARD_SCHEME_CHOICE]
            reward_options['minimize_trades'] = MINIMIZE_TRADES
            if REWARD_SCHEME_CHOICE == 'Sortino':
                reward_options['target_returns'] = TARGET_RETURNS

        #elif REWARD_SCHEME_CHOICE == 'max drawdown':
        #elif REWARD_SCHEME_CHOICE == 'max drawdown days':

        REWARD_SCHEME = reward_funcs[REWARD_SCHEME_CHOICE](**reward_options)

        # Screen + Chart output of results
        RENDERERS = [return_screenlogger(), 
                     return_plotlylogger(save_as_json=False)]

        from tensortrade.environments import TradingEnvironment
        ENVIRO = TradingEnvironment(
            action_scheme=ACTION_SCHEME, 
            reward_scheme=REWARD_SCHEME, 
            window_size=WINDOW_SIZE, 
            feed=DATAFEED,
            portfolio=PORTFOLIO,
            renderers=RENDERERS,
            price_history=PRICE_HISTORY,
            use_internal=True, # Use own assets as features
        )

        DATA_LENGTH = len(DATASET) # factor in slider val
        EPS_DECAY_STEPS = int(DATA_LENGTH * N_EPISODES / 2)

        AGENT = agent_funcs[AGENT_ALGO_CHOICE](ENVIRO)

        KWARGS = dict(
            n_steps = DATA_LENGTH, 
            n_episodes = N_EPISODES, 
            learning_rate = LEARNING_RATE,
            eps_decay_steps = EPS_DECAY_STEPS,
            render_interval = DATA_LENGTH,
            discount_factor = 0.9999,
            memory_capacity = 1000,
            update_target_every = 1000, # whats this do again?
            batch_size = 128,
            save_every = DATA_LENGTH * 10, # Every 10 episodes (i desire default overwrite)
            save_path = model_filepath,
        )

        # from os import system
        # system(f'mkdir -p ./examples/frontend/{ASSET_CHOICE_BOX[0].replace("/","_")}')

        AGENT, MEAN_REWARD = train_agent(AGENT, KWARGS)

        # import json
        # def load_json():
        #     SAVED_CHARTING_DATA = {}
        #     try:
        #         with open(f'./examples/frontend/{ASSET_CHOICE_BOX[0]}/charts.json', encoding='utf-8', mode="r") as f:
        #             SAVED_CHARTING_DATA = json.load(f)
        #     except:
        #         pass
        #     return SAVED_CHARTING_DATA


        # SAVED_CHARTING_DATA = load_json()
        # SAVED_CHARTING_DATA[
        #     ASSET_CHOICE_BOX[0]] = { # all this needed for plotly
        #     'episode': 0,
        #     'max_episodes': AGENT.env._max_episodes,
        #     'step': AGENT.env.clock.step - 1,
        #     'max_steps': AGENT.env._max_steps,
        #     'price_history': PRICE_HISTORY.to_json(),
        #     'net_worth': AGENT.env._portfolio.performance.net_worth.to_json(),
        #     'performance': AGENT.env._portfolio.performance.drop(columns=['base_symbol']).to_json(),
        #     'trades': AGENT.env._broker.trades,
        #     'features': FEATURE_CHOICES
        # }


        # with open(f'./examples/frontend/{ASSET_CHOICE_BOX[0]}/charts.json', encoding='utf-8', mode="w") as f:
        #     json.dump(SAVED_CHARTING_DATA, f, indent=4, sort_keys=True, separators=(',',' : '))



        # for num in range(0, N_EPISODES):
        #     AGENT.env._renderers[1].render(0)
        #     AGENT.env._renderers[1].save()

        st.balloons()
        st.success('Agent Training Complete!')

        # So we can access agent in other pages
        #AGENT = set_cached_agent_model(AGENT)

        # Save agent file button
        # save_button = False
        # save_button = st.button("Save agent", key="save")
        # if save_button is not False:
        #     if type(AGENT) == DQNAgent:
            #     AGENT.policy_network.save(model_filepath)
            # elif type(AGENT) == A2CAgent:
            #     AGENT.actor_network.save(model_filepath)
            #     AGENT.policy_network.save(model_filepath)
        #     st.success('Done Saving Agent @ ' + model_filename)



        st.header('QuantStats Analysis Metrics Report')

        # Prepare RETURNS
        if isinstance(AGENT.env.portfolio.performance.net_worth, 
                     (pd.DataFrame, pd.Series)):
            RETURNS = AGENT.env.portfolio.performance.net_worth.pct_change().iloc[1:]
        else: # Assuming type is np.ndarray
            RETURNS = np.diff(
                AGENT.env.portfolio.performance.net_worth, axis=0)
            # Get %
            np.divide(RETURNS, 
                      AGENT.env.portfolio.performance.net_worth[:-1],
                      out=RETURNS)

        # Add date index to Net_Worth values
        # probably a better way available using RETURNS.ix[DATASET.ix[:len(RETURNS)]] or something
        RETURNS = pd.concat([
             DATASET.reset_index()['datetime'].iloc[:len(RETURNS)],
             RETURNS.reset_index()['net_worth']], axis=1) 
        RETURNS = RETURNS[RETURNS.datetime.notnull()][1:]# Remove nulls
        RETURNS = RETURNS.set_index('datetime')


        # Calc/Plot QuantStats Metrics Report
        try:
            st.header('Drawdown')
            st.pyplot(
                qs.plots.drawdowns_periods(
                    RETURNS['net_worth'], 
                    show=False, 
                    #savefig=f'./examples/frontend/{ASSET_CHOICE_BOX[0]}/drawdowns_periods.png'
                )
            )
            st.pyplot(
                qs.plots.drawdown(
                    RETURNS['net_worth'], 
                    show=False, 
                    #savefig=f'./examples/frontend/{ASSET_CHOICE_BOX[0]}/drawdown.png' 
                )
            )
            st.header('Volatility')
            st.pyplot(
                qs.plots.rolling_volatility(
                    RETURNS['net_worth'], 
                    show=False, 
                    #savefig=f'./examples/frontend/{ASSET_CHOICE_BOX[0]}/rolling_volatility.png'  
                )
            )
            st.header('Sharpe')
            st.pyplot(
                qs.plots.rolling_sharpe(
                    RETURNS['net_worth'], 
                    show=False, 
                    #savefig=f'./examples/frontend/{ASSET_CHOICE_BOX[0]}/rolling_sharpe.png' 
                )
            )
            st.header('Sortino')
            st.pyplot(
                qs.plots.rolling_sortino(
                    RETURNS['net_worth'], 
                    show=False, 
                    #savefig=f'./examples/frontend/{ASSET_CHOICE_BOX[0]}/rolling_sortino.png'  
                )
            )
            st.header('Returns')
            st.pyplot(
                qs.plots.returns(
                    RETURNS['net_worth'], 
                    show=False, 
                    #savefig=f'./examples/frontend/{ASSET_CHOICE_BOX[0]}/returns.png' 
                )
            )
            st.pyplot(
                qs.plots.log_returns(
                    RETURNS['net_worth'], 
                    show=False, 
                    #savefig=f'./examples/frontend/{ASSET_CHOICE_BOX[0]}/log_returns.png' 
                )
            )
            st.pyplot(
                qs.plots.yearly_returns(
                    RETURNS['net_worth'], 
                    show=False, 
                    #savefig=f'./examples/frontend/{ASSET_CHOICE_BOX[0]}/yearly_returns.png' 
                )
            )
            st.pyplot(
                qs.plots.histogram(
                    RETURNS['net_worth'], 
                    show=False, 
                    #savefig=f'./examples/frontend/{ASSET_CHOICE_BOX[0]}/histogram.png'
                )
            )
            st.pyplot(
                qs.plots.daily_returns(
                    RETURNS['net_worth'], 
                    show=False, 
                    #savefig=f'./examples/frontend/{ASSET_CHOICE_BOX[0]}/daily_returns.png'   
                )
            )
            st.pyplot(
                qs.plots.monthly_heatmap(
                    RETURNS['net_worth'], 
                    show=False, 
                    #savefig=f'./examples/frontend/{ASSET_CHOICE_BOX[0]}/monthly_heatmap.png'  
                )
            )
            st.pyplot(
                qs.plots.distribution(
                    RETURNS['net_worth'], 
                    show=False, 
                    #savefig=f'./examples/frontend/{ASSET_CHOICE_BOX[0]}/distribution.png'  
                )
            )

        except Exception as e:
            #pass
            st.write('Exception: ' + str(e))



    # Havent got it working on Repl/Streamlit yet
    # Open TensorBoard Button
    #
    # st.title('TensorBoard Report')
    # st.header('TensorBoard Plots and Analysis ')
    #     #st.subheader('Agent configurations')
    # tensorboard_button = st.button("Launch tensorboard", key="tensorboard")
    # if tensorboard_button is not False:
    #     os.system("kill -9 $(lsof -t -i:6006)")
    #     proc = Popen(["tensorboard", "--logdir", "./logs"], stdout=PIPE)
    #     time.sleep(10)
    #     webbrowser.open("http://localhost:6006/")