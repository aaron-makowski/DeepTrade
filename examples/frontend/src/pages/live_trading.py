import pandas as pd

import streamlit as st
import tensorflow as tf

import time, os
from datetime import datetime
import webbrowser
from subprocess import Popen, PIPE # multiprocessing

from Pytrader_API import Pytrader_API
from src.pages.training_functions import *

import h5py
import io

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


def open_mt5_order(MT, 
                   instrument, 
                   ordertype, 
                   openprice,
                   volume, 
                   stoploss,
                   takeprofit,
                   slippage = 0.0, 
                   magicnumber = 2000, 
                   comment = 'PyTrader EA Executed Trade'
    ):
    NewOrder = MT.Open_order(
                instrument=instrument,
                ordertype=ordertype,
                openprice=openprice,
                volume=volume,
                stoploss=stoploss,
                takeprofit=takeprofit,
                slippage=slippage,
                comment=comment,
                magicnumber=magicnumber,
    )
    if (NewOrder == -1): # opening order failed
        print(MT.order_error)
    print(MT.order_return_message)

@st.cache
def fetch_ohlcv_data(exchange, symbol, timeframe, 
                     candle_amount, save_path=None, save_format='csv'):
    # Yahoo/CCXT/MT5
    if exchange == 'Yahoo':
        candles = Data.Yahoo.fetch_candles(
            symbol = symbol,
            timeframe = timeframe,
            candle_amount = candle_amount,
            TT_Format = False, #True if multiple assets
            save_path = save_path,
            save_format = save_format if save_path else None
        )
        #tf == 1m: candle_amount = 7d max
        #[5m, 15m, 30m] = 60d max
        #1h = 730d
    elif type(exchange) == str:
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

MT = None


def write():
    """Method used to write page in app.py for streamlit"""

    st.title('**DeepTrade** `Live Model Trading`')
    st.header('Trading Options')
    st.text('')

    trading_mode = st.selectbox(
        'Trading Mode', 
        ('MT5 Paper Trading','Yahoo Paper Trading','Crypto Paper Trading'))
        
    if 'MT5' in trading_mode:
        MT = mt5_api_object()

    # Mandatory identical to Training Parameters
    AGENT_ALGO_CHOICE = st.selectbox(
        "Reinforcement Learning Agent",
        ("DQN", "A2C"))
    
    
    # Load Saved/Trained Model File/s
    import os

    # Make Saved Model Files Selectbox
    # DIR_TREE = {}
    trained_models_folder = 'trained_models'
    ALL_MODEL_FOLDERS = []
    # Check if trained model folder exists and map the directory
    if trained_models_folder in os.listdir(f'.') and os.path.isdir(f'./{trained_models_folder}'):
        # DIR_TREE[trained_models_folder] = {}
        for item in os.listdir(f'./{trained_models_folder}'):
            if os.path.isdir(item):
                ALL_MODEL_FOLDERS.append(item)
                #DIR_TREE[trained_models_folder][item] = []
            # else: # Should be none in the top level
            #     DIR_TREE[trained_models_folder][item] = 'file'
        # Explore each model subfolder
        # for subfolder, files in DIR_TREE[trained_models_folder].items():
        #     if isinstance(files, list):
        #         for item in os.listdir(f'./{trained_models_folder}/{subfolder}'):
        #             'MODEL_FOLDERS'.append(item)
        #             if os.path.isdir(item):
        #                 'SUBFOLDERS'.append(item) # Should be 0
        #             else:
        #                 'MODEL_FILES'.append(item)
    else:
        st.write(f'DeepTrade expects `DeepTrade/{trained_models_folder}/` to be the folder with all of your saved trained models but couldn\'t find the folder and is making it now')
        try:
            os.mkdir(f'./{trained_models_folder}')
        except:
            st.write(f'ERROR: Failed to automatically make directory `DeepTrade/{trained_models_folder}/`')
    
    st.set_option('deprecation.showfileUploaderEncoding', False)
    uploaded_file = st.file_uploader(f"Upload a{'n Actor' if AGENT_ALGO_CHOICE == 'A2C' else ' DeepTrade'} Model file (`.HDF5`)") if st.checkbox('Upload a Model File', False) else st.selectbox("", ALL_MODEL_FOLDERS)
    
    if uploaded_file:
        try:
            uploaded_file = h5py.File(io.BytesIO(uploaded_file.read()), 'r')
            model = None
            model = tf.keras.models.load_model(uploaded_file)
            st.write('Loaded `TensorFlow.Keras` Model: ')# + uploaded_file.name)
        except Exception as e:
            st.write(f'Error in `tf.keras` Loading {str(uploaded_file)}:\n{str(e)}')

    if AGENT_ALGO_CHOICE == 'A2C': # Get second half of model
        critic_model_file = st.file_uploader(
            "Choose the corresponding DeepTrade `Critic` Model file") 
        if critic_model_file:
            try:
                critic_model_file = h5py.File(io.BytesIO(critic_model_file.read()), 'r')
                critic_model = None
                critic_model = tf.keras.models.load_model(critic_model_file)
                st.write('Loaded `TensorFlow.Keras` Critic Model: ')# + critic_model_file.name)
            except:
                st.write(f'Error in `TensorFlow.Keras` Model Loading {str(critic_model_file)}:\n{str(e)}')
    else:
        critic_model_file = None

    # Initialize an Agent loaded with the Trained Model
    # if critic_model_file:
    #     agent_restore_kwargs = dict(path = str(uploaded_file.name)[:str(uploaded_file.name).rfind('/')],
    #                   actor_filename = str(uploaded_file.name),
    #                   critic_filename = str(critic_model_file.name))
    # elif uploaded_file:
    #     agent_restore_kwargs = dict(path = str(uploaded_file.name))


    # Load all Model Params from bundled JSON
    # mkdir {'_'.join(ASSETS)_TIMEFRAME}
    # path = {'_'.join(ASSETS)_TIMEFRAME}
    # st.write(' '.join(str(uploaded_file).split('_')))
    # try:
    #     # Load/Deconstruct essential Parameters from model filename
    #     _, _,
    #     total_episodes,
    #     AGENT_CHOICE, WINDOW_SIZE,
    #     ACTION_SCHEME, TRADE_SIZES,
    #     SL_PERCENTS, TP_PERCENTS,
    #     FEATURE_CHOICES = str(uploaded_file).split('_')
    # except:
    #     st.write('Model Filename is missing essential parameters or is malformed.\nPlease manually input them below.')

    EXCHANGE = st.selectbox(
        "Choose a Data Provider", 
        ['MT5','Yahoo','Crypto'],
    )

    ASSET_CHOICE_BOX = st.text_input(
        "Choose an Asset for the AI Model to trade", value='EURUSD' if EXCHANGE == 'MT5' else 'MES=F' if EXCHANGE == 'Yahoo' else 'BTCUSDT')
    ASSET_CHOICE_BOX = [ASSET_CHOICE_BOX]
    
    if EXCHANGE == 'MT5' and MT and ASSET_CHOICE_BOX != ['']:
        if not mt5_asset_check(MT.get_broker_instrument_name, ASSET_CHOICE_BOX[0]):
            st.error(f'MT5 cannot find asset: {ASSET_CHOICE_BOX[0].upper()}')
        # check for valid crypto asset
        # check for valid yahoo asset
        
    TIMEFRAMES = [
        'MN1', 'W1', 'D1', 'H12', 'H8', 'H6', 
        'H4', 'H3', 'H2', 'H1', 'M30',
        'M20', 'M15', 'M12', 'M10', 'M6', 
        'M5', 'M4', 'M3', 'M2', 'M1'
    ] if EXCHANGE == 'MT5' else ['d1', 'h1'] if EXCHANGE == 'Yahoo' else ['d1', 'h1']
            
    TIMEFRAME = st.selectbox(
        "Candle/Bar Time Frame", 
        TIMEFRAMES, 
        index=2 if EXCHANGE =='MT5' else 0, # Default Choice
        format_func = lambda x: x[1:] + x[0] if x != 'MN1' else '1MN')

    st.subheader('Starting Asset Amounts')
    if len(ASSET_CHOICE_BOX[0]) > 0:
        if len(ASSET_CHOICE_BOX[0]) == 6:
            BASE_ASSET = ASSET_CHOICE_BOX[0][:3] # make sure these are right
            QUOTE_ASSET = ASSET_CHOICE_BOX[0][-3:]
        else:
            BASE_ASSET = st.text_input('Input the Traded asset symbol here (eg. EUR if EURUSD)', value=ASSET_CHOICE_BOX[0])
            QUOTE_ASSET = st.text_input('Input the Base Portfolio Asset as it looks in the symbol (USD if EURUSD)', value='')
    
    if EXCHANGE == 'MT5':
        ACCOUNT_INFO = MT.Get_dynamic_account_info()
        balance =      ACCOUNT_INFO[     'balance']
        equity =       ACCOUNT_INFO[      'equity']
        profit =       ACCOUNT_INFO[      'profit']
        margin =       ACCOUNT_INFO[      'margin']
        margin_free =  ACCOUNT_INFO[ 'margin_free']
        margin_level = ACCOUNT_INFO['margin_level']
        #
        #st.write(f'ACCOUNT INFO:\n{ACCOUNT_INFO}')
        # ACCOUNT INFO: {
        #     'balance': 3004.93, 
        #     'equity': 3004.93,
        #     'profit': 0.0,
        #     'margin': 0.0,
        #     'margin_level': 0.0,
        #     'margin_free': 3004.93
        # }
    
    # Set Starting Money/Assets for training agent
    QUOTE_CASH = int(st.number_input(
        f'{QUOTE_ASSET} starting amount', 
        min_value=0, 
        value=0
    ))
    BASE_CASH = int(st.number_input(
        f'{BASE_ASSET} starting amount. {margin_free}/{balance} {BASE_ASSET} Free Margin Balance' if EXCHANGE == 'MT5' else '{BASE_ASSET} starting amount.', 
        min_value=0, 
        value=int(margin_free) if EXCHANGE == 'MT5' else 10000
    ))

    OHLCV = ['Open', 'High', 'Low', 'Close', 'Volume']
    feature_list = OHLCV + ['Close Minus Open', 'High Minus Low'] \
                         + list(ta.pantulipy_indicators.keys()) \
                         + list(ta.unique_pandas_ta_indicators.keys())  

    # Mandatory identical to Training Parameters
    st.subheader('Features')
    FEATURE_CHOICES = st.multiselect(
        'Choose which data to use as features', 
        feature_list,
        # Default Feature Choices
        ('Volume', 'Close Minus Open', 'High Minus Low', 'macd', 'rsi'),
        format_func = lambda x: x[0].upper() + x[1:])


    st.subheader('Agent Options')
    # Mandatory identical to Training Parameters
    WINDOW_SIZE = st.number_input(
        "Amount of consecutive previous candles/bars that the Agent sees for each decision.", value=100)

    # POSSIBLY ALLOWED TO VARY BUT UNELSS INPUT GRID SIZE VARIES ERROR HAPPESN
    st.subheader('Action Scheme Options')
    ACTION_SCHEME_CHOICE = st.selectbox(
         'Action Scheme',
        ('Simple: Buy/Hold/Sell',
         'Managed Risk: Simple + TP & SL'))

    MAX_TRADE_SIZE = 33
    TRADE_SIZES = st.multiselect(
        f'Trade Sizes',
        [x/100 for x in range(1, MAX_TRADE_SIZE, 1)],
        [0.05, 0.10, 0.15]
    )

    if ACTION_SCHEME_CHOICE == 'Managed Risk: Simple + TP & SL':
        SL_PERCENTS = st.multiselect(
            f'Stop Loss Order Sizes',
            [x/100 for x in range(1, MAX_TRADE_SIZE, 1)],
            [x/100 for x in range(1, 6, 1)]
        )
        TP_PERCENTS = st.multiselect(
            f'Take Profit Order Sizes',
            [x/100 for x in range(1, MAX_TRADE_SIZE, 1)],
            [x/100 for x in range(1, 6, 1)] + [x/100 for x in range(6, 18, 3)]
        )
    else:
        SL_PERCENTS = TP_PERCENTS = None



    # Train Agent Button and Execution
    if st.button("Start Trading"):
        # tickers = ['BTC-USD']
        # yf = YahooFinancials(tickers)
        # retries = 1
        # while retries > 0:
        #     try:
        #         current_price = yf.get_current_price()
        #         print(current_price)
        #         break
        #     except Exception as e:
        #         retries -= 1
        #         print(str(e))
        
        
        # Grab max candles and make master dataset of all the indicators in the feature set so EMA shit is accurate
        FULL_HISTORY = fetch_ohlcv_data(
            exchange=MT, #'Yahoo',#
            symbol=ASSET_CHOICE_BOX[0],
            timeframe=TIMEFRAME,#'1d',#
            candle_amount=1000)#'max',

        NowTime = datetime.utcnow()
        LastBarDate = datetime.utcnow()
        DateTimeTF = MT.get_timeframe_value(TIMEFRAME)
        
        # loop forever on a schedule basis for hen new candles complete
        while True: 
            NowTime = datetime.utcnow()
            TimeDiff = NowTime - LastBarDate
            
            if TimeDiff.total_seconds() >= DateTimeTF.total_seconds():
                DATASET = fetch_ohlcv_data(
                    exchange=MT,
                    symbol=ASSET_CHOICE_BOX[0],
                    timeframe=TIMEFRAME,
                    candle_amount=WINDOW_SIZE + 1)
                
                # Verify we have a new Bar to process and arent double processing
                # after hours or closed down hours
                # right now this will loop 1 sec then every TF get candles
                # EVEN IF ITS THE WEEKEND
                NewBarDate = pd.to_datetime(DATASET.datetime.values[-1])
                if NewBarDate == LastBarDate:
                    time.sleep(1)
                    continue
                
                # to see if its new or whatever and get the right shit
                DATASET = DATASET[-(WINDOW_SIZE+1):-1].reset_index(drop=True)

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
                    'High Minus Low': high_minus_low
                }
                # Apply chosen features from all 3 sets of available features
                for feature in FEATURE_CHOICES: 
                    if feature not in OHLCV:
                        if feature in   feature_funcs.keys():
                            DATASET =   feature_funcs[feature](DATASET)
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
                    # Need loop to add wallets to portfolio
                        # Need CASH amount input box for each asset in training data
        
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
        
        
                from tensortrade.environments import TradingEnvironment
                ENVIRO = TradingEnvironment(
                    reward_scheme='simple',
                    action_scheme=ACTION_SCHEME,
                    window_size=WINDOW_SIZE,
                    feed=DATAFEED,
                    portfolio=PORTFOLIO,
                    renderers=[return_plotlylogger(save_as_json=False)],
                    price_history=PRICE_HISTORY,
                    use_internal=True, # Use own assets as features option needed
                )
        
                AGENT = agent_funcs[AGENT_ALGO_CHOICE](ENVIRO) # Init TT ML Agent
                if AGENT_ALGO_CHOICE == 'A2C':
                    AGENT.actor_network = model
                    AGENT.critic_network = critic_model_file
                else:
                    AGENT.policy_network = model
                    AGENT.target_network = tf.keras.models.clone_model(AGENT.policy_network)
                    AGENT.target_network.trainable = False
                
                #AGENT.restore(**agent_restore_kwargs) # Load Trained Model
                TRADES = AGENT.one_live_run() # runs once, for live one window -> action
                
                if TRADES != {}: 
                    st.write(TRADES)
                    
                    st.write(TRADES["type"])
                    st.write(TRADES["quantity"])

                    st.write(MT.Get_dynamic_account_info())

                    open_mt5_order(MT, 
                       instrument = ASSET_CHOICE_BOX[0], #TRADES["exchange_pair"], 
                       ordertype = TRADES["side"], 
                       openprice = TRADES["price"],
                       volume = TRADES['size'],
                       stoploss = round(TRADES['down_percent'] * TRADES["price"], 
                                        str(TRADES["price"])[::-1].find('.')) if 'down_percent' in TRADES.keys() else 0.0,
                       takeprofit = round(TRADES['up_percent'] * TRADES["price"], 
                                          str(TRADES["price"])[::-1].find('.')) if 'up_percent' in TRADES.keys() else 0.0,
                       slippage = 0.0)
                       
                    st.write(MT.Get_dynamic_account_info())
                else: 
                    st.write('No Trades for this Bar')
                    
            time.sleep(1)