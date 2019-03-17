# --------------------------------------- IMPORT LIBRARIES -------------------------------------------
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import random
import math
import warnings
import quandl
import time
warnings.filterwarnings("ignore")

from feature_select import FeatureSelector

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
import talib as tb
import matplotlib.pyplot as plt
from matplotlib import gridspec

# --------------------------------------- GLOBAL PARAMETERS -------------------------------------------


# Set total fund pool
TRAIN_PORTION = 0.9
ACCOUNT_FUND = 100000
ALLOCATION_RATIO = 0.2
SINGLE_TRADING_FUND = ACCOUNT_FUND * ALLOCATION_RATIO
PRICE_IMPACT = 0.1

# Start and end period of historical data in question
START_TRAIN = datetime(2000, 1, 1)
END_TRAIN = datetime(2017, 2, 12)
START_TEST = datetime(2017, 2, 12)
END_TEST = datetime(2019, 2, 22)

# DJIA component stocks
DJI = ['MMM', 'AXP', 'AAPL', 'BA', 'CAT', 'CVX', 'CSCO', 'KO', 'DIS', 'XOM', 'GE', 'GS',
          'HD', 'IBM', 'INTC', 'JNJ', 'JPM', 'MCD', 'MRK', 'MSFT', 'NKE', 'PFE', 'PG', 'UTX',
          'UNH', 'VZ', 'WMT']
DJI_N = ['3M','American Express', 'Apple','Boeing','Caterpillar','Chevron','Cisco Systems','Coca-Cola','Disney'
         ,'ExxonMobil','General Electric','Goldman Sachs','Home Depot','IBM','Intel','Johnson & Johnson',
         'JPMorgan Chase','McDonalds','Merck','Microsoft','NIKE','Pfizer','Procter & Gamble',
         'United Technologies','UnitedHealth Group','Verizon Communications','Wal Mart']

CONTEXT_DATA = ['^GSPC', '^DJI', '^IXIC', '^RUT', 'SPY', 'QQQ', '^VIX', 'GLD', '^TYX', '^TNX' , 'SHY', 'SHV']

CONTEXT_DATA_N = ['S&P 500', 'Dow Jones Industrial Average', 'NASDAQ Composite', 'Russell 2000', 'SPDR S&P 500 ETF',
 'Invesco QQQ Trust', 'CBOE Volatility Index', 'SPDR Gold Shares', 'Treasury Yield 30 Years',
 'CBOE Interest Rate 10 Year T Note', 'iShares 1-3 Year Treasury Bond ETF', 'iShares Short Treasury Bond ETF']

random.seed(633)
RANDOM_STOCK = random.sample(DJI, 1)
ADD_STOCKS = 4

#13 WEEK TREASURY BILL (^IRX)
# https://finance.yahoo.com/quote/%5EIRX?p=^IRX&.tsrc=fin-srch
RISK_FREE_RATE = ((1+0.02383)**(1.0/252))-1 # Assuming 1.43% risk free rate divided by 360 to get the daily risk free rate.
MAR = 0.05
# ------------------------------------------------ CLASSES --------------------------------------------

class DataRetrieval:
    """
    This class prepares data by loading historical data from pre-saved data.
    """

    def __init__(self):
        # Initiate component data downloads
        self._dji_components_data()

    def _get_daily_data(self, symbol):
        """
        Load pre-saved historical data stock by stock.

        """
        daily_price = pd.read_csv("{}{}{}".format('./data/', symbol, '.csv'), index_col='Date', parse_dates=True)

        return daily_price

    def _dji_components_data(self):
        """
        This function retrieve all components data and assembles the required OHLCV data into respective data
        """

        for i in DJI + CONTEXT_DATA:
            print("Loading {}'s historical data".format((DJI + CONTEXT_DATA_N)[(DJI + CONTEXT_DATA).index(i)]))
            daily_price = self._get_daily_data(i)
            if i == (DJI + CONTEXT_DATA)[0]:
                self.components_df_o = pd.DataFrame(index=daily_price.index, columns=(DJI + CONTEXT_DATA))
                self.components_df_c = pd.DataFrame(index=daily_price.index, columns=(DJI + CONTEXT_DATA))
                self.components_df_h = pd.DataFrame(index=daily_price.index, columns=(DJI + CONTEXT_DATA))
                self.components_df_l = pd.DataFrame(index=daily_price.index, columns=(DJI + CONTEXT_DATA))
                self.components_df_v = pd.DataFrame(index=daily_price.index, columns=(DJI + CONTEXT_DATA))
                # Since this span more than 10 years of data, many corporate actions could have happened,
                # adjusted closing price is used instead
                self.components_df_o[i] = daily_price["Open"]
                self.components_df_c[i] = daily_price["Adj Close"]
                self.components_df_h[i] = daily_price["High"]
                self.components_df_l[i] = daily_price["Low"]
                self.components_df_v[i] = daily_price["Volume"]
            else:
                self.components_df_o[i] = daily_price["Open"]
                self.components_df_c[i] = daily_price["Adj Close"]
                self.components_df_h[i] = daily_price["High"]
                self.components_df_l[i] = daily_price["Low"]
                self.components_df_v[i] = daily_price["Volume"]

    def get_dailyprice_df(self):
        """
        Gets all stocks' close price and separates them into train and test set.
        """
        self.dow_stocks_test = self.components_df_c.loc[START_TEST:END_TEST][DJI]
        self.dow_stocks_train = self.components_df_c.loc[START_TRAIN:END_TRAIN][DJI]

    def get_all(self):
        """
        Response to external request to get all stock price in train and test set.
        """

        self.get_dailyprice_df()
        return self.dow_stocks_train, self.dow_stocks_test

    def technical_indicators_df(self, daily_data):
        """
        Assemble a dataframe of technical indicator series for a single stock
        """
        o = daily_data['Open'].values
        c = daily_data['Close'].values
        h = daily_data['High'].values
        l = daily_data['Low'].values
        v = daily_data['Volume'].astype(float).values
        # define the technical analysis matrix

        # Most data series are normalized by their series' mean
        ta = pd.DataFrame()
        ta['MA5'] = tb.MA(c, timeperiod=5) / tb.MA(c, timeperiod=5).mean()
        ta['MA10'] = tb.MA(c, timeperiod=10) / tb.MA(c, timeperiod=10).mean()
        ta['MA20'] = tb.MA(c, timeperiod=20) / tb.MA(c, timeperiod=20).mean()
        ta['MA60'] = tb.MA(c, timeperiod=60) / tb.MA(c, timeperiod=60).mean()
        ta['MA120'] = tb.MA(c, timeperiod=120) / tb.MA(c, timeperiod=120).mean()
        ta['MA5'] = tb.MA(v, timeperiod=5) / tb.MA(v, timeperiod=5).mean()
        ta['MA10'] = tb.MA(v, timeperiod=10) / tb.MA(v, timeperiod=10).mean()
        ta['MA20'] = tb.MA(v, timeperiod=20) / tb.MA(v, timeperiod=20).mean()
        ta['ADX'] = tb.ADX(h, l, c, timeperiod=14) / tb.ADX(h, l, c, timeperiod=14).mean()
        ta['ADXR'] = tb.ADXR(h, l, c, timeperiod=14) / tb.ADXR(h, l, c, timeperiod=14).mean()
        ta['MACD'] = tb.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)[0] / \
                     tb.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)[0].mean()
        ta['RSI'] = tb.RSI(c, timeperiod=14) / tb.RSI(c, timeperiod=14).mean()
        ta['BBANDS_U'] = tb.BBANDS(c, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[0] / \
                         tb.BBANDS(c, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[0].mean()
        ta['BBANDS_M'] = tb.BBANDS(c, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[1] / \
                         tb.BBANDS(c, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[1].mean()
        ta['BBANDS_L'] = tb.BBANDS(c, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[2] / \
                         tb.BBANDS(c, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[2].mean()
        ta['AD'] = tb.AD(h, l, c, v) / tb.AD(h, l, c, v).mean()
        ta['ATR'] = tb.ATR(h, l, c, timeperiod=14) / tb.ATR(h, l, c, timeperiod=14).mean()
        ta['HT_DC'] = tb.HT_DCPERIOD(c) / tb.HT_DCPERIOD(c).mean()
        ta["High/Open"] = h / o
        ta["Low/Open"] = l / o
        ta["Close/Open"] = c / o

        self.ta = ta

    def label(self, df, seq_length):
        return (df['Returns'] > 0).astype(int)

    def preprocessing(self, symbol):
        """
        Preprocess all stock data into a big dataframe of features with the help of a feature selector , also creates label data

        """
        print("\n")
        print("Preprocessing {} & its technical data".format(symbol))
        print("============================================")
        self.daily_data = pd.DataFrame()
        self.daily_data['Returns'] = pd.Series(
            (self.components_df_c[symbol] / self.components_df_c[symbol].shift(1) - 1) * 100,
            index=self.components_df_c[symbol].index)
        self.daily_data['Open'] = self.components_df_o[symbol]
        self.daily_data['Close'] = self.components_df_c[symbol]
        self.daily_data['High'] = self.components_df_h[symbol]
        self.daily_data['Low'] = self.components_df_l[symbol]
        self.daily_data['Volume'] = self.components_df_v[symbol].astype(float)
        seq_length = 3
        self.technical_indicators_df(self.daily_data)
        self.X = self.daily_data[['Open', 'Close', 'High', 'Low', 'Volume']] / self.daily_data[
            ['Open', 'Close', 'High', 'Low', 'Volume']].mean()
        self.y = self.label(self.daily_data, seq_length)
        X_shift = [self.X]

        for i in range(1, seq_length):
            shifted_df = self.daily_data[['Open', 'Close', 'High', 'Low', 'Volume']].shift(i)
            X_shift.append(shifted_df / shifted_df.mean())
        ohlc = pd.concat(X_shift, axis=1)
        ohlc.columns = sum([[c + 'T-{}'.format(i) for c in ['Open', 'Close', 'High', 'Low', 'Volume']] \
                            for i in range(seq_length)], [])
        self.ta.index = ohlc.index
        self.X = pd.concat([ohlc, self.ta], axis=1)
        self.Xy = pd.concat([self.X, self.y], axis=1)

        fs = FeatureSelector(data=self.X, labels=self.y)
        fs.identify_all(selection_params={'missing_threshold': 0.6,
                                          'correlation_threshold': 0.9,
                                          'task': 'regression',
                                          'eval_metric': 'auc',
                                          'cumulative_importance': 0.99})
        self.X_fs = fs.remove(methods='all', keep_one_hot=True)

        return self.X_fs

    def get_feature_dataframe(self, selected_stock):
        """
        Get the preprocessed dataframe and extract only the stocks in interest. Returns a smaller dataframe

        """
        self.feature_df = pd.DataFrame()
        for s in selected_stock:
            if s == selected_stock[0]:
                df = self.preprocessing(s)
                df.columns = [str(s) + '_' + str(col) for col in df.columns]
                self.feature_df = df
            else:
                df = self.preprocessing(s)
                df.columns = [str(s) + '_' + str(col) for col in df.columns]
                self.feature_df = pd.concat([self.feature_df, df], axis=1)
        return self.feature_df

    def get_adj_close(self, selected):
        """
        Get a 3D dataframe of Adjusted close price from Quandl.
        """
        # get adjusted closing prices of 5 selected companies with Quandl
        quandl.ApiConfig.api_key = 'CxU5-dDyxppBFzVgGG6z'
        data = quandl.get_table('WIKI/PRICES', ticker=selected,
                                qopts={'columns': ['date', 'ticker', 'adj_close']},
                                date={'gte': START_TRAIN, 'lte': END_TRAIN}, paginate=True)
        return data


class MathCalc:
    """
    This class performs all the mathematical calculations
    """

    @staticmethod
    def calc_return(period):
        """
        This function computes the return of a series
        """
        period_return = period / period.shift(1) - 1
        return period_return[1:len(period_return)]

    @staticmethod
    def calc_monthly_return(series):
        """
        This function computes the monthly return

        """
        return MathCalc.calc_return(series.resample('M').last())

    @staticmethod
    def positive_pct(series):
        """
        This function calculates the probably of positive values from a series of values.
        """
        return (float(len(series[series > 0])) / float(len(series)))*100

    @staticmethod
    def calc_yearly_return(series):
        """
        This function computes the yearly return
        """
        return MathCalc.calc_return(series.resample('AS').last())

    @staticmethod
    def max_drawdown(r):
        """
        This function calculates maximum drawdown occurs in a series of cummulative returns
        """
        dd = r.div(r.cummax()).sub(1)
        maxdd = dd.min()
        return round(maxdd, 2)

    @staticmethod
    def calc_lake_ratio(series):

        """
        This function computes lake ratio
        """
        water = 0
        earth = 0
        series = series.dropna()
        water_level = []
        for i, s in enumerate(series):
            if i == 0:
                peak = s
            else:
                peak = np.max(series[0:i])
            water_level.append(peak)
            if s < peak:
                water = water + peak - s
            earth = earth + s
        return water / earth

    @staticmethod
    def calc_gain_to_pain(daily_series):
        """
        This function computes the gain to pain ratio given a series of cummulative returns
        """

        try:
            monthly_returns = MathCalc.calc_monthly_return(daily_series.dropna())
            sum_returns = monthly_returns.sum()
            sum_neg_months = abs(monthly_returns[monthly_returns < 0].sum())
            gain_to_pain = sum_returns / sum_neg_months
        except:
            gain_to_pain = 1.0

        # print "Gain to Pain ratio: ", gain_to_pain
        return gain_to_pain

    @staticmethod
    def sharpe_ratio(returns):
        """
        Calculates Sharpe ratio from a series of returns.
        """
        return ((returns.mean() - RISK_FREE_RATE) / returns.std()) * np.sqrt(252)

    @staticmethod
    def downside_deviation(returns):
        """
        This method returns a lower partial moment of the returns. Create an array he same length as returns containing the minimum return threshold
        """
        #

        target = 0
        df = pd.DataFrame(data=returns, columns=["Returns"], index=returns.index)
        df["Downside Returns"] = 0
        df.loc[df["Returns"] < target, "Downside Returns"] = df["Returns"] ** 2
        expected_return = df["Returns"].mean()

        return np.sqrt(df["Downside Returns"].mean())

    @staticmethod
    def sortino_ratio(returns):
        """
        Calculates Sortino ratio from a series of returns.
        """
        return ((returns.mean() - RISK_FREE_RATE) / MathCalc.downside_deviation(returns))* np.sqrt(252)

    @staticmethod
    def calc_kpi(portfolio):
        """
        This function calculates individual portfolio KPI related its risk profile
        """

        kpi = pd.DataFrame(index=['KPI'], columns=['Avg. monthly return', 'Pos months pct', 'Avg yearly return',
                                                   'Max monthly dd', 'Max drawdown', 'Lake ratio', 'Gain to Pain',
                                                   'Sharpe ratio', 'Sortino ratio'])
        kpi['Avg. monthly return'].iloc[0] = MathCalc.calc_monthly_return(portfolio['Total asset']).mean() * 100
        kpi['Pos months pct'].iloc[0] = MathCalc.positive_pct(portfolio['Returns'])
        kpi['Avg yearly return'].iloc[0] = MathCalc.calc_yearly_return(portfolio['Total asset']).mean() * 100
        kpi['Max monthly dd'].iloc[0] = MathCalc.max_drawdown(MathCalc.calc_monthly_return(portfolio['CumReturns']))
        kpi['Max drawdown'].iloc[0] = MathCalc.max_drawdown(MathCalc.calc_return(portfolio['CumReturns']))
        kpi['Lake ratio'].iloc[0] = MathCalc.calc_lake_ratio(portfolio['Total asset'])
        kpi['Gain to Pain'].iloc[0] = MathCalc.calc_gain_to_pain(portfolio['Total asset'])
        kpi['Sharpe ratio'].iloc[0] = MathCalc.sharpe_ratio(portfolio['Returns'])
        kpi['Sortino ratio'].iloc[0] = MathCalc.sortino_ratio(portfolio['Returns'])

        return kpi

    @staticmethod
    def assemble_cum_returns(returns_buyhold, returns_sharpe_optimized_buyhold, returns_minvar_optimized_buyhold):

        """
        This function assembles cumulative returns of all portfolios.
        """
        cum_returns = pd.DataFrame()
        cum_returns['BuyHold 5 Non-corr stocks'] = returns_buyhold
        cum_returns['BuyHold Sharpe-optimized'] = returns_sharpe_optimized_buyhold
        cum_returns['BuyHold MinVar-optimized'] = returns_minvar_optimized_buyhold

        return cum_returns

    @staticmethod
    def assemble_returns(returns_buyhold, returns_sharpe_optimized_buyhold, returns_minvar_optimized_buyhold):

        """
        This function assembles returns of all portfolios.
        """
        returns = pd.DataFrame()
        returns['BuyHold 5 Non-corr stocks'] = returns_buyhold
        returns['BuyHold Sharpe-optimized'] = returns_sharpe_optimized_buyhold
        returns['BuyHold MinVar-optimized'] = returns_minvar_optimized_buyhold

        return returns

    @staticmethod
    def colrow(i):
        """
        This function calculate the row and columns index number based on the total number of subplots in the plot.

        Return:
             row: axis's row index number
             col: axis's column index number
        """

        # Do odd/even check to get col index number
        if i % 2 == 0:
            col = 0
        else:
            col = 1
        # Do floor division to get row index number
        row = i // 2

        return col, row


class Trading:
    """
    This class performs trading and all other functions related to trading
    """

    def __init__(self, dow_stocks_train, dow_stocks_test, dow_stocks_volume):
        self._dow_stocks_test = dow_stocks_test
        self.dow_stocks_train = dow_stocks_train
        self.daily_v = dow_stocks_volume
        self.remaining_stocks()

    @staticmethod
    def slippage_price(price, stock_quantity, day_volume):
        """
        This function performs slippage price calculation using Zipline's volume share model
        https://www.zipline.io/_modules/zipline/finance/slippage.html
        """

        volumeShare = stock_quantity / float(day_volume)
        impactPct = volumeShare ** 2 * PRICE_IMPACT

        if stock_quantity > 0:
            slipped_price = price * (1 + impactPct)
        else:
            slipped_price = price * (1 - impactPct)

        return slipped_price

    @staticmethod
    def commission(num_share, share_value):
        """
        This function computes commission fee of every trade
        https://www.interactivebrokers.com/en/index.php?f=1590&p=stocks1
        """
        trade_value = num_share * share_value
        max_comm_fee = 0.01 * trade_value
        comm_fee = 0.005 * num_share

        if max_comm_fee < comm_fee:
            comm_fee = max_comm_fee
        elif comm_fee <= max_comm_fee and comm_fee > 1.0:
            pass
        elif comm_fee < 1.0 and num_share > 0:
            comm_fee = 1.0
        elif num_share == 0:
            comm_fee = 0.0

        return comm_fee

    def find_efficient_frontier(self, data, selected):
        """
        Find efficient frontier of a portfolio of stocks.
        Returns the stock weights for Sharpe ratio and minimum variance optimized portfolios.
        """

        # reorganise data pulled by setting date as index with
        # columns of tickers and their corresponding adjusted prices
        clean = data.set_index('date')
        table = clean.pivot(columns='ticker')

        # calculate daily and annual returns of the stocks
        returns_daily = table.pct_change()
        returns_annual = returns_daily.mean() * 250

        # get daily and covariance of returns of the stock
        cov_daily = returns_daily.cov()
        cov_annual = cov_daily * 250

        # empty lists to store returns, volatility and weights of imiginary portfolios
        port_returns = []
        port_volatility = []
        sharpe_ratio = []
        stock_weights = []

        # set the number of combinations for imaginary portfolios
        num_assets = len(selected)
        num_portfolios = 50000

        # set random seed for reproduction's sake
        np.random.seed(36)

        # populate the empty lists with each portfolios returns,risk and weights
        for single_portfolio in range(num_portfolios):
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)
            returns = np.dot(weights, returns_annual)
            volatility = np.sqrt(np.dot(weights.T, np.dot(cov_annual, weights)))
            sharpe = returns / volatility
            sharpe_ratio.append(sharpe)
            port_returns.append(returns)
            port_volatility.append(volatility)
            stock_weights.append(weights)

        # a dictionary for Returns and Risk values of each portfolio
        portfolio = {'Returns': port_returns,
                     'Volatility': port_volatility,
                     'Sharpe Ratio': sharpe_ratio}

        # extend original dictionary to accomodate each ticker and weight in the portfolio
        for counter, symbol in enumerate(selected):
            portfolio[symbol + ' Weight'] = [Weight[counter] for Weight in stock_weights]

        # make a nice dataframe of the extended dictionary
        df = pd.DataFrame(portfolio)

        # get better labels for desired arrangement of columns
        column_order = ['Returns', 'Volatility', 'Sharpe Ratio'] + [stock + ' Weight' for stock in selected]

        # reorder dataframe columns
        df = df[column_order]

        # find min Volatility & max sharpe values in the dataframe (df)
        min_volatility = df['Volatility'].min()
        max_sharpe = df['Sharpe Ratio'].max()
        sharpe_portfolio = df.loc[df['Sharpe Ratio'] == max_sharpe]
        min_variance_portfolio = df.loc[df['Volatility'] == min_volatility]

        UserDisplay().plot_efficient_frontier(df, sharpe_portfolio, min_variance_portfolio)
        # use the min, max values to locate and create the two special portfolios
        return sharpe_portfolio, min_variance_portfolio

    def remaining_stocks(self):
        """
        This function finds out the remaining Dow component stocks after the selected stocks are taken.
        """
        dow_remaining = self._dow_stocks_test.drop(RANDOM_STOCK, axis=1)
        self.dow_remaining = [i for i in dow_remaining.columns]

    def construct_book(self, dow_stocks_values, buyhold):
        """
        This function construct the trading book for the buy and hold trading strategy
        """
        portfolio = pd.DataFrame(index=dow_stocks_values.index,
                                 columns=["Total asset", "ProfitLoss", "Returns", "CumReturns"])

        if buyhold:
            portfolio["Total asset"] = dow_stocks_values.sum(axis=1) + (ACCOUNT_FUND * (1 - ALLOCATION_RATIO))
        else:
            portfolio["Total asset"] = dow_stocks_values.sum(axis=1)
        portfolio["ProfitLoss"] = portfolio["Total asset"] - portfolio["Total asset"].shift(1).fillna(
            portfolio["Total asset"][0])
        portfolio["Returns"] = portfolio["Total asset"] / portfolio["Total asset"].shift(1) - 1
        portfolio["CumReturns"] = portfolio["Returns"].add(1).cumprod().fillna(1)
        return portfolio

    def diversified_trade(self, ncs, dow_stocks):
        """
        This function create trading book for the diversifed portfolios
        """
        # Calculate equally weighted fund allocation for each stock
        single_component_fund = SINGLE_TRADING_FUND / len(ncs)
        # Randomly choose the set number of stocks from DJIA pool of component stocks
        share_distribution = single_component_fund / dow_stocks.iloc[0]
        dow_stocks_values = dow_stocks.mul(share_distribution, axis=1)
        portfolio = self.construct_book(dow_stocks_values, True)
        kpi = MathCalc.calc_kpi(portfolio)
        return dow_stocks_values, portfolio, kpi

    def optimized_diversified_trade(self, ncs, sharpe_portfolio, dow_stocks):
        """
        This function create trading book for the diversifed portfolios with asset weights that are optimized by modern portfolio theory
        """

        # Calculate equally weighted fund allocation for each stock
        single_component_fund = SINGLE_TRADING_FUND * sharpe_portfolio.T.iloc[3:].values.flatten()
        # Randomly choose the set number of stocks from DJIA pool of component stocks
        share_distribution = single_component_fund / dow_stocks[ncs].iloc[0]
        dow_stocks_values = dow_stocks[ncs].mul(share_distribution, axis=1)
        portfolio = self.construct_book(dow_stocks_values, True)
        kpi = MathCalc.calc_kpi(portfolio)
        return dow_stocks_values, portfolio, kpi

    def stocks_corr(self, portfolio_longonly_pre):
        """
        This function calculate the correlation coefficient between a portfolio returns and a stock returns
        """

        remaining_corr = pd.Series(index=self.dow_remaining)
        for stock in self.dow_remaining:
            stock_return = MathCalc.calc_return(self.dow_stocks_train[stock])
            remaining_corr[stock] = portfolio_longonly_pre['Returns'][1:].corr(stock_return)
        return remaining_corr.sort_values(ascending=True)

    def find_non_correlate_stocks(self, num_non_corr_stocks):
        """
        This function performs trade with a portfolio starting with the number of stocks specified and
        find the required number of most uncorrelated stocks.
        Only the train set data is used to perform this task to avoid look ahead bias.
        """
        add_stocks = (min(num_non_corr_stocks, len(DJI))) - 1
        # Get the returns of the long only returns of all Dow component stocks during the pre-trading period.
        single_component_fund = SINGLE_TRADING_FUND
        share_distribution = single_component_fund / self.dow_stocks_train[RANDOM_STOCK].iloc[0]
        dow_stocks_values = self.dow_stocks_train[RANDOM_STOCK].mul(share_distribution, axis=1)
        portfolio_longonly_train = self.construct_book(dow_stocks_values, True)

        # find the most uncorrelated stocks with the one randomly selected stock arranged from most
        # uncorrelated to most correlated
        remaining_corr = self.stocks_corr(portfolio_longonly_train)

        # Assemble the non-correlate stocks
        ncs = RANDOM_STOCK

        adding_stocks = [i for i in remaining_corr[0:add_stocks].index]

        # add stocks to the random portfolio stock
        ncs = ncs + adding_stocks

        # Do buy and hold trade with a simple equally weighted portfolio of the 5 non-correlate stocks
        portfolio_values, portfolio_nc_5, kpi_nc_5 = self.diversified_trade(ncs, self.dow_stocks_train[ncs])
        return portfolio_nc_5, kpi_nc_5, ncs


class UserDisplay:
    """
    The class displays plot(s) to users.
    """

    def plot_prediction(self, original, trained, train_len, nn):

        """
        Function to plot all stocks' actual price and price predicted by LSTM model.
        """
        # Set a palette so that all 14 lines can be better differentiated
        color_palette = ['#e6194b', '#3cb44b', '#4363d8']
        fig, ax = plt.subplots(5, 1, figsize=(16, 30))
        plt.subplots_adjust(hspace=0.5)
        for i, s in enumerate(original.columns):
            ax[i].plot(original.index, original[s], '-', label="Original price", linewidth=2, color=color_palette[0])
            ax[i].plot(trained.iloc[:train_len].index, trained[trained.columns[i]].iloc[:train_len], '-',
                       label="Trained price", linewidth=2,
                       color=color_palette[1], alpha=0.8)
            ax[i].plot(trained.iloc[train_len:].index, trained[trained.columns[i]].iloc[train_len:], '-',
                       label="Predicted price", linewidth=2,
                       color=color_palette[2])
            ax[i].set_title("{} trained model price prediction for {}".format(nn, s), fontsize=14)
        plt.legend()
        plt.xlabel('Date')
        plt.ylabel('Stock price')
        #plt.title('Original, trained & predicted stock price trained on {} model'.format(nn))
        plt.subplots_adjust(hspace=0.5)

        # Display and save the graph
        plt.savefig('./test_result/price_prediction_{}.png'.format(nn))
        # Inform user graph is saved and the program is ending.
        print(
            "Plot saved in ./test_result/prediction_{}.png. When done viewing, please close this plot for next plot. Thank You!".format(
                nn))
        plt.show()

    def plot_efficient_frontier(self, risk_return_dict, sharpe_portfolio, min_variance_portfolio):
        """
        Plot the efficient frontier of a portfolio of stocks.
        """

        # plot frontier, max sharpe & min Volatility values with a scatterplot
        plt.style.use('seaborn-dark')
        risk_return_dict.plot.scatter(x='Volatility', y='Returns', c='Sharpe Ratio',
                                      cmap='inferno', edgecolors='black', figsize=(10, 8), grid=True)

        plt.scatter(x=sharpe_portfolio['Volatility'], y=sharpe_portfolio['Returns'], c='red', marker='D', s=200)
        plt.scatter(x=min_variance_portfolio['Volatility'], y=min_variance_portfolio['Returns'], c='blue', marker='D',
                    s=200)
        plt.xlabel('Volatility (Std. Deviation)')
        plt.ylabel('Expected Returns')
        plt.title('Efficient Frontier')
        # Display and save the graph
        plt.savefig('./test_result/efficient_frontier.png')
        # Inform user graph is saved and the program is ending.
        print(
            "Plot saved in ./test_result/efficient_frontier.png. When done viewing, please close this plot for next plot. Thank You!")

        plt.show()

    def plot_portfolio_return(self, cum_returns):
        """
        Function to plot all portfolio cumulative returns.
        """
        # Set a palette so that all 14 lines can be better differentiated
        color_palette = ['#36C4FE', '#FF66F9', '#FF7E66', '#DE0049', '0038E7',
                         '#758CFF', '#4400E7', '#A2ED00', '#00EDC3', '#EECF00', '#EE5C00']
        fig, ax = plt.subplots(figsize=(14, 6))

        # Iterate the compared list to get correlation coefficient array for every compared index
        # Plot the correlation line on the plot canvas
        for i, d in enumerate(cum_returns):
            ax.plot(cum_returns.index, cum_returns[d], '-', label=cum_returns.columns[i], linewidth=2.5,
                    color=color_palette[i])

        plt.legend()
        plt.xlabel('Years')
        plt.ylabel('Cumulative returns')
        plt.title('Cumulative returns for portfolios with different trading models')
        # Display and save the graph
        plt.savefig('./test_result/portfolios_returns.png')
        # Inform user graph is saved and the program is ending.
        print(
            "Plot saved in ./test_result/portfolios_returns.png. When done viewing, please close this plot for next plot. Thank You!")

        plt.show()

    def plot_portfolio_risk(self, returns):
        """
        This function plot the histograms of returns for all portfolios.
        """

        plt.close('all')
        # Define axes, number of rows and columns
        f, ax = plt.subplots(3, 2, figsize=(20, 16))
        plt.subplots_adjust(hspace=0.5)

        for i, d in enumerate(returns):
            # Do odd/even check to col number for plot axes
            col, row = MathCalc.colrow(i)
            ret = returns[d].dropna()
            # plot line graph
            ax[row, col].hist(ret, bins=50, color='darkgreen')
            ax[row, col].axvline(ret.mean(), color='red',
                                 linestyle='-.', linewidth=2.5, label='Mean')
            ax[row, col].axvline(np.median(ret), color='#f1f442',
                                 linestyle='-.', linewidth=2.5, label='Median')
            ax[row, col].axvline(np.median(ret) + ret.std(), color='#b2f441', linestyle='--', linewidth=2,
                                 label='1 x sigma')
            ax[row, col].axvline(np.median(ret) - ret.std(),
                                 color='#b2f441', linestyle='--', linewidth=2)
            ax[row, col].set_title("Returns histogram for portfolio {}".format(returns.columns[i]), fontsize=14)
            ax[row, col].legend()

        plt.savefig('./test_result/portfolios_risk.png')

        print(
            "Plot saved in ./test_result/portfolios_risk.png. When done viewing, please close this plot to end program. Thank You!")

        plt.show()

