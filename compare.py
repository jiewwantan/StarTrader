# --------------------------- IMPORT LIBRARIES -------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import data_preprocessing as dp
from sklearn.preprocessing import MinMaxScaler

import keras
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
from keras.layers import Dense, Dropout

# ------------------------- GLOBAL PARAMETERS -------------------------
# Start and end period of historical data in question
START_TRAIN = datetime(2008, 12, 31)
END_TRAIN = datetime(2017, 2, 12)
START_TEST = datetime(2017, 2, 12)
END_TEST = datetime(2019, 2, 22)

STARTING_ACC_BALANCE = 100000
NUMBER_NON_CORR_STOCKS = 5
# Number of times of no-improvement before training is stop.
PATIENCE = 30

# Pools of stocks to trade
DJI = ['MMM', 'AXP', 'AAPL', 'BA', 'CAT', 'CVX', 'CSCO', 'KO', 'DIS', 'XOM', 'GE', 'GS', 'HD', 'IBM', 'INTC', 'JNJ',
       'JPM', 'MCD', 'MRK', 'MSFT', 'NKE', 'PFE', 'PG', 'UTX', 'UNH', 'VZ', 'WMT']

DJI_N = ['3M', 'American Express', 'Apple', 'Boeing', 'Caterpillar', 'Chevron', 'Cisco Systems', 'Coca-Cola', 'Disney'
    , 'ExxonMobil', 'General Electric', 'Goldman Sachs', 'Home Depot', 'IBM', 'Intel', 'Johnson & Johnson',
         'JPMorgan Chase', 'McDonalds', 'Merck', 'Microsoft', 'NIKE', 'Pfizer', 'Procter & Gamble',
         'United Technologies', 'UnitedHealth Group', 'Verizon Communications', 'Wal Mart']

# Market and macroeconomic data to be used as context data
CONTEXT_DATA = ['^GSPC', '^DJI', '^IXIC', '^RUT', 'SPY', 'QQQ', '^VIX', 'GLD', '^TYX', '^TNX', 'SHY', 'SHV']

# --------------------------------- CLASSES ------------------------------------
class Trading:
    def __init__(self, recovered_data_lstm, portfolio_stock_price, portfolio_stock_volume, test_set, non_corr_stocks):
        self.test_set = test_set
        self.ncs = non_corr_stocks
        self.stock_price = portfolio_stock_price
        self.stock_volume = portfolio_stock_volume
        self.generate_signals(recovered_data_lstm)

    def generate_signals(self, predicted_tomorrow_close):
        """
        Generate trade signla from the prediction of the LSTM model
        :param predicted_tomorrow_close:
        :return:
        """

        predicted_tomorrow_close.columns = self.stock_price.columns
        predicted_next_day_returns = (predicted_tomorrow_close / predicted_tomorrow_close.shift(1) - 1).dropna()
        next_day_returns = (self.stock_price / self.stock_price.shift(1) - 1).dropna()
        signals = pd.DataFrame(index=predicted_tomorrow_close.index, columns=self.stock_price.columns)

        for s in self.stock_price.columns:
            for d in next_day_returns.index:
                if predicted_tomorrow_close[s].loc[d] > self.stock_price[s].loc[d] and next_day_returns[s].loc[
                    d] > 0 and predicted_next_day_returns[s].loc[d] > 0:
                    signals[s].loc[d] = 2
                elif predicted_tomorrow_close[s].loc[d] < self.stock_price[s].loc[d] and next_day_returns[s].loc[
                    d] < 0 and predicted_next_day_returns[s].loc[d] < 0:
                    signals[s].loc[d] = -2
                elif predicted_tomorrow_close[s].loc[d] > self.stock_price[s].loc[d]:
                    signals[s].loc[d] = 2
                elif next_day_returns[s].loc[d] > 0:
                    signals[s].loc[d] = 1
                elif next_day_returns[s].loc[d] < 0:
                    signals[s].loc[d] = -1
                elif predicted_next_day_returns[s].loc[d] > 0:
                    signals[s].loc[d] = 2
                elif predicted_next_day_returns[s].loc[d] < 0:
                    signals[s].loc[d] = -1
                else:
                    signals[s].loc[d] = 0
        signals.loc[self.stock_price.index[0]] = [0, 0, 0, 0, 0]
        self.signals = signals

    def _sell(self, stock, sig, day):
        """
        Perform and record sell transactions.
        """

        # Get the index of the stock
        idx = self.ncs.index(stock)

        # Only need to sell the unit recommended by the trading agent, not necessarily all stock unit.
        num_share = min(abs(int(sig)), self.state[idx + 1])
        commission = dp.Trading.commission(num_share, self.stock_price.loc[day][stock])
        # Calculate slipped price. Though, at max trading volume of 10 shares, there's hardly any slippage
        transacted_price = dp.Trading.slippage_price(self.stock_price.loc[day][stock], -num_share,
                                                     self.stock_volume.loc[day][stock])

        # If there is existing stock holding
        if self.state[idx + 1] > 0:
            # Only need to sell the unit recommended by the trading agent, not necessarily all stock unit.
            # Update account balance after transaction
            self.state[0] += (transacted_price * num_share) - commission
            # Update stock holding
            self.state[idx + 1] -= num_share
            # Reset transacted buy price record to 0.0 if there is no more stock holding
            if self.state[idx + 1] == 0.0:
                self.buy_price[idx] = 0.0

        else:
            pass

    def _buy(self, stock, sig, day):
        """
        Perform and record buy transactions.
        """

        idx = self.ncs.index(stock)
        # Calculate the maximum possible number of stock unit the current cash can buy
        available_unit = self.state[0] // self.stock_price.loc[day][stock]
        num_share = min(available_unit, int(sig))
        # Deduct the traded amount from account balance. If available balance is not enough to purchase stock unit
        # recommended by trading agent's action, just use what is left.
        commission = dp.Trading.commission(num_share, self.stock_price.loc[day][stock])
        # Calculate slipped price. Though, at max trading volume of 10 shares, there's hardly any slippage
        transacted_price = dp.Trading.slippage_price(self.stock_price.loc[day][stock], num_share,
                                                     self.stock_volume.loc[day][stock])
        # Revise number of share to trade if account balance does not have enough
        if (self.state[0] - commission) < transacted_price * num_share:
            num_share = (self.state[0] - commission) // transacted_price
        self.state[0] -= (transacted_price * num_share) + commission

        # If there are existing stock holding already, calculate the average buy price
        if self.state[idx + 2] > 0.0:
            existing_unit = self.state[idx + 2]
            previous_buy_price = self.buy_price[idx]
            additional_unit = min(available_unit, int(sig))
            new_holding = existing_unit + additional_unit
            self.buy_price[idx] = ((existing_unit * previous_buy_price) + (
            self.stock_price.loc[day][stock] * additional_unit)) / new_holding
        # if there is no existing stock holding, simply record the current buy price
        elif self.state[idx + 2] == 0.0:
            self.buy_price[idx] = self.stock_price.loc[day][stock]

        # Update stock holding at its index
        self.state[idx + 1] += min(available_unit, int(sig))

    def execute_trading(self, non_corr_stocks):
        """
        This function performs long only trades for the LSTM model.
        """

        # The money in the trading account
        self.acc_balance = [STARTING_ACC_BALANCE]
        self.total_asset = self.acc_balance
        self.portfolio_asset = [0.0]
        self.buy_price = np.zeros((1, len(non_corr_stocks))).flatten()
        # Unrealized profit and loss
        self.unrealized_pnl = [0.0]
        # The value of all-stock holdings
        self.portfolio_value = 0.0

        # The state of the trading environment, defined by account balance, unrealized profit and loss, relevant
        # stock technical data & current stock holdings
        self.state = self.acc_balance + self.unrealized_pnl + [0 for i in range(len(non_corr_stocks))]

        # Slide through the timeline
        for d in self.test_set.index[:-1]:

            signals = self.signals.loc[d]

            # Get the stocks to be sold
            sell_stocks = signals[signals < 0].sort_values(ascending=True)
            # Get the stocks to be bought
            buy_stocks = signals[signals > 0].sort_values(ascending=True)

            for idx, sig in enumerate(sell_stocks):
                self._sell(sell_stocks.index[idx], sig, d)

            for idx, sig in enumerate(buy_stocks):
                self._buy(buy_stocks.index[idx], sig, d)

            self.unrealized_pnl = np.sum(np.array(self.stock_price.loc[d] - self.buy_price) * np.array(
                self.state[2:]))

            # Current state space
            self.state = [self.state[0]] + [self.unrealized_pnl] + list(self.state[2:])
            # Portfolio value is the current stock prices multiply with their respective holdings
            portfolio_value = sum(np.array(self.stock_price.loc[d]) * np.array(self.state[2:]))
            # Total asset = account balance + portfolio value
            total_asset_ending = self.state[0] + portfolio_value

            # Update account balance statement
            self.acc_balance = np.append(self.acc_balance, self.state[0])

            # Update portfolio value statement
            self.portfolio_asset = np.append(self.portfolio_asset, portfolio_value)

            # Update total asset statement
            self.total_asset = np.append(self.total_asset, total_asset_ending)

        trading_book = pd.DataFrame(index=self.test_set.index,
                                    columns=["Cash balance", "Portfolio value", "Total asset", "Returns", "CumReturns"])
        trading_book["Cash balance"] = self.acc_balance
        trading_book["Portfolio value"] = self.portfolio_asset
        trading_book["Total asset"] = self.total_asset
        trading_book["Returns"] = trading_book["Total asset"] / trading_book["Total asset"].shift(1) - 1
        trading_book["CumReturns"] = trading_book["Returns"].add(1).cumprod().fillna(1)
        trading_book.to_csv('./test_result/trading_book_backtest.csv')

        kpi = dp.MathCalc.calc_kpi(trading_book)
        kpi.to_csv('./test_result/kpi_backtest.csv')

        print("\n")
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print(
            " KPI of RNN-LSTM modelled trading strategy for a portfolio of {} non-correlated stocks".format(
                NUMBER_NON_CORR_STOCKS))
        print(kpi)
        print("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")

        return trading_book, kpi


class Data_ScaleSplit:
    """
    This class preprosses data for the LSTM model.
    """

    def __init__(self, X, selected_stocks_price, train_portion):
        self.X = X
        self.stock_price = selected_stocks_price
        self.generate_labels()
        self.scale_data()
        self.split_data(train_portion)


    def generate_labels(self):
        """
        Generate label data for tomorrow's prediction.
        """
        self.Y = self.stock_price.shift(-1)
        self.Y.columns = [c + '_Y' for c in self.Y.columns]

    def scale_data(self):
        """
        Scale the X and Y data with minimax scaller.
        The scaling is done separately for the train and test set to avoid look ahead bias.
        """
        self.XY = pd.concat([self.X, self.Y], axis=1).dropna()
        train_set = self.XY.loc[START_TRAIN:END_TRAIN]
        test_set = self.XY.loc[START_TEST:END_TEST]
        # MinMax scaling
        minmaxed_scaler = MinMaxScaler(feature_range=(0, 1))
        self.minmaxed = minmaxed_scaler.fit(train_set)
        train_set_matrix = minmaxed_scaler.transform(train_set)
        test_set_matrix = minmaxed_scaler.transform(test_set)
        self.train_set_matrix_df = pd.DataFrame(train_set_matrix, index=train_set.index, columns=train_set.columns)
        self.test_set_matrix_df = pd.DataFrame(test_set_matrix, index=test_set.index, columns=test_set.columns)
        self.XY = pd.concat([self.train_set_matrix_df, self.test_set_matrix_df], axis=0)

        # print ("Train set shape: ", train_set_matrix.shape)
        # print ("Test set shape: ", test_set_matrix.shape)

    def split_data(self, train_portion):
        """
        Perform train test split with cut off date defined.
        """
        df_values = self.XY.values
        # split into train and test sets

        train = df_values[:int(train_portion), :]
        test = df_values[int(train_portion):, :]
        # split into input and outputs
        train_X, self.train_y = train[:, :-5], train[:, -5:]
        test_X, self.test_y = test[:, :-5], test[:, -5:]
        # reshape input to be 3D [samples, timesteps, features]
        self.train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
        self.test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
        print("\n")
        print("Dataset shapes >")
        print("Train feature data shape:", self.train_X.shape)
        print("Train label data shape:", self.train_y.shape)
        print("Test feature data shape:", self.test_X.shape)
        print("Test label data shape:", self.test_y.shape)

    def get_prediction(self, model_lstm):
        """
        Get the model prediction, inverse transform scaling to get back to original price and
        reassemble the full XY dataframe.
        """
        # Get the model to predict test_y

        predicted_y_lstm = model_lstm.predict(self.test_X, batch_size=None, verbose=0, steps=None)
        # Get the model to generate train_y
        trained_y_lstm = model_lstm.predict(self.train_X, batch_size=None, verbose=0, steps=None)

        # combine the model generated train_y and test_y to create the full_y
        y_lstm = pd.DataFrame(data=np.vstack((trained_y_lstm, predicted_y_lstm)),
                              columns=[c + '_LSTM' for c in self.XY.columns[-5:]], index=self.XY.index)

        # Combine the original full length y with model generated y
        lstm_y_df = pd.concat([self.XY[self.XY.columns[-5:]], y_lstm], axis=1)
        # Get the full length XY data with the length of model generated y
        lstm_df = self.XY.loc[lstm_y_df.index]
        # Replace the full length XY data's Y with the model generated Y
        lstm_df[lstm_df.columns[-5:]] = lstm_y_df[lstm_y_df.columns[-5:]]
        # Inverse transform it to get back the original data, the model generated y would be transformed to reveal its true predicted value
        recovered_data_lstm = self.minmaxed.inverse_transform(lstm_df)
        # Create a dataframe from it
        self.recovered_data_lstm = pd.DataFrame(data=recovered_data_lstm, columns=self.XY.columns, index=lstm_df.index)
        return self.recovered_data_lstm

    def get_train_test_set(self):
        """
        Get the split X and y data.
        """
        return self.train_X, self.train_y, self.test_X, self.test_y

    def get_all_data(self):
        """
        Get the full XY data and the original stock price.
        """
        return self.XY, self.stock_price

class Model:
    """
    This class contains all the functions required to build a LSTM or LSTM-CNN model
    It also offer an option to load a pre-built model.
    """
    @staticmethod
    def train_model(model, train_X, train_y, model_type):
        """
        Try to load a pre-built model.
        Otherwise fit a new mode with the training data. Once training is done, save the model.
        """
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=PATIENCE)
        if model_type == "LSTM":
            batch_size = 4
            mc = ModelCheckpoint('./model/best_lstm_model.h5', monitor='val_loss', save_weights_only=False,
                                 mode='min', verbose=1, save_best_only=True)
            try:
                model = load_model('./model/best_lstm_model.h5')
                print("\n")
                print("Loading pre-saved model ...")
            except:
                print("\n")
                print("No pre-saved model, training new model.")
                pass
        elif model_type == "CNN":
            batch_size = 8
            mc = ModelCheckpoint('./model/best_cnn_model.h5'.format(symbol), monitor='val_loss', save_weights_only=False,
                                 mode='min', verbose=1, save_best_only=True)
            try:
                model = load_model('./model/best_cnn_model.h5')
                print("\n")
                print("Loading pre-saved model ...")
            except:
                print("\n")
                print("No pre-saved model, training new model.")
                pass
        # fit network
        history = model.fit(
            train_X,
            train_y,
            epochs=500,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=2,
            shuffle=True,
            # callbacks=[es, mc, tb, LearningRateTracker()])
            callbacks=[es, mc])

        if model_type == "LSTM":
            model.save('./model/best_lstm_model.h5')
        elif model_type == "CNN":
            model.save('./model/best_cnn_model.h5')

        return history, model

    @staticmethod
    def plot_training(history,nn):
        """
        Plot the historical training loss.
        """
        # plot history
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.title('Training loss history for {} model'.format(nn))
        plt.savefig('./train_result/training_loss_history_{}.png'.format(nn))
        plt.show()

    @staticmethod
    def build_rnn_model(train_X):
        """
        Build the RNN model architecture.
        """
        # design network
        print("\n")
        print("RNN LSTM model architecture >")
        model = Sequential()
        model.add(LSTM(128, kernel_initializer='random_uniform',
                       bias_initializer='zeros', return_sequences=True,
                       recurrent_dropout=0.2,
                       input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(Dropout(0.5))
        model.add(LSTM(64, kernel_initializer='random_uniform',
                       return_sequences=True,
                       # bias_regularizer=regularizers.l2(0.01),
                       # kernel_regularizer=regularizers.l1_l2(l1=0.01,l2=0.01),
                       # activity_regularizer=regularizers.l2(0.01),
                       bias_initializer='zeros'))
        model.add(Dropout(0.5))
        model.add(LSTM(64, kernel_initializer='random_uniform',
                       # bias_regularizer=regularizers.l2(0.01),
                       # kernel_regularizer=regularizers.l1_l2(l1=0.01,l2=0.01),
                       # activity_regularizer=regularizers.l2(0.01),
                       bias_initializer='zeros'))
        model.add(Dropout(0.5))
        model.add(Dense(5))
        # optimizer = keras.optimizers.RMSprop(lr=0.25, rho=0.9, epsilon=1e-0)
        # optimizer = keras.optimizers.Adagrad(lr=0.0001, epsilon=1e-08, decay=0.00002)
        # optimizer = keras.optimizers.Adam(lr=0.0001)
        # optimizer = keras.optimizers.Nadam(lr=0.0002, beta_1=0.9, beta_2=0.999, schedule_decay=0.004)
        # optimizer = keras.optimizers.Adamax(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
        optimizer = keras.optimizers.Adadelta(lr=0.2, rho=0.95, epsilon=None, decay=0.00001)

        model.compile(loss='mae', optimizer=optimizer, metrics=['mse', 'mae'])
        model.summary()
        print("\n")
        return model



# ------------------------------ Main Program ---------------------------------

def main():
    print("\n")
    print("######################### This program compare performance of trading strategies   ############################")
    print("\n")
    print( "1. Simple Buy and hold strategy of a portfolio with {} non-correlated stocks".format(NUMBER_NON_CORR_STOCKS))
    print( "2. Sharpe ratio optimized portfolio of {} non-correlated stocks".format(NUMBER_NON_CORR_STOCKS))
    print( "3. Minimum variance optimized portfolio of {} non-correlated stocks".format(NUMBER_NON_CORR_STOCKS))
    print( "4. Simple Buy and hold strategy ")
    print( "1. Simple Buy and hold strategy ")

    print("\n")

    print("Starting to pre-process data for trading environment construction ... ")
    # Data Preprocessing
    dataset = dp.DataRetrieval()
    dow_stocks_train, dow_stocks_test = dataset.get_all()
    train_portion = len(dow_stocks_train)
    dow_stock_volume = dataset.components_df_v[DJI]
    portfolios = dp.Trading(dow_stocks_train, dow_stocks_test, dow_stock_volume.loc[START_TEST:END_TEST])
    _, _, non_corr_stocks = portfolios.find_non_correlate_stocks(NUMBER_NON_CORR_STOCKS)
    non_corr_stocks_data = dataset.get_adj_close(non_corr_stocks)
    print("\n")
    print("Base on non-correlation preference, {} stocks are selected for portfolio construction:".format(NUMBER_NON_CORR_STOCKS))

    for stock in non_corr_stocks:
        print(DJI_N[DJI.index(stock)])
    print("\n")

    sharpe_portfolio, min_variance_portfolio = portfolios.find_efficient_frontier(non_corr_stocks_data, non_corr_stocks)
    print("Risk-averse portfolio with low variance:")
    print(min_variance_portfolio.T)
    print("High return portfolio with high Sharpe ratio")
    print(sharpe_portfolio.T)
    dow_stocks = pd.concat([dow_stocks_train, dow_stocks_test], axis=0)

    test_values_buyhold, test_returns_buyhold, test_kpi_buyhold = \
        portfolios.diversified_trade(non_corr_stocks, dow_stocks.loc[START_TEST:END_TEST][non_corr_stocks])
    print("\n")
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    print(" KPI of a simple buy and hold strategy for a portfolio of {} non-correlated stocks".format(NUMBER_NON_CORR_STOCKS))
    print("------------------------------------------------------------------------------------")
    print(test_kpi_buyhold)
    print("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")


    test_values_sharpe_optimized_buyhold, test_returns_sharpe_optimized_buyhold, test_kpi_sharpe_optimized_buyhold =\
        portfolios.optimized_diversified_trade(non_corr_stocks, sharpe_portfolio, dow_stocks.loc[START_TEST:END_TEST][non_corr_stocks])
    print("\n")
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    print(" KPI of a simple buy and hold strategy for a Sharpe ratio optimized portfolio of {} non-correlated stocks".format(NUMBER_NON_CORR_STOCKS))
    print("------------------------------------------------------------------------------------")
    print(test_kpi_sharpe_optimized_buyhold)
    print("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")

    test_values_minvar_optimized_buyhold, test_returns_minvar_optimized_buyhold, test_kpi_minvar_optimized_buyhold = \
        portfolios.optimized_diversified_trade(non_corr_stocks, min_variance_portfolio, dow_stocks.loc[START_TEST:END_TEST][non_corr_stocks])
    print("\n")
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    print(" KPI of a simple buy and hold strategy for a Minimum variance optimized portfolio of {} non-correlated stocks".format(NUMBER_NON_CORR_STOCKS))
    print("------------------------------------------------------------------------------------")
    print(test_kpi_minvar_optimized_buyhold)
    print("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")

    plot = dp.UserDisplay()
    test_returns = dp.MathCalc.assemble_returns(test_returns_buyhold['Returns'],
                                                    test_returns_sharpe_optimized_buyhold['Returns'],
                                                    test_returns_minvar_optimized_buyhold['Returns'])
    test_cum_returns = dp.MathCalc.assemble_cum_returns(test_returns_buyhold['CumReturns'],
                                                        test_returns_sharpe_optimized_buyhold['CumReturns'],
                                                test_returns_minvar_optimized_buyhold['CumReturns'])

    print("\n")
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    print("Buy and hold strategies computation completed. Now creating prediction model using RNN LSTM architecture")
    print("--------------------------------------------------------------------------------------------------------")
    print("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")


    # Use feature data preprocessed by StartTrader, so that they both use the same training data, to have a fair comparison
    input_states = pd.read_csv("./data/ddpg_input_states.csv", index_col='Date', parse_dates=True)
    scale_split = Data_ScaleSplit(input_states, dow_stocks[non_corr_stocks], train_portion)
    train_X, train_y, test_X, test_y = scale_split.get_train_test_set()

    modelling = Model
    model_lstm = modelling.build_rnn_model(train_X)
    history_lstm, model_lstm = modelling.train_model(model_lstm, train_X, train_y, "LSTM")
    print("RNN model loaded, now training the model again, training will stop after {} episodes no improvement")
    modelling.plot_training(history_lstm, "LSTM")
    print("Training completed, loading prediction using the trained RNN model >")
    recovered_data_lstm = scale_split.get_prediction(model_lstm)
    plot.plot_prediction(dow_stocks[non_corr_stocks].loc[recovered_data_lstm.index], recovered_data_lstm[recovered_data_lstm.columns[-5:]] , len(train_X), "LSTM")

    # Get the original stock price with the prediction length
    original_portfolio_stock_price = dow_stocks[non_corr_stocks].loc[recovered_data_lstm.index]
    # Get the predicted stock price with the prediction length
    predicted_portfolio_stock_price = recovered_data_lstm[recovered_data_lstm.columns[-5:]]
    print("Bactesting the RNN-LSTM model now")
    # Run backtest, the backtester is similar to those use by StarTrader too
    backtest = Trading(predicted_portfolio_stock_price, original_portfolio_stock_price, dow_stock_volume[non_corr_stocks].loc[recovered_data_lstm.index],  dow_stocks_test[non_corr_stocks], non_corr_stocks)
    trading_book, kpi = backtest.execute_trading(non_corr_stocks)
    # Load backtest result for StarTrader using DDPG as learning algorithm
    ddpg_backtest = pd.read_csv('./test_result/trading_book_test_1.csv', index_col='Unnamed: 0', parse_dates=True)
    print("Backtesting completed, plotting comparison of trading models")
    # Compare performance on all 4 trading type
    djia_daily = dataset._get_daily_data(CONTEXT_DATA[1]).loc[START_TEST:END_TEST]['Close']
    #print(djia_daily)
    all_benchmark_returns = test_returns
    all_benchmark_returns['DJIA'] = dp.MathCalc.calc_return(djia_daily)
    all_benchmark_returns['RNN LSTM'] = trading_book['Returns']
    all_benchmark_returns['DDPG'] = ddpg_backtest['Returns']
    all_benchmark_returns.to_csv('./test_result/all_strategies_returns.csv')
    plot.plot_portfolio_risk(all_benchmark_returns)

    all_benchmark_cum_returns = test_cum_returns
    all_benchmark_cum_returns['DJIA'] = all_benchmark_returns['DJIA'].add(1).cumprod().fillna(1)
    all_benchmark_cum_returns['RNN LSTM'] = trading_book['CumReturns']
    all_benchmark_cum_returns['DDPG'] = ddpg_backtest['CumReturns']
    all_benchmark_cum_returns.to_csv('./test_result/all_strategies_cum_returns.csv')
    plot.plot_portfolio_return(all_benchmark_cum_returns)


if __name__ == '__main__':
    main()