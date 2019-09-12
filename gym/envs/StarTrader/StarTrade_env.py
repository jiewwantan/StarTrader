# --------------------------- IMPORT LIBRARIES -------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from gym.utils import seeding
import gym
from gym import spaces
import data_preprocessing as dp

# ------------------------- GLOBAL PARAMETERS -------------------------
# Start and end period of historical data in question
START_TRAIN = datetime(2008, 12, 31)
END_TRAIN = datetime(2017, 2, 12)
START_TEST = datetime(2017, 2, 12)
END_TEST = datetime(2019, 2, 22)

STARTING_ACC_BALANCE = 100000
NUMBER_NON_CORR_STOCKS = 5
MAX_TRADE = 10
TRAIN_RATIO = 0.8
PRICE_IMPACT = 0.1

# Pools of stocks to trade
DJI = ['MMM', 'AXP', 'AAPL', 'BA', 'CAT', 'CVX', 'CSCO', 'KO', 'DIS', 'XOM', 'GE', 'GS', 'HD', 'IBM', 'INTC', 'JNJ',
       'JPM', 'MCD', 'MRK', 'MSFT', 'NKE', 'PFE', 'PG', 'UTX', 'UNH', 'VZ', 'WMT']

DJI_N = ['3M','American Express', 'Apple','Boeing','Caterpillar','Chevron','Cisco Systems','Coca-Cola','Disney'
         ,'ExxonMobil','General Electric','Goldman Sachs','Home Depot','IBM','Intel','Johnson & Johnson',
         'JPMorgan Chase','McDonalds','Merck','Microsoft','NIKE','Pfizer','Procter & Gamble',
         'United Technologies','UnitedHealth Group','Verizon Communications','Wal Mart']

#Market and macroeconomic data to be used as context data
CONTEXT_DATA = ['^GSPC', '^DJI', '^IXIC', '^RUT', 'SPY', 'QQQ', '^VIX', 'GLD', '^TYX', '^TNX' , 'SHY', 'SHV']


# ------------------------------ PREPROCESSING ---------------------------------
print ("\n")
print ("############################## Welcome to the playground of Star Trader!!   ###################################")
print ("\n")
print ("Hello, I am Star, I am learning to trade like a human. In this playground, I trade stocks and optimize my portfolio.")
print ("\n")

print ("Starting to pre-process data for trading environment construction ... ")
# Data Preprocessing
dataset = dp.DataRetrieval()
dow_stocks_train, dow_stocks_test = dataset.get_all()
dow_stock_volume = dataset.components_df_v[DJI]
portfolios = dp.Trading(dow_stocks_train, dow_stocks_test, dow_stock_volume.loc[START_TEST:END_TEST])
_, _, non_corr_stocks = portfolios.find_non_correlate_stocks(NUMBER_NON_CORR_STOCKS)
feature_df = dataset.get_feature_dataframe (non_corr_stocks)
context_df = dataset.get_feature_dataframe (CONTEXT_DATA)

# With context data
input_states = pd.concat([context_df, feature_df], axis=1)
input_states.to_csv('./data/ddpg_input_states.csv')
# Without context data
#input_states = feature_df
feature_length = len(input_states.columns)
data_length = len(input_states)
stock_price = dataset.components_df_o[non_corr_stocks]
stock_volume = dataset.components_df_v[non_corr_stocks]
stock_price.to_csv('./data/ddpg_stock_price.csv')

print("\n")
print("Base on non-correlation preference, {} stocks are selected for portfolio construction:".format(NUMBER_NON_CORR_STOCKS))

for stock in non_corr_stocks:
    print(DJI_N[DJI.index(stock)])
print("\n")

print("Pre-processing and stock selection complete, trading starts now ...")
print("_______________________________________________________________________________________________________________")


# ------------------------------ CLASSES ---------------------------------

class StarTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, day = START_TRAIN):

        """
        Initializing the trading environment, trading parameters starting values are defined.
        """
        self.iteration = 0
        self.day = day
        self.done = False


        # trading agent's action with low and high as the maximum number of stocks allowed to sell or buy,
        # defined using Gym's Box action space function
        self.action_space = spaces.Box(low = -MAX_TRADE, high = MAX_TRADE,shape = (NUMBER_NON_CORR_STOCKS,),dtype=np.int8)

        # [account balance]+[unrealized profit/loss] +[number of features, 36]+[portfolio stock of 5 stocks holdings]
        self.full_feature_length = 2 + feature_length
        print("full length", self.full_feature_length)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape = (self.full_feature_length + NUMBER_NON_CORR_STOCKS,))

        # Sliding the timeline window day-by-day, skipping the non-trading day as data is not available
        wrong_day = True
        add_day = 0
        while wrong_day:
            try:
                temp_date = self.day + timedelta(days=add_day)
                self.data = input_states.loc[temp_date]
                self.day = temp_date
                wrong_day = False
            except:
                add_day += 1

        self.timeline = [self.day]
        # The money in the trading account
        self.acc_balance = [STARTING_ACC_BALANCE]
        self.total_asset = self.acc_balance
        self.portfolio_asset = [0.0]
        self.buy_price =  np.zeros((1,NUMBER_NON_CORR_STOCKS)).flatten()
        # Unrealized profit and loss
        self.unrealized_pnl = [0.0]
        #The value of all-stock holdings
        self.portfolio_value = 0.0


        # The state of the trading environment, defined by account balance, unrealized profit and loss, relevant
        # stock technical data & current stock holdings
        self.state = self.acc_balance + self.unrealized_pnl + self.data.values.tolist() + [0 for i in range(NUMBER_NON_CORR_STOCKS)]

        # The current reward of the agent.
        self.reward = 0
        self._seed()
        self.reset()

    def _sell(self, idx, action):
        """
        Perform and record sell transactions. Commissions and slippage are taken into account.
        """

        # Only need to sell the unit recommended by the trading agent, not necessarily all stock unit.
        num_share = min(abs(int(action)) , self.state[idx + self.full_feature_length])
        commission = dp.Trading.commission(num_share, stock_price.loc[self.day][idx])
        # Calculate slipped price. Though, at max trading volume of 10 shares, there's hardly any slippage
        transacted_price = dp.Trading.slippage_price(stock_price.loc[self.day][idx], -num_share, stock_volume.loc[self.day][idx])
        
        # If there is existing stock holding
        if self.state[idx + self.full_feature_length] > 0:

            # Update account balance after transaction
            self.state[0] += (transacted_price * num_share) - commission
            # Update stock holding
            self.state[idx + self.full_feature_length] -= num_share
            # Reset transacted buy price record to 0.0 if there is no more stock holding
            if self.state[idx + self.full_feature_length] == 0.0:
                self.buy_price[idx] = 0.0
        else:
            pass

    def _buy(self, idx, action):
        """
        Perform and record buy transactions. Commissions and slippage are taken into account.
        """

        # Calculate the maximum possible number of stock unit the current cash can buy
        available_unit = self.state[0] // stock_price.loc[self.day][idx]
        num_share = min(available_unit, int(action))
        # Deduct the traded amount from account balance. If available balance is not enough to purchase stock unit
        # recommended by trading agent's action, just use what is left.
        commission = dp.Trading.commission(num_share, stock_price.loc[self.day][idx])
        # Calculate slipped price. Though, at max trading volume of 10 shares, there's hardly any slippage
        transacted_price = dp.Trading.slippage_price(stock_price.loc[self.day][idx], num_share,
                                                      stock_volume.loc[self.day][idx])

        # Revise number of share to trade if account balance does not have enough
        if (self.state[0] - commission) < transacted_price * num_share:
            num_share = (self.state[0] - commission) // transacted_price
        self.state[0] -= ( transacted_price * num_share ) + commission


        # If there are existing stock holding already, calculate the average buy price
        if self.state[idx + self.full_feature_length] > 0.0:
            existing_unit = self.state[idx + self.full_feature_length]
            previous_buy_price = self.buy_price[idx]
            additional_unit = min(available_unit, int(action))
            new_holding = existing_unit + additional_unit
            self.buy_price[idx] = ((existing_unit * previous_buy_price ) + (transacted_price * additional_unit))/ new_holding
        # if there is no existing stock holding, simply record the current buy price
        elif self.state[idx + self.full_feature_length] == 0.0:
            self.buy_price[idx] = transacted_price

        # Update stock holding at its index
        self.state[idx + self.full_feature_length] += num_share


    def step(self, actions):
        """
        The step of an episode. Perform all activities of an episode.
        """

        # Episode ends when timestep reaches the last day in feature data
        self.done = self.day >= END_TRAIN
        # Uncomment below to run a quick test
        #self.done = self.day >= START_TRAIN + timedelta(days=10)

        # If it is the last step, plot trading performance
        if self.done:
            print("@@@@@@@@@@@@@@@@@")
            print("Iteration", self.iteration-1)
            # Construct trading book and save to a spreadsheet for analysis
            trading_book = pd.DataFrame(index=self.timeline, columns=["Cash balance", "Portfolio value", "Total asset", "Returns", "Cum Returns"])
            trading_book["Cash balance"] = self.acc_balance
            trading_book["Portfolio value"] = self.portfolio_asset
            trading_book["Total asset"] = self.total_asset
            trading_book["Returns"] = trading_book["Total asset"] / trading_book["Total asset"].shift(1) - 1
            trading_book["CumReturns"] = trading_book["Returns"].add(1).cumprod().fillna(1)
            trading_book.to_csv('./train_result/trading_book_train_{}.csv'.format(self.iteration-1))

            kpi = dp.MathCalc.calc_kpi(trading_book)
            kpi.to_csv('./train_result/kpi_train_{}.csv'.format(self.iteration-1))
            print("===============================================================================================")
            print(kpi)
            print("===============================================================================================")

            # Visualize results
            plt.plot(trading_book.index, trading_book["Cash balance"], 'g', label='Account cash balance',alpha=0.8,)
            plt.plot(trading_book.index, trading_book["Portfolio value"], 'r', label='Portfolio value',alpha=1,lw=1.5)
            plt.plot(trading_book.index, trading_book["Total asset"], 'b', label='Total asset',alpha=0.6,lw=3)
            plt.xlabel('Timeline', fontsize=12)
            plt.ylabel('Value', fontsize=12)
            plt.title('Portfolio value + account fund evolution @ train iteration {}'.format(self.iteration-1), fontsize=13)
            plt.tight_layout()
            plt.legend()
            plt.savefig('./train_result/asset_evolution_train_{}.png'.format(self.iteration-1))
            plt.show()
            plt.close()

            plt.plot(trading_book.index, trading_book["CumReturns"], 'g', label='Cummulative returns')
            plt.xlabel('Timeline', fontsize=12)
            plt.ylabel('Returns', fontsize=12)
            plt.title('Cummulative returns @ train iteration {}'.format(self.iteration-1), fontsize=13)
            plt.tight_layout()
            plt.legend()
            plt.savefig('./train_result/cummulative_returns_train_{}.png'.format(self.iteration-1))
            plt.show()
            plt.close()

            return self.state, self.reward, self.done, {}

        else:
            # Portfolio value is current holdings multiply with current respective stock prices
            portfolio_value = sum(np.array(stock_price.loc[self.day]) * np.array(self.state[self.full_feature_length:]))
            # Total asset is account balance + portfolio value
            total_asset_starting = self.state[0] + portfolio_value

            # Sort the trade order in increasing order, stocks with higher number of units to sell will be
            # transacted first. Stocks with lesser number of units to buy will be transacted first
            sorted_actions = np.argsort(actions)
            # Get the stocks to be sold
            sell_stocks = sorted_actions[:np.where(actions < 0)[0].shape[0]]

            # Alternatively, sell with static order
            #sell_stocks = np.where(actions < 0)[0].flatten()
            #np.random.shuffle(sell_stocks)
            for stock_idx in sell_stocks:
                self._sell(stock_idx, actions[stock_idx])

            # Get the stocks to be bought
            buy_stocks = sorted_actions[::-1][:np.where(actions > 0)[0].shape[0]]
            # Alternatively, buy with static order
            #buy_stocks = np.where(actions > 0)[0].flatten()
            #np.random.shuffle(buy_stocks)
            for stock_idx in buy_stocks:
                self._buy(stock_idx, actions[stock_idx])

            # Update date and skip some date since not every day is trading day
            self.day += timedelta(days=1)
            wrong_day = True
            add_day = 0
            while wrong_day:
                try:
                    temp_date = self.day + timedelta(days=add_day)
                    self.data = input_states.loc[temp_date]
                    self.day = temp_date
                    wrong_day = False
                except:
                    add_day += 1

            # Calculate unrealized profit and loss for existing stock holdings
            self.unrealized_pnl = np.sum(np.array(stock_price.loc[self.day] - self.buy_price) * np.array(
                    self.state[self.full_feature_length:]))

            # Current state space
            self.state = [self.state[0]] + [self.unrealized_pnl] + self.data.values.tolist() + list(self.state[self.full_feature_length:])
            # Portfolio value is the current stock prices multiply with their respective holdings
            portfolio_value = sum(np.array(stock_price.loc[self.day]) * np.array(self.state[self.full_feature_length:]))
            # Total asset = account balance + portfolio value
            total_asset_ending = self.state[0] + portfolio_value

            # Update account balance statement
            self.acc_balance = np.append(self.acc_balance, self.state[0])

            # Update portfolio value statement
            self.portfolio_asset = np.append(self.portfolio_asset, portfolio_value)

            # Update total asset statement
            self.total_asset = np.append(self.total_asset, total_asset_ending)

            # Update timeline
            self.timeline = np.append(self.timeline, self.day)

            # Get the agent to consider gain-to-pain or lake ratio and be responsible for it if it has traded long enough
            if len(self.total_asset) > 9:
                returns = dp.MathCalc.calc_return(pd.Series(self.total_asset))

                self.reward = total_asset_ending - total_asset_starting \
                              + (100*dp.MathCalc.calc_gain_to_pain(returns))\
                              - (500 * dp.MathCalc.calc_lake_ratio(pd.Series(returns).add(1).cumprod().fillna(1)))
                              #+ (50 * dp.MathCalc.sharpe_ratio(pd.Series(returns)))

            # If agent has not traded long enough, it only has to bear total asset difference  at the end of the day
            else:
                self.reward = total_asset_ending - total_asset_starting

        return self.state, self.reward, self.done, {}

    def reset(self):
        """
        Reset the environment once an episode end.

        """
        self.acc_balance = [STARTING_ACC_BALANCE]
        self.total_asset = self.acc_balance
        self.portfolio_asset = [0.0]
        self.buy_price = np.zeros((1, NUMBER_NON_CORR_STOCKS)).flatten()
        self.unrealized_pnl = [0]
        self.day = START_TRAIN
        self.portfolio_value = 0.0

        wrong_day = True
        add_day = 0
        while wrong_day:
            try:
                temp_date = self.day + timedelta(days=add_day)
                self.data = input_states.loc[temp_date]
                self.day = temp_date
                wrong_day = False
            except:
                add_day += 1

        self.timeline = [self.day]

        self.state = self.acc_balance + self.unrealized_pnl + self.data.values.tolist() + [0 for i in range(NUMBER_NON_CORR_STOCKS)]
        self.iteration += 1

        return self.state
    
    def render(self, mode='human'):
        """
        Render the environment with current state.

        """
        return self.state

    def _seed(self, seed=None):
        """
        Seed the iteration.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
