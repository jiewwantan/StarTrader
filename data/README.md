# Capstone Project : Intelligent Trading Agent Development Using Deep Reinforcement Learning
## By Jiew Wan Tan
### 18 March 2019

## Usage instructions

Files included: 

1. /StarTrader/ folder - the environment files
2. /StarTraderTest/ folder - the environment files
3. __init__.py - register StarTrader and StarTraderTest environments to OpenAI Gym
4. run.py -  the file to execute agent's trading
5. /data/ folder - the folder that contains all required data
6. feature_select.py - the module to perform feature selection
7. data_preprocessing.py -  the module to perform data preprocessing
8. ddpg.py - the modified learning algorithm, along with hyperparameter tuning
9. ddpg_learner.py - the modified learning algorithm
10. /train_result/ folder - training results in plots and csv files
11. /test_result/ folder - testing results in plots and csv files
12. /model/ folder - the trained model
13. compare.py - the program to copare StarTrader with other trading strategies
14. JiewWanTan_capstone_report_18Mar2018.pdf - Capstone Report 
15. rnn_lstm_model_dev.ipynb - provides an option for user to train RNN-LSTM model
16. readme.txt - this file


## Prerequisites

Python 3.6 or Anaconda with Python 3.6 environment
Python packages: pandas, numpy, matplotlib, statsmodels, sklearn, tensorflow

The code is written in a Linux machine and has been tested only on a Linux Ubuntu 16.04 system.
Test run will be performed on Windows 10 Pro. 


## Installation instructions:


1. Installation of system packages CMake, OpenMPI on Mac

	- brew install cmake openmpi


2. Activate environemnt and install gym under this environment

	pip install gym

3. Download Official Baseline Package

    Go to directory where you want to install  Gym's baselines package.
	
	Clone the repo:

    git clone https://github.com/openai/baselines.git
	
    cd baselines
	
	pip install -e .

4. Install Tensorflow

    There are several ways of installing Tensorflow, this page provide a good description on how it can be done with system OS, Python version and GPU availability taken into consideration.

	https://www.tensorflow.org/install/

	In short, after environment activation, Tensorflow can be installed with these commands:

	Tensorflow for CPU: pip3 install --upgrade tensorflow

	Tensorflow for GPU: pip3 install --upgrade tensorflow-gpu

	Installing Tensorflow GPU allows faster training if your machine has nVidia GPU(s) built-in. However, Tensorflow GPU version requires the installation of the right cuDNN and CUDA, these pages provide instructions to ensure the right version is installed:
	Ubuntu: 
	https://www.tensorflow.org/install/install_linux

	MacOS: 
	https://www.tensorflow.org/install/install_mac (Tensorflow 1.2 no longer provides GPU support for MacOS) 

	Windows: 
	https://www.tensorflow.org/install/install_windows
	
5. Place the StarTrader and StarTraderTest folders to OpenAI Gym's environment folder: 

	gym/envs/
	
6. Replace the __init__.py file in the following folder with the __ini__.py provided in this package: 

	gym/envs/__init__.py

7. Place run.py in baselines folder to the folder where you want to execute run.py, for example:

   From Gym's installation: 
   baselines/baselines/run.py

   To: 
   run.py

8. Place 'data' folder to the folder where run.py resides

   /data/
   
9. Replace ddpg.py from Gym's baselines installation with ddpg.py provided in this package:

   /baselines/baselines/ddpg/ddpg.py

10.Replace ddpg_learner.py from Gym's baseline installation with ddpg_learner.py in this repository:

   /baselines/baselines/ddpg/ddpg_learner.py

11. Place feature_select.py and data_preprocessing.py in the same folder as run.py

12. Place the following folders in this repository into the folder where your run.py resides

	/test_result/ 
	/train_result/ 
	/model/

    You do not need to include the folders' content, they will be generated when the program executes. If contents are included, they will be replaced once program executes.

10. Under the same baselines folder enter the following command:  (under development)

	To train agent: <br />
	`python -m run --alg=ddpg --env=StarTrader-v0 --network=mlp --num_timesteps=2e4`

	To test agent: <br />
	`python -m run --alg=ddpg --env=StarTraderTest-v0 --network=mlp --num_timesteps=2e3 --num_timesteps=2e3 --load_path='./model/DDPG_trained_model_8'`
	
	If you have trained a better model, replace DDPG_trained_model_8 with your new model.
	
	After training and testing the agent successfully, the default saving path for the first iteration trading book is saved as ./test_result/trading_book_test_1.csv, which will be used by the next program. 

	Next, compare agent performance with benchmark index and other trading strategies: <br />

	`python compare.py`
	

Data and code related to instructions, program execution, train and test results can also be found at https://github.com/jiewwantan/StarTrader/

## Special intructions: 

1. The RNN-LSTM model is trained with Tensorflow-gpu version 1.7.0 and Keras 2.2.4. 

	Equivalent Tensorflow (with or without GPU) and Keras version maybe required to load the trained RNN-LSTM model:
	best_lstm_model.h5

	There may also be other issues such as variation on number of input features that cause failure in loading the RNN-LSTM model. 
	Such as if the user machine has selected a different number of input features. In that case, user may use the provided pre-processed data by uncommenting the line in /StarTrader/StarTrade_env.py

	input_states = pd.read_csv("./data/ddpg_input_states.csv", index_col='Date', parse_dates=True)

	and commenting out the following line: 
	input_states.to_csv('./data/ddpg_input_states.csv')

	If user prefer to trainig his/her own RNN-LSTM, it can be done with the enclosed working notebook: rnn_lstm_model_dev.ipynb 


2. Depends on machine configuration, the following intallation maybe necessary: 

	pip3 install -U numpy
	pip3 install opencv-python
	pip3 install mujoco-py==0.5.7
	pip3 install lockfile
	conda install h5py

	
3. The technical analysis library, TA-Lib may be tricky to install in some machines. The following page is a handy guide: 
	https://goldenjumper.wordpress.com/tag/ta-lib/

	graphiviz which is required to plot the XGBoost tree diagram, can be installed with the following command: 
	Windows: 
	conda install python-graphviz
	Mac/Linux: 
	conda install graphviz


4. Required Python packages: 

	import pandas as pd
	import math
	from time import sleep
	from datetime import datetime as dt
	import talib as tb
	from sklearn.model_selection import train_test_split
	from sklearn.preprocessing import StandardScaler
	from sklearn.preprocessing import Normalizer
	from sklearn.preprocessing import MinMaxScaler

	Libraries required by FeatureSelector()
	import lightgbm as lgb
	import gc
	from itertools import chain
	from sklearn.cluster import KMeans
	from mpl_toolkits.mplot3d import Axes3D
	import matplotlib.pyplot as plt
	from mpl_finance import candlestick_ohlc
	import copy
	from matplotlib.dates import (DateFormatter, WeekdayLocator, DayLocator, MONDAY)


#### Author
Jiew Wan Tan on 18th Mar 2019

