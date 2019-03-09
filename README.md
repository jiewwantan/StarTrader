[//]: # (Image References)

[image1]: https://github.com/jiewwantan/StarTrader/blob/master/train_iterations_9.gif "Training iterations"
[image2]: https://github.com/jiewwantan/StarTrader/blob/master/test_iteration_1.gif "Testing trained model with one iteration"
[image3]: https://github.com/jiewwantan/StarTrader/blob/master/test_result/portfolios_returns.png "Trading strategy performance comparison"

# **StarTrader:** <br />Intelligent Trading Agent Development<br /> with Deep Reinforcement Learning

### Introduction

This project aims to create a trading agent in an [OpenAI Gym](https://gym.openai.com/) environment and trained by deep reinforcement learning algorithms.
Two Gym environments are created to serve the purpose, one for training (StarTrader-v0), another testing
(StarTraderTest-v0). Both versions of StarTrader will utilize Gym's baseline implmentation of Deep deterministic policy gradient (DDPG). 

A portfolio of five stocks (out of 27 Dow Jones Industrial Average stocks) are selected based on non-correlation factor. StarTrader will trade these five non-correlated stocks by learning to maximize total asset (portfolio value + current account balance) as its goal. During the trading process, StarTrader-v0 will also optimize the portfolio by deciding how many stock units to trade for each of the five stocks.

### Training agent on 9 iterations
![Training iterations][image1]

### Testing agent on one iteration 
No learning or model refinement, purely on testing the trained model. 
Trading agent survived the major market correction in 2018 with 1.13 Sharpe ratio. <br />

![Testing trained model with one iteration][image2]

### Compare agent's performance with other trading strategies
DDPG is the best performer in terms of cumulative returns. However with a much less volatile ride, LSTM model has the highest Sharpe ratio (2.28) and Sortino ration (3.62). 
DDPG's reward system shall be modified to yield higher Sharpe and Sortino ratio. 

![Trading strategy performance comparison][image3]

## Prerequisites

Python 3.6 or Anaconda with Python 3.6 environment
Python packages: pandas, numpy, matplotlib, statsmodels, sklearn, tensorflow

The code is written in a Windows machine and has been tested on three operating systems: 
Linux Ubuntu 16.04 & Windows 10 Pro


## Installation instructions:

1. Installation of system packages CMake, OpenMPI on Mac

   ```brew install cmake openmpi```

2. Activate environemnt and install gym under this environment
 
   ```pip install gym```

3. Download Official Baseline Package

   Clone the repo:

   ```
   git clone https://github.com/openai/baselines.git

   cd baselines

   pip install -e .
   ```

4. Install Tensorflow

   There are several ways of installing Tensorflow, this page provide a good description on how it can be done with system OS, Python version and GPU availability taken into consideration.

   https://www.tensorflow.org/install/

   In short, after environment activation, Tensorflow can be installed with these commands: 

   Tensorflow for CPU:<br />
   ```pip3 install --upgrade tensorflow```

   Tensorflow for GPU: <br />
   ```pip3 install --upgrade tensorflow-gpu```

   Installing Tensorflow GPU allows faster training if your machine has nVidia GPU(s) built-in. 
   However, Tensorflow GPU version requires the installation of the right cuDNN and CUDA, these pages provide instructions to ensure the right version is installed: 

   [Ubuntu](https://www.tensorflow.org/install/install_linux)

   [MacOS](https://www.tensorflow.org/install/install_mac (Tensorflow 1.2 no longer provides GPU support for MacOS) )

   [Windows](https://www.tensorflow.org/install/install_windows)
	
5. Place StarTrader and StarTraderTest folders in this repository to your machine's OpenAI Gym's environment folder: 

   gym/envs/
	
6. Replace the ```__init__.py``` file in the following folder with the ```__ini__.py``` provided in this repository: 

   ```gym/envs/__init__.py```
  
7. Place run.py in baselines folder to the folder where you want to execute run.py, for example:

   From Gym's installation: <br />
   ```baselines/baselines/run.py```

   To: <br />
   ```run.py```
	
8. Place 'data' folder to the folder where run.py resides
  
   ```/data/```
   
9. Replace ddpg.py from Gym's installation with the ddpg.py in this repository:

   In your machine Gym's installation: <br />
   ```baselines/baselines/ddpg/ddpg.py```

   replaced by the ddpg.py in repository: <br />
   ```baselines/baselines/ddpg/ddpg.py```

10. Replace ddpg_learner.py from Gym's installation with the ddpg_learner.py in this repository:

      In your machine Gym's installation: <br />
      ```baselines/baselines/ddpg/ddpg_learner.py```

      replaced by the ddpg_learner.py in repository: <br />
      ```baselines/baselines/ddpg/ddpg_learner.py```
   
11. Place feature_select.py and data_preprocessing.py in this repository into the same folder as run.py

12. Place the following folders in this repository into the folder where your run.py resides

     ```/test_result/```<br />
     ```/train_result/```<br />
     ```/model/```<br />
    
      You do not need to include the folders' content, they will be generated when the program executes. If contents are included, they  will be replaced once program executes.

12. Under the folder where run.py resides enter the following command:

      To train agent:<br />
      ```python -m run --alg=ddpg --env=StarTrader-v0 --network=mlp --num_timesteps=2e4```

      To test agent:<br />
      ```python -m run --alg=ddpg --env=StarTraderTest-v0 --network=mlp --num_timesteps=2e3 --load_path='./model/DDPG_trained_model_8```
      
      If you have trained a better model, replace ```DDPG_trained_model_8``` with your new model. 
      
      After training and testing agent successfully, pick the best DDPG trading book saved in ./test_result/trading_book_test_1.csv or modify filename in compare.py. <br />
      Compare agent performance with benchmark and other trading strategies:<br />
      ```python compare.py```

## Special intructions: 
1. Depends on machine configuration, the following intallation maybe necessary: 

   ```pip3 install -U numpy```<br /> 
   ```pip3 install opencv-python```<br />
   ```pip3 install mujoco-py==0.5.7```<br />
   ```pip3 install lockfile```<br />
   
2. The technical analysis library, TA-Lib may be tricky to install in some machines. The following page is a handy guide: 
https://goldenjumper.wordpress.com/tag/ta-lib/

   graphiviz which is required to plot the XGBoost tree diagram, can be installed with the following command: <br />
   Windows: <br />
   ```conda install python-graphviz```<br />
   Mac/Linux: <br />
   ```conda install graphviz```<br />

