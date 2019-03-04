[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"

# StarTrader: An Intelligent Trading Agent Development Using Deep Reinforcement Learning

### Introduction

This project aims to create a trading agent in an [OpenAI Gym](https://gym.openai.com/) environment and trained by deep reinforcement learning algorithms.
Two Gym environments are created to serve the purpose, one for training (StarTrader-v0), another testing
(StarTraderTest-v0). Both versions of StarTrader will utilize Gym's baseline implmentation of Deep deterministic policy gradient (DDPG). 

A portfolio of five stocks (out of 27 Dow Jones Industrial Average stocks) are selected based on non-correlation factor. StarTrader will trade these five non-correlated stocks by learning to maximize total asset (portfolio value + current account balance) as its goal. During the trading process, StarTrader-v0 will also optimize the portfolio by deciding how many stock units to trade for each of the five stocks.

![Trained Agent][image1]

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

Tensorflow for CPU:
```pip3 install --upgrade tensorflow```

Tensorflow for GPU: 
```pip3 install --upgrade tensorflow-gpu```

Installing Tensorflow GPU allows faster training time if your machine has nVidia GPU(s) built-in. 
However, Tensorflow GPU version requires the installation of the right cuDNN and CUDA, these pages provide instructions on how it can be done: 

[Ubuntu](https://www.tensorflow.org/install/install_linux)

[MacOS](https://www.tensorflow.org/install/install_mac (Tensorflow 1.2 no longer provides GPU support for MacOS) )

[Windows](https://www.tensorflow.org/install/install_windows)
	
5. Place the StarTrader and StarTraderTest folders in this repository to your machine's OpenAI Gym's environment folder: 

gym/envs/
	
6. Replace the __init__.py file in the following folder with the __ini__.py provided in this repository: 

gym/envs/__init__.py
  
7. Place run.py in baselines folder to the folder where you want to execute run.py, for example:

From Gym's installation: 
baselines/baselines/run.py

To: 
run.py
	
8. Place 'data' folder to the folder where run.py resides
  
/data/
   
9. Replace ddpg.py from Gym's installation to the ddpg.py in this repository:

In your machine Gym's installation: 
baselines/baselines/ddpg/ddpg.py

replaced by the ddpg.py in repository: 
baselines/baselines/ddpg/ddpg.py

10. Replace ddpg_learner.py from Gym's installation to the ddpg_learner.py in this repository:

In your machine Gym's installation: 
baselines/baselines/ddpg/ddpg_learner.py

replaced by the ddpg_learner.py in repository: 
baselines/baselines/ddpg/ddpg_learner.py
   
11. Place feature_select.py and data_preprocessing.py in this repository into the same folder as run.py

12. Place the following folders in this repository into the folder where your run.py resides

/test_result/
/train_result/
/model/
    
You do not need to include the folders' content, they will be generated when the program executes. If contents are included, they will be replaced once program executes.

12. Under the folder where run.py resides enter the following command:

To train agent:
```python -m run --alg=ddpg --env=StarTrader-v0 --network=lstm --num_timesteps=2e4```

To test agent:
```python -m run --alg=ddpg --env=StarTrader-v0 --network=lstm --num_timesteps=2e4```


