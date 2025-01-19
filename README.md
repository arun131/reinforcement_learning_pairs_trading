**Introduction**

This repository implements a Deep Q-Network (DQN) based trading agent for pair trading cryptocurrencies or stocks. 
The agent leverages historical price data to learn trading strategies that exploit price correlations between a chosen pair of assets.

**Dependencies**

This code requires the following Python libraries:

numpy
yfinance
matplotlib.pyplot
pandas
scikit-learn
torch
pair_trading_env.py (custom environment class)
deep_q_network.py (custom DQN agent class)

**Instructions**

**Installation**:
Ensure you have the required libraries installed. You can install them using pip:
`pip install -r requirements.txt`

**Data Download and Preprocessing:**

Edit the stock_list variable in the main script (train.py) to specify the list of cryptocurrencies or stocks you want to analyze.
Run the script (python train.py). This will:
Download historical price data for the specified assets using yfinance.
Calculate price changes for each asset.
Identify the pair with the highest correlation in price changes.
Preprocess the data for the chosen pair using MinMax scaling.

**Training the DQN Agent:**

The script initializes a PairtradingEnv environment using the preprocessed data.
It then initializes a DQNAgent with hyperparameters defined in the training_config dictionary.
The agent is trained for a specified number of episodes (num_episodes). During each episode, the agent interacts with the environment, selects actions, and learns from the rewards received.
The trained model is saved as dqn_pair_trading_model.pth.

**Evaluation**:

After training, the script simulates the agent's trading performance on the environment.
It plots the final portfolio value over time steps.

**Work in progress**:
Some elements of the code still dosent work as expected. I need to correct the way the model output to return int values for the predicted actions

**Further Enhancements**

Experiment with different hyperparameters for the DQN agent.
Implement more sophisticated trading strategies within the environment.
Integrate risk management techniques into the agent's decision-making process.
Backtest the trained agent on historical data using VectorBT to evaluate its performance.

**Disclaimer**

This is a for-educational-purposes implementation and should not be considered financial advice. Always conduct your own research before making any investment decisions.
