import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
from pair_trading_env import PairtradingEnv
from deep_q_network import DQNAgent

# Define the list of stocks/cryptos for analysis
stock_list = ['BTC-USD', 'ETH-USD', 'DOGE-USD']
stock_tickers = ' '.join(stock_list)

# Download historical data
start_date = "2024-12-01"
end_date = "2024-12-04"
data = yf.download(tickers=stock_tickers, start=start_date, end=end_date, interval='5m')

# Extract adjusted close prices
data_close = data['Close'].copy()

# Calculate price changes
price_change_columns = []
for stock in data_close.columns:
    change_col = stock + '_change'
    price_change_columns.append(change_col)
    data_close.loc[:, change_col] = data_close[stock] - data_close[stock].shift(1)

# Compute pairwise correlations for price changes
correlations = {}
for col1 in price_change_columns:
    for col2 in price_change_columns:
        if col1 != col2 and (col1, col2) not in correlations and (col2, col1) not in correlations:
            correlations[(col1, col2)] = data_close[col1].corr(data_close[col2])

# Identify the pair with the highest correlation
best_pair = max(correlations, key=correlations.get)
pair_stocks = [best_pair[0][:-7], best_pair[1][:-7]]  # Extract original stock names

# Filter data for the selected pair
selected_data = data_close[pair_stocks]

# Visualize price changes for the selected pair
plt.plot(selected_data.index, selected_data[pair_stocks[1]], label=pair_stocks[1], color='red')
plt.title(f"Price Changes of {pair_stocks[0]} and {pair_stocks[1]}")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()

# Prepare data for the environment
stock1_prices = selected_data[pair_stocks[0]].values
stock2_prices = selected_data[pair_stocks[1]].values

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
stock1_prices = scaler.fit_transform(stock1_prices.reshape(-1, 1)).flatten()
stock2_prices = scaler.fit_transform(stock2_prices.reshape(-1, 1)).flatten()

# Initialize the environment with the selected pair
env = PairtradingEnv(stock1_prices, stock2_prices, max_trade_period=300)

# Initialize DQN Agent
input_shape = env.observation_space.shape
action_size = env.action_space.n
batch_size = 32
state_size = input_shape[0]
training_config = {
    'gamma': 0.99,
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay': 0.995,
    'learning_rate': 0.001,
    'device': 'cpu',
    'hidden_layer_units': [64, 64]
}
agent = DQNAgent(input_shape, action_size, batch_size, state_size, training_config)

# Train the agent
num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        # Select action from the agent
        action = agent.select_action(state)

        # Take a step in the environment
        next_state, reward, done, info = env.step(action)

        # Store the experience in the replay buffer (for training)
        experiences = [(state, action, reward, next_state, done)]
        agent.replay(experiences)

        # Update the total reward and state
        total_reward += reward
        state = next_state

    print(f"Episode {episode+1}/{num_episodes}, Total Reward: {total_reward}")

# Save the trained model
torch.save(agent.model.state_dict(), "dqn_pair_trading_model.pth")

# Plot the final portfolio value after training
portfolio_values = []
state = env.reset()
done = False
while not done:
    action = agent.select_action(state)
    next_state, reward, done, info = env.step(action)
    portfolio_values.append(info['portfolio'].value)
    state = next_state

plt.plot(portfolio_values)
plt.title("Portfolio Value Over Time")
plt.xlabel("Time Steps")
plt.ylabel("Portfolio Value")
plt.show()
