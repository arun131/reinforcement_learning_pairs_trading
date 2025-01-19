import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from portfolio import Portfolio

class PairtradingEnv(gym.Env):
    """
    Pair Trading Environment for reinforcement learning.
    
    Observations:
        - 0: Price of stock1
        - 1: Number of stock1 in hand
        - 2: Price of stock2
        - 3: Number of stock2 in hand
        - 4: Cash in hand

    Actions:
        - stock1_act: Ratio of stock1 to buy/sell
        - stock2_act: Ratio of stock2 to buy/sell
        - trade_flag: Action type (0: open, 1: close, 2: add, 3: swap, 4: hold)
    
    Reward:
        - Reward is the change in portfolio value from the last step.

    Episode Termination:
        - Ends after max_trade_period or at the end of the data.

    """
    
    metadata = {
        'render.modes': ['human']
    }

    def __init__(self, stock1, stock2, cash=1e5, max_trade_period=200):
        """
        Initialize the pair trading environment with stock prices and portfolio.
        
        Parameters:
        stock1 (list): List of prices for stock 1.
        stock2 (list): List of prices for stock 2.
        cash (float): Initial cash available for trading.
        max_trade_period (int): Maximum number of trading periods per episode.
        """
        self.stock1 = stock1
        self.stock2 = stock2
        self.cash = cash
        self.portfolio = Portfolio(self.cash)

        self.observation_space = spaces.Box(low=np.array([0, 0, 0, 0, 0]), high=np.array([np.inf, np.inf, np.inf, np.inf, np.inf]), dtype=np.float32)
        self.action_space = spaces.Discrete(5)  # 5 possible actions
        
        self.total_trade_period = len(stock1)
        self.max_trade_period = min(max_trade_period, self.total_trade_period)

        self.start_trade_time = 0
        self.current_trade_time = 0 
        
        self.seed()
        self.viewer = None

    def seed(self, seed=None):
        """Set the random seed for reproducibility."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def show(self):
        """Display the current state of the environment."""
        print(f"Total Trade Period: {self.total_trade_period}")
        print(f"Max Trade Period: {self.max_trade_period}")
        print(f"Start Trade Time: {self.start_trade_time}")
        print(f"Current Trade Time: {self.current_trade_time}")
        self.portfolio.show()

    def step(self, action):
        """
        Execute a trading action and calculate reward.
        
        Parameters:
        action (tuple): A tuple containing stock1 action, stock2 action, and trade flag.
        
        Returns:
        (numpy.ndarray, float, bool, dict): Next state, reward, done, info
        """
        stock1_num, stock2_num, trade_flag = action
        
        stock1_price = self.stock1[self.current_trade_time]
        stock2_price = self.stock2[self.current_trade_time]
        
        # Store old value to calculate reward
        old_value = self.portfolio.value    
        self.portfolio.updatePrice(stock1_price, stock2_price)

        # Compute the reward as the relative change in portfolio value
        reward = (self.portfolio.value / self.cash) - 1.0

        # Execute the action
        if trade_flag == 0:
            self.portfolio.openPosition(num1=stock1_num, num2=stock2_num)
        elif trade_flag == 1:
            self.portfolio.closePosition()
        elif trade_flag == 2:
            self.portfolio.addPosition(num1=stock1_num, num2=stock2_num)
        elif trade_flag == 3:
            self.portfolio.closePosition()
            self.portfolio.updatePrice(stock1_price, stock2_price)
            self.portfolio.openPosition(num1=stock1_num, num2=stock2_num)
        elif trade_flag == 4:
            pass  # No action
        
        # Check if the episode is done
        done = False
        if self.current_trade_time >= self.start_trade_time + self.max_trade_period - 1:
            done = True
        self.current_trade_time += 1

        # Get next price differences for observation
        price1 = self.stock1[self.current_trade_time]
        price2 = self.stock2[self.current_trade_time]
        price1diff = price1 - self.stock1[self.current_trade_time - 1]
        price2diff = price2 - self.stock2[self.current_trade_time - 1]

        # Return next state, reward, done, and additional info
        info = {'portfolio': self.portfolio}
        next_state = np.array([self.portfolio.stock1_num, price1, self.portfolio.stock2_num, price2, self.portfolio.cash])
        return next_state, reward, done, info

    def reset(self):
        """Reset the environment to a random starting point."""
        self.portfolio = Portfolio(cash=self.cash)
        
        # Randomly choose a start point for trading
        self.start_trade_time = np.random.randint(self.total_trade_period - self.max_trade_period - 5)
        self.current_trade_time = self.start_trade_time

        stock1_price = self.stock1[self.current_trade_time]
        stock2_price = self.stock2[self.current_trade_time]

        # Reset trade info
        self.trade_info = [self.portfolio, stock1_price, 0, stock2_price, 0]

        # Return initial state
        return np.array([self.portfolio.stock1_num, stock1_price, self.portfolio.stock2_num, stock2_price, self.portfolio.cash])

    def render(self, mode='human'):
        """Render the visualization process (Not implemented in this version)."""
        raise NotImplementedError()

    def close(self):
        """Clean up any resources (Not used in this version)."""
        if self.viewer:
            self.viewer.close()
            self.viewer = None
