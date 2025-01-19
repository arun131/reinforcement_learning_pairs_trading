import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class DQNAgent:
    def __init__(self, input_shape, action_size, batch_size, state_size, training_config):
        """
        Initializes the DQNAgent with the given parameters and model architecture.
        
        Parameters:
        input_shape (tuple): Shape of the input state.
        action_size (int): Number of possible actions (trade flags).
        batch_size (int): Number of experiences per batch for training.
        state_size (int): Size of the state space.
        training_config (dict): Dictionary containing training hyperparameters.
        """
        self.input_shape = input_shape
        self.action_size = action_size
        self.batch_size = batch_size
        self.state_size = state_size
        self.training_config = training_config

        # Model training configuration
        self.gamma = training_config['gamma']
        self.epsilon_start = self.epsilon = training_config['epsilon_start']
        self.epsilon_end = training_config['epsilon_end']
        self.epsilon_decay = training_config['epsilon_decay']
        self.device = training_config['device']  # 'cuda' or 'cpu'

        # Initialize the model
        self._build_model(training_config['hidden_layer_units'])
        self.optimizer = optim.Adam(self.model.parameters(), lr=training_config['learning_rate'])
        
    def _build_model(self, units_list):
        """
        Builds the deep Q-network (DQN) model using PyTorch.
        
        Parameters:
        units_list (list): List of units for hidden layers.
        
        Returns:
        model (nn.Module): The constructed neural network model.
        """
        # Shared hidden layers
        layers = []
        input_size = self.input_shape[0]

        # Adding hidden layers
        for units in units_list:
            layers.append(nn.Linear(input_size, units))
            layers.append(nn.ReLU())
            input_size = units

        # Stack shared layers into a Sequential model
        self.model = nn.Sequential(*layers).to(self.device)

        # Create two separate output layers (regression and classification)
        self.regression_model = nn.Linear(input_size, 2).to(self.device)  # Regression output: Predict stock quantities
        self.classification_model = nn.Linear(input_size, self.action_size).to(self.device)  # Classification output: Predict trade action
        

    def forward(self, state):
        """
        Forward pass through the network.
        
        Parameters:
        state (torch.Tensor): The current state input to the model.
        
        Returns:
        regression_output (torch.Tensor): Predicted stock quantities.
        classification_output (torch.Tensor): Predicted trade actions.
        """
        x = state
        x = self.model(x)
        import pdb; pdb.set_trace()
        
        # Split output into regression and classification parts
        regression_output = self.regression_model(x)  # Stock quantities
        classification_output = self.classification_model(x)  # Trade actions
        
        return regression_output, classification_output

    def select_action(self, state):
        """
        Selects an action using the epsilon-greedy policy.
        
        Parameters:
        state (np.ndarray): Current state of the environment.
        
        Returns:
        stock1_num (int): Number of stock 1 to buy/sell.
        stock2_num (int): Number of stock 2 to buy/sell.
        trade_flag (int): Action representing the trade type (0: open, 1: close, etc.)
        """
        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device).unsqueeze(0)
        
        if np.random.rand() < self.epsilon:
            # Exploration: Randomly select an action
            self.decay_epsilon()
            stock1_num = np.random.randint(-10, 11)  # Random quantity for stock 1 (range may vary)
            stock2_num = np.random.randint(-10, 11)  # Random quantity for stock 2 (range may vary)
            trade_flag = np.random.choice([0, 1, 2, 3, 4])  # Random trade flag (action)
            return stock1_num, stock2_num, trade_flag
        else:
            # Exploitation: Use the model to predict the best action
            regression_output, classification_output = self.forward(state_tensor)
            
            # Extract predicted quantities for stock1 and stock2
            stock1_num, stock2_num = regression_output.squeeze().cpu().detach().numpy()
            # Choose the trade flag with the highest probability
            trade_flag = torch.argmax(classification_output, dim=1).cpu().detach().numpy()[0]
            
            return int(stock1_num), int(stock2_num), trade_flag

    def decay_epsilon(self):
        """
        Decays epsilon after each action selection to shift from exploration to exploitation.
        """
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def replay(self, experiences):
        """
        Perform experience replay to update the model using a batch of experiences.
        
        Parameters:
        experiences (list): A list of experiences, each of which is a tuple of (state, action, reward, next_state, done).
        """
        # Convert experiences to tensors
        states, actions, rewards, next_states, dones = zip(*experiences)
        stock1_nums, stock2_nums, trade_flags = zip(*actions)
        stock1_nums = torch.tensor(stock1_nums, dtype=torch.long).to(self.device)
        stock2_nums = torch.tensor(stock2_nums, dtype=torch.long).to(self.device)
        trade_flags = torch.tensor(trade_flags, dtype=torch.long).to(self.device) 
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.device)

        # Predict Q-values for current states
        import pdb; pdb.set_trace()
        regression_output, classification_output = self.forward(states)

        # Calculate target values for the regression and classification outputs

        targets_regression = regression_output.clone()
        targets_classification = classification_output.clone()

        # Update the target values using rewards
        for i in range(self.batch_size):
            if dones[i]:
                targets_regression[i] = rewards[i]  # If done, use the reward as target for regression
            else:
                # Use the Bellman equation to calculate the target for regression
                targets_regression[i] = rewards[i] + self.gamma * torch.max(regression_output[i])

            # Update the classification target to the one-hot encoded trade flag
            targets_classification[i] = trade_flags[i]

        # Calculate loss
        loss_regression = nn.MSELoss()(regression_output, targets_regression)
        loss_classification = nn.CrossEntropyLoss()(classification_output, actions)

        total_loss = loss_regression + loss_classification

        # Perform backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

    def update_target_model(self):
        """Update target model with current model weights (if needed)."""
        pass  # If using target network, implement weight copying here
