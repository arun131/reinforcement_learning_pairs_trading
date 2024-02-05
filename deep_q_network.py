from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

class DQNAgent:
  def __init__(self, input_shape, action_size, batch_size, state_size, training_config):
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
    self._build_model(training_config['hidden_layer_units'])

  def _build_model(self, units_list):
    # Input layer
    input_layer = Input(shape=self.input_shape, name='input')

    # Shared hidden layers
    shared_hidden_layer = input_layer
    for units in units_list:
        shared_hidden_layer = Dense(units, activation='relu')(shared_hidden_layer)

    # Regression output
    regression_units = 2
    regression_output = Dense(regression_units, activation='linear', name='regression_output')(shared_hidden_layer)

    # Classification output
    classification_units = self.action_size
    classification_output = Dense(classification_units, activation='sigmoid', name='classification_output')(shared_hidden_layer)

    # Create the model
    self.model = Model(inputs=input_layer, outputs=[regression_output, classification_output])

    # Compile the model
    self.model.compile(optimizer='adam',
                  loss={'regression_output': 'mse', 'classification_output': 'binary_crossentropy'},
                  metrics={'regression_output': 'mae', 'classification_output': 'accuracy'})


  def select_action(self, state):

      if np.random.rand() < self.epsilon:
          self.decay_epsilon()
          return np.random.rand(1,2) * 100,   np.random.rand(1, 5)
      else:
          q_values = self.model.predict([state].reshape(1, -1))
          return np.argmax(q_values[0])
    
  def decay_epsilon(self):
      self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


