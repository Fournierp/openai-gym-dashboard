import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
from tensorflow.keras.optimizers import Adam


class CartpoleAgent:
    def __init__(self, render=False, episodes=10000, frames=200, gamma=0.97, epsilon=1.0, epsilon_min=0.01,
                 epsilon_decay=0.99, memory=100000, prod=False, neurons=[16, 8], batch_size=64, lr=0.01,
                 activation="tanh", log_file="", model_name="", plot_file=""):
        """
        Initialise the agent with user input.

        :param render: Boolean to render the game or not.
        :param episodes: Number of games used by the agent to learn.
        :param frames: Number of frames per game.
        :param gamma: Discount factor for the reward.
        :param epsilon: Value of randomness of the epsilon-greedy decision making.
        :param epsilon_min: Minimum epsilon value.
        :param epsilon_decay: Rate of decay of epsilon.
        :param memory: Number of frames the agent remembers.
        :param prod: Boolean for training or testing mode.
        :param neurons: Architecture of the DQN.
        :param batch_size: Number of frames fed to the DQN at once.
        :param activation: Activation function.
        :param lr: Learning Rate
        :param log_file: Name of the file the logs will be saved.
        :param model_name: Name the model will be saved to.
        :param plot_file: Name of the file the plot of the reward is saved to.
        """
        self.render = render
        self.env = gym.make('CartPole-v0')
        self.episodes = episodes
        self.frames = frames
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=memory)
        self.prod = prod
        self.batch_size = batch_size
        self.lr = lr
        self.neurons = neurons
        self.activation = activation
        self.model = self.build_model() if model_name == '' else load_model(model_name)
        self.log_file = 'logs/env_CartPole-v0_gamma_' + str(gamma) + '_episodes_' + str(episodes) + "_epsilon_decay_" +\
                        str(epsilon_decay) + '_batch_size_' + str(batch_size) + '_lr_' + str(lr) + '_neurons_' +\
                        str(neurons) + '_activation_' + str(activation) + '.csv' if log_file == '' else log_file
        self.model_file = 'models/env_CartPole-v0_gamma_' + str(gamma) + '_episodes_' + str(episodes) +\
                          '_epsilon_decay_' + str(epsilon_decay) + '_batch_size_' + str(batch_size) + '_lr_' + str(lr)\
                          + '_neurons_' + str(neurons) + '_activation_' + str(activation) if model_name == '' else model_name
        self.plot_file = 'plots/env_CartPole-v0_gamma_' + str(gamma) + '_episodes_' + str(episodes) +\
                         '_epsilon_decay_' + str(epsilon_decay) + '_batch_size_' + str(batch_size) + '_lr_' + str(lr)\
                         + '_neurons_' + str(neurons) + '_activation_' + str(activation) + '.png' \
            if plot_file == '' else plot_file

    def reset(self, epsilon=1.0, memory=100000):
        """
        Reset the attributes that change during play.

        :param epsilon:
        :param memory:
        :return:
        """
        self.env = self.env.reset()
        self.epsilon = epsilon
        self.memory = deque(maxlen=memory)
        self.model = self.build_model()

    def build_model(self):
        """
        Function that creates the Q-Network.

        :return: DQN model
        """
        x = Input(shape=(1, 4))
        hidden = Dense(units=self.neurons[0], activation=self.activation)(x)
        for layer in self.neurons[1:]:
            hidden = Dense(units=layer, activation=self.activation)(hidden)
        y = Dense(2, activation="linear")(hidden)

        model = Model(inputs=x, outputs=y)
        optimizer = Adam(lr=self.lr)
        model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])

        return model

    def decay(self):
        """
        Function that decays the threshold that determines when the agent takes random decisions in training.
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def choose_action(self, state):
        """
        Function that determines what the agent does. In training it will randomly select a random decision to
        accelerate learning through exploration.

        :param state: Environment setting (position of the pole)
        :return: optimal or random decision
        """
        # If the agent is in training and with a random chance.
        if not self.prod and np.random.rand() <= self.epsilon:
            # Return a random decision
            return self.env.action_space.sample()
        # Else, return the optimal decision chosen by the model
        state = np.reshape(state, (1, 1, 4))
        options = self.model.predict([state])

        return np.argmax(options[0])

    def generate_batch(self):
        """
        Function that gets the last (batch_size) elements in memory for training.

        :return: Shuffled batch
        """
        batch = []
        length = len(self.memory)
        for i in range(length - self.batch_size + 1, length):
            batch.append(self.memory[i])
        # Shuffle the data to avoid the order influencing the training
        return np.random.permutation(batch)

    def learn(self):
        """
        Function that trains the model given a batch of data.

        :return: Mean Squared Error of the batch.
        """
        x, y = [], []
        minibatch = self.generate_batch()
        for state, action, reward, next_state, done in minibatch:
            # Get the model prediction for the given state
            state = np.reshape(state, (1, 1, 4))
            y_target = self.model.predict([state])
            # Set the predicted reward for the selected action to be the actual reward added to the predicted reward at
            # the next state (unless it is the last state)
            next_state = np.reshape(next_state, (next_state.shape[0], 1, next_state.shape[1]))
            y_target[0][0][action] = reward if done \
                else reward + self.gamma * np.max(self.model.predict([next_state])[0])
            x.append(state[0])
            y.append(y_target[0])

        # Give the model the data
        hist = self.model.fit(np.array(x), np.array(y), verbose=0)
        # Reduce the randomness of the decision making
        self.decay()

        return hist.history["loss"][0]

    def play(self):
        """
        Function that simulates the game.
        """
        if not self.prod:
            log = open(self.log_file, 'w')
            log.write('Batch,Reward,Loss')
            log.close()
            scores = deque(maxlen=100)

        for e in range(self.episodes):
            # Restart the cartpole
            state = self.env.reset()
            i = 0

            # For each frame in the game loop
            for f in range(self.frames):
                # Render
                if self.render:
                    self.env.render()

                # Select an action
                action = self.choose_action(state)
                # Simulate a frame
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, 4])
                # Save the simulation to the agent memory
                self.memory.append((state, action, reward, next_state, done))
                state = next_state
                i += 1

                # Stop when game is over
                if done:
                    break

            # Record the track run
            scores.append(i)
            mean_score = np.mean(scores)

            if not (e+1) % self.batch_size and not self.prod:
                # Feed a batch into the DQN
                loss = self.learn()
                # Log results to console
                print('Batch {} - Survival time over last {} episodes was {} frames. -- {}'.
                      format((e+1)/self.batch_size, self.batch_size, mean_score, loss))
                if mean_score > 195.0:
                    print('Game is solved at Episode {}'.format(e))
                # Log results to files
                log = open(self.log_file, 'a')
                log.write("\n" + str((e+1)/self.batch_size) + ',' + str(i) + ',' + str(loss))
                log.close()

    def demo_run(self, render):
        """
        Run the game with the Agent making the decisions and not learning.

        :param render: Boolean to display the game or not.
        """
        # Restart the cartpole
        state = self.env.reset()
        i = 0

        # For each frame in the game loop
        for f in range(self.frames):
            # Render
            if render:
                self.env.render()

            # Select an action
            action = self.choose_action(state)
            # Simulate a frame
            next_state, reward, done, _ = self.env.step(action)
            next_state = np.reshape(next_state, [1, 4])
            state = next_state
            i += 1

            # Stop when game is over
            if done:
                break

        print('Demo - Survival time was {} frames.'.format(i))
        self.env.reset()

    def save_model(self):
        """
        Function to save the model.
        """
        # Serialize model to JSON
        model_json = self.model.to_json()
        with open(self.model_file + ".json", "w") as json_file:
            json_file.write(model_json)
        # Serialize weights to HDF5
        self.model.save_weights(self.model_file + ".h5")

    def plot_model(self):
        """
        Plot the reward per episode.
        """
        df = pd.read_csv(self.log_file)
        fig, ax = plt.subplots()
        ax.plot(df.Batch, df.Reward)
        ax.set(xlabel='Batch', ylabel='Reward', title=self.log_file[:-4])
        ax.grid()
        fig.savefig(self.plot_file)


if __name__ == '__main__':
    agent = CartpoleAgent(episodes=100000, epsilon_decay=0.99)
    agent.demo_run(render=False)
    agent.play()
    agent.demo_run(render=True)
    agent.save_model()
    agent.plot_model()
