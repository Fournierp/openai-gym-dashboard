import gym
import numpy as np
from collections import deque
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
from tensorflow.keras.optimizers import Adam


class CartpoleAgent:
    def __init__(self, render=False, episodes=100, frames=200, gamma=0.97, epsilon=1.0, epsilon_min=0.01,
                 epsilon_decay=2*1e-4, alpha=0.618, memory=100000, prod=False, model_name="", neurons=16,
                 batch_size=64, lr=0.01, activation="tanh"):
        """
        Initialise the agent with user input.

        :param render:
        :param episodes:
        :param frames:
        :param gamma:
        :param epsilon:
        :param epsilon_min:
        :param epsilon_decay:
        :param alpha:
        :param memory:
        :param prod:
        :param model_name:
        :param neurons:
        :param batch_size:
        :param lr:
        :param activation:
        """
        self.render = render
        self.env = gym.make('CartPole-v0')
        self.episodes = episodes
        self.frames = frames
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha
        self.memory = deque(maxlen=memory)
        self.prod = prod
        self.batch_size = batch_size
        self.lr = lr
        self.neurons = neurons
        self.model = self.build_model()  # load_model(model_name) if prod else
        self.activation = activation

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

        :return: model
        """
        x = Input(shape=(1, 4))
        hidden = Dense(units=self.neurons, activation="tanh")(x)
        hidden = Dense(units=self.neurons / 2, activation="tanh")(hidden)
        # hidden = Dense(units=self.neurons, activation="tanh")(hidden)
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

        :param state:
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

        :return: shuffle batch
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

        :return:
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
        self.model.fit(np.array(x), np.array(y), batch_size=len(x), verbose=0)

        # Reduce the randomness of the decision making
        self.decay()

    def play(self):
        """
        Function that simulates the game

        :return:
        """
        log = open('tmp.csv', 'w')
        log.write('Episode, Reward, Epsilon')
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
            # Log results
            log.write("\n" + str(e) + ',' + str(i) + ',' + str(self.epsilon))

            if e % 100 == 0 and not self.prod:
                print('Episode {} - Survival time over last 100 episodes was {} frames. -- {}'.
                      format(e, mean_score, self.epsilon))

            if e % self.batch_size:
                self.learn()

        log.close()

    def baseline(self):
        """

        :return:
        """
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
            state = next_state
            i += 1

            # Stop when game is over
            if done:
                break

        print('Baseline - Survival time was {} frames. -- {}'.format(i, self.epsilon))

    def render(self):
        """

        :return:
        """


if __name__ == '__main__':
    agent = CartpoleAgent(episodes=6000, epsilon_decay=0.999)
    agent.baseline()
    agent.play()
