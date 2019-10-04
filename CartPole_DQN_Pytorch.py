# ====== IMPORT ================
import datetime
import pybullet as p
import pybullet_data
import random
import gym
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

# ====== CART POLE SOLVER CLASS ================

class CartPole_DQN():
    def __init__(self, n_episodes=2000,
                 n_win_ticks=195,
                 max_env_steps=None,
                 gamma=1.0,
                 alpha=0.1,
                 alpha_decay=0.01,
                 batch_size=64,
                 type_of_env=0):

        # Selection of environment type (3D or not)
        self.type_of_env = type_of_env
        if type_of_env == 0:
            self.env = gym.make('CartPole-v0')
        else:
            # 3D cart pole
            # Not properly working
            self.env = gym.make('CartPoleBulletEnv-v0')
        self.env.reset()

        np.random.seed(50)
        self.env.seed(50)

        self.memory = deque(maxlen=100000)

        # Iteration limits
        self.n_episodes = n_episodes  # training episodes
        self.max_t = 200  # time limit for each episode

        self.gamma = gamma
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.n_win_ticks = n_win_ticks
        self.batch_size = batch_size
        if max_env_steps is not None: self.env._max_episode_steps = max_env_steps

        # Flag activation
        self.debug = False  # Print debug information over the console
        self.plot = True  # Plot the trend of the reward during the n_episodes
        self.render = False  # Show or not the animation of the cart

        # Init model
        self.model = Sequential()
        self.model.add(Dense(24, input_dim=4, activation='relu'))
        self.model.add(Dense(48, activation='relu'))
        self.model.add(Dense(2, activation='linear'))
        # self.model = Sequential()
        # self.model.add(Flatten(input_shape=(1,) + self.env.observation_space.shape))
        # self.model.add(Dense(16))
        # self.model.add(Activation('relu'))
        # self.model.add(Dense(4))
        # self.model.add(Activation('linear'))
        self.model.compile(loss='mse', optimizer=Adam(lr=self.alpha, decay=self.alpha_decay))
        print(self.model.summary())



    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        epsilon = 0.3  # ϵ for greedy: explore_rate

        # Select a random action
        if random.uniform(0, 1) < epsilon:
            # Choose action randomly
            if random.uniform(0, 1) > 0.5:
                action = 0
            else:
                action = 1
        # Select the action with the highest q
        else:
            action = np.argmax(self.model.predict(state))
            if self.debug:
                print("Action " + str(action))
                print("State " + str(state))
        return action

    def preprocess_state(self, state):
        return np.reshape(state, [1, 4])

    def replay(self, batch_size):
        x_batch, y_batch = [], []
        minibatch = random.sample(
            self.memory, min(len(self.memory), batch_size))
        for state, action, reward, next_state, done in minibatch:
            y_target = self.model.predict(state)
            y_target[0][action] = reward if done else reward + self.gamma * np.max(self.model.predict(next_state)[0])
            x_batch.append(state[0])
            y_batch.append(y_target[0])

        self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)

    def run(self):
        scores = deque(maxlen=100)

        if self.type_of_env == 1:
            physicsClient = p.connect(p.DIRECT)  # or p.DIRECT for non-graphical version
            p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
            p.setGravity(0, 0, -9.81)
            planeId = p.loadURDF("plane.urdf")
            #cartStartPos = [0, 0, 0]
            #cartStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
            cartPoleId = p.loadURDF("cartPole.urdf")
            if self.debug:
                print('=========================')
                print(p.getNumJoints(cartPoleId))
                print('jointIndex jointName jointType qIndex uIndex flags jointDamping jointFriction jointLowerLimit jointUpperLimit jointMaxForce jointMaxVelocity linkName jointAxis parentFramePos parentFrameOrn parentIndex')
                print(p.getJointInfo(cartPoleId, 0))
                print(p.getJointInfo(cartPoleId, 1))
                print('=========================')

        reward_for_each_episode = [0 for x in range(self.n_episodes)]
        start_time = datetime.datetime.now()
        print("timestamp =" + str(start_time))

        for e in range(self.n_episodes):

            # Initialization
            state = self.preprocess_state(self.env.reset())
            i = 0
            cumulative_reward = 0
            time = 0
            done = False
            start_time_episode = datetime.datetime.now()

            for t in range(self.max_t):
                if self.render:
                    self.env.render()

                action = self.choose_action(state)
                if self.debug:
                    print()
                    if action == 1:
                        print("Action: " + "RIGHT")
                    else:
                        print("Action: " + "LEFT")

                next_state, reward, done, _ = self.env.step(action)
                if self.debug:
                    print("Reward: " + str(reward))
                    pole_angle_rad = next_state[2]
                    pole_angle_degrees = pole_angle_rad * 180 / math.pi
                    print("New cart position: " + str(next_state[0]))
                    print("New cart velocity: " + str(next_state[1]))
                    print("New pole position [degrees]: " + str(pole_angle_degrees))
                    print("New pole velocity: " + str(next_state[3]))
                    print("")

                if done:
                    end_time_episode = datetime.datetime.now()
                    print(str(e) + "/" + str(self.n_episodes) +
                          "; episode steps: " + str(time) +
                          "; duration: " +
                          str((end_time_episode - start_time_episode).total_seconds()) + "s "
                          "; episode reward: " + str(cumulative_reward))
                    break

                next_state = self.preprocess_state(next_state)
                self.remember(state, action, reward, next_state, done)
                state = next_state

                cumulative_reward += reward
                reward_for_each_episode[e] = cumulative_reward
                time += 1


            scores.append(i)
            mean_score = np.mean(scores)
            self.replay(self.batch_size)

        end_time = datetime.datetime.now()
        print("timestamp =" + str(end_time))
        if self.plot:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            fig.subplots_adjust(top=0.85)
            ax.text(0.95, 0.01, 'Execution time ' + str((end_time - start_time)),
                    verticalalignment='bottom', horizontalalignment='right',
                    transform=ax.transAxes,
                    color='black', fontsize=12)

            plt.plot([x for x in range(self.n_episodes)], reward_for_each_episode, 'b', label='Reward')
            ysmoothed = gaussian_filter1d(reward_for_each_episode, sigma=2)
            plt.plot([x for x in range(self.n_episodes)], ysmoothed, 'r', label='Reward smoothed')
            plt.axhline(y=self.n_win_ticks, color='g', linestyle='-', label='Threshold ticks')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.legend()
            plt.show()

        return e


if __name__ == '__main__':
    agent = CartPole_DQN()
    agent.run()