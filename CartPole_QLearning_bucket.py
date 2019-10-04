# ====== IMPORT ================
import datetime
import gym
import numpy as np
import math
import random
import pybullet as p
import pybullet_data
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d

# ====== CART POLE SOLVER CLASS ================

class CartPole_QLearning():
    def __init__(self, buckets=(1, 1, 4, 5,),
                 n_episodes=2000,
                 gamma=1,
                 type_of_env=0):

        # Selection of environment type (3D or not)
        self.type_of_env = type_of_env
        if type_of_env == 0:
            self.env = gym.make('CartPole-v1')
        else:
            # 3D cart pole
            # Not properly working
            self.env = gym.make('CartPoleBulletEnv-v1')
        self.env.reset()

        # Limits for Q-table discretization (only linear and angular position are discretized)
        # 0 --> cart position (x)
        # 1 --> cart velocity (x')
        # 2 --> pole angle (theta)
        # 3 --> pole angular velocity (theta')

        self.max_x = self.env.observation_space.high[0]
        self.min_x = self.env.observation_space.low[0]
        self.max_theta = self.env.observation_space.high[2]
        self.min_theta = self.env.observation_space.low[2]

        self.max_dot_x = 1
        self.min_dot_x = -1
        self.max_dot_theta = 1
        self.min_dot_theta = -1

        # Iteration limits
        self.n_episodes = n_episodes  # training episodes
        self.max_t = 200  # time limit for each episode

        self.buckets = buckets  # down-scaling feature space to discrete range
        self.n_win_ticks = 195  # average ticks over 100 episodes required for win
        self.gamma = gamma  # discount factor

        # Q matrix
        self.Q = np.zeros(self.buckets + (self.env.action_space.n,))

        # Flag activation
        self.debug = False  # Print debug information over the console
        self.plot = True  # Plot the trend of the reward during the n_episodes
        self.render = False  # Show or not the animation of the cart

    def discretize(self, obs):

        upper_bounds = [self.max_x, self.max_dot_x, self.max_theta, self.max_dot_theta]
        lower_bounds = [self.min_x, self.min_dot_x, self.min_theta, self.min_dot_theta]

        ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
        new_obs = [int(round((self.buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
        new_obs = [min(self.buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
        return tuple(new_obs)

    def choose_action(self, state, t):
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
            action = np.argmax(self.Q[state])
            if self.debug:
                print("Action " + str(action))
                print("State " + str(state))
                print("Maximize Q table that has this fields \r\n" + str(self.Q))
        return action

    def update_q(self, state_old, action, reward, state_new, t):
        alpha = 0.7  # Alpha
        discount_rate = self.gamma
        self.Q[state_old][action] += alpha * (reward + discount_rate * np.max(self.Q[state_new]) - self.Q[state_old][action])

    def run(self):
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
            current_state = self.discretize(self.env.reset())
            cumulative_reward = 0
            time = 0
            done = False
            start_time_episode = datetime.datetime.now()

            for t in range(self.max_t):
                if self.render:
                    self.env.render()
                action = self.choose_action(current_state, time)

                if self.debug:
                    print()
                    if action == 1:
                        print("Action: " + "RIGHT")
                    else:
                        print("Action: " + "LEFT")
                new_state_continuos, reward, done, info = self.env.step(action)

                if self.debug:
                    print("Reward: " + str(reward))
                    pole_angle_rad = new_state_continuos[2]
                    pole_angle_degrees = pole_angle_rad * 180 / math.pi
                    print("New cart position: " + str(new_state_continuos[0]))
                    print("New cart velocity: " + str(new_state_continuos[1]))
                    print("New pole position [degrees]: " + str(pole_angle_degrees))
                    print("New pole velocity: " + str(new_state_continuos[3]))
                    print("")

                if done:
                    end_time_episode = datetime.datetime.now()
                    print(str(e) + "/" + str(self.n_episodes) +
                          "; episode steps: " + str(time) +
                          "; duration: " +
                          str((end_time_episode - start_time_episode).total_seconds()) + "s "
                          "; episode reward: " + str(cumulative_reward))
                    break

                # Update structures after action
                new_state_discrete = self.discretize(new_state_continuos)
                self.update_q(current_state, action, reward, new_state_discrete, time)
                current_state = new_state_discrete

                cumulative_reward += reward
                reward_for_each_episode[e] = cumulative_reward
                time += 1

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

# ============= MAIN EXECUTION ============================
if __name__ == "__main__":
    solver = CartPole_QLearning()
    solver.run()