# ====== IMPORT ================
import timeit
import warnings
import gym
import rl
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import Callback
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from ann_visualizer.visualize import ann_viz
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy
from rl.policy import GreedyQPolicy


# ====== CALLBACKS TO RECEIVE DATA FROM NN LIBRARY ================
class MyCustomCallback(Callback):
    def __init__(self):
        # Some algorithms compute multiple episodes at once since they are multi-threaded.
        # We therefore use a dictionary that is indexed by the episode to separate episodes
        # from each other.
        self.episode_start = {}
        self.observations = {}
        self.rewards = {}
        self.actions = {}
        self.metrics = {}
        self.step = 0
        self.duration = 0

    def on_train_begin(self, logs):
        """ Print training values at beginning of training """
        self.train_start = timeit.default_timer()
        self.metrics_names = self.model.metrics_names
        print('Training for {} steps ...'.format(self.params['nb_steps']))

    def on_train_end(self, logs):
        """ Print training time at end of training """
        self.duration = timeit.default_timer() - self.train_start
        print('done, took {:.3f} seconds'.format(self.duration))

    def on_episode_begin(self, episode, logs):
        """ Reset environment variables at beginning of each episode """
        self.episode_start[episode] = timeit.default_timer()
        self.observations[episode] = []
        self.rewards[episode] = []
        self.actions[episode] = []
        self.metrics[episode] = []

    def on_episode_end(self, episode, logs):
        """ Compute and print training statistics of the episode when done """
        duration = timeit.default_timer() - self.episode_start[episode]
        episode_steps = len(self.observations[episode])

        # Format all metrics.
        metrics = np.array(self.metrics[episode])
        metrics_template = ''
        metrics_variables = []
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            for idx, name in enumerate(self.metrics_names):
                if idx > 0:
                    metrics_template += ', '
                try:
                    value = np.nanmean(metrics[:, idx])
                    metrics_template += '{}: {:f}'
                except Warning:
                    value = '--'
                    metrics_template += '{}: {}'
                metrics_variables += [name, value]
        metrics_text = metrics_template.format(*metrics_variables)

        nb_step_digits = str(int(np.ceil(np.log10(self.params['nb_steps']))) + 1)
        template = '{nb_steps}: episode: {episode}, duration: {duration:.3f}s, episode steps: {episode_steps}, steps per second: {sps:.0f}, episode reward: {episode_reward:.3f}, mean reward: {reward_mean:.3f} [{reward_min:.3f}, {reward_max:.3f}], mean action: {action_mean:.3f} [{action_min:.3f}, {action_max:.3f}], mean observation: {obs_mean:.3f} [{obs_min:.3f}, {obs_max:.3f}], {metrics}'
        variables = {
            'nb_steps': self.params['nb_steps'],
            'episode': episode + 1,
            'duration': duration,
            'episode_steps': episode_steps,
            'sps': float(episode_steps) / duration,
            'episode_reward': np.sum(self.rewards[episode]),
            'reward_mean': np.mean(self.rewards[episode]),
            'reward_min': np.min(self.rewards[episode]),
            'reward_max': np.max(self.rewards[episode]),
            'action_mean': np.mean(self.actions[episode]),
            'action_min': np.min(self.actions[episode]),
            'action_max': np.max(self.actions[episode]),
            'obs_mean': np.mean(self.observations[episode]),
            'obs_min': np.min(self.observations[episode]),
            'obs_max': np.max(self.observations[episode]),
            'metrics': metrics_text,
        }
        print("My callaback:  " + template.format(**variables))

        # Free up resources.
        del self.episode_start[episode]
        del self.observations[episode]
        del self.rewards[episode]
        del self.actions[episode]
        del self.metrics[episode]

    def on_step_end(self, step, logs):
        """ Update statistics of episode after each step """
        episode = logs['episode']
        self.observations[episode].append(logs['observation'])
        self.rewards[episode].append(logs['reward'])
        self.actions[episode].append(logs['action'])
        self.metrics[episode].append(logs['metrics'])
        self.step += 1

# ====== CART POLE SOLVER ================

ENV_NAME = 'CartPole-v0'
env = gym.make(ENV_NAME)
np.random.seed(50)
env.seed(50)
nb_actions = env.action_space.n

# ====== BUILDING OF NN ================
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('sigmoid'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))

# ====== PRINT OF NN ================
#ann_viz(model, title="Neural network")
print(model.summary())

# ====== SETUP OF MEMORY ================
memory = SequentialMemory(limit=50000, window_length=1)


if True:
    # ====== FIRST POLICY ================
    policy = EpsGreedyQPolicy()

    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
                   target_model_update=1e-2, policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    callbackClass = MyCustomCallback()
    history = dqn.fit(env, nb_steps=2000, visualize=False, callbacks=[callbackClass], verbose=0, nb_max_episode_steps=200)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.subplots_adjust(top=0.85)
    ax.text(0.95, 0.01, 'Execution time ' + str((callbackClass.duration)),
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes,
            color='black', fontsize=12)

    plt.plot(history.history['episode_reward'])
    plt.plot(history.history['nb_episode_steps'])
    plt.title('Trend of reward with EpsGreedyQPolicy')
    plt.ylabel('Reward')
    plt.xlabel('Episodes')
    plt.show()
else:
    # ====== SECOND POLICY ================
    policy = GreedyQPolicy()

    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
                   target_model_update=1e-2, policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    callbackClass = MyCustomCallback()
    history = dqn.fit(env, nb_steps=2000, visualize=False, callbacks=[callbackClass], verbose=0, nb_max_episode_steps=200)

    dqn.test(env, nb_episodes=5, visualize=False)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.subplots_adjust(top=0.85)
    ax.text(0.95, 0.01, 'Execution time ' + str(callbackClass.duration),
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes,
            color='black', fontsize=12)

    plt.plot(history.history['episode_reward'])
    plt.plot(history.history['nb_episode_steps'])
    plt.title('Trend of reward with GreedyQPolicy')
    plt.ylabel('Reward')
    plt.xlabel('Episodes')
    plt.show()
