from torch import nn
import torch

from collections import deque
import itertools
import numpy as np
import random

from Model import PolicyNetwork
from simulation import QuantumEnvironment

GAMMA = 0.99
BATCH_SIZE = 32
BUFFER_SIZE = 50000
MIN_REPLAY_SIZE = 1000
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 10000
TARGET_UPDATE_FREQ = 1000
LEARNING_RATE = 5e-4


class Trainer:
    """
    Supervises the the training of a given network.
    """

    def __init__(self, energy_gap, runtime, dt, skips):
        self.env = QuantumEnvironment(energy_gap, runtime, dt, skips)

        self.__replay_buffer = deque(maxlen=BUFFER_SIZE)
        self.__rew_buffer = deque(maxlen=BUFFER_SIZE)

        self.__episode_reward = 0.0

        self.online_net = PolicyNetwork(self.env)
        self.target_net = PolicyNetwork(self.env)

        self.target_net.load_state_dict(self.online_net.state_dict())
        self.__initialize_buffers()

    def __initialize_buffers(self):
        """
        Fills the buffers with transitions and corresponding rewards.
        """
        obs = self.env.reset()

        bar = (range(MIN_REPLAY_SIZE))
        for _ in bar:
            action = self.env.sample()

            new_obs, rew, done, _ = self.env.step(action)
            transition = (obs, action, rew, done, new_obs)
            obs = new_obs

            self.__replay_buffer.append(transition)
            # this following isn't actually correct but it doesnt affect the training.
            self.__rew_buffer.append(self.env.fidelity())

            if done:
                obs = self.env.reset()

    def save_net(self, path):
        # Todo - we also need to figure out where in the training loop to use it.
        # torch.save(self.network.state_dict(), path)
        pass

    def load_net(self, path):
        # Todo - likewise. If a ready net exists, train on it.
        pass

    def train(self):
        obs = self.env.reset()

        optimizer = torch.optim.Adam(self.online_net.parameters(), lr=LEARNING_RATE)

        # TODO - The following loop never stops. we need to apply a apply a save network condition
        # TODO - make this work with cuda if it is even possible.
        for step in itertools.count():
            epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])

            rnd_sample = random.random()  # single sample ~ U[0, 1]

            if rnd_sample <= epsilon:
                # Make a random action
                action = self.env.sample()
            else:
                # Use previous knowledge
                action = self.online_net.act(obs)

            new_obs, rew, done, info = self.env.step(action)
            transition = (obs, action, rew, done, new_obs)
            self.__replay_buffer.append(transition)
            obs = new_obs

            self.__episode_reward += rew

            if done:
                self.__rew_buffer.append(self.env.fidelity())
                obs = self.env.reset()
                self.__episode_reward = 0.0

            # Start gradient step
            transitions = random.sample(self.__replay_buffer, BATCH_SIZE)

            obses = np.asarray([t[0] for t in transitions])
            actions = np.asarray([t[1] for t in transitions])
            rews = np.asarray([t[2] for t in transitions])
            dones = np.asarray([t[3] for t in transitions])
            new_obses = np.asarray([t[4] for t in transitions])

            obses_t = torch.as_tensor(obses, dtype=torch.float32)
            actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1)
            rews_t = torch.as_tensor(rews, dtype=torch.float32).unsqueeze(-1)
            dones_t = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(-1)
            new_obses_t = torch.as_tensor(new_obses, dtype=torch.float32)

            # Compute targets
            target_q_values = self.target_net(new_obses_t)
            # the following line is equiv to argmax in dim1. We pick what the network
            # thinks is the best action.
            max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

            targets = rews_t + GAMMA * (1 - dones_t) * max_target_q_values

            # Compute loss
            q_values = self.online_net(obses_t)

            action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)

            loss = nn.functional.smooth_l1_loss(action_q_values, targets)

            # Gradient descent
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # update target network
            if step % TARGET_UPDATE_FREQ == 0:
                self.target_net.load_state_dict(self.online_net.state_dict())

            # logging
            step_skips = 70000
            if step % step_skips == 0:
                print('Step ', step_skips / 1000, 'k')
                print('Avg rew ', np.mean(self.__rew_buffer), '\n')
