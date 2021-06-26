from torch import nn
import torch
import gym
from collections import deque
import itertools
import numpy as np
import random
import qutip
from qutip import sigmax, sigmay, sigmaz
from Model import Network

GAMMA = 0.99
BATCH_SIZE = 32
BUFFER_SIZE = 50000
MIN_REPLAY_SIZE = 1000
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 10000
TARGET_UPDATE_FREQ = 1000
LEARNING_RATE = 5e-4

BOOST_CONTROL = 0.01
AMPLITUDE_ERR = 0.01

THETA = 0
THETA_DOT = 1
AMP = 2
AMP_DOT = 3


class QuantumEnvironment:
    """
    The model is implemeneted as such - we have some energy gap w0. We induce a Hamiltonian in
    the x-y plane whose parameters are the angle θ and amplitude A. The model keeps θ_dot and
    A_dot. At every time step, the environment user is asked whether he wants, for both θ_dot
    and A_dot, whether he wants to increase/decrease them. The parameters are updated, a time
    step is applied.
    The inputs to the network are:
        <σ_x>,
        <σ_y>,
        <σ_z>,
        A_x,
        A_y,
        theta_dot,
        A_dot.
    """

    N_ACTIONS = 4
    INPUT_SIZE = 7

    def __init__(self, energy_gap, runtime, dt, skips):
        """
        Add driving of the form envelope(t)*n*sigma to the Hamiltonian.
        :param energy_gap: Induced energy gap. Qobj type. i.e. w0*sigmaz()
        :param runtime: duration of the simulation
        :param dt: time step
        :param skips: Number of time steps passed between each action.
                      This was not implemented yet.
        :return:
        """
        self.energy_gap = energy_gap * sigmaz()

        self.dt = dt
        self.runtime = runtime
        self.steps = 0

        self.skips = skips
        self.state = qutip.basis(2, 0)

        self.ham_parameters = [0] * 4

    def state_to_vec(self):
        """
        Vector bloch representation of the state self.state
        :return: list of expectation values for the state self.state.
        """
        psi = self.state
        psi_d = self.state.dag()
        x = psi_d * sigmax() * psi
        y = psi_d * sigmay() * psi
        z = psi_d * sigmaz() * psi
        return [abs(x[0][0][0]), abs(y[0][0][0]), abs(z[0][0][0])]

    def fidelity(self):
        """
        Fidelity with respect to |1> (which is the target state)
        :return: F(|ψ>,|1>) of type float.
        """
        psi = self.state
        down = qutip.basis(2, 1)
        proj = (psi.dag() * down)[0][0][0]
        return abs(proj * proj)

    def step(self, action):
        """
        Perform a single action: 
                dθ/dt increase/decrease
                dA/dt increase/decrease
        so 2*2 actions. Observe the ad
        :param action: an integer n∈[
        :return:
        """
        self.steps += 1
        # Update Hamiltonian parameters (including its derivatives)
        #        0  1  2  3
        # θ_dot  -  -  +  +
        # A_dot  -  +  -  +
        theta_boost = (action // 2 - 0.5) * 2
        amp_boost = (action % 2 - 0.5) * 2

        self.ham_parameters[THETA_DOT] += theta_boost * BOOST_CONTROL
        self.ham_parameters[AMP_DOT] += amp_boost * BOOST_CONTROL

        self.ham_parameters[THETA] += self.ham_parameters[THETA_DOT] * self.dt
        self.ham_parameters[AMP] += abs(self.ham_parameters[AMP_DOT] * self.dt)

        amp = self.ham_parameters[AMP]
        amp_d = self.ham_parameters[AMP_DOT]
        theta_d = self.ham_parameters[THETA_DOT]
        theta = self.ham_parameters[THETA]

        old_fidelity = self.fidelity()

        # We want a high cost for high amps and high frequencies
        reward = -(amp * amp + theta_d * theta_d) * AMPLITUDE_ERR

        H = self.energy_gap + amp * (np.cos(theta) * sigmax() + np.sin(theta) * sigmay())
        U = (1j * H * self.dt).expm()
        self.state = (U * self.state).unit()
        new_fidelity = self.fidelity()

        fidelity_shift = new_fidelity - old_fidelity

        reward += fidelity_shift
        # I made this condition to punish a decrease in fidelity.
        # If you understand why, to save space, you can change it to
        # reward += (1 + fidelity_shift < 0) * fidelity_shift
        if fidelity_shift < 0:
            reward += fidelity_shift

        done = (self.steps * self.dt) >= self.runtime

        # This is what the neural network sees
        observation = self.state_to_vec() + [amp * np.cos(theta),
                                             amp * np.sin(theta),
                                             amp_d,
                                             theta_d]

        # Some useless parameter resulting from a previous implementation.
        info = {}
        return observation, reward, done, info

    def reset(self):
        """
        Restarts steps and randomizes a state.
        :return: 7 parameters of the state and Hamiltonian.
        """

        c1 = 1j * random.random() + random.random()
        c2 = 1j * random.random() + random.random()

        psi = c1 * qutip.basis(2, 0) + c2 * qutip.basis(2, 1)
        self.state = psi.unit()

        self.steps = 0
        self.ham_parameters = list(np.random.uniform(0, 2, 4))

        amp = self.ham_parameters[AMP]
        amp_d = self.ham_parameters[AMP_DOT]
        theta_d = self.ham_parameters[THETA_DOT]
        theta = self.ham_parameters[THETA]
        observation = self.state_to_vec() + [amp * np.cos(theta),
                                             amp * np.sin(theta),
                                             amp_d,
                                             theta_d]
        return observation

    def sample(self):
        """
        :return: A random action represented by an integer.
        """
        return random.randint(0, self.N_ACTIONS - 1)


class Trainer:
    """
    Supervises the the training of a given network.
    """

    def __init__(self, net):
        energy_gap, runtime, dt, skips = 1, 20, 0.01, 5
        self.env = QuantumEnvironment(energy_gap, runtime, dt, skips)

        self.replay_buffer = deque(maxlen=BUFFER_SIZE)
        self.rew_buffer = deque(maxlen=BUFFER_SIZE)

        self.episode_reward = 0.0

        self.online_net = Network(self.env)
        self.target_net = Network(self.env)

        self.target_net.load_state_dict(self.online_net.state_dict())
        self.initialize_buffers()

    def initialize_buffers(self):
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

            self.replay_buffer.append(transition)
            self.rew_buffer.append(rew)

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

        # The following loop never stops.
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
            self.replay_buffer.append(transition)
            obs = new_obs

            self.episode_reward += rew

            if done:
                self.rew_buffer.append(self.env.fidelity())
                obs = self.env.reset()
                self.episode_reward = 0.0

            # Start gradient step
            transitions = random.sample(self.replay_buffer, BATCH_SIZE)

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

            # # logging
            # if step % 10000 == 0:
            #     print()
            #     print('Step ', step)
            #     print('Avg rew ', np.mean(self.rew_buffer))
