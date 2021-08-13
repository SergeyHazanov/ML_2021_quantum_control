import numpy as np
import qutip as qt

import itertools
import random

from Model import NET_INPUT_SIZE

THETA_BOOST_CONTROL = 1e-2
PHI_BOOST_CONTROL = 1e-2
AMP_BOOST_CONTROL = 1e-2

COUPLING = 1


class QuantumEnvironment:

    def __init__(self, energy_gap, runtime, dt):
        # For know we are working in the int. picture, thus the zero.
        self.energy_gap = 0 * energy_gap

        self.dt = dt
        self.runtime = runtime
        self.steps = 0

        self.state = qt.tensor(qt.basis(2, 0), qt.basis(2, 0))

        self.ham_theta = 0.0
        self.ham_theta_dot = 0.0

        self.ham_phi = 0.0
        self.ham_phi_dot = 0.0

        self.ham_amp = 0.0

        # create action lookup table
        theta_dot_list = [-1, 0, 1]
        phi_dot_list = [-1, 0, 1]
        amp_list = [-1, 0, 1]
        self.action_lookup = list(itertools.product(theta_dot_list, phi_dot_list, amp_list))
        self.num_actions = len(self.action_lookup)

    def state2vec(self, qubit=1):
        """
        Bloch vector representation of the state self.state
        :return: list of expectation values for the state self.state.
        """
        x, y, z = qt.expect(qt.sigmax(), qt.ptrace(self.state, qubit)),\
                  qt.expect(qt.sigmay(), qt.ptrace(self.state, qubit)),\
                  qt.expect(qt.sigmaz(), qt.ptrace(self.state, qubit))
        return [x, y, z]

    def fidelity(self, qubit=1):
        """
        fidelity with respect to |1> (which is the target state)
        :return: F(|ψ>,|1>) of type float.
        """
        return qt.fidelity(qt.ptrace(self.state, qubit), qt.basis(2, 1))

    def step(self, action):
        """
        Perform a single action: increase or decrease omega.
        so 3*3*3 actions. Observe the ad
        :param action: an integer
        :return:
        """

        prev_fidelity = self.fidelity(qubit=1)

        self.steps += 1
        if action > (self.num_actions - 1) or action < 0 or type(action) is not int:
            raise Exception('Wrong action value!')
        else:
            self.ham_theta_dot += self.action_lookup[action][0] * THETA_BOOST_CONTROL
            self.ham_phi_dot += self.action_lookup[action][1] * PHI_BOOST_CONTROL
            self.ham_amp += self.action_lookup[action][1] * AMP_BOOST_CONTROL
            # keep amplitude above zero:
            self.ham_amp = self.ham_amp if self.ham_amp > 0 else 0

        self.ham_theta += self.ham_theta_dot * self.dt

        hx = self.ham_amp * np.sin(self.ham_theta) * np.cos(self.ham_phi)
        hy = self.ham_amp * np.sin(self.ham_theta) * np.sin(self.ham_phi)
        hz = self.energy_gap + self.ham_amp * np.cos(self.ham_theta)

        ham_tot = hx * qt.tensor(qt.sigmax(), qt.identity(2)) + \
                  hy * qt.tensor(qt.sigmay(), qt.identity(2)) + \
                  hz * qt.tensor(qt.sigmaz(), qt.identity(2)) + \
                  COUPLING * qt.tensor(qt.sigmax(), qt.sigmaz())

        unitary_op = (- 1j * ham_tot * self.dt).expm()

        self.state = (unitary_op * self.state).unit()

        delta_fidelity = self.fidelity(qubit=1) - prev_fidelity
        reward = 1 / np.sqrt(1 - self.fidelity(qubit=1))
        if delta_fidelity < 0:
            reward = 0
        else:
            reward = (1 + delta_fidelity) ** 2 * reward

        if self.fidelity(qubit=1) > 0.999:
            done = True
            reward += 1e3
        elif (self.steps * self.dt) >= self.runtime:
            done = True
        else:
            done = False

        # This is what the neural network sees
        observation = [hx, hy, hz, self.ham_theta, self.ham_phi] + self.state2vec(qubit=1)

        if len(observation) != NET_INPUT_SIZE:
            print('Wrong observation size')

        # Some useless parameter resulting from a previous implementation where I tried
        # to inherit from gym.env class.
        info = {}

        return observation, reward, done, info

    def reset(self):
        """
        Restarts steps and randomizes a state.
        :return: 8 parameters of the state and Hamiltonian.
        """

        self.state = qt.tensor(qt.basis(2, 0), qt.basis(2, 0))

        self.steps = 0
        self.ham_theta_dot = 0.0
        self.ham_theta = 0.0
        self.ham_phi_dot = 0.0
        self.ham_phi = 0.0
        self.ham_amp = 0.0
        hx = 0
        hy = 0
        hz = self.energy_gap

        observation = [hx, hy, hz, self.ham_theta, self.ham_phi] + self.state2vec()
        return observation

    def sample(self):
        """
        :return: A random action represented by an integer.
        """
        return random.randint(0, self.num_actions - 1)
