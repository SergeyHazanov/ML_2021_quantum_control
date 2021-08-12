import numpy as np
import qutip as qt

import itertools
import random


class QuantumEnvironment:

    def __init__(self, energy_gap, runtime, dt):
        # For know we are working in the int. picture, thus the zero.
        self.energy_gap = 0 * energy_gap

        self.dt = dt
        self.runtime = runtime
        self.steps = 0

        self.state = qt.basis(2, 0)

        self.ham_theta = 0.0
        self.ham_omega = 0.0
        self.ham_amp = 0.0

        # create action lookup table
        omega_list = [-1, 0, 1]
        amp_list = [-1, 0, 1]
        self.action_lookup = list(itertools.product(omega_list, amp_list))

