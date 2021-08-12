import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from qutip import sigmax, sigmay, sigmaz
import random
from Model import NET_INPUT_SIZE
import itertools

THETA_BOOST_CONTROL = 1e-2
AMP_BOOST_CONTROL = 1e-2

OMEGA_ERR_FACTOR = 1
AMP_ERR_FACTOR = 1

MAX_OMEGA = 1
MAX_AMP = 1

REACH_TARGET = 1e4
TOO_LARGE = 1e4
TIME = 1e2
FIDELITY_DELTA = 2e2

N_ACTIONS = 9


class QuantumEnvironment:
    """
    The model is implemented as such - we have some energy gap w0. We induce a Hamiltonian in
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

    def __init__(self, energy_gap, runtime, dt):
        """
        :param energy_gap: Induced energy gap. Qobj type. i.e. w0*sigmaz()
        :param runtime: duration of the simulation
        :param dt: time step
        :param skips: Number of time steps passed between each action.
                      This was not implemented yet.
        :return:
        """

        # For now we are working in the int. picture, thus the zero.
        self.energy_gap = 0 * energy_gap

        self.dt = dt
        self.runtime = runtime
        self.steps = 0

        self.state = qt.tensor(qt.basis(2, 0), qt.basis(2, 0))

        self.ham_theta = 0.0
        self.ham_omega = 0.0
        self.ham_amp = 0.0

        # create action lookup table
        omega_list = [-1, 0, 1]
        amp_list = [-1, 0, 1]
        self.action_lookup = list(itertools.product(omega_list, amp_list))

    def state_to_vec(self, label='target'):
        """
        Bloch vector representation of the state of the target (index 0) or control (index 1) qubit.
        :return: list of expectation values for the state self.state.
        """
        if label == 'target':
            x, y, z = qt.expect(sigmax(), self.state.ptrace(1)), qt.expect(sigmay(), self.state.ptrace(1)), \
                      qt.expect(sigmaz(), self.state.ptrace(1))
        elif label == 'control':
            x, y, z = qt.expect(sigmax(), self.state.ptrace(0)), qt.expect(sigmay(), self.state.ptrace(0)), \
                      qt.expect(sigmaz(), self.state.ptrace(0))
        else:
            raise Exception('Wrong label! Please use the labels "target" or "control".')

        return [x, y, z]

    def fidelity(self, label='target'):
        """
        fidelity with respect to |1> (which is the target state)
        :return: F(|ψ>,|1>) of type float.
        """
        if label == 'target':
            fidelity = qt.fidelity(self.state.ptrace(1), qt.basis(2, 1))
        elif label == 'control':
            fidelity = qt.fidelity(self.state.ptrace(0), qt.basis(2, 1))
        else:
            raise Exception('Wrong label! Please use the labels "target" or "control".')

        return fidelity

    def step(self, action):
        """
        Perform a single action: increase or decrease omega.
        so 2*2 actions. Observe the ad
        :param action: an integer
        :return:
        """
        self.steps += 1

        if action > (N_ACTIONS - 1) or action < 0 or type(action) is not int:
            raise Exception('Wrong action value!')
        else:
            self.ham_omega += self.action_lookup[action][0] * THETA_BOOST_CONTROL
            self.ham_amp += self.action_lookup[action][1] * AMP_BOOST_CONTROL

        self.ham_theta += self.ham_omega * self.dt

        g = 1e-2
        hx = self.ham_amp * np.cos(self.ham_theta)
        hy = self.ham_amp * np.sin(self.ham_theta)
        hz = self.energy_gap

        ham_tot = qt.tensor((hx * sigmax() + hy * sigmay() + hz * sigmaz()), qt.identity(2)) \
                  + g * (qt.tensor(qt.sigmap(), qt.sigmam()) + qt.tensor(qt.sigmam(), qt.sigmap()))

        unitary_op = 1 - 1j * ham_tot * self.dt

        # Maybe remove unit
        curr_fidelity = self.fidelity()
        self.state = (unitary_op * self.state).unit()

        delta_fidelity = self.fidelity() - curr_fidelity
        reward = 1 / (1 - self.fidelity())
        if delta_fidelity < 0:
            reward = 0
        else:
            reward = (1 + delta_fidelity) * reward

        # reward = (self.fidelity() - curr_fidelity) * FIDELITY_DELTA
        # # reward -= np.abs(self.ham_omega) * OMEGA_ERR_FACTOR
        # # reward -= np.abs(self.ham_amp) * AMP_ERR_FACTOR
        #
        # if np.abs(self.ham_amp) > MAX_AMP or np.abs(self.ham_omega) > MAX_OMEGA:
        #     reward -= TOO_LARGE
        # if self.fidelity() > 0.99:
        #     reward += REACH_TARGET
        #     # reward += (1 - (self.runtime/(self.steps * self.dt))) * TIME
        #     done = True
        # elif (self.steps * self.dt) >= self.runtime:
        #     done = True
        # else:
        #     done = False

        if (self.steps * self.dt) >= self.runtime:
            done = True
        else:
            done = False

        # This is what the neural network sees
        observation = [hx, hy, hz, self.ham_omega] + self.state_to_vec() + self.state_to_vec(label='control')

        if len(observation) != NET_INPUT_SIZE:
            print('Wrong observation size')

        # Some useless parameter resulting from a previous implementation where I tried
        # to inherit from gym.env class.
        info = {}

        return observation, reward, done, info

    def reset(self):
        """
        Restarts steps and randomizes a state.
        :return: 7 parameters of the state and Hamiltonian.
        """

        self.state = qt.tensor(qt.basis(2, 0), qt.basis(2, 0))

        self.steps = 0
        self.ham_theta = 0.0
        self.ham_omega = 0.0
        self.ham_amp = 0.0
        hx = 0
        hy = 0
        hz = self.energy_gap

        observation = [hx, hy, hz, self.ham_omega] + self.state_to_vec() + self.state_to_vec(label='control')
        return observation

    def sample(self):
        """
        :return: A random action represented by an integer.
        """
        return random.randint(0, self.N_ACTIONS - 1)


class TLSSimulation:
    """ Simulate time evolution of a two level system subjected to an external driving."""

    def __init__(self, **kwargs):

        # General parameters:
        self.dimensions = 2

        # Hamiltonian:
        self.H = []
        self.hamiltonian_type = None

        # Temporal parameters:
        self.duration = kwargs.pop('duration')
        self.time_step = kwargs.pop('time_step')
        self.times = np.arange(0, self.duration, self.time_step)

        # State parameters:
        self.initial_state = None
        self.final_dm = None

        # simulation output
        self.simulation = None
        self.envelopes = []
        self.bloch_vectors_a = []
        self.bloch_vectors_b = []

    def add_initial_state(self, state_string):
        """
        Generate the initial state of the simulation.
        :param state_string: String determining the state of the first qubit.
        The second qubit is assumed to start in the ground state (?).
        """

        if state_string == 'ground':
            psi0 = qt.basis(dimensions=self.dimensions, n=0)
        elif state_string == 'excited':
            psi0 = qt.basis(dimensions=self.dimensions, n=1)
        else:
            raise Exception('Add support for arbitrary TLS state.')

        if self.hamiltonian_type == 'two_qubits':
            psi0 = qt.tensor(psi0, qt.basis(dimensions=self.dimensions, n=0))

        self.initial_state = psi0

    def add_hamiltonian(self, hamiltonian_type: str, **kwargs):
        """
        Generate the Hamiltonian of the system.
        Supports: 1) Single qubit - w_a * sigma_z
                  2) Two coupled qubits - w_a * sigma_z + w_b * sigma_z + g * sigma * sigma.
        :param hamiltonian_type: Determines the type of the simulation: 'single_qubit' or 'two_qubits'.
        """

        self.hamiltonian_type = hamiltonian_type

        if hamiltonian_type == 'single_qubit':
            sa_x, sa_y, sa_z = self.get_operators()
            w_a = kwargs.pop('w_a')
            hamiltonian = w_a * sa_z

        elif hamiltonian_type == 'two_qubits':
            sa_x, sa_y, sa_z, sb_x, sb_y, sb_z = self.get_operators()
            sa_plus = 0.5 * (sa_x - 1j * sa_y)
            sa_minus = sa_plus.dag()

            sb_plus = 0.5 * (sb_x - 1j * sb_y)
            sb_minus = sb_plus.dag()

            w_a = kwargs.pop('w_a')
            w_b = kwargs.pop('w_b')
            g = kwargs.pop('g')

            hamiltonian = (w_a / 2) * sa_z + (w_b / 2) * sb_z + g * (sa_plus * sb_minus + sa_minus * sb_plus)

        else:
            raise Exception('No valid Hamiltonian model was found.')

        self.H.append(hamiltonian)

    def add_drive(self, envelope_type: str, direction, **kwargs):
        """
        Add driving of the form envelope(t)*n*sigma to the Hamiltonian.
        :param direction: The n vector in the formula above (no need to normalize).
        :param envelope_type: Supports 1) 'step': specify 't_initial', 'duration' and 'amplitude' via kwargs.
                                       2) 'gaussian': specify 'center', 'amplitude' and 'width' via kwargs.
        :param kwargs: Envelope function parameters.
        :return:
        """

        # define the driving operators
        if self.hamiltonian_type == 'single_qubit':
            sa_x, sa_y, sa_z = self.get_operators()
        elif self.hamiltonian_type == 'two_qubits':
            sa_x, sa_y, sa_z, _, _, _ = self.get_operators()
        else:
            raise Exception('No Hamiltonian was found.')

        # normalize the direction vector
        direction = np.array(direction)
        direction = direction / np.linalg.norm(direction)

        # generate a handle for the envelope function.
        if envelope_type == 'step':
            t_initial = kwargs.pop('t_initial')
            duration = kwargs.pop('duration')
            amplitude = kwargs.pop('amplitude')

            envelope_func = lambda t, args: 0 if t < t_initial or t > t_initial + duration else amplitude

        elif envelope_type == 'gaussian':
            center = kwargs.pop('center')
            amplitude = kwargs.pop('amplitude')
            width = kwargs.pop('width')

            envelope_func = lambda t, args: amplitude * np.exp(-(t - center) ** 2 / (2 * width ** 2))
        else:
            envelope_func = lambda t, args: 1

        self.envelopes.append([envelope_func, direction])
        self.H.append([direction[0] * sa_x + direction[1] * sa_y + direction[2] * sa_z, envelope_func])

    def simulate(self, store_trajectory=False):
        """
        Simulate the time evolution of the qubit.
        """

        if self.hamiltonian_type == 'single_qubit':
            sa_x, sa_y, sa_z = self.get_operators()
        elif self.hamiltonian_type == 'two_qubits':
            sa_x, sa_y, sa_z, sb_x, sb_y, sb_z = self.get_operators()
        else:
            raise Exception('No Hamiltonian was found.')

        self.simulation = qt.sesolve(H=self.H, psi0=self.initial_state, tlist=self.times)

        # get final density matrix:
        if self.hamiltonian_type == 'single_qubit':
            self.final_dm = qt.ket2dm(self.simulation.states[-1])
        elif self.hamiltonian_type == 'two_qubits':
            self.final_dm = qt.ptrace(self.simulation.states[-1], 1)

        if store_trajectory:
            for state in self.simulation.states:
                self.bloch_vectors_a.append([qt.expect(sa_x, state), qt.expect(sa_y, state), qt.expect(sa_z, state)])
                if self.hamiltonian_type == 'two_qubits':
                    self.bloch_vectors_b.append(
                        [qt.expect(sb_x, state), qt.expect(sb_y, state), qt.expect(sb_z, state)])

    def draw_trajectory(self):
        """
        Present 3D bloch sphere with the trajectory of the qubit state as function of time (indicates by the color of
        points.
        """

        def get_sphere(steps=20):
            """
            Generate the coordinates for a sphere.
            :param steps: The number of angle steps.
            :return: x, y and z arrays for generating a sphere with plt.plot_surface(x, y, z).
            """
            phi = np.linspace(0, np.pi, steps)
            theta = np.linspace(0, 2 * np.pi, steps)
            phi, theta = np.meshgrid(phi, theta)

            # The Cartesian coordinates of the unit sphere
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)

            return x, y, z

        if not self.bloch_vectors_a:
            raise Exception(r'No Bloch vectors were found.')

        # Get points on the Bloch-sphere
        xs_a = np.array([point[0] for point in self.bloch_vectors_a])
        ys_a = np.array([point[1] for point in self.bloch_vectors_a])
        zs_a = np.array([point[2] for point in self.bloch_vectors_a])
        if self.hamiltonian_type == 'two_qubits':
            xs_b = np.array([point[0] for point in self.bloch_vectors_b])
            ys_b = np.array([point[1] for point in self.bloch_vectors_b])
            zs_b = np.array([point[2] for point in self.bloch_vectors_b])

        colors = np.linspace(0, 1, len(xs_a))

        x, y, z = get_sphere()

        # Set the aspect ratio to 1 so our sphere looks spherical
        fig = plt.figure(figsize=plt.figaspect(1.))

        if self.hamiltonian_type == 'single_qubit':
            axes = [fig.add_subplot(111, projection='3d')]
        elif self.hamiltonian_type == 'two_qubits':
            axes = [fig.add_subplot(121, projection='3d'), fig.add_subplot(122, projection='3d')]
        else:
            raise Exception('No Hamiltonian was found.')

        [ax.set_box_aspect([1.1, 1.1, 1]) for ax in axes]

        # plot Bloch sphere and points
        [ax.plot_surface(x, y, z, alpha=0.2, cmap='bone', edgecolor='black', linewidth=0.25) for ax in axes]
        axes[0].scatter(xs_a, ys_a, zs_a, s=50, c=colors, cmap='Reds', edgecolors='black')
        if self.hamiltonian_type == 'two_qubits':
            axes[1].scatter(xs_b, ys_b, zs_b, s=50, c=colors, cmap='Reds', edgecolors='black')

        # add lines
        [ax.plot([-1, 1], [0, 0], [0, 0], color='k') for ax in axes]
        [ax.plot([0, 0], [-1, 1], [0, 0], color='k') for ax in axes]
        [ax.plot([0, 0], [0, 0], [-1, 1], color='k') for ax in axes]

        # add text
        [ax.text(0, 0, 1.2, r'$\left|+,z\right\rangle$') for ax in axes]
        [ax.text(0, 0, -1.2, r'$\left|-,z\right\rangle$') for ax in axes]
        [ax.text(1.1, 0, 0, r'$\left|+,x\right\rangle$') for ax in axes]
        [ax.text(-1.55, 0, 0, r'$\left|-,x\right\rangle$') for ax in axes]
        [ax.text(0, 1.1, 0, r'$\left|+,y\right\rangle$') for ax in axes]
        [ax.text(0, -1.2, 0, r'$\left|-,y\right\rangle$') for ax in axes]

        [ax.axis('off') for ax in axes]
        plt.show()

    def draw_pulse(self):
        """
        Plot the driving pulses.
        """
        fig, ax = plt.subplots()
        args = []
        for i, (envelope_func, direction) in enumerate(self.envelopes):
            pulse = np.array([envelope_func(t, args) for t in self.times])
            ax.plot(self.times, pulse, label=str(direction))
            ax.fill_between(self.times, pulse, alpha=0.1)

        ax.set(xlabel='time', xlim=(self.times[0], self.times[-1]))
        ax.set_ylim(bottom=0)
        ax.legend()
        ax.grid()
        plt.show()

    def get_operators(self):
        """
        Generate Pauli operators for single and two qubits simulations,
         assuming the drive is applied to the first qubit.
        :return: Pauli matrices sigma_x, sigma_y and sigma_z
        """
        s_x = qt.sigmax()
        s_y = qt.sigmay()
        s_z = qt.sigmaz()
        iden = qt.identity(self.dimensions)

        if self.hamiltonian_type == 'single_qubit':
            sa_x = s_x
            sa_y = s_y
            sa_z = s_z
            return sa_x, sa_y, sa_z

        if self.hamiltonian_type == 'two_qubits':
            sa_x = qt.tensor(s_x, iden)
            sa_y = qt.tensor(s_y, iden)
            sa_z = qt.tensor(s_z, iden)
            sb_x = qt.tensor(iden, s_x)
            sb_y = qt.tensor(iden, s_y)
            sb_z = qt.tensor(iden, s_z)
            return sa_x, sa_y, sa_z, sb_x, sb_y, sb_z


def simulate_single_qubit(centers, amplitudes, widths):
    """
    Wrapper for the TLSSimulation class. Simulates the time evolution of a single driven qubit.
    The qubit is assumed to be in the ground state (?).
    The energy gap is assumed to be 1 (?).
    The simulation duration is 10 and the time step is 0.1 (?).
    :param centers: The centers of the three Gaussian pulses (in the x, y and z direction), [c_x, c_y, c_z].
    :param amplitudes: The amplitudes of the three Gaussian pulses, [a_x, a_y, a_z].
    :param widths: The widths of the three Gaussian pulses, [w_x, w_y, w_z].
    :return: The final density matrix of the qubit in the format: array([rho_00, rho01; rho_10, rho_11]) and a list of
             lists containing (gaussian envelope, direction).
             Note that the envelope functions require an 'arg' input (can be set to None).
    """

    sim_params = {'duration': 10,
                  'time_step': 0.1}

    sim = TLSSimulation(**sim_params)
    sim.add_hamiltonian(hamiltonian_type='single_qubit', **{'w_a': 1})
    sim.add_initial_state(state_string='ground')
    sim.add_drive(envelope_type='gaussian', direction=[1, 0, 0], **{'center': centers[0],
                                                                    'amplitude': amplitudes[0],
                                                                    'width': widths[0]})
    sim.add_drive(envelope_type='gaussian', direction=[0, 1, 0], **{'center': centers[1],
                                                                    'amplitude': amplitudes[1],
                                                                    'width': widths[1]})
    sim.add_drive(envelope_type='gaussian', direction=[0, 0, 1], **{'center': centers[2],
                                                                    'amplitude': amplitudes[2],
                                                                    'width': widths[2]})
    sim.simulate(store_trajectory=False)

    return sim.final_dm.data.A, sim.envelopes


def simulate_two_qubits(centers, amplitudes, widths):
    """
    Wrapper for the TLSSimulation class. Simulates the time evolution of two qubits, coupled via the interaction term:
    g * (sigma_plus * sigma_minus + sigma_minus * sigma_plus).

    Both qubits are assumed to be in the ground state (?).
    The qubit energy gaps are assumed to be w_a = w_b = 1 (?).
    The coupling strength is assumed to be g = 1 (?).
    The simulation duration is 10 and the time step is 0.1 (?).
    :param centers: The centers of the three Gaussian pulses (in the x, y and z direction), [c_x, c_y, c_z].
    :param amplitudes: The amplitudes of the three Gaussian pulses, [a_x, a_y, a_z].
    :param widths: The widths of the three Gaussian pulses, [w_x, w_y, w_z].
    :return: The final density matrix of the second qubit in the format: array([rho_00, rho01; rho_10, rho_11]) and a
             lists of tuples containing (gaussian envelope, direction).
             Note that the envelope functions require an 'arg' input (can be set to None).
    """

    sim_params = {'duration': 10,
                  'time_step': 0.1}

    sim = TLSSimulation(**sim_params)
    sim.add_hamiltonian(hamiltonian_type='two_qubits', **{'w_a': 1,
                                                          'w_b': 1,
                                                          'g': 1})
    sim.add_initial_state(state_string='ground')
    sim.add_drive(envelope_type='gaussian', direction=[1, 0, 0], **{'center': centers[0],
                                                                    'amplitude': amplitudes[0],
                                                                    'width': widths[0]})
    sim.add_drive(envelope_type='gaussian', direction=[0, 1, 0], **{'center': centers[1],
                                                                    'amplitude': amplitudes[1],
                                                                    'width': widths[1]})
    sim.add_drive(envelope_type='gaussian', direction=[0, 0, 1], **{'center': centers[2],
                                                                    'amplitude': amplitudes[2],
                                                                    'width': widths[2]})
    sim.simulate(store_trajectory=False)

    return sim.final_dm.data.A, sim.envelopes


if __name__ == '__main__':
    """ Direct simulation example """
    # sim_params = {'duration': 10,
    #               'time_step': 0.1}
    #
    # sim = TLSSimulation(**sim_params)
    # # sim.add_hamiltonian(hamiltonian_type='single_qubit', **{'w_a': 1})
    # sim.add_hamiltonian(hamiltonian_type='two_qubits', **{'w_a': 0,
    #                                                       'w_b': 0,
    #                                                       'g': 1})
    # sim.add_initial_state(state_string='ground')
    # sim.add_drive(envelope_type='gaussian', direction=[1, 0, 0], **{'center': 3, 'amplitude': 1, 'width': 0.5})
    # sim.add_drive(envelope_type='gaussian', direction=[1, 0, 0], **{'center': 6, 'amplitude': 0.6, 'width': 0.4})
    # sim.simulate(store_trajectory=True)
    # sim.draw_trajectory()
    # sim.draw_pulse()

    """ Single/Two qubits Wrapper simulation example """
    output, envelopes = simulate_single_qubit(centers=[4, 5, 6], amplitudes=[0.8, 1, 1.2], widths=[0.4, 0.5, 0.6])
    # output, envelopes = simulate_two_qubits(centers=[4, 5, 6], amplitudes=[0.8, 1, 1.2], widths=[0.4, 0.5, 0.6])
