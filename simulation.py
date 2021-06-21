import numpy as np
import qutip as qt
import matplotlib.pyplot as plt


class TLSSimulation:
    """ Simulate time evolution of a two level system subjected to an external driving."""
    def __init__(self, **kwargs):
        # General parameters:
        self.dimensions = 2
        # Hamiltonian parameters:
        self.energy_gap = kwargs.pop('energy_gap')
        self.H = [(self.energy_gap / 2) * qt.sigmaz()]
        # Temporal parameters:
        self.duration = kwargs.pop('duration')
        self.time_step = kwargs.pop('time_step')
        self.times = np.arange(0, self.duration, self.time_step)
        # State parameters:
        self.initial_state = self.get_initial_state(kwargs.pop('initial_state'))
        self.final_state = None

        # simulation output
        self.simulation = None
        self.envelopes = []
        self.bloch_vectors = []

    def get_initial_state(self, state_string):
        if state_string == 'ground':
            psi0 = qt.basis(dimensions=self.dimensions, n=0)
        elif state_string == 'excited':
            psi0 = qt.basis(dimensions=self.dimensions, n=1)
        else:
            raise Exception('Add support for arbitrary TLS state.')

        return psi0

    def add_drive(self, envelope_type: str, direction, **kwargs):
        """
        Add driving of the form envelope(t)*n*sigma to the Hamiltonian.
        :param direction: The n vector in the formula above (no need to normalize).
        :param envelope_type: Supports 1) 'step': specify 't_initial', 'duration' and 'amplitude' via kwargs.
                                       2) 'gaussian': specify 'center', 'amplitude' and 'width' via kwargs.
        :param kwargs: Envelope function parameters.
        :return:
        """
        s_x = qt.sigmax()
        s_y = qt.sigmay()
        s_z = qt.sigmaz()

        # normalize the direction vector
        direction = np.array(direction)
        direction = direction / np.linalg.norm(direction)

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
        self.H.append([direction[0] * s_x + direction[1] * s_y + direction[2] * s_z, envelope_func])

    def simulate(self, store_trajectory=False):
        """
        Simulate the time evolution of the qubit.
        """
        s_x = qt.sigmax()
        s_y = qt.sigmay()
        s_z = qt.sigmaz()

        self.simulation = qt.sesolve(H=self.H, psi0=self.initial_state, tlist=self.times)
        self.final_state = self.simulation.states[-1]

        if store_trajectory:
            for state in self.simulation.states:
                self.bloch_vectors.append([qt.expect(s_x, state), qt.expect(s_y, state), qt.expect(s_z, state)])

    def draw_trajectory(self):
        """
        Present 3D bloch sphere with the trajectory of the qubit state as function of time (indicates by the color of
        points.
        """
        if not self.bloch_vectors:
            raise Exception(r'No Bloch vectors were found.')

        # Get points on the Bloch-sphere
        xs = np.array([point[0] for point in self.bloch_vectors])
        ys = np.array([point[1] for point in self.bloch_vectors])
        zs = np.array([point[2] for point in self.bloch_vectors])

        colors = np.linspace(0, 1, len(xs))

        phi = np.linspace(0, np.pi, 20)
        theta = np.linspace(0, 2 * np.pi, 20)
        phi, theta = np.meshgrid(phi, theta)

        # The Cartesian coordinates of the unit sphere
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)

        # Set the aspect ratio to 1 so our sphere looks spherical
        fig = plt.figure(figsize=plt.figaspect(1.))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_box_aspect([1.1, 1.1, 1])

        # plot Bloch sphere and points
        ax.plot_surface(x, y, z, alpha=0.2, cmap='bone', edgecolor='black', linewidth=0.25)
        ax.scatter(xs, ys, zs, s=50, c=colors, cmap='Reds', edgecolors='black')

        # add lines
        ax.plot([-1, 1], [0, 0], [0, 0], color='k')
        ax.plot([0, 0], [-1, 1], [0, 0], color='k')
        ax.plot([0, 0], [0, 0], [-1, 1], color='k')

        # add text
        ax.text(0, 0, 1.2, r'$\left|+,z\right\rangle$')
        ax.text(0, 0, -1.2, r'$\left|-,z\right\rangle$')
        ax.text(1.1, 0, 0, r'$\left|+,x\right\rangle$')
        ax.text(-1.55, 0, 0, r'$\left|-,x\right\rangle$')
        ax.text(0, 1.1, 0, r'$\left|+,y\right\rangle$')
        ax.text(0, -1.2, 0, r'$\left|-,y\right\rangle$')

        ax.axis('off')
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


def simulate_single_qubit(centers, amplitudes, widths):
    """
    Wrapper for the TLSSimulation class. Simulates the time evolution of a single driven qubit.
    The qubit is assumed to be in the ground state (?).
    The energy gap is assumed to be 1 (?).
    The simulation duration is 10 and the time step is 0.1 (?).
    :param centers: The centers of the three Gaussian pulses (in the x, y and z direction), [c_x, c_y, c_z].
    :param amplitudes: The amplitudes of the three Gaussian pulses, [a_x, a_y, a_z].
    :param widths: The widths of the three Gaussian pulses, [w_x, w_y, w_z].
    :return: The final qubit state in the evolution.
    """

    sim_params = {'duration': 10,
                  'time_step': 0.1,
                  'energy_gap': 1,
                  'initial_state': 'ground'}

    sim = TLSSimulation(**sim_params)
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

    return [sim.final_state.data.A[0][0], sim.final_state.data.A[1][0]]


if __name__ == '__main__':
    """ Direct simulation example """
    sim_params = {'duration': 10,
                  'time_step': 0.1,
                  'energy_gap': 1,
                  'initial_state': 'ground'}

    sim = TLSSimulation(**sim_params)
    sim.add_drive(envelope_type='gaussian', direction=[1, 0, 0], **{'center': 3, 'amplitude': 1, 'width': 0.5})
    sim.simulate(store_trajectory=True)
    sim.draw_trajectory()
    sim.draw_pulse()

    """ Single qubit Wrapper simulation example """
    # centers = [4, 5, 6]
    # amplitudes = [0.8, 1, 1.2]
    # widths = [0.4, 0.5, 0.6]
    # output = simulate_single_qubit(centers, amplitudes, widths)
    # print(output)
