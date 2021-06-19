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

        self.simulation = None
        self.bloch_vectors = []

    def get_initial_state(self, state_string):
        if state_string == 'ground':
            psi0 = qt.basis(dimensions=self.dimensions, n=0)
        elif state_string == 'excited':
            psi0 = qt.basis(dimensions=self.dimensions, n=1)
        else:
            raise Exception('Add support for arbitrary TLS state.')

        return psi0

    def add_drive(self, t_initial, duration, direction):
        """
        Add driving of the form n*sigma to the Hamiltonian
        :param t_initial: When the pulse starts.
        :param duration: The pulse's duration.
        :param direction: The vector n in the above formula.
        :return: Updates the simulation's Hamiltonian.
        """
        s_x = qt.sigmax()
        s_y = qt.sigmay()
        s_z = qt.sigmaz()
        step_func = lambda t, args: 0 if t < t_initial or t > t_initial + duration else 1
        self.H.append([direction[0] * s_x + direction[1] * s_y + direction[2] * s_z, step_func])

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


if __name__ == '__main__':
    sim_params = {'duration': 3,
                  'time_step': 0.1,
                  'energy_gap': 1,
                  'initial_state': 'ground'}

    sim = TLSSimulation(**sim_params)
    sim.add_drive(t_initial=0, duration=1, direction=[1, 0, 0])
    sim.add_drive(t_initial=1, duration=2, direction=[0, 1, 0])
    sim.simulate(store_trajectory=True)
    sim.draw_trajectory()

