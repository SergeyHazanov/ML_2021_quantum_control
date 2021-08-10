import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

from DataLoader import GamesMemoryBank
from policy_loss import PolicyLoss
from Model import PolicyNetwork, NET_INPUT_SIZE
from simulation import QuantumEnvironment

from IPython.display import clear_output
from scipy.ndimage import uniform_filter1d
from os.path import isfile
import torch
from tqdm import tqdm


class Trainer:

    def __init__(self, **kwargs):
        self.kwargs = kwargs

        # learning parameters
        self.n_epochs = kwargs.get('n_epochs', 300)
        self.games_per_epoch = kwargs.get('games_per_epoch', 10)
        self.batch_size = kwargs.get('batch_size', 24000)
        self.num_batches = kwargs.get('num_batches', 5)
        self.learning_rate = kwargs.get('learning_rate', 1e-3)

        # quantum environment
        self.dt = kwargs.get('dt', 0.01)
        self.runtime = kwargs.get('runtime', 15)
        self.energy_gap = kwargs.get('energy_gap', 1)
        self.env = QuantumEnvironment(energy_gap=self.energy_gap, runtime=self.runtime, dt=self.dt)

        # initialize net, loss function and the memory bank
        self.net = PolicyNetwork()
        self.loss_func = PolicyLoss()
        self.memory_bank = GamesMemoryBank()

        # load previously trained net (if exists)
        load_net = kwargs.get('load_net', True)
        self.model_name = kwargs.get('net_name', 'trained_model.pt')
        if isfile(self.model_name) and load_net:
            self.net.load_state_dict(torch.load(self.model_name))

        # initialize optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)

    def train(self):
        final_fidelity = []
        running_mean_fid = []
        running_time = []
        losses = []
        rewards = []
        discounted_rewards = []

        fig, axes = plt.subplots()
        # [ax.legend() for ax in axes]

        for epoch in tqdm(range(self.n_epochs)):

            self.memory_bank.clear_memory()

            self.net.eval()
            self.net.cpu()

            for game_i in range(self.games_per_epoch):

                state = self.env.reset()

                state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
                prev_state = torch.zeros_like(state)

                while True:
                    with torch.no_grad():
                        action, action_p = self.net.sample_action(state, prev_state)

                    new_state, reward, done, info = self.env.step(action)
                    self.memory_bank.add_event(state, prev_state, action, action_p, reward)

                    prev_state = state
                    state = torch.tensor(new_state, dtype=torch.float).view(-1).unsqueeze(0)

                    if done:
                        final_fidelity.append(self.env.fidelity())
                        running_time.append(self.env.steps * self.dt / self.runtime)
                        break

            self.memory_bank.compute_reward_history()

            clear_output(wait=True)
            running_mean_fid = uniform_filter1d(final_fidelity, 50)
            mean_fid = np.mean(final_fidelity[-self.games_per_epoch:])
            mean_fid = str(mean_fid)[:5]

            plt.title('epoch ' + str(epoch)
                      + ' mean fidelity in last '
                      + str(self.games_per_epoch)
                      + ' simulations '
                      + mean_fid)

            plt.plot(final_fidelity, label='Fidelity', alpha=0.5)

            plt.plot(running_mean_fid, label='Running Mean', color='black')
            plt.plot(running_time, label="Running time", color="rebeccapurple", alpha=0.5)
            plt.legend()
            plt.show()

            self.net.train()
            if torch.cuda.is_available():
                self.net.cuda()

            for batch_i in range(self.num_batches):
                self.optimizer.zero_grad()

                state, prev_state, action, action_p, reward, discounted_reward =\
                    self.memory_bank.get_sample(self.batch_size)
                state = state.view((state.shape[0], NET_INPUT_SIZE))
                prev_state = prev_state.view((prev_state.shape[0]), NET_INPUT_SIZE)

                logits = self.net(state, prev_state)

                loss = self.loss_func(logits, action, action_p, discounted_reward)
                losses.append(loss.item())
                loss.backward()
                self.optimizer.step()

                losses.append(loss.item())
                rewards.append(torch.mean(reward).item())
                discounted_rewards.append(torch.mean(discounted_reward).item())

            torch.save(self.net.state_dict(), self.model_name)

    def probe_learning(self):
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

        state = self.env.reset()
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        prev_state = state

        z = list()
        omega = list()
        theta = list()
        amps = list()

        s_x = list()
        s_y = list()
        s_z = list()

        # play a game
        while True:
            action, action_p = self.net.sample_action(state, prev_state)
            new_state, reward, done, info = self.env.step(action)
            state = torch.tensor(new_state, dtype=torch.float).view(-1).unsqueeze(0)

            state_data = state[0].numpy()

            curr_amp = np.sqrt(state_data[0] ** 2 + state_data[1] ** 2 + state_data[2] ** 2)
            curr_omega = state_data[3]
            if not theta:
                curr_theta = curr_omega * self.dt
            else:
                curr_theta = theta[-1] + curr_omega * self.dt
            z.append(state_data[-1])

            amps.append(curr_amp)
            omega.append(curr_omega)
            theta.append(curr_theta)

            s_x.append(state_data[4])
            s_y.append(state_data[5])
            s_z.append(state_data[6])

            if done:
                break

        times = np.arange(0, self.runtime, self.dt)
        times = times[:len(amps)]
        fig, ax = plt.subplots()
        ax.plot(times, amps, label='amp')
        ax.plot(times, omega, label='omega')
        ax.plot(times, theta, label='theta')

        ax.legend()
        plt.show()

        # draw game trajectory
        # Set the aspect ratio to 1 so our sphere looks spherical
        fig = plt.figure(figsize=plt.figaspect(1.))
        axes = [fig.add_subplot(111, projection='3d')]
        x, y, z = get_sphere()

        [ax.set_box_aspect([1.1, 1.1, 1]) for ax in axes]

        # plot Bloch sphere and points
        s_x = np.array(s_x)
        s_y = np.array(s_y)
        s_z = np.array(s_z)
        colors = np.linspace(0, 1, len(s_x))

        [ax.plot_surface(x, y, z, alpha=0.2, cmap='bone', edgecolor='black', linewidth=0.25) for ax in axes]
        axes[0].scatter(s_x, s_y, s_z, s=50, c=colors, cmap='Reds', edgecolors='black')

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


if __name__ == '__main__':
    train_session = Trainer(**{'runtime': 2, 'load_net': False})
    train_session.train()
