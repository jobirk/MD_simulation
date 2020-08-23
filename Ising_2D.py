import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import scipy.constants as con
from IPython.display import HTML
import time


class Ising_2D():

    def __init__(self):
        self.spins = []
        self.spin_trajectory = []

    def Ising_energy(self, J=2):  # J in J/mol
        """ returns the energy of the ising model """

        spins_shift_x = np.roll(self.spins, 1, 0)
        spins_shift_y = np.roll(self.spins, 1, 1)

        spin_products_x = self.spins * spins_shift_x 
        spin_products_y = self.spins * spins_shift_y
        spin_sum = np.sum(spin_products_x, 
                          axis=(0, 1)) + np.sum(spin_products_y, axis=(0, 1)) 

        return - 0.5 * J * spin_sum

    def MC_simulation(self, N, simulation_steps=100000, J=2, T=0.18, 
                      new_spins=True, print_progress=False):
        """ performs a MC simulation of the Ising model """

        if new_spins:
            self.spins = np.random.choice([-1, 1], size=(N, N))

        self.steps = simulation_steps
        self.ising_energies = np.zeros(simulation_steps)
        self.magnetisation  = np.zeros(simulation_steps)
        self.spin_trajectory = np.zeros((N, N, simulation_steps))

        E = self.Ising_energy()
        m = np.mean(self.spins, axis=(0, 1))

        for step in (range(simulation_steps)):

            if print_progress:  # print some information about the progress
                if (step/simulation_steps*100) % 1 == 0:
                    print(int(step/simulation_steps*100), "% completed", end='\r')

            # generate a random pair (i,j) of indices
            i = np.random.randint(N)
            j = np.random.randint(N)
            # flip the spin
            self.spins[i, j] *= -1 

            # calculate the change in energy
            dE = - 2 * 0.5 * J * self.spins[i, j] *  \
                (self.spins[i, (j-1) % N] + self.spins[i, (j+1) % N]
                 + self.spins[(i-1) % N, j] + self.spins[(i+1) % N, j])

            if dE < 0:
                E += dE                         # flip accepted, change energy
                m += self.spins[i, j] * 2 / N**2  # update the magnetisation

            else:
                P = np.exp(- (dE) / (con.R * T))
                q = np.random.uniform(0, 1)
                if q < P:
                    E += dE                         # accepted, update energy
                    m += self.spins[i, j] * 2 / N**2  # update the magnetisation
                else:
                    # step not accepted -> flip spin back
                    self.spins[i, j] *= -1 

            # save energy, magnetisation and spins of this step
            self.ising_energies[step] = E
            self.magnetisation[step]  = m
            self.spin_trajectory[:, :, step] = self.spins

        return self.ising_energies, self.magnetisation, self.spin_trajectory

    def Ising_analysis(self, N, n_temp=5, steps_1=1000, steps_2=1000):
        """ runs the Ising MC simulation for several temperatures
            and evaluates them """
        temperatures = np.linspace(0.18, 0.4, n_temp)
        mean_energies           = np.zeros(n_temp)
        mean_magnetisations     = np.zeros(n_temp)
        mean_squared_energies       = np.zeros(n_temp)
        mean_squared_magnetisations = np.zeros(n_temp)

        for i, T in enumerate(temperatures):
            t1 = time.time()
            # perform simulation for equilibration
            a, b, c                 = self.MC_simulation(N, T=T, new_spins=True, simulation_steps=steps_1)
            # perfrom simulation in equilibrium
            energy_T, mag_T, spins  = self.MC_simulation(N, T=T, new_spins=False, simulation_steps=steps_2)
            # calculate energie and magnetisation properties
            mean_energies[i]                = np.mean(energy_T)
            mean_squared_energies[i]        = np.mean(energy_T**2)
            mean_magnetisations[i]          = np.mean(mag_T)
            mean_squared_magnetisations[i]  = np.mean(mag_T**2)
            t2 = time.time()
            # plt.imshow(spins[:,:,1], cmap="hot")
            # plt.show()
            print("Completed simulation at temperature", i+1, "/", n_temp,
                  ", time per simulation: %.0f" % (t2-t1), "seconds", end="\r")

        C_V = (mean_squared_energies - mean_energies**2) / (con.R * temperatures**2)
        Chi_T = (mean_squared_magnetisations - mean_magnetisations**2) / (con.R * temperatures)

        fig, axs = plt.subplots(2, 2, figsize=(11, 7))
        axs = axs.flatten()
        axs[0].set_title(r"Mean energy $\left<E\right>$")
        axs[1].set_title(r"Mean magnetisation $\left<m\right>$")
        axs[2].set_title(r"Specific heat at constant Volume $C_{V}$")
        axs[3].set_title(r"Magnetic susceptibility $\chi_{T}$")
        axs[0].set_xlabel(r"Temperature $T$")
        axs[1].set_xlabel(r"Temperature $T$")
        axs[2].set_xlabel(r"Temperature $T$")
        axs[3].set_xlabel(r"Temperature $T$")
        axs[0].set_ylabel(r"$\left<E\right>$")
        axs[1].set_ylabel(r"$\left<m\right>$")
        axs[2].set_ylabel(r"$C_{V}$")
        axs[3].set_ylabel(r"$\chi_{T}$")
        axs[0].plot(temperatures, mean_energies)
        axs[1].plot(temperatures, mean_magnetisations, marker="o", ls="")
        axs[2].plot(temperatures, C_V, marker="o", ls="--")
        axs[3].plot(temperatures, Chi_T, marker="o", ls="--")

        plt.tight_layout()
        plt.show()

    def visualisation(self, steps_per_frame=5, number_of_plots=0,
                      saveas=""):

        if number_of_plots:
            if number_of_plots % 3 != 0:
                print(">>> ERROR: please give multiple of 3 as number of plots <<<")
            n_rows = int((number_of_plots)/3)
            fig, axs = plt.subplots(n_rows, 3, figsize=(15, n_rows*5))
            axs = axs.flatten()
            for i in range(number_of_plots):
                step = int(i / number_of_plots * self.steps)
                axs[i].set_title(r"step %i" % (step))
                axs[i].imshow(self.spin_trajectory[:, :, step])

            plt.show()

        else:
            fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
            ax.set_title(r"Phase transition of the Ising model")

            im = plt.imshow(self.spin_trajectory[:, :, 0])

            def animate(i):
                im.set_data(self.spin_trajectory[:, :, i])
                return im

            plt.close()
            anim = animation.FuncAnimation(fig, animate, np.arange(0, self.steps, steps_per_frame), interval=100, blit=False)
            if saveas != "":
                Writer = animation.writers['ffmpeg']
                writer = Writer(  # fps=int(1000 / ms_between_frames), 
                                metadata=dict(artist='Me'), bitrate=1800)
                if 'mp4' in saveas:
                    print("Saving as .mp4")
                    anim.save(saveas, writer=writer)
                elif '.gif' in saveas:
                    print("Saving as .gif")
                    anim.save(saveas, writer='imagemagick')
                else:
                    print("Saving as .mp4")
                    anim.save(saveas+".mp4", writer=writer)
                    print("Saving as .gif")
                    anim.save(saveas+".gif", writer='imagemagick')
            else:
                return HTML(anim.to_html5_video())

