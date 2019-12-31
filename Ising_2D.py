import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from scipy.stats import rv_continuous
import scipy.constants as con
from IPython.display import HTML
from tqdm import tqdm
import time
import sys

class Ising_2D():

    def __init__(self):
        self.spins = []
        self.spin_trajectory = []

    def Ising_energy(self, J=2): #J in J/mol
        """ returns the energy of the ising model """

        spins_shift_x = np.roll(self.spins, 1, 0)
        spins_shift_y = np.roll(self.spins, 1, 1)

        spin_products_x = self.spins * spins_shift_x 
        spin_products_y = self.spins * spins_shift_y
        spin_sum = np.sum(spin_products_x, axis=(0,1)) + np.sum(spin_products_y, axis=(0,1)) 

        return - 0.5 * J * spin_sum


    def MC_Ising_simulation(self, N, ising_steps=100000, J=2, T=0.18, new_spins=True,
                            print_progress=False):
        """ performs a MC simulation of the Ising model """

        counter_accepted = 0

        if new_spins:
            self.spins = np.random.choice([-1, 1], size=(N,N))

        self.ising_energies = np.zeros(ising_steps)
        self.magnetisation  = np.zeros(ising_steps)
        self.spin_trajectory = np.zeros((N, N, ising_steps))

        E = self.Ising_energy()
        m = np.mean(self.spins, axis=(0, 1))

        for step in (range(ising_steps)):

            if print_progress:
                if (step/ising_steps*100)%1==0:
                    print(int(step/ising_steps*100), "% completed", end='\r')

            i = np.random.randint(N)
            j = np.random.randint(N)
            self.spins[i,j] *= -1

            dE = - 2 * 0.5 * J * self.spins[i,j] *  \
                (self.spins[i, (j-1)%N] + self.spins[i, (j+1)%N] \
                +self.spins[(i-1)%N, j] + self.spins[(i+1)%N, j])

            if dE < 0:
                E += dE                         # flip accepted, change energy
                m += self.spins[i,j] * 2 / N**2 # update the magnetisation
                counter_accepted += 1

            else:
                P = np.exp(- (dE) / (con.R * T))
                q = np.random.uniform(0,1)
                if np.log(q) < np.log(P):
                    E += dE                         # update energy
                    m += self.spins[i,j] * 2 / N**2 # update the magnetisation
                    counter_accepted += 1
                else:
                    self.spins[i,j] *= -1 # step not accepted -> flip spin back

            self.ising_energies[step] = E
            self.magnetisation[step]  = m
            self.spin_trajectory[:,:,step]= self.spins
        # print("Total flips:", ising_steps, "Accepted flips:", counter_accepted)

        return self.ising_energies, self.magnetisation, self.spin_trajectory


    def Ising_analysis(self, N, n_temp=5, steps_1=1000, steps_2=1000):
        """ runs the Ising MC simulation for several temperatures
            and evaluates them """
        temperatures = np.linspace(0.18, 0.4, n_temp)
        mean_energies           = np.zeros(n_temp)
        mean_magnetisations     = np.zeros(n_temp)
        mean_squared_energies       = np.zeros(n_temp)
        mean_squared_magnetisations = np.zeros(n_temp)

        new_spin_orientations = False
        for i, T in enumerate(temperatures):
            t1 = time.time()
            a, b, c                 = self.MC_Ising_simulation(N, T=T, new_spins=True, ising_steps=steps_1)
            energy_T, mag_T, spins  = self.MC_Ising_simulation(N, T=T, new_spins=False, ising_steps=steps_2)
            mean_energies[i]                = np.mean(energy_T)
            mean_squared_energies[i]        = np.mean(energy_T**2)
            mean_magnetisations[i]          = np.mean(mag_T)
            mean_squared_magnetisations[i]  = np.mean(mag_T**2)
            t2 = time.time()
            # plt.imshow(spins[:,:,1], cmap="hot")
            # plt.show()
            print("Completed simulation at temperature", i+1, "/", n_temp, \
                  ", time per simulation: %.0f"%(t2-t1),"seconds", end="\r")

        C_V = (mean_squared_energies - mean_energies**2) / (con.R * temperatures**2)
        Chi_T = (mean_squared_magnetisations - mean_magnetisations**2) / (con.R * temperatures)

        fig, axs = plt.subplots(2, 2, figsize=(13, 7))
        axs = axs.flatten()
        axs[0].set_title(r"Mean energy $\left<E\right>$")
        axs[1].set_title(r"Mean magnetisation $\left<m\right>$")
        axs[2].set_title(r"Specific heat at constant Volume $C_{V}$")
        axs[3].set_title(r"Magnetic susceptibility $\chi_{T}$")
        axs[0].set_xlabel(r"$Temperature T$")
        axs[1].set_xlabel(r"$Temperature T$")
        axs[2].set_xlabel(r"$Temperature T$")
        axs[3].set_xlabel(r"$Temperature T$")
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


    def Ising_visualisation(self, N, T, steps=1000, steps_per_frame=5,
                            number_of_plots=0):
        print("Starting simulation.")
        energy_T, mag_T, spins = self.MC_Ising_simulation(N, T=T, new_spins=True, 
                                 ising_steps=steps, print_progress=True)
        print("Done with simulation.")

        if number_of_plots:
            if number_of_plots%3!=0:
                print(">>> ERROR: please give multiple of 3 as number of plots <<<")
            n_rows = int((number_of_plots)/3)
            fig, axs = plt.subplots(n_rows, 3, figsize=(15,n_rows*5))
            axs = axs.flatten()
            for i in range(number_of_plots):
                step = int(i * steps/number_of_plots)
                axs[i].set_title(r"step %i"%(step))
                axs[i].imshow(spins[:,:,step])

            plt.show()
        
        else:
            fig, ax = plt.subplots(figsize=(3,3), dpi=100)
            ax.set_title(r"Phase transition of the Ising model")

            im = plt.imshow(spins[:,:,0])

            def animate(i):
                im.set_data(spins[:,:,i])
                return im

            plt.close()
            anim = animation.FuncAnimation(fig, animate, np.arange(0, steps, steps_per_frame), interval=100, blit=False);
            return HTML(anim.to_html5_video());





