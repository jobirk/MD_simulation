
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
import scipy.constants as con
from IPython.display import HTML
from tqdm import tqdm
import time
import sys


def U(x, y, T=300, x_shift=0, y_shift=0, a=0.809, b=0.588):
    x = x - x_shift
    y = y - y_shift
    return con.R * T * (0.28 * (0.25 * (a*x + b*y)**4 + 0.1 * (a*x + b*y)**3 - 3.24*(a*x + b*y)**2 + 6.856*(a*y - b*x)**2)+3.5)


class energy_surface_simulation():

    def __init__(self, start_x=2, start_y=2, simulation_steps=1e5, T=300):
        self.T = T
        self.steps = int(simulation_steps)
        self.trajectory = np.zeros([2, self.steps+1])
        self.trajectory[:, 0] = [start_x, start_y]
        self.population_matrix = 0
        print("initial position:", self.trajectory[:,0])

    def MC_simulation(self, step_size=0.01):

        accepted_steps = 0
        E_old = U(self.trajectory[0,0], self.trajectory[1,0])

        for step in tqdm(range(self.steps)):

            # if accepted_steps == 1e6:
            #     print("Reached 1 000 000 accepted steps.")
            #     break

            self.trajectory[:, step+1] = np.copy(self.trajectory[:,step])
            # move particle
            phi = np.random.uniform(0, 2*np.pi)
            r_x = np.cos(phi)
            r_y = np.sin(phi)
            self.trajectory[0, step+1] += r_x
            self.trajectory[1, step+1] += r_y

            # calculate new energy
            E_new = U(self.trajectory[0, step+1], self.trajectory[1, step+1], T=self.T)

            if E_new < E_old:
                E_old = E_new
                accepted_steps +=1
                continue

            else: # check for metropolis criteria
                P = np.exp(- (E_new - E_old) / (con.R * self.T))
                q = np.random.uniform(0,1)
                if q < P:
                    E_old = E_new
                    accepted_steps += 1
                    continue
            
                else: # not accepted, move particle back
                    self.trajectory[0, step+1] -= r_x
                    self.trajectory[1, step+1] -= r_y

        print("Accepted steps:", accepted_steps)
        print("Acceptance: %.2f"%(accepted_steps/self.steps))

    def plot_population(self, trajectory=0, nbins=30, x_range=[-5,5], y_range=[-5,5], interpol='gaussian'):

        if type(trajectory)==int:
            trajectory = self.trajectory

        fig, ax = plt.subplots(figsize=(3,3), dpi=100)
        ax.set_title(r"Population")
        ax.set_xlabel(r"x")
        ax.set_ylabel(r"y")
        self.population_matrix, x, y = np.histogram2d(trajectory[0,:], trajectory[1,:], range=[x_range, y_range], bins=nbins)
        self.population_matrix = self.population_matrix.T

        im = ax.imshow(self.population_matrix, interpolation=interpol, origin="lower", 
                extent=[x_range[0], x_range[1], y_range[0], y_range[1]])
        fig.colorbar(im, ax=ax)
        







