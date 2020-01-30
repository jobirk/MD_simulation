import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
import scipy.constants as con
from IPython.display import HTML
from tqdm import tqdm
import time
import sys

def U(x, T=300):
    """ returns the potential at a position x

    Parameters
    ----------
    T : int (optional, default is 300. Given in Kelvin)

    Returns
    ----------
    float

    """
    return con.R * T * (0.28 * (0.25*x**4 + 0.1*x**3 - 3.24*x**2) + 3.5)

def dUdx(x, T=300):
    """ derivative of the potential U

    Parameters
    ----------
    T : int (optional, default is 300. Given in Kelvin)

    Returns
    ----------
    float

    """
    return con.R * T * (0.28 * (0.25*4*x**3 + 0.1*3*x**2 - 3.24*2*x))


class Markov_simulation():
    """
    Object to perform a simulation using the Markovian Langevin equation

    Attributes
    ----------
    traj : array 
        stores the position x and velocity v for each simulation step

    R : array
        normal distributed noise

    """

    def __init__(self, steps, dt=0.001, m=1, T=300):
        """
        Parameters
        ----------
        steps : int
            number for simulation steps

        dt : float, optional
            integration time in the simulation, default is 0.001, unit is ps

        m : float, optional
            mass of the simulated particle, default is 1, unit is ps^-1

        T : float, optional
            temperature at which the simulation is carried out, unit is K

        """
        self.traj   = np.zeros([2, steps+1])
        self.steps  = steps
        self.dt     = dt
        self.m      = m
        self.T      = T
        # normal distributed noise with gaussian of mean 0 and variance 1
        self.R      = np.random.normal(0, 1, size=steps+1)

    def integrator(self, step, Gamma=100):
        """
        Numerical integrator derived from the Markovian Langevin Equation

        Parameters
        ----------
        step : int
            the simulation step. Based on the information at this step the 
            coordinate and velocity at step+1 is calculated

        Gamma : float, optional
            parameter for the friction, default is 100

        """
        # calculate the position at step+1
        self.traj[0, step+1] = self.traj[0, step] + self.traj[1, step] * self.dt

        # calculate the velocity at step+1
        self.traj[1, step+1] = self.traj[1, step] \
                                - 1/self.m * dUdx(self.traj[0, step], T=self.T) * self.dt\
                                - 1/self.m * Gamma * self.traj[1, step] * self.dt \
                                + 1/self.m * np.sqrt(2 * con.R * self.T * Gamma * self.dt) * self.R[step]

    def simulate(self):
        """
        simple method that runs the simulation
        """
        for i in tqdm(range(self.steps)):
            self.integrator(i)

    def plot_trajectory(self, end_step, start_step=0):
        """
        method to plot the trajectory of the particle as a function of time

        Parameters
        ----------
        end_step : int
            last step that is included in the plot

        start_step : int, optional
            first step that is included in the plot, default is 0
        """

        fig, ax = plt.subplots(figsize=(6,4))
        ax.set_title(r"Trajectory of the particle")
        ax.set_xlabel(r"Time $t$ [ps]")
        ax.set_ylabel(r"Position $x$")

        timesteps = self.dt * np.arange(self.steps)
        ax.plot(timesteps[start_step:end_step], self.traj[0, start_step:end_step])

    def plot_histogram(self, end_step, start_step=0, nbins=30):
        """
        method to plot a histogram of the particle position

        Parameters
        ----------
        end_step : int
            last step that is included in the plot

        start_step : int, optional
            first step that is included in the plot, default is 0

        nbins : int, optional
            number of bins in the histogram, default is 30
        """
        fig, ax = plt.subplots(figsize=(6,4))
        ax.set_title(r"Position of the particle")
        ax.set_xlabel(r"Position $x$")
        ax.set_ylabel(r"Occupation")

        ax.hist(self.traj[0, start_step:end_step], bins=nbins)

    def save_trajectory(self, filename, interval=1):
        """
        method to save the trajectory of the particle

        Parameters
        ----------
        filename : str
            name of the text file in which the numpy array is saved
        interval : int, optional
            save only every n*interval'th step, default is 1 (every step saved)
        """
        np.savetxt(filename, self.traj[:,::interval])
        if interval!=1:
            print("Saved the trajectory of every %ith step to the file '%s'" %(interval, filename))
        else:
            print("Saved the trajectory to the file '%s'" %(filename))



def calculate_states(trajectory):
    """
    caculates the matrix defining the Markov State Model

    Parameters
    ----------
    trajectory : array
        saved trajectory from previous simulation

    Returns
    ----------
    array
    """ 

    x = np.copy(trajectory[0,:])

    states = np.zeros(len(x))
    # set the first state randomly to -1 or 1
    states[0] = np.random.choice([-1,1])

    left_indices   = np.where(x < -1)
    right_indices  = np.where(1 <  x)
    middle_indices = np.where((-1 < x) & (x < 1))
    # print(middle_indices)

    # set states of left and right populated positions
    states[left_indices]  = -1
    states[right_indices] =  1

    # now loop over the positions between the cores
    # these states are assigned to the state visited before
    for i in middle_indices[0]:
        if i==0:
            continue
        states[i] = states[i-1]

    # calculate population of the two states
    N_left  = len(np.where(states==-1)[0])
    N_right = len(np.where(states== 1)[0])

    # calculate the transitions
    states_shift = np.roll(states, 1, axis=0) # all states are shifted one index further
    # calculate difference of states[i] and states[i-1]
    diff = states - states_shift
    # diff = -2 corresponds to transition right -> left
    # diff =  2 corresponds to transition left  -> right
    N_left_right = len(np.where(diff== 2)[0])
    N_right_left = len(np.where(diff==-2)[0])
    # plt.plot(diff)

    # calculate the transition probabilities
    p_left_right = N_left_right / N_left
    p_right_left = N_right_left / N_right
    p_left_left   = 1 - p_left_right
    p_right_right = 1 - p_right_left

    # build Markov matrix
    M = np.array([[p_left_left,  p_left_right ],
                  [p_right_left, p_right_right]])
    # print(p_left_right, p_left_left)
    # print(p_right_left, p_right_right)
    # plt.plot(diff[1:], ls="--")
    # print("Nleft + Nright", N_left+N_right)

    return M, x[left_indices], x[middle_indices], x[right_indices], states, diff


            



















