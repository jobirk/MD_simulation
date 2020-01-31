import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
import scipy.constants as con
from IPython.display import HTML
from tqdm import tqdm
import time
import sys
import os

kbT = 38

def U(x):
    """ returns the potential at a given position
    Parameters
    ----------
    x : float
        position x of the particle

    Returns
    ----------
    U : float
        value of the potential at the given position x

    """
    return kbT * (0.28 * (0.25*x**4 + 0.1*x**3 - 3.24*x**2) + 3.5)

def dUdx(x):
    """ derivative of the potential at given position

    Parameters
    ----------
    x : float
        position x of the particle

    Returns
    ----------
    dUdx : float
        value of the first derivative of the potential at position x
    """
    return kbT * (0.28 * (0.25*4*x**3 + 0.1*3*x**2 - 3.24*2*x))


class Markov_simulation():
    """
    class to perform a simulation using the Markovian Langevin equation

    Attributes
    ----------
    x : array
        positions x at all simulation steps
    v : array
        velocity v at all simulation steps
    R : array
        normal distributed noise used for the integration
    """

    def __init__(self, steps, dt=0.001, m=1):
        """
        Parameters
        ----------
        steps : int
            number for simulation steps

        dt : float, optional
            integration time in the simulation, default is 0.001, unit is ps

        m : float, optional
            mass of the simulated particle, default is 1, unit is ps^-1
        """
        self.x      = np.zeros(steps+1)
        self.v      = np.zeros(steps+1)
        self.steps  = steps
        self.dt     = dt
        self.m      = m
        # normal distributed noise with gaussian of mean 0 and variance 1
        np.random.seed(42)
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
        self.x[step+1] = self.x[step] + self.v[step] * self.dt

        # calculate the velocity at step+1
        self.v[step+1] =   self.v[step] \
                         - 1/self.m * dUdx(self.x[step]) * self.dt \
                         - 1/self.m * Gamma * self.v[step] * self.dt \
                         + 1/self.m * np.sqrt(2 * kbT * Gamma * self.dt) \
                           * self.R[step]

    def simulate(self):
        """
        simple method that runs the simulation
        """
        for i in tqdm(range(self.steps)):
            self.integrator(i)

    def save_trajectory(self, filename, interval=1, printing=False):
        """
        method to save the trajectory of the particle

        Parameters
        ----------
        filename : str
            name of the text file in which the numpy array is saved
        interval : int, optional
            save only every n*interval'th step, default is 1 (every step saved)
        printing : bool
            option to get a printed output telling what was saved
        """
        if os.path.exists(filename):
            if printing:
                print("file already exists -> not saved again")
            return

        np.savetxt(filename, self.x[::interval])
        if printing:
            if interval!=1:
                print("Saved the trajectory of every %ith step to the file '%s'" %(interval, filename))
            else:
                print("Saved the trajectory to the file '%s'" %(filename))


def plot_trajectory(trajectory, end_step, start_step=0):
    """
    method to plot the trajectory of the particle as a function of time

    Parameters
    ----------
    end_step : int
        last step that is included in the plot

    start_step : int, optional
        first step that is included in the plot, default is 0
    """
    fig, ax = plt.subplots(figsize=(5,3))
    ax.set_title(r"Trajectory of the particle")
    ax.set_xlabel(r"Step")
    ax.set_ylabel(r"Position $x$")
    ax.ticklabel_format(style='sci')

    ax.plot(trajectory[start_step:end_step])
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    plt.tight_layout()

def plot_histogram(trajectory, end_step, start_step=0, nbins=30):
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
    fig, ax = plt.subplots(figsize=(5,3))
    ax.set_title(r"Position distribution of the particle")
    ax.set_xlabel(r"Position $x$")
    ax.set_ylabel(r"Occupation")

    ax.hist(trajectory[start_step:end_step], bins=nbins, density=1)
    plt.tight_layout()


def calculate_states_and_M(trajectory):
    """
    caculates the matrix defining the Markov State Model

    Parameters
    ----------
    trajectory : array_like
        saved trajectory from previous simulation

    Returns
    ----------
    M : array_like
        transition matrix of the trajectory
    states : array_like
        numpy array including the state (-1 or 1) of each simulation step
    """ 

    x = np.copy(trajectory)

    states = np.zeros(len(x))
    # set the first state randomly to -1 or 1
    np.random.seed(42)
    states[0] = np.random.choice([-1,1])

    left_indices   = np.where(x < -1)
    right_indices  = np.where(1 <  x)
    middle_indices = np.where((-1 < x) & (x < 1))

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
    # calculate difference of states[i] and states[i-1]
    diff = states - np.roll(states, 1, axis=0)
    # diff = -2 corresponds to transition right -> left
    # diff =  2 corresponds to transition left  -> right
    N_left_right = len(np.where(diff== 2)[0])
    N_right_left = len(np.where(diff==-2)[0])

    # calculate the transition probabilities
    p_left_right = N_left_right / N_left
    p_right_left = N_right_left / N_right
    p_left_left   = 1 - p_left_right
    p_right_right = 1 - p_right_left

    # build Markov matrix
    M = np.array([[p_left_left,  p_left_right ],
                  [p_right_left, p_right_right]])

    return M, states #x[left_indices], x[middle_indices], x[right_indices], states, diff


            




