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
    trajectory : array 
        stores the position x for each simulation step

    """

    def __init__(self, steps):
        """
        Parameters
        ----------
        steps : int
            number for simulation steps

        """
        self.trajectory = 
        return 0

    def


















