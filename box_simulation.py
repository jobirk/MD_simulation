import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
import scipy.constants as con
from IPython.display import HTML
from tqdm import tqdm
import time
import sys

def distance_pbc_transformation(distance, box_length):
    """ transforms the distance dx or dy with respect to periodic
        boundary conditions """
    return (distance + (box_length / 2)) % box_length - (box_length / 2)

def Lennard_Jones_force(r, dx, dy, C_12=9.847044e-6, C_6=6.2647225e-3, 
        cutoff=0.33, use_lower_cutoff=False, use_upper_cutoff=False):
    """ Lennard Jones Force in N/mol """
    dx_values = np.copy(dx)
    dy_values = np.copy(dy)
    r_values  = np.copy(r)

    #dirty workaround to avoid division by zero
    zero_indices = np.where(r_values==0)
    r_values[zero_indices] = 42e10

    if use_lower_cutoff==True:
        # for distances below the cutoff we want assign them the same force
        # Therefore, adjust the dx, dy, r appropriately
        cutoff_indices = np.where(r<cutoff)
        dx_values[cutoff_indices] = dx_values[cutoff_indices] * \
                                    cutoff/r_values[cutoff_indices]
        dy_values[cutoff_indices] = dy_values[cutoff_indices] * \
                                    cutoff/r_values[cutoff_indices]
        r_values[cutoff_indices] = cutoff

    F = (12 * C_12 / r_values**13 - 6 * C_6 / r_values**7) \
        * 1 / r_values * np.array([dx_values, dy_values])

    if use_upper_cutoff==True:
        # print('using upper cutoff')
        # set all Forces for distances > 0.9nm to zero
        up_cut_indices = np.where(r_values > 0.9)
        F[0][up_cut_indices] = 0
        F[1][up_cut_indices] = 0

    # set F=0 at entries where r=0
    F[0][zero_indices] = 0
    F[1][zero_indices] = 0
    return F

def Lennard_Jones(distances, C_12=9.847044e-6, C_6=6.2647225e-3, cutoff=0.33, \
                  use_lower_cutoff=True, upper_cutoff=0):
    """ function that calculates an array of V_ij values (in J/mol) 
        corresponding to a distance array containing r_ij values (distances) """
    # the distance array usually contains entries with r=0, e.g. r_11
    # but we only want to use the non-zero entries for the calculation
    r = np.copy(distances)
    non_diagonal_indices = np.where(r!=0)
    diagonal_indices = np.where(r==0)
    V = np.zeros(r.shape)
    r_pow_6 = r[non_diagonal_indices]**6

    V[non_diagonal_indices] = C_12 / r_pow_6**2 - C_6 / r_pow_6
    LJ_cutoff = C_12 / cutoff**12 - C_6 / cutoff**6

    if use_lower_cutoff==True:
        # correct the values below the cutoff
        cutoff_indices = np.where(r<cutoff)
        # this is the potential if assuming a 
        # constant slope for r < cutoff
        V[cutoff_indices] = LJ_cutoff + \
            Lennard_Jones_force(np.array([cutoff]), np.array([cutoff]), 
                                np.array([0]))[0] * (cutoff-r[cutoff_indices]) 
        V[diagonal_indices] = 0

    if upper_cutoff==True:
        # correct the values above the cutoff
        cutoff_indices = np.where(r>upper_cutoff)
        # this is the potential if assuming a 
        # constant slope for r < cutoff
        V[cutoff_indices] = 0

    return V


""">>>>>>>>>>>>>>>>>>> simulation class <<<<<<<<<<<<<<<<<"""

class box_simulation():

    def __init__(self, box_x=50, box_y=50, n_particles=50, n_steps=2000, \
                 particle_mass=0.018, pbc=True, grid=False):
        self.box = [box_x, box_y]
        self.n_particles = n_particles
        self.pbc = pbc
        self.grid = grid
        self.steps = n_steps
        self.particle_distances = []
        self.trajectories = []
        self.kin_energies = []
        self.pot_energies = []
        self.Lennard_Jones_matrix = []
        self.particle_mass = particle_mass
        self.step_interval = 0
        self.temperatures = []

    def generate_particles(self, test_particles=[], v=1, m=1, T=0):
        """ method to generate a starting position of particles and velocities
            distributed randomly in the box. Also creates the essential arrays
            to save the simulation result. """

        # generate the new particles with random numbers
        n_random = self.n_particles
        #total number incl. test particles
        self.n_particles += len(test_particles) 
        self.trajectories = np.zeros([self.n_particles, 4, self.steps+1])

        # >>> positions <<<
        if self.grid:
            # place particles in a grid
            n_row = np.sqrt(self.n_particles)
            if n_row%1 != 0:
                print(">>> ERROR: Please choose number of particles which is the square root of an integer")
            n_row = int(n_row)
            dist_neighbour = self.box[0] / n_row
            self.dist_neighbour = dist_neighbour
            positions = np.zeros([2, self.n_particles])
            print("generating particles arranged in a grid")
            particle = 0
            for i in range(n_row):
                x = dist_neighbour/2 + i * dist_neighbour
                for j in range(n_row):
                    y = dist_neighbour/2 + j * dist_neighbour
                    positions[:, particle] = (x,y)
                    particle += 1
            self.temperatures = np.zeros([n_row, n_row, self.steps+1])

        else:
            positions = np.random.uniform(0, self.box[0], (2, n_random))

        # >>> set the positions and random velocities <<<
        v_angles = np.random.uniform(0, 2*np.pi, n_random)
        self.trajectories[:n_random,0,0] = positions[0]
        self.trajectories[:n_random,1,0] = positions[1]
        self.trajectories[:n_random,2,0] = v * np.cos(v_angles)
        self.trajectories[:n_random,3,0] = v * np.sin(v_angles)

        # option to add specific test_particles
        if test_particles != []:
            i = 0
            for particle in test_particles:
                self.trajectories[n_random+i,:,0] = particle
                i += 1
        if T:
            # assign velocities according to 2D boltzmann distrib.
            # -> overwrite the previously defined velocities
            print('Assigning velocities according to Maxwell Boltzmann distribution at T=%s'%(T))
            sigma = np.sqrt(con.R * T / self.particle_mass)
            vx = np.random.normal(scale=sigma, size=self.n_particles)
            vy = np.random.normal(scale=sigma, size=self.n_particles)
            self.trajectories[:,2,0] = vx
            self.trajectories[:,3,0] = vy

        # also setup the necessary arrays to save the simulaiton
        self.particle_distances   = np.zeros([self.n_particles, \
                                            self.n_particles, 3, self.steps+1])
        self.Lennard_Jones_matrix = np.zeros([self.n_particles, \
                                            self.n_particles, 3, self.steps+1])
        self.kin_energies         = np.zeros([self.n_particles, self.steps+1])
        self.pot_energies         = np.zeros([self.n_particles, self.steps+1])

    def circle_temperature(self, T=10, r=1, plot_structure=False, mean_velocities=False):
        print('Generating velocities in circle of radius %.2f and center at center of box'%(r))
        print('Assigning velocities according to Maxwell Boltzmann distribution at T=%s'%(T))

        # first calculate which particles are in this circle
        step_index = 0
        x_values = self.trajectories[:, 0, step_index]
        y_values = self.trajectories[:, 1, step_index]

        # subtract the center values from the coordinates
        x_center_distance = distance_pbc_transformation(x_values - self.box[0]/2, self.box[0])
        y_center_distance = distance_pbc_transformation(y_values - self.box[1]/2, self.box[1])

        center_distance = np.sqrt(x_center_distance**2 + y_center_distance**2) #shape (n_particles)

        in_circle = np.where(center_distance < r)

        if plot_structure:
            # print("in_circle indices:", in_circle[0])
            self.plot_trajectories(circle_radius=r, markersize=5)

        # print("shape of center_distance array: ", center_distance.shape)
        if self.grid:
            T_values = self.temperatures[:,:,0].flatten()
            T_values[in_circle[0]] = T
            self.temperatures[:,:,0] = T_values.reshape(self.temperatures[:,:,0].shape)

        else:
            if mean_velocities:
                v = np.sqrt(2 / self.particle_mass * con.R * T)
                # v = 
                v_angles = np.random.uniform(0, 2*np.pi, len(in_circle[0]))
                self.trajectories[:,2,0] = 0
                self.trajectories[:,3,0] = 0
                self.trajectories[in_circle,2,0] = v * np.cos(v_angles)
                self.trajectories[in_circle,3,0] = v * np.sin(v_angles)
            else:
                sigma = np.sqrt(con.R * T / self.particle_mass)
                vx = np.random.normal(scale=sigma, size=len(in_circle[0]))
                vy = np.random.normal(scale=sigma, size=len(in_circle[0]))
                self.trajectories[:,2,0] = 0
                self.trajectories[:,3,0] = 0
                self.trajectories[in_circle,2,0] = vx
                self.trajectories[in_circle,3,0] = vy

    def save_particle_positions(self, step, filename="no_filename.txt"):
        """ method to save the particle positions at given step to file """
        np.savetxt(filename, self.trajectories[:,:,step])
        print("Saved the particle positions of step %i to file '%s'" \
               %(step,filename))

    def load_particles(self, filename, new_velocities=False, v=1, T=2):
        """ method to load saved particle positions """
        traj_from_file = np.loadtxt(filename, dtype=float)
        self.n_particles = traj_from_file.shape[0]
        self.trajectories = np.zeros([self.n_particles, 4, self.steps+1])
        self.trajectories[:,:,0] = traj_from_file

        print("The initial state loaded from the file '%s'" %(filename))
        if new_velocities:
            v_angles = np.random.uniform(0, 2*np.pi, self.n_particles)
            self.trajectories[:,2,0] = v * np.cos(v_angles)
            self.trajectories[:,3,0] = v * np.sin(v_angles)
        if T:
            # assign velocities according to 2D boltzmann distrib.
            print('Assigning velocities according to Maxwell Boltzmann distribution at T=%s'%(T))
            sigma = np.sqrt(con.R * T / self.particle_mass)
            vx = np.random.normal(scale=sigma, size=self.n_particles)
            vy = np.random.normal(scale=sigma, size=self.n_particles)
            self.trajectories[:,2,0] = vx #* np.cos(v_angles)
            self.trajectories[:,3,0] = vy #* np.sin(v_angles)

        # also setup the necessary arrays to save the simulaiton
        self.particle_distances   = np.zeros([self.n_particles, \
                                            self.n_particles, 3, self.steps+1])
        self.Lennard_Jones_matrix = np.zeros([self.n_particles, \
                                            self.n_particles, 3, self.steps+1])
        self.kin_energies         = np.zeros([self.n_particles, self.steps+1])
        self.pot_energies         = np.zeros([self.n_particles, self.steps+1])

    def calculate_distances(self, step_index):
        """ method for calculating all distances between particles at a step"""
        # if self.pbc:
        x_values = self.trajectories[:, 0, step_index]
        x_mesh = np.meshgrid(x_values, x_values)
        dx = x_mesh[1] - x_mesh[0] # x_mesh[i,j] = p_i.x - p_j.x
        dx = distance_pbc_transformation(dx, box_length=self.box[0])

        y_values = self.trajectories[:, 1, step_index]
        y_mesh = np.meshgrid(y_values, y_values)
        dy = y_mesh[1] - y_mesh[0] # y_mesh[i,j] = p_i.y - p_j.y
        dy = distance_pbc_transformation(dy, box_length=self.box[0])

        r = np.sqrt(dx**2 + dy**2)

        self.particle_distances[:, :, 0, step_index] = dx
        self.particle_distances[:, :, 1, step_index] = dy
        self.particle_distances[:, :, 2, step_index] = r 

    def calculate_LJ_potential(self, step_index, \
                        use_lower_cutoff=False, upper_cutoff=0):
        """ function that calculates the matrix of LJ potentials 
            and forces between all particles """
        self.Lennard_Jones_matrix[:, :, 0, step_index] = \
                Lennard_Jones(self.particle_distances[:, :, 2, step_index], 
                              use_lower_cutoff=use_lower_cutoff, 
                              upper_cutoff=upper_cutoff)

    def calculate_LJ_force(self, step_index, \
                        use_lower_cutoff=False, upper_cutoff=False):
        """ function that calculates the matrix of LJ potentials 
            and forces between all particles """
        self.Lennard_Jones_matrix[:, :, 1, step_index] = \
            Lennard_Jones_force(self.particle_distances[:, :, 2, step_index], 
                                self.particle_distances[:, :, 0, step_index],
                                self.particle_distances[:, :, 1, step_index], 
                                use_lower_cutoff=use_lower_cutoff,
                                use_upper_cutoff=upper_cutoff)[0]
        self.Lennard_Jones_matrix[:, :, 2, step_index] = \
            Lennard_Jones_force(self.particle_distances[:, :, 2, step_index], 
                                self.particle_distances[:, :, 0, step_index],
                                self.particle_distances[:, :, 1, step_index],
                                use_lower_cutoff=use_lower_cutoff,
                                use_upper_cutoff=upper_cutoff)[1]

    def verlet_update_positions(self, step_index, dt=1):
        """ function that moves a particle according to the 
            velocity verlet algorithm """
        m = self.particle_mass

        # sum potential and forces along one (the second) particle index
        V  = np.sum(self.Lennard_Jones_matrix[:, :, 0, step_index], axis=1)
        Fx = np.sum(self.Lennard_Jones_matrix[:, :, 1, step_index], axis=1)*1000
        Fy = np.sum(self.Lennard_Jones_matrix[:, :, 2, step_index], axis=1)*1000

        ax, ay = Fx/m , Fy/m
        self.trajectories[:, 0, step_index+1] = \
                self.trajectories[:, 0, step_index] + \
                self.trajectories[:, 2, step_index] * dt + 0.5 * ax * dt**2
        self.trajectories[:, 1, step_index+1] = \
                self.trajectories[:, 1, step_index] + \
                self.trajectories[:, 3, step_index] * dt + 0.5 * ay * dt**2
        if self.pbc:
            self.trajectories[:, 0, step_index+1] %= self.box[0]
            self.trajectories[:, 1, step_index+1] %= self.box[1]

    def verlet_update_velocities(self, step_index, dt=1):
        """ function that updates all velocities according to the 
            velocity verlet algorithm """
        m = self.particle_mass
        # sum potential and forces along one (the second) particle index
        V  = np.sum(self.Lennard_Jones_matrix[:, :, 0, step_index], axis=1)
        Fx = np.sum(self.Lennard_Jones_matrix[:, :, 1, step_index], axis=1)*1000
        Fy = np.sum(self.Lennard_Jones_matrix[:, :, 2, step_index], axis=1)*1000

        V_new  = np.sum(self.Lennard_Jones_matrix[:,:,0, step_index+1],axis=1)
        Fx_new = np.sum(self.Lennard_Jones_matrix[:,:,1, step_index+1],axis=1) \
                 *1000
        Fy_new = np.sum(self.Lennard_Jones_matrix[:,:,2, step_index+1],axis=1) \
                 *1000

        ax, ay, ax_new, ay_new = Fx/m , Fy/m, Fx_new/m , Fy_new/m

        self.trajectories[:, 2, step_index+1] = \
                self.trajectories[:, 2, step_index] + 0.5 * (ax + ax_new) * dt
        self.trajectories[:, 3, step_index+1] = \
                self.trajectories[:, 3, step_index] + 0.5 * (ay + ay_new) * dt
    
    def move_all_particles(self, step, r_x=0, r_y=0, length=0.05):
        """ moves all particles along r """
        self.trajectories[:, 0, step+1] = (self.trajectories[:, 0, step] + \
                                           r_x*length) % self.box[0]
        self.trajectories[:, 1, step+1] = (self.trajectories[:, 1, step] + \
                                           r_y*length) % self.box[1]

    def move_single_particle(self, part_index, step, r_x=0, r_y=0, \
                             random_direction=False, length=0.05):
        if random_direction:
            phi = np.random.uniform(0, 2*np.pi)
            r_x = np.cos(phi)
            r_y = np.sin(phi)
        self.trajectories[part_index, 0, step+1] = \
            (self.trajectories[part_index, 0, step] + r_x*length) % self.box[0]
        self.trajectories[part_index, 1, step+1] = \
            (self.trajectories[part_index, 1, step] + r_y*length) % self.box[1]

    def calc_F_direction(self, step):
        """ returns the normalised F vector at given time step """
        self.calculate_distances(step)
        self.calculate_LJ_potential(step, use_lower_cutoff=False)
        self.calculate_LJ_force(step, use_lower_cutoff=False)
        Fx = np.sum(self.Lennard_Jones_matrix[:, :, 1, step], axis=1)
        Fy = np.sum(self.Lennard_Jones_matrix[:, :, 2, step], axis=1)
        F_abs = np.sqrt(Fx**2 + Fy**2)
        # build unit vector r along F
        r_x = Fx / F_abs
        r_y = Fy / F_abs
        return np.array([r_x, r_y])

    def E_pot(self, step):
        """ returns the potential energie at a given simulation step """
        return np.sum(self.Lennard_Jones_matrix[:, :, 0, step], axis=(0,1))/2

    def get_T(self, step, N=0):
        """ returns the temperature in the box at given sim. step """
        if N==0:
            N = self.n_particles
        k = 8.13 # Gas constant
        T = self.particle_mass / (2 * k * N) * \
            np.sum(self.trajectories[:N,2,step]**2 + self.trajectories[:N,3,step]**2)
        return T

    def berendsen_thermo(self, step, tau, T0):
        """ method to scale the velocities with the berendsen thermostat """
        T = self.get_T(step)
        if T == 0:
            T = 1e-42
        lam = np.sqrt( 1 + self.step_interval / tau * (T0 / T -1 ))
        # multiply the velocities of the given step with the factor lambda
        self.trajectories[:, 2:4, step] *= lam
        return T, lam

    def MD_simulation(self, step_interval=1, use_lower_cutoff=False, 
                      upper_cutoff=False, T=100):
        """ method to run the simulation and create the trajectories """

        self.step_interval = step_interval
        self.calculate_distances(0)
        self.calculate_LJ_potential(0, use_lower_cutoff=use_lower_cutoff, upper_cutoff=upper_cutoff)
        self.calculate_LJ_force(0, use_lower_cutoff=use_lower_cutoff, upper_cutoff=upper_cutoff)
        self.thermostat = np.zeros([self.steps+1, 2]) # T and lambda for all steps

        self.thermostat[0, 0] = self.get_T(0)

        # >>>> simulation <<<<
        for step in tqdm(range(self.steps)):
            # update all positions
            self.verlet_update_positions(step, dt=step_interval)
            # calculate the new values of the LJ potential and force
            self.calculate_distances(step+1)
            self.calculate_LJ_potential(step+1,
                    use_lower_cutoff=use_lower_cutoff,
                    upper_cutoff=upper_cutoff)
            self.calculate_LJ_force(step+1,
                    use_lower_cutoff=use_lower_cutoff,
                    upper_cutoff=upper_cutoff)
            # update all velocities
            self.verlet_update_velocities(step, dt=step_interval)
            # rescale new velocities with berendsen thermostat
            if T:
                self.thermostat[step+1,:] = self.berendsen_thermo(step+1, 0.0002, T)

        self.kin_energies = 0.5 * self.particle_mass * \
                (self.trajectories[:,2,:]**2 + self.trajectories[:,3,:]**2)

    def MC_simulation(self, r_length=0.01, T=50, maximal_steps=100000):
        """ method to perform a MC simulation with the Metropolis algorithm """
        step = 0
        i_particle = 0
        self.calculate_distances(0)
        self.calculate_LJ_potential(0, upper_cutoff=0.9)
        E_1 = self.E_pot(0)
        counter = 0
        print("MC simulation will be performed over %i accepted steps" % (self.steps))
        P_list = [] # list to store the probabilities when checking for the Metropolis criteria
        energies = np.zeros(self.steps)

        for i in range(3*self.steps):
            if step >= self.steps:
                break
            # progress
            if (step/self.steps*100)%1==0:
                print(int(step/self.steps*100), "% completed", end='\r')

            counter += 1
            energies[step] = E_1
            # copy coordinates of current step to next step
            self.trajectories[:, :, step+1] = np.copy(self.trajectories[:, :, step])
            # move the particle i into random direction
            self.move_single_particle(i_particle, step, length=r_length, random_direction=True)
            self.calculate_distances(step+1)
            self.calculate_LJ_potential(step+1, upper_cutoff=0.9)
            E_2 = self.E_pot(step+1)
            # if energy decreased, accept step
            if E_1 > E_2:
                step += 1
                E_1 = E_2
            else:
                # else, check for Metropolis criteria
                P = np.exp(- (E_2-E_1) * 1000 / (con.R * T))
                P_list.append(P)
                q = np.random.uniform(0,1)
                if np.log(q) < np.log(P):
                    step += 1
                    E_1 = E_2
            # if the move is not accepted, step is not increased and thereby the
            # coordinated of step+1 will be overwritten in the next move
            i_particle = (i_particle + 1) % self.n_particles
        print("100% completed.")
        print(30*'-')
        print("Total steps:", counter)

        # plt.hist(P_list)
        # plt.show()
        # plt.plot(energies)
        # plt.show()

    def run_SD(self, step_length=0.01, plot_from=False):
        """ method to run the minimisation of the total potential energy of
            the system using the method of steepest descent """

        step = 0
        r = self.calc_F_direction(0)
        E_1 = self.E_pot(0)
        E_2 = E_1 - 1
        stop_SD = False

        for i in tqdm(range(self.steps)):

            continue_moving = True
            steps_in_this_direction = 0

            while continue_moving and step<self.steps:
                # move all particles in the direction of the force vector
                self.move_all_particles(step, r[0], r[1], length=step_length)
                # calculate new distances, potential and energy
                self.calculate_distances(step+1)
                self.calculate_LJ_potential(step+1, use_lower_cutoff=False)
                E_2 = self.E_pot(step+1)
                if E_1 > E_2:
                    # continue, and set the old E_2 as new E_1
                    E_1 = E_2
                    steps_in_this_direction += 1
                    if step > self.steps:
                        break
                else:
                    # dont move in that direction any more
                    # reverse the last step
                    self.move_all_particles(step, r[0], r[1], length=0)
                    continue_moving = False
                    if steps_in_this_direction==0:
                        stop_SD = True
                step += 1

            if step > self.steps or stop_SD:
                break

            # calculate the new direction of the Force vector
            r = self.calc_F_direction(step)

        if plot_from:
            """ plot the evolution of the total potential energy """
            E_pot = np.sum(self.Lennard_Jones_matrix[:,:,0,:step], axis=(0,1)) / 2
            fig, (ax1) = plt.subplots(1, 1, figsize=(6,4))

            ax1.set_xlabel(r"simulation step")
            ax1.set_ylabel('Energy [kJ/mol]')
            ax1.set_title("Evolution of potential energy")

            ax1.plot(range(plot_from,len(E_pot)), E_pot[plot_from:], label=r"$E_{pot}$", color="r")
            ax1.legend()

            plt.tight_layout()
            plt.show()

        # return the number of the step after which the minimisation has ended
        print("Steepest descent minimisation finished at step", step)
        return step

    def plot_energies(self, only_Epot=False):
        E_pot = np.sum(self.Lennard_Jones_matrix[:,:,0,:], axis=(0,1)) / 2
        # divide by two because otherwise all values are double counted
        # set up the figure for the box plot
        if only_Epot:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13,4))
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(7,4))

        ax1.set_xlabel(r'Time t [ns]')
        ax1.set_ylabel('Energy [kJ/mol]')
        ax1.set_title('Energy as a function of time')

        time_range = np.arange(self.steps+1) * 2e-6 #time in ns

        if only_Epot:
            n_start=0
            print("E_pot at time 0:", E_pot[0])
            ax1.plot(time_range[n_start:], E_pot[n_start:], label=r"$E_{pot}$", color="r")
            ax2.hist(E_pot, bins=20, density=True, label=r"Distribution of energy states in MC")
            E_lin = np.linspace(E_pot.min(), E_pot.max(), 100)
            ax2.set_xlabel('Energy [kJ/mol]')
            ax2.set_ylabel('Normalised events')
            ax2.legend()

        else:
            E_kin = np.sum(self.kin_energies, axis=0) / 1000 # to make it kJ/mol
            E_kin_diff = E_kin[1:] - E_kin[:-1]
            E_pot_diff = E_pot[1:] - E_pot[:-1]
            E_tot = E_kin + E_pot
            E_diff = E_kin - E_pot
            E_tot_diff = E_kin_diff + E_pot_diff
            ax1.plot(time_range, E_pot, label=r"$E_{pot}$", color="r")
            ax1.plot(time_range, E_kin, label=r"$E_{kin}$", color="g")
            ax1.plot(time_range, E_tot, label=r"$E_{kin} + E_{pot}$", color="b")

        ax1.legend()

        plt.tight_layout()
        plt.show()

    def plot_temperatures(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13,4))

        ax1.set_xlabel(r'Time t [ns]')
        ax1.set_ylabel('Temperature [K]')
        ax2.set_xlabel(r'Time t [ns]')
        ax2.set_ylabel(r' $\lambda$ ')
        ax1.set_title('Temperature as a function of time')
        ax2.set_title('Velocity scaling factor')
        
        ax1.plot(range(self.steps+1), self.thermostat[:,0])
        ax2.plot(range(self.steps+1), self.thermostat[:,1])

        plt.tight_layout()
        plt.show()

    def plot_trajectories(self, circle_radius=0, markersize=2):
        """ method to plot the trajectories of all particles """
        x = np.array([self.trajectories[j][0,:] for j in range(self.n_particles)])
        y = np.array([self.trajectories[j][1,:] for j in range(self.n_particles)])

        # set up the figure for the box plot
        fig, ax1 = plt.subplots(figsize=(4,4))
        ax1.set_xlim((0, self.box[0]))
        ax1.set_ylim((0, self.box[1]))

        ax1.set_xlabel('sposition x')
        ax1.set_ylabel('position y')

        if circle_radius:
            circle = plt.Circle((self.box[0]/2, self.box[1]/2), circle_radius, fill=False)
            ax1.add_artist(circle)

        for xi, yi in zip(x, y):
            ax1.plot(xi, yi,"o", ms=markersize)
        plt.tight_layout()
        plt.show()

    def animate_trajectories(self, ms_between_frames=30, dot_size=3, steps_per_frame=5):
        """ method to animate the particle movement """
        fig, ax = plt.subplots(figsize=(4,4), dpi=130)

        ax.set_xlim(0, self.box[0])
        ax.set_ylim(0, self.box[1])

        plt.xlabel('position x')
        plt.ylabel('position y')

        lines = [ax.plot([], [], marker = 'o', linestyle='', markersize=dot_size)[0]
                 for i in range(self.n_particles)]

        def init():
            for line in lines:
                line.set_data([], [])
            return lines

        x_animate = self.trajectories[:,0,::steps_per_frame]
        y_animate = self.trajectories[:,1,::steps_per_frame]
        print(x_animate.shape)

        def animate(i):
            for j in range(len(lines)):
                x = x_animate[j,i]
                y = y_animate[j,i]
                lines[j].set_data(x, y)
            return lines

        plt.close()
        anim = animation.FuncAnimation(fig, animate, init_func=init, \
                                   frames=x_animate.shape[1], interval=ms_between_frames, blit=True)
        # Writer = animation.writers['ffmpeg']
        # writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        # anim.save("animation.mp4", writer=writer)
        return HTML(anim.to_html5_video())

    def heat_propagation(self, steps_per_frame=5, number_of_plots=0, nbins=15, 
            ms_between_frames=100, interpolation='gaussian'):


        if number_of_plots:
            if number_of_plots%3!=0:
                print(">>> ERROR: please give multiple of 3 as number of plots <<<")
            n_rows = int((number_of_plots)/3)
            fig, axs = plt.subplots(n_rows, 3, figsize=(15,n_rows*5))
            axs = axs.flatten()

            for i in range(number_of_plots):
                step = int(i / number_of_plots * self.steps)
                x_i = self.trajectories[:,0,step]
                y_i = self.trajectories[:,1,step]
                if self.grid:
                    im = axs[i].imshow(self.temperatures[:,:,step], cmap=plt.cm.get_cmap(name='hot'), interpolation=interpolation, extent=[0,self.box[0],0,self.box[1]])
                    fig.colorbar(im, ax=axs[i])
                else:
                    T_i = self.kin_energies[:,step]
                    data, x, y = np.histogram2d(x_i, y_i, weights=T_i, bins=nbins)
                    im = axs[i].imshow(data.T, cmap=plt.cm.get_cmap(name='hot'), interpolation=interpolation, extent=[0,self.box[0],0,self.box[1]])
                    fig.colorbar(im, ax=axs[i])

            plt.show()

        else:
            fig, ax = plt.subplots(figsize=(3,3), dpi=100)
            ax.set_title(r"Propagation of heat")

            x0 = self.trajectories[:,0,0]
            y0 = self.trajectories[:,1,0]

            if self.grid:
                T_0 = self.temperatures[:,:,0].flatten()
            else:
                T_0 = self.kin_energies[:,0]

            data, x, y = np.histogram2d(x0, y0, weights=T_0, bins=nbins)

            im = plt.imshow(data.T, cmap=plt.cm.get_cmap(name='hot'), interpolation=interpolation, extent=[0,self.box[0],0,self.box[1]])
            fig.colorbar(im, ax=ax)

            def animate(i):
                x_i = self.trajectories[:,0,i]
                y_i = self.trajectories[:,1,i]
                if self.grid:
                    im.set_data(self.temperatures[:,:,i])
                else:
                    T_i = self.kin_energies[:,i]
                    data, x, y = np.histogram2d(x_i, y_i, weights=T_i, bins=nbins)
                    im.set_data(data.T)
                return im

            plt.close()
            anim = animation.FuncAnimation(fig, animate, frames=np.arange(0, self.steps, steps_per_frame), 
                    interval=ms_between_frames, blit=False);
            return HTML(anim.to_html5_video());

    def occupation(self, start=0, end=0, n_bins=50):
        """ method to plot the occupation on the x-y plane as well
        as the projections on the x and y axis """
        if end==0:
            end = self.steps
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,3))

        ax1.set_xlim((0, self.box[0]))
        ax1.set_ylim((0, self.box[1]))

        ax1.set_xlabel('position x')
        ax1.set_ylabel('position y')
        ax2.set_xlabel('position x')
        ax2.set_ylabel('probability')
        ax3.set_xlabel('position y')
        ax3.set_ylabel('probability')
        ax1.set_title('Occupation probability (x,y)')
        ax2.set_title('Occupation probability along x')
        ax3.set_title('Occupation probability along y')

        # plot a 2D histogram of the box with 50x50 bins,
        # the returned 2D array "counter_squares" stores the number of entries of the bins
        x = np.concatenate([self.trajectories[j][0,start:end] for j in range(self.n_particles)])
        y = np.concatenate([self.trajectories[j][1,start:end] for j in range(self.n_particles)])
        counter_squares, d, f, mappable = ax1.hist2d(x, y, bins=n_bins, density=1)
        fig.colorbar(mappable, ax=ax1, orientation='vertical')

        ax2.hist(x, bins=n_bins, density=1)
        ax3.hist(y, bins=n_bins, density=1)
        plt.tight_layout()
        plt.show()

    def velocity_distributions(self, start=0, end="standard", n_bins=50, width=100):
        """ methdo to plot the velocity distributions """
        if end=="standard":
            end = self.steps
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,3))

        ax1.set_xlabel(r'velocity $v_x$')
        ax1.set_ylabel('probability')
        ax2.set_xlabel(r'velocity $v_y$')
        ax2.set_ylabel('probability')
        ax3.set_xlabel(r'total velocity $|\vec{v}|$')
        ax3.set_ylabel('probability')
        ax1.set_title(r'velocity $v_x$ distribution')
        ax2.set_title(r'velocity $v_y$ distribution')
        ax3.set_title(r'total velocity $|\vec{v}|$ distribution')
        # extract numpy arrays of all vx and vy values between step "start" and step "end"
        vx = np.concatenate([self.trajectories[j][2,start:end] for j in range(self.n_particles)])
        vy = np.concatenate([self.trajectories[j][3,start:end] for j in range(self.n_particles)])
        # plot the normalised histograms
        ax1.hist(vx, bins=n_bins, density=1, range=(-width, width))
        ax2.hist(vy, bins=n_bins, density=1, range=(-width, width))
        v_tot = np.sqrt(vx**2+vy**2)
        ax3.hist(v_tot, bins=n_bins, density=1, range=(0, 2*width))

        plt.tight_layout()
        plt.show()

    def RDF(self, n_bins=50, dr=0.01):
        """ method to plot the RDF """

        distances_time_average = self.particle_distances[:,:,2,:].flatten()
        mask = (distances_time_average != 0) #remove entiries with distance=0
        distances_time_average = distances_time_average[mask]

        r_linspace = np.linspace(0, self.box[0]/2, n_bins)

        dn, bin_edges = np.histogram(distances_time_average, bins=n_bins, \
                                     range=(r_linspace[0], r_linspace[-1]))
        dr = self.box[0] / 2 / (n_bins-1)
        rho = self.n_particles / (self.box[0] * self.box[1])
        g_r =  1 / (2 * np.pi * r_linspace[1:] * rho) * dn[1:] / dr \
               * 1 / (self.steps * self.n_particles)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13,4))
        ax1.set_xlabel(r'distance $r$')
        ax2.set_xlabel(r'distance $r$')
        ax1.set_ylabel(r'probability')
        ax2.set_ylabel(r'$g(r)$')
        ax1.set_title(r"Distribution of distances")
        ax2.set_title(r"Time and particle averaged RDF")
        ax1.hist(distances_time_average, bins=n_bins)
        ax2.plot(r_linspace[1:], g_r)
        ax2.axhline(1, ls='--')
        plt.tight_layout()
        plt.show()

        return r_linspace[1:], g_r

    def temperature_analysis(self, N_start=3, deltaN=1):
        N_array = np.arange(N_start, self.n_particles, deltaN)
        rel_T_variances = np.zeros(len(N_array))
        for i in tqdm(range(len(N_array))):
            temperatures = np.array([self.get_T(step, N_array[i]) for step in range(self.steps)])
            var_T = np.var(temperatures)
            mean_T = np.mean(temperatures)
            rel_T_variances[i] = var_T / mean_T**2

        fig, (ax1) = plt.subplots(1, 1, figsize=(6,4))

        ax1.set_xlabel(r'N')
        ax1.set_ylabel(r'$\sigma_{T}^2$ / $\left<T\right>^2$')
        ax1.set_title('Relative variance of the temperature')
        
        ax1.plot(N_array, rel_T_variances, label=r'$\sigma_{T}^2$ / $\left<T\right>^2$')
        ax1.plot(N_array, 1/N_array, label=r'1/N')
        ax1.legend()

        plt.tight_layout()
        plt.show()
    
    def Epot_convergence(self, n_start=0):
        E_pot = np.sum(self.Lennard_Jones_matrix[:,:,0,:], axis=(0,1)) / 2
        E_pot_mean      = np.zeros(len(E_pot))
        E_pot_mean_std  = np.zeros(len(E_pot))
        for i in tqdm(range(len(E_pot))):
            E_pot_mean[i]     = np.mean(E_pot[:i+1])
            E_pot_mean_std[i] = np.std(E_pot_mean[:i+1], ddof=1)

        # set up the figure for the box plot
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13,4))
        ax1.set_title(r"Mean of $E_{pot}$")
        ax1.set_xlabel("N")
        ax1.set_ylabel(r"$\left<E_{pot}\right>$")
        ax2.set_title(r"Standard deviation of $\left<E_{pot}\right>$")
        ax2.set_xlabel("N")
        ax2.set_ylabel(r"$\sigma \cdot \sqrt{N}$")
        ax1.plot(E_pot_mean[n_start:], label=r"$\left<E_{pot}\right>$")
        ax2.plot(E_pot_mean_std[n_start:] * np.sqrt(np.arange(1,len(E_pot_mean_std[n_start:])+1)), label=r"$\sigma * \sqrt{N}$")
        ax3.plot(E_pot_mean_std[n_start:], label=r"$\sigma$")
        ax2.legend()
        ax3.legend()

        plt.tight_layout()
        plt.show()
        return E_pot

    def heat_integrator(self, step, dt=1, alpha=0.1):
        """ calculates the temperatures of the next step """

        T_shift_x_left  = np.roll(self.temperatures[:,:,step], -1, 0)
        T_shift_x_right = np.roll(self.temperatures[:,:,step],  1, 0)
        T_shift_y_left  = np.roll(self.temperatures[:,:,step], -1, 1)
        T_shift_y_right = np.roll(self.temperatures[:,:,step],  1, 1)

        dT_over_dt = alpha * (T_shift_x_left + T_shift_x_right + T_shift_y_left + T_shift_y_right - 4*self.temperatures[:,:,step]) / self.dist_neighbour**2

        self.temperatures[:,:,step+1] = self.temperatures[:,:,step] + dT_over_dt * dt

    def heat_grid_simulation(self, dt, alpha):
        """ performs the simulation of the heat transfer """

        for step in tqdm(range(self.steps)):
            self.heat_integrator(step, dt=dt, alpha=alpha)











