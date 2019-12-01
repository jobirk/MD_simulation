import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from IPython.display import HTML
from tqdm import tqdm
import time
import sys


def distance_pbc_transformation(distance, box_length):
    """ transforms the distance dx or dy with respect to periodic
        boundary conditions """
    return (distance + (box_length / 2)) % box_length - (box_length / 2)

def Lennard_Jones_force(r, dx, dy, C_12=9.847044e-6, C_6=6.2647225e-3, \
                        cutoff=0.33, use_cutoff=True):
    """ Lennard Jones Force in N/mol """
    dx_values = np.copy(dx)
    dy_values = np.copy(dy)
    r_values  = np.copy(r)

    #dirty workaround to avoid division by zero
    zero_indices = np.where(r_values==0)
    r_values[zero_indices] = 42e10

    if use_cutoff==True:
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
    # set F=0 at entries where r=0
    F[0][zero_indices] = 0
    F[1][zero_indices] = 0
    return F

def Lennard_Jones(r, C_12=9.847044e-6, C_6=6.2647225e-3, cutoff=0.33, \
                  use_cutoff=True):
    """ function that calculates an array of V_ij values (in J/mol) 
        corresponding to a distance array containing r_ij values (distances) """
    # the distance array usually contains entries with r=0, e.g. r_11
    # but we only want to use the non-zero entries for the calculation
    non_diagonal_indices = np.where(r!=0)
    V = np.zeros(r.shape)
    r_pow_6 = r[non_diagonal_indices]**6

    V[non_diagonal_indices] = C_12 / r_pow_6**2 - C_6 / r_pow_6
    LJ_cutoff = C_12 / cutoff**12 - C_6 / cutoff**6

    if use_cutoff==True:
        # correct the values below the cutoff
        cutoff_indices = np.where(r<cutoff)
        # this is the potential if assuming a 
        # constant slope for r < cutoff
        V[cutoff_indices] = LJ_cutoff + \
            Lennard_Jones_force(np.array([cutoff]), np.array([cutoff]), 
                                np.array([0]))[0] * (cutoff-r[cutoff_indices]) 
        diagonal_indices = np.where(r==0)
        V[diagonal_indices] = 0

    return V


""">>>>>>>>>>>>>>>>>>> simulation class <<<<<<<<<<<<<<<<<"""

class box_simulation():

    def __init__(self, box_x=50, box_y=50, n_particles=50, n_steps=2000, \
                 particle_mass=0.018, pbc=True):
        self.box = [box_x, box_y]
        self.n_particles = n_particles
        self.pbc = pbc
        self.steps = n_steps
        self.particle_distances = []
        self.trajectories = []
        self.particles = []
        self.kin_energies = []
        self.pot_energies = []
        self.Lennard_Jones_matrix = []
        self.particle_mass = particle_mass

    def init_box(self, x, y):
        self.box = [x, y]

    def generate_particles(self, test_particles=[], particle_radius=1, v=1, m=1, 
                           load_from_file=False, new_velocities=False):
        """ method to generate a starting position of particles and velocities
            distributed randomly in the box. Also creates the essential arrays
            to save the simulation result. """

        if load_from_file:
            traj_from_file = np.loadtxt(load_from_file, dtype=float)
            self.n_particles = traj_from_file.shape[0]
            self.trajectories = np.zeros([self.n_particles, 4, self.steps+1])
            self.trajectories[:,:,0] = traj_from_file
            print("The initial state is loaded from the file '%s'" \
                  %(load_from_file))
            if new_velocities:
                v_angles = np.random.uniform(0, 2*np.pi, self.n_particles)
                self.trajectories[:,2,0] = v * np.cos(v_angles)
                self.trajectories[:,3,0] = v * np.sin(v_angles)


        else:
            # generate the new particles with random numbers
            n_random = self.n_particles
            #total number incl. test particles
            self.n_particles += len(test_particles) 
            self.trajectories = np.zeros([self.n_particles, 4, self.steps+1])

            positions = np.random.uniform(0, self.box[0], (2, n_random))
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

        # also setup the necessary arrays to save the simulaiton
        self.particle_distances   = np.zeros([self.n_particles, \
                                            self.n_particles, 3, self.steps+1])
        self.Lennard_Jones_matrix = np.zeros([self.n_particles, \
                                            self.n_particles, 3, self.steps+1])
        self.kin_energies         = np.zeros([self.n_particles, self.steps+1])
        self.pot_energies         = np.zeros([self.n_particles, self.steps+1])

    def save_particle_positions(self, step, filename="no_filename.txt"):
        """ method to save the particle positions at given step to file """
        np.savetxt(filename, self.trajectories[:,:,step])
        print("Saved the particle positions of step %i to file '%s'" \
               %(step,filename))

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

    def calculate_LJ_potential_and_force(self, step_index, use_cutoff=True):
        """ function that calculates the matrix of LJ potentials 
            and forces between all particles """
        self.Lennard_Jones_matrix[:, :, 0, step_index] = \
                Lennard_Jones(self.particle_distances[:, :, 2, step_index], 
                              use_cutoff=use_cutoff)
        self.Lennard_Jones_matrix[:, :, 1, step_index] = \
            Lennard_Jones_force(self.particle_distances[:, :, 2, step_index], 
                                self.particle_distances[:, :, 0, step_index],
                                self.particle_distances[:, :, 1, step_index], 
                                use_cutoff=use_cutoff)[0]
        self.Lennard_Jones_matrix[:, :, 2, step_index] = \
            Lennard_Jones_force(self.particle_distances[:, :, 2, step_index], 
                                self.particle_distances[:, :, 0, step_index],
                                self.particle_distances[:, :, 1, step_index],
                                use_cutoff=use_cutoff)[1]

    def verlet_update_positions(self, step_index, dt=1):
        """ function that moves a particle according to the 
            velocity verlet algorithm """
        #p = self.particles[particle_index]
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

        #print("Step:", step_index, self.trajectories[:, :, step_index])

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

    def simulate_MD(self, step_interval=1, use_cutoff=True):
        """ method to run the simulation and create the trajectories """

        self.calculate_distances(0)
        self.calculate_LJ_potential_and_force(0, use_cutoff=use_cutoff)

        # >>>> simulation <<<<
        for step in tqdm(range(self.steps)):
            # update all positions
            self.verlet_update_positions(step, dt=step_interval)
            # calculate the new values of the LJ potential and force
            self.calculate_distances(step+1)
            self.calculate_LJ_potential_and_force(step+1, use_cutoff=use_cutoff)
            # update all velocities
            self.verlet_update_velocities(step, dt=step_interval)

        self.kin_energies = 0.5 * self.particle_mass * \
                (self.trajectories[:,2,:]**2 + self.trajectories[:,3,:]**2)

    def calc_F_direction(self, step):
        """ returns the normalised F vector at given time step """
        self.calculate_distances(step)
        self.calculate_LJ_potential_and_force(step, use_cutoff=False)
        Fx = np.sum(self.Lennard_Jones_matrix[:, :, 1, step], axis=1)
        Fy = np.sum(self.Lennard_Jones_matrix[:, :, 2, step], axis=1)
        F_abs = np.sqrt(Fx**2 + Fy**2)
        # build r vector of length 0.05nm along F
        r_x = Fx / F_abs
        r_y = Fy / F_abs
        return np.array([r_x, r_y])

    def E_pot(self, step):
        """ returns the potential energie at a given simulation step """
        return np.sum(self.Lennard_Jones_matrix[:, :, 0, step], axis=(0,1))/2
    
    def move(self, step, r_x, r_y, length=0.05):
        """ moves all particles along r """
        self.trajectories[:, 0, step+1] = (self.trajectories[:, 0, step] + \
                                           r_x*length) % self.box[0]
        self.trajectories[:, 1, step+1] = (self.trajectories[:, 1, step] + \
                                           r_y*length) % self.box[1]

    def run_SD(self, step_length=0.01, plot_from=False):
        """ method to run the minimisation of the total potential energy of
            the system using the method of steepest descent """

        step = 0
        run_loop = True
        counts_else = 0
        E_1_minus_E_2 = []

        r = self.calc_F_direction(0)
        E_1 = self.E_pot(0)
        E_2 = E_1 - 1
        counter_E1_larger_E2 = 0

        while run_loop and step<self.steps:

            continue_moving = True

            while continue_moving and step<self.steps:
                self.move(step, r[0], r[1], length=step_length)
                self.calculate_distances(step+1)
                self.calculate_LJ_potential_and_force(step+1, use_cutoff=False)
                E_2 = self.E_pot(step+1)
                E_1_minus_E_2.append(E_1-E_2)
                step += 1
                if E_1 > E_2:
                    # continue, and set the old E_2 as new E_1
                    E_1 = E_2
                    counter_E1_larger_E2 += 1
                else:
                    # dont move in that direction any more
                    continue_moving = False

            # calculate the new direction of the Force vector
            r = self.calc_F_direction(step)

        print("counts E1 > E2:", counter_E1_larger_E2, "steps total:", step) 
        steps = range(len(E_1_minus_E_2))

        if plot_from:
            """ plot the evolution of the total potential energy and the
                difference E_1 - E_2 for each iteration step """
            E_pot = np.sum(self.Lennard_Jones_matrix[:,:,0,:], axis=(0,1)) / 2
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_figheight(4)
            fig.set_figwidth(13)

            ax1.set_xlabel(r"simulation step")
            ax1.set_ylabel('Energy [kJ/mol]')
            ax1.set_title("Evolution of potential energy")
            ax2.set_title(r"Energy difference $E_1 - E_2$")
            ax2.set_ylabel(r"$E_1 - E_2$")
            ax2.set_xlabel(r"simulation step")

            ax1.plot(range(plot_from,len(E_pot)), E_pot[plot_from:], label=r"$E_{pot}$", color="r")
            ax2.plot(steps[plot_from:], E_1_minus_E_2[plot_from:], 
                     label=r"$E_1 - E_2$")
            ax2.axhline(0, color="g")

            ax1.legend()
            ax2.legend()

            plt.tight_layout()
            plt.show()

    def plot_energies(self):
        E_kin = np.sum(self.kin_energies, axis=0) / 1000 # to make it kJ/mol
        E_pot = np.sum(self.Lennard_Jones_matrix[:,:,0,:], axis=(0,1)) / 2
        # divide by two because otherwise all values are double counted
        E_kin_diff = E_kin[1:] - E_kin[:-1]
        E_pot_diff = E_pot[1:] - E_pot[:-1]
        E_tot = E_kin + E_pot
        E_diff = E_kin - E_pot
        E_tot_diff = E_kin_diff + E_pot_diff
        # set up the figure for the box plot
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_figheight(4)
        fig.set_figwidth(13)

        ax1.set_xlabel(r'Time t [ns]')
        ax1.set_ylabel('Energy [kJ/mol]')
        ax2.set_xlabel(r'Time t [ns]')
        ax2.set_ylabel('Energy [kJ/mol]')
        ax1.set_title('Energy as a function of time')
        ax2.set_title('Difference in total energy per step')

        time_range = np.arange(self.steps+1) * 2e-6 #time in ns

        ax1.plot(time_range, E_kin, label=r"$E_{kin}$", color="g")
        ax1.plot(time_range, E_pot, label=r"$E_{pot}$", color="r")
        ax1.plot(time_range, E_tot, label=r"$E_{kin} + E_{pot}$", color="b")
        ax1.axhline(E_tot[0], ls="--", color="b")
        ax1.legend()

        ax2.plot(time_range[1:], E_tot_diff, label=r"$E_{tot}(t+\Delta t) - E_{tot}(t)$", color="b")
        ax2.axhline(0, ls="--", color="b")
        ax2.legend()

        plt.tight_layout()
        plt.show()

    def plot_trajectories(self):
        """ method to plot the trajectories of all particles """
        x = np.array([self.trajectories[j][0,:] for j in range(self.n_particles)])
        y = np.array([self.trajectories[j][1,:] for j in range(self.n_particles)])

        # set up the figure for the box plot
        fig, ax1 = plt.subplots()
        fig.set_figheight(4)
        fig.set_figwidth(4)
        ax1.set_xlim((0, self.box[0]))
        ax1.set_ylim((0, self.box[1]))

        ax1.set_xlabel('sposition x')
        ax1.set_ylabel('position y')

        for xi, yi in zip(x, y):
            ax1.plot(xi, yi,"o", ms=1)
        plt.tight_layout()
        plt.show()

    def animate_trajectories(self, animation_interval=30, dot_size=5):
        """ method to animate the particle movement """
        fig, ax = plt.subplots()
        fig.set_figheight(6)
        fig.set_figwidth(6)

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

        def animate(i):
            for j in range(len(lines)):
                x = self.trajectories[j][0,i]
                y = self.trajectories[j][1,i]
                lines[j].set_data(x, y)
            return lines

        anim = animation.FuncAnimation(fig, animate, init_func=init, \
                                   frames=self.steps, interval=animation_interval, blit=True)
        #plt.show()
        return HTML(anim.to_html5_video())
        #return HTML(anim.to_jshtml())

    def occupation(self, start=0, end=0, n_bins=50):
        """ method to plot the occupation on the x-y plane as well
        as the projections on the x and y axis """
        if end==0:
            end = self.steps
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        fig.set_figheight(3)
        fig.set_figwidth(15)

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

    def velocity_distributions(self, start=0, end="standard", n_bins=50, width_factor=1):
        """ methdo to plot the velocity distributions """
        if end=="standard":
            end = self.steps
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        fig.set_figheight(3)
        fig.set_figwidth(15)

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
        ax1.hist(vx, bins=n_bins, density=1,
                 range=(-width_factor*self.particle_start_velocity, width_factor*self.particle_start_velocity))
        ax2.hist(vy, bins=n_bins, density=1,
                 range=(-width_factor*self.particle_start_velocity, width_factor*self.particle_start_velocity))
        v_tot = np.sqrt(vx**2+vy**2)
        ax3.hist(v_tot, bins=n_bins, density=1, range=(0, 6.1*self.particle_start_velocity))

        plt.tight_layout()
        plt.show()

    def radial_distribution_function(self, n_bins=20, dr=0.01, n_linspace=200, x_offset=5):
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

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_figheight(4)
        fig.set_figwidth(13)
        ax1.set_xlabel(r'distance $r$')
        ax2.set_xlabel(r'distance $r$')
        ax1.set_ylabel(r'probability')
        ax2.set_ylabel(r'$g(r)$')
        ax1.set_title(r"Distribution of distances")
        ax2.set_title(r"Time and particle averaged RDF")
        ax1.hist(distances_time_average, bins=n_bins)
        ax2.plot(r_linspace[1:], g_r)
        plt.tight_layout()
        plt.show()

        return r_linspace[1:], g_r

