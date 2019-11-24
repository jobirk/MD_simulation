import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from IPython.display import HTML
from tqdm import tqdm

class Particle:

    def __init__(self, x=0, y=0, vx=0, vy=0, ax=0, ay=0, m=1, r=0.5):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.ax = ax
        self.ay = ay
        self.r = r
        self.m = m

    def __repr__(self):
        return str("This is a particle at %0.2f, %0.2f with v=%0.2f,%0.2f, v_tot=%0.8f"\
                   %(self.x, self.y, self.vx, self.vy, np.sqrt(self.vx**2 + self.vy**2)))

    def random_position(self, x_min=0, x_max=1, y_min=0, y_max=1):
        # method to set a random position for the particle in 
        # [x_min, x_max] and [y_min, y_max]
        self.x = np.random.uniform(x_min, x_max)
        self.y = np.random.uniform(y_min, y_max)

    def random_velocity(self, v_total = 1):
        # method to set a random velocity vector to the particle. 
        # The absolut value of the velocity can be chosen
        phi = np.random.uniform(0, 2*np.pi)
        self.vx = v_total * np.cos(phi)
        self.vy = v_total * np.sin(phi)
    
    # two methods which switch the sign of the x/y velocity if 
    # the particle is reflected at the y/x axis
    def reflection_x_axis(self):
        self.vy *= -1
    
    def reflection_y_axis(self):
        self.vx *= -1

# function for the movement of a particle    
def move(p, box_length_x, box_length_y, dt=1, pbc=True):
    if pbc:
        p.x = (p.x + p.vx*dt) % box_length_x
        p.y = (p.y + p.vy*dt) % box_length_y
    else:
        p.x = (p.x + p.vx*dt)
        p.y = (p.y + p.vy*dt)
    return p    

def dx_dy(p1, p2, box_length_x, box_length_y, pbc=True, verbose=False):
    dx = p1.x - p2.x
    dy = p1.y - p2.y
    if pbc:
        # if the distance is larger than half the box length
        # subtract the length, which gives the distance when
        # calculated over the borders
        if abs(dx) > box_length_x/2:
            if dx > 0:
                dx = dx - box_length_x
            elif dx < 0:
                dx = box_length_x + dx
            #print("used the modified dx =", dx)
    
        if abs(dy) > box_length_y/2:
            if dy > 0:
                dy = dy - box_length_y
            elif dy < 0:
                dy = box_length_y + dy
    d = np.sqrt(dx**2 + dy**2)
    if verbose:
        print("dx =", dx, ", dy =", dy)
    return dx, dy, d


""">>>>>> simulation class <<<<<<<"""

class box_simulation_many_particles():

    def __init__(self):
        self.box = [50,50]
        self.trajectories = []
        self.particles = []
        self.steps = 2000
        self.x_list = []
        self.y_list = []
        self.particle_start_velocity = 0.5
        self.particle_distances = np.zeros(1)

    def init_box(self, x, y):
        self.box = [x, y]

    def elastic_collision(self, p1, p2, pbc=True):

        # to calculate the velocity vectors of the two colliding particles
        # perform a basis transformation to a basis with one basis-vector
        # pointing along the connection line between p1 and p2 and the other
        # being perpendicular to that line.
        # The velocity components parallel to the connection line are
        # swapped and the components perpendicular to this line remain the same

        #dx, dy, distance = self.particle_distances(step_index, particle_index1, particle_index2)
        #dx_dy(p1, p2, self.box[0], self.box[1], pbc=pbc)

        # define the normalised vector along the connection line
        p = np.array([dx, dy]) / distance
        # define the basis transformation matrix B and the inverse
        B = np.array([[p[0], p[1]], [-p[1], p[0]]])
        B_inv = B.transpose()
        #print(np.dot(B, B_inv))

        v1 = np.array([p1.vx, p1.vy])
        v2 = np.array([p2.vx, p2.vy])
        v1_transformed = np.dot(B, v1)
        v2_transformed = np.dot(B, v2)
        # store the parallel components
        v1_parallel = v1_transformed[0]
        v2_parallel = v2_transformed[0]
        # print("Parallel components:", v1_parallel, v2_parallel)
        # update the velocity vectors, swap parallel component
        v1_transformed[0] = v2_parallel
        v2_transformed[0] = v1_parallel
        # transform the new vectors back in the original basis
        v1_prime = np.dot(B_inv, v1_transformed)
        v2_prime = np.dot(B_inv, v2_transformed)
        # update the velicity in the particle objects
        p1.vx, p1.vy = v1_prime[0], v1_prime[1]
        p2.vx, p2.vy = v2_prime[0], v2_prime[1]


    def calculate_distances(self, collide=True, pbc=True, print_dx_dy=False):
        """ method for checking which collisions are taking place """
        # to get the distances of the particles for the calculation of the
        # radial distribution function, store all distances in a numpy array
        # store the distances in an array, all the arrays in one super-array
        distances = np.array([np.zeros([3, len(self.particles)]) for p in self.particles])
        for i in range(len(self.particles)):
            counter = 0 
            for j in range(i, len(self.particles)):
                if i==j:
                    dx, dy, distance = 0, 0, 0
                else:
                    dx, dy, distance = dx_dy(self.particles[i], self.particles[j], self.box[0], self.box[1], 
                                             pbc=pbc, verbose=print_dx_dy)
                # update the distance entry in the array
                distances[i][:,j] = [ dx,  dy, distance]
                distances[j][:,i] = [-dx, -dy, distance]

                # check if the distance between the particles is smaller than 2 times the radius
                if (distance < (self.particles[i].r + self.particles[j].r) and collide):
                    # distance of particle centers is smaller than the
                    # sum of the two radi

                    # also check if the particles are moving away from each other
                    # -> then they should not collide (this is the case if
                    #    particles collided in the previous simulation step but
                    #    are still not far enough away from each other)
                    #    therefore, project the velocity vectors onto the connection line
                    #    between the two particles
                    # print("distance = ", distance)
                    if ((self.particles[i].vx * dx + \
                         self.particles[i].vy * dy) > 0 and\
                        (self.particles[j].vx * dx + \
                         self.particles[j].vy * dy) < 0):
                        #print("no collision, because moving away from each other", "distance:", distance,
                                #"dx", dx, "dy", dy)
                        pass
                    else:
                        self.elastic_collision(self.particles[i], self.particles[j])
        return distances

    def check_for_reflection(self):
        """ method for checking if a reflection at a border is taking place """
        for p in self.particles:
            # check for reflection
            if ((abs(p.x - self.box[0]) < p.r) and p.vx>0) or \
                (abs(p.x < p.r) and p.vx < 0) or p.x<0 or p.x>self.box[0]:
                p.reflection_y_axis()
            if ((abs(p.y - self.box[1]) < p.r) and p.vy>0) or \
                (abs(p.y < p.r) and p.vy < 0) or p.y<0 or p.y>self.box[1]:
                p.reflection_x_axis()

    def LJ_potential_and_force(self, particle_index, step_index,
                               C_12=9.847044e-6, C_6=6.2647225e-3, cutoff=0.33):
        """ function that returns the value of the LJ potential and the resulting force for particle i """
        # initialise a value of 0 for the potential particle i sees and a vector (0,0) for the corresponding force
        V_i = np.float(0) 
        F_i = np.array([0, 0])

        for j in range(len(self.particles)):
            if j == particle_index:
                continue
            #print("dx,dy,rij", self.particle_distances[step_index, particle_index][:,j])
            dx, dy, r_ij = self.particle_distances[step_index, particle_index][:,j]
            if r_ij < cutoff:
                dx = dx * cutoff/r_ij
                dy = dy * cutoff/r_ij
                r_ij = cutoff
            V_ij = C_12 / r_ij**12 - C_6 / r_ij**6
            F_ij = (12 * C_12 / r_ij**13 - 6 * C_6 / r_ij**7) * 1 / r_ij * np.array([dx, dy]) * 1000 
            #factor 1000 to get the right unit when translated into a velocity
            # add the force and potential value to the total value
            V_i += V_ij
            F_i = F_i + F_ij
        return V_i, F_i

    def verlet_update_positions(self, particle_index, step_index, dt=1, pbc=True, forces=True):
        """ function that moves a particle according to the velocity verlet algorithm """
        p = self.particles[particle_index]

        if step_index==0 or forces==False:
            V, Fx, Fy = 0, 0, 0
        else:
            V, Fx, Fy = self.Lennard_Jones[particle_index, :, step_index-1]

        ax, ay = Fx/p.m , Fy/p.m
        #print("acceleration in nm/ns**2:", ax, ay)
        p.x = p.x + p.vx * dt + 0.5 * ax * dt**2
        p.y = p.y + p.vy * dt + 0.5 * ay * dt**2

        if pbc:
            p.x = p.x % self.box[0]
            p.y = p.y % self.box[1]

        self.trajectories[particle_index][0:2,step_index] = \
                                    [self.particles[particle_index].x,  self.particles[particle_index].y]

    def verlet_update_velocities(self, particle_index, step_index, dt=1, pbc=True, forces=True):
        p = self.particles[particle_index]

        V, F_x_old, F_y_old, ax, ay = 0, 0, 0, 0, 0
        F = np.array([0,0])

        if forces and step_index!=0:
            V_old, F_x_old, F_y_old = self.Lennard_Jones[particle_index, :, step_index-1]
        ax_old, ay_old = F_x_old / p.m , F_y_old / p.m

        if forces:
            V, F = self.LJ_potential_and_force(particle_index, step_index)
            ax, ay = F[0] / p.m , F[1] / p.m

        p.vx = p.vx + 0.5 * (ax_old + ax) * dt
        p.vy = p.vy + 0.5 * (ay_old + ay) * dt

        self.trajectories[particle_index][2:4,step_index] = \
                                    [self.particles[particle_index].vx,  self.particles[particle_index].vy]
        self.Lennard_Jones[particle_index][:,step_index] = [V, F[0], F[1]]


    def simulate(self, n_particles, particle_radius=0.5, particle_mass=1, steps=2000,
                 particle_velocity=0.5, pbc=True, test_particles=[], step_interval=1,
                 print_dx_dy=False, forces=True, collisions=True):
        """ method to run the simulation and create the trajectories of the particles"""
        self.steps = steps
        self.particle_start_velocity = particle_velocity
        # array of arrays containing the particle trajectories
        self.trajectories = np.array([np.zeros([4,steps]) for i in range(n_particles+len(test_particles))])
        self.Lennard_Jones = np.array([np.zeros([3,steps]) for i in range(n_particles+len(test_particles))])
        self.kin_energies = np.array([np.zeros(self.steps) for i in range(n_particles+len(test_particles))])
        self.pot_energies = np.array([np.zeros(self.steps) for i in range(n_particles+len(test_particles))])

        # initialise particle with random position in the box and random velocity direction
        self.particles = []
        for i in range(n_particles):
            p = Particle(r=particle_radius, m=particle_mass)
            p.random_position(0, self.box[0], 0, self.box[1])
            p.random_velocity(particle_velocity)
            self.particles.append(p)

        if test_particles != []:
            for particle in test_particles:
                p = Particle(x=particle[0], y=particle[1], vx=particle[2], vy=particle[3], m=particle[4])
                self.particles.append(p)

        # create a numpy array containing arrays that store the distances between particles
        distances_placeholder = np.array([np.zeros([3, len(self.particles)]) for p in self.particles])
        self.particle_distances = np.array([ distances_placeholder for step in range(steps) ])

        for i in tqdm(range(steps)):
            #if i!=0: #dont want to collide them already in the start position
            self.particle_distances[i] = self.calculate_distances(pbc=pbc, print_dx_dy=print_dx_dy,
                                                                  collide=collisions)
            if not pbc:
                self.check_for_reflection()

            # append a copy of the particle to the trajectory
            for j in range(len(self.particles)):
                #move(self.particles[j], self.box[0], self.box[1], dt=step_interval, pbc=pbc)
                self.verlet_update_positions(j, i, dt=step_interval, pbc=pbc, forces=forces)
            # loop to update all velocities
            for j in range(len(self.particles)):
                self.verlet_update_velocities(j, i, dt=step_interval, pbc=pbc, forces=forces)

        self.kin_energies = 0.5 * self.particles[0].m * (self.trajectories[:,2,:]**2 + self.trajectories[:,3,:]**2)

    def plot_energies(self):

        E_kin = np.sum(self.kin_energies, axis=0) / 1000 # to make it kJ/mol
        E_pot = np.sum(self.Lennard_Jones[:,0,:], axis=0) / 2
        # set up the figure for the box plot
        fig, ax1 = plt.subplots()
        fig.set_figheight(4)
        fig.set_figwidth(4)
        #ax1.set_xlim((0, self.box[0]))
        #ax1.set_ylim((0, self.box[1]))

        ax1.set_xlabel('Time step')
        ax1.set_ylabel('Energy [kJ/mol]')

        ax1.plot(range(self.steps), E_kin, label=r"$E_{kin}$", color="g")
        ax1.plot(range(self.steps), E_pot, label=r"$E_{pot}$", color="r")
        ax1.plot(range(self.steps), E_kin + E_pot, label=r"$E_{kin} + E_{pot}$", color="b")
        ax1.legend()

        plt.tight_layout()
        plt.show()

    def plot_trajectories(self):
        """ method to plot the trajectories of all particles """
        x = np.array([self.trajectories[j][0,:] for j in range(len(self.particles))])
        y = np.array([self.trajectories[j][1,:] for j in range(len(self.particles))])

        # set up the figure for the box plot
        fig, ax1 = plt.subplots()
        fig.set_figheight(4)
        fig.set_figwidth(4)
        ax1.set_xlim((0, self.box[0]))
        ax1.set_ylim((0, self.box[1]))

        ax1.set_xlabel('position x')
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
                 for i in range(len(self.particles))]

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
        x = np.concatenate([self.trajectories[j][0,start:end] for j in range(len(self.particles))])
        y = np.concatenate([self.trajectories[j][1,start:end] for j in range(len(self.particles))])
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
        vx = np.concatenate([self.trajectories[j][2,start:end] for j in range(len(self.particles))])
        vy = np.concatenate([self.trajectories[j][3,start:end] for j in range(len(self.particles))])
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
        """ method to plot the radial distrib. fuc. """

        distances_time_average = self.particle_distances[:,:,2,:].flatten()
        mask = (distances_time_average != 0) #remove entires with distance=0
        distances_time_average = distances_time_average[mask]

        r_linspace = np.linspace(x_offset*dr, self.box[0]/2, n_linspace)

        def g(r, dr=dr):
            dn = np.zeros(len(r))
            rho = len(self.particles) / (self.box[0] * self.box[1])
            for i in tqdm(range(len(r))):
                dn[i] = len(distances_time_average[(r[i]<distances_time_average) & (distances_time_average<(r[i]+dr))])
            return 1 / (2 * np.pi * r * rho) * dn / dr * 1 / (self.steps * len(self.particles))

        g_r_linspace = g(r_linspace)

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_figheight(3)
        fig.set_figwidth(8)
        ax1.set_xlabel(r'distance $r$')
        ax2.set_xlabel(r'distance $r$')
        ax1.set_ylabel(r'probability')
        ax2.set_ylabel(r'$g(r)$')
        ax1.set_title(r"Distribution of distances")
        ax2.set_title(r"Time and particle averaged RDF")
        ax1.hist(distances_time_average, bins=n_bins)
        ax2.plot(r_linspace, g_r_linspace)
        plt.tight_layout()
        plt.show()

        return r_linspace, g_r_linspace








