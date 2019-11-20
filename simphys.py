import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from IPython.display import HTML
from tqdm import tqdm

class Particle:

    def __init__(self, x=0, y=0, vx=0, vy=0, m=1, r=0.5):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
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
    if verbose:
        print("dx =", dx, ", dy =", dy)
    return dx, dy


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

        dx, dy = dx_dy(p1, p2, self.box[0], self.box[1], pbc=pbc)

        # define the normalised vector along the connection line
        p = np.array([dx, dy]) / np.sqrt(dx**2 + dy**2)
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


    def elastic_collision_wiki(self, p1, p2, pbc=True):

        # to calculate the directions of the velocities after the collision,
        # one needs to calculate the tangent t along the two particles at the collision point
        # this can be done by inverting and inserting a factor -1 to the slope s between the
        # center points of the two particles
        # ...the calculations are from the german Wikipedia page "Stoß"

        delta_x, delta_y = dx_dy(p1, p2, self.box[0], self.box[1], pbc=pbc)
        # if delta_x = 0 or delta_y = 0, the collision is equivalent to a collision at the box border
        if delta_y == 0:
            p1.vx *= -1
            p2.vx *= -1
            return

        if delta_x == 0:
            p1.vy *= -1
            p2.vy *= -1
            return

        # calculating the tangent
        s = delta_y / delta_x
        t = -1 / s

        # calculate slope sv1 and sv2 of vectors v1 to p1 and v2 to p2
        if p1.vx==0:
            p1.vx=1e-99
        if p2.vx==0:
            p2.vx=1e-99
        sv1 = p1.vy / p1.vx
        sv2 = p2.vy / p2.vx

        # for each particle, calculate the projections of vx (vy) onto the tangent
        # and perpendicular to the tanget (on s)

        # particle 1
        xt1 = p1.vx * (s - sv1) / (s - t)
        xs1 = p1.vx * (t - sv1) / (t - s)

        yt1 = t * xt1
        ys1 = s * xs1

        #particle 2
        xt2 = p2.vx * (s - sv2) / (s - t)
        xs2 = p2.vx * (t - sv2) / (t - s)

        yt2 = t * xt2
        ys2 = s * xs2

        # calculate the resulting velocity vectors
        p1.vx = xt1 + xs2
        p1.vy = yt1 + ys2

        p2.vx = xt2 + xs1
        p2.vy = yt2 + ys1

    def calculate_distances(self, collide=True, pbc=True, print_dx_dy=False):
        """ method for checking which collisions are taking place """
        # to get the distances of the particles for the calculation of the
        # radial distribution function, store all distances in a numpy array
        # store the distances in an array, all the arrays in one super-array
        distances = np.array([np.zeros(len(self.particles)-1) for p in self.particles])
        for i in range(len(self.particles)):
            counter = 0 
            for j in range(len(self.particles)):
                if i==j:
                    continue
                dx, dy = dx_dy(self.particles[i], self.particles[j], self.box[0], self.box[1], 
                               pbc=pbc, verbose=print_dx_dy)
                distance = np.sqrt(dx**2 + dy**2)
                # update the distance entry in the array
                distances[i][counter] = distance
                counter += 1
                if j>i: # these particles were already collided if necessary
                    continue
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


    def simulate(self, n_particles, particle_radius=0.5, particle_mass=1, steps=2000,
                 particle_velocity=0.5, pbc=True, test_particles=[], step_interval=1,
                 print_dx_dy=False):
        """ method to run the simulation and create the trajectories of the particles"""
        self.steps = steps
        self.particle_start_velocity = particle_velocity
        # array of arrays containing the particle trajectories
        self.trajectories = np.array([np.zeros([4,steps]) for i in range(n_particles+len(test_particles))])

        # initialise particle with random position in the box and random velocity direction
        self.particles = []
        for i in range(n_particles):
            p = Particle(r=particle_radius, m=particle_mass)
            p.random_position(0, self.box[0], 0, self.box[1])
            p.random_velocity(particle_velocity)
            self.particles.append(p)

        if test_particles != []:
            for particle in test_particles:
                p = Particle(x=particle[0], y=particle[1], vx=particle[2], vy=particle[3], r=particle[4])
                self.particles.append(p)

        # create a numpy array containing arrays that store the distances between particles
        distances_placeholder = np.array([np.zeros(len(self.particles)-1) for p in self.particles])
        self.particle_distances = np.array([ distances_placeholder for step in range(steps) ])

        for i in tqdm(range(steps)):
            #if i!=0: #dont want to collide them already in the start position
            self.particle_distances[i] = self.calculate_distances(pbc=pbc, print_dx_dy=print_dx_dy)
            if not pbc:
                self.check_for_reflection()

            # append a copy of the particle to the trajectory
            for j in range(len(self.particles)):
                self.trajectories[j][:,i] = [self.particles[j].x,  self.particles[j].y, \
                                             self.particles[j].vx, self.particles[j].vy]
                move(self.particles[j], self.box[0], self.box[1], dt=step_interval, pbc=pbc)

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
            for j,line in enumerate(lines):
                x = self.trajectories[j][0,i]
                y = self.trajectories[j][1,i]
                line.set_data(x, y)
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
        fig.set_figheight(4)
        fig.set_figwidth(13)

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

    def velocity_distributions(self, start=0, end="standard", n_bins=50):
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
                 range=(-2.5*self.particle_start_velocity, 2.5*self.particle_start_velocity))
        ax2.hist(vy, bins=n_bins, density=1,
                 range=(-2.5*self.particle_start_velocity, 2.5*self.particle_start_velocity))
        v_tot = np.sqrt(vx**2+vy**2)
        ax3.hist(v_tot, bins=n_bins, density=1, range=(0, 3.1*self.particle_start_velocity))

        plt.tight_layout()
        plt.show()

    def radial_distribution_function(self, n_bins=20, dr=0.01, n_linspace=200):
        """ method to plot the radial distrib. fuc. """

        n_distances_total = self.steps * len(self.particles) * (len(self.particles) -1)
        distances_time_average = np.reshape(self.particle_distances, n_distances_total)

        r_linspace = np.linspace(dr, self.box[0]/2, n_linspace)

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








