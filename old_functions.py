
    def elastic_collision(self, p1, p2, pbc=True):

        # to calculate the velocity vectors of the two colliding particles
        # perform a basis transformation to a basis with one basis-vector
        # pointing along the connection line between p1 and p2 and the other
        # being perpendicular to that line.
        # The velocity components parallel to the connection line are
        # swapped and the components perpendicular to this line remain the same

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

    def calculate_distances(self, step_index, collide=True, pbc=True):
        """ method for calculating all distances between particles at a given step"""
        # to get the distances of the particles for the calculation of the
        # radial distribution function, store all distances in a numpy array
        distances = np.zeros([self.n_particles, self.n_particles, 3])
        for i in range(self.n_particles):
            for j in range(i+1, self.n_particles):
                dx, dy, distance = dx_dy(self.particles[i], self.particles[j], self.box[0], self.box[1], 
                                             pbc=pbc)
                # update the distance entry in the array
                distances[i, j, :] = [ dx,  dy, distance]
                distances[j, i, :] = [-dx, -dy, distance]

                # this is a leftover from the elastic collisions
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
                        pass
                        #self.elastic_collision(self.particles[i], self.particles[j])
        self.particle_distances[:, :, :, step_index] = distances
