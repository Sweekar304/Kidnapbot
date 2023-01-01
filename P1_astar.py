import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils import plot_line_segments
from sre_constants import FAILURE

class AStar(object):
    """Represents a motion planning problem to be solved using A*"""

    def __init__(self, statespace_lo, statespace_hi, x_init, x_goal, occupancy, resolution=1.0)->None: # probably we can avoid the corners of the walls by changing the resolution
        self.statespace_lo = np.array(statespace_lo)         # state space lower bound (e.g., [-5, -5])
        self.statespace_hi = np.array(statespace_hi)         # state space upper bound (e.g., [5, 5])
        self.occupancy = occupancy                 # occupancy grid (a DetOccupancyGrid2D object)
        self.resolution = resolution               # resolution of the discretization of state space (cell/m)
        self.x_init = self.snap_to_grid(x_init)    # initial state
        self.x_goal = self.snap_to_grid(x_goal)    # goal state

        self.closed_set = set()    # the set containing the states that have been visited
        self.open_set = set()      # the set containing the states that are condidate for future expension
        self.est_cost_through = {} #np.inf*np.ones((self.statespace_hi - self.statespace_lo)//self.resolution + 1)  # 2d map of the estimated cost from start to goal passing through state (often called f score)
        self.cost_to_arrive = {} #np.inf*np.ones((self.statespace_hi - self.statespace_lo)//self.resolution + 1)    # 2d map of the cost-to-arrive at state from start (often called g score)
        self.came_from = {}         # dictionary keeping track of each state's parent to reconstruct the path

        self.open_set.add(self.x_init)
        self.cost_to_arrive[self.x_init] = 0
        self.est_cost_through[self.x_init] = self.distance(self.x_init,self.x_goal)

        self.path = None        # the final path as a list of states

    def is_free(self, x):
        """
        Checks if a give state x is free, meaning it is inside the bounds of the map and
        is not inside any obstacle.
        Inputs:
            x: state tuple
        Output:
            Boolean True/False
        Hint: self.occupancy is a DetOccupancyGrid2D object, take a look at its methods for what might be
              useful here
        """
        ########## Code starts here ##########
        if x==self.x_init or x==self.x_goal: #Checks for initial and goal position
            return True
        if x[0]<self.statespace_lo[0] or x[0]>self.statespace_hi[0]: #Binds x limits
            return False
        if x[1]<self.statespace_lo[1] or x[1]>self.statespace_hi[1]: #binds y limits
            return False
        if not self.occupancy.is_free(x): #Checks obstacles
            return False
        return True

        ########## Code ends here ##########

    def distance(self, x1, x2):
        """
        Computes the Euclidean distance between two states.
        Inputs:
            x1: First state tuple
            x2: Second state tuple
        Output:
            Float Euclidean distance

        HINT: This should take one line. Tuples can be converted to numpy arrays using np.array().
        """
        ########## Code starts here ##########
        #return np.linalg.norm(np.array(x2)-np.array(x1))
       #return np.sqrt(np.dot(np.transpose(a2-a1),(a2-a1)))
        return np.sqrt(np.sum((np.array(x2)-np.array(x1))**2))  #Most simplest means for computation
        ########## Code ends here ##########

    def snap_to_grid(self, x):
        """ Returns the closest point on a discrete state grid
        Input:
            x: tuple state
        Output:
            A tuple that represents the closest point to x on the discrete state grid
        """
        return (self.resolution*round(x[0]/self.resolution), self.resolution*round(x[1]/self.resolution))

    def get_neighbors(self, x):
        """
        Gets the FREE neighbor states of a given state x. Assumes a motion model
        where we can move up, down, left, right, or along the diagonals by an
        amount equal to self.resolution.
        Input:
            x: tuple state
        Ouput:
            List of neighbors that are free, as a list of TUPLES

        HINTS: Use self.is_free to check whether a given state is indeed free.
               Use self.snap_to_grid (see above) to ensure that the neighbors
               you compute are actually on the discrete grid, i.e., if you were
               to compute neighbors by adding/subtracting self.resolution from x,
               numerical errors could creep in over the course of many additions
               and cause grid point equality checks to fail. To remedy this, you
               should make sure that every neighbor is snapped to the grid as it
               is computed.
        """
        neighbors = []
        ########## Code starts here ##########
        #Assume resolution length is the same even along diagonal
        x = np.array(x) #Could have done same without use of this line and tuple
        x_0 = (x[0]-self.resolution,x[1]) #x-dx,y
        x_0 = self.snap_to_grid(tuple(x_0))
        if self.is_free(x_0):
            neighbors.append(x_0)
        x_1 = (x[0]-(self.resolution/1),x[1]+(self.resolution/1)) #x-dx,y+dy
        x_1 = self.snap_to_grid(tuple(x_1))
        if self.is_free(x_1):
            neighbors.append(x_1)
        x_2 = (x[0],x[1]+self.resolution) #x,y+dy
        x_2 = self.snap_to_grid(tuple(x_2))            
        if self.is_free(x_2):
            neighbors.append(x_2)
        x_3 = (x[0]+(self.resolution/1),x[1]+(self.resolution/1)) #x+dx,y+dy
        x_3 = self.snap_to_grid(tuple(x_3))
        if self.is_free(x_3):
            neighbors.append(x_3)
        x_4 = (x[0]+self.resolution,x[1]) #x+dx,y
        x_4 = self.snap_to_grid(tuple(x_4))
        if self.is_free(x_4):
            neighbors.append(x_4)
        x_5 = (x[0]+(self.resolution/1),x[1]-(self.resolution/1)) #x+dx,y-dy
        x_5 = self.snap_to_grid(tuple(x_5))
        if self.is_free(x_5):
            neighbors.append(x_5)
        x_6 = (x[0],x[1]-self.resolution) #x,y-dy
        x_6 = self.snap_to_grid(tuple(x_6))
        if self.is_free(x_6):
            neighbors.append(x_6)
        x_7 = (x[0]-(self.resolution/1),x[1]-(self.resolution/1)) #x-dx,y-dy
        x_7 = self.snap_to_grid(tuple(x_7))
        if self.is_free(x_7):
            neighbors.append(x_7)
            
        """
        neighbors.append(self.snap_to_grid(x[0]-self.resolution,x[1]))
        neighbors.append(self.snap_to_grid(x[0]-self.resolution,x[1]+self.resolution))
        neighbors.append(self.snap_to_grid(x[0],x[1]+self.resolution))
        neighbors.append(self.snap_to_grid(x[0]+self.resolution,x[1]+self.resolution))
        neighbors.append(self.snap_to_grid(x[0]+self.resolution,x[1]))
        neighbors.append(self.snap_to_grid(x[0]-self.resolution,x[1]-self.resolution))
        neighbors.append( self.snap_to_grid(x[0],x[1]-self.resolution))
        neighbors.append(self.snap_to_grid(x[0]-self.resolution,x[1]-self.resolution))
        neighbors2 = np.copy(neighbors)
        for i in neighbors2:
            if not self.is_free(i):
                neighbors.remove(i)
        
        """
        ########## Code ends here ##########
        return neighbors

    def find_best_est_cost_through(self):
        """
        Gets the state in open_set that has the lowest est_cost_through
        Output: A tuple, the state found in open_set that has the lowest est_cost_through
        """
        return min(self.open_set, key=lambda x: self.est_cost_through[x])

    def reconstruct_path(self):
        """
        Use the came_from map to reconstruct a path from the initial location to
        the goal location
        Output:
            A list of tuples, which is a list of the states that go from start to goal
        """
        self.path = [self.x_goal]
        current = self.path[-1]
        while current != self.x_init:
            self.path.append(self.came_from[current])
            current = self.path[-1]
        self.path = list(reversed(self.path))
        return self.path

    def plot_path(self, fig_num=0, show_init_label=True):
        """Plots the path found in self.path and the obstacles"""
        if not self.path:
            return

        self.occupancy.plot(fig_num)

        solution_path = np.array(self.path) * self.resolution
        plt.plot(solution_path[:,0],solution_path[:,1], color="green", linewidth=2, label="A* solution path", zorder=10)
        plt.scatter([self.x_init[0]*self.resolution, self.x_goal[0]*self.resolution], [self.x_init[1]*self.resolution, self.x_goal[1]*self.resolution], color="green", s=30, zorder=10)
        if show_init_label:
            plt.annotate(r"$x_{init}$", np.array(self.x_init)*self.resolution + np.array([.2, .2]), fontsize=16)
        plt.annotate(r"$x_{goal}$", np.array(self.x_goal)*self.resolution + np.array([.2, .2]), fontsize=16)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), fancybox=True, ncol=3)

        plt.axis([0, self.occupancy.width, 0, self.occupancy.height])

    def plot_tree(self, point_size=15):
        plot_line_segments([(x, self.came_from[x]) for x in self.open_set if x != self.x_init], linewidth=1, color="blue", alpha=0.2)
        plot_line_segments([(x, self.came_from[x]) for x in self.closed_set if x != self.x_init], linewidth=1, color="blue", alpha=0.2)
        px = [x[0] for x in self.open_set | self.closed_set if x != self.x_init and x != self.x_goal]
        py = [x[1] for x in self.open_set | self.closed_set if x != self.x_init and x != self.x_goal]
        plt.scatter(px, py, color="blue", s=point_size, zorder=10, alpha=0.2)

    def solve(self):
        """
        Solves the planning problem using the A* search algorithm. It places
        the solution as a list of tuples (each representing a state) that go
        from self.x_init to self.x_goal inside the variable self.path
        Input:
            None
        Output:
            Boolean, True if a solution from x_init to x_goal was found

        HINTS:  We're representing the open and closed sets using python's built-in
                set() class. This allows easily adding and removing items using
                .add(item) and .remove(item) respectively, as well as checking for
                set membership efficiently using the syntax "if item in set".
        """
        ########## Code starts here ##########
        while len(self.open_set)>0:
            x_current = self.find_best_est_cost_through() #Calculate state with least cost
            if x_current == self.x_goal:  #Checks goal and exits loop
                return self.reconstruct_path()
            self.open_set.remove(x_current) #remove from future set and add it past path
            self.closed_set.add(x_current)
            for x_neigh in self.get_neighbors(x_current):
                if x_neigh in self.closed_set: #No need to continue checking since path is traversed
                    continue
                tentative_cost_to_arrive = self.cost_to_arrive[x_current]+self.distance(x_current,x_neigh)
                if x_neigh not in self.open_set: #Add neighbour to the plan
                    self.open_set.add(x_neigh)
                elif tentative_cost_to_arrive>self.cost_to_arrive[x_neigh]: #Next neighbour cost is high; break
                    continue
                self.came_from[x_neigh]= x_current #if cost is less, update the x_neigh
                self.cost_to_arrive[x_neigh] = tentative_cost_to_arrive
                self.est_cost_through[x_neigh] = tentative_cost_to_arrive+self.distance(x_neigh,self.x_goal)
        return FAILURE
                
            
        
        ########## Code ends here ##########

class DetOccupancyGrid2D(object):
    """
    A 2D state space grid with a set of rectangular obstacles. The grid is
    fully deterministic
    """
    def __init__(self, width, height, obstacles):
        self.width = width
        self.height = height
        self.obstacles = obstacles

    def is_free(self, x):
        """Verifies that point is not inside any obstacles"""
        for obs in self.obstacles:
            inside = True
            for dim in range(len(x)):
                if x[dim] < obs[0][dim] or x[dim] > obs[1][dim]:  ###Sweekar Changed this remive 0.1
                    inside = False
                    break
            if inside:
                return False
        return True

    def plot(self, fig_num=0):
        """Plots the space and its obstacles"""
        fig = plt.figure(fig_num)
        ax = fig.add_subplot(111, aspect='equal')
        for obs in self.obstacles:
            ax.add_patch(
            patches.Rectangle(
            obs[0],
            obs[1][0]-obs[0][0],
            obs[1][1]-obs[0][1],))
        ax.set(xlim=(0,self.width), ylim=(0,self.height))
