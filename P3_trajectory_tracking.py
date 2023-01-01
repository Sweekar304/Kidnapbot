import typing as T

import numpy as np
from numpy import linalg

V_PREV_THRES = 0.0001

class TrajectoryTracker:
    """ Trajectory tracking controller using differential flatness """
    def __init__(self, kpx: float, kpy: float, kdx: float, kdy: float,
                 V_max: float = 0.5, om_max: float = 1) -> None:
        self.kpx = kpx
        self.kpy = kpy
        self.kdx = kdx
        self.kdy = kdy

        self.V_max = V_max
        self.om_max = om_max

        self.coeffs = np.zeros(8) # Polynomial coefficients for x(t) and y(t) as
                                  # returned by the differential flatness code

    def reset(self) -> None:
        self.V_prev = 0.
        self.om_prev = 0.
        self.t_prev = 0.

    def load_traj(self, times: np.ndarray, traj: np.ndarray) -> None:
        """ Loads in a new trajectory to follow, and resets the time """
        self.reset()
        self.traj_times = times
        self.traj = traj

    def get_desired_state(self, t: float) -> T.Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                     np.ndarray, np.ndarray, np.ndarray]:
        """
        Input:
            t: Current time
        Output:
            x_d, xd_d, xdd_d, y_d, yd_d, ydd_d: Desired state and derivatives
                at time t according to self.coeffs
        """
        x_d = np.interp(t,self.traj_times,self.traj[:,0])
        y_d = np.interp(t,self.traj_times,self.traj[:,1])
        xd_d = np.interp(t,self.traj_times,self.traj[:,3])
        yd_d = np.interp(t,self.traj_times,self.traj[:,4])
        xdd_d = np.interp(t,self.traj_times,self.traj[:,5])
        ydd_d = np.interp(t,self.traj_times,self.traj[:,6])

        return x_d, xd_d, xdd_d, y_d, yd_d, ydd_d

    def compute_control(self, x: float, y: float, th: float, t: float) -> T.Tuple[float, float]:
        """
        Inputs:
            x,y,th: Current state
            t: Current time
        Outputs:
            V, om: Control actions
        """

        dt = t - self.t_prev
        x_d, xd_d, xdd_d, y_d, yd_d, ydd_d = self.get_desired_state(t)

        ########## Code starts here ##########
        if self.V_prev<V_PREV_THRES: #Resetting to prevent velocity dropping to zero and to prevent system singularity
            self.V_prev = np.sqrt(xd_d**2 + yd_d**2)
        u1 = xdd_d+self.kpx*(x_d-x)+self.kdx*(xd_d-self.V_prev*np.cos(th)) #Defining control law
        u2 = ydd_d+self.kpy*(y_d-y)+self.kdy*(yd_d-self.V_prev*np.sin(th)) #Defining control law
        A_matrix = np.array([[np.cos(th),-self.V_prev*np.sin(th)],[np.sin(th),self.V_prev*np.cos(th)]]) #defining coefficient matrix
        B_matrix = np.array([u1,u2]) #Defining controls matrix
        C_matrix = np.linalg.solve(A_matrix,B_matrix) #Solving unknown
        om = C_matrix[1] #Finding omega
        V = C_matrix[0]*dt + self.V_prev #Integrating to find velocity
            
        ########## Code ends here ##########

        # apply control limits
        V = np.clip(V, -self.V_max, self.V_max)
        om = np.clip(om, -self.om_max, self.om_max)

        # save the commands that were applied and the time
        self.t_prev = t
        self.V_prev = V
        self.om_prev = om

        return V, om
