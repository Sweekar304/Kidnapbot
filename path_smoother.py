import numpy as np
import scipy.interpolate as intp

def compute_smoothed_traj(path, V_des, k, alpha, dt):
    """
    Fit cubic spline to a path and generate a resulting trajectory for our
    wheeled robot.

    Inputs:
        path (np.array [N,2]): Initial path
        V_des (float): Desired nominal velocity, used as a heuristic to assign nominal
            times to points in the initial path
        k (int): The degree of the spline fit.
            For this assignment, k should equal 3 (see documentation for
            scipy.interpolate.splrep)
        alpha (float): Smoothing parameter (see documentation for
            scipy.interpolate.splrep)
        dt (float): Timestep used in final smooth trajectory
    Outputs:
        t_smoothed (np.array [N]): Associated trajectory times
        traj_smoothed (np.array [N,7]): Smoothed trajectory
    Hint: Use splrep and splev from scipy.interpolate
    """
    ########## Code starts here ##########
    assert(path and k > 2 and k < len(path))
    ########## Code starts here ##########
    # Hint 1 - Determine nominal time for each point in the path using V_des
    # Hint 2 - Use splrep to determine cubic coefficients that best fit given path in x, y
    # Hint 3 - Use splev to determine smoothed paths. The "der" argument may be useful.
    time = np.zeros(len(path))
    #x = np.zeros(len(path))
    #y = np.zeros(len(path))
    time[0] = 0
    ##x[0] = path[0][0]
    #y[0] = path[0][1]
    path = np.array(path) #Converting tuple to np.array
    for i in range(1,len(path)):
        time[i] = (np.linalg.norm(path[i,:]-path[i-1,:])/V_des) + time[i-1] #calculating time
        #x[i] = path[i][0]
        #y[i] = path[i][1]
    #print(time)
    spl_x = intp.splrep(time,path[:,0],k=k,s=alpha) #Splining
    spl_y = intp.splrep(time,path[:,1],k=k,s=alpha) #Splining
    #spl_x = intp.splrep(time,x,k=k,s=alpha)
    #spl_y = intp.splrep(time,y,k=k,s=alpha)
    #spl_th = intp.splrep(time,path[:,2],k=3)
    t_smoothed = np.arange(time[0],time[-1],dt) #Returns evenly spaced values for given dt
    x_d = intp.splev(t_smoothed,spl_x) #Smoothed x
    y_d = intp.splev(t_smoothed,spl_y) #Smoothed y
    #theta_d = intp.splev(time,spl_th)
    xd_d = intp.splev(t_smoothed,spl_x,der=1) #Smoothed veloity x
    yd_d = intp.splev(t_smoothed,spl_y,der=1) #Smoothed velocity y
    theta_d = np.arctan2(yd_d,xd_d) #Calculating theta
    xdd_d = intp.splev(t_smoothed,spl_x,der=2) #Smoothed x acceleration
    ydd_d = intp.splev(t_smoothed,spl_y,der=2) #Smoothed y acceleration
    #t_smoothed = time.copy()

    ########## Code ends here ##########
    traj_smoothed = np.stack([x_d, y_d, theta_d, xd_d, yd_d, xdd_d, ydd_d]).transpose()
    ########## Code ends here ##########

    return t_smoothed, traj_smoothed
