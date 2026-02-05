import numpy as np
import numpy.linalg as nl
import numpy.random as nr
import time
from macspo import MACSPOProb, MACSPOSolver


"""================== Waypoint generation for case 1 =================="""

### waypoints ###
wp1  = 10* np.array([[-10 * 1.5 ,10 * 1.75]])
wp2  = 10* np.array([[-10 * 1.5 ,8 * 1.75]])
wp3  = 10* np.array([[-10 * 1.5 ,6 * 1.75]])
wp4  = 10* np.array([[-10 * 1.5 ,4 * 1.75]])
wp5  = 10* np.array([[-10 * 1.5 ,2 * 1.75]])
wp6  = 10* np.array([[-10 * 1.5 ,0 * 1.75]])
wp7  = 10* np.array([[-10 * 1.5 ,-2 * 1.75]])
wp8  = 10* np.array([[-10 * 1.5 ,-4 * 1.75]])
wp9  = 10* np.array([[-10 * 1.5 ,-6 * 1.75]])
wp10 = 10* np.array([[-10 * 1.5 ,-8 * 1.75]])
wp11 = 10* np.array([[-10 * 1.5 ,-10 * 1.75]])

### destination ###
des1  = 10* np.array([[10 * 1.75 ,-10 * 1.75]])
des2  = 10* np.array([[10 * 1.75 ,-8 * 1.75]])
des3  = 10* np.array([[10 * 1.75 ,-6 * 1.75]])
des4  = 10* np.array([[10 * 1.75 ,-4 * 1.75]])
des5  = 10* np.array([[10 * 1.75 ,-2 * 1.75]])
des6  = 10* np.array([[10 * 1.75 ,0 * 1.75]])
des7  = 10* np.array([[10 * 1.75 ,2 * 1.75]])
des8  = 10* np.array([[10 * 1.75 ,4 * 1.75]])
des9  = 10* np.array([[10 * 1.75 ,6 * 1.75]])
des10 = 10* np.array([[10 * 1.75 ,8 * 1.75]])
des11 = 10* np.array([[10 * 1.75 ,10 * 1.75]])

wps_list = [wp1,wp2,wp3,wp4,wp5,wp6,wp7,wp8,wp9,wp10,wp11]
dss = [des1,des2,des3,des4,des5,des6,des7,des8,des9,des10,des11]

nr.seed(3)

for i in range(len(wps_list)):

    wp = wps_list[i]
    des = dss[i]

    # direction = nr.rand()*np.pi/3 - np.pi/6
    direction = np.arctan2(des[0,1]-wp[0,1],des[0,0]-wp[0,0])

    num = 5 + (abs(len(wps_list) - 2*i)//3)

    # random delta alphas of length num
    delta_alpha = nr.rand(num)*np.pi/6 - np.pi/12

    alpha = direction + delta_alpha

    # random segment lengths of length num whose sum is 10
    dist = nl.norm(des - wp)
    l = 0.3*nr.rand(num) + dist/num - 0.5

    # random waypoints
    for n in range(num-1): wp = np.append(wp,wp[-1,:] + l[n]*np.array([np.cos(alpha[n]),np.sin(alpha[n])]).reshape(1,2),axis=0)

    wp = np.append(wp,des,axis=0)

    wps_list[i] = wp/9


"""================== Solve MACSPO problem =================="""

# number of agents
K = len(wps_list) 

# parameters for MACSPO
params = {
"constraints":
  {
  "speed_min"                 : 0.1,
  "speed_max"                 : 3.0,
  "safety_distance"           : 1.8
  },
"solver":
  {
  "step_size"                 : 1e2,
  "smoothing_param"           : 0.0,
  "max_iterations"            : 500,
  "primal_residual_criterion" : 1e-3,
  "penalty_update_criterion"  : 1e-4,
  "num_discrete_time_steps"   : 100
  },
"flight_controller":
  {
  "velocity_controller_tau"   : 0.1,
  "control_loop_dt"           : 0.1
  }
}

# initial conditions
init_cond = {
  "optimization_variable" : None,
  "initial_velocities"    : None
}

# terminal times and task times
tf_list = [None for _ in range(K)]
tt_list = [np.full(wps_list[i].shape[0]-1, 0.0) for i in range(K)]

st = time.time()

prob   = MACSPOProb(params=params)
solver = MACSPOSolver(params)

prob.update_prob(list_waypoints=wps_list,
                 list_term_time=tf_list,
                 list_task_time=tt_list)

solver.solve(prob=prob,
             init_cond=init_cond)

tspace, list_tvar, list_vtraj, list_vcmd = solver.get_vtraj(prob=prob)
tspace, list_ptraj = solver.get_ptraj(prob=prob)

et = time.time()