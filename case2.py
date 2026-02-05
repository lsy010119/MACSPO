import numpy as np
import numpy.linalg as nl
import numpy.random as nr
import time
from macspo import MACSPOProb, MACSPOSolver


"""================== Waypoint generation for case 1 =================="""

def generate_wps_bottleneck(K):
    wps_list = [0]*K

    ymax = np.linspace(-1,1,K)

    for i in range(K):
        n = 2
        d = 3

        x = np.linspace(0,1,6)**1.5 * (10-d)
        y = ymax[i]*(1-(1-x/(10-d))**n)**(1/n)

        wps = np.block([[-x[::-1].reshape(-1,1)-d,y[::-1].reshape(-1,1)*10],
                        [np.array([[-1,0],
                                    [0,0],
                                    [1,0]])],
                        [x.reshape(-1,1)+d,y.reshape(-1,1)*10]])

        wps_list[i] = wps

    return wps_list


"""================== Solve MACSPO problem =================="""

# number of agents
K = 11
wps_list = generate_wps_bottleneck(K)

# parameters for MACSPO
params = {
"constraints":
  {
  "speed_min"                 : 0.02,
  "speed_max"                 : 2.0,
  "safety_distance"           : 1.59
  },
"solver":
  {
  "step_size"                 : 1e2,
  "max_iterations"            : 1000,
  "primal_residual_criterion" : 1e-3,
  "penalty_update_criterion"  : 1e-5,
  "num_discrete_time_steps"   : 100
  },
"flight_controller":
  {
  "velocity_controller_tau"   : 0.1,
  "control_loop_dt"           : 0.1
  }
}

init_cond = {
  "optimization_variable" : None,
  "initial_velocities"    : None
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