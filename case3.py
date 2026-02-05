import numpy as np
import networkx as nx
from scipy.spatial import Delaunay
import time
from macspo import MACSPOProb, MACSPOSolver


"""================== Waypoint generation for case 1 =================="""

N = 60
r = 10
d = 1.5


def generate_nodes(N, r, d):
    samples = [0]*N
    counter = 0
    while counter < N:
        theta = nr.uniform(0, 2*np.pi)
        rad   = r * np.sqrt(nr.uniform(0,1))
        p = np.array([rad*np.cos(theta), rad*np.sin(theta)])
        if all(np.linalg.norm(p - q) >= d for q in samples):
            samples[counter] = p
            counter += 1
    nodes = np.array(samples)
    return nodes


def generate_uam_paths(nodes, num_agents, max_edge_len=7.0):

    N = nodes.shape[0]
    DG = nx.DiGraph()

    tri = Delaunay(nodes)
    temp_edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            u, v = sorted([simplex[i], simplex[(i+1)%3]])
            temp_edges.add((u, v))

    for u, v in temp_edges:
        dist = np.linalg.norm(nodes[u] - nodes[v])
        if dist <= max_edge_len:
            if nodes[u, 0] < nodes[v, 0]: DG.add_edge(u, v, weight=dist)
            else: DG.add_edge(v, u, weight=dist)

    center = np.mean(nodes, axis=0)
    dist_from_center = np.linalg.norm(nodes - center, axis=1)
    peripheral_indices = np.argsort(dist_from_center)[::-1]

    wps_list = []
    used_starts_ends = set()
    all_path_nodes_except_ends = set()

    potential_paths = []
    for s in peripheral_indices:
        try:
            descendants = list(nx.descendants(DG, s))
            for g in descendants:
                if s == g: continue
                path = nx.shortest_path(DG, s, g, weight='weight')
                if len(path) >= 4:
                    potential_paths.append(path)
        except: continue

    potential_paths.sort(key=len, reverse=True)

    for path in potential_paths:
        if len(wps_list) >= num_agents: break

        s, g = path[0], path[-1]

        if s in used_starts_ends or g in used_starts_ends: continue

        if g in all_path_nodes_except_ends: continue

        is_safe = True
        existing_ends = {tuple(p[-1]) for p in wps_list}
        for node in path[:-1]:
            if node in existing_ends:
                is_safe = False
                break
        if not is_safe: continue

        wps_list.append(nodes[path])
        used_starts_ends.add(s)
        used_starts_ends.add(g)
        for node in path[:-1]:
            all_path_nodes_except_ends.add(node)

    return wps_list, DG


"""================== Solve MACSPO problem =================="""

# number of agents
K = 11

# parameters for MACSPO
params = {
"constraints":
  {
  "speed_min"                 : 0.02,
  "speed_max"                 : 2.0,
  "safety_distance"           : 1.5
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

# initial conditions
init_cond = {
  "optimization_variable" : None,
  "initial_velocities"    : None
}

nr.seed(4)

nodes = generate_nodes(N, r, d)
wps_list, G = generate_uam_paths(nodes, K)

tf_list = [None for _ in range(K)]
tt_list = [np.full(wps_list[i].shape[0]-1, 1.0) for i in range(K)]
for i in range(K): tt_list[i][0] = 0

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