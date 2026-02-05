from .base  import *
from .utils import *


class MACSPOProb:
    def __init__(self, params):
        self.load_params(params)

        self.list_waypoints = None
        self.list_vinit     = None

        self.q          = None
        self.Si         = None
        self.Sd         = None
        self.d          = None
        self.A          = None
        self.M          = None
        self.L          = None
        self.R          = None
        self.Rinv      = None

        self.K          = None
        self.Ni         = None
        self.Ni0        = None
        self.N          = None
        self.Np         = None
        self.id_pairs   = None

        self.t          = None
        self.x          = None
        self.z          = None
        self.s          = None
        self.u          = None
        self.w          = None
        self.f          = None

        self.res_prim  = None
        self.res_dual  = None
        self.sum_time  = None


    def load_params(self, params):
        self.vmin       = params["constraints"      ]["speed_min"               ]
        self.vmax       = params["constraints"      ]["speed_max"               ]
        self.dsafe      = params["constraints"      ]["safety_distance"         ]
        self.rho        = params["solver"           ]["step_size"               ]
        self.Nt         = params["solver"           ]["num_discrete_time_steps" ]
        ctrl_tau        = params["flight_controller"]["velocity_controller_tau" ]

        self.bet        = 1.5*(1/ctrl_tau)
        self.tbias      = 1.68/(1/ctrl_tau)
        self.thor       = self.tbias + ss.softplus(-self.bet*self.tbias)/self.bet



    def update_prob(self, list_waypoints, list_term_time, list_task_time):
        self.list_waypoints = list_waypoints

        self.K   = len(self.list_waypoints)
        self.Ni  = [ wps.shape[0] for wps in self.list_waypoints ]
        self.Ni0 = [ 0, *np.cumsum(self.Ni) ]
        self.N   = sum(self.Ni)
        self.Np  = self.K*(self.K-1) // 2

        _i_idx, _j_idx = np.triu_indices(self.K, k=1)
        self.id_pairs  = np.vstack([_i_idx, _j_idx]).astype(np.int8)

        self.list_task_time = [0]*self.K
        _list_eN   = [0]*self.K
        _list_e0   = [0]*self.K
        _list_diff = [0]*self.K
        _list_dist = [0]*self.K
        _list_tmin = [0]*self.K
        _list_tf   = []
        _list_ef   = []

        for i in range(self.K):
            n0  = self.Ni0[i]
            n   = self.Ni[i]
            wps = self.list_waypoints[i]

            _list_diff[i] = -np.eye(n-1, n) + np.eye(n-1, n, k=1)
            _list_dist[i] = nl.norm(np.diff(wps, axis=0), axis=1)
            _list_tmin[i] = np.sum(_list_dist[i]) / self.vmax
            _list_e0[i]   = onehot_vector(0,  n).T
            _list_eN[i]   = onehot_vector(-1, n) / _list_tmin[i] / self.K

            self.list_task_time[i] = np.cumsum(list_task_time[i])

            if list_term_time[i] is None: continue
            _list_ef.append(onehot_vector(n0+n-1, self.N).reshape(1,-1))
            _list_tf.append([list_term_time[i] - self.list_task_time[i][-1]])

        self.d  = np.concatenate(_list_dist).reshape(-1, 1)
        self.q  = np.concatenate(_list_eN  )
        self.Si = sl.block_diag(*_list_e0  )
        self.Sd = sl.block_diag(*_list_diff)

        self._scaler_diff = 1/(np.mean(self.d) / self.vmax)
        self._scaler_time = 1/(np.min(_list_tmin))

        self.dt = np.min(_list_tmin) / self.Nt

        if len(_list_tf) != 0:
            self.tf = np.concatenate(_list_tf  ).reshape(-1, 1)
            self.Sf = np.concatenate(_list_ef  )
            self.A = np.vstack([self.Si, self.Sd, np.eye(self.N), self.Sf])
            self.b = np.vstack([np.zeros((2*self.N,1)), self.tf])

            Ntf = self.tf.shape[0]

        else:
            self.A = np.vstack([self.Si, self.Sd, np.eye(self.N)])
            self.b = np.vstack([np.zeros((2*self.N,1))])

            Ntf = 0

        self.M = np.zeros((2*self.N + Ntf,      self.N - self.K ))
        self.L = np.zeros((2*self.N + Ntf,      self.N          ))

        self.M[self.K:self.N,  :] = np.eye(self.N - self.K)
        self.L[self.N:2*self.N,:] = np.eye(self.N)

        self.R = self.A.T @ self.A
        self.Rinv = nl.inv(self.R)

        self.t   = np.zeros((self.N,        1), dtype=np.float32)
        self.x   = np.zeros((self.N-self.K, 1), dtype=np.float32)
        self.z   = np.zeros((self.N,        1), dtype=np.float32)
        self.u   = np.zeros((2*self.N + Ntf,1), dtype=np.float32)
        self.f   = 0
        self.res_prim  = 0
        self.res_dual  = 0
        self.sum_time  = 0

        self.q      = self.q.astype     (np.float32)
        self.Si     = self.Si.astype    (np.float32)
        self.Sd     = self.Sd.astype    (np.float32)
        self.A      = self.A.astype     (np.float32)
        self.b      = self.b.astype     (np.float32)
        self.M      = self.M.astype     (np.float32)
        self.L      = self.L.astype     (np.float32)
        self.d      = self.d.astype     (np.float32)
        self.R      = self.R.astype     (np.float32)
        self.Rinv  = self.Rinv.astype (np.float32)


    def initialize(self, init_cond):
        tinit       = init_cond["optimization_variable"]
        list_vinit  = init_cond["initial_velocities"   ]

        if tinit is not None:
            self.t = tinit.copy()

        else:
            for i in range(self.K):
                Ni0 = self.Ni0[i]
                Ni  = self.Ni[i]
                self.t[Ni0 + 1:Ni0 + Ni] = np.cumsum(self.d[Ni0 - i :Ni0 + Ni - i - 1], axis=0) / self.vmax

        if list_vinit is not None:
            self.list_vinit = list_vinit.copy()
        else:
            self.list_vinit = [ np.zeros((1,2), dtype=np.float32) for _ in range(self.K) ]

        self.x = self.Sd @ self.t
        self.z = self.t.copy()
