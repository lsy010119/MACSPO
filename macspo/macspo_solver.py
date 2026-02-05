from .base        import *
from .utils       import *
from .macspo_prob import MACSPOProb


class MACSPOSolver:
    def __init__(self, params):
        self.load_params(params)

        self.buff_t = None
        self.buff_x = None
        self.buff_z = None
        self.buff_u = None
        
        self._buff_f = 0.
        self.alpha   = 0.

        self.solver_status = 0

    def load_params(self, params):
        self.rho        = params["solver"           ]["step_size"                 ]
        self.Niter      = params["solver"           ]["max_iterations"            ]
        self.crit_prim  = params["solver"           ]["primal_residual_criterion" ]
        self.crit_frate = params["solver"           ]["penalty_update_criterion"  ]
        self.Nt         = params["solver"           ]["num_discrete_time_steps"   ]
        self.ctrl_dt    = params["flight_controller"]["control_loop_dt"           ]


    def solve(self, prob: MACSPOProb, init_cond):
        prob.initialize(init_cond)

        for i in range(self.Niter):
            self.update_t(prob)
            self.update_x(prob)
            self.update_z(prob)
            self.update_u(prob)

            if i % 10 == 0:
                if prob.f > 0 and self._buff_f > 0:
                    if np.abs(prob.f - self._buff_f)/self._buff_f < 1e-1:
                        self.alpha = 0.7
                else:
                    self.alpha = 0.
                self._buff_f = prob.f

            if self.is_converged(prob):
                break

        print("Solving | " + ">" * int(40*i/self.Niter) + "-" * int(40*(self.Niter-i)/self.Niter) + f"| {i}/{self.Niter} | pr: {(prob.res_prim):.3e} | f: {prob.f:.3e} | tmax: {np.max(prob.t):.3e} | cost: {self.logger__sum_time[i]:.3e}")


    def get_vtraj(self, prob : MACSPOProb):
        list_wps    = prob.list_waypoints
        list_v0     = prob.list_vinit
        list_ttask  = prob.list_task_time
        topt        = prob.t
        param_bet   = prob.bet
        param_tbias = prob.tbias
        param_thor  = prob.thor

        list_vtraj = [0]*prob.K
        list_vcmd  = [0]*prob.K
        list_tvar  = [0]*prob.K

        mission_time = np.max([topt[prob.Ni0[i] + prob.Ni[i] - 1] + list_ttask[i][-1] for i in range(prob.K)])
        tspace = np.arange(0, mission_time + param_tbias*2, self.ctrl_dt, dtype=np.float32)

        for i in range(prob.K):
            wps    = list_wps[i]
            vinit  = list_v0[i]
            ttask  = list_ttask[i]

            tvar   = topt[ prob.Ni0[i] : prob.Ni0[i+1] ]

            vtraj, vcmd = calc_vel_traj(wps, tvar, tspace, vinit, ttask, param_bet, param_tbias, param_thor)
            list_vtraj[i] = vtraj
            list_vcmd[i]  = vcmd
            list_tvar[i]  = tvar

        return tspace, list_tvar, list_vtraj, list_vcmd


    def get_ptraj(self, prob : MACSPOProb):
        list_wps    = prob.list_waypoints
        list_v0     = prob.list_vinit
        list_ttask  = prob.list_task_time
        topt        = prob.t
        param_bet   = prob.bet
        param_tbias = prob.tbias
        param_thor  = prob.thor

        list_ptraj = [0]*prob.K

        mission_time = np.max([topt[prob.Ni0[i] + prob.Ni[i] - 1] + list_ttask[i][-1] for i in range(prob.K)])
        tspace = np.arange(0, mission_time + param_tbias*2, self.ctrl_dt, dtype=np.float32)

        for i in range(prob.K):
            wps    = list_wps[i]
            vinit  = list_v0[i]
            ttask  = list_ttask[i]

            tvar   = topt[ prob.Ni0[i] : prob.Ni0[i+1] ]
            ptraj = calc_pos_traj(wps, tvar, tspace, vinit, ttask, param_bet, param_tbias, param_thor)
            list_ptraj[i] = ptraj

        return tspace, list_ptraj


    def update_t(self, prob : MACSPOProb):
        self.buff_t = prob.t.copy()

        x           = prob.x
        z           = prob.z
        u           = prob.u
        q           = prob.q
        A           = prob.A
        b           = prob.b
        M           = prob.M
        L           = prob.L
        Rinv        = prob.Rinv

        r = A.T @ ( - M @ x - L @ z - b + u ) + (1/self.rho) * q
        t = - Rinv @ r

        prob.t = t


    def update_x(self, prob : MACSPOProb):
        self.buff_x = prob.x.copy()

        t           = prob.t
        u           = prob.u
        Sd          = prob.Sd
        d           = prob.d
        N           = prob.N
        K           = prob.K
        vmax        = prob.vmax
        vmin        = prob.vmin

        ux = u[K:N]
        x  = Sd @ t + ux

        under = x < d / vmax
        over  = x > d / vmin
        x[under] = d[under] / vmax
        x[over]  = d[over]  / vmin

        prob.x = x


    def update_z(self, prob : MACSPOProb):
        self.buff_z = prob.z.copy()

        t           = prob.t
        z           = prob.z
        u           = prob.u
        N           = prob.N
        rho         = prob.rho
        param_bet   = prob.bet
        param_tbias = prob.tbias
        param_thor  = prob.thor
        param_dsafe = prob.dsafe
        list_ttask  = prob.list_task_time

        if not hasattr(self, 'v_momentum'):
                self.v_momentum = np.zeros_like(prob.z)

        uz = u[N:2*N]

        z  = t + uz

        mission_time = np.max([z[prob.Ni0[i] + prob.Ni[i] - 1] + list_ttask[i][-1] for i in range(prob.K)])
        tspace = np.linspace(0, mission_time, self.Nt, dtype=np.float32)
        dt = tspace[1] - tspace[0]

        f, J = calc_penalty(z           = z,
                            tspace      = tspace,
                            param_bet   = param_bet,
                            param_tbias = param_tbias,
                            param_thor  = param_thor,
                            param_dsafe = param_dsafe,
                            prob        = prob)

        if not hasattr(self, 'f0'):
              self.f0 = f

        f, J = f/(self.f0 + 1e-8), J/(self.f0 + 1e-8)

        z_update = - f / (nl.norm(J)**2 + 1e-8) * J.reshape(-1, 1)
        self.v_momentum = self.alpha * self.v_momentum + z_update
        z += self.v_momentum

        prob.z = z
        prob.f = f


    def update_u(self, prob : MACSPOProb):
        self.buff_u = prob.u.copy()

        t           = prob.t
        x           = prob.x
        z           = prob.z
        u           = prob.u
        A           = prob.A
        b           = prob.b
        M           = prob.M
        L           = prob.L

        u = u + A @ t - M @ x - L @ z - b

        prob.u = u


    def is_converged(self, prob : MACSPOProb):
        t           = prob.t
        x           = prob.x
        z           = prob.z
        f           = prob.f
        A           = prob.A
        b           = prob.b
        M           = prob.M
        L           = prob.L

        prob.res_prim = nl.norm( A @ t - M @ x - L @ z - b )**2

        return prob.res_prim < self.crit_prim and f*self.f0 < self.crit_frate and (np.abs(prob.q.T@(t - self.buff_t))).item() <= 1e-2