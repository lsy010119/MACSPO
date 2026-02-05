from .base import *


def onehot_vector(i, N):
    e    = np.zeros((N,1))
    e[i] = 1.0
    return e


def calc_vel_traj(wps, tvar, tspace, vinit, ttask, param_bet, param_tbias, param_thor):
    Nt, N = tspace.shape[0], tvar.shape[0]
    _tvar = tvar.reshape(-1)

    pbias = vinit * param_thor

    refined_wps = wps.copy()
    refined_wps[1:] -= pbias

    dp   = np.diff(refined_wps, axis=0)
    _dt  = np.diff(_tvar)
    dt   = np.clip(np.abs(_dt), 1e-12, np.inf) * np.sign(_dt + 1e-12)
    vcmd = dp / dt[:, None]

    shifted_tspace = tspace[:, None] - param_tbias

    xf    = param_bet * (shifted_tspace - _tvar[None, 1: ] - ttask)
    xb    = param_bet * (shifted_tspace - _tvar[None, :-1] - ttask)
    sigxf = ss.expit(xf)
    sigxb = ss.expit(xb)

    vtraj = np.sum(vcmd[None, :, :] * (sigxf[:, :, None] - sigxb[:, :, None]), axis=1) \
            + vinit*(1 - ss.expit(param_bet*(shifted_tspace - _tvar[0])))

    vcmd  = np.concatenate((vcmd, np.zeros((1,2))))

    return vtraj, vcmd


def calc_pos_traj(wps, tvar, tspace, vinit, ttask, param_bet, param_tbias, param_thor, calc_grad=False):
    Nt, N = tspace.shape[0], tvar.shape[0]
    _tvar = tvar.reshape(-1)

    pbias = vinit * param_thor

    refined_wps = wps.copy()
    refined_wps[1:] -= pbias

    dp   = np.diff(refined_wps, axis=0)
    _dt  = np.diff(_tvar)
    dt   = np.clip(np.abs(_dt), 1e-8, np.inf) * np.sign(_dt + 1e-8)
    vcmd = dp / dt[:, None]

    shifted_tspace = tspace[:, None] - param_tbias

    xf    = param_bet * (shifted_tspace - _tvar[None, :-1] - ttask)
    xb    = param_bet * (shifted_tspace - _tvar[None, 1: ] - ttask)
    spxf  = ss.softplus(xf)
    spxb  = ss.softplus(xb)
    sigxf = ss.expit(xf)
    sigxb = ss.expit(xb)
    W     = (spxf - spxb)/ param_bet

    ptraj = refined_wps[0] + pbias \
            + vinit*(shifted_tspace - ss.softplus(param_bet*(shifted_tspace - _tvar[0])) / param_bet) \
            + np.sum(W[:, :, None] * vcmd[None, :, :], axis=1)  # (Nt,2)

    if not calc_grad:
        return ptraj

    else:
        grad = np.zeros((Nt, N, 2), dtype=np.float32)

        inv_dt2     = 1.0 / (dt ** 2)
        dp_over_dt2 = dp * inv_dt2[:,None]
        W_dp_over_dt2 = W[:, :, None] * dp_over_dt2[None, :, :]

        if N >= 2:

            buff = vcmd[None, :, :] * sigxf[:, :, None]
            grad[:, :-1, :] -= buff
            grad[:, :-1, :] += W_dp_over_dt2

            buff = vcmd[None, :, :] * sigxb[:, :, None]
            grad[:, 1:, :] += buff
            grad[:, 1:, :] -= W_dp_over_dt2

        grad[:, 0, :] += vinit * sigxb[:, 0, None]

        return ptraj, grad



def calc_penalty(z, tspace, param_bet, param_tbias, param_thor, param_dsafe, prob):
    list_wps   = prob.list_waypoints
    list_v0    = prob.list_vinit
    list_ttask = prob.list_task_time
    idx_pairs  = prob.id_pairs
    K          = prob.K
    N          = prob.N
    Ni         = prob.Ni
    Ni0        = prob.Ni0
    Np         = prob.Np
    Nt         = tspace.shape[0]
    dt         = tspace[1] - tspace[0]

    grad      = np.zeros(N,             dtype=np.float32)
    traj_mtx  = np.zeros((K, Nt, 2),    dtype=np.float32)
    traj_grad = np.zeros((K, Nt, N, 2), dtype=np.float32)

    for i in range(K):
        ti  = z[Ni0[i]: Ni0[i] + Ni[i]]
        vi0 = list_v0[i]
        ttask = list_ttask[i]

        pi, dpi_dti = calc_pos_traj(
            wps         = list_wps[i],
            tvar        = ti,
            tspace      = tspace,
            vinit       = vi0,
            ttask       = ttask,
            param_bet   = param_bet,
            param_tbias = param_tbias,
            param_thor  = param_thor,
            calc_grad   = True)

        traj_mtx[i] = pi
        traj_grad[i, :, Ni0[i]:Ni0[i]+Ni[i], :] = dpi_dti

    diff = traj_mtx[idx_pairs[0]] - traj_mtx[idx_pairs[1]]
    dist = nl.norm(diff, axis=2)

    mask = dist < param_dsafe

    if np.sum(mask) == 0:
        return 0, grad

    f = np.sum(np.clip(1 - dist/param_dsafe, 0, np.inf))

    for p_idx, tau_idx in zip(*np.where(mask)):

        i_idx = idx_pairs[0][p_idx]
        j_idx = idx_pairs[1][p_idx]

        pi_tau = traj_mtx[i_idx, tau_idx]
        pj_tau = traj_mtx[j_idx, tau_idx]

        diff_vec = pi_tau - pj_tau
        dist_ij = dist[p_idx, tau_idx]

        unit_vec = (diff_vec / (dist_ij + 1e-8))

        dpi_dt_all = traj_grad[i_idx, tau_idx]  # (N, 2)
        dpj_dt_all = traj_grad[j_idx, tau_idx]  # (N, 2)

        grad_contrib = (dpi_dt_all - dpj_dt_all) @ unit_vec / param_dsafe # (N,)
        grad -= grad_contrib.reshape(-1)

    return f*dt/Np*prob._scaler_time, grad*dt/Np*prob._scaler_time