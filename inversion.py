import numpy as np
import copy
from ray_tracer import trace_ray_with_derivatives

def invert_event(
        event,
        stations,
        picks,
        velocity_model,
        *,
        max_iter: int = 10,
        tol: float = 0.001,
        damp_xyz: float = 1e-3,   # ridge damping on x,y,z  (km)
        damp_t0:  float = 1e-3,   # ridge damping on origin time (s)
        max_dxyz: float = 0.01,    # hard cap on xyz step per iter (km)
        max_dt0:  float = 0.01,    # hard cap on t0 step per iter (s)
        differential: bool = False,
        t0_min: float = -np.inf    # allow negative origin times unless you set a floor
):
    """
    Locate a single earthquake in a *fixed* 1-D velocity model.

    Parameters
    ----------
    event : dict  – initial guess {'x','y','z','t0'} in km and s
    stations : list[dict] – each {'x','y','z'} in km
    picks : list[float] – P-phase arrival times (s) matching `stations`
    velocity_model : list[dict] – each {'top','bottom','velocity'} in km/s
    differential : bool
        If True, subtract the reference-station time so t0 drops out.
    Returns
    -------
    best_event : dict
        Hypocentre corresponding to the lowest RMS achieved, *even if
        formal convergence is not reached*.
    rms_history : list[float]
        RMS after each iteration.
    """

    evt   = copy.deepcopy(event)
    nsta  = len(stations)
    ntot  = 4                           # x, y, z, t0
    rms_hist = []

    # “best so far”
    best_evt = copy.deepcopy(evt)
    best_rms = np.inf
    best_it  = -1

    # choose first station as reference for differential times
    ref_idx = 0
    picks = np.asarray(picks)
    if differential:
        picks = picks - picks[ref_idx]

    for it in range(max_iter):
        r = []
        J = []

        for i, (sta, t_obs) in enumerate(zip(stations, picks)):
            t_cal, _, dtdx, dtdy, dtdz = trace_ray_with_derivatives(
                evt, sta, velocity_model
            )

            if differential:
                t_ref, _, *_ = trace_ray_with_derivatives(
                    evt, stations[ref_idx], velocity_model
                )
                t_cal -= t_ref

            residual = t_obs - (t_cal + (0 if differential else evt['t0']))
            r.append(residual)

            row = np.zeros(ntot)
            row[0:3] = np.array([dtdx, dtdy, dtdz])     # sign fixed
            row[3]   = 0.0 if differential else 1.0
            J.append(row)

        r = np.asarray(r)
        J = np.asarray(J)

        # ---------- damped least squares ---------------------------------
        D = np.diag([damp_xyz, damp_xyz, damp_xyz, damp_t0])
        delta = np.linalg.solve(J.T @ J + D, J.T @ r)

        # ---------- step-length control ----------------------------------
        delta[0:3] = np.clip(delta[0:3], -max_dxyz, max_dxyz)
        delta[3]   = np.clip(delta[3],  -max_dt0,  max_dt0)

        # ---------- apply update -----------------------------------------
        evt['x']  += delta[0]
        evt['y']  += delta[1]
        evt['z']  += delta[2]
        evt['t0'] = max(t0_min, evt['t0'] + delta[3])

        # ---------- diagnostics ------------------------------------------
        rms = np.sqrt(np.mean(r**2))
        rms_hist.append(rms)

        if rms < best_rms:                 # update “best so far”
            best_rms = rms
            best_evt = copy.deepcopy(evt)
            best_it  = it + 1

        print(f"Iter {it+1:02d}  RMS={rms:.4f}  |Δm|={np.linalg.norm(delta):.2e}")

        # convergence test
        if np.linalg.norm(delta) < tol:
            print("Converged.")
            return evt, rms_hist           # final model is already the best

    # --------------------------------------------------------------------
    # fell out of the loop without convergence → return best RMS model
    # --------------------------------------------------------------------
    print(f"\nNo convergence after {max_iter} iterations.  "
          f"Best RMS = {best_rms:.4f} reached at iter {best_it}.")
    return best_evt, rms_hist

def invert_event_and_velocity(
        event,
        stations,
        picks,
        velocity_model,
        max_iter: int = 10,
        tol: float = 1e-4,
        damp_xyz: float = 1e-3,
        damp_t0:  float = 1e-3,
        damp_slow: float = 1e-1,
        max_dxyz: float = 2.0,
        max_dv_rel: float = 0.05,
        typeB_every: int = 2,
        t0_min: float = 0.0          # << NEW: hard lower bound for origin-time
):
    """
    Joint inversion for a single earthquake's location and the velocity model.

    Parameters:
    - event: dict with initial 'x', 'y', 'z', 't0'
    - stations: list of dicts with 'x', 'y', 'z'
    - picks: list of observed arrival times, same order as stations
    - velocity_model: list of dicts with 'top', 'bottom', 'velocity'
    - max_iter: maximum number of iterations
    - damping: damping parameter for regularization
    - tol: threshold for convergence

Returns
    -------
    best_event  : dict   – model that achieved the lowest RMS
    best_model  : list   – corresponding velocity layers
    rms_history : list   – RMS after each iteration (useful for diagnostics)
    """

    # ---------- working copies ------------------------------------------
    evt   = copy.deepcopy(event)
    model = [lay.copy() for lay in velocity_model]
    slo   = 1.0 / np.array([lay['velocity'] for lay in model])  # slowness
    nlay  = len(model)
    ntot  = 4 + nlay

    top_z = model[0]['top']
    bot_z = model[-1]['bottom']

    # ---------- “best-so-far” trackers ----------------------------------
    best_evt   = copy.deepcopy(evt)
    best_model = [lay.copy() for lay in model]
    best_rms   = np.inf
    best_it    = -1
    rms_hist   = []

    picks = np.asarray(picks)

    for it in range(max_iter):
        # ---------- forward modelling + Jacobian ------------------------
        r, J = [], []
        for sta, t_obs in zip(stations, picks):
            t_cal, seg, dtdx, dtdy, dtdz = trace_ray_with_derivatives(
                evt, sta, model
            )
            r.append(t_obs - (t_cal + evt['t0']))

            row       = np.zeros(ntot)
            row[:3]   = np.array([dtdx, dtdy, dtdz])
            row[3]    = 1.0                    # ∂t/∂t0
            row[4:]   = np.asarray(seg)        # ∂t/∂s_i = path length
            J.append(row)

        r = np.asarray(r)
        J = np.asarray(J)

        # ---------- damping matrix --------------------------------------
        D = np.diag(
            [damp_xyz, damp_xyz, damp_xyz, damp_t0] +
            [damp_slow] * nlay
        )
        if typeB_every and (it % typeB_every):
            D[4:, 4:] += 1.0e5                 # freeze velocities (“type-B”)

        # ---------- solve for parameter update --------------------------
        delta = np.linalg.solve(J.T @ J + D, J.T @ r)

        # ---------- step-length control ---------------------------------
        delta[:3] = np.clip(delta[:3], -max_dxyz, max_dxyz)
        rel       = delta[4:] / slo
        delta[4:] = np.clip(rel, -max_dv_rel, max_dv_rel) * slo

        # ---------- apply update ----------------------------------------
        evt['x']  += delta[0]
        evt['y']  += delta[1]
        evt['z']  = np.clip(evt['z'] + delta[2], top_z, bot_z)

        evt['t0'] = max(t0_min, evt['t0'] + delta[3])

        slo += delta[4:]
        if np.any(slo <= 0.0):
            raise RuntimeError(
                "Slowness became non-positive – "
                "increase damp_slow or decrease max_dv_rel."
            )
        for j, lay in enumerate(model):
            lay['velocity'] = 1.0 / slo[j]

        # ---------- diagnostics & “best-so-far” update ------------------
        rms = np.sqrt(np.mean(r ** 2))
        rms_hist.append(rms)

        if rms < best_rms:
            best_rms   = rms
            best_it    = it + 1
            best_evt   = copy.deepcopy(evt)
            best_model = [lay.copy() for lay in model]

        print(f"Iter {it+1:02d}  RMS={rms:.4f}  |Δm|={np.linalg.norm(delta):.2e}")

        if np.linalg.norm(delta) < tol:
            print("Converged.")
            return evt, model, rms_hist    # final model is already the best

    # ---------- no convergence -----------------------------------------
    print(f"\nNo convergence after {max_iter} iterations.  "
          f"Best RMS = {best_rms:.4f} reached at iter {best_it}.")
    return best_evt, best_model, rms_hist
