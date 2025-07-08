import numpy as np
import copy
from ray_tracer import trace_ray_with_derivatives

# --------------------------------------------------------------------------
# ===== 1.  forward modelling for a distance-dependent velocity model  =====
# --------------------------------------------------------------------------
import numpy as np

def compute_travel_time_radial(event, station, velocity_model):
    """
    Straight-line travel time through concentric distance shells.
    
    Parameters
    ----------
    event, station : dict
        {'x','y','z'}   – coordinates in km (z positive downward)
    velocity_model : list[dict]
        [{'start':0.0,'end':1.0,'velocity':4.0}, …]  (distances in km)
        The last layer *must* have ``end=np.inf`` (or a very large number).
    Returns
    -------
    tt        : float               – travel time (s)
    segments  : list[float] len=n   – path length in each shell (km)
    """
    # straight-line source-receiver distance
    dx = station['x'] - event['x']
    dy = station['y'] - event['y']
    dz = station['z'] - event['z']
    r  = np.sqrt(dx*dx + dy*dy + dz*dz)
    
    tt, segments = 0.0, []
    travelled = 0.0
    for lay in velocity_model:
        d0, d1, v = lay['start'], lay['end'], lay['velocity']
        # overlap of [travelled, r] with [d0, d1]
        seg = max(0.0, min(r, d1) - max(travelled, d0))
        segments.append(seg)
        tt += seg / v
        travelled += seg
        if travelled >= r:                      # finished
            segments.extend([0.0]*(len(velocity_model)-len(segments)))
            break
    return tt, segments


def tt_with_derivatives(event, station, velocity_model, eps=1e-3):
    """
    Finite-difference ∂t/∂x,y,z for the radial model.
    """
    base, seg = compute_travel_time_radial(event, station, velocity_model)
    shift = lambda comp: {**event, comp: event[comp] + eps}
    
    dtdx = (compute_travel_time_radial(shift('x'), station, velocity_model)[0] - base)/eps
    dtdy = (compute_travel_time_radial(shift('y'), station, velocity_model)[0] - base)/eps
    dtdz = (compute_travel_time_radial(shift('z'), station, velocity_model)[0] - base)/eps
    return base, seg, dtdx, dtdy, dtdz


# --------------------------------------------------------------------------
# ===== 2.  locate a single event in a fixed radial velocity model  ========
# --------------------------------------------------------------------------
def invert_event_radial(event, stations, picks, velocity_model, *,
                        max_iter=10, tol=1e-3,
                        damp_xyz=1e-3, damp_t0=1e-3,
                        max_dxyz=0.01, max_dt0=0.01,
                        t0_min=-np.inf):
    """
    Analogue of ``invert_event`` but for a distance-dependent model.
    
    velocity_model : see `compute_travel_time_radial`
    """
    evt       = event.copy()
    nsta      = len(stations)
    ntot      = 4                      # x, y, z, t0
    rms_hist  = []
    
    for it in range(max_iter):
        r, J = [], []
        for sta, t_obs in zip(stations, picks):
            t_cal, _, dtdx, dtdy, dtdz = tt_with_derivatives(evt, sta, velocity_model)
            residual = t_obs - (t_cal + evt['t0'])
            r.append(residual)
            J.append([dtdx, dtdy, dtdz, 1.0])
        
        r = np.asarray(r)
        J = np.asarray(J)
        
        # damped least squares
        D     = np.diag([damp_xyz, damp_xyz, damp_xyz, damp_t0])
        delta = np.linalg.solve(J.T @ J + D, J.T @ r)
        
        # step-length control
        delta[:3] = np.clip(delta[:3], -max_dxyz, max_dxyz)
        delta[ 3] = np.clip(delta[ 3], -max_dt0, max_dt0)
        
        # update model
        evt['x']  += delta[0]
        evt['y']  += delta[1]
        evt['z']  += delta[2]
        evt['t0']  = max(t0_min, evt['t0'] + delta[3])
        
        rms_hist.append(float(np.sqrt(np.mean(r**2))))
        if np.linalg.norm(delta) < tol:
            break
    
    return evt, rms_hist


# --------------------------------------------------------------------------
# ===== 3.  joint inversion: hypocentre + radial velocity shells  ==========
# --------------------------------------------------------------------------
def invert_event_and_velocity_radial(event, stations, picks, velocity_model, *,
                                     max_iter=10, tol=1e-4,
                                     damp_xyz=1e-3, damp_t0=1e-3,
                                     damp_slow=1e-1,                 # slowness damping
                                     max_dxyz=2.0, max_dv_rel=0.05,
                                     typeB_every=2, t0_min=0.0):
    """
    Jointly solve for (x,y,z,t0) **and** slownesses s_i = 1/v_i in the distance shells.
    
    The update and damping logic mirrors the original VELEST-style routine.
    """
    evt    = event.copy()
    model  = [lay.copy() for lay in velocity_model]
    slo    = 1.0 / np.array([lay['velocity'] for lay in model])   # slowness vector
    nlay   = len(model)
    ntot   = 4 + nlay
    rms_hist = []
    
    for it in range(max_iter):
        r, J = [], []
        for sta, t_obs in zip(stations, picks):
            t_cal, seg, dtdx, dtdy, dtdz = tt_with_derivatives(evt, sta, model)
            r.append(t_obs - (t_cal + evt['t0']))
            
            row       = np.zeros(ntot)
            row[:3]   = [dtdx, dtdy, dtdz]
            row[3]    = 1.0                       # ∂t/∂t0
            row[4:]   = seg                      # ∂t/∂s_i = segment length
            J.append(row)
        
        r = np.asarray(r)
        J = np.asarray(J)
        
        # regularisation (type-A vs type-B)
        D = np.diag([damp_xyz, damp_xyz, damp_xyz, damp_t0] +
                    [damp_slow]*nlay)
        if typeB_every and (it % typeB_every):
            D[4:, 4:] += 1e5                     # freeze velocities on type-B steps
        
        delta = np.linalg.solve(J.T @ J + D, J.T @ r)
        
        # clip location step
        delta[:3] = np.clip(delta[:3], -max_dxyz, max_dxyz)
        # clip velocity step (relative)
        rel       = delta[4:] / slo
        delta[4:] = np.clip(rel, -max_dv_rel, max_dv_rel) * slo
        
        # apply update
        evt['x']  += delta[0]
        evt['y']  += delta[1]
        evt['z']  += delta[2]
        evt['t0']  = max(t0_min, evt['t0'] + delta[3])
        
        slo += delta[4:]
        if np.any(slo <= 0.0):
            raise RuntimeError("Slowness became non-positive – increase damping.")
        for j, lay in enumerate(model):
            lay['velocity'] = 1.0 / slo[j]
        
        rms_hist.append(float(np.sqrt(np.mean(r**2))))
        if np.linalg.norm(delta) < tol:
            break
    
    return evt, model, rms_hist
