import numpy as np
from ray_tracer import *

def compute_ray_segments_all_rays(all_ray_paths, n_layers):
    """
    Converts ray tracer output to layer-wise segment lengths for inversion.

    Parameters:
        all_ray_paths: list of ray tracer outputs (dicts), one per ray/event
        n_layers: total number of velocity layers

    Returns:
        ray_segments: list of lists (one per ray), each of length n_layers
                      where ray_segments[i][j] is length of ray in layer j
    """
    ray_segments = []

    for ray in all_ray_paths:

        # -----------------------------------------------------------------
        # 1. Accept three possible formats:
        #    (a) Full dict from compute_travel_time_1d  → ray['segments']
        #    (b) Tuple returned by trace_ray            → ray[1] (segments)
        #    (c) Plain list/array that is already layer-wise lengths
        # -----------------------------------------------------------------
        if isinstance(ray, dict) and 'segments' in ray:
            segs = ray['segments']                       # list of (idx,dz,dx,dt)
            layer_lengths = np.zeros(n_layers)
            for layer_idx, dz, dx, _ in segs:
                layer_lengths[layer_idx] += np.hypot(dz, dx)
        elif isinstance(ray, (tuple, list)) and len(ray) == 3:
            # tuple from trace_ray: (t_calc, segments, p)
            layer_lengths = np.asarray(ray[1], float)    # already per-layer
        else:
            layer_lengths = np.asarray(ray, float)       # assume correct shape

        # enforce length n_layers
        if layer_lengths.size != n_layers:
            raise ValueError("Segment list has wrong length. "
                             "Expected {}, got {}."
                             .format(n_layers, layer_lengths.size))

        ray_segments.append(layer_lengths)

    return ray_segments  # shape: (n_rays, n_layers)


def calibrate_velocity_model(events, stations, picks, velocity_model, max_iter=10, damping=0.01, tol=1e-4):
    """
    Calibrate 1D velocity model using known explosion events.
    
    Parameters:
    - events: list of dicts, each with keys 'x', 'y', 'z', 't0' for explosion events
    - stations: list of dicts, each with keys 'x', 'y', 'z'
    - picks: dict keyed by (event_idx, station_idx), value is observed travel time
    - velocity_model: list of dicts with 'top', 'bottom', 'velocity'
    - max_iter: maximum inversion iterations
    - damping: damping factor for stability
    - tol: convergence threshold on RMS misfit
    
    Returns:
    - updated velocity_model (new velocities)
    """

    # --------------------------------------------------------------------
    # Working arrays
    # --------------------------------------------------------------------
    slowness = np.array([1.0 / layer['velocity'] for layer in velocity_model],
                        dtype=float)          # s / km
    n_layers = len(velocity_model)

    for iteration in range(max_iter):

        ray_segments_all = []     # ∑_ray  L_ij
        residuals_all    = []     # ∑_ray  (t_obs - t_calc)

        for ev_idx, event in enumerate(events):
            for st_idx, station in enumerate(stations):

                key = (ev_idx, st_idx)
                if key not in picks:
                    continue                       # missing pick → skip

                t_obs = picks[key]

                # -------- trace the ray with CURRENT model --------------
                t_calc, seg_lengths, _ = trace_ray(event, station,
                                                   velocity_model)

                ray_segments_all.append(seg_lengths)                # L_i•
                residuals_all.append(t_obs - (t_calc + event['t0']))

        if not ray_segments_all:
            raise RuntimeError("No rays found – check picks dictionary.")

        # -------- assemble normal equations -----------------------------
        L = np.vstack(ray_segments_all)           # shape (n_rays, n_layers)
        r = np.asarray(residuals_all)             # shape (n_rays,)

        A = L.T @ L + damping * np.eye(n_layers)
        b = L.T @ r

        delta_s = np.linalg.solve(A, b)           # slowness perturbation

        # -------- update model ------------------------------------------
        slowness += delta_s
        if np.any(slowness <= 0.0):
            raise RuntimeError("Negative slowness encountered. "
                               "Reduce damping or iteration step.")

        velocity = 1.0 / slowness                 # km / s
        for i in range(n_layers):
            velocity_model[i]['velocity'] = velocity[i]

        # -------- convergence check & report ----------------------------
        rms = np.sqrt(np.mean(r**2))
        print(f"Iter {iteration+1:02d}: RMS = {rms:.6f}  "
              f"|Δs| = {np.linalg.norm(delta_s):.2e}")

        if np.linalg.norm(delta_s) < tol:
            print("Converged.")
            break

    return velocity_model


