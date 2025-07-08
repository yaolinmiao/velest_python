import numpy as np
from scipy.optimize import minimize_scalar

def find_layer(depth, layer_tops, layer_bottoms):
    for i, (top, bottom) in enumerate(zip(layer_tops, layer_bottoms)):
        if top <= depth < bottom:
            return i
    raise ValueError(f"Depth {depth} km is outside model bounds.")

def compute_ray_path(p, source_depth, receiver_depth, velocity_model):
    """
    Compute ray path segments, travel time, and horizontal distance for ray parameter p.
    """
    layer_tops = np.array([layer['top'] for layer in velocity_model])
    layer_bottoms = np.array([layer['bottom'] for layer in velocity_model])
    velocity = np.array([layer['velocity'] for layer in velocity_model])

    source_idx = find_layer(source_depth, layer_tops, layer_bottoms)
    receiver_idx = find_layer(receiver_depth, layer_tops, layer_bottoms)

    total_time = 0.0
    total_offset = 0.0
    segments = []

    direction = 1 if receiver_depth > source_depth else -1
    idx_range = range(source_idx, receiver_idx + direction, direction)

    for i in idx_range:
        v = velocity[i]
        sin_theta = p * v
#         if sin_theta > 1:
#             raise ValueError("Total internal reflection")
        if sin_theta >= 1.0:
            raise ValueError("Total internal reflection")

        cos_theta = np.sqrt(1 - sin_theta**2)
        tan_theta = sin_theta / cos_theta

        top = layer_tops[i]
        bottom = layer_bottoms[i]

        # Vertical extent in this layer
        if i == source_idx:
            z_start = source_depth
        else:
            z_start = top if direction > 0 else bottom

        if i == receiver_idx:
            z_end = receiver_depth
        else:
            z_end = bottom if direction > 0 else top

        dz = abs(z_end - z_start)
        dx = dz * tan_theta
        dt = dz / (v * cos_theta)

        total_time += dt
        total_offset += dx
        segments.append((i, dz, dx, dt))

    return total_time, total_offset, segments

def compute_travel_time_1d(velocity_model, source_depth, receiver_depth, horizontal_dist, p_tol=1e-4):
    """
    Compute travel time for given source and receiver using ray parameter shooting.
    """
    layer_tops = np.array([layer['top'] for layer in velocity_model])
    layer_bottoms = np.array([layer['bottom'] for layer in velocity_model])
    velocity = np.array([layer['velocity'] for layer in velocity_model])

    def travel_time_objective(p):
        try:
            _, dx, _ = compute_ray_path(p, source_depth, receiver_depth, velocity_model)
            return abs(dx - horizontal_dist)
        except ValueError:
            return np.inf

    v_min = np.min(velocity)
    p_max = 1.0 / v_min - 1e-6

    res = minimize_scalar(travel_time_objective, bounds=(0, p_max), method='bounded', options={'xatol': p_tol})
    if not res.success:
        raise RuntimeError("Failed to optimize ray parameter")

    best_p = res.x
    tt, _, segments = compute_ray_path(best_p, source_depth, receiver_depth, velocity_model)

    return {
        "travel_time": tt,
        "ray_parameter": best_p,
        "segments": segments
    }

def trace_ray(event, station, velocity_model):
    dx = station['x'] - event['x']
    dy = station['y'] - event['y']
    dz = station['z'] - event['z']
    horizontal_dist = np.sqrt(dx**2 + dy**2)

    result = compute_travel_time_1d(
        velocity_model,
        source_depth=event['z'],
        receiver_depth=station['z'],
        horizontal_dist=horizontal_dist
    )

    n_layers = len(velocity_model)
    segments = [0.0] * n_layers
    for i, dz, dx, _ in result["segments"]:
        segments[i] += np.sqrt(dz**2 + dx**2)

    return result["travel_time"], segments, result["ray_parameter"]


def trace_ray_with_derivatives(event, station, velocity_model, eps=1e-3):
    """
    Compute finite-difference derivatives of travel time w.r.t. event x, y, z.
    """
    base_t, segments, _ = trace_ray(event, station, velocity_model)

    dtdx = (trace_ray({**event, 'x': event['x'] + eps}, station, velocity_model)[0] - base_t) / eps
    dtdy = (trace_ray({**event, 'y': event['y'] + eps}, station, velocity_model)[0] - base_t) / eps
    dtdz = (trace_ray({**event, 'z': event['z'] + eps}, station, velocity_model)[0] - base_t) / eps

    return base_t, segments, dtdx, dtdy, dtdz
