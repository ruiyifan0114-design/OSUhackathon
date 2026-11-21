"""Short-horizon motion used by steering and muzzle prediction."""
import math


def planar_delta(vx, vy, tx, ty, accel, seconds):
    dx, dy = tx - vx, ty - vy
    ramp = math.hypot(dx, dy) / max(0.1, accel)
    gain = 0.5 * seconds * seconds / ramp if seconds < ramp else seconds - 0.5 * ramp
    return vx * seconds + dx * gain, vy * seconds + dy * gain


def flight_delta(vz, acceleration, cap, seconds):
    if acceleration <= 0:
        return vz * seconds + 0.5 * acceleration * seconds * seconds
    ramp = min(seconds, max(0.0, (cap - vz) / acceleration))
    return vz * ramp + 0.5 * acceleration * ramp * ramp + cap * (seconds - ramp)


def feet_after(me, action, seconds, arena=None, charging=False):
    """Predict feet at launch time from the command being sent this frame."""
    pos, stats = me["pos"], me["stats"]
    vx, vy, vz = me["vel"]
    jumping = me["grounded"] and action.get("jump", False)
    grounded = me["grounded"] and not jumping
    boosting = action.get("boost", False) and me["energy"] > 0
    mx, my = action.get("move", (0, 0))
    speed = stats["boost_speed"] if grounded and boosting else stats["walk_speed"]
    if charging or any(w and w.get("charge", 0) > 0 for w in me["weapons"]):
        speed *= stats["charge_move_mult"]
    accel = stats["ground_accel" if grounded else "air_accel"]
    dx, dy = planar_delta(vx, vy, mx * speed, my * speed, accel, seconds)
    if grounded and arena is not None and (dx or dy):
        fraction = arena.walk_fraction(pos, dx, dy, stats["radius"], stats["height"], stats["step_up"])
        dx, dy = dx * fraction, dy * fraction
    x, y = pos[0] + dx, pos[1] + dy
    z = pos[2]
    if not grounded:
        jet = stats["jet_accel"] if boosting and not jumping else 0.0
        vz = stats["jump_speed"] if jumping else vz
        dz = flight_delta(vz, jet - stats["gravity"], stats["max_rise_speed"], seconds)
        ceiling_clear = (arena is None or dz <= 0 or arena.clear(
            (pos[0], pos[1], z + stats["height"]), (x, y, z + stats["height"] + dz)
        ))
        if ceiling_clear:
            z += dz
        if arena is not None:
            ground = arena.ground(x, y)
            if ground is not None:
                z = max(z, ground)
    elif arena is not None:
        here, ground = arena.ground(*pos[:2]), arena.ground(x, y)
        if here is not None and ground is not None and abs(pos[2] - here) <= stats["step_up"]:
            if abs(ground - z) <= stats["step_up"]:
                z = ground
    return x, y, z
