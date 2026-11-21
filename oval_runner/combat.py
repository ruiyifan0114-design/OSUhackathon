"""Aim and weapon timing for the mechfight/1 simulator."""

from dataclasses import dataclass
import math


def angle_error(first, second):
    return math.remainder(first - second, math.tau)


@dataclass(slots=True)
class Shot:
    yaw: float
    pitch: float
    distance: float
    horizontal_tolerance: float
    vertical_tolerance: float


class Combat:
    def __init__(self, hello, arena=None):
        self.arena = arena
        self.config = hello["config"]
        match = self.config["match"]
        self.step = 1 / match["tick_hz"]
        self.steps = max(1, int(match["decide_every"]))
        self.frame = self.step * self.steps
        self.laser_wait = 0

    def dimensions(self, target):
        if target.get("kind") == "mech" or target["id"] < 16:
            stats = self.config["mech"]
            scale = 1 + self.config["cubes"]["size_per_cube"] * target.get("cubes", 0)
            return stats["radius"] * scale, stats["height"] * scale
        group = "npc" if target.get("kind") == "npc" or target["id"] < 1000 else "crates"
        stats = self.config[group]
        return stats["radius"], stats["height"]

    def visible(self, me, target):
        """Check muzzle line of sight, including a target peeking over cover."""
        if self.arena is None:
            return True
        p = me["pos"]
        start = (p[0], p[1], p[2] + me["stats"]["muzzle_height"])
        x, y, z = target["pos"]
        _, height = self.dimensions(target)
        return self.arena.clear(start, (x, y, z + height * 0.52)) or self.arena.clear(
            start, (x, y, z + height * 0.86)
        )

    def _lead(self, dx, dy, dz, velocity, speed, offset, gravity, ttl):
        """Solve the first ballistic intercept, including the muzzle offset."""
        flight = max(0, (math.sqrt(dx * dx + dy * dy + dz * dz) - offset) / speed)
        if not gravity and not any(velocity):
            return flight
        vx, vy, vz = velocity
        for _ in range(7):
            x = dx + vx * flight
            y = dy + vy * flight
            z = dz + (vz + 0.5 * gravity * flight) * flight
            length = math.sqrt(x * x + y * y + z * z)
            miss = length - (speed * flight + offset)
            if abs(miss) < 0.04:
                return flight
            derivative = (x * vx + y * vy + z * (vz + gravity * flight)) / max(length, 0.01) - speed
            if derivative > -0.05 * speed:
                return None
            flight = max(0, flight - miss / derivative)
            if flight > ttl:
                return None
        return None

    def _trajectory_clear(self, start, end, flight, gravity):
        if self.arena is None:
            return True
        if not gravity or flight < 0.25:
            return self.arena.clear(start, end)
        # Check short chords, with their endpoints on the simulated parabola.
        chords = min(10, max(2, math.ceil(flight * 3.5)))
        dx, dy, dz = (end[i] - start[i] for i in range(3))
        bend = 0.5 * gravity * flight * flight
        previous = start
        for i in range(1, chords + 1):
            fraction = i / chords
            point = (
                start[0] + dx * fraction,
                start[1] + dy * fraction,
                start[2] + (dz + bend * (1 - fraction)) * fraction,
            )
            if not self.arena.clear(previous, point):
                return False
            previous = point
        return True

    def _solution(self, me, target, weapon, latency, aim_at):
        stats = weapon["stats"]
        kind = weapon["kind"]
        own = me["stats"]
        radius, height = self.dimensions(target)
        p = me["pos"]
        own_v = me["vel"]
        muzzle = own["height"] + 0.5 if kind == "mortar" else own["muzzle_height"]
        origin = (p[0] + own_v[0] * latency, p[1] + own_v[1] * latency,
                  p[2] + own_v[2] * latency + muzzle)
        target_pos = target["pos"]
        velocity = target.get("vel", (0, 0, 0))
        dx = target_pos[0] + velocity[0] * latency - origin[0]
        dy = target_pos[1] + velocity[1] * latency - origin[1]
        dz = target_pos[2] + velocity[2] * latency + height * aim_at - origin[2]
        offset = own["radius"] + 0.5
        gravity = stats.get("gravity", 0)
        flight = 0.0
        projectile_speed = stats.get("speed", 0)
        # Skip obviously out-of-range trajectories before solving or ray tracing.
        if projectile_speed:
            ttl = stats["ttl"] if gravity else stats["range"] / projectile_speed
            flight = self._lead(dx, dy, dz, velocity, projectile_speed, offset, gravity, ttl)
            if flight is None or flight > ttl:
                return None
        ex, ey, ez = dx + velocity[0] * flight, dy + velocity[1] * flight, dz + velocity[2] * flight
        horiz = math.hypot(ex, ey)
        distance = math.hypot(horiz, ez)
        if kind == "lightning" and distance > stats["range"]:
            return None
        if not projectile_speed and kind != "lightning" and distance > stats["range"] + offset:
            return None
        if gravity and horiz < own["radius"] + stats["splash_radius"] + 2:
            return None
        arc_rise = 0.5 * gravity * flight * flight
        pitch = math.atan2(ez + arc_rise, horiz)
        if not own["min_pitch"] <= pitch <= own["max_pitch"] and kind != "lightning":
            return None
        pitch = min(own["max_pitch"], max(own["min_pitch"], pitch))
        endpoint = (origin[0] + ex, origin[1] + ey, origin[2] + ez)
        if gravity:
            # The shell spawns offset along the ray and only then starts falling.
            arc_length = math.hypot(horiz, ez + arc_rise)
            fraction = min(1.0, offset / max(arc_length, 0.01))
            start = (origin[0] + ex * fraction, origin[1] + ey * fraction,
                     origin[2] + (ez + arc_rise) * fraction)
            if self.arena is not None and not self.arena.clear(origin, start):
                return None
        else:
            start = origin
        if not self._trajectory_clear(start, endpoint, flight, gravity):
            return None
        horizontal = math.atan2(radius * 0.90, max(0.1, horiz))
        vertical = math.atan2(height * min(aim_at, 1 - aim_at) * 0.9, max(0.1, horiz))
        if kind == "lightning":
            horizontal = math.radians(stats["cone_deg"]) * 0.49
            vertical = math.pi
        return Shot(math.atan2(ey, ex), pitch, distance, horizontal, vertical)

    def _shot(self, me, target, weapon, latency):
        shot = self._solution(me, target, weapon, latency, 0.52)
        if shot is None and weapon["kind"] in ("gun", "laser"):
            shot = self._solution(me, target, weapon, latency, 0.83)
        return shot

    def _heading_hits(self, me, aimed, shot, delay=0):
        """Check the aim after the first simulation step on which a shot can fire."""
        if shot is None:
            return False
        steps = min(self.steps, max(1, math.ceil((delay - 1e-6) / self.step)))
        elapsed = steps * self.step
        own_yaw, own_pitch = me["aim"]
        yaw_delta = angle_error(aimed.yaw, own_yaw)
        pitch_delta = aimed.pitch - own_pitch
        yaw_step = me["stats"]["turn_rate"] * elapsed
        pitch_step = me["stats"]["pitch_rate"] * elapsed
        yaw = own_yaw + min(yaw_step, max(-yaw_step, yaw_delta))
        pitch = own_pitch + min(pitch_step, max(-pitch_step, pitch_delta))
        return (
            abs(angle_error(yaw, shot.yaw)) <= shot.horizontal_tolerance
            and abs(pitch - shot.pitch) <= shot.vertical_tolerance
        )

    def _range(self, weapon):
        stats = weapon["stats"]
        return stats.get("range", stats.get("speed", 0) ** 2 / max(stats.get("gravity", 1), 1))

    def select(self, obs, candidates, prior_id=None):
        """Choose an in-range, shootable target with a bias toward finishing."""
        me = obs["self"]
        x, y, _ = me["pos"]
        weapons = [weapon for weapon in me["weapons"] if weapon]
        reach = max((self._range(weapon) for weapon in weapons), default=0)
        scored = []
        for target in candidates:
            tx, ty, _ = target["pos"]
            distance = math.hypot(tx - x, ty - y)
            if distance > reach + 10:
                continue
            remaining = min(1.5, target.get("hp", 1400) / 1400)
            turn = abs(angle_error(math.atan2(ty - y, tx - x), me["aim"][0]))
            score = distance * 0.55 + 66 * remaining + 12 * turn
            if target["id"] == prior_id:
                score -= 14
            elif target.get("charging") and distance < 90:
                score -= 7
            scored.append((score, target["id"], target))
        scored.sort()
        for _, _, target in scored[:5]:
            if self.visible(me, target):
                return target
        mortar = next((w for w in weapons if w["kind"] == "mortar" and
                       w["cooldown"] <= self.frame + 0.20), None)
        if mortar:
            for _, _, target in scored[:2]:
                if self._shot(me, target, mortar, self.step):
                    return target
        return None

    def apply(self, action, obs, target, scan_yaw, allow_fire=True):
        me = obs["self"]
        weapons = me["weapons"]
        action["aim"] = [math.remainder(scan_yaw, math.tau), 0]
        action["fire"] = [False, False]
        gun = next((weapon for weapon in weapons if weapon and weapon["kind"] == "gun"), None)
        if target is None:
            self.laser_wait = 0
            action["reload"] = bool(gun and gun.get("reload", 0) <= 0
                                    and gun.get("ammo", 0) < gun["stats"]["mag"])
            return action

        p = me["pos"]
        t = target["pos"]
        _, height = self.dimensions(target)
        dx, dy = t[0] - p[0], t[1] - p[1]
        distance = math.hypot(dx, dy)
        own = me["stats"]
        yaw = math.atan2(dy, dx)
        pitch = math.atan2(t[2] - p[2] + 0.52 * height - own["muzzle_height"], distance)
        default = Shot(yaw, min(own["max_pitch"], max(own["min_pitch"], pitch)), distance, 0, 0)
        action["aim"] = [default.yaw, default.pitch]
        if not allow_fire:
            self.laser_wait = 0
            action["reload"] = bool(gun and 0 < gun.get("ammo", 0) < gun["stats"]["mag"])
            return action

        shots = {}
        laser = None
        charge_goal = 0.0
        for slot, weapon in enumerate(weapons):
            if weapon is None:
                continue
            kind = weapon["kind"]
            cooldown = weapon.get("cooldown", 0)
            if kind == "gun" and weapon.get("reload", 0) > 0.25:
                continue
            if kind == "mortar" and cooldown > self.frame + 0.20:
                continue
            if kind == "lightning" and (cooldown > self.frame + 0.05 or distance > weapon["stats"]["range"] + 2):
                continue
            if kind == "laser":
                if cooldown > self.frame and not weapon.get("charge", 0):
                    continue
                laser = slot
                charge_goal = self._charge_goal(me, target, weapon)
                if not weapon.get("charge", 0) and "lightning" in target.get("weapons", ()) and distance < 48:
                    continue
            latency = self.step * 0.55 + min(self.step, max(0, cooldown - self.step))
            shot = self._shot(me, target, weapon, latency)
            if shot is not None:
                shots[slot] = shot

        aim = default
        best = -1
        for slot, shot in shots.items():
            weapon = weapons[slot]
            kind = weapon["kind"]
            if kind == "gun":
                score = 30
            elif kind == "lightning":
                score = 22 if gun else 36
            elif kind == "mortar":
                score = 60
            else:
                charge = weapon.get("charge", 0)
                # Track the gun while charging until it is time to release.
                score = 80 if charge >= charge_goal - 0.04 else 27
            if score > best:
                best, aim = score, shot
        action["aim"] = [aim.yaw, aim.pitch]

        for slot, shot in shots.items():
            weapon = weapons[slot]
            kind = weapon["kind"]
            if kind == "laser":
                continue
            if weapon.get("cooldown", 0) > self.frame + 1e-4:
                continue
            if kind == "gun" and (weapon.get("ammo", 0) <= 0 or weapon.get("reload", 0) > self.frame):
                continue
            if kind == "lightning":
                cost = weapon["stats"]["energy"]
                reserve = 14 if not me["grounded"] else 5
                if me["energy"] < cost + reserve:
                    continue
            action["fire"][slot] = self._heading_hits(me, aim, shot, weapon.get("cooldown", 0))

        if laser is not None:
            weapon = weapons[laser]
            charge = weapon.get("charge", 0)
            shot = shots.get(laser)
            if charge:
                ready = charge >= charge_goal - 1e-3
                aligned = self._heading_hits(me, aim, shot)
                self.laser_wait = self.laser_wait + self.frame if ready or shot is None else 0
                action["fire"][laser] = not (ready and aligned or self.laser_wait > 0.45)
            elif shot is not None:
                self.laser_wait = 0
                action["fire"][laser] = abs(angle_error(shot.yaw, me["aim"][0])) < 0.32
        return action

    def _charge_goal(self, me, target, weapon):
        stats = weapon["stats"]
        full = stats["charge"]
        hp = target.get("hp")
        if hp is None:
            return full
        cubes = self.config["cubes"]
        damage = stats["damage"] * (1 + cubes["damage_per_cube"] * min(me["cubes"], cubes["stat_cap"]))
        if "mortar" in target.get("weapons", ()):
            damage *= 1 + stats["cookoff"]
        if target.get("kind") == "mech":
            # Enemy armor is private. Do not release early on a marginal estimate.
            damage *= 0.92
        needed = (hp / damage - stats["min_frac"]) / (1 - stats["min_frac"])
        return min(full, max(stats["min_charge"], needed * full + self.step))
