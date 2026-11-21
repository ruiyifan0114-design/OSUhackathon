"""Mech Royale entry point."""
import math
from pathlib import Path
import sys

# The container copies the wrapper alongside this file.
sys.path.append(str(Path(__file__).parent / "mechfight-kit" / "python"))
import mechfight as mf


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def angle_error(a, b):
    return abs(math.remainder(a - b, math.tau))


class Bot:
    name = "Oval Runner"

    def __init__(self, hello):
        self.config = hello["config"]
        self.anchor = None
        self.unstick_until = 0
        self.was_moving = False
        self.side = 1 if hello["slot"] % 2 else -1
        self.last_gate = 0
        self.recovering = False

    def act(self, obs):
        me = obs["self"]
        pos = me["pos"]
        now = obs["t"]
        zone = obs["zone"]
        center = zone["next_center"]
        radius = zone["next_radius"]
        weapons = me["weapons"]
        enemies = [actor for actor in obs["mechs"] if actor["kind"] == "mech"]
        enemy = min(enemies, key=lambda actor: distance(pos, actor["pos"]) +
                    45 * actor.get("hp", 1400) / max(1, actor.get("max_hp", 1400)),
                    default=None)
        nearest = min((distance(pos, actor["pos"]) for actor in enemies), default=math.inf)
        separation = distance(pos, enemy["pos"]) if enemy else math.inf
        hp_fraction = me["hp"] / me["max_hp"]
        if hp_fraction < 0.42:
            self.recovering = True
        elif hp_fraction > 0.78:
            self.recovering = False
        urgent = not me["in_zone"] or (
            distance(pos, center) > radius - 10 and
            distance(pos, center) - radius + 18 >
            max(0, zone["shrink_end"] - now - 5) * me["stats"]["walk_speed"]
        )
        goal = center
        action = {}
        reach = self.config["loot"]["pickup_radius"]
        ratings = {"gun": 0, "mortar": 0, "laser": 3 if radius > 70 else 2,
                   "lightning": 2 if radius > 70 else 3}
        held = {weapon["kind"] for weapon in weapons if weapon}
        pickups = [
            item for item in obs["pickups"]
            if item["kind"] not in held and ratings[item["kind"]] >
            (ratings[weapons[1]["kind"]] if weapons[1] else 0)
            and abs(item["pos"][2] - pos[2]) <= 4
        ]
        pickup = min(pickups, key=lambda item: distance(pos, item["pos"]), default=None)
        if pickup and distance(pos, pickup["pos"]) < reach:
            action["take"] = pickup["id"]
            if weapons[1]:
                if weapons[1].get("charge", 0):
                    action.pop("take")
                else:
                    action["drop"] = 1
        if not urgent:
            objectives = []
            if nearest > 45:
                objectives.extend((distance(pos, item["pos"]) - 18, item["pos"])
                                  for item in obs["cubes"]
                                  if abs(item["pos"][2] - pos[2]) < 4)
                if pickup and distance(pos, pickup["pos"]) < 55:
                    objectives.append((distance(pos, pickup["pos"]) - 12, pickup["pos"]))
                if not self.recovering and nearest > 100 and me["cubes"] < 10:
                    objectives.extend((distance(pos, crate["pos"]) + 18, crate["pos"])
                                      for crate in obs["crates"]
                                      if distance(pos, crate["pos"]) < 65
                                      and abs(crate["pos"][2] - pos[2]) < 4)
                for drop in obs["drops"]:
                    d = distance(pos, drop["pos"])
                    if d < 105 and drop["holder"] in (None, me["id"]) and drop["lands_in"] < 8:
                        objectives.append((d * 0.65 - 45, drop["pos"]))
            if objectives:
                goal = min(objectives, key=lambda item: item[0])[1]
            elif distance(pos, center) < max(8, radius * 0.65):
                goal = pos

        move_yaw = math.atan2(goal[1] - pos[1], goal[0] - pos[0])
        moving = distance(pos, goal) > 3
        if separation < 115 and not urgent:
            bearing = math.atan2(enemy["pos"][1] - pos[1], enemy["pos"][0] - pos[0])
            preferred = 30 if "lightning" in held else 65
            retreat = self.recovering or separation < preferred - 12
            close = separation > preferred + 12 and len(enemies) <= 1
            move_yaw = bearing + self.side * (2.65 if retreat else 0.85 if close else 1.57)
            moving = True
        if self.anchor is None or now - self.last_gate >= 0.8:
            if self.anchor and self.was_moving and moving and distance(pos, self.anchor) < 2:
                self.unstick_until = now + 1.25
                self.side *= -1
            self.anchor = pos
            self.last_gate = now
            self.was_moving = moving
        unsticking = moving and now < self.unstick_until
        if unsticking:
            move_yaw += self.side * 1.05
        action["move"] = [math.cos(move_yaw), math.sin(move_yaw)] if moving else [0, 0]
        action["boost"] = moving and (
            (me["grounded"] and me["energy"] > 25 and
             (urgent or separation < 28 or self.recovering and separation < 100))
            or (unsticking and not me["grounded"] and me["energy"] > 35)
        )
        action["jump"] = unsticking and me["grounded"]

        target = enemy
        if not target and not urgent and not self.recovering:
            farm = list(obs["crates"])
            farm.extend(actor for actor in obs["mechs"] if actor["kind"] == "npc")
            target = min(farm, key=lambda item: distance(pos, item["pos"]) +
                         (12 if item["id"] >= 1000 else 0), default=None)
            if target and distance(pos, target["pos"]) > 50:
                target = None
        self.aim_and_fire(action, me, target, move_yaw if moving else me["aim"][0] + 0.16)
        if self.recovering and separation > 50:
            action["fire"] = [False, False]
        if obs["offer"]:
            priorities = {"armor": 5, "gun_rate": 4, "sensors": 2}
            action["pick"] = max(range(len(obs["offer"]["choices"])),
                                 key=lambda i: priorities.get(obs["offer"]["choices"][i]["id"], 1))
        return action

    def aim_and_fire(self, action, me, target, scan_yaw):
        weapons = me["weapons"]
        action["aim"] = [math.remainder(scan_yaw, math.tau), 0]
        action["fire"] = [False, False]
        if not target:
            action["reload"] = any(w and w["kind"] == "gun" and 0 < w["ammo"] < w["stats"]["mag"]
                                   for w in weapons)
            return
        pos = me["pos"]
        d = distance(pos, target["pos"])
        secondary = weapons[1]
        instant = secondary and d < secondary["stats"]["range"] and (
            secondary["kind"] == "lightning" or
            secondary["kind"] == "laser" and (secondary["cooldown"] <= 0.1 or secondary.get("charge", 0))
        )
        gun = weapons[0]
        lead = 0.05 if instant else 0.05 + d / gun["stats"]["speed"]
        velocity = target.get("vel", (0, 0, 0))
        dx, dy, dz = (target["pos"][i] + velocity[i] * lead - pos[i] for i in range(3))
        height = 4.5 * (1 + 0.05 * target.get("cubes", 0)) if target.get("kind") == "mech" else 1.5
        yaw = math.atan2(dy, dx)
        pitch = math.atan2(dz + height - me["stats"]["muzzle_height"], math.hypot(dx, dy))
        action["aim"] = [yaw, pitch]
        error = angle_error(yaw, me["aim"][0])
        tolerance = math.atan2(2.0, max(1, d))
        aligned = error < tolerance and abs(pitch - me["aim"][1]) < tolerance
        gun_dx = dx + (velocity[0] * d / gun["stats"]["speed"] if instant else 0)
        gun_dy = dy + (velocity[1] * d / gun["stats"]["speed"] if instant else 0)
        gun_yaw = math.atan2(gun_dy, gun_dx)
        # The action can keep its current aim at the first simulation tick,
        # then slew toward the new aim before the second one.
        gun_error = max(angle_error(gun_yaw, me["aim"][0]), angle_error(gun_yaw, yaw))
        action["fire"][0] = (gun_error < tolerance and abs(pitch - me["aim"][1]) < tolerance
                             and d < gun["stats"]["range"])
        if instant and secondary["kind"] == "lightning":
            action["fire"][1] = error < math.radians(secondary["stats"]["cone_deg"]) * 0.45 and me["energy"] > 10
        elif instant and secondary["kind"] == "laser":
            charge = secondary.get("charge", 0)
            action["fire"][1] = (charge > 0 or error < 0.24) and not (
                charge >= secondary["stats"]["charge"] - 0.025 and aligned
            )


if __name__ == "__main__":
    mf.run(Bot)
