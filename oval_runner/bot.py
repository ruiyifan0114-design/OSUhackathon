"""Mechfight strategy: build an early lead, then isolate opponents."""
import math

import mechfight as mf
from arena import Arena
from combat import Combat
from movement import Movement, distance


UPGRADES = {
    "nanites": 9, "gun_mag": 8, "lightning_range": 8, "plating": 7,
    "gun_damage": 6, "armor": 6, "lightning_efficiency": 6,
    "mortar_reload": 6, "servos": 5, "laser_damage": 5, "mortar_splash": 5,
    "laser_charge": 4, "jets": 4, "gun_velocity": 4, "gun_rate": 3,
    "reactor": 3, "capacitor": 2, "sensors": 2, "gyros": 2,
}


class Bot:
    name = "Oval Runner II"

    def __init__(self, hello):
        self.config = hello["config"]
        self.arena = Arena(hello)
        self.combat = Combat(hello, self.arena)
        self.movement = Movement(hello, self.arena)
        self.recovering = False
        self.hit_rate = 0
        self.last_time = 0
        self.last_bearing = None
        self.last_damage = -100
        self.combat_target = None
        self.goal_id = None
        self.loot = {}
        self.explore_angle = (hello["slot"] / 16) * math.tau

    def update(self, obs):
        now = obs["t"]
        dt = min(2, max(0, now - self.last_time))
        self.hit_rate *= math.exp(-dt / 2.0)
        self.last_time = now
        for event in obs["events"]:
            if event["type"] == "damaged" and event["weapon"] != "zone":
                self.hit_rate += event["amount"] / 2.0
                self.last_bearing = event["bearing"]
                self.last_damage = now

        visible = set()
        for kind in ("cubes", "pickups", "crates"):
            for item in obs[kind]:
                visible.add(item["id"])
                self.loot[item["id"]] = (kind, item, now)
        pos = obs["self"]["pos"]
        for key, (kind, item, seen) in list(self.loot.items()):
            if key not in visible and (now - seen > 16 or distance(pos, item["pos"]) < 8):
                del self.loot[key]

    def act(self, obs):
        self.update(obs)
        me = obs["self"]
        pos = me["pos"]
        now = obs["t"]
        zone = obs["zone"]
        center = zone["next_center"]
        guns = me["weapons"]
        secondary = guns[1]
        enemies = sorted((x for x in obs["mechs"] if x["kind"] == "mech"),
                         key=lambda x: distance(pos, x["pos"]))
        nearby = [x for x in enemies if distance(pos, x["pos"]) < 115]
        nearest = distance(pos, enemies[0]["pos"]) if enemies else math.inf
        previous_target = self.combat_target
        candidate = self.combat.select(obs, enemies[:7], previous_target)
        self.combat_target = candidate["id"] if candidate else None
        hp = me["hp"] / me["max_hp"]
        low = 0.50 if len(nearby) < 2 else 0.69
        if zone["next_radius"] == 0:
            self.recovering = False
        elif hp < low or (hp < 0.83 and self.hit_rate * 4 > me["hp"]):
            self.recovering = True
        elif hp >= 0.85:
            self.recovering = False

        zone_goal, urgent = self.zone_goal(obs)
        action = {}
        if obs["offer"]:
            choices = obs["offer"]["choices"]
            action["pick"] = max(range(len(choices)), key=lambda i: UPGRADES.get(choices[i]["id"], 0))
        ratings = self.weapon_ratings(obs, enemies)
        rating = ratings[secondary["kind"]] if secondary else 0
        reach = self.config["loot"]["pickup_radius"]
        options = [p for p in obs["pickups"] if p["kind"] != "gun"
                   and ratings[p["kind"]] > rating + (0.8 if secondary else 0)
                   and distance(pos, p["pos"]) < reach and abs(p["pos"][2] - pos[2]) < 4]
        if options and not (secondary and secondary.get("charge", 0)):
            action["take"] = max(options, key=lambda p: ratings[p["kind"]])["id"]
            if secondary:
                action["drop"] = 1

        goal, farm_target, mode = self.objective(obs, zone_goal, urgent, nearby, ratings, rating)
        enemy_distance = distance(pos, candidate["pos"]) if candidate else math.inf
        ally_pressure = len(nearby) >= 2
        easy_finish = candidate and candidate.get("hp", 1400) < 230 and enemy_distance < 85
        can_finish = easy_finish and hp > 0.22 and not (ally_pressure and self.hit_rate > 120)
        fighting = candidate and enemy_distance < (165 if candidate["id"] == previous_target else 145)
        evasive = bool(zone["next_radius"] > 0 and
                       (self.recovering or ally_pressure and self.hit_rate > 85) and not can_finish)
        committed_drop = mode == "drop" and (
            distance(pos, goal) < 6 and hp > 0.28
            or distance(pos, goal) < 37 and nearest > 35 and hp < 0.8
        )
        seeking_gear = mode == "weapon" and not secondary and nearest > 38
        if committed_drop or seeking_gear and hp > 0.36:
            # A nearby repair or first secondary can flip an opening duel.
            pass
        elif not urgent and (fighting or nearby):
            if evasive or fighting and (mode not in ("cube", "weapon") or nearest < 44):
                opponent = candidate if candidate is not None else nearby[0]
                weapon_range = secondary["stats"]["range"] if secondary and secondary["kind"] == "lightning" else None
                preferred = weapon_range * 0.79 if weapon_range and me["energy"] > 15 else 67
                if secondary and secondary["kind"] == "mortar":
                    preferred = 51
                if not weapon_range and "lightning" in opponent.get("weapons", ()):
                    preferred = max(preferred, 62)
                if can_finish:
                    preferred = min(preferred, 43)
                goal = self.movement.engage(obs, opponent, preferred, evasive, zone_goal)
                mode = "retreat" if evasive else "fight"
        elif evasive and nearest < 150 and not urgent:
            goal = self.movement.engage(obs, enemies[0], 70, True, zone_goal)
            mode = "retreat"
        elif evasive and not urgent and self.last_bearing is not None and now - self.last_damage < 2:
            # Canopy or a wall can hide the attacker even just after a hit.
            bearing = self.last_bearing + 0.36 * self.movement.side
            goal = (pos[0] - 28 * math.cos(bearing), pos[1] - 28 * math.sin(bearing), pos[2])
            mode = "retreat"

        self.movement.apply(action, obs, goal, mode, nearby)
        target = candidate or farm_target
        allow_fire = not evasive or committed_drop
        if not allow_fire and nearest < 24 and hp > 0.16:
            allow_fire = True
        if not allow_fire:
            target = None
        if target is None and self.last_bearing is not None and now - self.last_damage < 1.6:
            scan = self.last_bearing
        elif goal is not None and distance(pos, goal) > 5:
            scan = math.atan2(goal[1] - pos[1], goal[0] - pos[0]) + 0.50 * math.sin(now * 1.25)
        else:
            scan = me["aim"][0] + 0.31
        self.combat.apply(action, obs, target, scan, allow_fire)
        return action

    def weapon_ratings(self, obs, enemies):
        small = obs["zone"]["next_radius"] <= 70
        has_mortar_enemy = any("mortar" in e.get("weapons", ()) for e in enemies[:3])
        ratings = {"gun": 0, "mortar": 5.1, "laser": 4.4, "lightning": 6.0}
        if small:
            ratings["lightning"] += 2
        elif has_mortar_enemy:
            ratings["laser"] += 1.6
        for upgrade in obs["self"]["upgrades"]:
            prefix = upgrade.split("_", 1)[0]
            if prefix in ratings and prefix != "gun":
                ratings[prefix] += 0.65
        return ratings

    @staticmethod
    def zone_goal(obs):
        zone = obs["zone"]
        me = obs["self"]
        pos = me["pos"]
        now = obs["t"]
        zone_room = zone["radius"] - distance(pos, zone["center"])
        center = zone["next_center"]
        radius = zone["next_radius"]
        d = distance(pos, center)
        speed = me["stats"]["walk_speed"] * 0.62
        required = max(0, d - max(0, radius - 18)) / speed
        urgent = (
            not me["in_zone"]
            or zone_room < (12 if now >= zone["shrink_start"] else 1)
            or required > max(3, zone["shrink_end"] - now - 12)
            or radius == 0 and zone["shrink_end"] - now < 10
        )
        if d <= 2 or radius <= 15:
            return center, urgent
        inner = 0 if radius <= 25 else max(0, radius - (42 if urgent else 75))
        ratio = min(d, inner) / d
        return (center[0] + (pos[0] - center[0]) * ratio,
                center[1] + (pos[1] - center[1]) * ratio), urgent

    def objective(self, obs, zone_goal, urgent, nearby, ratings, current_weapon):
        me = obs["self"]
        pos = me["pos"]
        zone = obs["zone"]
        if urgent:
            self.goal_id = None
            return zone_goal, None, "zone"
        candidates = []
        hp = me["hp"] / me["max_hp"]
        now = obs["t"]
        vision_range = 125
        can_farm = not self.recovering and len(nearby) < 2 and self.hit_rate < 35 and me["cubes"] < 10

        def add(identity, point, mode, value, actor=None, radius=vision_range):
            d = distance(pos, point)
            if d > radius or distance(point, zone["center"]) > zone["radius"] - 8:
                return
            if mode != "drop" and abs(point[2] - pos[2]) > 5:
                return
            threat = min((distance(point, e["pos"]) for e in nearby), default=150)
            danger = max(0, 38 - threat) * (2.6 if self.recovering else 1.25)
            commitment = 9 if identity == self.goal_id else 0
            score = value + commitment - d - danger
            candidates.append((score, identity, point, mode, actor))

        for key, (kind, item, seen) in self.loot.items():
            if kind == "cubes":
                add(key, item["pos"], "cube", 61 if me["cubes"] < 10 else 30, radius=74)
            elif kind == "pickups":
                value = ratings[item["kind"]]
                if item["kind"] != "gun" and value > current_weapon + (0.8 if current_weapon else 0):
                    add(key, item["pos"], "weapon", 127 if not current_weapon else 42,
                        radius=127 if not current_weapon else 90)
            elif can_farm and now - seen < 5:
                add(key, item["pos"], "farm", 41 + 0.061 * (300 - item.get("hp", 300)), item, radius=87)

        if can_farm:
            for npc in obs["mechs"]:
                if npc["kind"] == "npc":
                    add(npc["id"], npc["pos"], "farm", 65 + 0.07 * (350 - npc.get("hp", 350)), npc, radius=90)
        for drop in obs["drops"]:
            d = distance(pos, drop["pos"])
            if drop["lands_in"] <= d / max(1, me["stats"]["walk_speed"]) + 8 and drop["holder"] in (None, me["id"]):
                add(drop["id"], drop["pos"], "drop", 115 + 60 * (1 - hp) + 18 * drop["progress"], radius=185)

        if candidates:
            score, self.goal_id, point, mode, target = max(candidates, key=lambda item: item[0])
            if score > -19:
                if self.recovering and mode == "farm":
                    return None, None, "idle"
                return point, target, mode
        self.goal_id = None
        if self.recovering:
            return None, None, "idle"
        if distance(pos, zone_goal) > 20:
            return zone_goal, None, "travel"
        if zone["next_radius"] < 55:
            return zone["next_center"], None, "travel"
        if me["cubes"] < 10 and not nearby:
            # Explore in persistent bursts so the forward vision wedge finds loot.
            angle = self.explore_angle + math.floor(now / 9) * 1.49
            point = (pos[0] + 46 * math.cos(angle), pos[1] + 46 * math.sin(angle), pos[2])
            if distance(point, zone["next_center"]) < zone["next_radius"] - 25:
                return point, None, "travel"
            return zone["next_center"], None, "travel"
        return None, None, "idle"


if __name__ == "__main__":
    mf.run(Bot)
