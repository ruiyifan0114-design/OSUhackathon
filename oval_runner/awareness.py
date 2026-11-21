"""Short-lived sound memory for threats outside the vision wedge.

The server reports a world-frame bearing, a coarse distance bucket, and the
age of every sound heard since the previous observation.  The bearing is
approximate, so this module deliberately treats sounds as *evidence* rather
than as exact targets: it feeds a scan direction and a conservative pressure
estimate back to the tactical layer.
"""
from dataclasses import dataclass
import math


BUCKET_DISTANCE = {"near": 28.0, "mid": 100.0, "far": 220.0}
BUCKET_WEIGHT = {"near": 1.0, "mid": 0.58, "far": 0.25}

# Relative danger of hearing each kind, in the same rough units as the
# visible-threat pressure estimate in Bot.threats (HP/s equivalent).
BASE_PRESSURE = {
    "laser_charge": 78.0,
    "lightning": 64.0,
    "mortar": 58.0,
    "explosion": 56.0,
    "laser": 52.0,
    "gun": 44.0,
    "boost": 27.0,
    "step": 25.0,
    "jet": 22.0,
    "npc_gun": 14.0,
    "reload": 8.0,
    "crate": 6.0,
}
MAX_AGE = 6.0


@dataclass(slots=True)
class Contact:
    kind: str
    bearing: float
    distance: float
    bucket: str
    heard_at: float
    pressure: float

    def age(self, now):
        return max(0.0, now - self.heard_at)

    def score(self, now):
        return self.pressure * math.exp(-self.age(now) / 1.7)

    def point(self, pos):
        return (
            pos[0] + self.distance * math.cos(self.bearing),
            pos[1] + self.distance * math.sin(self.bearing),
            pos[2],
        )


class Awareness:
    """Keep a small, recency-weighted memory of heard enemy activity."""

    def __init__(self, hello):
        sound = hello.get("config", {}).get("sound", {})
        self.near = float(sound.get("near", 60))
        self.mid = float(sound.get("mid", 160))
        self.contacts = []

    def update(self, obs):
        now = obs["t"]
        self.contacts = [c for c in self.contacts if now - c.heard_at <= MAX_AGE]
        for raw in obs.get("sounds", ()):
            kind = raw.get("kind")
            if kind not in BASE_PRESSURE:
                continue
            age = max(0.0, float(raw.get("age", 0.0)))
            heard = now - age
            if heard < now - MAX_AGE:
                continue
            bucket = raw.get("dist", "far")
            if bucket not in BUCKET_DISTANCE:
                bucket = "far"
            bearing = math.remainder(float(raw.get("bearing", 0.0)), math.tau)
            distance = BUCKET_DISTANCE[bucket]
            pressure = BASE_PRESSURE[kind] * BUCKET_WEIGHT[bucket]
            # Repeated steps/boosts can arrive every few ticks.  Collapse a
            # near-identical contact instead of letting one runner dominate
            # the memory with a dozen copies.
            duplicate = next(
                (c for c in self.contacts
                 if c.kind == kind and c.bucket == bucket
                 and heard - c.heard_at < 0.30
                 and abs(math.remainder(c.bearing - bearing, math.tau)) < 0.20),
                None,
            )
            if duplicate is not None:
                duplicate.heard_at = max(duplicate.heard_at, heard)
                duplicate.bearing = bearing
                duplicate.pressure = max(duplicate.pressure, pressure)
                continue
            self.contacts.append(Contact(kind, bearing, distance, bucket, heard, pressure))
        if len(self.contacts) > 36:
            self.contacts.sort(key=lambda c: c.heard_at, reverse=True)
            del self.contacts[36:]

    def alert(self, now, max_age=3.4):
        """Return the strongest recent hostile sound, or None."""
        best = None
        best_score = 0.0
        for contact in self.contacts:
            age = contact.age(now)
            if age > max_age:
                continue
            score = contact.score(now)
            if score > best_score:
                best, best_score = contact, score
        # The score threshold corresponds to a faint far gunshot; ignore it.
        if best is None or best_score < 8.0:
            return None
        return best

    def scan_bearing(self, now, max_age=2.5):
        contact = self.alert(now, max_age=max_age)
        return contact.bearing if contact is not None else None

    def pressure(self, now, max_age=2.4):
        contact = self.alert(now, max_age=max_age)
        return contact.score(now) if contact is not None else 0.0
