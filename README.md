# Oval Runner IV

A Python Mech Royale bot for the Builders Buildathon. The repository root is the Docker build context. The image reads match observations on stdin, sends actions on stdout, and loads the arena from the path in the match handshake.

```sh
docker build --platform linux/amd64 -t oval-runner .
```

The bot uses only the Python standard library and runs as an unprivileged user on a read-only filesystem. `oval_runner/mechfight.py` is the protocol wrapper from the official [Mech Royale starter kit](https://www.joinbuilders.org/buildathon/tracks/builders). The arena and conformance checker are provided separately by the organizers.

## What it does

- **Map-aware combat**: reads the arena height field and building/bridge prisms, traces shots through cover, and validates local movement against walls and ledges.
- **Predictive weapons**: leads gun targets, arcs mortar shells, charges/releases lasers, and schedules both weapon slots around one torso.
- **Sound awareness**: keeps a short recency-weighted memory of gunfire, jets, footsteps, laser charges, and other sounds; turns to scan unseen threats, backs away from close contacts, and can use a nearby sound as a coarse lightning target.
- **Mobility build**: values `servos`, `jets`, `capacitor`, and `reactor`; boosts normal travel, closing fights, and projectile dodges while preserving energy for lightning and braking earlier at pickups.
- **Survival first**: disengages to cover at low HP or against multiple threats, avoids long open-ground chases, values wounded/isolated targets, and uses zone timing and supply drops.
- **Config-driven upgrades**: values an offer from its `hello.config.upgrades` effect paths instead of a hard-coded id list, including `lightning_chain`, `laser_pierce`, and `mortar_velocity`.
- **Projectile evasion**: uses warning time and speed in metres (not a dimensionless gain) to choose lateral movement away from bullets and mortar splash.

## Local verification

Use the official kit's checker with this repository as the build context:

```sh
docker build --platform linux/amd64 -t oval-runner .
python3 /path/to/mechfight-kit/check/check.py \
  --map-dir /arena -- \
  docker run --rm -i --network none --cpus 1 --memory 512m --read-only \
  --tmpfs /tmp:rw,size=64m --user 65534:65534 \
  -v /path/to/mechfight-kit/arena:/arena:ro oval-runner
```

The conformance checker only validates protocol and timing; it is not a match simulator.

The original Oval Runner code is released under the MIT License.
