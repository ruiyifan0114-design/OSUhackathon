# Oval Runner

A Python Mech Royale bot for the Builders Buildathon. The repository root is the Docker build context. The image reads match observations on stdin, sends actions on stdout, and loads the arena from the path in the match handshake.

```sh
docker build --platform linux/amd64 -t oval-runner .
```

The bot uses only the Python standard library. `oval_runner/mechfight.py` is the protocol wrapper from the official [Mech Royale starter kit](https://www.joinbuilders.org/buildathon/tracks/builders). The arena and conformance checker are provided separately by the organizers.

The original Oval Runner code is released under the MIT License.
