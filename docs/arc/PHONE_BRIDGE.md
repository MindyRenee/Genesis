# Genesis ARC phone bridge

This package connects a Dart/Flutter phone client to the existing Python ARC-AGI-3 adapter. It does **not** embed Python in Dart and does not add an LLM. The official ARC toolkit remains in Python; Dart provides typed phone-side transport and can submit explicit actions.

## Run on the same Android phone

From the Genesis repository root in Termux:

```sh
python -m pip install -e '.[arc]'
PYTHONPATH=python:. python scripts/arc_phone_bridge.py --host 127.0.0.1 --port 8765 --mind-socket /path/to/running/genesis.sock
```

The service intentionally binds to loopback only and has no authentication. Do not change it to listen on a LAN/public interface. It stores active sessions in memory; restarting the service ends them. Replace `/path/to/running/genesis.sock` with the actual socket path of an already-running Genesis daemon. The bridge starts a headless Mind connected to that daemon and forwards each initial/resulting observation through `Mind.world.perceive()` and each executed action through `Mind.world.it_acted()`. Those are the existing event boundaries used by Genesis's live cognitive integration. If `--mind-socket` is omitted, the bridge remains transport-only.

In the Dart/Flutter app, add this package as a path dependency:

```yaml
dependencies:
  genesis_arc_bridge:
    path: ../dart/genesis_arc_bridge
```

Example:

```dart
final arc = GenesisArcClient();
try {
  if (!await arc.health()) throw StateError('Bridge is not healthy');
  final games = await arc.listGames();
  final session = await arc.openGame(games.first.gameId);
  // The caller's policy must choose only from session.legalActions.
  final next = await arc.step(session, action: session.legalActions.first);
  print(next.observation);
  await arc.closeSession(next);
} finally {
  arc.close();
}
```

The example selects the first legal action only to demonstrate transport; it is **not** a reasoning policy. A real policy must decide which legal action to take and pass that choice to `step`.

## Protocol v1

- `GET /health`: protocol version and service status.
- `GET /games`: official toolkit game identifiers.
- `POST /sessions`: `{"game_id":"…","seed":0,"offline":false}`.
- `POST /sessions/{session_id}/step`: `{"action":"ACTION…","data":null,"reasoning":null}`.
- `DELETE /sessions/{session_id}`: close a session.

Each session response includes the current observation, current legal action names, step count, terminal state, and remaining step budget. The server revalidates actions against the environment before executing them. Session IDs are unguessable, requests are size-limited, and the default bridge binds only to loopback.

## Limits

This is a transport and event-integration bridge, not proof of autonomous ARC solving. With `--mind-socket`, ARC observations/actions enter Genesis through the existing OuterWorld event path, which is wired to self-state and cognitive-topos updates. This does not make the policy a solver, and it does not directly translate ARC records into language or guarantee learned semantic alignment. The Python service also needs the optional official ARC toolkit and any required API access to list/open live games.
