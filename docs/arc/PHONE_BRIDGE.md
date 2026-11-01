# Genesis ARC phone bridge

This package connects a Dart/Flutter phone client to the existing Python ARC-AGI-3 adapter. It does **not** embed Python in Dart and does not add an LLM. The official ARC toolkit remains in Python; Dart provides typed phone-side transport and can submit explicit actions.

## Run on the same Android phone

From the Genesis repository root in Termux:

```sh
python -m pip install -e '.[arc]'
PYTHONPATH=python:. python scripts/arc_phone_bridge.py --host 127.0.0.1 --port 8765
```

The service intentionally binds to loopback only and has no authentication. Do not change it to listen on a LAN/public interface. It stores active sessions in memory; restarting the service ends them.

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

This is a transport bridge, not proof of autonomous ARC solving. It does not automatically attach a live `Mind`/OuterWorld instance, the cognitive topos engine, telemetry, or the binary-language substrate. Those require a separate explicit integration and end-to-end tests. The Python service also needs the optional official ARC toolkit and any required API access to list/open live games.
