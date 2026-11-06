import 'dart:convert';
import 'dart:io';

/// A bounded view of the official ARC-AGI-3 environment state.
///
/// After `GAME_OVER` the only legal action is `RESET`, which restarts the
/// current level; `WIN` means every level is cleared and nothing more can
/// be played.
class ArcSession {
  /// Creates a session view; normally built with [ArcSession.fromJson].
  const ArcSession({
    required this.sessionId,
    required this.gameId,
    required this.step,
    required this.observation,
    required this.legalActions,
    required this.terminalState,
    required this.stepBudgetRemaining,
    required this.closed,
    this.board,
    this.state,
    this.levelsCompleted,
    this.winLevels,
    this.genesisReasoning,
  });

  /// Bridge-assigned session identifier.
  final String sessionId;

  /// Official ARC-AGI-3 game identifier.
  final String gameId;

  /// Number of actions submitted in this session, `RESET` included.
  final int step;

  /// The raw toolkit observation as JSON.
  final Object? observation;

  /// Actions the server will accept next (`['RESET']` after `GAME_OVER`).
  final List<String> legalActions;

  /// `WIN` or `GAME_OVER` when the game is in a terminal state, else null.
  final String? terminalState;

  /// Actions left before the bridge's per-session budget is exhausted.
  final int stepBudgetRemaining;

  /// Whether the session has been closed.
  final bool closed;

  /// The visible 64x64 board (rows of colours 0..15), if the server sent one.
  final List<List<int>>? board;

  /// Toolkit game state: `NOT_PLAYED`, `NOT_FINISHED`, `WIN`, or `GAME_OVER`.
  final String? state;

  /// Levels cleared so far.
  final int? levelsCompleted;

  /// Levels needed to win the game.
  final int? winLevels;

  /// Genesis's own decision telemetry when she chose the move
  /// ([GenesisArcClient.genesisStep]); null for moves you chose.
  ///
  /// Typical keys: `mode` (e.g. `untested_here`, `walk_to_frontier`,
  /// `procedural_replay`), `dopamine_rpe`, `striatal_disinhibition`,
  /// `gaba_inhibited`, `masked_border_lines`.
  final Map<String, Object?>? genesisReasoning;

  /// Whether the game is over or won (a `GAME_OVER` can still be RESET).
  bool get isTerminal => terminalState == 'WIN' || terminalState == 'GAME_OVER';

  /// Whether every level has been cleared.
  bool get isWon => terminalState == 'WIN';

  /// Parses a bridge session response.
  ///
  /// Throws [FormatException] if required fields are missing or malformed.

  factory ArcSession.fromJson(Map<String, dynamic> json) {
    final rawActions = json['legal_actions'];
    if (rawActions is! List || rawActions.any((item) => item is! String)) {
      throw const FormatException('legal_actions must be a list of strings');
    }
    return ArcSession(
      sessionId: _requiredString(json, 'session_id'),
      gameId: _requiredString(json, 'game_id'),
      step: _requiredInt(json, 'step'),
      observation: json['observation'],
      legalActions: List<String>.unmodifiable(rawActions.cast<String>()),
      terminalState: json['terminal_state'] as String?,
      stepBudgetRemaining: _requiredInt(json, 'step_budget_remaining'),
      closed: json['closed'] == true,
      board: _optionalBoard(json['board']),
      state: json['state'] as String?,
      levelsCompleted: json['levels_completed'] as int?,
      winLevels: json['win_levels'] as int?,
      genesisReasoning: _optionalObject(json['genesis_reasoning']),
    );
  }
}

/// An ARC-AGI-3 game the bridge can open.
class ArcGame {
  /// Creates a game descriptor.
  const ArcGame(
      {required this.gameId, required this.title, this.tags = const []});

  /// Official game identifier.
  final String gameId;

  /// Human-readable title, or empty.
  final String title;

  /// Toolkit tags for the game.
  final List<String> tags;

  /// Parses one entry of the `/games` response.
  factory ArcGame.fromJson(Map<String, dynamic> json) => ArcGame(
        gameId: _requiredString(json, 'game_id'),
        title: json['title'] is String ? json['title'] as String : '',
        tags: json['tags'] is List
            ? List<String>.unmodifiable(
                (json['tags'] as List).whereType<String>())
            : const [],
      );
}

String _requiredString(Map<String, dynamic> json, String key) {
  final value = json[key];
  if (value is! String || value.isEmpty) {
    throw FormatException('$key must be a non-empty string');
  }
  return value;
}

int _requiredInt(Map<String, dynamic> json, String key) {
  final value = json[key];
  if (value is! int || value < 0) {
    throw FormatException('$key must be a non-negative integer');
  }
  return value;
}

/// Dart-side transport only. Action selection remains the responsibility of
/// Genesis' policy; the server validates each action against the live legal set.
class GenesisArcClient {
  GenesisArcClient({
    Uri? baseUri,
    HttpClient? httpClient,
    this.requestTimeout = const Duration(seconds: 15),
  })  : baseUri = baseUri ?? Uri.parse('http://127.0.0.1:8765'),
        _http = httpClient ?? HttpClient();

  final Uri baseUri;
  final HttpClient _http;
  final Duration requestTimeout;

  /// Returns true when the bridge is up and speaks protocol 1.
  Future<bool> health() async {
    final response = await _request('GET', '/health');
    return response['ok'] == true && response['protocol'] == 1;
  }

  /// Lists the games available through the bridge.
  Future<List<ArcGame>> listGames() async {
    final response = await _request('GET', '/games');
    final games = response['games'];
    if (games is! List) throw const FormatException('games must be a list');
    return games
        .whereType<Map<String, dynamic>>()
        .map(ArcGame.fromJson)
        .toList(growable: false);
  }

  /// Opens a new session for [gameId].
  Future<ArcSession> openGame(
    String gameId, {
    int seed = 0,
    bool offline = false,
  }) async {
    if (gameId.trim().isEmpty) throw ArgumentError.value(gameId, 'gameId');
    final response = await _request('POST', '/sessions', {
      'game_id': gameId,
      'seed': seed,
      'offline': offline,
    });
    return ArcSession.fromJson(response);
  }

  /// Submits [action] and returns the next session view.
  ///
  /// `ACTION6` is a click and needs [data] `{'x': column, 'y': row}` with
  /// integers 0..63. After `GAME_OVER` only `RESET` is accepted.
  Future<ArcSession> step(
    ArcSession session, {
    required String action,
    Map<String, Object?>? data,
    Map<String, Object?>? reasoning,
  }) async {
    if (session.isWon) throw StateError('Cannot step a won session');
    if (session.closed) throw StateError('Cannot step a closed session');
    if (!session.legalActions.contains(action)) {
      throw ArgumentError.value(
          action, 'action', 'not in the current legal action set');
    }
    if (action == 'ACTION6' && !_validClick(data)) {
      throw ArgumentError.value(
          data, 'data', 'ACTION6 needs integer x and y in 0..63');
    }
    final response = await _request(
      'POST',
      '/sessions/${Uri.encodeComponent(session.sessionId)}/step',
      {'action': action, 'data': data, 'reasoning': reasoning},
    );
    return ArcSession.fromJson(response);
  }

  /// Lets Genesis choose and play the next move of [session] herself.
  ///
  /// She decides from the board pixels only, keeps her state graph for the
  /// whole session, and answers `GAME_OVER` with `RESET`. Read her reasons
  /// from [ArcSession.genesisReasoning] on the returned session.
  Future<ArcSession> genesisStep(ArcSession session) async {
    if (session.isWon) throw StateError('Cannot step a won session');
    if (session.closed) throw StateError('Cannot step a closed session');
    final response = await _request(
      'POST',
      '/sessions/${Uri.encodeComponent(session.sessionId)}/genesis',
      <String, Object?>{},
    );
    return ArcSession.fromJson(response);
  }

  /// Closes [session] on the bridge.
  Future<void> closeSession(ArcSession session) async {
    await _request(
        'DELETE', '/sessions/${Uri.encodeComponent(session.sessionId)}');
  }

  Future<Map<String, dynamic>> _request(
    String method,
    String path, [
    Map<String, Object?>? payload,
  ]) async {
    final uri = baseUri.resolve(path);
    final request = await _http.openUrl(method, uri).timeout(requestTimeout);
    request.headers.contentType = ContentType.json;
    request.headers.set(HttpHeaders.acceptHeader, 'application/json');
    if (payload != null) request.write(jsonEncode(payload));
    final response = await request.close().timeout(requestTimeout);
    final body =
        await utf8.decoder.bind(response).join().timeout(requestTimeout);
    final decoded = body.isEmpty ? <String, dynamic>{} : jsonDecode(body);
    if (decoded is! Map<String, dynamic>) {
      throw const FormatException('server response must be an object');
    }
    if (response.statusCode < 200 || response.statusCode >= 300) {
      final message = decoded['message'] ??
          decoded['error'] ??
          'HTTP ${response.statusCode}';
      throw HttpException('$message', uri: uri);
    }
    return decoded;
  }

  /// Closes the underlying HTTP client.
  void close() => _http.close(force: true);
}

bool _validClick(Map<String, Object?>? data) {
  bool coordinate(Object? value) => value is int && value >= 0 && value <= 63;
  return data != null && coordinate(data['x']) && coordinate(data['y']);
}

Map<String, Object?>? _optionalObject(Object? value) {
  if (value == null) return null;
  if (value is! Map<String, dynamic>) {
    throw const FormatException('genesis_reasoning must be an object');
  }
  return Map<String, Object?>.unmodifiable(value);
}

List<List<int>>? _optionalBoard(Object? value) {
  if (value == null) return null;
  if (value is! List ||
      value.any((row) => row is! List || row.any((c) => c is! int))) {
    throw const FormatException('board must be a list of integer rows');
  }
  return List<List<int>>.unmodifiable(
    value.map((row) => List<int>.unmodifiable((row as List).cast<int>())),
  );
}
