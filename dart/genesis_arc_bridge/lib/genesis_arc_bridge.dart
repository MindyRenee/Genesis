import 'dart:convert';
import 'dart:io';

/// A bounded view of the official ARC-AGI-3 environment state.
class ArcSession {
  const ArcSession({
    required this.sessionId,
    required this.gameId,
    required this.step,
    required this.observation,
    required this.legalActions,
    required this.terminalState,
    required this.stepBudgetRemaining,
    required this.closed,
  });

  final String sessionId;
  final String gameId;
  final int step;
  final Object? observation;
  final List<String> legalActions;
  final String? terminalState;
  final int stepBudgetRemaining;
  final bool closed;

  bool get isTerminal => terminalState == 'WIN' || terminalState == 'GAME_OVER';

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
    );
  }
}

class ArcGame {
  const ArcGame({required this.gameId, required this.title, this.tags = const []});
  final String gameId;
  final String title;
  final List<String> tags;

  factory ArcGame.fromJson(Map<String, dynamic> json) => ArcGame(
        gameId: _requiredString(json, 'game_id'),
        title: json['title'] is String ? json['title'] as String : '',
        tags: json['tags'] is List
            ? List<String>.unmodifiable((json['tags'] as List).whereType<String>())
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

  Future<bool> health() async {
    final response = await _request('GET', '/health');
    return response['ok'] == true && response['protocol'] == 1;
  }

  Future<List<ArcGame>> listGames() async {
    final response = await _request('GET', '/games');
    final games = response['games'];
    if (games is! List) throw const FormatException('games must be a list');
    return games
        .whereType<Map<String, dynamic>>()
        .map(ArcGame.fromJson)
        .toList(growable: false);
  }

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

  Future<ArcSession> step(
    ArcSession session, {
    required String action,
    Map<String, Object?>? data,
    Map<String, Object?>? reasoning,
  }) async {
    if (session.isTerminal) throw StateError('Cannot step a terminal session');
    if (session.closed) throw StateError('Cannot step a closed session');
    if (!session.legalActions.contains(action)) {
      throw ArgumentError.value(action, 'action', 'not in the current legal action set');
    }
    final response = await _request(
      'POST',
      '/sessions/${Uri.encodeComponent(session.sessionId)}/step',
      {'action': action, 'data': data, 'reasoning': reasoning},
    );
    return ArcSession.fromJson(response);
  }

  Future<void> closeSession(ArcSession session) async {
    await _request('DELETE', '/sessions/${Uri.encodeComponent(session.sessionId)}');
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
    final body = await utf8.decoder.bind(response).join().timeout(requestTimeout);
    final decoded = body.isEmpty ? <String, dynamic>{} : jsonDecode(body);
    if (decoded is! Map<String, dynamic>) {
      throw const FormatException('server response must be an object');
    }
    if (response.statusCode < 200 || response.statusCode >= 300) {
      final message = decoded['message'] ?? decoded['error'] ?? 'HTTP ${response.statusCode}';
      throw HttpException('$message', uri: uri);
    }
    return decoded;
  }

  void close() => _http.close(force: true);
}
