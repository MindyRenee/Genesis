import 'package:genesis_arc_bridge/genesis_arc_bridge.dart';
import 'package:test/test.dart';

void main() {
  group('ArcSession', () {
    test('parses a nonterminal response', () {
      final session = ArcSession.fromJson({
        'session_id': 'session-1',
        'game_id': 'game-1',
        'step': 2,
        'observation': {'state': 'PLAY'},
        'legal_actions': ['ACTION1', 'ACTION2'],
        'terminal_state': null,
        'closed': false,
        'step_budget_remaining': 98,
      });
      expect(session.step, 2);
      expect(session.legalActions, ['ACTION1', 'ACTION2']);
      expect(session.isTerminal, isFalse);
    });

    test('detects terminal state and rejects malformed legal actions', () {
      final session = ArcSession.fromJson({
        'session_id': 'session-1',
        'game_id': 'game-1',
        'step': 1,
        'observation': {},
        'legal_actions': [],
        'terminal_state': 'WIN',
        'closed': false,
        'step_budget_remaining': 99,
      });
      expect(session.isTerminal, isTrue);
      expect(
        () => ArcSession.fromJson({
          'session_id': 'session-1',
          'game_id': 'game-1',
          'step': 0,
          'observation': {},
          'legal_actions': ['ACTION1', 2],
          'terminal_state': null,
          'closed': false,
          'step_budget_remaining': 100,
        }),
        throwsFormatException,
      );
    });
  });

  test('parses game metadata', () {
    final game = ArcGame.fromJson({
      'game_id': 'game-1',
      'title': 'Demo',
      'tags': ['test'],
    });
    expect(game.gameId, 'game-1');
    expect(game.tags, ['test']);
  });

  test('parses board, level progress, and RESET after GAME_OVER', () {
    final session = ArcSession.fromJson({
      'session_id': 's',
      'game_id': 'g',
      'step': 3,
      'observation': null,
      'legal_actions': ['RESET'],
      'terminal_state': 'GAME_OVER',
      'step_budget_remaining': 10,
      'closed': false,
      'board': [
        [0, 1],
        [2, 3],
      ],
      'state': 'GAME_OVER',
      'levels_completed': 1,
      'win_levels': 7,
    });
    expect(session.isTerminal, isTrue);
    expect(session.isWon, isFalse);
    expect(session.legalActions, ['RESET']);
    expect(session.board, [
      [0, 1],
      [2, 3],
    ]);
    expect(session.levelsCompleted, 1);
    expect(session.winLevels, 7);
  });
}
