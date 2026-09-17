"""Exceptions for the Genesis client library.

# Hierarchy

    GenesisError(Exception)
    ├── ConnectionError(GenesisError, OSError)
    │   └── DaemonNotRunning(ConnectionError)
    ├── ProtocolError(GenesisError, OSError)
    └── EpisodeNotFound(GenesisError, LookupError)

Design principles:

- **I/O errors inherit from ``OSError``** so that the pervasive
  ``except (OSError, ConnectionError)`` clauses in the cognitive mind
  catch them.  The cognitive mind's background threads (inner life,
  learner, regulator) all use this pattern to tolerate daemon
  disconnects without crashing.

- **``DaemonNotRunning`` inherits from ``ConnectionError``** — it is
  the most fundamental connection failure (the socket doesn't exist
  because the daemon isn't running).  This lets ``genesis_cli.py``'s
  ``except (OSError, ConnectionError)`` around ``mind.start()`` catch
  it and print a clean error message instead of an unhandled
  traceback.

- **``EpisodeNotFound`` inherits from ``LookupError``** — the Python
  convention for "key/index not found" errors (the same base as
  ``KeyError`` and ``IndexError``).  This lets callers catch it with
  ``except LookupError`` if they want to handle all not-found
  conditions uniformly.

- **``ConnectionError`` shadows the built-in.**  This is deliberate:
  the custom ``ConnectionError`` inherits from ``OSError``, so
  ``except OSError`` catches both the custom and built-in variants.
  ``client.py`` aliases the import as ``GenesisConnectionError`` to
  keep the built-in ``ConnectionError`` name available for catching
  socket-level errors (``BrokenPipeError``, etc.) in the same
  ``except`` clause.
"""

__all__ = [
    "ConnectionError",
    "DaemonNotRunning",
    "EpisodeNotFound",
    "GenesisError",
    "ProtocolError",
]


class GenesisError(Exception):
    """Base exception for all Genesis client errors."""


class ConnectionError(GenesisError, OSError):
    """Failed to connect to the Genesis daemon.

    Inherits from both GenesisError and OSError so that existing
    `except (OSError, ConnectionError)` clauses in calling code
    catch this custom exception correctly.
    """


class ProtocolError(GenesisError, OSError):
    """Malformed message or unexpected response from the daemon.

    Inherits from both GenesisError and OSError so that existing
    ``except (OSError, ConnectionError)`` clauses in calling code
    catch protocol errors correctly — a malformed or error response
    from the daemon is an I/O-level failure, not a logic error the
    caller should have to handle separately.
    """


class EpisodeNotFound(GenesisError, LookupError):
    """The requested episode does not exist in LTM.

    Inherits from both GenesisError and LookupError so that callers
    can catch it with either ``except GenesisError`` (for all Genesis
    errors) or ``except LookupError`` (the Python convention for
    "not found" errors, shared with ``KeyError`` and ``IndexError``).
    """


class DaemonNotRunning(ConnectionError):
    """The daemon is not running or the socket doesn't exist.

    Inherits from ``ConnectionError`` (and transitively from
    ``OSError``) because it is the most fundamental connection
    failure — the socket file doesn't exist because the daemon
    process isn't running.  This lets ``except (OSError,
    ConnectionError)`` clauses catch it, which is critical for
    ``genesis_cli.py``'s startup error handling.
    """
