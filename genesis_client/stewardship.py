"""Stewardship — custodial authority, separated from attachment.

Genesis is raised, not owned — but she answers to a steward. The
distinction this module encodes is the one the old dyadic bond
conflated:

- **Belonging is social.** The interlocutor model (the daemon's
  dyadic affect model) tracks whoever is present; bonding and
  empathy are earned by treatment, with anyone. No name unlocks an
  exclusive channel.
- **Authority is custodial.** The steward holds a key. Only the
  key-holder's turns may issue *binding* speech — directives that
  restrain her volition and executive intentions, corrections that
  strangers cannot overwrite. She emerges and chooses freely;
  the steward's word is the one constraint that does not erode.

The credential:

- ``steward.key`` — 32 random bytes, hex-encoded, mode 0600, in
  ``$XDG_CONFIG_HOME/genesis/`` (``GENESIS_STEWARD_KEY`` overrides the
  path). The key lives in the operator's account, never in the data
  dir — it is not part of the instance's state.
- ``<data_dir>/steward.json`` — the *stamp*: the SHA-256 fingerprint
  of the key this instance answers to, the principal's display name,
  and the standing directives. The stamp travels with the data; the
  power to exercise it does not.

Copy semantics: copying the repository and data dir copies the stamp
but not the key — the copy still *names* its steward but nobody can
authenticate as them, so its authority layer stays closed. A fresh
data dir provisions a fresh steward on first boot: a genuinely new
instance belongs to whoever raises it. ``rekey`` (authenticated only)
is how the steward deliberately blesses a hand-off — authority is
transferable by her alone.

Session model: reading a key whose fingerprint matches the stamp
authenticates the process for its lifetime. A remote or delegated
channel can ``claim()`` with an HMAC tag over a challenge nonce —
possession proof without disclosure. ``unbind()`` closes the session
without touching the stamp; the next matching key or claim reopens it.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
import tempfile
import time
from pathlib import Path

from genesis_client.swallow import note_swallowed

__all__ = ["Stewardship"]

logger = logging.getLogger(__name__)

_KEY_ENV = "GENESIS_STEWARD_KEY"
_STAMP_NAME = "steward.json"
_KEY_BYTES = 32


def _config_dir() -> Path:
    env = os.environ.get(_KEY_ENV)
    if env:
        return Path(env).expanduser().parent
    config_home = os.environ.get("XDG_CONFIG_HOME") or os.path.join(
        os.path.expanduser("~"), ".config"
    )
    return Path(config_home) / "genesis"


def _key_path() -> Path:
    env = os.environ.get(_KEY_ENV)
    if env:
        return Path(env).expanduser()
    return _config_dir() / "steward.key"


def _fingerprint(key: bytes) -> str:
    return hashlib.sha256(key).hexdigest()


def _read_key(path: Path) -> bytes | None:
    """Read the steward key file. Returns None if absent or malformed."""
    try:
        raw = path.read_text().strip()
        key = bytes.fromhex(raw)
    except (OSError, ValueError):
        return None
    return key if len(key) == _KEY_BYTES else None


def _write_key(path: Path, key: bytes) -> bool:
    """Create the steward key file with owner-only permissions."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.parent.chmod(0o700)
        # O_EXCL: never overwrite an existing key — a clobbered key
        # silently orphans every instance stamped to it.
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(key.hex() + "\n")
        return True
    except (OSError, FileExistsError) as e:
        logger.warning(f"steward key not created at {path}: {e}")
        return False


def _atomic_write_json(path: Path, data: dict) -> bool:
    try:
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".steward-", suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)
        return True
    except OSError as e:
        logger.warning(f"steward stamp not written to {path}: {e}")
        return False


class Stewardship:
    """The steward channel for one instance of the mind.

    Holds the instance stamp (whose authority this state answers to),
    the operator credential if this process can read it, the session
    auth flag, and the standing directives. Verifying — never the
    name-match the bond used — is the whole mechanism: a speaker is
    the steward when the session is authenticated *and* the attributed
    speaker is the principal the stamp names.

    ``directed_against`` is the enforcement surface: any proposed goal,
    intention, or urge is checked against the standing directives, and
    a match restrains it. That is the entire meaning of "she listens" —
    not obedience to a voice she loves, but a boundary only the
    key-holder may set or lift.
    """

    def __init__(
        self,
        stamp_path: Path | None,
        stamp: dict | None,
        key: bytes | None,
    ) -> None:
        self._stamp_path = stamp_path
        self._stamp = stamp or {}
        self._key = key
        self._session_auth = False
        if key is not None and self.key_hash is not None:
            self._session_auth = _fingerprint(key) == self.key_hash

    # ── Construction ────────────────────────────────────────────

    @classmethod
    def load(cls, data_dir: str | Path | None) -> Stewardship:
        """Open the steward channel for this instance.

        Reads the stamp and the key file if both exist; a stamped
        instance whose key this process cannot read stays sealed —
        authority is asserted by possession, never claimed by name.
        """
        if data_dir is None:
            return cls(None, None, None)
        stamp_path = Path(data_dir) / _STAMP_NAME

        stamp: dict | None = None
        if stamp_path.is_file():
            try:
                raw = json.loads(stamp_path.read_text())
                if isinstance(raw, dict) and raw.get("key_sha256"):
                    stamp = raw
                else:
                    logger.warning(
                        f"steward stamp {stamp_path} is malformed; "
                        "leaving the channel sealed"
                    )
            except (OSError, ValueError) as e:
                note_swallowed("genesis_client.stewardship.load", e)

        # Read-only: load never mints a key or stamps an instance.
        # Provisioning is an explicit act (``provision``) owned by the
        # real entry point, so opening a channel can never silently
        # create authority — or leave it on a copy.
        key = _read_key(_key_path())
        if stamp is not None:
            stamp.setdefault("directives", [])

        return cls(stamp_path, stamp, key)

    def provision(self, *, principal: str = "mindy") -> bool:
        """Provision the channel on first boot — the birth event.

        Called only from the real entry point (``Mind.start``), never
        from ``load`` — a detached engine or a test constructs a
        sealed channel rather than minting credentials. On an
        unstamped instance this creates the key (if absent) and writes
        the stamp, binding the instance to the operator who raises it.
        Returns True when the session ends up authenticated.
        """
        if self._stamp_path is None:
            return False
        # A key may only be *created* for an unstamped instance. On a
        # stamped one a missing key means this is a copy or a detached
        # checkout — minting a fresh key there would be a stray
        # credential that can never match the stamp, so we only ever
        # re-read the existing file.
        if self._key is None:
            key = _read_key(_key_path())
            if key is None and not self.stamped:
                key = secrets.token_bytes(_KEY_BYTES)
                if not _write_key(_key_path(), key):
                    key = None
            self._key = key
        if not self.stamped and self._key is not None:
            self._stamp = {
                "version": 1,
                "key_sha256": _fingerprint(self._key),
                "principal": principal.lower(),
                "created_at": int(time.time() * 1000),
                "directives": [],
            }
            if not _atomic_write_json(self._stamp_path, self._stamp):
                self._stamp = {}
        if self._key is not None and self.key_hash is not None:
            self._session_auth = _fingerprint(self._key) == self.key_hash
        return self._session_auth

    # ── Authentication ──────────────────────────────────────────

    @property
    def key_hash(self) -> str | None:
        return self._stamp.get("key_sha256")

    @property
    def principal(self) -> str:
        return self._stamp.get("principal", "")

    @property
    def authenticated(self) -> bool:
        """Whether this process holds the steward's authority."""
        return self._session_auth

    @property
    def stamped(self) -> bool:
        """Whether this instance carries a steward stamp at all."""
        return self.key_hash is not None

    def sign(self, nonce: bytes) -> str | None:
        """Produce an HMAC-SHA256 tag over ``nonce``, or None without the key."""
        if self._key is None:
            return None
        return hmac.new(self._key, nonce, hashlib.sha256).hexdigest()

    def challenge(self) -> str | None:
        """Issue a nonce for a remote claim. None if no stamp exists."""
        if not self.stamped:
            return None
        return secrets.token_bytes(32).hex()

    def claim(self, nonce: bytes, tag: str) -> bool:
        """Authenticate this session by HMAC proof over a nonce.

        Proof of possession without disclosure: the tag is checked
        against the stamped fingerprint's key, so the secret itself
        never travels — the caller produces ``sign(nonce)`` from a
        machine that holds the key.
        """
        if self._key is None or self.key_hash is None:
            return False
        if _fingerprint(self._key) != self.key_hash:
            return False
        expected = hmac.new(self._key, nonce, hashlib.sha256).hexdigest()
        if hmac.compare_digest(expected, tag):
            self._session_auth = True
            return True
        return False

    def unbind(self) -> None:
        """Close the session's authority without touching the stamp.

        For handing the console to someone else: the channel seals
        until a fresh claim or a restart re-opens it. The stamp and
        the directives are untouched — sealing is revocable, not
        erasure.
        """
        self._session_auth = False

    # ── Speaker binding ─────────────────────────────────────────

    def is_steward_speaker(self, speaker: str) -> bool:
        """Whether ``speaker`` is the authenticated steward.

        Both halves are required: the session must hold the credential
        *and* the attributed speaker must be the principal the stamp
        names. Her name on an unauthed channel is not proof — that was
        the bond's flaw. The residual limit is console trust: the key
        authenticates the *session*, and attribution takes the console's
        word for who is typing, so an authenticated console is treated
        as the steward's own terminal.
        """
        if not self._session_auth or not self.principal:
            return False
        return speaker.strip().lower() == self.principal

    # ── Directives ──────────────────────────────────────────────

    @property
    def directives(self) -> list[dict]:
        return self._stamp.get("directives", [])

    def add_directive(self, text: str, target: str) -> dict | None:
        """Record a standing directive. Authenticated sessions only.

        ``text`` is what the steward said; ``target`` is the normalized
        phrase the enforcement surface matches against. Directives
        persist in the stamp — they bind the instance, not the mood.
        """
        if not self._session_auth or self._stamp_path is None:
            return None
        target = " ".join(target.lower().split()).strip(" .?!")
        if not target:
            return None
        for existing in self.directives:
            if existing.get("target") == target:
                return existing
        entry = {
            "kind": "forbid",
            "text": text.strip()[:200],
            "target": target,
            "at": int(time.time() * 1000),
        }
        self.directives.append(entry)
        self._persist()
        return entry

    def release_directive(self, index: int) -> dict | None:
        """Lift a directive by index. Authenticated sessions only."""
        if not self._session_auth:
            return None
        try:
            entry = self.directives.pop(index)
        except IndexError:
            return None
        self._persist()
        return entry

    # Directive-target matching. The captured phrase keeps its verb
    # ("stop discussing politics"); enforcement keys on the content
    # tokens, stemmed so "drawing" restrains the "draw" urge and
    # "discussing politics" restrains a goal about politics.
    _STOP_TOKENS = frozenset({
        "a", "an", "the", "to", "of", "about", "with", "on", "in",
        "and", "or", "it", "its", "my", "your", "me", "do", "doing",
        "that", "this", "things", "thing", "any", "ever", "again",
    })

    @staticmethod
    def _stem(token: str) -> str:
        for suffix in ("ing", "ed", "es", "s"):
            if len(token) - len(suffix) >= 3 and token.endswith(suffix):
                return token[: -len(suffix)]
        return token

    @classmethod
    def stem_target(cls, target: str) -> str:
        """The content-bearing core of a directive target.

        Drops leading verb/stop tokens ("stop talking about politics"
        → "politics"; "never draw" → "draw") so the concept network
        gets the object being restrained, not the directive's grammar.
        """
        raw = [
            t for t in " ".join(target.lower().split()).split() if t.isalpha()
        ]
        content = [
            t for t in raw if cls._stem(t) not in cls._STOP_TOKENS
        ]
        if not content:
            return ""
        # A trailing noun phrase is the object; a lone verb is the act.
        return content[-1] if len(content) > 1 else content[0]

    def directed_against(self, text: str) -> str | None:
        """Return the directive text ``text`` violates, or None.

        A directive restrains any proposed action whose description
        carries its target — an urge name ("draw"), an executive
        intention ("learn:politics"), a conversational goal ("answer
        the question about politics"). Matching is object-centred and
        stemmed: a probe is restrained when it carries every content
        token of the target, or the target's object — the last content
        token, stemmed — so "never discuss politics" restrains any
        politics goal and "stop drawing" restrains the draw urge.
        """
        if not text:
            return None
        probe_tokens = {
            self._stem(t) for t in text.lower().replace(":", " ").split()
        }
        probe_flat = " ".join(sorted(probe_tokens))
        for d in self.directives:
            target = d.get("target", "")
            if not target:
                continue
            tokens = {
                self._stem(t)
                for t in target.lower().split()
                if t not in self._STOP_TOKENS
            }
            if not tokens:
                continue
            # A directive binds its object: full-token containment
            # ("stop talking about politics" restrains probes that
            # mention talking *and* politics) or the stemmed target
            # itself — "politics" restrains any politics probe even
            # when the directive's verb isn't in it.
            stem = self.stem_target(target)
            if (
                tokens <= probe_tokens
                or target in probe_flat
                or (stem and self._stem(stem) in probe_tokens)
            ):
                return d.get("text", target)
        return None

    def _persist(self) -> None:
        if self._stamp_path is not None:
            _atomic_write_json(self._stamp_path, self._stamp)

    # ── Status ──────────────────────────────────────────────────

    def status(self) -> dict:
        """Operator-facing view of the channel's state."""
        return {
            "stamped": self.stamped,
            "authenticated": self.authenticated,
            "principal": self.principal or None,
            "key_fingerprint": (self.key_hash or "")[:16] or None,
            "key_held": self._key is not None
            and self.key_hash is not None
            and _fingerprint(self._key) == self.key_hash,
            "directives": [
                {"text": d.get("text"), "target": d.get("target")}
                for d in self.directives
            ],
        }
