"""Tests for genesis_client.threads — named_thread + comm_name."""

import threading

import pytest

from genesis_client.threads import call_on_named_thread, comm_name, named_thread


def test_named_thread_sets_python_and_comm_name():
    seen = {}

    def target():
        seen["py"] = threading.current_thread().name
        seen["comm"] = comm_name()

    t = named_thread("organ-test", target, comm="g-organ-test")
    t.start()
    t.join(timeout=5.0)

    assert not t.is_alive()
    assert seen["py"] == "organ-test"
    assert seen["comm"] == "g-organ-test"


def test_named_thread_defaults_daemon_and_truncates_comm():
    seen = {}

    t = named_thread("a-very-long-thread-name", lambda: seen.update(comm=comm_name()))
    assert t.daemon is True
    t.start()
    t.join(timeout=5.0)

    # No explicit comm — the truncated Python name is used (≤15 bytes).
    assert seen["comm"] == "a-very-long-thr"


def test_named_thread_passes_args_and_kwargs():
    seen = {}

    def target(a, b, *, c):
        seen["total"] = a + b + c

    t = named_thread("args-test", target, args=(1, 2), kwargs={"c": 3}, comm="g-args")
    t.start()
    t.join(timeout=5.0)

    assert seen["total"] == 6


def test_named_thread_target_exception_does_not_hide_naming():
    # A dying target still carries the name — that's the point of the
    # diagnostic: the stack dump shows which organ it was.
    def target():
        raise RuntimeError("boom")

    t = named_thread("doomed", target, comm="g-doomed")
    t.start()
    t.join(timeout=5.0)
    assert not t.is_alive()


def test_comm_name_of_main_thread():
    # The main thread's comm name is the process name — sanity check
    # that the prctl GET path works at all.
    assert comm_name() != ""


def test_call_on_named_thread_returns_result_and_names():
    seen = {}

    def open_thing():
        seen["comm"] = comm_name()
        seen["py"] = threading.current_thread().name
        return 42

    result = call_on_named_thread("io-open", open_thing, comm="g-io-open")

    assert result == 42
    assert seen["comm"] == "g-io-open"
    assert seen["py"] == "io-open"


def test_call_on_named_thread_reraises_errors():
    def broken():
        raise RuntimeError("device gone")

    with pytest.raises(RuntimeError, match="device gone"):
        call_on_named_thread("io-broken", broken, comm="g-io-broken")


def test_call_on_named_thread_timeout():
    import time

    with pytest.raises(TimeoutError):
        call_on_named_thread("io-stuck", lambda: time.sleep(2.0), comm="g-io-stuck", timeout=0.2)
