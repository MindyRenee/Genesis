"""RealtimeKit — word-layer companion to ``src/daemon/rtkit.rs``.

A minimal rtkit client over the system bus: just enough D-Bus to
request and *hold* a realtime scheduling grant for the daemon's
hot threads (the body's 200 ms tick cadence wants it). Disabled
on Android where no system bus or RT grant exists — priority becomes
a recommendation the mind applies to itself instead.
"""
