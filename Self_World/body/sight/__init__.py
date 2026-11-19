"""Sight — the eye and its afferent path.

``optic_nerve`` is the mind's end of the body's optic nerve: the
retina (``genesis-retina``, the camera's sole carrier) writes raw
frames into a shared-memory segment and this module wraps them
zero-copy for the LGN.
``optic_relay`` publishes an X display into a second nerve-style
segment — her screen eye. Processing begins downstream, in
``occipital_lobe``.
"""
