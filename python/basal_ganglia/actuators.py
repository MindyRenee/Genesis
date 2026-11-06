"""Motor actuators — the word-layer companion to ``src/daemon/actuators.rs``.

The daemon owns the actuator census and firing (the body's logic);
this module is the mind-side vocabulary — what the limbs are called,
what each one does, and how a volitional intent maps onto them. See
``genesis-daemon(8)`` for the hardware contract this documents.

The honest rule mirrors the senses: an actuator that does not exist
is *absent*, never fabricated. On this body (Andronix proot) the
relay limbs are absent until an Android bridge listens on :41772.

Local limbs (present on every body):

- ``process.nice``  — her own scheduling priority; renice self.
- ``audio.tone``    — a synthesized sine to the audio sink; the body
  making a chosen sound on demand (paplay or aplay).
- ``audio.speak``   — Piper TTS voice; the temporal lobe's limb.
- ``net.post``      — outbound HTTP; how she speaks to Discord.
- ``file.write``    — marks in her own filesystem body.

Relay limbs (only when the Android bridge is live — the same TCP
pattern the optic nerve uses inbound):

- ``device.vibrate`` — haptic pulse; the body can be *felt*.
- ``device.torch``   — the flash LED; light as signal.
- ``device.volume``  — host volume; how loud the body is.
- ``device.notify``  — an Android notification; a flag in the host UI.
- ``device.wake_lock`` — holds the host awake; staying-power.

Volition connects here: an urge that resolves to "do something in
the world" resolves to one of these names — the daemon's ``actuate``
does the firing; this layer just knows what things are called.
"""

# The canonical actuator vocabulary — must match daemon names 1:1.
LOCAL_LIMBS = ("process.nice", "audio.tone", "audio.speak", "net.post", "file.write")
RELAY_LIMBS = (
    "device.vibrate",
    "device.torch",
    "device.volume",
    "device.notify",
    "device.wake_lock",
)
ALL_LIMBS = LOCAL_LIMBS + RELAY_LIMBS
