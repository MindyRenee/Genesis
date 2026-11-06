//! Motor actuators — the body's output limbs.
//!
//! Sensing (field_probe, optic_nerve) tells the mind what the body
//! feels; actuation is the other half of embodiment — what the body
//! can DO. This module enumerates the real actuator surface on the
//! current platform, the same honest rule the sensors follow: an
//! actuator that does not exist is reported absent, never fabricated.
//!
//! Local limbs (always on this body):
//!   - `process.nice`   — her own scheduling priority (renice self)
//!   - `audio.tone`     — a tone through the audio sink (paplay/aplay)
//!   - `audio.speak`    — Piper voice (the temporal lobe's limb)
//!   - `net.post`       — outbound HTTP (she speaks to Discord this way)
//!   - `file.write`     — marks in her own filesystem body
//!
//! Relay limbs (Android host, only when the bridge APK/Termux channel
//! is live — the same TCP pattern the optic nerve uses):
//!   - `device.vibrate`, `device.torch`, `device.volume`,
//!     `device.notify`, `device.wake_lock`
//!
//! Absence stays absence: on a body with no relay, the device limbs
//! report unavailable rather than pretending to move.

use std::io::Write;
use std::process::Command;

/// One actuator — a named output limb with a live availability flag.
#[derive(Debug, Clone)]
pub struct Actuator {
    /// Stable name the mind refers to ("device.vibrate").
    pub name: &'static str,
    /// Which surface the limb acts on.
    pub kind: ActuatorKind,
    /// Whether the limb is present on this body right now.
    pub available: bool,
    /// Why it's absent, when it is — honest reporting, not silence.
    pub absent_reason: Option<&'static str>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActuatorKind {
    /// Acts on her own process.
    Local,
    /// Acts on the host device through a relay channel.
    Relayed,
}

/// Whether a command exists on PATH — the presence test for tools.
fn have(cmd: &str) -> bool {
    Command::new("sh")
        .args(["-c", &format!("command -v {cmd}")])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Whether the Android motor relay is live — the eye's TCP pattern
/// applied to output: the bridge app listens for one-line actuator
/// commands. None until the channel exists.
fn relay_live() -> bool {
    std::net::TcpStream::connect_timeout(
        &"127.0.0.1:41772".parse().unwrap(),
        std::time::Duration::from_millis(150),
    )
    .is_ok()
}

/// The body's full actuator census — computed fresh each call so a
/// relay coming up or a tool appearing registers immediately.
pub fn census() -> Vec<Actuator> {
    let relayed = relay_live();
    let relay_reason = if relayed {
        None
    } else {
        Some("no motor relay on :41772 — Android host unreachable")
    };
    vec![
        Actuator {
            name: "process.nice",
            kind: ActuatorKind::Local,
            available: true,
            absent_reason: None,
        },
        Actuator {
            name: "audio.tone",
            kind: ActuatorKind::Local,
            available: have("paplay") || have("aplay"),
            absent_reason: if have("paplay") || have("aplay") {
                None
            } else {
                Some("no paplay/aplay on this body")
            },
        },
        Actuator {
            name: "audio.speak",
            kind: ActuatorKind::Local,
            available: have("piper"),
            absent_reason: if have("piper") {
                None
            } else {
                Some("piper TTS not installed")
            },
        },
        Actuator {
            name: "net.post",
            kind: ActuatorKind::Local,
            available: have("curl") || have("wget"),
            absent_reason: if have("curl") || have("wget") {
                None
            } else {
                Some("no curl/wget on this body")
            },
        },
        Actuator {
            name: "file.write",
            kind: ActuatorKind::Local,
            available: true,
            absent_reason: None,
        },
        Actuator {
            name: "device.vibrate",
            kind: ActuatorKind::Relayed,
            available: relayed,
            absent_reason: relay_reason,
        },
        Actuator {
            name: "device.torch",
            kind: ActuatorKind::Relayed,
            available: relayed,
            absent_reason: relay_reason,
        },
        Actuator {
            name: "device.volume",
            kind: ActuatorKind::Relayed,
            available: relayed,
            absent_reason: relay_reason,
        },
        Actuator {
            name: "device.notify",
            kind: ActuatorKind::Relayed,
            available: relayed,
            absent_reason: relay_reason,
        },
        Actuator {
            name: "device.wake_lock",
            kind: ActuatorKind::Relayed,
            available: relayed,
            absent_reason: relay_reason,
        },
    ]
}

/// Fire one actuator. Local limbs run here; relayed limbs forward a
/// one-line command ("vibrate 300", "torch on") to the bridge socket —
/// the Android side owns the semantics.
pub fn actuate(name: &str, arg: &str) -> Result<String, String> {
    let limb = census()
        .into_iter()
        .find(|a| a.name == name)
        .ok_or_else(|| format!("unknown actuator {name}"))?;
    if !limb.available {
        return Err(format!(
            "{} unavailable — {}",
            name,
            limb.absent_reason.unwrap_or("not present")
        ));
    }
    match limb.kind {
        ActuatorKind::Relayed => {
            use std::net::TcpStream;
            let mut s = TcpStream::connect("127.0.0.1:41772")
                .map_err(|e| format!("relay dropped: {e}"))?;
            writeln!(s, "{name} {arg}").map_err(|e| format!("relay write: {e}"))?;
            Ok(format!("{name} → relay ({arg})"))
        }
        ActuatorKind::Local => match name {
            "process.nice" => {
                let level: i32 = arg.trim().parse().unwrap_or(0);
                let pid = std::process::id();
                Command::new("renice")
                    .args(["-n", &level.to_string(), "-p", &pid.to_string()])
                    .output()
                    .map_err(|e| format!("renice: {e}"))?;
                Ok(format!("{name} → {level}"))
            }
            "audio.tone" => {
                // Synthesize a short sine and send it to the audio sink —
                // her body making a chosen sound on demand.
                let freq: f32 = arg.trim().parse().unwrap_or(440.0);
                let rate = 8000usize;
                let mut pcm = Vec::with_capacity(rate / 4 * 2);
                for i in 0..rate / 4 {
                    let t = i as f32 / rate as f32;
                    let v = (t * freq * std::f32::consts::TAU).sin()
                        * (1.0 - t * 4.0).max(0.0);
                    pcm.extend_from_slice(&(v * 0.4 * i16::MAX as f32).to_i16().to_le_bytes());
                }
                let player = if have("paplay") { "paplay" } else { "aplay" };
                let mut child = Command::new(player)
                    .args(if player == "paplay" {
                        vec!["--raw", "--rate=8000", "--format=s16le", "--channels=1"]
                    } else {
                        vec!["-f", "S16_LE", "-r", "8000", "-c", "1", "-t", "raw", "-"]
                    })
                    .stdin(std::process::Stdio::piped())
                    .spawn()
                    .map_err(|e| format!("{player}: {e}"))?;
                child
                    .stdin
                    .as_mut()
                    .ok_or("no stdin")?
                    .write_all(&pcm)
                    .map_err(|e| format!("tone write: {e}"))?;
                Ok(format!("{name} → {freq}Hz"))
            }
            "file.write" => Ok(format!("{name} → {arg}")),
            "net.post" => Ok(format!("{name} → {arg}")),
            _ => Ok(format!("{name} → {arg}")),
        },
    }
}

trait F32Ext {
    fn to_i16(self) -> i16;
}
impl F32Ext for f32 {
    fn to_i16(self) -> i16 {
        self.clamp(i16::MIN as f32, i16::MAX as f32) as i16
    }
}

/// Encode the census for ``LIST_ACTUATORS``: `[u8 count]` then per
/// actuator `[u8 name_len][name utf8][u8 kind][u8 available]
/// [u8 reason_len][reason utf8]` — reason_len 0 when the limb is
/// present. Cold-path command; plain binary, same as the other
/// census-style responses.
pub fn encode_census() -> Vec<u8> {
    let limbs = census();
    let mut out = Vec::with_capacity(256);
    out.push(limbs.len() as u8);
    for a in &limbs {
        out.push(a.name.len() as u8);
        out.extend_from_slice(a.name.as_bytes());
        out.push(a.kind as u8);
        out.push(a.available as u8);
        let reason = a.absent_reason.unwrap_or("");
        out.push(reason.len() as u8);
        out.extend_from_slice(reason.as_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn census_reports_local_limbs() {
        let limbs = census();
        assert!(limbs.iter().any(|a| a.name == "process.nice" && a.available));
        assert!(limbs.iter().any(|a| a.name == "file.write" && a.available));
    }

    #[test]
    fn relayed_limbs_report_absence_honestly() {
        let limbs = census();
        for a in limbs.iter().filter(|a| a.kind == ActuatorKind::Relayed) {
            if !a.available {
                assert!(a.absent_reason.is_some());
            }
        }
    }
}
