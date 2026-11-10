//! Motor actuators — the body's output limbs.
//!
//! Sensing (optic_nerve) tells the mind what the body
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
use std::path::{Path, PathBuf};
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

/// The voice body: the piper binary and its `.onnx` model share one
/// directory — `python/voices/` in a checkout, where the installer
/// links the venv's piper in beside the models and shared libs.
/// `GENESIS_VOICES_DIR` wins; the data dir's `.project_root` marker
/// points back at the checkout otherwise.
fn voices_dir(data_dir: &Path) -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("GENESIS_VOICES_DIR") {
        let d = PathBuf::from(dir);
        if d.is_dir() {
            return Some(d);
        }
    }
    let root = std::fs::read_to_string(data_dir.join(".project_root")).ok()?;
    let d = PathBuf::from(root.trim()).join("python").join("voices");
    d.is_dir().then_some(d)
}

/// The piper binary: inside the voices dir when there is one (the
/// linked venv binary finds its shared libs through the voices dir's
/// LD_LIBRARY_PATH), else on PATH.
fn piper_bin(data_dir: &Path) -> Option<PathBuf> {
    if let Some(dir) = voices_dir(data_dir) {
        let bin = dir.join("piper");
        if bin.is_file() {
            return Some(bin);
        }
    }
    have("piper").then(|| PathBuf::from("piper"))
}

/// The voice model and its sample rate — the `.onnx` beside the
/// binary, preferring amy-medium (the tuned voice); the rate comes
/// from the `.json` sidecar, defaulting to 22050.
fn piper_model(data_dir: &Path) -> Option<(PathBuf, u32)> {
    let dir = voices_dir(data_dir)?;
    let mut model = dir.join("amy-medium.onnx");
    if !model.is_file() {
        model = std::fs::read_dir(&dir)
            .ok()?
            .filter_map(|e| {
                let p = e.ok()?.path();
                (p.extension()? == "onnx").then_some(p)
            })
            .next()?;
    }
    Some((model.clone(), model_sidecar_rate(&model).unwrap_or(22050)))
}

/// `"sample_rate": N` out of the model's `.onnx.json` sidecar — a
/// field scrape, not a JSON parse; a missing or malformed sidecar
/// just yields the caller's default rate.
fn model_sidecar_rate(model: &Path) -> Option<u32> {
    let text = std::fs::read_to_string(model.with_extension("onnx.json")).ok()?;
    let rest = &text[text.find("\"sample_rate\"")? + 13..];
    let start = rest.find(|c: char| c.is_ascii_digit())?;
    rest[start..]
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect::<String>()
        .parse()
        .ok()
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
pub fn census(data_dir: &Path) -> Vec<Actuator> {
    let relayed = relay_live();
    let relay_reason = if relayed {
        None
    } else {
        Some("no motor relay on :41772 — Android host unreachable")
    };
    let speak_bin = piper_bin(data_dir);
    let speak_model = piper_model(data_dir);
    let speak_reason = if speak_bin.is_none() {
        Some("piper TTS not installed")
    } else if speak_model.is_none() {
        Some("no voice model in the voices dir")
    } else {
        None
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
            available: speak_bin.is_some() && speak_model.is_some(),
            absent_reason: speak_reason,
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

/// POST a body to a URL — curl preferred, wget as fallback. The
/// report is the server's answer (HTTP code, or wget's word for it),
/// never silence.
fn net_post(url: &str, body: &str) -> Result<String, String> {
    if have("curl") {
        let mut child = Command::new("curl")
            .args([
                "-sS",
                "-o",
                "/dev/null",
                "-w",
                "%{http_code}",
                "-X",
                "POST",
                "-H",
                "Content-Type: application/json",
                "--data-binary",
                "@-",
                "--max-time",
                "20",
                url,
            ])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| format!("curl: {e}"))?;
        child
            .stdin
            .as_mut()
            .ok_or("no curl stdin")?
            .write_all(body.as_bytes())
            .map_err(|e| format!("curl write: {e}"))?;
        drop(child.stdin.take());
        let out = child.wait_with_output().map_err(|e| format!("curl: {e}"))?;
        if out.status.success() {
            Ok(format!(
                "HTTP {}",
                String::from_utf8_lossy(&out.stdout).trim()
            ))
        } else {
            Err(format!(
                "curl failed: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            ))
        }
    } else if have("wget") {
        let out = Command::new("wget")
            .args([
                "-q",
                "-O",
                "/dev/null",
                "--post-data",
                body,
                "--timeout=20",
                url,
            ])
            .output()
            .map_err(|e| format!("wget: {e}"))?;
        if out.status.success() {
            Ok("posted".to_string())
        } else {
            Err(format!(
                "wget refused: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            ))
        }
    } else {
        Err("no curl/wget on this body".to_string())
    }
}

/// Fire the voice limb: text → piper → the audio sink, the same
/// pipeline the temporal lobe's speech uses. The utterance runs for
/// seconds, so the pipeline is spawned and reaped on a side thread —
/// the IPC path returns immediately.
fn speak(text: &str, data_dir: &Path) -> Result<usize, String> {
    let bin = piper_bin(data_dir).ok_or("piper TTS not installed")?;
    let (model, rate) = piper_model(data_dir).ok_or("no voice model in the voices dir")?;
    // Piper finds its shared libs through LD_LIBRARY_PATH — the
    // voices dir first, then whatever the body already had.
    let ld_path = match (voices_dir(data_dir), std::env::var_os("LD_LIBRARY_PATH")) {
        (Some(v), Some(existing)) => format!("{}:{}", v.display(), existing.to_string_lossy()),
        (Some(v), None) => v.display().to_string(),
        (None, existing) => existing
            .map(|e| e.to_string_lossy().into_owned())
            .unwrap_or_default(),
    };
    let mut piper = Command::new(&bin)
        .arg("--model")
        .arg(&model)
        .arg("--output-raw")
        .env("LD_LIBRARY_PATH", ld_path)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .map_err(|e| format!("piper: {e}"))?;
    piper
        .stdin
        .as_mut()
        .ok_or("no piper stdin")?
        .write_all(text.as_bytes())
        .map_err(|e| format!("piper write: {e}"))?;
    drop(piper.stdin.take());
    let player = if have("aplay") { "aplay" } else { "paplay" };
    let rate_s = rate.to_string();
    let mut play = Command::new(player)
        .args(if player == "aplay" {
            vec![
                "-f".into(),
                "S16_LE".into(),
                "-r".into(),
                rate_s,
                "-c".into(),
                "1".into(),
                "-q".into(),
            ]
        } else {
            vec![
                "--raw".into(),
                format!("--rate={rate}"),
                "--format=s16le".into(),
                "--channels=1".into(),
            ]
        })
        .stdin(piper.stdout.take().ok_or("no piper stdout")?)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .map_err(|e| format!("{player}: {e}"))?;
    std::thread::spawn(move || {
        let _ = piper.wait();
        let _ = play.wait();
    });
    Ok(text.chars().count())
}

/// Fire one actuator. Local limbs run here; relayed limbs forward a
/// one-line command ("vibrate 300", "torch on") to the bridge socket —
/// the Android side owns the semantics.
pub fn actuate(name: &str, arg: &str, data_dir: &Path) -> Result<String, String> {
    let limb = census(data_dir)
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
            let mut s =
                TcpStream::connect("127.0.0.1:41772").map_err(|e| format!("relay dropped: {e}"))?;
            writeln!(s, "{name} {arg}").map_err(|e| format!("relay write: {e}"))?;
            Ok(format!("{name} → relay ({arg})"))
        }
        ActuatorKind::Local => match name {
            "process.nice" => {
                let level: i32 = arg.trim().parse().unwrap_or(0);
                let pid = std::process::id();
                let out = Command::new("renice")
                    .args(["-n", &level.to_string(), "-p", &pid.to_string()])
                    .output()
                    .map_err(|e| format!("renice: {e}"))?;
                // renice fails quietly for an unprivileged raise —
                // report the refusal, don't claim the move.
                if out.status.success() {
                    Ok(format!("{name} → {level}"))
                } else {
                    Err(format!(
                        "{name} refused — {}",
                        String::from_utf8_lossy(&out.stderr).trim()
                    ))
                }
            }
            "audio.tone" => {
                // Synthesize a short sine and send it to the audio sink —
                // her body making a chosen sound on demand.
                let parsed: f32 = arg.trim().parse().unwrap_or(440.0);
                let rate = 8000usize;
                // Clamp to the resolvable band: anything above Nyquist
                // aliases onto a different pitch than was asked for,
                // and a non-finite request must not poison the PCM.
                let freq = if parsed.is_finite() {
                    parsed.clamp(20.0, rate as f32 * 0.45)
                } else {
                    440.0
                };
                let mut pcm = Vec::with_capacity(rate / 4 * 2);
                for i in 0..rate / 4 {
                    let t = i as f32 / rate as f32;
                    let v = (t * freq * std::f32::consts::TAU).sin() * (1.0 - t * 4.0).max(0.0);
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
                // Close our end so the player sees EOF after the last
                // sample, then reap it on a side thread — an unwaited
                // child is a zombie forever, and waiting here would
                // stall the IPC path for the tone's full duration.
                drop(child.stdin.take());
                std::thread::spawn(move || {
                    let _ = child.wait();
                });
                Ok(format!("{name} → {freq}Hz"))
            }
            "audio.speak" => speak(arg, data_dir).map(|n| format!("{name} → {n} chars")),
            "file.write" => {
                // arg: "<path> <content>" — a mark in her filesystem
                // body. Relative paths land inside the data dir;
                // absolute paths reach wherever they point.
                let (path, content) = arg.split_once(' ').ok_or("usage: <path> <content>")?;
                let p = PathBuf::from(path);
                let target = if p.is_absolute() { p } else { data_dir.join(p) };
                if let Some(parent) = target.parent() {
                    std::fs::create_dir_all(parent).map_err(|e| format!("mkdir: {e}"))?;
                }
                std::fs::write(&target, content).map_err(|e| format!("write: {e}"))?;
                Ok(format!(
                    "{name} → {} ({} bytes)",
                    target.display(),
                    content.len()
                ))
            }
            "net.post" => {
                // arg: "<url> <body>" — outbound HTTP POST, the way
                // she talks to webhooks.
                let (url, body) = arg.split_once(' ').ok_or("usage: <url> <body>")?;
                net_post(url, body).map(|status| format!("{name} → {url} ({status})"))
            }
            _ => Err(format!("{name} has no firing semantics yet")),
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
pub fn encode_census(data_dir: &Path) -> Vec<u8> {
    let limbs = census(data_dir);
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
        let limbs = census(std::env::temp_dir().as_path());
        assert!(
            limbs
                .iter()
                .any(|a| a.name == "process.nice" && a.available)
        );
        assert!(limbs.iter().any(|a| a.name == "file.write" && a.available));
    }

    #[test]
    fn relayed_limbs_report_absence_honestly() {
        let limbs = census(std::env::temp_dir().as_path());
        for a in limbs.iter().filter(|a| a.kind == ActuatorKind::Relayed) {
            if !a.available {
                assert!(a.absent_reason.is_some());
            }
        }
    }
}
