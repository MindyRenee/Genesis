//! Minimal RealtimeKit client over the system bus — just enough
//! D-Bus to request and *hold* a scheduling grant.
//!
//! rtkit revokes an elevation when the bus client that requested it
//! disappears: a one-shot `dbus-send` is rolled back seconds after
//! it exits (measured on this host: the grant to a probe thread was
//! gone within ~45 s while the identical request from a client that
//! stayed connected held indefinitely). PulseAudio's grants survive
//! because it keeps a persistent connection — this module does the
//! same, speaking the wire protocol directly so the daemon grows no
//! dependencies: SASL EXTERNAL auth, the required Hello, one
//! METHOD_CALL, then the socket stays open for as long as the
//! caller keeps the returned stream.
//!
//! Everything here is best-effort: an absent bus, a refused auth,
//! or a denied grant all surface as `Err` and the caller continues
//! at default scheduling — degraded jitter, never a failure.

use std::io::{self, Read, Write};
use std::os::unix::net::UnixStream;
use std::time::Duration;

const SYSTEM_BUS_PATH: &str = "/run/dbus/system_bus_socket";

/// Levels tried strictest-first — rtkit clamps or denies a request
/// below its advertised MinNiceLevel, so walking down covers daemons
/// configured tighter than -15.
const NICE_LEVELS: [i32; 3] = [-15, -10, -5];

fn align_up(n: usize, a: usize) -> usize {
    (n + a - 1) & !(a - 1)
}

fn pad(buf: &mut Vec<u8>, align: usize) {
    while buf.len() % align != 0 {
        buf.push(0);
    }
}

/// `"s"`/`"o"` wire form: u32 length + bytes + NUL.
fn enc_str(s: &str) -> Vec<u8> {
    let mut v = Vec::with_capacity(s.len() + 5);
    v.extend_from_slice(&(s.len() as u32).to_le_bytes());
    v.extend_from_slice(s.as_bytes());
    v.push(0);
    v
}

/// `"g"` wire form: u8 length + bytes + NUL.
fn enc_sig(s: &str) -> Vec<u8> {
    let mut v = Vec::with_capacity(s.len() + 2);
    v.push(s.len() as u8);
    v.extend_from_slice(s.as_bytes());
    v.push(0);
    v
}

/// One header field — an 8-aligned struct of a byte code and a
/// variant. The `fields` buffer begins at absolute offset 16 (itself
/// 8-aligned), so buffer-relative padding is wire-correct.
fn marshal_field(buf: &mut Vec<u8>, code: u8, sig: &str, value: &[u8], val_align: usize) {
    pad(buf, 8);
    buf.push(code);
    buf.push(sig.len() as u8);
    buf.extend_from_slice(sig.as_bytes());
    buf.push(0);
    pad(buf, val_align);
    buf.extend_from_slice(value);
}

/// METHOD_CALL in little-endian framing. Field codes: 1 PATH,
/// 2 INTERFACE, 3 MEMBER, 6 DESTINATION, 8 SIGNATURE. The bus
/// fills in SENDER; we never set it.
fn marshal_call(
    serial: u32,
    path: &str,
    iface: &str,
    member: &str,
    dest: &str,
    sig: &str,
    body: &[u8],
) -> Vec<u8> {
    let mut fields = Vec::new();
    marshal_field(&mut fields, 1, "o", &enc_str(path), 4);
    marshal_field(&mut fields, 2, "s", &enc_str(iface), 4);
    marshal_field(&mut fields, 3, "s", &enc_str(member), 4);
    marshal_field(&mut fields, 6, "s", &enc_str(dest), 4);
    if !sig.is_empty() {
        marshal_field(&mut fields, 8, "g", &enc_sig(sig), 1);
    }

    let mut m = Vec::with_capacity(24 + fields.len() + body.len());
    m.extend_from_slice(b"l\x01\x00\x01");
    m.extend_from_slice(&(body.len() as u32).to_le_bytes());
    m.extend_from_slice(&serial.to_le_bytes());
    m.extend_from_slice(&(fields.len() as u32).to_le_bytes());
    pad(&mut m, 8);
    m.extend_from_slice(&fields);
    pad(&mut m, 8);
    m.extend_from_slice(body);
    m
}

struct Msg {
    mtype: u8,
    reply_serial: Option<u32>,
    error_name: Option<String>,
}

/// Read one full message: 16-byte fixed header, the header-fields
/// array, then the body. Only the pieces we need are decoded —
/// type, REPLY_SERIAL (code 5, "u"), and ERROR_NAME (code 4, "s").
fn read_message(s: &mut UnixStream) -> io::Result<Msg> {
    let debug = std::env::var_os("GENESIS_DBUS_DEBUG").is_some();
    let mut fixed = [0u8; 16];
    s.read_exact(&mut fixed)?;
    // Fixed header: 0 endianness, 1 type, 2 flags, 3 version,
    // 4-7 body length, 8-11 serial, 12-15 header-fields array len.
    let body_len = u32::from_le_bytes([fixed[4], fixed[5], fixed[6], fixed[7]]) as usize;
    let fields_len = u32::from_le_bytes([fixed[12], fixed[13], fixed[14], fixed[15]]) as usize;
    if debug {
        eprintln!(
            "[dbus] msg type={} blen={} flen={} hdr={:02x?}",
            fixed[1], body_len, fields_len, fixed
        );
    }
    if fields_len > (1 << 20) || body_len > (1 << 20) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("oversized dbus frame: hdr={:02x?}", fixed),
        ));
    }
    let mut rest = vec![0u8; align_up(fields_len, 8) + body_len];
    s.read_exact(&mut rest)?;
    let fields = &rest[..fields_len];

    let mut reply_serial = None;
    let mut error_name = None;
    let mut i = 0usize;
    while i + 2 <= fields.len() {
        let code = fields[i];
        i += 1;
        let sl = fields[i] as usize;
        i += 1;
        if i + sl + 1 > fields.len() {
            break;
        }
        let vsig = &fields[i..i + sl];
        i += sl + 1;
        match vsig {
            b"o" | b"s" => {
                i = align_up(i, 4);
                if i + 4 > fields.len() {
                    break;
                }
                let l = u32::from_le_bytes([fields[i], fields[i + 1], fields[i + 2], fields[i + 3]])
                    as usize;
                let start = i + 4;
                if start + l + 1 > fields.len() {
                    break;
                }
                if code == 4 {
                    error_name =
                        Some(String::from_utf8_lossy(&fields[start..start + l]).into_owned());
                }
                i = start + l + 1;
            }
            b"g" => {
                if i >= fields.len() {
                    break;
                }
                let l = fields[i] as usize;
                i += 1 + l + 1;
            }
            b"u" => {
                i = align_up(i, 4);
                if i + 4 > fields.len() {
                    break;
                }
                if code == 5 {
                    reply_serial = Some(u32::from_le_bytes([
                        fields[i],
                        fields[i + 1],
                        fields[i + 2],
                        fields[i + 3],
                    ]));
                }
                i += 4;
            }
            _ => break,
        }
        while i % 8 != 0 && i < fields.len() {
            i += 1;
        }
    }
    Ok(Msg {
        mtype: fixed[1],
        reply_serial,
        error_name,
    })
}

enum Reply {
    Return,
    Error(String),
}

/// A D-Bus method target — everything marshal_call needs that isn't
/// the serial or the body.
struct CallSpec<'a> {
    path: &'a str,
    iface: &'a str,
    member: &'a str,
    dest: &'a str,
    sig: &'a str,
    body: Vec<u8>,
}

/// Send one METHOD_CALL and read until its reply, skipping signals
/// (e.g. the bus's own NameAcquired for our unique name).
fn dbus_call(s: &mut UnixStream, serial: &mut u32, c: &CallSpec) -> io::Result<Reply> {
    *serial += 1;
    let mine = *serial;
    s.write_all(&marshal_call(
        mine, c.path, c.iface, c.member, c.dest, c.sig, &c.body,
    ))?;
    loop {
        let m = read_message(s)?;
        match (m.mtype, m.reply_serial) {
            (2, Some(rs)) if rs == mine => return Ok(Reply::Return),
            (3, Some(rs)) if rs == mine => {
                return Ok(Reply::Error(
                    m.error_name.unwrap_or_else(|| "dbus error".into()),
                ));
            }
            _ => {}
        }
    }
}

/// SASL EXTERNAL over the unix socket: the one leading NUL byte,
/// then the uid's decimal string hex-encoded.
fn sasl_auth(s: &mut UnixStream) -> io::Result<()> {
    // SAFETY: getuid takes no arguments, touches no memory, and
    // cannot fail — it always returns the caller's real uid.
    let uid = unsafe { libc::getuid() };
    let hex: String = uid
        .to_string()
        .bytes()
        .map(|b| format!("{b:02x}"))
        .collect();
    s.write_all(b"\0")?;
    s.write_all(format!("AUTH EXTERNAL {hex}\r\n").as_bytes())?;
    let mut line = Vec::with_capacity(64);
    let mut byte = [0u8; 1];
    while line.len() < 4096 {
        s.read_exact(&mut byte)?;
        if byte[0] == b'\n' {
            break;
        }
        line.push(byte[0]);
    }
    let line = String::from_utf8_lossy(&line);
    if !line.starts_with("OK") {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            format!("system bus auth: {}", line.trim()),
        ));
    }
    s.write_all(b"BEGIN\r\n")
}

fn connect() -> io::Result<UnixStream> {
    let addr = std::env::var("DBUS_SYSTEM_BUS_ADDRESS").unwrap_or_default();
    let path = addr.strip_prefix("unix:path=").unwrap_or(SYSTEM_BUS_PATH);
    let s = UnixStream::connect(path)?;
    s.set_read_timeout(Some(Duration::from_secs(10)))?;
    Ok(s)
}

/// Ask rtkit for high priority (nice) on `tid` of `pid`. Returns
/// the granted level *and the open socket* — the grant is owned by
/// this bus client, so the caller must keep the stream alive for
/// as long as the elevation should hold (rtkit revokes on client
/// disconnect). `Err` covers every failure: no bus, no rtkit, or a
/// denial at all levels — the caller continues unboosted.
pub fn grant_high_priority(pid: u64, tid: u64) -> io::Result<(i32, UnixStream)> {
    let mut s = connect()?;
    sasl_auth(&mut s)?;
    let mut serial = 0u32;
    // Hello is mandatory before any real call — it registers our
    // unique name, the identity rtkit watches.
    match dbus_call(
        &mut s,
        &mut serial,
        &CallSpec {
            path: "/org/freedesktop/DBus",
            iface: "org.freedesktop.DBus",
            member: "Hello",
            dest: "org.freedesktop.DBus",
            sig: "",
            body: Vec::new(),
        },
    )? {
        Reply::Return => {}
        Reply::Error(e) => {
            return Err(io::Error::other(format!("Hello refused: {e}")));
        }
    }
    let mut denied = String::new();
    for nice in NICE_LEVELS {
        // Signature "tti": u64 process, u64 thread, i32 level — each
        // aligned to its own size within the body.
        let mut body = Vec::with_capacity(24);
        pad(&mut body, 8);
        body.extend_from_slice(&pid.to_le_bytes());
        pad(&mut body, 8);
        body.extend_from_slice(&tid.to_le_bytes());
        pad(&mut body, 4);
        body.extend_from_slice(&nice.to_le_bytes());
        match dbus_call(
            &mut s,
            &mut serial,
            &CallSpec {
                path: "/org/freedesktop/RealtimeKit1",
                iface: "org.freedesktop.RealtimeKit1",
                member: "MakeThreadHighPriorityWithPID",
                dest: "org.freedesktop.RealtimeKit1",
                sig: "tti",
                body,
            },
        )? {
            Reply::Return => return Ok((nice, s)),
            Reply::Error(e) => denied = e,
        }
    }
    Err(io::Error::new(io::ErrorKind::PermissionDenied, denied))
}
