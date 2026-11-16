//! genesis-retina — the eye as its own organ.
//!
//! The retina opens the V4L2 capture device itself and publishes raw
//! frames into the POSIX shared-memory segment `/genesis_optic_nerve`
//! — the same segment the daemon's embedded capture limb writes when
//! no retina is running. One capture implementation, two carriers:
//! the daemon runs it on a thread (`OpticCapture`), this binary runs
//! it on the main thread of a dedicated process.
//!
//! The CLI launches the retina before the daemon; ownership is
//! settled on the nerve itself — the daemon's own optic reader sees
//! these frames arriving and keeps its embedded limb closed, and if
//! this process dies the daemon takes the camera back. Exactly one
//! process owns the device at a time. On SIGINT/SIGTERM the limb
//! winds down and the segment's name is unlinked, so a dead eye
//! never publishes stale frames under a live name.

use std::ffi::CString;
use std::sync::atomic::{AtomicBool, Ordering};

use genesis::daemon::optic_nerve::SHM_NAME;
use genesis::daemon::optic_nerve::capture_limb;

/// Set by the signal handler when the eye should close.
static SHOULD_EXIT: AtomicBool = AtomicBool::new(false);

/// Signal handler — flags the capture limb out and returns. Only an
/// atomic store runs inside a signal context; unlinking happens in
/// `main` after the limb returns.
extern "C" fn signal_handler(_sig: libc::c_int) {
    SHOULD_EXIT.store(true, Ordering::Relaxed);
}

fn main() {
    // SAFETY: signal handlers are atomic flag setters — no
    // non-reentrant work. The cast is valid because `signal_handler`
    // matches the `sighandler_t` signature.
    unsafe {
        libc::signal(
            libc::SIGTERM,
            signal_handler as *const () as libc::sighandler_t,
        );
        libc::signal(
            libc::SIGINT,
            signal_handler as *const () as libc::sighandler_t,
        );
    }

    // The segment to publish — the main optic nerve by default; a
    // named segment (tests, a second eye) overrides.
    let shm_name =
        std::env::var("GENESIS_OPTIC_SEGMENT").unwrap_or_else(|_| SHM_NAME.to_string());

    capture_limb(&shm_name, &SHOULD_EXIT);

    // Take the name back on the way out: readers already holding the
    // mapping keep it until they detach, but nothing new attaches to
    // a dead eye. ENOENT when the limb never got as far as creating
    // the segment.
    if let Ok(name) = CString::new(shm_name.as_str()) {
        // SAFETY: `name` is a valid NUL-terminated CString;
        // shm_unlink of a missing name returns ENOENT, ignored.
        unsafe {
            let _ = libc::shm_unlink(name.as_ptr());
        }
    }
}
