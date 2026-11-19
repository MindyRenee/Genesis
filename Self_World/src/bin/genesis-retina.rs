//! genesis-retina — the eye as its own organ, and the only one.
//!
//! The retina opens the V4L2 capture device itself and publishes raw
//! frames into the POSIX shared-memory segment `/genesis_optic_nerve`.
//! It is the sole writer of the nerve: the daemon carries no capture
//! limb of its own and only reads, so the retina is the eye's only
//! connection — no retina means no eye.
//!
//! The CLI launches the retina before the daemon. On SIGINT/SIGTERM
//! the limb winds down and the segment's name is unlinked, so a dead
//! eye never publishes stale frames under a live name.

use std::sync::atomic::{AtomicBool, Ordering};

use genesis::daemon::optic_nerve::SHM_NAME;
use genesis::daemon::optic_nerve::capture_limb;

/// Set by the signal handler when the eye should close.
static SHOULD_EXIT: AtomicBool = AtomicBool::new(false);

/// Set by the limb while it holds a segment it created — the exit
/// path only unlinks a name this process actually owns, never a
/// nerve another publisher claimed.
static OWNS_NERVE: AtomicBool = AtomicBool::new(false);

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
    let shm_name = std::env::var("GENESIS_OPTIC_SEGMENT").unwrap_or_else(|_| SHM_NAME.to_string());

    capture_limb(&shm_name, &SHOULD_EXIT, &OWNS_NERVE);

    // Take the name back on the way out — but only if the limb ever
    // owned a segment. A retina that stood down to a live publisher
    // or died before creating one must not unlink another eye's
    // nerve. (In practice the owner `Shmem`'s drop already unlinks
    // before `capture_limb` returns, so this fires only as the
    // belt-and-suspenders path.)
    if OWNS_NERVE.load(Ordering::Relaxed) {
        if let Ok(name) = std::ffi::CString::new(shm_name.as_str()) {
            // SAFETY: `name` is a valid NUL-terminated CString;
            // shm_unlink of a missing name returns ENOENT, ignored.
            unsafe {
                let _ = libc::shm_unlink(name.as_ptr());
            }
        }
    }
}
