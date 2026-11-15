use std::ffi::CString;
use std::process;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;
use std::time::Duration;

use v4l::Device;
use v4l::buffer::Type;
use v4l::format::FourCC;
use v4l::io::mmap::Stream as MmapStream;
use v4l::io::traits::CaptureStream;
use v4l::video::Capture;

const HEADER_SIZE: usize = 28;
const SHM_NAME: &str = "/genesis_retina";

/// Set to true by the signal handler when the process should exit.
static SHOULD_EXIT: AtomicBool = AtomicBool::new(false);

/// Signal handler — sets the exit flag and returns.
extern "C" fn signal_handler(_sig: libc::c_int) {
    SHOULD_EXIT.store(true, Ordering::Relaxed);
}

fn main() {
    // Register signal handlers so SIGTERM/SIGINT break the capture
    // loop cleanly, allowing us to shm_unlink the shared memory
    // object before exiting. Without this, the POSIX shared memory
    // object /genesis_retina would persist in /dev/shm after the
    // process is killed (it's only cleaned up on the next run's
    // pre-creation shm_unlink).
    // SAFETY: signal handlers are simple atomic flag setters — no
    // non-reentrant operations. The handler pointer cast is valid
    // because signal_handler matches the sighandler_t signature.
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

    // Get camera device from environment variable or default to /dev/video0
    let camera_path = std::env::var("GENESIS_CAMERA").unwrap_or_else(|_| "/dev/video0".to_string());
    let camera_index = camera_path
        .strip_prefix("/dev/video")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(0);

    // Open the V4L2 capture device. EBUSY is transient — a previous
    // retina can still be releasing the device through a restart
    // overlap — so it gets a short retry window before giving up.
    let mut dev = None;
    for _ in 0..10 {
        match Device::new(camera_index) {
            Ok(d) => {
                dev = Some(d);
                break;
            }
            Err(e) if e.raw_os_error() == Some(libc::EBUSY) => {
                thread::sleep(Duration::from_millis(200));
            }
            Err(e) => {
                eprintln!("retina: cannot open {camera_path}: {e}");
                process::exit(1);
            }
        }
    }
    let dev = match dev {
        Some(d) => d,
        None => {
            eprintln!("retina: cannot open {camera_path}: device stayed busy");
            process::exit(1);
        }
    };

    // Request an uncompressed RGB24 format. Fall back to YUYV if RGB3 is not available.
    let mut fmt = match dev.format() {
        Ok(f) => f,
        Err(e) => {
            eprintln!("retina: cannot read current format: {e}");
            process::exit(1);
        }
    };
    fmt.width = 640;
    fmt.height = 480;

    // Try RGB24 first, then the camera's native uncompressed
    // fallback. A driver rejects an unsupported fourcc with EINVAL —
    // move on to the next candidate — and answers EBUSY while a
    // previous handle still owns the device, which gets the same
    // brief retry window as open.
    let mut negotiated = None;
    'fourcc: for cand in [b"RGB3", b"YUYV"] {
        fmt.fourcc = FourCC::new(cand);
        for _ in 0..10 {
            match dev.set_format(&fmt) {
                Ok(f) if f.fourcc == FourCC::new(cand) => {
                    negotiated = Some(f);
                    break 'fourcc;
                }
                // Driver accepted the request but negotiated a
                // different fourcc — try the next candidate.
                Ok(_) => continue 'fourcc,
                Err(e) if e.raw_os_error() == Some(libc::EBUSY) => {
                    thread::sleep(Duration::from_millis(200));
                }
                Err(_) => continue 'fourcc,
            }
        }
    }
    let fmt = match negotiated {
        Some(f) => f,
        None => {
            eprintln!("retina: no usable format on {camera_path}");
            process::exit(1);
        }
    };

    println!(
        "retina: {}x{} fourcc={:?} size={} stride={}",
        fmt.width,
        fmt.height,
        fmt.fourcc.str().unwrap_or("????"),
        fmt.size,
        fmt.stride
    );

    // Create a memory-mapped, zero-copy V4L2 stream.
    let mut stream = match MmapStream::with_buffers(&dev, Type::VideoCapture, 4) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("retina: cannot create mmap stream: {e}");
            process::exit(1);
        }
    };

    // Drop any stale shared memory segment so the create below will succeed.
    let cname = CString::new(SHM_NAME).expect("valid shm name");
    // SAFETY: `cname` is a valid NUL-terminated CString whose pointer
    // remains valid for the duration of the call. `shm_unlink` is
    // safe to call even if the shared memory object does not exist
    // (it returns an error which we intentionally discard).
    unsafe {
        let _ = libc::shm_unlink(cname.as_ptr());
    }

    let shm_size = HEADER_SIZE + fmt.size as usize;
    let shmem: &'static mut shared_memory::Shmem = match shared_memory::ShmemConf::new()
        .os_id(SHM_NAME)
        .size(shm_size)
        .create()
    {
        Ok(mut s) => {
            // Keep the POSIX object alive after this process exits so Python can attach.
            s.set_owner(false);
            Box::leak(Box::new(s))
        }
        Err(e) => {
            eprintln!("retina: cannot create shared memory {SHM_NAME}: {e}");
            process::exit(1);
        }
    };

    {
        let base = shmem.as_ptr();
        // SAFETY: `base` is a valid pointer to the shared memory region
        // of `shm_size` bytes obtained from `Shmem`, which guarantees the
        // mapping is valid and writable. The header fields (offsets 0–27)
        // and the width/height/fourcc/size/stride writes are all within
        // bounds (`shm_size = HEADER_SIZE + fmt.size`, and HEADER_SIZE ≥ 28).
        // `write_volatile` prevents the compiler from optimizing away the
        // stores, which is necessary because another process reads this
        // memory concurrently.
        unsafe {
            // Initialize the header once.
            // The frame counter is an AtomicU64 using a seqlock:
            // even = frame complete, odd = write in progress.
            let counter = &*(base as *const AtomicU64);
            counter.store(0, Ordering::Relaxed);
            core::ptr::write_volatile(base.add(8) as *mut u32, fmt.width);
            core::ptr::write_volatile(base.add(12) as *mut u32, fmt.height);
            core::ptr::copy_nonoverlapping(fmt.fourcc.repr.as_ptr(), base.add(16), 4);
            core::ptr::write_volatile(base.add(20) as *mut u32, fmt.size);
            core::ptr::write_volatile(base.add(24) as *mut u32, fmt.stride);
        }
    }

    println!(
        "retina: os_id={} size={} owner={}",
        shmem.get_os_id(),
        shmem.len(),
        shmem.is_owner()
    );
    println!(
        "retina: shared memory {SHM_NAME} ready, {} bytes",
        shmem.len()
    );

    loop {
        if SHOULD_EXIT.load(Ordering::Relaxed) {
            break;
        }

        let base = shmem.as_ptr();

        let (buf, meta) = match stream.next() {
            Ok(frame) => frame,
            Err(e) => {
                // A signal (SIGTERM/SIGINT) interrupts the blocking
                // dequeue — that's the exit path, not a capture fault.
                if SHOULD_EXIT.load(Ordering::Relaxed) {
                    break;
                }
                // EINTR from any other signal is transient; retry
                // without logging or delay.
                if e.raw_os_error() == Some(libc::EINTR) {
                    continue;
                }
                eprintln!("retina: capture error: {e}");
                // Back off to avoid a tight error spin that floods the
                // log and burns CPU when the camera is disconnected.
                // The error is typically persistent (device gone), so
                // we sleep before retrying. The exit flag is checked
                // at the top of the loop on the next iteration.
                thread::sleep(Duration::from_millis(500));
                continue;
            }
        };

        // Use bytesused from the V4L2 buffer metadata instead of
        // buf.len(). The buffer is a full mmap'd arena and may be
        // larger than the actual frame data (e.g., page-aligned
        // padding). bytesused tells us exactly how many bytes the
        // driver filled, which is what we need to copy. Without this,
        // padded frames would be silently dropped when buf.len()
        // exceeds fmt.size even though the actual frame fits.
        let frame_size = meta.bytesused as usize;
        if frame_size == 0 {
            // Driver reported zero bytes — skip this frame
            continue;
        }
        if frame_size > fmt.size as usize {
            eprintln!(
                "retina: frame size {frame_size} exceeds expected {}",
                fmt.size
            );
            continue;
        }

        // SAFETY: `base` is a valid pointer to the shared memory region
        // obtained from `Shmem`, mapped with `shm_size` bytes. The frame
        // counter at offset 0 is an `AtomicU64` initialized once during
        // setup and only accessed via atomic operations. The frame data
        // copy (`base.add(HEADER_SIZE)`) stays within bounds because
        // `frame_size` was checked against `fmt.size` above, and
        // `shm_size = HEADER_SIZE + fmt.size`. The seqlock protocol
        // ensures readers see consistent data.
        unsafe {
            let counter = &*(base as *const AtomicU64);

            // Seqlock write protocol:
            // 1. Increment counter to odd → signals "write in progress"
            //    Any reader seeing an odd counter will retry.
            // 2. Copy the frame data.
            // 3. Increment counter to even → signals "frame complete"
            //    The Release ordering ensures the frame copy is visible
            //    before the even counter is observed by the reader.
            let seq = counter.load(Ordering::Relaxed);
            counter.store(seq.wrapping_add(1), Ordering::Relaxed);
            std::sync::atomic::fence(Ordering::SeqCst);

            // Copy the raw frame into the shared buffer after the header.
            core::ptr::copy_nonoverlapping(buf.as_ptr(), base.add(HEADER_SIZE), frame_size);

            // Publish the completed frame.
            counter.store(seq.wrapping_add(2), Ordering::Release);
        }

        // Frame-rate throttling — cap at ~30 FPS to avoid burning CPU
        // when the camera delivers frames faster than the consumer
        // (the cognitive mind) needs them. The cognitive mind samples
        // at its own rate via the seqlock, so extra frames are wasted
        // work. 33ms ≈ 30 FPS.
        thread::sleep(Duration::from_millis(33));
    }

    // Clean up the POSIX shared memory object on graceful exit so it
    // doesn't persist in /dev/shm after the process stops. The Shmem
    // was intentionally leaked with set_owner(false) so Python can
    // attach; now that we're exiting, we unlink it. Python readers
    // that already have the mapping open will continue to work until
    // they close it (shm_unlink only removes the name, not existing
    // mappings).
    let cname = CString::new(SHM_NAME).expect("valid shm name");
    // SAFETY: `cname` is a valid NUL-terminated CString. `shm_unlink`
    // only removes the name; existing mappings remain valid. The
    // return value is intentionally discarded (ENOENT is fine).
    unsafe {
        let _ = libc::shm_unlink(cname.as_ptr());
    }
    eprintln!("retina: shared memory unlinked, exiting.");
}
