//! Dump the field-probe signature — what the daemon serves on
//! GET_FIELD_PROBE. Runs the probe for a few seconds so the rings
//! warm, then decodes the v3 body: band energies, carrier lock
//! fields, the electrical channels, and the Maxwell estimates.
//!
//! `cargo run --example field_probe_dump`

use genesis::daemon::field_probe::{self, FIELD_BANDS};

fn f32s(body: &[u8], off: usize) -> f32 {
    body.get(off..off + 4)
        .and_then(|s| s.try_into().ok())
        .map(f32::from_le_bytes)
        .unwrap_or(f32::NAN)
}

fn main() {
    field_probe::start(|| "wake");
    std::thread::sleep(std::time::Duration::from_secs(5));
    let Some(probe) = field_probe::shared() else {
        eprintln!("probe did not start");
        std::process::exit(1);
    };
    let sig = probe.signature();
    println!("version {} · {} bytes", sig[0], sig.len());
    let names = [
        "eps", "del", "the", "alp", "sig", "b1", "b2", "b3", "gam", "lam",
    ];
    print!("thru  ");
    for (i, f) in names.iter().enumerate() {
        print!("{f}:{:.3}@{:.1}Hz  ", f32s(&sig, 1 + i * 4), FIELD_BANDS[i]);
    }
    println!();
    print!("jitt  ");
    for i in 0..10 {
        print!("{:.3}  ", f32s(&sig, 41 + i * 4));
    }
    println!();
    print!("powr  ");
    for i in 0..10 {
        print!("{:.3}  ", f32s(&sig, 114 + i * 4));
    }
    println!();
    println!(
        "thru_mean {:.0} it/s · jcv {:.2} · pstate {:.1}/s · irq {:.0}/s",
        f32s(&sig, 81),
        f32s(&sig, 85),
        f32s(&sig, 89),
        f32s(&sig, 93)
    );
    println!(
        "carrier {:.2} Hz · ok {} · lag {:+.3} rad · phase {:.3} · coh {:.3}",
        f32s(&sig, 97),
        sig[101],
        f32s(&sig, 102),
        f32s(&sig, 106),
        f32s(&sig, 110)
    );
    let mask = sig[178];
    println!(
        "power {:.2} W · sensor {:.1} Hz · rail {:.3} V · amps {:.3} A · mask {:03b}",
        f32s(&sig, 154),
        f32s(&sig, 158),
        f32s(&sig, 162),
        f32s(&sig, 166),
        mask
    );
    println!(
        "power-vs-carrier lag {:+.3} rad · coherence {:.3}",
        f32s(&sig, 170),
        f32s(&sig, 174)
    );
    println!(
        "maxwell: carrier ΔI {:.2} A · B(0.3m) {:.2e} µT · E_ind {:.2e} V/m · P_rad {:.2e} W",
        f32s(&sig, 179),
        f32s(&sig, 183),
        f32s(&sig, 187),
        f32s(&sig, 191)
    );
}
