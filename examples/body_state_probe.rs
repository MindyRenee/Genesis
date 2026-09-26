//! Diagnostic probe — read the machine-native body state twice and
//! print the v3 timing/involuntary/senescence fields.
//!
//! The rate-based fields (pulse_hz, top_freq_share) need two reads
//! to establish a delta window, so this sleeps briefly between them.
//!
//!   cargo run --example body_state_probe

use genesis::daemon::interoception::Interoceptor;

fn main() {
    let mut intero = Interoceptor::new();
    // First read primes the delta counters (LOC count, time_in_state).
    let _ = intero.read();
    std::thread::sleep(std::time::Duration::from_millis(600));
    let body = intero.read();

    println!("─── body state (second read) ───");
    println!("cpu_temp_c      {:>8.1} °C", body.cpu_temp_c);
    println!("arousal_freq    {:>8.3} (frac of max)", body.arousal_freq);
    println!("metabolic_rate  {:>8.3} (power, normalized)", body.metabolic_rate);
    println!("core_voltage    {:>8.2} V", body.core_voltage);
    println!("supply_voltage  {:>8.2} V", body.supply_voltage);
    println!("autonomic_rate  {:>8.2} GPE/s", body.autonomic_rate);
    println!();
    println!("─── timing ───");
    println!("pulse_hz        {:>8.0} beats/s", body.pulse_hz);
    println!("pulse           {:>8.3} [0-1]", body.pulse);
    println!("clocksource     {:>8} (0=? 1=tsc 2=hpet 3=acpi_pm 4=other)", body.clocksource);
    println!("suspend_caps    {:>8} (bit0=mem bit1=wakealarm)", body.suspend_caps);
    println!();
    println!("─── involuntary ───");
    println!("throttle_state  {:>8.3} [0-1]", body.throttle_state);
    println!("top_freq_share  {:>8.3} of window at max P-state", body.top_freq_share);
    println!("psi_cpu         {:>8.3}", body.psi_cpu);
    println!("psi_io          {:>8.3}", body.psi_io);
    println!("psi_mem         {:>8.3}", body.psi_mem);
    println!();
    println!("─── senescence ───");
    println!("battery_cycles  {:>8.0}", body.battery_cycles);
    println!("entropy_level   {:>8.3} [0-1]", body.entropy_level);
    println!();
    println!("distressed      {:>8}", body.distressed);
}
