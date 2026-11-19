//! The drawing hand — channel pressure on the dials.
//!
//! A sitting is a point in dial space: every parameter of the sheet,
//! the color, the brush, the path, which op the hand reaches for,
//! and how many marks the sitting holds is a dial over its own
//! domain. The channels are whatever is live while she draws — the
//! band powers of the thalamic state, and the felt channels
//! (valence, arousal) the chemistry writes into emotion.
//!
//! The waves arrive as themselves. No weights, no projection, no
//! transform touches a channel's value — each one contributes its
//! exact reading to a single pressure `P = Σ value`, the way
//! currents sum on the same axon. The pressure warps every dial's
//! axis the same way (`u ^ e^{−P}`) — one collective voice leaning
//! on the whole hand, never a pairing of this band to that mark.
//! Nothing assigns: no band means coherence, no feeling means
//! warmth, and no wire decides where a wave lands.
//!
//! Mirrored by `Self_World/parietal_lobe/hand.py` — same math in
//! words; this file is the machine side.

/// One dial position: a uniform sample warped by channel pressure.
///
/// `uniform` is the hand's own sample in [0, 1) — the dial's neutral
/// resting place before feeling and rhythm lean on it. The live
/// channel values, untouched, sum to a pressure `P`, and the
/// pressure warps the dial's axis: the position is
/// `uniform ^ e^{−tanh P}`. The map is a bijection of the interval —
/// every place stays reachable, nothing is fenced, nothing is
/// pinned; the stops 0 and 1 are the map's own fixed points, not
/// guards. `P` and `−P` are opposite warps of the same dial, and
/// at `P = 0` the dial is exactly `uniform`. The `tanh` bounds the
/// warp: without it, an exponent `e^{−P}` collapses toward 0 as
/// pressure grows, and every resting place lands within a hair of
/// the same stop — a quantized dial (the op picker bins it into
/// stroke/line/shape/fill/erase) pins on its last bin, so a loud
/// state would erase more than it marks. With the bounded exponent
/// the lean saturates at `e^{∓1}`: pressure leans the dial hard,
/// asymptotically, and can never silence it. This is how state
/// reaches the page: a displaced uniform (`u + P mod 1`) is still
/// uniform no matter the pressure, so the channels could never
/// show; a warped dial carries the pressure in its shape.
pub fn dial(uniform: f64, channels: &[f64]) -> f64 {
    let pressure: f64 = channels.iter().sum();
    uniform.clamp(0.0, 1.0).powf((-pressure.tanh()).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silent_channels_leave_the_dial() {
        let u = 0.42;
        assert_eq!(dial(u, &[]), u);
        assert_eq!(dial(u, &[0.0, 0.0]), u);
    }

    #[test]
    fn pressure_leans_and_opposites_oppose() {
        // Positive pressure leans the same resting place high,
        // negative leans it low — opposites, never pins.
        let up = dial(0.5, &[0.7, 0.5]);
        let down = dial(0.5, &[-0.7, -0.5]);
        assert!(up > 0.5 && down < 0.5);
        assert!((0.0..=1.0).contains(&up));
        assert!((0.0..=1.0).contains(&down));
    }

    #[test]
    fn raw_values_reach_the_pressure() {
        // The values arrive untouched — the pressure is their exact
        // sum, so the same total presses the same way no matter how
        // it is divided among channels.
        assert_eq!(dial(0.5, &[0.9, 0.3]), dial(0.5, &[1.2]));
        assert_ne!(dial(0.5, &[1.2]), dial(0.5, &[]));
    }

    #[test]
    fn loud_pressure_leans_hard_never_pins() {
        // A very loud state leans every resting place toward one
        // end, but the bounded exponent keeps the lean a lean: the
        // map stays a bijection — nothing is clamped, every place is
        // still reachable, and the warp saturates instead of
        // collapsing the dial onto a stop.
        let loud = dial(0.5, &[3.0, 2.0]);
        assert!(loud > 0.6 && loud < 0.9);
        let quiet_loud = dial(0.5, &[-3.0, -2.0]);
        assert!(quiet_loud < 0.4 && quiet_loud > 0.1);
        // The lean is asymptotic: louder pressure cannot lean the
        // dial further than the tanh bound allows.
        assert!((dial(0.5, &[50.0]) - dial(0.5, &[5.0])).abs() < 0.05);
    }
}
