//! Number sense — the approximate number system (ANS) and subitizing.
//!
//! Machine-side counterpart to `python/parietal_lobe/math/number_sense.py`.
//! Encodes quantities on a Weber–Fechner logarithmic mental number line
//! (Dehaene 2003; Feigenson, Dehaene & Spelke 2004).

/// Largest quantity recognized instantly and exactly without counting.
pub const SUBITIZING_LIMIT: f64 = 4.0;

/// Adult Weber fraction governing approximate discriminability.
pub const WEBER_FRACTION: f64 = 0.15;

/// Ordering relation returned by [`compare`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MagnitudeRelation {
    Less,
    Equal,
    Greater,
}

/// Approximate comparison between two quantities.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MagnitudeJudgment {
    pub a: f64,
    pub b: f64,
    pub relation: MagnitudeRelation,
    pub confidence: f64,
    pub exact: bool,
}

/// Whether `n` falls inside the exact subitizing range {1, 2, 3, 4}.
pub fn subitizes(n: f64) -> bool {
    n.is_finite() && n > 0.0 && n <= SUBITIZING_LIMIT && n.fract() == 0.0
}

/// Logarithmic position on the mental number line (`ln n` for `n > 0`).
pub fn represent(n: f64) -> f64 {
    if !n.is_finite() || n <= 0.0 {
        0.0
    } else {
        n.ln()
    }
}

/// Approximate magnitude `(mean, sigma)` under scalar variability (Gibbon 1977).
pub fn estimate(n: f64) -> (f64, f64) {
    if subitizes(n) {
        (n, 0.0)
    } else {
        (n, (n * WEBER_FRACTION).max(0.5))
    }
}

/// Compare two magnitudes under subitizing + Weber–Fechner discriminability.
pub fn compare(a: f64, b: f64) -> MagnitudeJudgment {
    if subitizes(a) && subitizes(b) {
        let relation = if a < b {
            MagnitudeRelation::Less
        } else if a > b {
            MagnitudeRelation::Greater
        } else {
            MagnitudeRelation::Equal
        };
        return MagnitudeJudgment {
            a,
            b,
            relation,
            confidence: 1.0,
            exact: true,
        };
    }
    if a == b {
        return MagnitudeJudgment {
            a,
            b,
            relation: MagnitudeRelation::Equal,
            confidence: 1.0,
            exact: false,
        };
    }
    let (bigger, smaller) = if a > b { (a, b) } else { (b, a) };
    let ratio_gap = if bigger > 0.0 {
        (bigger - smaller) / bigger
    } else {
        0.0
    };
    let confidence = (ratio_gap / WEBER_FRACTION).clamp(0.0, 1.0);
    let relation = if a > b {
        MagnitudeRelation::Greater
    } else {
        MagnitudeRelation::Less
    };
    MagnitudeJudgment {
        a,
        b,
        relation,
        confidence,
        exact: false,
    }
}

/// Whether `claimed` lies within one Weber fraction of `expected` on the log line.
pub fn plausible(claimed: f64, expected: f64) -> bool {
    if expected == 0.0 {
        return claimed == 0.0;
    }
    if claimed <= 0.0 || expected <= 0.0 {
        return false;
    }
    (represent(claimed) - represent(expected)).abs() < (1.0 + WEBER_FRACTION).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subitizing_and_weber_law() {
        assert!(subitizes(3.0));
        assert!(!subitizes(7.0));
        let exact = compare(2.0, 4.0);
        assert!(exact.exact && exact.confidence == 1.0);
        let easy = compare(10.0, 20.0);
        let hard = compare(88.0, 90.0);
        assert!(easy.confidence > hard.confidence);
        assert!(plausible(105.0, 100.0));
        assert!(!plausible(500.0, 100.0));
    }
}
