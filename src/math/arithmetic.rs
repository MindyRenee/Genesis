//! Arithmetic — exact expression evaluation and spoken-number primitives.
//!
//! Machine-side counterpart to `python/parietal_lobe/math/arithmetic.py`.
//! Pure deterministic evaluation over binary operations, integer powers,
//! Euclidean division/modulo, and real square roots without host `eval`.

/// Binary operations supported by the exact arithmetic engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArithOp {
    Add,
    Sub,
    Mul,
    Div,
    FloorDiv,
    Mod,
    Pow,
}

const MAX_POW_ABS: f64 = 10_000.0;

/// Evaluate one binary arithmetic step; returns `None` on division by zero,
/// exponent overflow, or non-real/non-finite results.
pub fn apply_op(op: ArithOp, lhs: f64, rhs: f64) -> Option<f64> {
    if !lhs.is_finite() || !rhs.is_finite() {
        return None;
    }
    let out = match op {
        ArithOp::Add => lhs + rhs,
        ArithOp::Sub => lhs - rhs,
        ArithOp::Mul => lhs * rhs,
        ArithOp::Div => {
            if rhs == 0.0 {
                return None;
            }
            lhs / rhs
        }
        ArithOp::FloorDiv => {
            if rhs == 0.0 {
                return None;
            }
            (lhs / rhs).floor()
        }
        ArithOp::Mod => {
            if rhs == 0.0 {
                return None;
            }
            lhs.rem_euclid(rhs)
        }
        ArithOp::Pow => {
            if rhs.abs() > MAX_POW_ABS {
                return None;
            }
            if lhs < 0.0 && rhs.fract() != 0.0 {
                return None;
            }
            lhs.powf(rhs)
        }
    };
    out.is_finite().then_some(out)
}

/// Principal real square root; `None` for negative or non-finite inputs.
pub fn real_sqrt(x: f64) -> Option<f64> {
    if !x.is_finite() || x < 0.0 {
        return None;
    }
    Some(x.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_ops_and_guards() {
        assert_eq!(apply_op(ArithOp::Add, 7.0, 5.0), Some(12.0));
        assert_eq!(apply_op(ArithOp::Div, 1.0, 0.0), None);
        assert_eq!(apply_op(ArithOp::Pow, 2.0, 10.0), Some(1024.0));
        assert_eq!(apply_op(ArithOp::Pow, -1.0, 0.5), None);
        assert_eq!(real_sqrt(144.0), Some(12.0));
        assert_eq!(real_sqrt(-1.0), None);
    }
}
