//! Combinatorics — the art of counting without counting.
//!
//! Exact integers throughout — u128 where u64 would overflow.
//! Factorials and binomials saturate honestly: None past the
//! type's reach rather than a wrapped lie. The named numbers
//! (derangements, Catalan, Stirling, Bell, partition counts)
//! come from their recurrences, because a recurrence is a
//! definition the machine can run.

/// n! — exact through 34!.
pub fn factorial(n: u64) -> Option<u128> {
    let mut f = 1u128;
    for k in 2..=n {
        f = f.checked_mul(k as u128)?;
    }
    Some(f)
}

/// n! saturating at u64::MAX — for callers that only compare.
pub fn factorial64(n: u64) -> u64 {
    let mut f = 1u64;
    for k in 2..=n {
        f = f.saturating_mul(k);
    }
    f
}

/// C(n, k) — multiplicative form, exact.
pub fn binomial(n: u64, k: u64) -> Option<u128> {
    if k > n {
        return Some(0);
    }
    let k = k.min(n - k);
    let mut r = 1u128;
    for i in 0..k {
        r = r.checked_mul((n - i) as u128)? / ((i + 1) as u128);
    }
    Some(r)
}

/// Falling factorial (n)_k = n!/(n-k)! — arrangements of k from n.
pub fn arrangements(n: u64, k: u64) -> Option<u128> {
    if k > n {
        return Some(0);
    }
    let mut r = 1u128;
    for i in 0..k {
        r = r.checked_mul((n - i) as u128)?;
    }
    Some(r)
}

/// Multinomial coefficient n! / (k1! k2! ... km!).
pub fn multinomial(n: u64, parts: &[u64]) -> Option<u128> {
    if parts.iter().sum::<u64>() != n {
        return None;
    }
    let mut r = 1u128;
    let mut used = 0u64;
    for &k in parts {
        r = r.checked_mul(binomial(n - used, k)?)?;
        used += k;
    }
    Some(r)
}

/// Stars and bars: ways to place n identical objects into k
/// distinct bins — C(n + k - 1, k - 1).
pub fn stars_and_bars(n: u64, k: u64) -> Option<u128> {
    if k == 0 {
        return if n == 0 { Some(1) } else { Some(0) };
    }
    binomial(n.checked_add(k)?.checked_sub(1)?, k - 1)
}

/// Derangements !n — permutations with no fixed point.
/// !n = (n-1)(!(n-1) + !(n-2));  !0 = 1, !1 = 0.
/// Sequence: 1, 0, 1, 2, 9, 44, 265, 1854, ...
pub fn derangements(n: u64) -> Option<u128> {
    match n {
        0 => Some(1),
        1 => Some(0),
        _ => {
            let (mut a, mut b) = (1u128, 0u128); // !0, !1
            for k in 2..=n {
                let next = (k as u128 - 1).checked_mul(a.checked_add(b)?)?;
                (a, b) = (b, next);
            }
            Some(b)
        }
    }
}

/// Catalan number C_n = C(2n, n) / (n+1) — the count that keeps
/// reappearing: trees, paths, bracketings, triangulations.
pub fn catalan(n: u64) -> Option<u128> {
    binomial(n.checked_mul(2)?, n)?.checked_div(n as u128 + 1)
}

/// Stirling numbers of the second kind S(n, k): partitions of an
/// n-set into k nonempty blocks. Recurrence
/// S(n,k) = k·S(n-1,k) + S(n-1,k-1), S(0,0) = 1.
pub fn stirling2(n: u64, k: u64) -> Option<u128> {
    if k > n || k == 0 {
        return if n == 0 && k == 0 { Some(1) } else { Some(0) };
    }
    let mut prev = vec![0u128; (k + 1) as usize];
    prev[0] = 1;
    for i in 1..=n {
        let mut cur = vec![0u128; (k + 1) as usize];
        for j in 1..=k.min(i) {
            cur[j as usize] = (j as u128)
                .checked_mul(prev[j as usize])?
                .checked_add(prev[(j - 1) as usize])?;
        }
        prev = cur;
    }
    Some(prev[k as usize])
}

/// Bell number B_n — total partitions of an n-set, the row sums
/// of Stirling's triangle.
pub fn bell(n: u64) -> Option<u128> {
    let mut total = 0u128;
    for k in 0..=n {
        total = total.checked_add(stirling2(n, k)?)?;
    }
    Some(total)
}

/// Integer partition count p(n) — ways to write n as a sum of
/// positive integers, order ignored. DP over coin-style
/// generating function.
pub fn partitions(n: u64) -> Option<u128> {
    let mut p = vec![0u128; (n + 1) as usize];
    p[0] = 1;
    for part in 1..=n {
        for s in part..=n {
            p[s as usize] = p[s as usize].checked_add(p[(s - part) as usize])?;
        }
    }
    Some(p[n as usize])
}

/// Fibonacci F(n) — F(0)=0, F(1)=1.
pub fn fibonacci(n: u64) -> Option<u128> {
    let (mut a, mut b) = (0u128, 1u128);
    for _ in 0..n {
        let next = a.checked_add(b)?;
        (a, b) = (b, next);
    }
    Some(a)
}

/// Pigeonhole check: can `items` fit into `bins` with at most
/// `cap` per bin? The trivially powerful principle.
pub fn pigeonhole_holds(items: u64, bins: u64, cap: u64) -> bool {
    items <= bins.saturating_mul(cap)
}

/// Minimum collisions guaranteed: ceil(items / bins).
pub fn pigeonhole_min(items: u64, bins: u64) -> u64 {
    if bins == 0 {
        return if items == 0 { 0 } else { u64::MAX };
    }
    items.div_ceil(bins)
}

/// Powers of two as the subsets count: 2^n.
pub fn power_set_size(n: u64) -> Option<u128> {
    1u128.checked_shl(u32::try_from(n).ok()?)
}

/// Handshake lemma check: does a degree sequence sum to twice
/// the edge count (even)? — graph-realizability's first filter.
pub fn degree_sequence_possible(degrees: &[u64]) -> bool {
    degrees
        .iter()
        .fold(0, |parity, degree| parity ^ (degree & 1))
        == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts() {
        assert_eq!(factorial(10), Some(3_628_800));
        assert_eq!(binomial(5, 2), Some(10));
        assert_eq!(binomial(52, 5), Some(2_598_960));
        assert_eq!(arrangements(5, 2), Some(20));
        assert_eq!(multinomial(4, &[2, 2]), Some(6));
        assert_eq!(stars_and_bars(5, 3), Some(21));
    }

    #[test]
    fn named_numbers() {
        // !n: 1, 0, 1, 2, 9, 44, 265
        let ds: Vec<u128> = (0..7).map(|n| derangements(n).unwrap()).collect();
        assert_eq!(ds, vec![1, 0, 1, 2, 9, 44, 265]);
        // Catalan: 1, 1, 2, 5, 14, 42
        let cs: Vec<u128> = (0..6).map(|n| catalan(n).unwrap()).collect();
        assert_eq!(cs, vec![1, 1, 2, 5, 14, 42]);
        // Stirling: S(5,2) = 15, S(4,2) = 7
        assert_eq!(stirling2(5, 2), Some(15));
        assert_eq!(bell(5), Some(52));
        // p(5) = 7, p(10) = 42
        assert_eq!(partitions(5), Some(7));
        assert_eq!(partitions(10), Some(42));
        assert_eq!(fibonacci(10), Some(55));
    }

    #[test]
    fn pigeonhole() {
        assert!(pigeonhole_holds(10, 3, 4));
        assert!(!pigeonhole_holds(13, 3, 4));
        assert_eq!(pigeonhole_min(13, 3), 5); // 13 items, 3 bins: some bin has >= 5
    }

    #[test]
    fn overflow_is_honest() {
        assert_eq!(factorial(34), Some(295232799039604140847618609643520000000));
        assert_eq!(factorial(35), None); // past u128 — say so
        assert_eq!(power_set_size(10), Some(1024));
    }
}
