//! Number theory — the integers' own machinery.
//!
//! Exact arithmetic on `u64`/`u128`/`i64`/`i128`: gcd and its
//! extended form, primality, factorization, modular powers and
//! inverses, the Chinese remainder theorem, continued fractions.
//! No floats anywhere in this file — the integers are exact or
//! they are nothing.

/// Greatest common divisor — Euclid, unsigned.
pub fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

/// Least common multiple.
pub fn lcm(a: u64, b: u64) -> u64 {
    if a == 0 || b == 0 {
        return 0;
    }
    a / gcd(a, b) * b
}

/// Extended gcd: returns (g, x, y) with ax + by = g = gcd(a,b).
/// Signed throughout — Bezout's coefficients.
pub fn extended_gcd(a: i64, b: i64) -> (i64, i64, i64) {
    let (mut r, mut rp) = (a.abs(), b.abs());
    let (mut x, mut xp) = (1i64, 0i64);
    let (mut y, mut yp) = (0i64, 1i64);
    while rp != 0 {
        let q = r / rp;
        (r, rp) = (rp, r - q * rp);
        (x, xp) = (xp, x - q * xp);
        (y, yp) = (yp, y - q * yp);
    }
    (r, if a < 0 { -x } else { x }, if b < 0 { -y } else { y })
}

/// Trial-division primality — honest and sufficient to 10¹².
pub fn is_prime(n: u64) -> bool {
    match n {
        0 | 1 => return false,
        2 | 3 => return true,
        _ => {}
    }
    if n % 2 == 0 || n % 3 == 0 {
        return false;
    }
    let mut k = 5u64;
    while k <= n / k {
        if n % k == 0 || n % (k + 2) == 0 {
            return false;
        }
        k += 6;
    }
    true
}

/// Deterministic Miller–Rabin for u64 — the known witness set.
pub fn is_prime_mr(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    for &p in &[2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        if n == p {
            return true;
        }
        if n % p == 0 {
            return false;
        }
    }
    let mut d = n - 1;
    let s = d.trailing_zeros();
    d >>= s;
    for &a in &[2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        let mut x = mod_pow(a, d, n);
        if x == 1 || x == n - 1 {
            continue;
        }
        let mut composite = true;
        for _ in 1..s {
            x = (x as u128 * x as u128 % n as u128) as u64;
            if x == n - 1 {
                composite = false;
                break;
            }
        }
        if composite {
            return false;
        }
    }
    true
}

/// Prime factorization — sorted (prime, exponent) pairs.
pub fn factorize(mut n: u64) -> Vec<(u64, u32)> {
    let mut out = Vec::new();
    let mut push = |p: u64, n: &mut u64| {
        let mut e = 0;
        while *n % p == 0 {
            *n /= p;
            e += 1;
        }
        if e > 0 {
            out.push((p, e));
        }
    };
    push(2, &mut n);
    let mut k = 3u64;
    while k <= n / k {
        push(k, &mut n);
        k += 2;
    }
    if n > 1 {
        out.push((n, 1));
    }
    out
}

/// Euler's totient φ(n) — count of units mod n.
pub fn totient(n: u64) -> u64 {
    if n == 0 {
        return 0;
    }
    let mut result = n;
    for (p, _) in factorize(n) {
        result = result / p * (p - 1);
    }
    result
}

/// All divisors, sorted.
pub fn divisors(n: u64) -> Vec<u64> {
    let mut small = Vec::new();
    let mut large = Vec::new();
    let mut k = 1u64;
    while k <= n / k.max(1) && k * k <= n {
        if n % k == 0 {
            small.push(k);
            if k * k != n {
                large.push(n / k);
            }
        }
        k += 1;
    }
    large.reverse();
    small.extend(large);
    small
}

/// Modular exponentiation — a^e mod m, u128-safe.
pub fn mod_pow(mut a: u64, mut e: u64, m: u64) -> u64 {
    if m == 1 {
        return 0;
    }
    let mut r = 1u128;
    let m = m as u128;
    a %= m as u64;
    let mut a = a as u128;
    while e > 0 {
        if e & 1 == 1 {
            r = r * a % m;
        }
        a = a * a % m;
        e >>= 1;
    }
    r as u64
}

/// Modular inverse of a mod m — exists iff gcd(a,m)=1.
pub fn mod_inverse(a: i64, m: i64) -> Option<i64> {
    let (g, x, _) = extended_gcd(a, m);
    if g != 1 {
        return None;
    }
    Some(x.rem_euclid(m))
}

/// Chinese remainder theorem: x ≡ aᵢ (mod mᵢ). Works with
/// non-coprime moduli too — returns None when the congruences
/// are incompatible. (x, product-modulus) on success.
pub fn crt(residues: &[(u64, u64)]) -> Option<(u64, u64)> {
    // Iterative pairwise merge in signed arithmetic:
    // x' = x + M·t with (M/g)·t ≡ (a−x)/g (mod m/g).
    let mut x: i128 = 0;
    let mut big_m: i128 = 1;
    for &(a, m) in residues {
        if m == 0 || m > i64::MAX as u64 || big_m > i64::MAX as i128 {
            return None;
        }
        let (g, s, _t) = extended_gcd(big_m as i64, m as i64);
        let (g, a, m) = (g as i128, (a % m) as i128, m as i128);
        if (a - x) % g != 0 {
            return None; // incompatible congruences
        }
        let m_g = m / g;
        let modulus = big_m.checked_mul(m_g)?;
        if modulus > u64::MAX as i128 {
            return None;
        }
        let t = (((a - x) / g) * s as i128).rem_euclid(m_g);
        x = (x + big_m * t).rem_euclid(modulus);
        big_m = modulus;
    }
    Some((x as u64, big_m as u64))
}

/// Continued fraction of p/q — the Euclidean algorithm's record.
pub fn continued_fraction(p: i64, q: i64) -> Option<Vec<i64>> {
    if q == 0 {
        return None;
    }
    let (mut p, mut q) = (p, q);
    let mut out = Vec::new();
    while q != 0 {
        out.push(p.div_euclid(q));
        (p, q) = (q, p.rem_euclid(q));
    }
    Some(out)
}

/// Legendre symbol (a/p) for odd prime p: 1 residue, -1 not, 0 zero.
pub fn legendre(a: u64, p: u64) -> i64 {
    if p <= 2 || !is_prime(p) {
        return 0;
    }
    match mod_pow(a, (p - 1) / 2, p) {
        0 => 0,
        1 => 1,
        _ => -1,
    }
}

/// Solve ax + by = c over the integers — one particular
/// solution plus the homogeneous step, or None when
/// gcd(a,b) ∤ c.
pub fn solve_diophantine(a: i64, b: i64, c: i64) -> Option<((i64, i64), (i64, i64))> {
    if a == 0 && b == 0 {
        return if c == 0 { Some(((0, 0), (0, 0))) } else { None };
    }
    let (g, x, y) = extended_gcd(a, b);
    if c % g != 0 {
        return None;
    }
    let k = c / g;
    Some(((x * k, y * k), (b / g, -a / g)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gcd_lcm() {
        assert_eq!(gcd(48, 36), 12);
        assert_eq!(lcm(4, 6), 12);
        let (g, x, y) = extended_gcd(35, 15);
        assert_eq!(g, 5);
        assert_eq!(35 * x + 15 * y, 5);
    }

    #[test]
    fn primes_factors() {
        assert!(is_prime(97) && is_prime_mr(97));
        assert!(!is_prime(91) && !is_prime_mr(561)); // Carmichael
        assert!(is_prime_mr(2_147_483_647));
        assert_eq!(factorize(360), vec![(2, 3), (3, 2), (5, 1)]);
        assert_eq!(totient(9), 6);
        assert_eq!(divisors(28), vec![1, 2, 4, 7, 14, 28]);
    }

    #[test]
    fn modular() {
        assert_eq!(mod_pow(3, 200, 50), mod_pow(3, 200 % 40, 50));
        assert_eq!(mod_inverse(3, 11), Some(4)); // 3·4 = 12 ≡ 1
        assert_eq!(crt(&[(2, 3), (3, 5), (2, 7)]), Some((23, 105)));
        assert_eq!(continued_fraction(415, 93), Some(vec![4, 2, 6, 7]));
        assert_eq!(legendre(2, 7), 1);
        assert_eq!(legendre(3, 7), -1);
    }

    #[test]
    fn diophantine() {
        let ((x, y), (dx, dy)) = solve_diophantine(12, 18, 30).unwrap();
        assert_eq!(12 * x + 18 * y, 30);
        assert_eq!(12 * dx + 18 * dy, 0);
        assert_eq!(solve_diophantine(12, 18, 35), None);
    }
}
