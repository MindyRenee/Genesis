//! Proof — the third pillar of mathematical thinking.
//!
//! The machine's role in proof is the one that is decidable:
//! checking. A formula is a tree of atoms joined by connectives;
//! a sequent Γ ⊢ φ asks whether φ holds under every assignment
//! that makes Γ true — and with finitely many atoms the answer is
//! a truth table, not a guess. The inference rules below are the
//! classical kernel — each takes premises and a claimed
//! conclusion and says whether the step is licensed.
//!
//! Induction is handled honestly: the machine can verify a base
//! case and the step at every integer in a bounded range — that
//! is evidence, never a proof, and the return type says so.

use std::collections::BTreeSet;

/// A propositional formula.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Prop {
    Atom(char),
    Not(Box<Prop>),
    And(Box<Prop>, Box<Prop>),
    Or(Box<Prop>, Box<Prop>),
    Implies(Box<Prop>, Box<Prop>),
    Iff(Box<Prop>, Box<Prop>),
    Truth,
    Falsity,
}

impl Prop {
    pub fn atom(c: char) -> Self {
        Prop::Atom(c)
    }
    pub fn negated(p: Prop) -> Self {
        Prop::Not(Box::new(p))
    }
    pub fn and(a: Prop, b: Prop) -> Self {
        Prop::And(Box::new(a), Box::new(b))
    }
    pub fn or(a: Prop, b: Prop) -> Self {
        Prop::Or(Box::new(a), Box::new(b))
    }
    pub fn implies(a: Prop, b: Prop) -> Self {
        Prop::Implies(Box::new(a), Box::new(b))
    }
    pub fn iff(a: Prop, b: Prop) -> Self {
        Prop::Iff(Box::new(a), Box::new(b))
    }

    /// The atoms occurring in the formula, sorted.
    pub fn atoms(&self) -> BTreeSet<char> {
        let mut out = BTreeSet::new();
        self.collect_atoms(&mut out);
        out
    }

    fn collect_atoms(&self, out: &mut BTreeSet<char>) {
        match self {
            Prop::Atom(c) => {
                out.insert(*c);
            }
            Prop::Not(p) => p.collect_atoms(out),
            Prop::And(a, b) | Prop::Or(a, b) | Prop::Implies(a, b) | Prop::Iff(a, b) => {
                a.collect_atoms(out);
                b.collect_atoms(out);
            }
            Prop::Truth | Prop::Falsity => {}
        }
    }

    /// Evaluate under an assignment (atoms absent count false).
    pub fn eval(&self, assignment: &BTreeSet<char>) -> bool {
        match self {
            Prop::Atom(c) => assignment.contains(c),
            Prop::Not(p) => !p.eval(assignment),
            Prop::And(a, b) => a.eval(assignment) && b.eval(assignment),
            Prop::Or(a, b) => a.eval(assignment) || b.eval(assignment),
            Prop::Implies(a, b) => !a.eval(assignment) || b.eval(assignment),
            Prop::Iff(a, b) => a.eval(assignment) == b.eval(assignment),
            Prop::Truth => true,
            Prop::Falsity => false,
        }
    }

    /// Truth table: every assignment over the formula's atoms
    /// paired with its value.
    pub fn truth_table(&self) -> Vec<(BTreeSet<char>, bool)> {
        let atoms: Vec<char> = self.atoms().into_iter().collect();
        (0..(1u64 << atoms.len()))
            .map(|mask| {
                let assign: BTreeSet<char> = atoms
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| mask & (1 << i) != 0)
                    .map(|(_, &c)| c)
                    .collect();
                let v = self.eval(&assign);
                (assign, v)
            })
            .collect()
    }
}

/// Is the formula true under every assignment — a tautology?
/// Decidable: 2^n assignments, checked exhaustively.
pub fn is_tautology(p: &Prop) -> bool {
    p.truth_table().iter().all(|(_, v)| *v)
}

/// Is the formula true under no assignment — a contradiction?
pub fn is_contradiction(p: &Prop) -> bool {
    p.truth_table().iter().all(|(_, v)| !*v)
}

/// Sequent check Γ ⊢ φ: under every assignment where all
/// premises hold, does the conclusion hold? The semantic turnstile
/// — entailment, decided by the table.
pub fn entails(premises: &[Prop], conclusion: &Prop) -> bool {
    let mut atoms = BTreeSet::new();
    for p in premises.iter().chain(std::iter::once(conclusion)) {
        atoms.extend(p.atoms());
    }
    let atoms: Vec<char> = atoms.into_iter().collect();
    (0..(1u64 << atoms.len())).all(|mask| {
        let assign: BTreeSet<char> = atoms
            .iter()
            .enumerate()
            .filter(|(i, _)| mask & (1 << i) != 0)
            .map(|(_, &c)| c)
            .collect();
        let all_premises = premises.iter().all(|p| p.eval(&assign));
        !all_premises || conclusion.eval(&assign)
    })
}

// ─── The inference kernel ────────────────────────────────────────
// Each rule checks the SHAPE of premises and conclusion — the
// step is licensed when the syntactic pattern matches. Soundness
// then comes free: every licensed step preserves truth.

/// Modus ponens: from (A → B) and A, conclude B.
pub fn modus_ponens(implication: &Prop, antecedent: &Prop, conclusion: &Prop) -> bool {
    match implication {
        Prop::Implies(a, b) => **a == *antecedent && **b == *conclusion,
        _ => false,
    }
}

/// Modus tollens: from (A → B) and ¬B, conclude ¬A.
pub fn modus_tollens(implication: &Prop, neg_consequent: &Prop, conclusion: &Prop) -> bool {
    match (implication, neg_consequent, conclusion) {
        (Prop::Implies(a, b), Prop::Not(nb), Prop::Not(na)) => **nb == **b && **na == **a,
        _ => false,
    }
}

/// Conjunction introduction: from A and B, conclude A ∧ B.
pub fn and_intro(a: &Prop, b: &Prop, conclusion: &Prop) -> bool {
    matches!(conclusion, Prop::And(x, y) if **x == *a && **y == *b)
}

/// Conjunction elimination: from A ∧ B conclude A (or B).
pub fn and_elim(conjunction: &Prop, conclusion: &Prop) -> bool {
    match conjunction {
        Prop::And(a, b) => **a == *conclusion || **b == *conclusion,
        _ => false,
    }
}

/// Disjunction introduction: from A conclude A ∨ B for any B.
pub fn or_intro(premise: &Prop, conclusion: &Prop) -> bool {
    match conclusion {
        Prop::Or(a, b) => **a == *premise || **b == *premise,
        _ => false,
    }
}

/// Hypothetical syllogism: from (A → B) and (B → C), conclude (A → C).
pub fn hypothetical_syllogism(p: &Prop, q: &Prop, conclusion: &Prop) -> bool {
    match (p, q, conclusion) {
        (Prop::Implies(a1, b1), Prop::Implies(b2, c), Prop::Implies(a2, c2)) => {
            **b1 == **b2 && **a1 == **a2 && **c == **c2
        }
        _ => false,
    }
}

/// De Morgan's law as a rewrite: ¬(A ∧ B) ⇄ (¬A ∨ ¬B), checked in
/// both directions.
pub fn de_morgan(negated: &Prop, distributed: &Prop) -> bool {
    match (negated, distributed) {
        (Prop::Not(inner), Prop::Or(na, nb)) => match (&**inner, &**na, &**nb) {
            (Prop::And(a, b), Prop::Not(x), Prop::Not(y)) => **a == **x && **b == **y,
            _ => false,
        },
        (Prop::Or(na, nb), Prop::Not(inner)) => match (&**na, &**nb, &**inner) {
            (Prop::Not(a), Prop::Not(b), Prop::And(x, y)) => **a == **x && **b == **y,
            _ => false,
        },
        _ => false,
    }
}

/// Deduction theorem (implication introduction): if premises ∪ {A}
/// ⊢ B, then premises ⊢ A → B. Checked semantically — the meta-step
/// that turns an assumption into an implication.
pub fn deduction(premises: &[Prop], assumed: &Prop, derived: &Prop) -> bool {
    let mut with = premises.to_vec();
    with.push(assumed.clone());
    entails(&with, derived) && entails(premises, &Prop::implies(assumed.clone(), derived.clone()))
}

/// Proof by contradiction: premises ∪ {¬φ} ⊢ ⊥  iff  premises ⊢ φ.
/// The sequent form of reductio — if assuming the opposite is
/// unsatisfiable, the claim follows. Both sides are checked, and
/// they agree only when the meta-theorem holds here.
pub fn by_contradiction(premises: &[Prop], claim: &Prop) -> bool {
    let mut with = premises.to_vec();
    with.push(Prop::negated(claim.clone()));
    entails(&with, &Prop::Falsity) && entails(premises, claim)
}

/// Bounded induction evidence: verify base(n0) and every step
/// k → k+1 across [lo, hi] under predicate `pred`. Returns the
/// checked range on success — EVIDENCE, never a proof. The
/// quantifier over all naturals stays beyond the machine's reach,
/// and this signature refuses to pretend otherwise.
pub fn induction_evidence(pred: &dyn Fn(u64) -> bool, lo: u64, hi: u64) -> Option<(u64, u64)> {
    for k in lo..=hi {
        if !pred(k) {
            return None;
        }
    }
    Some((lo, hi))
}

/// Case analysis: if premises ⊢ A∨B, premises∪{A} ⊢ C and
/// premises∪{B} ⊢ C, then premises ⊢ C. Exhaustion of cases.
pub fn by_cases(premises: &[Prop], case_a: &Prop, case_b: &Prop, conclusion: &Prop) -> bool {
    let mut wa = premises.to_vec();
    wa.push(case_a.clone());
    let mut wb = premises.to_vec();
    wb.push(case_b.clone());
    entails(premises, &Prop::or(case_a.clone(), case_b.clone()))
        && entails(&wa, conclusion)
        && entails(&wb, conclusion)
        && entails(premises, conclusion)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(c: char) -> Prop {
        Prop::atom(c)
    }

    #[test]
    fn tautologies_and_contradictions() {
        // A ∨ ¬A — excluded middle, always true.
        let em = Prop::or(p('a'), Prop::negated(p('a')));
        assert!(is_tautology(&em));
        // A ∧ ¬A — never true.
        let contra = Prop::and(p('a'), Prop::negated(p('a')));
        assert!(is_contradiction(&contra));
        // (A → B) ↔ (¬B → ¬A) — contraposition.
        let cp = Prop::iff(
            Prop::implies(p('a'), p('b')),
            Prop::implies(Prop::negated(p('b')), Prop::negated(p('a'))),
        );
        assert!(is_tautology(&cp));
    }

    #[test]
    fn entailment() {
        // {A, A→B} ⊢ B
        let imp = Prop::implies(p('a'), p('b'));
        assert!(entails(&[p('a'), imp], &p('b')));
        // {A} ⊬ B
        assert!(!entails(&[p('a')], &p('b')));
    }

    #[test]
    fn rules() {
        let a = p('a');
        let b = p('b');
        let imp = Prop::implies(a.clone(), b.clone());
        assert!(modus_ponens(&imp, &a, &b));
        assert!(!modus_ponens(&imp, &b, &a)); // affirming the consequent — refused
        assert!(modus_tollens(
            &imp,
            &Prop::negated(b.clone()),
            &Prop::negated(a.clone())
        ));
        assert!(and_intro(&a, &b, &Prop::and(a.clone(), b.clone())));
        assert!(and_elim(&Prop::and(a.clone(), b.clone()), &a));
        assert!(or_intro(&a, &Prop::or(a.clone(), p('z'))));
        let bc = Prop::implies(b.clone(), p('c'));
        assert!(hypothetical_syllogism(
            &imp,
            &bc,
            &Prop::implies(a.clone(), p('c'))
        ));
        let nand = Prop::negated(Prop::and(a.clone(), b.clone()));
        let dem = Prop::or(Prop::negated(a.clone()), Prop::negated(b.clone()));
        assert!(de_morgan(&nand, &dem));
    }

    #[test]
    fn meta_rules() {
        let a = p('a');
        let b = p('b');
        // Assuming A, derive A∨B; so ⊢ A → A∨B.
        assert!(deduction(&[], &a, &Prop::or(a.clone(), b.clone())));
        // Cases: {A∨B, A→C, B→C} ⊢ C — the disjunction is a
        // premise too; the implications alone can't produce it.
        let c = p('c');
        let premises = vec![
            Prop::or(a.clone(), b.clone()),
            Prop::implies(a.clone(), c.clone()),
            Prop::implies(b.clone(), c.clone()),
        ];
        assert!(by_cases(&premises, &a, &b, &c));
        // Without A∨B the case analysis is unlicensed.
        assert!(!by_cases(&premises[1..], &a, &b, &c));
    }

    #[test]
    fn induction_is_bounded() {
        // k*k >= k for naturals — evidence over [1, 100].
        let r = induction_evidence(&|k| k * k >= k, 1, 100);
        assert_eq!(r, Some((1, 100)));
        // Fails at k = 0 inside the range? 0*0 >= 0 holds; try k^2 > k:
        assert_eq!(induction_evidence(&|k| k * k > k, 0, 100), None);
    }
}
