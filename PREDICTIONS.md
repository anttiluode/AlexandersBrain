# Predictions, written before G1-G4 were run

Medium: overdamped complex Ginzburg-Landau field psi on a 64 x 64 grid (dx = 0.5, coherence
length xi = 1), with a material map mu(x): mu = +1 is ordered material, mu = -1 is normal.
A ring (annulus) of radius R = 10 and width W = 4 is cut out of normal material. G0 found
deterministically stable winding states n = 0..4 (n = 5 decays), so the ring can hold the
9 integers -4..4.

Two memories live in the same ring:
- **the integer n** (winding number, gauge invariant);
- **the global phase Theta** (psi ~ |psi| e^{i(n phi + Theta)}), a continuous phase memory of
  the VMN kind.

## G1 - noise

1. Theta diffuses with variance exactly `T t / integral |psi|^2 dA` (no free parameter).
2. Snapped to 9 levels, Theta recall falls to chance (1/9) once that variance passes about
   (pi/9)^2; n recall stays at 1.00 over the same time at every T where the ring stays ordered.
3. A disk (no hole) stores Theta no better than the ring: holding a phase gains nothing from
   the hole. Only the integer does.
4. Integer errors, when they happen, start with the highest |n| (closest to depairing).

## G2 - reading

1. A "pinch" ping (lower mu in one patch of the ring for a while) produces a phase drop
   across the patch, measured between two points of the field itself (no external
   reference). That drop separates all 9 values of n with margin > 3 noise sd.
2. 100 pinch reads change n in 0 rings for pinches above a threshold mu_p*; below it the
   read is destructive (a phase slip). The threshold is higher (more fragile) for larger |n|.
3. Reading Theta needs an external reference wave; each read pulls Theta toward the
   reference, so 100 reads produce a cumulative shift that grows with read strength - VMN
   Gate 5's read damage, reproduced. The pinch read leaves the n memory intact at the
   same time.
4. Local damage: zeroing psi in a disk of radius r on the ring does not change n while
   2r < W; it changes n only when the hole cuts the whole width.

## G3 - constriction lifetime ("ladder")

The ring is narrowed to width w over a short arc. With noise T, n decays by phase slips
at the constriction.
1. Mean escape time tau rises with w; across w = 1.0..3.0 it spans at least 2 decades at
   one T.
2. ln tau is linear in 1/T at fixed w (Arrhenius); the slope (the barrier) grows with w.

## G4 - the clasp (ring cut by a gap of normal material, width g)

This is the 2-D stand-in for one finite stage of Alexander's clasp: a loop that does not
quite close.
1. At T = 0, winding n >= 1 is metastable only for g below a critical g* of about 2-3 xi;
   g* is smaller for larger n.
2. With noise, the n = 1 lifetime falls faster than exponentially in g (the barrier is a
   Josephson energy ~ e^{-g/xi}, so ln tau ~ e^{-g}).
3. So a clasp traps an integer only if its gap is within a few coherence lengths. The
   finite-depth clasps that a real medium can build trap nothing wider than that.

None of G1-G4 tests 3-D linking. The 2-D ring is the 2-D cross-section of a linked tube.

---

# Predictions for G5-G7 (writing), written before they were run, 21 Oct 2026

Write line: a uniform rotation-like vector potential, so every ring of the same size is
threaded by the same flux f (in flux quanta). Gate: a gap of normal material of arc length 3
opened in one ring. A first check showed that with the gate open the arc's phase twist
builds up diffusively, too slowly to reach f within 600 t. So write time is a variable.

## G5 - write map, one ring, T = 0.05
1. Gated arm with enough open time (1800 t): written n = round(f) for integer targets
   |f| <= 2. At half-integer-ish f the result depends on start state and noise.
2. Short open time (200 t) under-writes: |n| falls short of round(f) for |f| >= 1.5.
3. Gate open, no field (f = 0): erases to n = 0. Not random as in AlexandersBrainSol's
   1-D gate-only arm, because a 3-xi gap drains the arc's current (G4).
4. Closed arm (field on, gate shut): n0 kept whenever |n0 - f| <= 2; crosstalk appears
   at |n0 - f| >= 3, i.e. when the closed ring's current approaches what G1 showed
   decays at T = 0.05.

## G6 - addressed array: 4 rings, one global field, local gates, alphabet {-1, 0, 1}
1. Random sequences of 15 writes: target correct >= 95% of ops; crosstalk (a non-target
   ring changing) <= 1% of ring-ops.

## G7 - heat as the gate (no material switch)
The ring has a constriction (w = 2: holds n = 1 for > 36000 t at T = 0.07, G3). Heating a
patch over the constriction to T_hot while the field is on should write.
1. There is a T_hot window: too cold, no write; hot enough, written n = round(f) for
   |f| <= 1.
2. Heat with no field erases toward 0 or randomises; field with no heat writes nothing.
3. If G7.1 holds, the G6 array sequence works with heat addressing too: target >= 95%,
   crosstalk <= 1%.

## Addendum before G6c / G7c (written after the first array runs, before the reruns)

G6 at R = 8 and R = 10 and G7b gave target accuracy 94-99% but crosstalk 5-12.5%, all at
ops where the field changed sign. Diagnosis: the write field was a rotation about the box
centre. A ring off that axis also sees a uniform flow of (B/2)|c| ~ 0.2 across it per flux
quantum. It is pure gauge in steady state, but each switch of the field must be absorbed
by the ring's phase, and the transient adds to the circulating current. The G5 ring sat on
the axis and showed no crosstalk at the same flux difference.
Rerun with the same flux threaded through each hole (field centred on each ring):
1. material gates: target >= 95%, crosstalk <= 1%;
2. heat gates (T_hot = 1, 1000 t): target >= 95%, crosstalk <= 1%.
If crosstalk stays, the diagnosis is wrong.
