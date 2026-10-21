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
