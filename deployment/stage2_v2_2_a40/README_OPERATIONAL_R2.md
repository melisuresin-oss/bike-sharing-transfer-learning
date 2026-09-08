# Stage 2/Stage 2B V2.2 operational R2 overlay

R2 fixes the controller-module interface failure that prevented R1 from
dispatching any new job. It preserves the complete R1 canonical metadata
validator and delegates every unchanged function or attribute to R1 or the
byte-identical base `a40.py` module.

The unchanged original controller can now call `verify_package`, `preflight`,
atomic output helpers, runtime constants, and every other expected interface
member. Spawned scientific workers explicitly run
`research.stage2_v2_2.a40_r2`, which delegates scientific work to the R1/base
implementation and therefore retains corrected canonical validation.

Apply this tiny overlay over the existing TS9 run root after verifying its ZIP
hash. Run package-check, strict preflight, and validate-existing. The detached
launcher repeats preflight and the four-completion retention gate before entering
the original dispatch loop. Concurrency remains four.

No base, R1, scientific-contract, job-map, checkpoint, prediction, completion,
or result file is included in or overwritten by R2. Stage 3, Stage 4, and
final-target-label access remain unauthorized.
