# Stage 2 + Stage 2B V2.2 UNIVERSITY_A40 package

This package contains the frozen V2.2 Stage-2 Graph-GRU and Stage-2B
Vanilla-GRU development-selection workloads. It is bound to the validated
Stage-1 V2.2 `LOG1P / log1p_target` decision. Stage 2 has 144 source-only fits;
Stage 2B has 48 source-only fits. Both are zero-shot development evaluations,
with no target adaptation.

Extract the ZIP into a new directory on TS9, preserving repository-relative
paths. Use only `%USERPROFILE%\bike_env\Scripts\python.exe` through the supplied
PowerShell launcher. Do not merge V2.1 fitted checkpoints or local CPU results
into the extraction.

Run `preflight` first with the exact bundle-manifest SHA-256 in `COMMANDS.txt`.
Preflight checks the package, scientific job maps, upstream bindings, A40/CUDA
runtime, deterministic settings, resource gate, and synthetic model forwards.
It performs zero optimizer steps and zero scientific evaluations. Only a PASS
on TS9 authorizes `launch`.

The selected operational concurrency is four workers. Six is accepted only if
the runtime resource gate proves adequate A40 memory, host RAM, CPU capacity,
and aggregate cache footprint. Worker count is recorded as runtime provenance;
it is not a scientific hyperparameter.

Launch starts a hidden detached controller and returns its process ID and log
paths. The controller survives closure of the invoking PowerShell window and an
ordinary RDP disconnect, but not Windows logoff, reboot, shutdown, or a host
policy that terminates the session. It maintains one fresh Python process per
immutable fit and never gives multiple scientific fits to a long-lived worker.

Every attempt, checkpoint, prediction commitment, completion, exit record, and
log has a unique stage-isolated path. File contents are flushed and atomically
renamed. OS locks prevent overlapping controllers and duplicate jobs. A restart
skips an existing job only after full validation; a partial job starts again
from its registered seed and optimizer update zero.

The shared queue interleaves three Stage-2 jobs and one Stage-2B job. After all
192 jobs, it invokes separate read-only validators, freezes each stage using
only its registered selection rule, records both decision manifests, and stops.
It cannot launch Stage 3 or Stage 4. Final-target labels are outside the package
and blocked by path and output firewalls.

This local package verification does not claim an A40 runtime PASS. The strict
device-specific preflight must be run after extraction on TS9.
