# Stage-1 V2.2 A40 prelaunch throughput audit

Status: **PASS — operational scheduling amendment only**  
Scientific launch performed: **no**

The original authoritative A40 controller had a maximum concurrency of one. Its `launch()` loop called a fresh worker through blocking `subprocess.run(..., check=True)` and waited for that worker to finish before dispatching the next immutable job.

The frozen jobs are safe to execute concurrently. Every fit runs in a fresh Python process; constructs its own model, optimizer, Python/NumPy/PyTorch/CUDA RNG state, and CUDA context; reads immutable caches; holds a per-job OS lock; and writes checkpoints and predictions under its immutable job ID plus an attempt UUID. Final completion records have one unique path per immutable job and are committed with flush, `fsync`, and append-only rename. The coordinator has no model, optimizer, or scientific RNG state.

The revised operational controller permits one through four workers and defaults to four. It executes all 24 source fits first and only then dispatches the 24 dependent adaptations, so the interleaved job-map ordering cannot cause an adaptation to race its source checkpoint. Every immutable fit still receives one fresh `research.stage1_v2_2.a40 job` Python process.

Four workers are the recommended initial UNIVERSITY_A40 setting. The complete immutable cache payload is 670,747,102 bytes per source worker; four copies total 2,682,988,408 bytes before modest model/activation overhead. This is below 3 GiB on the 48 GiB A40. The model has 12,939 parameters, and four workers request eight CPU threads in total. Actual GPU memory and utilization should still be monitored during the first wave.

The PowerShell wrapper uses asynchronous `Start-Process` without `-Wait`, a hidden process, and redirected controller stdout/stderr. Closing the invoking PowerShell window or disconnecting RDP without logging off does not terminate the coordinator. Workers use `CREATE_NO_WINDOW` and `CREATE_NEW_PROCESS_GROUP`, and each has separate stdout/stderr files. This mechanism does not survive Windows logoff, reboot, or host shutdown.

Worker count is written to the controller-run runtime provenance and is explicitly not a scientific hyperparameter. Batch size, update counts, optimizer, architecture, seeds, data, evaluation, and selection remain unchanged. The immutable job-map SHA-256 remains `4709c222b799a9a2d4ef209ce80ae5ee859b5a818db5c1cb97df2954ffce3937`.

Twelve non-scientific throughput checks and PowerShell syntax validation passed. No Stage-1 fit was launched, no scientific evaluation ran, and no final-target label was accessed.
