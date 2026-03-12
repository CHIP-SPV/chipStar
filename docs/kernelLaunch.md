# hipLaunchKernel overhead: per-stream scaling analysis

Benchmark: `tests/benchmarks/manySmallKernels`

Measures median null-stream launch+sync time (ns) as idle blocking stream count
increases.  Each idle stream is created but never receives work, so the runtime
must check every one of them for cross-queue dependencies on every launch.

Hardware: Intel Arc A770, Level0 backend.

---

## Baseline (no fix)

`zeCommandListAppendBarrier` with a signal event is called in `finish()` on an
in-order command list.  With `ZE_COMMAND_QUEUE_FLAG_IN_ORDER`, the Level0 driver
must fence against all other in-order CLs on the same physical engine (all
queues share engine 0 on this platform). Cost scales O(N streams).

Additionally, `addDependenciesQueueSyncImpl` calls `CreateMarkerInQueue` for
every blocking stream unconditionally — each call does `zeCommandListAppendBarrier`
on the *other* queue's command list.

| Idle streams | Median ns/launch |
|--------------|-----------------|
|            0 |       ~328,000   |
|            1 |       ~367,000   |
|            2 |       ~386,000   |
|            4 |       ~421,000   |
|            8 |       ~482,000   |
|           16 |       ~544,000   |
|           32 |       ~805,000   |
|           64 |     ~1,320,000   |

**~4x slowdown at 64 streams** (real-world apps can show 10–30x depending on
workload mix and queue counts).

---

## After Colleen fix (isEmptyQueue guard)

Skip `CreateMarkerInQueue` for queues that have no pending work, using an
`IsEmptyQueue_` atomic bool (set `false` on every submit, reset `true` in
`finish()`).  Because no signal-event is appended to idle CLs, those CLs
have no pending GPU work, so `zeCommandListAppendBarrier` in `finish()` no
longer needs to fence against them — O(N) cost drops to O(1).

| Idle streams | Median ns/launch |
|--------------|-----------------|
|            0 |        ~83,000   |
|            1 |        ~76,000   |
|            2 |        ~76,000   |
|            4 |        ~75,000   |
|            8 |        ~75,000   |
|           16 |        ~76,000   |
|           32 |        ~76,000   |
|           64 |        ~76,000   |

**Flat — no scaling with stream count.**

---

## After Paulius fix (simplify finish())

Replace the two-barrier + event-sync + 2×clSync pattern in `finish()` with a
single `zeCommandListHostSynchronize(ZeCmdListImm_, UINT64_MAX)`.

Background: `ZeCmdListImmCopy_ == ZeCmdListImm_` (issue #1136 was never
implemented), so the cross-barrier pattern was self-referential dead code.
The redundant `zeCommandListAppendBarrier` with a signal event on an in-order
CL would still scale O(N) if the Colleen fix were ever bypassed. Removing it
makes `finish()` simpler, correct, and robust against future regressions.

| Idle streams | Median ns/launch |
|--------------|-----------------|
|            0 |        ~55,000   |
|            1 |        ~54,000   |
|            2 |        ~52,000   |
|            4 |        ~46,000   |
|            8 |        ~46,000   |
|           16 |        ~46,000   |
|           32 |        ~46,000   |
|           64 |        ~46,000   |

**Flat — ~46 µs regardless of stream count.**
Improvement vs baseline: **~29x** at 64 streams (~1,320 µs → ~46 µs).
