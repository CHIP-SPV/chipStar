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
