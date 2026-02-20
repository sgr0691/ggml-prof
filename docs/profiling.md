# GGML Graph Profiler

The GGML graph profiler is an **opt-in** facility that records per-operation timing, graph-level totals, and memory estimates during `ggml_graph_compute`. When disabled, it adds zero overhead.

## Enabling the Profiler

1. Create a profiler: `ggml_profiler_t * p = ggml_profiler_new();`
2. Build your compute plan: `struct ggml_cplan plan = ggml_graph_plan(cgraph, n_threads, NULL);`
3. Attach the profiler: `ggml_cplan_set_profiler(&plan, p);`
4. Allocate work buffer and run: `ggml_graph_compute(cgraph, &plan);`
5. Inspect results (see below), then free: `ggml_profiler_free(p);`

## Example

```c
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-profiler.h"

// ... build graph gf ...

ggml_profiler_t * profiler = ggml_profiler_new();
struct ggml_cplan plan = ggml_graph_plan(gf, n_threads, NULL);
plan.work_data = (uint8_t *)malloc(plan.work_size);
ggml_cplan_set_profiler(&plan, profiler);

ggml_graph_compute(gf, &plan);

// Print human-readable table (top 10 ops by time)
ggml_profiler_dump_table(profiler, stdout, 10);

// Or export JSON for tooling
ggml_profiler_dump_json(profiler, stdout);

ggml_profiler_free(profiler);
```

## Output

### Table (ggml_profiler_dump_table)

```
=== GGML Profiler Summary ===
Total wall time: 12.345 ms
Threads: 8 | Nodes executed: 742
Work buffer: 1048576 B | Backend buffers: 2147483648 B | Peak estimate: 2148532224 B

--- Top 10 ops by time ---
OP                  COUNT    TOTAL_MS       MAX_MS
------------------------------------------------
MUL_MAT                64      80.100        2.900
RMS_NORM              128       5.200        0.120
...
```

### JSON (ggml_profiler_dump_json)

```json
{
  "version": 1,
  "summary": {
    "wall_ms_total": 12.345,
    "n_threads": 8,
    "n_nodes_executed": 742,
    "work_size_bytes": 1048576,
    "buffers_size_bytes": 2147483648,
    "peak_estimate_bytes": 2148532224
  },
  "ops": [
    { "op": "MUL_MAT", "count": 64, "wall_ms_total": 80.1, "wall_ms_max": 2.9 },
    ...
  ]
}
```

## API Summary

| Function | Description |
|----------|-------------|
| `ggml_profiler_new()` | Create a new profiler |
| `ggml_profiler_free(p)` | Free the profiler |
| `ggml_profiler_reset(p)` | Reset stats for reuse |
| `ggml_cplan_set_profiler(cplan, p)` | Attach profiler to compute plan |
| `ggml_profiler_get_summary(p, summary)` | Fill summary struct |
| `ggml_profiler_dump_table(p, fp, top_k)` | Print top-k ops table |
| `ggml_profiler_dump_json(p, fp)` | Export JSON |

## Memory Estimate

The profiler computes an **upper-bound estimate** of peak memory:

- **work_size_bytes**: Size of the temporary work buffer (from `ggml_graph_plan`)
- **buffers_size_bytes**: Sum of unique backend buffer sizes used by tensors in the graph
- **peak_estimate_bytes**: work_size + buffers_size

This is a conservative estimate; actual peak may be lower due to buffer reuse.

## Scope

- **CPU backend only**: Profiling is implemented for the CPU graph compute path. GPU backends (CUDA, Metal, Vulkan, etc.) are not instrumented.
- **Per-op timing**: Recorded only from thread 0 (`ith == 0`) to avoid double-counting; internal barriers ensure correct ordering.
- **NOPs skipped**: Reshape, transpose, view, permute are not counted as executed nodes.

## Run the Example

```bash
./bin/simple-profiler
```

## Benchmark Overhead

To measure profiler overhead (disabled vs enabled):

```bash
./bin/bench-profiler-overhead [n_elements] [n_iterations]
```

- **n_elements**: Graph size (default 512). Larger graphs amortize timing overhead.
- **n_iterations**: Number of runs (default 100).

Example output:

```
=== Profiler overhead benchmark ===
Graph: n=512 elements (mul + add), 100 iterations (+ 5 warmup)

Profiler disabled: 1234 us total, 12.340 us/iter (avg)
Profiler enabled:  1250 us total, 12.500 us/iter (avg)

Overhead: 0.160 us/iter (1.30%)
```
