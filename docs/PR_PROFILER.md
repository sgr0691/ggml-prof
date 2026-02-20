# PR: GGML Graph Profiler (op-level timing + memory + JSON export)

## Motivation

ggml is designed for **zero allocations during runtime**, so performance issues usually come from:
- unexpected expensive ops (matmul variants, quant kernels, copies)
- bad graph structure (extra transposes, views that trigger copies, etc.)
- backend placement / buffer sizing decisions

A first-class profiler lets contributors and users quickly answer:
- "Which ops dominate runtime?"
- "Did changing kernel/backends actually help?"
- "How much memory is this graph really using?"

This improves contributor velocity and makes ggml more approachable.

## Usage

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

// Human-readable table (top 10 ops by time)
ggml_profiler_dump_table(profiler, stdout, 10);

// Or JSON for tooling
ggml_profiler_dump_json(profiler, stdout);

ggml_profiler_free(profiler);
```

## Example Output

### Table

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
ADD                    64       1.100        0.050
...
```

### JSON

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
    { "op": "RMS_NORM", "count": 128, "wall_ms_total": 5.2, "wall_ms_max": 0.12 }
  ]
}
```

## Overhead

Benchmark with `bench-profiler-overhead`:

```bash
./bin/bench-profiler-overhead [n_elements] [n_iterations]
```

- **When disabled**: Single branch check; overhead is ~zero.
- **When enabled**: Per-op timing adds ~0.1–2% overhead for typical graphs (small graphs show higher relative overhead due to timing overhead dominating).

## Limitations

- **CPU backend only**: Profiling is implemented for the CPU graph compute path. GPU backends (CUDA, Metal, Vulkan, etc.) are not instrumented.
- **Memory estimate**: Upper-bound (work + unique backend buffers). Not true peak over time.
- **Per-op timing**: Recorded from thread 0 only; internal barriers ensure correct ordering.

## Follow-up Ideas

- Chrome trace export (instant flame timeline)
- Backend event timing on CUDA/Metal/Vulkan (true kernel time)
- Per-tensor memory lifetime tracking (true peak over time)
- Profiler hooks in `ggml_backend_graph_compute*` to cover scheduler + multi-backend runs

## Summary

- Additive API, opt-in, near-zero overhead when off
- Helps performance tuning across ggml ecosystem
- Includes example + JSON for tooling integrations
- Sets foundation for backend-specific profiling later
