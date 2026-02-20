#pragma once

//
// GGML Graph Profiler
//
// Opt-in profiling facility for ggml_graph_compute:
// - per-op wall time (and count)
// - graph-level totals (wall time, n_ops, n_threads)
// - peak memory estimate (work buffer + unique backend buffers)
// - JSON and human-readable table export
//
// Zero overhead when profiler is not attached.
//

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef GGML_API
#define GGML_API
#endif

typedef struct ggml_profiler ggml_profiler_t;

// A single aggregated op record (by op type)
struct ggml_profiler_op_stat {
    enum ggml_op op;
    uint64_t count;
    double   wall_ms_total;
    double   wall_ms_max;
};

// Graph summary
struct ggml_profiler_summary {
    double   wall_ms_total;
    int      n_threads;
    int      n_nodes_executed;
    size_t   work_size_bytes;
    size_t   buffers_size_bytes;      // sum unique backend buffers
    size_t   peak_estimate_bytes;     // work + buffers
};

// Lifecycle
GGML_API ggml_profiler_t * ggml_profiler_new(void);
GGML_API void              ggml_profiler_free(ggml_profiler_t * p);
GGML_API void              ggml_profiler_reset(ggml_profiler_t * p);

// Attach/detach profiler to a compute plan (lowest friction)
GGML_API void ggml_cplan_set_profiler(struct ggml_cplan * cplan, ggml_profiler_t * p);

// Read results
GGML_API void ggml_profiler_get_summary(const ggml_profiler_t * p, struct ggml_profiler_summary * out);

// Export
GGML_API void ggml_profiler_dump_json(const ggml_profiler_t * p, FILE * fp);
GGML_API void ggml_profiler_dump_table(const ggml_profiler_t * p, FILE * fp, int top_k);

// Internal: monotonic time in nanoseconds (for instrumentation)
GGML_API uint64_t ggml_profiler_time_ns(void);

// Internal: record op timing (called by CPU backend when ith == 0)
GGML_API void ggml_profiler_record_op(ggml_profiler_t * p, enum ggml_op op, uint64_t wall_ns);

// Internal: start/end graph timing, compute memory estimate (called by CPU backend)
GGML_API void ggml_profiler_begin_graph(ggml_profiler_t * p, const struct ggml_cgraph * cgraph,
                                        const struct ggml_cplan * cplan);
GGML_API void ggml_profiler_end_graph(ggml_profiler_t * p, uint64_t wall_ns_total);

#ifdef __cplusplus
}
#endif
