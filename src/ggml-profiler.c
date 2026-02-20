//
// GGML Graph Profiler - per-op timing, memory estimate, JSON/table export
//

#include "ggml-profiler.h"
#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <inttypes.h>
#include <time.h>

#define GGML_PROFILER_OP_COUNT (GGML_OP_COUNT + 1)

struct ggml_profiler {
    // per-op stats: index by enum ggml_op (0..GGML_OP_COUNT-1)
    uint64_t count[GGML_PROFILER_OP_COUNT];
    uint64_t total_ns[GGML_PROFILER_OP_COUNT];
    uint64_t max_ns[GGML_PROFILER_OP_COUNT];

    // summary (filled at end_graph)
    double   wall_ms_total;
    int      n_threads;
    int      n_nodes_executed;
    size_t   work_size_bytes;
    size_t   buffers_size_bytes;
    size_t   peak_estimate_bytes;
};

#if defined(_WIN32) && (defined(_MSC_VER) || defined(__MINGW32__))
#include <windows.h>
#endif

uint64_t ggml_profiler_time_ns(void) {
#if defined(_WIN32) && (defined(_MSC_VER) || defined(__MINGW32__))
    LARGE_INTEGER t, f;
    QueryPerformanceCounter(&t);
    QueryPerformanceFrequency(&f);
    return (uint64_t)((t.QuadPart * 1000000000ULL) / f.QuadPart);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
#endif
}

ggml_profiler_t * ggml_profiler_new(void) {
    ggml_profiler_t * p = (ggml_profiler_t *)malloc(sizeof(struct ggml_profiler));
    if (p) {
        ggml_profiler_reset(p);
    }
    return p;
}

void ggml_profiler_free(ggml_profiler_t * p) {
    free(p);
}

void ggml_profiler_reset(ggml_profiler_t * p) {
    if (!p) return;
    memset(p->count, 0, sizeof(p->count));
    memset(p->total_ns, 0, sizeof(p->total_ns));
    memset(p->max_ns, 0, sizeof(p->max_ns));
    p->wall_ms_total = 0;
    p->n_threads = 0;
    p->n_nodes_executed = 0;
    p->work_size_bytes = 0;
    p->buffers_size_bytes = 0;
    p->peak_estimate_bytes = 0;
}

void ggml_cplan_set_profiler(struct ggml_cplan * cplan, ggml_profiler_t * p) {
    if (cplan) {
        cplan->profiler = p;
    }
}

void ggml_profiler_record_op(ggml_profiler_t * p, enum ggml_op op, uint64_t wall_ns) {
    if (!p || op >= GGML_OP_COUNT) return;
    int idx = (int)op;
    p->count[idx]++;
    p->total_ns[idx] += wall_ns;
    if (wall_ns > p->max_ns[idx]) {
        p->max_ns[idx] = wall_ns;
    }
}

static void add_buffer_unique(ggml_backend_buffer_t * bufs, int * n_bufs, int max_bufs, ggml_backend_buffer_t buf) {
    if (!buf) return;
    for (int i = 0; i < *n_bufs; i++) {
        if (bufs[i] == buf) return;
    }
    if (*n_bufs < max_bufs) {
        bufs[(*n_bufs)++] = buf;
    }
}

static size_t ggml_profiler_compute_buffers_size(const struct ggml_cgraph * cgraph) {
    // Collect unique backend buffers from nodes and leafs
#define MAX_BUFFERS 4096
    ggml_backend_buffer_t bufs[MAX_BUFFERS];
    int n_bufs = 0;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * t = cgraph->nodes[i];
        if (t->buffer) add_buffer_unique(bufs, &n_bufs, MAX_BUFFERS, t->buffer);
    }
    for (int i = 0; i < cgraph->n_leafs; i++) {
        struct ggml_tensor * t = cgraph->leafs[i];
        if (t->buffer) add_buffer_unique(bufs, &n_bufs, MAX_BUFFERS, t->buffer);
    }

    size_t total = 0;
    for (int i = 0; i < n_bufs; i++) {
        total += ggml_backend_buffer_get_size(bufs[i]);
    }
    return total;
}

void ggml_profiler_begin_graph(ggml_profiler_t * p, const struct ggml_cgraph * cgraph,
                               const struct ggml_cplan * cplan) {
    if (!p || !cgraph || !cplan) return;
    p->n_threads = cplan->n_threads;
    p->work_size_bytes = cplan->work_size;
    p->buffers_size_bytes = ggml_profiler_compute_buffers_size(cgraph);
    p->peak_estimate_bytes = p->work_size_bytes + p->buffers_size_bytes;
}

void ggml_profiler_end_graph(ggml_profiler_t * p, uint64_t wall_ns_total) {
    if (!p) return;
    p->wall_ms_total = (double)wall_ns_total / 1e6;
    int n_executed = 0;
    for (int i = 0; i < GGML_PROFILER_OP_COUNT; i++) {
        n_executed += (int)p->count[i];
    }
    p->n_nodes_executed = n_executed;
}

void ggml_profiler_get_summary(const ggml_profiler_t * p, struct ggml_profiler_summary * out) {
    if (!p || !out) return;
    out->wall_ms_total = p->wall_ms_total;
    out->n_threads = p->n_threads;
    out->n_nodes_executed = p->n_nodes_executed;
    out->work_size_bytes = p->work_size_bytes;
    out->buffers_size_bytes = p->buffers_size_bytes;
    out->peak_estimate_bytes = p->peak_estimate_bytes;
}

void ggml_profiler_dump_json(const ggml_profiler_t * p, FILE * fp) {
    if (!p || !fp) return;
    fprintf(fp, "{\n");
    fprintf(fp, "  \"version\": 1,\n");
    fprintf(fp, "  \"summary\": {\n");
    fprintf(fp, "    \"wall_ms_total\": %.6f,\n", p->wall_ms_total);
    fprintf(fp, "    \"n_threads\": %d,\n", p->n_threads);
    fprintf(fp, "    \"n_nodes_executed\": %d,\n", p->n_nodes_executed);
    fprintf(fp, "    \"work_size_bytes\": %zu,\n", (size_t)p->work_size_bytes);
    fprintf(fp, "    \"buffers_size_bytes\": %zu,\n", (size_t)p->buffers_size_bytes);
    fprintf(fp, "    \"peak_estimate_bytes\": %zu\n", (size_t)p->peak_estimate_bytes);
    fprintf(fp, "  },\n");
    fprintf(fp, "  \"ops\": [\n");
    int first = 1;
    for (int i = 0; i < GGML_OP_COUNT; i++) {
        if (p->count[i] == 0) continue;
        const char * op_name = ggml_op_name((enum ggml_op)i);
        if (!op_name) op_name = "UNKNOWN";
        if (!first) fprintf(fp, ",\n");
        fprintf(fp, "    { \"op\": \"%s\", \"count\": %" PRIu64 ", \"wall_ms_total\": %.6f, \"wall_ms_max\": %.6f }",
                op_name, p->count[i],
                (double)p->total_ns[i] / 1e6,
                (double)p->max_ns[i] / 1e6);
        first = 0;
    }
    fprintf(fp, "\n  ]\n");
    fprintf(fp, "}\n");
}

void ggml_profiler_dump_table(const ggml_profiler_t * p, FILE * fp, int top_k) {
    if (!p || !fp) return;
    if (top_k <= 0) top_k = 20;

    // Build sorted list of (op_idx, total_ms) for top-k
    struct { int op_idx; double total_ms; } rows[GGML_PROFILER_OP_COUNT];
    int n_rows = 0;
    for (int i = 0; i < GGML_OP_COUNT; i++) {
        if (p->count[i] == 0) continue;
        rows[n_rows].op_idx = i;
        rows[n_rows].total_ms = (double)p->total_ns[i] / 1e6;
        n_rows++;
    }
    // Simple sort by total_ms descending
    for (int i = 0; i < n_rows - 1; i++) {
        for (int j = i + 1; j < n_rows; j++) {
            if (rows[j].total_ms > rows[i].total_ms) {
                int tmp_idx = rows[i].op_idx;
                double tmp_ms = rows[i].total_ms;
                rows[i].op_idx = rows[j].op_idx;
                rows[i].total_ms = rows[j].total_ms;
                rows[j].op_idx = tmp_idx;
                rows[j].total_ms = tmp_ms;
            }
        }
    }

    fprintf(fp, "=== GGML Profiler Summary ===\n");
    fprintf(fp, "Total wall time: %.3f ms\n", p->wall_ms_total);
    fprintf(fp, "Threads: %d | Nodes executed: %d\n", p->n_threads, p->n_nodes_executed);
    fprintf(fp, "Work buffer: %zu B | Backend buffers: %zu B | Peak estimate: %zu B\n",
            (size_t)p->work_size_bytes, (size_t)p->buffers_size_bytes, (size_t)p->peak_estimate_bytes);
    fprintf(fp, "\n--- Top %d ops by time ---\n", top_k);
    fprintf(fp, "%-16s %8s %12s %12s\n", "OP", "COUNT", "TOTAL_MS", "MAX_MS");
    fprintf(fp, "------------------------------------------------\n");

    int k = (top_k < n_rows) ? top_k : n_rows;
    for (int i = 0; i < k; i++) {
        int idx = rows[i].op_idx;
        const char * name = ggml_op_name((enum ggml_op)idx);
        if (!name) name = "UNKNOWN";
        fprintf(fp, "%-16s %8" PRIu64 " %12.3f %12.3f\n",
                name, p->count[idx],
                (double)p->total_ns[idx] / 1e6,
                (double)p->max_ns[idx] / 1e6);
    }
}
