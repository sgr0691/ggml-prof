// Simple example demonstrating the GGML graph profiler
// Build a small graph (mul + add), run with profiling, and print results

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-profiler.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(void) {
    ggml_time_init();

    // Create context and tensors for: c = a * b + d
    size_t ctx_size = 256 * 1024;  // 256 KB
    struct ggml_init_params params = {
        .mem_size   = ctx_size,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    struct ggml_context * ctx = ggml_init(params);

    int n = 128;
    struct ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    struct ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    struct ggml_tensor * d = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);

    // Initialize with simple values
    for (int i = 0; i < n; i++) {
        ((float *)a->data)[i] = 1.0f / (i + 1);
        ((float *)b->data)[i] = (float)(i + 1);
        ((float *)d->data)[i] = 0.1f;
    }

    // Build graph: c = a * b + d
    struct ggml_tensor * mul = ggml_mul(ctx, a, b);
    struct ggml_tensor * c  = ggml_add(ctx, mul, d);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, c);

    // Create profiler and attach to plan
    ggml_profiler_t * profiler = ggml_profiler_new();
    struct ggml_cplan plan = ggml_graph_plan(gf, 1, nullptr);

    if (plan.work_size > 0) {
        plan.work_data = (uint8_t *)malloc(plan.work_size);
    }
    ggml_cplan_set_profiler(&plan, profiler);

    // Run computation with profiling
    ggml_graph_compute(gf, &plan);

    // Print results
    printf("\n=== Profiler table output (top 10) ===\n");
    ggml_profiler_dump_table(profiler, stdout, 10);

    printf("\n=== Profiler JSON output ===\n");
    ggml_profiler_dump_json(profiler, stdout);

    // Summary
    struct ggml_profiler_summary summary;
    ggml_profiler_get_summary(profiler, &summary);
    printf("\n=== Summary ===\n");
    printf("Total: %.3f ms, %d threads, %d nodes\n",
           summary.wall_ms_total, summary.n_threads, summary.n_nodes_executed);
    printf("Peak memory estimate: %zu bytes (%.2f KB)\n",
           summary.peak_estimate_bytes, summary.peak_estimate_bytes / 1024.0);

    // Cleanup
    if (plan.work_data) {
        free(plan.work_data);
    }
    ggml_profiler_free(profiler);
    ggml_free(ctx);

    return 0;
}
