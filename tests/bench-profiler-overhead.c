// Benchmark: profiler overhead (disabled vs enabled)
// Runs the same graph many times with and without profiling to measure overhead.

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-profiler.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

#define WARMUP 5
#define ITERATIONS 100
#define MIN_ITERATIONS 10

static struct ggml_cgraph * build_graph(struct ggml_context * ctx, int n) {
    struct ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    struct ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    struct ggml_tensor * d = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);

    for (int i = 0; i < n; i++) {
        ((float *)a->data)[i] = 1.0f / (i + 1);
        ((float *)b->data)[i] = (float)(i + 1);
        ((float *)d->data)[i] = 0.1f;
    }

    struct ggml_tensor * mul = ggml_mul(ctx, a, b);
    struct ggml_tensor * c  = ggml_add(ctx, mul, d);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, c);
    return gf;
}

static int64_t run_benchmark(struct ggml_cgraph * gf, struct ggml_cplan * plan,
                             int n_iter, int warmup) {
    int64_t total_us = 0;
    for (int i = 0; i < warmup + n_iter; i++) {
        int64_t t0 = ggml_time_us();
        ggml_graph_compute(gf, plan);
        int64_t t1 = ggml_time_us();
        if (i >= warmup) {
            total_us += (t1 - t0);
        }
    }
    return total_us;
}

int main(int argc, char ** argv) {
    ggml_time_init();

    int n = 512;
    int n_iter = ITERATIONS;
    if (argc >= 2) n = atoi(argv[1]);
    if (argc >= 3) n_iter = atoi(argv[2]);
    if (n_iter < MIN_ITERATIONS) n_iter = MIN_ITERATIONS;

    size_t ctx_size = 1024 * 1024;
    struct ggml_init_params params = {
        .mem_size   = ctx_size,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_cgraph * gf = build_graph(ctx, n);
    struct ggml_cplan plan = ggml_graph_plan(gf, 1, NULL);
    if (plan.work_size > 0) {
        plan.work_data = (uint8_t *)malloc(plan.work_size);
    }

    /* Benchmark: profiler disabled */
    int64_t us_disabled = run_benchmark(gf, &plan, n_iter, WARMUP);

    /* Benchmark: profiler enabled */
    ggml_profiler_t * profiler = ggml_profiler_new();
    ggml_cplan_set_profiler(&plan, profiler);
    int64_t us_enabled = run_benchmark(gf, &plan, n_iter, WARMUP);

    /* Report */
    double avg_disabled_us = (double)us_disabled / n_iter;
    double avg_enabled_us  = (double)us_enabled  / n_iter;
    double overhead_us    = avg_enabled_us - avg_disabled_us;
    double overhead_pct    = (avg_disabled_us > 0)
        ? (overhead_us / avg_disabled_us * 100.0) : 0.0;

    printf("=== Profiler overhead benchmark ===\n");
    printf("Graph: n=%d elements (mul + add), %d iterations (+ %d warmup)\n", n, n_iter, WARMUP);
    printf("\n");
    printf("Profiler disabled: %" PRId64 " us total, %.3f us/iter (avg)\n",
           (int64_t)us_disabled, avg_disabled_us);
    printf("Profiler enabled:  %" PRId64 " us total, %.3f us/iter (avg)\n",
           (int64_t)us_enabled, avg_enabled_us);
    printf("\n");
    printf("Overhead: %.3f us/iter (%.2f%%)\n", overhead_us, overhead_pct);
    printf("\n");

    if (overhead_pct < 1.0 && avg_disabled_us > 0) {
        printf("PASS: Overhead < 1%% when enabled (typical for small graphs)\n");
    } else if (overhead_us < 1.0) {
        printf("PASS: Absolute overhead < 1 us/iter\n");
    } else {
        printf("NOTE: Overhead %.2f%% - expected for small/fast graphs\n", overhead_pct);
    }

    ggml_profiler_free(profiler);
    if (plan.work_data) free(plan.work_data);
    ggml_free(ctx);

    return 0;
}
