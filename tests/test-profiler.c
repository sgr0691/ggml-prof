// Simple test for GGML graph profiler
// Validates: JSON has required keys, stats are non-zero for a toy graph

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-profiler.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(_WIN32)
#include <io.h>
#define unlink _unlink
#else
#include <unistd.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

static int test_profiler_basic(void) {
    ggml_time_init();

    struct ggml_init_params params = {
        .mem_size   = 256 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    struct ggml_context * ctx = ggml_init(params);

    int n = 64;
    struct ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    struct ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    struct ggml_tensor * d = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);

    for (int i = 0; i < n; i++) {
        ((float *)a->data)[i] = 1.0f;
        ((float *)b->data)[i] = 2.0f;
        ((float *)d->data)[i] = 0.5f;
    }

    struct ggml_tensor * mul = ggml_mul(ctx, a, b);
    struct ggml_tensor * c  = ggml_add(ctx, mul, d);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, c);

    ggml_profiler_t * profiler = ggml_profiler_new();
    if (!profiler) {
        fprintf(stderr, "ggml_profiler_new failed\n");
        ggml_free(ctx);
        return 1;
    }

    struct ggml_cplan plan = ggml_graph_plan(gf, 1, NULL);
    if (plan.work_size > 0) {
        plan.work_data = (uint8_t *)malloc(plan.work_size);
    }
    ggml_cplan_set_profiler(&plan, profiler);

    ggml_graph_compute(gf, &plan);

    struct ggml_profiler_summary summary;
    ggml_profiler_get_summary(profiler, &summary);

    if (summary.n_nodes_executed == 0) {
        fprintf(stderr, "expected n_nodes_executed > 0\n");
        if (plan.work_data) free(plan.work_data);
        ggml_profiler_free(profiler);
        ggml_free(ctx);
        return 1;
    }

    if (summary.wall_ms_total < 0) {
        fprintf(stderr, "expected wall_ms_total >= 0\n");
        if (plan.work_data) free(plan.work_data);
        ggml_profiler_free(profiler);
        ggml_free(ctx);
        return 1;
    }

    if (plan.work_data) free(plan.work_data);
    ggml_profiler_free(profiler);
    ggml_free(ctx);
    return 0;
}

static int test_profiler_json_keys(void) {
    ggml_time_init();

    struct ggml_init_params params = {
        .mem_size   = 256 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    struct ggml_tensor * b = ggml_add(ctx, a, a);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, b);

    ggml_profiler_t * profiler = ggml_profiler_new();
    struct ggml_cplan plan = ggml_graph_plan(gf, 1, NULL);
    if (plan.work_size > 0) plan.work_data = (uint8_t *)malloc(plan.work_size);
    ggml_cplan_set_profiler(&plan, profiler);

    ggml_graph_compute(gf, &plan);

    const char * path = "test-profiler-out.json";
    FILE * fp = fopen(path, "w");
    if (!fp) {
        if (plan.work_data) free(plan.work_data);
        ggml_profiler_free(profiler);
        ggml_free(ctx);
        return 1;
    }
    ggml_profiler_dump_json(profiler, fp);
    fclose(fp);

    fp = fopen(path, "r");
    if (!fp) {
        unlink(path);
        if (plan.work_data) free(plan.work_data);
        ggml_profiler_free(profiler);
        ggml_free(ctx);
        return 1;
    }
    char buf[4096];
    size_t n = fread(buf, 1, sizeof(buf) - 1, fp);
    buf[n] = '\0';
    fclose(fp);
    unlink(path);

    int ok = 1;
    if (strstr(buf, "\"version\"") == NULL) ok = 0;
    if (strstr(buf, "\"summary\"") == NULL) ok = 0;
    if (strstr(buf, "\"wall_ms_total\"") == NULL) ok = 0;
    if (strstr(buf, "\"n_threads\"") == NULL) ok = 0;
    if (strstr(buf, "\"n_nodes_executed\"") == NULL) ok = 0;
    if (strstr(buf, "\"ops\"") == NULL) ok = 0;

    if (plan.work_data) free(plan.work_data);
    ggml_profiler_free(profiler);
    ggml_free(ctx);

    if (!ok) {
        fprintf(stderr, "JSON missing required keys. Output:\n%s\n", buf);
        return 1;
    }
    return 0;
}

int main(int argc, char ** argv) {
    (void)argc;
    (void)argv;

    if (test_profiler_basic() != 0) {
        fprintf(stderr, "test_profiler_basic failed\n");
        return 1;
    }

    if (test_profiler_json_keys() != 0) {
        fprintf(stderr, "test_profiler_json_keys failed\n");
        return 1;
    }

    printf("test-profiler: OK\n");
    return 0;
}

#ifdef __cplusplus
}
#endif
