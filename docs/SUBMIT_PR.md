# How to Submit the Profiler to ggml-org/llama.cpp

This guide walks through submitting the GGML Graph Profiler as a pull request
to `ggml-org/llama.cpp`. The ggml library lives under the `ggml/` subdirectory
there, so a path remapping is needed.

---

## 1. One-time Setup

```bash
# Fork ggml-org/llama.cpp on GitHub (use the web UI), then clone your fork:
git clone https://github.com/<your-github-username>/llama.cpp
cd llama.cpp
git remote add upstream https://github.com/ggml-org/llama.cpp
git fetch upstream
```

## 2. Create a Branch

```bash
git checkout -b feat/ggml-graph-profiler upstream/master
```

## 3. Apply the Patch

A pre-generated patch with paths already adjusted for llama.cpp is at:
`docs/../profiler-llamacpp.patch` in this repo.

Copy it over and apply:

```bash
# From inside your llama.cpp clone:
git apply /path/to/profiler-llamacpp.patch
```

If `git apply` rejects some hunks (due to upstream changes), apply manually
using the **File Mapping** table below.

---

## File Mapping: ggml repo → llama.cpp repo

| This repo (`ggml-org/ggml` fork)         | llama.cpp target                              |
|------------------------------------------|-----------------------------------------------|
| `CMakeLists.txt`                         | `ggml/CMakeLists.txt`                         |
| `include/ggml-profiler.h`               | `ggml/include/ggml-profiler.h`               |
| `include/ggml-cpu.h`                    | `ggml/include/ggml-cpu.h`                    |
| `src/ggml-profiler.c`                   | `ggml/src/ggml-profiler.c`                   |
| `src/CMakeLists.txt`                    | `ggml/src/CMakeLists.txt`                    |
| `src/ggml-cpu/ggml-cpu.c`              | `ggml/src/ggml-cpu/ggml-cpu.c`              |
| `tests/test-profiler.c`                 | `ggml/tests/test-profiler.c`                 |
| `tests/bench-profiler-overhead.c`       | `ggml/tests/bench-profiler-overhead.c`       |
| `tests/CMakeLists.txt` (additions)      | `ggml/tests/CMakeLists.txt`                  |
| `examples/simple/simple-profiler.cpp`  | `examples/simple/simple-profiler.cpp`        |
| `examples/simple/CMakeLists.txt` (add) | `examples/simple/CMakeLists.txt`             |
| `docs/profiling.md`                     | `docs/profiling.md`                           |

---

## 4. Build and Test

```bash
cmake -B build -DGGML_BUILD_TESTS=ON -DGGML_BUILD_EXAMPLES=ON
cmake --build build --parallel

# Run profiler unit tests
./build/bin/test-profiler

# Run overhead benchmark
./build/bin/bench-profiler-overhead

# Run the example
./build/bin/simple-profiler
```

## 5. Commit

```bash
git add ggml/include/ggml-profiler.h \
        ggml/include/ggml-cpu.h \
        ggml/src/ggml-profiler.c \
        ggml/src/CMakeLists.txt \
        ggml/src/ggml-cpu/ggml-cpu.c \
        ggml/CMakeLists.txt \
        ggml/tests/test-profiler.c \
        ggml/tests/bench-profiler-overhead.c \
        ggml/tests/CMakeLists.txt \
        examples/simple/simple-profiler.cpp \
        examples/simple/CMakeLists.txt \
        docs/profiling.md

git commit -m "ggml : add graph profiler (op-level timing, memory estimate, JSON export)"
```

## 6. Push and Open the PR

```bash
git push -u origin feat/ggml-graph-profiler
```

Then open the PR on GitHub:
- **Base:** `ggml-org/llama.cpp` → `master`
- **Head:** `<your-username>/llama.cpp` → `feat/ggml-graph-profiler`
- **Title:** `ggml : add graph profiler (op-level timing, memory estimate, JSON export)`
- **Body:** copy from `docs/PR_PROFILER.md`

---

## PR Checklist (llama.cpp contribution requirements)

- [ ] No compiler warnings on default build
- [ ] `test-profiler` passes
- [ ] `bench-profiler-overhead` runs and shows ~0 overhead when disabled
- [ ] Changes are additive/opt-in — existing behavior unchanged
- [ ] Public API added to `ggml/include/ggml-profiler.h` (new file)
- [ ] `ggml_cplan` field `profiler` defaults to `NULL` (no-op when unset)
- [ ] Example works: `./bin/simple-profiler`
