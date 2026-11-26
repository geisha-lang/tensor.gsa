# tensor.gsa — Architecture & Engineering Rules

## Design Philosophy
- **Pure Geisha first**: CPU kernels (`src/cpu.gsa`), Apple Metal + MPS (`src/metal.gsa`), reverse-mode AD (`src/tensor.gsa`, `src/ad.gsa`), JIT fusion (`src/jit.gsa`), optimizers (`src/optim.gsa`), and Safetensors I/O (`src/safetensors.gsa`) are written in pure Geisha without C glue.
- **Keep `tensor.gsa` domain-agnostic**: Do not put model- or task-specific custom kernels (e.g. RWKV-7 recurrence scans) into `tensor.gsa`. Downstream projects should build custom ops on top of `customVjp` and `tensor.cuda` / `tensor.metal`.
- **Declarative native builds**: Use `[[target.native.when]]` in `geisha.toml` for platform/toolchain rules (`os`, `tool`, `pkg_config`). Never add hardware-specific logic to the `geisha` compiler.
- **No dead C glue**: Avoid C vtables or redundant header files in `native/`.

## Critical Pitfalls & Rules

### 1. FFI `extern noalloc` for Async GPU Launches
- In Geisha, plain `extern "sym"` wraps every call in `_setjmp` + `gc_enter_blocking`/`gc_leave_blocking`.
- Every **asynchronous** CUDA kernel launch or lightweight query in `src/cuda.gsa` **must** use `extern noalloc "gs_cu_*"`; otherwise hundreds of microsecond-scale launches per training step burn significant CPU overhead.
- Reserve plain `extern` only for genuinely blocking calls (`gs_cu_sync`, `gs_cu_h2d`, `gs_cu_d2h`, reductions with D2H readback like `gs_cu_sum_all` / `gs_cu_sum_sq` / `gs_cu_ce_f` / `gs_cu_ce_b`, and host-side `gs_cu_randn`).

### 2. CUDA `block_reduce_*` Trailing `__syncthreads()`
- In `native/cuda.cu`, `block_reduce_sum` and `block_reduce_max` use static `__shared__ float sh[256]`.
- Both helpers **must** end with `__syncthreads()` after reading `float res = sh[0]` and before returning.
- Without the trailing barrier, back-to-back calls inside the same kernel (`k_rms_b`, `k_ln_b`, `k_sm_f`, `k_ce_fwd_loss`) let fast warps overwrite `sh[tid]` for the second reduction before slow warps read `sh[0]` from the first, causing silent data races and NaN by step 2.

### 3. Apple Metal ARM64 ABI & Unified Memory
- `MTLSize` is 24 bytes (> 16B), so AAPCS64 passes it **indirectly by pointer** (`withAlloca(24, ...)` `ScopedPtr`), never as 3 integer register arguments to `objc_msgSend`.
- Metal tensors use Unified Memory (`MTLResourceStorageModeShared = 0`) with deferred `MTLCommandBuffer` batching. Any CPU read/write on a Metal tensor **must** call `syncDevice(dev)` first to commit and wait for pending GPU work.

### 4. JIT Fusion Backend Safety (`src/jit.gsa`)
- `jit1` / `jit2` / `jit3` must guard against unsupported devices before tracing: until NVRTC execution is wired for CUDA, `Cuda` tensors must fall back to eager `f(...)` so `executeTrace` never dereferences a CUDA device pointer on CPU.
- Keep JIT topology signatures **shape-agnostic and scalar-value-agnostic** (lift all scalar constants into `tr.scalars`, pass `numel` at runtime).
- In `src/ops.gsa`, route JIT-traceable ops through `tryJitBin` / `tryJitUnary` rather than duplicating eager + `customVjp` blocks.

### 5. Cross-Platform Static `extern` Stubs
- Because Geisha currently resolves `extern "sym"` statically without indirect function-pointer calls:
  - `native/cuda_weak_stubs.c` provides `__attribute__((weak))` no-op fallbacks for `gs_cu_*` when `nvcc` is absent; `native/cuda.cu` overrides them with strong definitions when `nvcc` is present.
  - `native/metal_stubs_linux.c` is compiled **only** on Linux (`[[target.native.when]] os = "linux"`) to satisfy link-time Metal/ObjC symbols.
- Whenever adding, removing, or changing the signature of any `gs_cu_*` symbol, update **all three** files together: `native/cuda.cu`, `native/cuda_weak_stubs.c`, and `src/cuda.gsa`.

### 6. CUDA Async Memory Pool, Uninitialized Allocations & Tensor Cores
- `allocTensorOn` on CUDA allocates from the default async memory pool (`cudaMallocAsync` / `cudaFreeAsync`, `ReleaseThreshold = UINT64_MAX`) **without zero-filling**. Any op that accumulates into an output buffer (e.g., `catAxis` or custom atomic kernels) **must** use `zerosOn` or `cudaZero` (`cudaMemsetAsync`), never assume `allocTensorOn` is zeroed.
- Never call synchronous `cudaMemset` in `native/cuda.cu` — always use `cudaMemsetAsync(..., 0)`.
- `cublasSgemm` enables `CUBLAS_TF32_TENSOR_OP_MATH` by default; `setCudaBf16(true)` switches GEMM/BGEMM to `cublasGemmEx` with `CUBLAS_COMPUTE_32F_FAST_16BF` (keeping FP32 buffers/accumulators).

### 7. Geisha Language Gotchas
- `Int` literals are 32-bit; express 64-bit constants via `Int64` conversions or keep 64-bit RNG bit-mixing in C/CUDA.
- Lambda parameters that access record fields need explicit type annotations (e.g. `(g: Tensor) -> ...`).

## Verification
- Run `~/.geisha/bin/geisha test` on macOS (verifies CPU, broadcasting, gradcheck, AdamW, Muon, HF Safetensors, Metal/MPS, JIT fusion zero-recompile, and activation checkpointing).
- When modifying `native/cuda.cu`, also verify on a CUDA host (`nvcc` available) that `geisha test` compiles and runs cleanly.
