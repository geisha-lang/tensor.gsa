# tensor.gsa

Functional, zero-boilerplate Deep Learning framework for **Geisha**.

- **Plain ADTs as Models**: `data Linear : Params, StateDict { weight: Tensor, bias: Option Tensor }` — zero `Mk*`, zero proc-macros, zero `.grad` / `zero_grad()`.
- **Local-Tape Reverse-Mode AD**: `valueAndGrad(model, lossFn)` with `customVjp` hook for fused kernels.
- **Multi-Backend Native Kernels (`GsTensorBackend`)**:
  - **CPU**: portable cache-tiled C11 (`-DGS_TENSOR_NO_ACCELERATE`), Linux OpenBLAS (`-DGS_TENSOR_OPENBLAS`), or macOS Apple Accelerate (`cblas_sgemm` + `vDSP`).
  - **CUDA**: `native/gs_backend_cuda.cu` (`cublasSgemm` / `cublasSgemmStridedBatched` + CUDA kernels, enable with `-DGS_TENSOR_CUDA -lcudart -lcublas`).
- **Optimizers**: Pure functional `AdamW` and `Muon` (5-step Newton-Schulz orthogonalization for 2D matrices).
- **HuggingFace Interoperability**: Pure-Geisha `safetensors` reader/writer + `StateDict` auto-deriving (`Named` field paths match `LlamaForCausalLM` 1-to-1) + `config.json` exporter.

## Quickstart

```sh
geisha test
```

## Example

```geisha
use tensor.tensor.{Cpu, fromI32List, crossEntropy}
use tensor.ad.valueAndGrad
use tensor.nn.{LMConfig, rngNew, initCausalLM, causalLMFwd}
use tensor.optim.{defaultMuon, muonInit, muonStep}
use tensor.safetensors.saveSafetensors

let cfg = LMConfig {
    vocabSize: 16, hiddenSize: 32, intermediateSize: 64,
    numLayers: 2, numHeads: 4, rmsNormEps: 1.0e-5, ropeTheta: 10000.0,
    device: Cpu,
};
let (lm0, _) = initCausalLM(cfg, rngNew(42));
let (loss, grads) = valueAndGrad(lm0, (m) -> crossEntropy(causalLMFwd(m, inp), tgt));
let (lm1, st1) = muonStep(defaultMuon(0.03), muonInit(lm0), lm0, grads);
let _ = saveSafetensors("model.safetensors", lm1);
```

## Building with Pure C11 / OpenBLAS / CUDA

- **macOS (default)**: `geisha.toml` uses Apple Accelerate automatically.
- **Linux (pure C11 CPU)**: remove `-framework Accelerate` from `geisha.toml` `link_flags` (uses portable cache-tiled SGEMM in `native/gs_backend_cpu.c`).
- **Linux (OpenBLAS)**: add `-DGS_TENSOR_OPENBLAS` to `c_flags` and `-lopenblas` to `link_flags`.
- **Linux (CUDA + cuBLAS)**: compile `native/gs_backend_cuda.cu` with `nvcc -O3 -DGS_TENSOR_CUDA -c`, add `-DGS_TENSOR_CUDA` to `c_flags`, and link `-lcudart -lcublas`.
