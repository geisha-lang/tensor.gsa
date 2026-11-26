# tensor.gsa

Functional, zero-boilerplate Deep Learning framework for **Geisha**.

- **Plain ADTs as Models**: `data Linear : Params, StateDict { weight: Tensor, bias: Option Tensor }` — zero `Mk*`, zero proc-macros, zero `.grad` / `zero_grad()`.
- **Local-Tape Reverse-Mode AD**: `valueAndGrad(model, lossFn)` with `customVjp` hook for fused kernels.
- **Multi-Backend Native Kernels (`GsTensorBackend`)**:
  - **CPU**: portable cache-tiled C11 (`-DGS_TENSOR_NO_ACCELERATE`), Linux OpenBLAS (`-DGS_TENSOR_OPENBLAS`), or macOS Apple Accelerate (`cblas_sgemm` + `vDSP`).
  - **CUDA**: `native/cuda.cu` (`cublasSgemm` / `cublasSgemmStridedBatched` + CUDA kernels, auto-detected when `nvcc` is on `PATH`).
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

## Multi-Backend Auto-Detection

`tensor.gsa` declares platform and toolchain rules via `[[target.native.when]]` in `geisha.toml` — downstream apps just add `tensor` to `[deps]` and run `geisha build` / `geisha test` with zero manual flags:

- **macOS**: automatically enables Apple Accelerate (`cblas_sgemm` + `vDSP`).
- **Linux (CPU)**: automatically detects `openblas` via `pkg-config`, falling back to portable cache-tiled C11.
- **Linux (NVIDIA GPU)**: automatically detects `nvcc` on `PATH`, compiles `native/cuda.cu`, and links `cudart` + `cublas` (disable with `GEISHA_DISABLE_TOOL=nvcc`).
