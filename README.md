# Verum Bespoke Singularity

![macOS](https://img.shields.io/badge/platform-macOS-8e8e93)
![Objective-C](https://img.shields.io/badge/language-Objective--C-438eff)
![Metal](https://img.shields.io/badge/GPU-Metal-ff9f0a)
![License: MIT](https://img.shields.io/badge/license-MIT-34c759)

This tree is a clean restart from the uploaded prototype.  It intentionally has
no model-family switch and no fixed transformer layer struct.

The runtime is split into four native Objective-C responsibilities:

1. `VBSGGUFModel` keeps the GGUF file mmap-backed, preserves metadata, validates
   canonical GGML type geometry, and exposes block-aligned two-dimensional tiles.
2. `VBSGraphCompiler` derives a declarative operation graph from metadata, tensor
   names, shapes, and repeated scopes.  Ambiguous semantics are a hard error.
3. `VBSMetalCompiler` generates and compiles the exact MSL function required for
   an operation/encoding/tile shape.  There is no static Llama execution graph.
4. `VBSSliceExecutor` owns two bounded staging/private windows per GPU and streams
   rows *and* columns from mmap.  Tensor size is therefore independent of VRAM;
   only the selected packed tile and live activation state must fit.

`VBSFabric` treats multiple Metal devices as separate memory domains.  It probes
for peer groups when the SDK exposes them and otherwise uses an explicit shared
staging copy.  It never pretends that copying into a newly allocated destination
buffer changed the caller's source buffer.

## Build on the target Mac

```sh
make
./verum-slice --inspect /path/model.gguf
./verum-slice --compile /path/model.gguf
```

`--compile` validates the inferred graph and compiles its generated MSL kernels.
Execution is rejected when the file does not contain enough semantic evidence to
derive a unique graph; the engine does not guess a model family.

This clean restart currently ends at graph and GPU-harness compilation. It does
not falsely expose an interactive generation command before the inferred graph
executor and tokenizer are connected. The old tree could print `READY`, but it
could not correctly execute its own claimed memory or model contract.

## Recovered reference material (2026-09-30)

- `Sources/tokenizer.h` / `Sources/tokenizer.c` — HF `tokenizer.json` BPE
  implementation recovered from the 2026-09-19 iMac revision: UTF-8 JSON
  decoding, hash-map vocab, rank-based merges, GPT-style pre-tokenization,
  full 256-byte mapping, `tokenizer_eos_id()` API. **Not yet wired into the
  build or the runtime** — the Makefile currently compiles `Sources/*.m` only.
- `Sources/Reference/dequant_kernels.metal` — the 35-kernel MSL set from the
  same revision (dequant matvec / row-extract for F32/F16/Q8_0/Q4_0/Q4_1/
  Q5_0/Q5_1/Q2_K..Q6_K plus discrete-GPU `_amd` variants, rms_norm, rope,
  GQA attention, silu_mul, Gumbel-max sampling). **Reference only, not build
  input, not numerically validated against ggml** — the ground-truth
  candidate the runtime MSL generator must match. Parity tests required
  before any correctness claim.
