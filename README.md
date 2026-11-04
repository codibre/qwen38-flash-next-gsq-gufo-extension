# Qwen3.8 Flash-Next GSQ — Gufo model extension

Out-of-tree [Gufo](https://github.com/gufo-org/gufo) model extension for
**Qwen3.8-Flash-Next GSQ-RCO** checkpoints: pruned 256-routed-expert variants
such as
[ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF),
with native IQ-codebook quants (IQ2_S/IQ3_XXS/IQ4_XS routed experts) and the
MTP draft handshake for the 256-expert family.

It also loads **Swift 1.5 GSQ-RCO** (`ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF`,
512-expert `qwen4exp` finetune) after the reader fixes in this branch; see
[SWIFT-1.5.md](SWIFT-1.5.md) for the two metadata encodings that needed support
and the served/bare numbers.

It is a vendored fork of the in-tree `qwen38_flash_next` extension carrying
the changes of [gufo-org/gufo#939](https://github.com/gufo-org/gufo/pull/939),
packaged in the shape the maintainers asked for new model/quant support: a
self-contained extension directory built into Gufo via
`GUFO_EXTENSION_DIRS`, like [gufo-org/lfm3-gufo-extension](https://github.com/gufo-org/lfm3-gufo-extension).

## Relationship to gufo-org/gufo#939

PR #939 has three layers, and only two of them can live in an extension:

| Layer | Where it lives now |
| --- | --- |
| Model support (config, weights, IQ decode kernels, tests, ~2.6k lines) | **this repository** (`src/model/`, `tests/`) |
| IQ block structs / codebooks / scalar accessors in the extension-visible quant header | this repository vendors its own copies; upstream #939 proposes publishing them in core (`include/gufo/ext/changes/0.1/native-quant-formats.md`) |
| Rejected-GGUF extension failover (`src/core/gguf_reader.*`, `src/ext/`, `src/cli/model_open.hpp`) | **cannot live in an extension** — the core parser rejects the file before any provider is consulted. Supersedes #939's loader hunk: strict parse fails -> re-offered to opted-in providers under `TensorLayoutValidation::kFusedExpertSlabs`, with extension-declared storage types. Until it merges, build Gufo from the branch in `scripts/build.sh` (`feat/rejected-gguf-extension-fallback`, drafted at `Farenheith/gufo#2`). |

When the failover lands on Gufo main, re-point `GUFO_REF` in
`scripts/build.sh` to `gufo-org/main` and this extension is pure
out-of-tree (see [TODO.md](TODO.md)).

## Why explicit extension selection is required

Selecting `GUFO_EXTENSIONS=qwen38_flash_next_gsq` puts the build in explicit
mode: the in-tree `qwen38_flash_next` is excluded from the shipped set, so
its probe (which claims every `qwen4exp` file) never competes with ours, its
artifact key no longer collides with ours in discovery validation, and the
vendored qwen-vision translation units do not collide with the ones the
`qwen` extension links. Verified: `gufo diagnose` reports `excluded
qwen38_flash_next ... not selected in explicit mode` with `extensions.lock`
untouched. Do **not** edit `extensions.lock` to exclude the in-tree model:
the Nix release package generates `mkServe` options from the lock, and
removing the entry silently drops its declared serve options (breaks
`mk-serve-nixos-tests`).

Consequence: this extension **replaces** the in-tree Flash-Next in the binary;
it serves both the 256-expert GSQ variants and the 512-expert original.

## Build

```sh
./scripts/build.sh            # clones/pin-builds gufo into ./build-gufo
```

or from an existing Gufo checkout:

```sh
cmake --preset release \
  -DGUFO_EXTENSION_DIRS=/path/to/qwen38-flash-next-gsq-gufo-extension \
  -DGUFO_EXTENSIONS=qwen38_flash_next_gsq
cmake --build --preset release --target gufo --parallel
./build/release/gufo diagnose --section extensions
```

Requirements: Linux, AMD Strix Halo / ROCm toolchain (`gfx1151` is the
qualification target), Gufo extension API >= 0.1.11 with #939's core hunk.

## Weights and usage

```sh
hf download ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF --local-dir models/gsq-coder
build/release/gufo chat --model models/gsq-coder/IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf
build/release/gufo serve llm --model <same shard 1> --context 200000
build/release/gufo bench --model <same shard 1> -p 2048 -n 128 -r 3
```

Sharded GGUFs: pass shard 1; the loader follows the shard set. MTP
speculative decoding with the matching 256-expert draft
(`Davidmg0815/Qwen3.8-Flash-Next-MTP-256exp-GGUF`) via
`--speculative mtp --mtp-model <draft>`; the base model's own 512-expert MTP
sidecar is rejected by design (expert-count mismatch, see
`MtpMatches`).

## Layout

```
extension.json          # manifest: provider flash-next-gsq-coder, --mtp-model option
extension.cmake         # gufo_ext_runtime-only targets (+ vendored mmq kernels)
src/model/              # flash-next model fork (from #939), includes rewritten to <gufo/ext/...>
src/family/             # qwen-family pieces needed by the model (headers; vision TUs compiled in)
tests/                  # projection + routed-WMMA op tests ported from #939
scripts/build.sh      # pinned-branch configure+build entry point
scripts/build.sh        # pinned build entry point
```

## Provenance and license

Derived from [gufo-org/gufo](https://github.com/gufo-org/gufo)
`src/models/qwen38_flash_next` + `src/families/qwen` at
`Farenheith/gufo@feat/qwen4exp-gsq-coder-support` (PR #939 head). Gufo is MIT
licensed; see [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md). Model weights are
distributed separately under their own licenses.
