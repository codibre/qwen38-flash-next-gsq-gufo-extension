# Swift 1.5 Qwen3.8-Flash-Next GSQ-RCO support

This extension loads the `ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF`
tiers in addition to the GSQ-RCO Coder variants it was developed against.
Swift 1.5 is a finetune of Qwen3.8-Flash-Next with the same `qwen4exp`
architecture and 512 routed experts, so the model path is the in-tree
Flash-Next fork; only the container encoding needed changes.

Verified tier: **IQ3_S** (2 shards, 83.74 GB, KLD 0.144674 vs Swift BF16),
HF revision `99bb8f7f95c7aa7b24a36a7786a4f657b30f5d3d`.

## Reader fixes (this branch)

Two standard Qwen3.8 GGUF encodings were rejected by `GsqReader`:

1. **`qwen4exp.attention.recurrent_layers` is an array of `bool`** (GGUF
   element type 7). `ReadArray` had no `kBool` arm, so the value failed with
   `c.ok == false`, the caller treated the header as truncated and kept
   growing its read buffer to the whole shard, then failed with
   `malformed GGUF header`. Bool arrays are now accepted and stored as 0/1
   `uint64` values with the declared element type (the metadata variant has no
   `vector<bool>`; no consumer reads this key directly).
2. **`split.count` is `u16`** in Swift's converter (the Coder files use
   `u64`). The loader only recognised `std::get_if<std::uint64_t>`, left
   `total_shards = 1`, and bound shard 1 only (373 of 1224 tensors). It is now
   read through `GetMetadataUint64`, which accepts u8/u16/u32/u64.

With both fixes the claim hook (`qwen4exp` + a 3-D `ffn_down_exps.weight`)
fires, both shards bind, and 1224 tensors load.

## Build and serve

```sh
# Build gufo with the extension (the core branch carries the rejected-GGUF
# fallback from gufo-org/gufo#966).
JOBS=8 GUFO_REF=feat/rejected-gguf-extension-fallback ./scripts/build.sh

hf download ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF \
  --revision 99bb8f7f95c7aa7b24a36a7786a4f657b30f5d3d \
  --include "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-0000*-of-00002.gguf" \
  --local-dir models/swift-iq3s

build-gufo/build/release/gufo serve llm \
  --model models/swift-iq3s/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
  --sessions 4 --context 262144 \
  --speculative mtp \
  --mtp-model <mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf>
```

Swift ships no draft head. The 512-expert shared Flash-Next MTP sidecar
(`unsloth/MTP-shared/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`) matches the
expert count and loads; acceptance measured 84% on a ~25-token burst.

## Measured (Strix Halo gfx1151, ROCm 10.0, 128 GiB unified)

Served with `--sessions 4 --context 262144`, MTP on, 1 GiB RAM cache, on a
100 GiB `amdttm` device pool (the 61 GiB default cannot claim 4 sessions):

| probe | this build | in-tree Flash-Next UD-Q4_K_XL |
|---|---|---|
| load | 13.2 s, 90.7 GiB device | 22 s, 39.6 GiB device |
| prefill, cold 8K / 16K | ~1169 / ~1159 t/s | ~2388 / ~2244 t/s |
| decode, 8K / 16K, 256 gen | 29.1 / 29.5 t/s | 38.3 / 42.0 t/s |
| retrieval 16K / 32K | 3/3, 3/3 | 3/3, 3/3 |
| tool calls | 8/8 | 8/8 |
| reasoning probe (5 prompts) | 5/5, 103 median tokens | 5/5, 110 median tokens |
| 4-way concurrency (64 gen) | 4x6.2 t/s (serial) | — |

The PLE/engram table stays disk-backed (`pread`, `ngram_disk_reads > 0`); RSS
after load is 2.3 GiB.

## Notes and limits

- IQ3_S only. The IQ2_XS tier stores routed experts as IQ2_XS, which this
  extension does not execute; the Q2_0 tier is experimental upstream.
- Four concurrent requests serialise on this build (`plan=serial-fallback`);
  slots buy headroom, not aggregate throughput.
- The published numbers are one box, one seed of prompts; the upstream KLD
  table remains the quant-quality reference.
