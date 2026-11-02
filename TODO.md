# TODO

## Verification status (update honestly with each push)

- [x] Include boundary: no `src/...` includes outside the repository except
      forwarded public headers (`<gufo/ext/...>`).
- [x] Gufo configure with `GUFO_EXTENSION_DIRS` (isolation validator green,
      2026-11-02, gufo @59f94a4a + lock patch, `extension mode=explicit;
      included=[qwen38_flash_next_gsq]`).
- [ ] Compile green: `gufo_qwen38_flash_next_gsq`, `..._hip`, `qwen38_flash_next_gsq_mmq`.
- [ ] `ctest -L qwen38_flash_next_gsq` (projection ops, routed WMMA ops).
- [ ] `gufo diagnose --section extensions` lists `qwen38_flash_next_gsq`.
- [ ] Load smoke: GSQ Coder IQ1_M shard 1 loads and answers a prompt.
- [ ] MTP smoke with the 256-expert draft.
- [ ] Benchmark parity vs. in-tree #939 build.

## Known blockers / caveats

- Core GGUF loader alignment (the 19-line `gguf_reader.cpp` hunk of #939) is
  required before load; shipped today only via the pinned gufo ref. Exit
  paths: #939 core half merges upstream, or the extension grows its own GGUF
  tensor reader (not planned).
- Provider probes are first-claim-wins over registration order; the in-tree
  `qwen38_flash_next` probe claims every `qwen4exp` file. Hence the lock
  patch + explicit `GUFO_EXTENSIONS` selection. If Gufo ever adds probe
  specificity ranking, this can coexist with the in-tree model instead.
- Building this extension together with the `qwen` extension duplicates the
  vendored `gufo::models::qwen::vision` symbols at final link. Exclude
  `qwen` from the selection (default in `scripts/build.sh`).
- `api_version ">= 0.1.11"` cannot express the #939 core-header additions;
  the pin covers it today. If upstream publishes the quant-format change as a
  new API version, raise the floor accordingly.

## Porting backlog

- [ ] Windows packaging (upstream model supports the platform; we vendored
      Linux only so far).
- [ ] Functional suite (upstream `functional/` registry) as an extension-side
      script.
- [ ] `reference.cpp` / `cpu_ops.cpp` reference decoder + tools (need the
      compare harness; only used by validation tooling today).
- [ ] Docs: QUALITY/BENCHMARKS in lfm3-extension style.
- [ ] Reconcile with #939 reviews: re-vendor on upstream PR changes until it
      merges, then rebase onto main and drop the pin.
