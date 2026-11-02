# TODO

## Verification status (update honestly with each push)

- [x] Include boundary: no `src/...` includes outside the repository except
      forwarded public headers (`<gufo/ext/...>`).
- [x] Gufo configure with `GUFO_EXTENSION_DIRS` (isolation validator green,
      2026-11-02, gufo @59f94a4a + lock patch, `extension mode=explicit;
      included=[qwen38_flash_next_gsq]`).
- [x] Compile green: full `gufo` binary with the extension (`--target gufo`,
      2026-11-02, gfx1151).
- [x] `gufo diagnose --section extensions` lists
      `qwen38_flash_next_gsq 0.1.0` with `provider flash-next-gsq-coder:
      available`.
- [ ] `ctest -L qwen38_flash_next_gsq` (projection ops, routed WMMA ops).
      Blocked upstream: extension test executables cannot reach core
      implementations (`gufo_core` is link-prohibited for external targets
      and `gufo_ext_runtime` exposes headers, not their objects), so
      out-of-tree tests that touch `GgufReader`/`HostAllocations`/crypto
      cannot link. The same test bodies pass upstream as part of
      gufo#939's suite against byte-identical code paths.
- [x] Load smoke E2E via llama-swap `intermediate` (2026-11-02): gufo main +
      rejected-GGUF fallback PR + this extension loads the GSQ Coder IQ1_M
      shard (`load_completed elapsed_ms=6532`, GPU 43.9 GB) and generates
      (28 tok/s decode, MTP drafts 39/61 accepted, coherent reasoning).
      Plain main without the PR fails at strict parse on
      `blk.1.ffn_down_exps.weight`.
- [ ] Full-answer smoke with a larger token budget (64-token cap only showed
      the reasoning channel; content came back empty for that reason).
- [ ] MTP smoke with the 256-expert draft.
- [ ] Benchmark parity vs. in-tree #939 build.

## Verified E2E recipe
- gufo branch `feat/rejected-gguf-extension-fallback` (main + core failover:
  strict parse → re-offer to `handles_layout_rejected` providers under
  `TensorLayoutValidation::kFusedExpertSlabs`, plus
  `ext::RegisterUnknownQuantFormat` for extension-declared storage types).
  All unit coverage lives in that PR (`gguf_reader_test`,
  `ext_runtime_neutral_contracts_test`).
- No pin to #939 is required anymore: `kQ2_0` (id 42) is a vendored compat
  constant, its geometry (64 el/18 B) is registered in `provider.cpp`, and
  every model/draft open uses the slab policy (`extension.json` now requires
  API `>= 0.1.16`).

## Known blockers / caveats

- Depends on the rejected-GGUF fallback PR being present in the host build
  (it replaces the old #939 core-loader requirement). Until it merges,
  `scripts/build.sh` defaults to `GUFO_REF=feat/rejected-gguf-extension-fallback`.
- Provider probes are first-claim-wins over registration order; the in-tree
  `qwen38_flash_next` probe claims every `qwen4exp` file. Explicit
  `GUFO_EXTENSIONS` selection excludes it from the shipped set (verified
  with `extensions.lock` intact); the old `patches/gufo-lock.patch` was
  removed because editing the lock breaks the Nix release-package serve
  options. If Gufo ever adds probe specificity ranking, this can coexist
  with the in-tree model without explicit mode.
- Building this extension together with the `qwen` extension duplicates the
  vendored `gufo::models::qwen::vision` symbols at final link. Exclude
  `qwen` from the selection (default in `scripts/build.sh`).

## Porting backlog

- [ ] Upstream quirk to report: with `BUILD_TESTING=ON`, core
      `tests/models/*` link in-tree extension targets even when those
      extensions are excluded by `GUFO_EXTENSIONS`, so an extension-only
      selection cannot configure with tests. Workaround used for
      `ctest -L qwen38_flash_next_gsq`: explicitly select every other
      extension as well (and keep `qwen38_flash_next` out of
      `extensions.lock`), then build only this extension's test targets.
- [ ] Windows packaging (upstream model supports the platform; we vendored
      Linux only so far).
- [ ] Functional suite (upstream `functional/` registry) as an extension-side
      script.
- [ ] `reference.cpp` / `cpu_ops.cpp` reference decoder + tools (need the
      compare harness; only used by validation tooling today).
- [ ] Docs: QUALITY/BENCHMARKS in lfm3-extension style.
- [ ] Reconcile with #939 reviews: re-vendor on upstream PR changes until it
      merges, then rebase onto main and drop the pin.
