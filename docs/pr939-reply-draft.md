# Draft — comment for gufo-org/gufo#939 (do not paste blind; review first)

Follow-up on the extension-system suggestion: I packaged the model side of
this PR as an out-of-tree extension —
**[codibre/qwen38-flash-next-gsq-gufo-extension](https://github.com/codibre/qwen38-flash-next-gsq-gufo-extension)** —
in the shape of `gufo-org/lfm3-gufo-extension` (`extension.json` +
`extension.cmake` + sources under `<gufo/ext/...>` only; builds via
`GUFO_EXTENSION_DIRS` with `GUFO_EXTENSIONS=qwen38_flash_next_gsq`;
configure passes the isolation validator).

Findings from the port, in case they're useful for the RFC:

1. **A ~19-line piece of this PR cannot live in an extension**: the fused
   3-D expert-tensor alignment rule in `GgufReader::ParseHeaders` runs before
   any provider/probe is consulted, so GSQ 256-expert GGUFs (k=640 vs a
   256-element block) are rejected at parse time. The extension README pins a
   ref carrying exactly that hunk until it merges.
2. The `include/gufo/ext/changes/0.1/native-quant-formats.md` proposal (IQ
   block structs + codebooks + accessors) is the other core-side wish; the
   extension vendors its own copies today, so it is a "nice to have", not a
   blocker.
3. Two discovery behaviors worth documenting for extension authors: the
   shipped/unshipped key/option tier check (an external extension reusing
   `gguf:qwen4exp` / `--mtp-model` hard-fails configure while the in-tree
   model remains locked), and first-claim-wins probe ordering (the in-tree
   `qwen4exp` probe would shadow any coder-specific extension). A related
   test-side gap: extension `role:test` executables can't link core
   implementations (`gufo_core` is allowlist-prohibited), so out-of-tree
   suites that use `GgufReader`/crypto can't currently link; `lfm3` builds
   its extension but I hit this the moment I tried `ctest`.

So the split that works long-term: the loader alignment (+ optionally the
quant-format publish) upstream, model/kernel code in the extension. Happy to
narrow this PR to the core half if that's the preferred shape.
