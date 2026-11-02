# Notices

This repository is a derivative work of [gufo-org/gufo](https://github.com/gufo-org/gufo)
(MIT, Copyright (c) 2026 gufo contributors), vendored at
`Farenheith/gufo@59f94a4a` (`src/models/qwen38_flash_next`, parts of
`src/families/qwen`, `src/testing/compare/logit_comparator.hpp`), together
with the changes contributed in gufo-org/gufo#939.

`src/model/kernels/rocm/mmq/` derives from ggml/llama.cpp quantized GEMM
kernels (MIT, The ggml authors), as vendored by Gufo.

Model weights (Qwen, GSQ-RCO Coder, MTP drafts) are separate artifacts under
their own licenses and are not distributed here.
