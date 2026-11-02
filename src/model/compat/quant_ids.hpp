#pragma once

// Vendored quant type ids that the out-of-tree extension must not require in
// the core GgmlType enum. GGUF files produced by GSQ-style quantizers may
// store expert weights with type id 42 (Q2_0); keeping the id local lets this
// extension build against plain gufo main, whose enum has no kQ2_0 member.
// When core grows the member (gufo#939's ext-api proposal), the value matches
// core's, so behavior is identical on both trees.

#include <gufo/ext/gguf_reader.hpp>

namespace gufo::models::qwen38_flash_next::compat {

inline constexpr gufo::core::GgmlType kQ2_0 = static_cast<gufo::core::GgmlType>(42);

}  // namespace gufo::models::qwen38_flash_next::compat
