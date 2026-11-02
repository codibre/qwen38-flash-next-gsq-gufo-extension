#pragma once

#include <string>

#include "model/gguf/gsq_reader.hpp"

namespace gufo::models::qwen38_flash_next::gsq {

/// Stable artifact fingerprint for claimed containers: region sizes, each
/// region's head/tail bytes and the full tensor table (names, types,
/// offsets). Purpose is internal runner identity only; the scheme is
/// extension-owned and need not match the core identity hash.
[[nodiscard]] std::string GsqIdentityHex(const GsqReader& reader);

}  // namespace gufo::models::qwen38_flash_next::gsq
