#pragma once

#include <string>
#include <string_view>

#include "gufo/ext/gguf_reader.hpp"

namespace gufo::models::qwen38_flash_next {

[[nodiscard]] bool ProbeFlashNext(std::string_view model_root);
[[nodiscard]] bool ProbeFlashNextGguf(const core::GgufReader& reader);
/// First-refusal claim: takes GSQ coder containers that the core loader
/// rejects (fused 3-D expert slabs / storage types outside the core table)
/// and leaves everything else to the native flow.
[[nodiscard]] bool ClaimGsqCoderFile(const std::string& model_path);

}  // namespace gufo::models::qwen38_flash_next
