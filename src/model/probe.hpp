#pragma once

#include <string>
#include <string_view>

#include "gufo/ext/gguf_reader.hpp"

namespace gufo::models::qwen38_flash_next {

[[nodiscard]] bool ProbeFlashNext(std::string_view model_root);
/// Accepts qwen4exp containers, including ones opened with
/// core::GgufOpenOptions::allow_unsized_tensors (GSQ coder artifacts);
/// the provider sets accepts_unsized_tensors and loads those tensors itself.
[[nodiscard]] bool ProbeFlashNextGguf(const core::GgufReader& reader);

}  // namespace gufo::models::qwen38_flash_next
