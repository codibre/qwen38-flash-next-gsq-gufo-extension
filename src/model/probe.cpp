#include "model/probe.hpp"
#include "family/text/tokenizer.hpp"

#include <filesystem>
#include <string>

#include "model/gguf/gsq_reader.hpp"

namespace gufo::models::qwen38_flash_next {



bool ProbeFlashNextGguf(const core::GgufReader& reader) {
  const auto architecture =
      reader.GetMetadataString("general.architecture").value_or("");
  return architecture == "qwen4exp";
}

bool ProbeFlashNext(std::string_view model_root) {
  try {
    std::filesystem::path p(model_root);
    if (!std::filesystem::is_regular_file(p)) {
      return false;
    }
    std::string err;
    const auto reader =
        gsq::GsqReader::OpenFile(p.string(), &err, /*map_data=*/false);
    return reader != nullptr &&
           reader->GetMetadataString("general.architecture").value_or("") ==
               "qwen4exp";
  } catch (...) {
    return false;
  }
}

bool ClaimGsqCoderFile(const std::string& model_path) {
  try {
    std::string err;
    const auto reader =
        gsq::GsqReader::OpenFile(model_path, &err, /*map_data=*/false);
    if (reader == nullptr ||
        reader->GetMetadataString("general.architecture").value_or("") !=
            "qwen4exp") {
      return false;
    }
    // Claim only what the core strict loader cannot parse today: fused
    // expert weight slabs (3-D) that native qwen4exp handling rejects.
    // Standard flash-next GGUFs keep flowing through the native providers.
    for (const auto& tensor : reader->Tensors()) {
      if (tensor.dimensions.size() == 3 &&
          tensor.name.size() >= 20 &&
          tensor.name.compare(tensor.name.size() - 20, 20,
                              "ffn_down_exps.weight") == 0) {
        return true;
      }
    }
    return false;
  } catch (...) {
    return false;
  }
}

}  // namespace gufo::models::qwen38_flash_next
