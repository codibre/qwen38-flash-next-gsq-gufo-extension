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

}  // namespace gufo::models::qwen38_flash_next
