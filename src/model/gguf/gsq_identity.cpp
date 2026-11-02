#include "model/gguf/gsq_identity.hpp"

#include <gufo/ext/sha256.hpp>

#include <algorithm>
#include <cstdint>

namespace gufo::models::qwen38_flash_next::gsq {
namespace {

void HashBytes(crypto::Sha256Hasher& hash, const void* data,
               std::size_t length) {
  hash.Update({static_cast<const std::uint8_t*>(data), length});
}

void HashString(crypto::Sha256Hasher& hash, std::string_view value) {
  HashBytes(hash, value.data(), value.size());
}

}  // namespace

std::string GsqIdentityHex(const GsqReader& reader) {
  crypto::Sha256Hasher hash;
  HashString(hash, "gsq-coder-identity-v1");
  for (const auto& region : reader.GetMappedRegions()) {
    HashString(hash, ":" + std::to_string(region.size) + ":");
    if (region.data != nullptr) {
      const std::size_t head =
          std::min<std::size_t>(region.size, 1U << 20);
      HashBytes(hash, region.data, head);
      if (region.size > head) {
        const std::size_t tail =
            std::min<std::size_t>(region.size - head, 1U << 16);
        HashBytes(hash, static_cast<const std::uint8_t*>(region.data) +
                            region.size - tail,
                  tail);
      }
    }
  }
  for (const auto& tensor : reader.Tensors()) {
    HashString(hash, tensor.name);
    std::string shape;
    for (const auto dimension : tensor.dimensions) {
      shape += "x" + std::to_string(dimension);
    }
    HashString(hash, shape + ":" +
                         std::to_string(static_cast<std::uint32_t>(
                             tensor.type)) +
                         ":" + std::to_string(tensor.offset));
  }
  return hash.FinishHex();
}

}  // namespace gufo::models::qwen38_flash_next::gsq
