#include "model/gguf/gsq_reader.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <variant>

namespace gufo::models::qwen38_flash_next::gsq {
namespace {

constexpr std::uint32_t kMagic = 0x46554747u;  // "GGUF"
constexpr std::size_t kGrowStep = 8u << 20;

struct Cursor {
  const char* p{nullptr};
  const char* end{nullptr};
  bool ok{true};

  template <typename T>
  [[nodiscard]] T Read() {
    if (end - p < static_cast<std::ptrdiff_t>(sizeof(T))) {
      ok = false;
      return {};
    }
    T value;
    std::memcpy(&value, p, sizeof(T));
    p += sizeof(T);
    return value;
  }

  [[nodiscard]] std::string_view ReadString() {
    const auto length = Read<std::uint64_t>();
    if (!ok || static_cast<std::uint64_t>(end - p) < length) {
      ok = false;
      return {};
    }
    std::string_view value(p, static_cast<std::size_t>(length));
    p += static_cast<std::size_t>(length);
    return value;
  }
};

bool ReadValueInto(Cursor& c, GgufValueType type, GgufMeta* out);

bool ReadArray(Cursor& c, GgufMeta* out) {
  const auto element_type =
      static_cast<GgufValueType>(c.Read<std::uint32_t>());
  const auto count = c.Read<std::uint64_t>();
  if (!c.ok) {
    return false;
  }
  switch (element_type) {
    case GgufValueType::kString: {
      if (count > static_cast<std::uint64_t>(c.end - c.p)) {
        c.ok = false;
        return false;
      }
      std::vector<std::string_view> values;
      values.reserve(static_cast<std::size_t>(count));
      for (std::uint64_t i = 0; i < count; ++i) {
        values.push_back(c.ReadString());
        if (!c.ok) {
          return false;
        }
      }
      out->type = element_type;
      out->value = std::move(values);
      return true;
    }
    case GgufValueType::kBool: {
      // Qwen3.8 GGUF stores attention.recurrent_layers as [bool]. The metadata
      // variant has no vector<bool>, so keep the values as 0/1 u64 with the
      // declared element type; no consumer reads this key directly.
      std::vector<std::uint64_t> values;
      values.reserve(static_cast<std::size_t>(count));
      for (std::uint64_t i = 0; i < count; ++i) {
        GgufMeta item;
        if (!ReadValueInto(c, element_type, &item)) {
          return false;
        }
        values.push_back(std::get<bool>(item.value) ? 1U : 0U);
      }
      out->type = element_type;
      out->value = std::move(values);
      return true;
    }
    case GgufValueType::kUint8:
    case GgufValueType::kUint16:
    case GgufValueType::kUint32:
    case GgufValueType::kUint64: {
      std::vector<std::uint64_t> values;
      values.reserve(static_cast<std::size_t>(count));
      for (std::uint64_t i = 0; i < count; ++i) {
        GgufMeta item;
        if (!ReadValueInto(c, element_type, &item)) {
          return false;
        }
        values.push_back(std::get<std::uint64_t>(item.value));
      }
      out->type = element_type;
      out->value = std::move(values);
      return true;
    }
    case GgufValueType::kInt8:
    case GgufValueType::kInt16:
    case GgufValueType::kInt32:
    case GgufValueType::kInt64: {
      std::vector<std::int64_t> values;
      values.reserve(static_cast<std::size_t>(count));
      for (std::uint64_t i = 0; i < count; ++i) {
        GgufMeta item;
        if (!ReadValueInto(c, element_type, &item)) {
          return false;
        }
        values.push_back(std::get<std::int64_t>(item.value));
      }
      out->type = element_type;
      out->value = std::move(values);
      return true;
    }
    case GgufValueType::kFloat32:
    case GgufValueType::kFloat64: {
      std::vector<double> values;
      values.reserve(static_cast<std::size_t>(count));
      for (std::uint64_t i = 0; i < count; ++i) {
        GgufMeta item;
        if (!ReadValueInto(c, element_type, &item)) {
          return false;
        }
        values.push_back(std::get<double>(item.value));
      }
      out->type = element_type;
      out->value = std::move(values);
      return true;
    }
    default:
      c.ok = false;
      return false;
  }
}

bool ReadValueInto(Cursor& c, GgufValueType type, GgufMeta* out) {
  switch (type) {
    case GgufValueType::kUint8:
      out->value = static_cast<std::uint64_t>(c.Read<std::uint8_t>());
      break;
    case GgufValueType::kInt8:
      out->value = static_cast<std::int64_t>(c.Read<std::int8_t>());
      break;
    case GgufValueType::kUint16:
      out->value = static_cast<std::uint64_t>(c.Read<std::uint16_t>());
      break;
    case GgufValueType::kInt16:
      out->value = static_cast<std::int64_t>(c.Read<std::int16_t>());
      break;
    case GgufValueType::kUint32:
      out->value = static_cast<std::uint64_t>(c.Read<std::uint32_t>());
      break;
    case GgufValueType::kInt32:
      out->value = static_cast<std::int64_t>(c.Read<std::int32_t>());
      break;
    case GgufValueType::kFloat32:
      out->value = static_cast<double>(c.Read<float>());
      break;
    case GgufValueType::kBool:
      out->value = c.Read<std::uint8_t>() != 0;
      break;
    case GgufValueType::kString:
      out->value = c.ReadString();
      break;
    case GgufValueType::kUint64:
      out->value = c.Read<std::uint64_t>();
      break;
    case GgufValueType::kInt64:
      out->value = c.Read<std::int64_t>();
      break;
    case GgufValueType::kFloat64:
      out->value = c.Read<double>();
      break;
    case GgufValueType::kArray:
      return ReadArray(c, out);
    default:
      c.ok = false;
      return false;
  }
  out->type = type;
  return c.ok;
}

struct LocalTensor {
  std::string_view name;
  std::vector<std::uint64_t> dimensions;
  std::uint32_t type{0};
  std::uint64_t offset{0};
};

struct ParsedShard {
  std::vector<std::pair<std::string, GgufMeta>> kv;
  std::vector<LocalTensor> tensors;
  std::uint64_t header_length{0};
  bool truncated{false};  // prefix too short: caller may grow and retry
  bool fatal{false};      // malformed beyond repair
};

ParsedShard ParseHeaderSection(const char* base, std::size_t length) {
  ParsedShard parsed;
  Cursor c{base, base + length};
  if (c.Read<std::uint32_t>() != kMagic) {
    parsed.fatal = true;
    return parsed;
  }
  const std::uint32_t version = c.Read<std::uint32_t>();
  if (version < 2 || version > 3) {
    parsed.fatal = true;
    return parsed;
  }
  const std::uint64_t n_tensors =
      version >= 3 ? c.Read<std::uint64_t>() : c.Read<std::uint32_t>();
  const std::uint64_t n_kv = c.Read<std::uint64_t>();
  if (!c.ok) {
    parsed.truncated = true;
    return parsed;
  }
  for (std::uint64_t i = 0; i < n_kv; ++i) {
    const std::string_view key = c.ReadString();
    const auto type = static_cast<GgufValueType>(c.Read<std::uint32_t>());
    GgufMeta value;
    if (!c.ok || !ReadValueInto(c, type, &value)) {
      if (c.ok) {
        parsed.fatal = true;
      } else {
        parsed.truncated = true;
      }
      return parsed;
    }
    parsed.kv.emplace_back(std::string(key), std::move(value));
  }
  if (n_tensors > (1ULL << 26)) {
    parsed.fatal = true;
    return parsed;
  }
  parsed.tensors.reserve(static_cast<std::size_t>(n_tensors));
  for (std::uint64_t i = 0; i < n_tensors; ++i) {
    LocalTensor tensor;
    tensor.name = c.ReadString();
    const std::uint32_t n_dims = c.Read<std::uint32_t>();
    if (!c.ok || n_dims > 8 || n_dims == 0) {
      if (c.ok) {
        parsed.fatal = true;
      } else {
        parsed.truncated = true;
      }
      return parsed;
    }
    tensor.dimensions.resize(n_dims);
    for (std::uint32_t d = 0; d < n_dims; ++d) {
      tensor.dimensions[d] = c.Read<std::uint64_t>();
    }
    tensor.type = c.Read<std::uint32_t>();
    tensor.offset = c.Read<std::uint64_t>();
    if (!c.ok) {
      parsed.truncated = true;
      return parsed;
    }
    parsed.tensors.push_back(std::move(tensor));
  }
  parsed.header_length = static_cast<std::uint64_t>(c.p - base);
  return parsed;
}

std::filesystem::path NthShardPath(const std::filesystem::path& first,
                                   std::uint32_t index) {
  std::string name = first.filename().string();
  const std::size_t pos = name.find("-00001-of-");
  if (pos == std::string::npos) {
    return first;
  }
  char digits[6];
  std::snprintf(digits, sizeof(digits), "%05u", index + 1U);
  name.replace(pos + 1, 5, digits);
  return first.parent_path() / name;
}

std::uint64_t AlignUp(std::uint64_t value, std::uint64_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

}  // namespace

std::unique_ptr<GsqReader> GsqReader::OpenFile(const std::string& path,
                                               std::string* error_msg,
                                               bool map_data) {
  auto fail = [&](const std::string& message) {
    if (error_msg != nullptr) {
      *error_msg = message;
    }
    return nullptr;
  };
  auto reader = std::unique_ptr<GsqReader>(new GsqReader());

  std::vector<ParsedShard> parsed_shards;
  std::vector<std::uint64_t> data_sizes;  // data region bytes per shard
  std::uint64_t alignment = 32;
  std::uint32_t total_shards = 1;

  for (std::uint32_t s = 0; s < total_shards; ++s) {
    const std::filesystem::path shard_path =
        s == 0 ? std::filesystem::path(path) : NthShardPath(path, s);
    std::error_code ec;
    const std::uint64_t file_size =
        static_cast<std::uint64_t>(std::filesystem::file_size(shard_path, ec));
    if (ec) {
      return fail("cannot stat " + shard_path.string() + ": " + ec.message());
    }

    Shard shard;
    shard.path = shard_path;
    ParsedShard parsed;

    if (map_data) {
      const int fd = ::open(shard_path.c_str(), O_RDONLY);
      if (fd < 0) {
        return fail("cannot open " + shard_path.string());
      }
      void* mapped = ::mmap(nullptr, static_cast<std::size_t>(file_size),
                            PROT_READ, MAP_PRIVATE, fd, 0);
      if (mapped == MAP_FAILED) {
        ::close(fd);
        return fail("cannot mmap " + shard_path.string());
      }
      shard.fd = fd;
      shard.mapped = mapped;
      shard.mapped_size = static_cast<std::size_t>(file_size);
      std::size_t prefix =
          static_cast<std::size_t>(std::min<std::uint64_t>(file_size, kGrowStep));
      while (true) {
        parsed = ParseHeaderSection(static_cast<const char*>(mapped), prefix);
        if (!parsed.truncated || prefix >= file_size) {
          break;
        }
        prefix = static_cast<std::size_t>(
            std::min<std::uint64_t>(file_size, prefix + kGrowStep));
      }
    } else {
      const int fd = ::open(shard_path.c_str(), O_RDONLY);
      if (fd < 0) {
        return fail("cannot open " + shard_path.string());
      }
      std::size_t have = 0;
      std::size_t target = static_cast<std::size_t>(
          std::min<std::uint64_t>(file_size, kGrowStep));
      shard.blob.resize(target);
      while (have < target) {
        const ssize_t got =
            ::read(fd, shard.blob.data() + have, target - have);
        if (got <= 0) {
          break;
        }
        have += static_cast<std::size_t>(got);
      }
      ::close(fd);
      while (true) {
        parsed = ParseHeaderSection(shard.blob.data(), have);
        if (!parsed.truncated || have >= file_size) {
          break;
        }
        const std::size_t next_target = static_cast<std::size_t>(
            std::min<std::uint64_t>(file_size, have + kGrowStep));
        if (next_target == have) {
          break;
        }
        shard.blob.resize(next_target);
        const int grow_fd = ::open(shard_path.c_str(), O_RDONLY);
        if (grow_fd < 0) {
          return fail("cannot reopen " + shard_path.string());
        }
        while (have < next_target) {
          const ssize_t got =
              ::pread(grow_fd, shard.blob.data() + have, next_target - have,
                      static_cast<off_t>(have));
          if (got <= 0) {
            break;
          }
          have += static_cast<std::size_t>(got);
        }
        ::close(grow_fd);
      }
    }
    if (parsed.fatal || parsed.truncated) {
      return fail("malformed GGUF header: " + shard_path.string());
    }
    if (s == 0 && parsed.tensors.empty()) {
      return fail("no tensors in " + shard_path.string());
    }

    // First shard decides alignment and shard count; KV is first-wins.
    for (auto& [key, value] : parsed.kv) {
      reader->metadata_.emplace(key, std::move(value));
    }
    if (s == 0) {
      if (const auto* declared = reader->FindMetadata("general.alignment");
          declared != nullptr) {
        if (const auto* raw = std::get_if<std::uint64_t>(&declared->value);
            raw != nullptr && *raw > 0) {
          alignment = *raw;
        }
      }
      if (const auto split_count = reader->GetMetadataUint64("split.count");
          split_count.has_value() && *split_count > 1 && *split_count <= 4096) {
        total_shards = static_cast<std::uint32_t>(*split_count);
      }
    }
    shard.data_start = AlignUp(parsed.header_length, alignment);

    GgufRegion region;
    region.data = shard.mapped;
    region.size = static_cast<std::size_t>(file_size);
    region.file_descriptor = shard.fd;
    region.source_path = shard.path;
    reader->regions_.push_back(region);
    data_sizes.push_back(file_size > shard.data_start
                             ? file_size - shard.data_start
                             : 0);
    parsed_shards.push_back(std::move(parsed));
    reader->shards_.push_back(std::move(shard));
  }

  // Flatten tensors. Sizes come from the container's own offset layout, not
  // from any quant type table: a tensor runs to the next tensor's offset,
  // and the last tensor of a shard runs to the end of its data region.
  for (std::size_t s = 0; s < parsed_shards.size(); ++s) {
    const auto& parsed = parsed_shards[s];
    const Shard& shard = reader->shards_[s];
    for (std::size_t i = 0; i < parsed.tensors.size(); ++i) {
      const LocalTensor& local = parsed.tensors[i];
      GgufTensor info;
      info.name = local.name;
      info.dimensions = local.dimensions;
      info.type = static_cast<GgmlType>(local.type);
      info.offset = local.offset;
      std::uint64_t size = 0;
      if (i + 1 < parsed.tensors.size() &&
          parsed.tensors[i + 1].offset > local.offset) {
        size = parsed.tensors[i + 1].offset - local.offset;
      } else if (i + 1 == parsed.tensors.size() &&
                 data_sizes[s] > local.offset) {
        size = data_sizes[s] - local.offset;
      }
      info.size_bytes = static_cast<std::size_t>(size);
      if (shard.mapped != nullptr) {
        info.data = static_cast<const char*>(shard.mapped) + shard.data_start +
                    local.offset;
      }
      reader->tensors_.push_back(info);
    }
  }
  return reader;
}

GsqReader::~GsqReader() {
  for (auto& shard : shards_) {
    if (shard.mapped != nullptr) {
      ::munmap(shard.mapped, shard.mapped_size);
    }
    if (shard.fd != gufo::platform::kInvalidFile) {
      ::close(shard.fd);
    }
  }
}

const GgufTensor* GsqReader::FindTensor(std::string_view name) const
    noexcept {
  for (const auto& tensor : tensors_) {
    if (tensor.name == name) {
      return &tensor;
    }
  }
  return nullptr;
}

const GgufMeta* GsqReader::FindMetadata(std::string_view key) const
    noexcept {
  const auto it = metadata_.find(key);
  return it == metadata_.end() ? nullptr : &it->second;
}

std::optional<std::string_view> GsqReader::GetMetadataString(
    std::string_view key) const {
  const auto* value = FindMetadata(key);
  if (value == nullptr || value->type != GgufValueType::kString) {
    return std::nullopt;
  }
  if (const auto* raw = std::get_if<std::string_view>(&value->value);
      raw != nullptr) {
    return *raw;
  }
  return std::nullopt;
}

std::optional<std::uint64_t> GsqReader::GetMetadataUint64(
    std::string_view key) const {
  const auto* value = FindMetadata(key);
  if (value == nullptr) {
    return std::nullopt;
  }
  switch (value->type) {
    case GgufValueType::kUint8:
    case GgufValueType::kUint16:
    case GgufValueType::kUint32:
    case GgufValueType::kUint64:
      if (const auto* raw = std::get_if<std::uint64_t>(&value->value);
          raw != nullptr) {
        return *raw;
      }
      return std::nullopt;
    default:
      return std::nullopt;
  }
}

std::optional<float> GsqReader::GetMetadataFloat32(std::string_view key) const {
  const auto* value = FindMetadata(key);
  if (value == nullptr) {
    return std::nullopt;
  }
  // Core's metadata variant normalizes both F32 and F64 to double.
  if (value->type == GgufValueType::kFloat32 ||
      value->type == GgufValueType::kFloat64) {
    if (const auto* raw = std::get_if<double>(&value->value); raw != nullptr) {
      return static_cast<float>(*raw);
    }
  }
  return std::nullopt;
}

std::optional<std::uint32_t> GsqReader::GetMetadataUint32(
    std::string_view key) const {
  const auto wide = GetMetadataUint64(key);
  if (!wide.has_value() || *wide > std::numeric_limits<std::uint32_t>::max()) {
    return std::nullopt;
  }
  return static_cast<std::uint32_t>(*wide);
}

std::vector<std::string_view> GsqReader::GetMetadataStringArray(
    std::string_view key) const {
  const auto* value = FindMetadata(key);
  if (value == nullptr) {
    return {};
  }
  if (const auto* raw =
          std::get_if<std::vector<std::string_view>>(&value->value);
      raw != nullptr) {
    return *raw;
  }
  return {};
}

std::uint64_t GsqReader::GetSize() const noexcept {
  std::uint64_t total = 0;
  for (const auto& region : regions_) {
    total += region.size;
  }
  return total;
}

}  // namespace gufo::models::qwen38_flash_next::gsq
