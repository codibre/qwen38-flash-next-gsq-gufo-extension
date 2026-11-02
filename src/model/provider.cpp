#include <gufo/ext/generated/qwen38_flash_next_gsq_constants.hpp>
#include <gufo/ext/registry.hpp>
#include <gufo/ext/text.hpp>
#include <gufo/ext/gguf_quant.hpp>

#include "model/probe.hpp"
#include "model/runner.hpp"

namespace gufo::models::qwen38_flash_next {

#if defined(ENGINE_ENABLE_HIP)
int RunQwen38FlashNextBenchmark(
    const ext::TextBenchOptions& opt,
    const std::shared_ptr<const core::GgufReader>& reader,
    std::chrono::steady_clock::time_point model_load_start);

int RunFlashNextPrompt(const ext::TextPromptOptions& opt,
                       const core::GgufReader& reader,
                       std::chrono::steady_clock::time_point model_load_start);

int RunFlashNextChat(const ext::TextPromptOptions& opt,
                     const core::GgufReader& reader,
                     std::chrono::steady_clock::time_point model_load_start);
#endif

}  // namespace gufo::models::qwen38_flash_next

void qwen38_flash_next_gsq_provider_register(gufo::ext::Registry& registry) {
  // GSQ coder files store some expert slabs with storage type id 42 (Q2_0),
  // which gufo main's GgmlType does not include. Teach the rejected-GGUF
  // extension fallback its geometry (64 elements / 18 bytes per block,
  // matching kQ20BlockBytes in the rocm kernels) so the re-opened headers
  // size correctly. Strict parsing on main still rejects id 42 untouched.
  gufo::ext::RegisterUnknownQuantFormat({42, 64, 18});
  registry.AddProvider(gufo::ext::generated::extension_qwen38_flash_next_gsq::
                           kProvider_flash_next_gsq_coder);
#if defined(ENGINE_ENABLE_HIP)
  registry.AddTextProvider({
      .id = "flash-next-gsq-coder",
      .probe = &gufo::models::qwen38_flash_next::ProbeFlashNext,
      .probe_gguf = &gufo::models::qwen38_flash_next::ProbeFlashNextGguf,
      .create = &gufo::models::qwen38_flash_next::CreateQwenFlashNextTextRunner,
      .bench_driver =
          &gufo::models::qwen38_flash_next::RunQwen38FlashNextBenchmark,
      .prompt_driver = &gufo::models::qwen38_flash_next::RunFlashNextPrompt,
      .chat_driver = &gufo::models::qwen38_flash_next::RunFlashNextChat,
      // The GSQ coder GGUFs carry fused 3-D expert tensors with a ragged
      // leading dimension (k=640 vs 256-element IQ blocks), which the core
      // strict parser rejects; this provider takes the re-opened file from
      // the rejected-GGUF extension fallback (see the fallback PR).
      .handles_layout_rejected = true,
  });
#endif
}
