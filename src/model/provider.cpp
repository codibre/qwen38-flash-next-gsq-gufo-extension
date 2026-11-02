#include <gufo/ext/generated/qwen38_flash_next_gsq_constants.hpp>
#include <gufo/ext/registry.hpp>
#include <gufo/ext/text.hpp>

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
  });
#endif
}
