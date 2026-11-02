# Qwen3.8 Flash-Next GSQ (256-expert Coder variants) model extension.
#
# Vendored from gufo src/models/qwen38_flash_next at PR gufo-org/gufo#939
# plus the qwen family pieces this model needs (headers under src/family).
# Out-of-tree rules: link gufo_ext_runtime only; never gufo_core/gufo_qwen_*.

gufo_extension_target(
  NAME gufo_qwen38_flash_next_gsq
  SOURCES
    src/model/config.cpp
    src/model/continuation_layout.cpp
    src/model/ngram.cpp
    src/model/rope_scaling.cpp
    src/model/weights.cpp
    src/model/bench_yarn.cpp
    src/model/probe.cpp
    src/model/continuation_identity.cpp
    src/model/extension.cpp
    src/family/vision/image_cache.cpp
    src/family/vision/prompt.cpp
  LINK_LIBRARIES
    gufo_ext_runtime
    Threads::Threads
)

if(TARGET gufo_qwen38_flash_next_gsq)
  target_include_directories(gufo_qwen38_flash_next_gsq PUBLIC
    "$<BUILD_INTERFACE:${GUFO_EXTENSION_BUILD_ROOT}/src>"
  )
endif()

if(TARGET gufo_qwen38_flash_next_gsq AND ENGINE_ENABLE_HIP)
  set(QFN_GSQ_MMQ_DIR "${GUFO_EXTENSION_BUILD_ROOT}/src/model/kernels/rocm/mmq")
  set(QFN_GSQ_MMQ_SOURCES
    ${QFN_GSQ_MMQ_DIR}/qfn_ggml_stubs.hip.cpp
    ${QFN_GSQ_MMQ_DIR}/qfn_mmq.hip.cpp
    ${QFN_GSQ_MMQ_DIR}/mmid.hip.cpp
    ${QFN_GSQ_MMQ_DIR}/mmvq.hip.cpp
    ${QFN_GSQ_MMQ_DIR}/moe_wave64.hip.cpp
    ${QFN_GSQ_MMQ_DIR}/quantize.hip.cpp)

  add_library(qwen38_flash_next_gsq_mmq STATIC ${QFN_GSQ_MMQ_SOURCES})
  set_source_files_properties(${QFN_GSQ_MMQ_SOURCES} PROPERTIES LANGUAGE HIP)
  set_source_files_properties(${QFN_GSQ_MMQ_DIR}/moe_wave64.hip.cpp
    PROPERTIES COMPILE_OPTIONS "-mwavefrontsize64")
  set_target_properties(qwen38_flash_next_gsq_mmq PROPERTIES
    HIP_STANDARD 17
    HIP_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)
  target_compile_definitions(qwen38_flash_next_gsq_mmq PRIVATE
    GGML_HIP_NO_VMM ENGINE_ENABLE_HIP=1)
  target_compile_options(qwen38_flash_next_gsq_mmq PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3>
    $<$<COMPILE_LANGUAGE:HIP>:-ffast-math>
    $<$<COMPILE_LANGUAGE:HIP>:-fno-finite-math-only>
    $<$<COMPILE_LANGUAGE:HIP>:-Wno-unused-value>
    $<$<COMPILE_LANGUAGE:HIP>:-Wno-unused-command-line-argument>)
  target_include_directories(qwen38_flash_next_gsq_mmq SYSTEM PUBLIC ${QFN_GSQ_MMQ_DIR})
  target_link_libraries(qwen38_flash_next_gsq_mmq PUBLIC gufo_ext_runtime hip::host roc::hipblas)
endif()

gufo_extension_target(
  NAME gufo_qwen38_flash_next_gsq_hip
  SOURCES
    src/model/continuation_adapter.cpp
    src/model/kernels/rocm/blaslt.cpp
    src/model/kernels/rocm/device_model.cpp
    src/model/kernels/rocm/executor.cpp
    src/model/kernels/rocm/batch.cpp
    src/model/engine.cpp
    src/model/runner.cpp
    src/model/drivers.cpp
    src/model/provider.cpp
  LINK_LIBRARIES
    gufo_qwen38_flash_next_gsq
    qwen38_flash_next_gsq_mmq
    gufo_ext_runtime
    hip::host
    roc::hipblas
    roc::hipblaslt
)

if(TARGET gufo_qwen38_flash_next_gsq_hip)
  set(QFN_GSQ_ROCM_DIR "${GUFO_EXTENSION_BUILD_ROOT}/src/model/kernels/rocm")
  gufo_extension_hip_sources(gufo_qwen38_flash_next_gsq_hip
    SOURCES
      src/model/kernels/rocm/kernels.hip.cpp
      src/model/kernels/rocm/w8a8_wave64.hip
      src/family/vision/encoder.hip
      src/family/vision/device_input.hip
  )
  set_source_files_properties(${QFN_GSQ_ROCM_DIR}/w8a8_wave64.hip
    PROPERTIES COMPILE_OPTIONS "-mwavefrontsize64")
  target_compile_definitions(gufo_qwen38_flash_next_gsq_hip PRIVATE QFN_ROCM_BUILD=1)
  if(ENGINE_ENABLE_HIP)
    target_compile_definitions(gufo_qwen38_flash_next_gsq_hip PRIVATE ENGINE_ENABLE_HIP=1)
  endif()
  target_compile_options(gufo_qwen38_flash_next_gsq_hip PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O2>
    $<$<COMPILE_LANGUAGE:HIP>:-O3>
    $<$<COMPILE_LANGUAGE:HIP>:-ffast-math>
    $<$<COMPILE_LANGUAGE:HIP>:-fno-finite-math-only>
    $<$<COMPILE_LANGUAGE:HIP>:-Wno-unused-command-line-argument>)
  target_include_directories(gufo_qwen38_flash_next_gsq_hip SYSTEM PUBLIC
    ${HIPCUB_INCLUDE_DIR}
    ${ROCPRIM_INCLUDE_DIR}
    ${ROCWMMA_INCLUDE_DIR})
  target_include_directories(gufo_qwen38_flash_next_gsq_hip PUBLIC
    "$<BUILD_INTERFACE:${GUFO_EXTENSION_BUILD_ROOT}/src>"
  )
endif()

gufo_extension_target(
  NAME qwen38_flash_next_gsq_projection_ops_test
  SOURCES
    tests/projection_ops_test.cpp
  LINK_LIBRARIES
    gufo_qwen38_flash_next_gsq_hip
    gufo_qwen38_flash_next_gsq
    gufo_ext_runtime
    hip::host
)

if(TARGET qwen38_flash_next_gsq_projection_ops_test)
  add_test(NAME qwen38_flash_next_gsq.projection_ops
    COMMAND qwen38_flash_next_gsq_projection_ops_test)
  set_tests_properties(qwen38_flash_next_gsq.projection_ops
    PROPERTIES LABELS "qwen38_flash_next_gsq;hip;gpu")
endif()

gufo_extension_target(
  NAME qwen38_flash_next_gsq_routed_wmma_ops_test
  SOURCES
    tests/routed_wmma_ops_test.cpp
  LINK_LIBRARIES
    gufo_qwen38_flash_next_gsq_hip
    gufo_qwen38_flash_next_gsq
    gufo_ext_runtime
    hip::host
)

if(TARGET qwen38_flash_next_gsq_routed_wmma_ops_test)
  add_test(NAME qwen38_flash_next_gsq.routed_wmma_ops
    COMMAND qwen38_flash_next_gsq_routed_wmma_ops_test)
  set_tests_properties(qwen38_flash_next_gsq.routed_wmma_ops
    PROPERTIES LABELS "qwen38_flash_next_gsq;hip;gpu")
endif()
