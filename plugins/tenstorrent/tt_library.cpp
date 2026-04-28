#include <tt_library.h>

#include <tt-metalium/tt_metal.hpp>


nxs_status TTLibrary::jitProgram(std::shared_ptr<ttmd::MeshDevice> device, ttm::Program &program,
    const ttm::CoreRange &cores, const CompileTimeArgs &ctas) {
    // create 3 source files

    auto load_kernel_str = [&](const char *kernel_enum_name) {
        std::string kernel_str = "#define " + std::string(kernel_enum_name) + "\n";
        if (auto kernel_override_path = std::getenv("TT_KERNEL_OVERRIDE_PATH")) {
          kernel_str += "#include \"" + std::string(kernel_override_path) + "\"\n";
        } else {
        if (is_filename) {
            kernel_str += "#include \"" + file + "\"\n";
        } else {
            kernel_str += file;
        }
        }
        return kernel_str;
    };
    std::string reader_kernel_str = load_kernel_str("READER_KERNEL");
    TT_NOBJ_CHECK(reader_kernel, ttm::CreateKernelFromString,
        program, reader_kernel_str, cores,
        ttm::DataMovementConfig{.processor = ttm::DataMovementProcessor::RISCV_1,
                        .noc = ttm::NOC::RISCV_1_default,
                        .compile_args = ctas});
    std::string writer_kernel_str = load_kernel_str("WRITER_KERNEL");
    TT_NOBJ_CHECK(writer_kernel, ttm::CreateKernelFromString,
        program, writer_kernel_str, cores,
        ttm::DataMovementConfig{.processor = ttm::DataMovementProcessor::RISCV_0,
                        .noc = ttm::NOC::RISCV_0_default,
                        .compile_args = ctas});
    std::string compute_kernel_str = load_kernel_str("COMPUTE_KERNEL");
    TT_NOBJ_CHECK(compute_kernel, ttm::CreateKernelFromString,
        program, compute_kernel_str, cores,
        ttm::ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .compile_args = ctas});

    try {
      TT_CHECK(ttm::detail::CompileProgram, device->get_devices()[0], program);
    } catch (const std::exception& e) {
      // Compiler error — e.what() contains the error message
      NXSLOG_ERROR("Compilation failed: {}", e.what());
      return NXS_InvalidCommand;
    }

    return NXS_Success;
}

void TTLibrary::setupCommonRuntime(ttm::Program &program, const RunTimeArgs &run_time_args) {
    TT_CHECK(ttm::SetCommonRuntimeArgs, program, reader_kernel, run_time_args);
    TT_CHECK(ttm::SetCommonRuntimeArgs, program, writer_kernel, run_time_args);
    TT_CHECK(ttm::SetCommonRuntimeArgs, program, compute_kernel, run_time_args);
}

void TTLibrary::setupCoreRuntime(ttm::Program &program, const ttm::CoreCoord &core, const RunTimeArgs &run_time_args) {
    TT_CHECK(ttm::SetRuntimeArgs, program, reader_kernel, core, run_time_args);
    TT_CHECK(ttm::SetRuntimeArgs, program, writer_kernel, core, run_time_args);
    TT_CHECK(ttm::SetRuntimeArgs, program, compute_kernel, core, run_time_args);
}
