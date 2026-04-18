#include <tt_library.h>


void TTLibrary::jitProgram(ttm::Program &program, const ttm::CoreRange &cores, const CompileTimeArgs &compile_time_args) {
    // Q: why does it need cores
    // create 3 source files

    auto load_kernel_str = [&](const char *kernel_enum_name) {
        std::string kernel_str = "#define " + std::string(kernel_enum_name) + "\n";
        if (is_filename) {
            kernel_str += "#include \"" + file + "\"\n";
        } else {
            kernel_str += file;
        }
        return kernel_str;
    };
    std::string reader_kernel_str = load_kernel_str("READER_KERNEL");
    TT_OBJ_CHECK(reader_kernel, ttm::CreateKernelFromString,
        program, reader_kernel_str, cores,
        ttm::DataMovementConfig{.processor = ttm::DataMovementProcessor::RISCV_0,
                        .noc = ttm::NOC::RISCV_0_default,
                        .compile_args = compile_time_args});
    std::string writer_kernel_str = load_kernel_str("WRITER_KERNEL");
    TT_OBJ_CHECK(writer_kernel, ttm::CreateKernelFromString,
        program, writer_kernel_str, cores,
        ttm::DataMovementConfig{.processor = ttm::DataMovementProcessor::RISCV_1,
                        .noc = ttm::NOC::RISCV_1_default,
                        .compile_args = compile_time_args});
    std::string compute_kernel_str = load_kernel_str("COMPUTE_KERNEL");
    TT_OBJ_CHECK(compute_kernel, ttm::CreateKernelFromString,
        program, compute_kernel_str, cores,
        ttm::ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .compile_args = compile_time_args});
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
