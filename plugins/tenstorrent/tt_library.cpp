#include <tt_library.h>

std::string make_lib(const char *def, const std::string &file, bool is_filename) {
  std::string kernel_str = std::string("#define ") + def + "\n";
  kernel_str += " // namespace NAMESPACE\n";
  if (is_filename)
    kernel_str += "#include \"";
  kernel_str += file;
  if (is_filename)
    kernel_str += "\"\n";
  return kernel_str;
}

nxs_status TTLibrary::jitProgram(TTMeshDevice device, ttm::Program &program,
                                 const ttm::CoreRange &cores, const CompileTimeArgs &compile_time_args) {
    // create 3 source files
    NXSLOG_INFO("jitProgram: is_filename={} - {}", isFilename, file);
    std::string reader_kernel_str = make_lib("READER_KERNEL", file, isFilename);
    TT_NOBJ_CHECK(reader_kernel, ttm::CreateKernelFromString,
        program, reader_kernel_str, cores,
        ttm::DataMovementConfig{.processor = ttm::DataMovementProcessor::RISCV_1,
                        .noc = ttm::NOC::RISCV_1_default,
                        .compile_args = compile_time_args});
    std::string writer_kernel_str = make_lib("WRITER_KERNEL", file, isFilename);
    TT_NOBJ_CHECK(writer_kernel, ttm::CreateKernelFromString,
        program, writer_kernel_str, cores,
        ttm::DataMovementConfig{.processor = ttm::DataMovementProcessor::RISCV_0,
                        .noc = ttm::NOC::RISCV_0_default,
                        .compile_args = compile_time_args});
    std::string compute_kernel_str = make_lib("COMPUTE_KERNEL", file, isFilename);
    TT_NOBJ_CHECK(compute_kernel, ttm::CreateKernelFromString,
        program, compute_kernel_str, cores,
        ttm::ComputeConfig{.math_fidelity = MathFidelity::HiFi4,
                    //.fp32_dest_acc_en = false, .dst_full_sync_en = false, .math_approx_mode = false,
                    .compile_args = compile_time_args});

    NXSLOG_INFO("jitProgram: r = {}, w = {}, c = {}", reader_kernel, writer_kernel, compute_kernel);
    try {
      for (auto *dev : device->get_devices())
        ttm::detail::CompileProgram(dev, program);
    } catch (const std::exception& e) {
      NXSLOG_ERROR("jitProgram: failed to compile");
      return NXS_InvalidProgram;
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
