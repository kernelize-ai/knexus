#include <tt_command.h>
#include <tt_runtime.h>
#include <rt_buffer.h>

#include <tt-metalium/bfloat16.hpp>

#include <algorithm>

/************************************************************************
 * @def _cpu_barrier
 * @brief Barrier for CPU fibers
 * @return void
 ***********************************************************************/
nxs_status TTCommand::runCommand(TTDevice *device, nxs_int stream, ttmd::MeshWorkload &workload,
                                 ttmd::MeshCoordinateRange &dev_range, ttm::CoreRange &cores) {
  NXSLOG_INFO("runCommand {} - {},{}", (intptr_t)kernel, cores.start_coord.x,
             cores.start_coord.y);

  assert(kernel);
  auto *library = kernel->getLibrary();
  assert(library);

  ttm::Program program = ttm::CreateProgram();

  // load the compile-time args
  TTLibrary::CompileTimeArgs ctas;

  auto make_cb_config = [&](tt::CBIndex cb_index, size_t tile_count, size_t tile_size_bytes, tt::DataFormat data_format) {
    TT_NOBJ_CHECK(cb_config, ttm::CircularBufferConfig, (tile_count * tile_size_bytes), {{cb_index, data_format}});
    TT_CHECK(cb_config.set_page_size, cb_index, tile_size_bytes);
    return cb_config;
  }; 

  size_t tile_size = 1024;

  for (nxs_uint i = 0; i < getNumConstants(); ++i) {
    auto &cst = consts[i];
    if (std::string("CB") == consts[i].name) {
      size_t tile_count = *(nxs_int*)consts[i].value;
      size_t tile_size_bytes = tile_size * getDataTypeSize(cst.settings);
      auto data_format = getDataFormat(consts[i].settings);
      NXSLOG_INFO("CB size ({}): tile_count={}, tile_size_bytes={}, format={}", i, tile_count, tile_size_bytes,
                 static_cast<int>(data_format));
      auto cb_config = make_cb_config(static_cast<tt::CBIndex>(i), tile_count, tile_size_bytes, data_format);
      TT_CHECK(ttm::CreateCircularBuffer, program, cores, cb_config);
      ctas.push_back(i);
    } else {
      NXSLOG_ERROR("Constant not supported: {}", consts[i].name);
      //assert(0); // unsupported cta
    }
  }

  ctas.push_back(ttm::CreateSemaphore(program, cores, 0));
  ctas.push_back(ttm::CreateSemaphore(program, cores, 0));

  // jit the programs
  nxs_status status = library->jitProgram(device->get(), program, cores, ctas);
  if (!nxs_success(status)) return status;

  // collect uniform args
  TTLibrary::RunTimeArgs rt_args;
  size_t numArgs = getArgsCount();
  assert(numArgs <= NXS_KERNEL_MAX_ARGS - 5);
  for (size_t i = 0; i < numArgs; i++) {
    uint32_t arg_val = *static_cast<uint32_t *>(args[i].value);
    NXSLOG_INFO("Runtime arg: {}={}", i, arg_val);
    rt_args[i] = arg_val;
  }

  // compute persistent grid size
  int total_grid_size = grid_size.x * grid_size.y * grid_size.z;
  int persistent_grid_stride = rt::cdiv(total_grid_size, (int)cores.size());
  NXSLOG_INFO("Total grid size: {}, cores: {}, persistent grid stride: {}", total_grid_size, cores.size(), persistent_grid_stride);

  library->setupCommonRuntime(program, rt_args);

  // set params
  int persistent_grid_idx = 0;
  for (const auto& core : cores) {
    TTLibrary::RunTimeArgs core_rt_args;
    core_rt_args[0] = persistent_grid_idx * persistent_grid_stride;
    core_rt_args[1] = persistent_grid_idx * persistent_grid_stride + persistent_grid_stride;
    if (core_rt_args[1] > total_grid_size)
      core_rt_args[1] = total_grid_size;
    NXSLOG_INFO("Launch params: grid_idx={}, start={}, end={}", persistent_grid_idx,
               core_rt_args[0], core_rt_args[1]);
    library->setupCoreRuntime(program, core, core_rt_args);
    persistent_grid_idx++;
  }

  // local or passed in?
  TT_CHECK(workload.add_program, dev_range, std::move(program));
  return NXS_Success;
}
