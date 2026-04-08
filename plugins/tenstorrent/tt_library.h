#ifndef RT_TT_LIBRARY_H
#define RT_TT_LIBRARY_H

#include "tenstorrent.h"

#include <array>
#include <string>
#include <vector>

class TTRuntime;
class TTLibrary;

class TTKernel {
  TTLibrary *library;
 public:
  TTKernel(TTLibrary *_l) : library(_l) {}
  TTLibrary *getLibrary() const { return library; }
};

class TTLibrary {
  TTRuntime *rt;
  std::string file;
  bool isFilename;
  TTKernel kernel;
  bool loaded;

  ttm::KernelHandle reader_kernel;
  ttm::KernelHandle writer_kernel;
  ttm::KernelHandle compute_kernel;

 public:
  TTLibrary(TTRuntime *rt = nullptr, const std::string &_file = "", bool _isFilename = false,
            nxs_uint library_settings = 0) : rt(rt), file(_file), isFilename(_isFilename), kernel(this), loaded(false) {
  }
  TTLibrary(const TTLibrary &other) = default;

  ~TTLibrary() = default;

  TTKernel *getKernel() { return &kernel; }

  typedef std::vector<uint32_t> CompileTimeArgs;
  typedef std::array<uint32_t, NXS_KERNEL_MAX_ARGS> RunTimeArgs;

  nxs_status jitProgram(TTMeshDevice device, ttm::Program &program,
                        const ttm::CoreRange &cores, const CompileTimeArgs &compile_time_args);
  void setupCommonRuntime(ttm::Program &program, const RunTimeArgs &run_time_args);
  void setupCoreRuntime(ttm::Program &program, const ttm::CoreCoord &core, const RunTimeArgs &run_time_args);
};

#endif  // RT_TT_LIBRARY_H
