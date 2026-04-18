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
  bool is_filename;

  TTKernel kernel;
  bool loaded;

  ttm::KernelHandle reader_kernel;
  ttm::KernelHandle writer_kernel;
  ttm::KernelHandle compute_kernel;

 public:
  TTLibrary(TTRuntime *rt = nullptr, const std::string &filename = "", nxs_uint library_settings = 0)
      : rt(rt), file(filename), is_filename(true), kernel(this), loaded(false) {
  }
  TTLibrary(TTRuntime *rt, void *library_data, nxs_uint data_size, nxs_uint library_settings = 0)
      : rt(rt), file(reinterpret_cast<char *>(library_data), data_size), is_filename(false), kernel(this), loaded(false) {
  }
  TTLibrary(const TTLibrary &other) = default;

  ~TTLibrary() = default;

  TTKernel *getKernel() { return &kernel; }

  typedef std::vector<uint32_t> CompileTimeArgs;
  typedef std::array<uint32_t, NXS_KERNEL_MAX_ARGS> RunTimeArgs;

  void jitProgram(ttm::Program &program, const ttm::CoreRange &cores, const CompileTimeArgs &compile_time_args);
  void setupCommonRuntime(ttm::Program &program, const RunTimeArgs &run_time_args);
  void setupCoreRuntime(ttm::Program &program, const ttm::CoreCoord &core, const RunTimeArgs &run_time_args);
};

#endif  // RT_TT_LIBRARY_H