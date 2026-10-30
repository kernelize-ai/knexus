#ifndef RT_NEURON_LIBRARY_H
#define RT_NEURON_LIBRARY_H

#include "neuron.h"

#include <nrt/nrt_experimental.h>

#include <string>
#include <vector>

struct NeuronDevice;
class NeuronLibrary;

// One input or output tensor of a loaded NEFF, from nrt_get_model_tensor_info.
// The tensor's name is what libnrt binds by (nrt_add_tensor_to_tensor_set).
struct NeuronTensorInfo {
  std::string name;
  nrt_tensor_usage_t usage = NRT_TENSOR_USAGE_INPUT;
  size_t size = 0;  // bytes
  nrt_dtype_t dtype = NRT_DTYPE_UNKNOWN;
  std::vector<nxs_long> shape;

  bool isInput() const { return usage == NRT_TENSOR_USAGE_INPUT; }
  std::string describe() const;  // "a: INPUT float32[128,512] (262144 bytes)"
};

// libnrt's name for a dtype ("float32", "bfloat16"), and the matching KNexus
// data type (NXS_DataType_Undefined where there is none, the fp8 kinds, say).
const char *nrtDtypeName(nrt_dtype_t dtype);
nxs_data_type nrtDtypeToNxs(nrt_dtype_t dtype);

// The kernel of a library: a NEFF is one compiled graph with one entry point,
// so the kernel is the loaded model itself. Its arguments are the model's
// tensors, inputs first then outputs, each in nrt_get_model_tensor_info order.
class NeuronKernel {
  NeuronLibrary *library;
  std::string name;  // the name it was first looked up by

 public:
  NeuronKernel(NeuronLibrary *lib = nullptr) : library(lib) {}
  NeuronLibrary *getLibrary() const { return library; }
  const std::string &getName() const { return name; }
  void setName(const std::string &n) {
    if (name.empty()) name = n;
  }
};

// A library: one NEFF loaded onto its device's logical core with nrt_load.
// libnrt keeps its own copy of the NEFF, so the bytes are not held here. The
// model is loaded by load(), not the constructor, so a failure is a status,
// and unloaded by unload() (or the destructor, for one still live at
// teardown): the pool reuses a released slot without destroying it.
class NeuronLibrary {
  NeuronDevice *device;
  std::string path;  // the NEFF's file, empty for one created from bytes
  size_t neff_size = 0;
  nrt_model_t *model = nullptr;
  std::vector<NeuronTensorInfo> tensors;  // inputs, then outputs
  size_t num_inputs = 0;
  NeuronKernel kernel;

 public:
  NeuronLibrary(NeuronDevice *dev = nullptr, const std::string &file = "")
      : device(dev), path(file), kernel(this) {}
  NeuronLibrary(const NeuronLibrary &) = delete;
  NeuronLibrary &operator=(const NeuronLibrary &) = delete;
  ~NeuronLibrary() { unload(); }

  nxs_status load(const void *neff, size_t size);
  nxs_status loadFile();  // reads `path`, then load()
  nxs_status unload();

  NeuronDevice *getDevice() const { return device; }
  const std::string &getPath() const { return path; }
  size_t getNeffSize() const { return neff_size; }
  nrt_model_t *getModel() const { return model; }
  const std::vector<NeuronTensorInfo> &getTensors() const { return tensors; }
  size_t getNumInputs() const { return num_inputs; }
  NeuronKernel *getKernel() { return &kernel; }

  // The tensor `name` names, -1 where the model has none.
  int findTensor(const char *name) const;
};

#endif  // RT_NEURON_LIBRARY_H
