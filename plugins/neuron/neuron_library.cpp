#include <neuron_device.h>
#include <neuron_library.h>

#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>

const char *nrtDtypeName(nrt_dtype_t dtype) {
  switch (dtype) {
    case NRT_DTYPE_FLOAT32: return "float32";
    case NRT_DTYPE_FP32R: return "fp32r";
    case NRT_DTYPE_FLOAT16: return "float16";
    case NRT_DTYPE_BFLOAT16: return "bfloat16";
    case NRT_DTYPE_FP8_E3: return "fp8_e3";
    case NRT_DTYPE_FP8_E4: return "fp8_e4";
    case NRT_DTYPE_FP8_E5: return "fp8_e5";
    case NRT_DTYPE_UINT8: return "uint8";
    case NRT_DTYPE_UINT16: return "uint16";
    case NRT_DTYPE_UINT32: return "uint32";
    case NRT_DTYPE_UINT64: return "uint64";
    case NRT_DTYPE_INT8: return "int8";
    case NRT_DTYPE_INT16: return "int16";
    case NRT_DTYPE_INT32: return "int32";
    case NRT_DTYPE_INT64: return "int64";
    default: return "unknown";
  }
}

nxs_data_type nrtDtypeToNxs(nrt_dtype_t dtype) {
  switch (dtype) {
    case NRT_DTYPE_FLOAT32: return NXS_DataType_F32;
    case NRT_DTYPE_FLOAT16: return NXS_DataType_F16;
    case NRT_DTYPE_BFLOAT16: return NXS_DataType_BF16;
    case NRT_DTYPE_UINT8: return NXS_DataType_U8;
    case NRT_DTYPE_UINT16: return NXS_DataType_U16;
    case NRT_DTYPE_UINT32: return NXS_DataType_U32;
    case NRT_DTYPE_UINT64: return NXS_DataType_U64;
    case NRT_DTYPE_INT8: return NXS_DataType_I8;
    case NRT_DTYPE_INT16: return NXS_DataType_I16;
    case NRT_DTYPE_INT32: return NXS_DataType_I32;
    case NRT_DTYPE_INT64: return NXS_DataType_I64;
    default: return NXS_DataType_Undefined;
  }
}

std::string NeuronTensorInfo::describe() const {
  std::stringstream ss;
  ss << name << ": " << (isInput() ? "INPUT " : "OUTPUT ") << nrtDtypeName(dtype) << "[";
  for (size_t d = 0; d < shape.size(); ++d) ss << (d ? "," : "") << shape[d];
  ss << "] (" << size << " bytes)";
  return ss.str();
}

// The caller (NeuronRuntime::getLibrary) has initialized libnrt. The model is
// placed on the device's core, never "the first free core" (vnc -1): a
// buffer of the same device must be on the core the model runs on.
nxs_status NeuronLibrary::load(const void *neff, size_t size) {
  if (!device) return NXS_InvalidDevice;
  if (!neff || size == 0) {
    NXSLOG_ERROR("NeuronLibrary: empty NEFF");
    return NXS_InvalidBinary;
  }
  NRT_STATUS st = nrt_load(neff, size, device->vnc, -1, &model);
  if (st != NRT_SUCCESS) {
    NXSLOG_ERROR("nrt_load of a {}-byte NEFF on vnc {} failed: {}", size, device->vnc,
                 nrt_get_status_as_str(st));
    model = nullptr;
    switch (st) {
      case NRT_INVALID:
      case NRT_UNSUPPORTED_NEFF_VERSION:
        return NXS_InvalidBinary;
      case NRT_RESOURCE:
      case NRT_LOAD_NOT_ENOUGH_NC:
      case NRT_FAIL_HOST_MEM_ALLOC:
        return NXS_OutOfResources;
      case NRT_EXEC_NC_BUSY:
        return NXS_DeviceNotAvailable;
      default:
        return NXS_InvalidLibrary;
    }
  }
  neff_size = size;

  nrt_tensor_info_array_t *info = nullptr;
  st = nrt_get_model_tensor_info(model, &info);
  if (st != NRT_SUCCESS) {
    NXSLOG_ERROR("nrt_get_model_tensor_info failed: {}", nrt_get_status_as_str(st));
    unload();
    return NXS_InvalidLibrary;
  }
  std::vector<NeuronTensorInfo> ins, outs;
  for (uint64_t i = 0; i < info->tensor_count; ++i) {
    const nrt_tensor_info_t &ti = info->tensor_array[i];
    NeuronTensorInfo t;
    t.name.assign(ti.name, strnlen(ti.name, NRT_TENSOR_NAME_MAX));
    t.usage = ti.usage;
    t.size = ti.size;
    t.dtype = ti.dtype;
    for (uint32_t d = 0; d < ti.ndim; ++d) t.shape.push_back(ti.shape[d]);
    (t.isInput() ? ins : outs).push_back(t);
  }
  nrt_free_model_tensor_info(info);
  num_inputs = ins.size();
  tensors = ins;
  tensors.insert(tensors.end(), outs.begin(), outs.end());

  NXSLOG_INFO("NeuronLibrary: loaded {} ({} bytes) on vnc {}: {} inputs, {} outputs",
              path.empty() ? "<bytes>" : path, size, device->vnc, ins.size(), outs.size());
  for (const auto &t : tensors) NXSLOG_INFO("  {}", t.describe());
  return NXS_Success;
}

nxs_status NeuronLibrary::loadFile() {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    NXSLOG_ERROR("NeuronLibrary: cannot read NEFF file '{}'", path);
    return NXS_InvalidLibrary;
  }
  std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  return load(bytes.data(), bytes.size());  // libnrt copies; `bytes` goes now
}

nxs_status NeuronLibrary::unload() {
  if (model) {
    NXSLOG_INFO("NeuronLibrary: unload {}", path.empty() ? "<bytes>" : path);
    NRT_STATUS st = nrt_unload(model);
    if (st != NRT_SUCCESS) NXSLOG_WARN("nrt_unload failed: {}", nrt_get_status_as_str(st));
    model = nullptr;
  }
  tensors.clear();
  num_inputs = 0;
  return NXS_Success;
}

int NeuronLibrary::findTensor(const char *name) const {
  if (!name) return -1;
  for (size_t i = 0; i < tensors.size(); ++i)
    if (tensors[i].name == name) return (int)i;
  return -1;
}
