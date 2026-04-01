#ifndef _NEXUS_RUNTIME_IMPL_H
#define _NEXUS_RUNTIME_IMPL_H

#include <nexus-api.h>
#include <nexus/device.h>

#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#pragma push_macro("NEXUS_LOG_MODULE")
#undef NEXUS_LOG_MODULE
#define NEXUS_LOG_MODULE "runtime"
#include <nexus/log.h>
#pragma pop_macro("NEXUS_LOG_MODULE")

namespace nexus {

namespace detail {
class RuntimeImpl : public Impl {
 public:
  RuntimeImpl(Impl base, const std::string &path);
  ~RuntimeImpl();

  void release();

  std::optional<Property> getProperty(nxs_int prop) const;

  Devices getDevices() const { return devices; }
  Device getDevice(nxs_int deviceId) const;

  template <nxs_function Tfn,
            typename Tfnp = typename nxsFunctionType<Tfn>::type>
  Tfnp getFunction() const {
    return (Tfnp)runtimeFns[Tfn];
  }

  template <nxs_function Tfn, typename... Args>
  nxs_int runAPIFunction(Args... args) {
    nxs_int apiResult = NXS_InvalidDevice;  // invalid runtime
    if (auto *fn = getFunction<Tfn>()) {
      apiResult = (*fn)(args...);
      if (nxs_failed(apiResult))
        NXSLOG_ERROR("{}: {}", nxsGetFuncName(Tfn), nxsGetStatusName(apiResult));
      else
        NXSLOG_TRACE("{}: {}", nxsGetFuncName(Tfn), apiResult);
    } else {
      NXSLOG_ERROR("{}: API not present", nxsGetFuncName(Tfn));
    }
    return apiResult;
  }

  template <nxs_function Tfn, typename... Args>
  std::optional<Property> getAPIProperty(nxs_int prop, Args... args) const {
    if (auto fn = getFunction<Tfn>()) {
      auto npt_prop = nxs_property_type_map[prop];
      switch (npt_prop) {
        case NPT_INT: {
          nxs_long val = 0;
          size_t size = sizeof(val);
          if (nxs_success((*fn)(args..., prop, &val, &size))) {
            NXSLOG_INFO("{}: {} = {}", nxsGetFuncName(Tfn), nxsGetPropName(prop), val);
            return Property(val);
          }
          break;
        }
        case NPT_FLT: {
          nxs_double val = 0.;
          size_t size = sizeof(val);
          if (nxs_success((*fn)(args..., prop, &val, &size))) {
            NXSLOG_INFO("{}: {} = {}", nxsGetFuncName(Tfn), nxsGetPropName(prop), val);
            return Property(val);
          }
          break;
        }
        case NPT_STR: {
          size_t size = 256;
          char name[size];
          name[0] = '\0';
          if (nxs_success((*fn)(args..., prop, &name, &size))) {
            NXSLOG_INFO("{}: {} = {}", nxsGetFuncName(Tfn), nxsGetPropName(prop),
                      static_cast<const char*>(name));
            return std::string(name);
          }
          break;
        }
        case NPT_INT_VEC: {
          nxs_long vals[1024];
          size_t size = sizeof(vals);
          if (nxs_success((*fn)(args..., prop, vals, &size))) {
            NXSLOG_INFO("{}: {} ({} elements)", nxsGetFuncName(Tfn), nxsGetPropName(prop),
                      static_cast<unsigned>(size / sizeof(nxs_long)));
            std::vector<nxs_long> vec(size / sizeof(nxs_long));
            std::memcpy(vec.data(), vals, size);
            return Property(vec);
          }
          break;
        }
        default: {
          NXSLOG_ERROR("{}: Unknown property type for - {}", nxsGetFuncName(Tfn),
                    nxsGetPropName(prop));
          break;
        }
      }
    }
    return std::nullopt;
  }

 private:
  void loadPlugin();

  std::string pluginLibraryPath;
  void *library;
  void *runtimeFns[NXS_FUNCTION_CNT];

  Objects<Device> devices;
};

}  // namespace detail
}  // namespace nexus

#endif  // _NEXUS_RUNTIME_IMPL_H
