#ifndef RT_UTILITIES_H
#define RT_UTILITIES_H

#include <knexus-api.h>

#include <cstring>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace nxs {
namespace rt {

static nxs_status getPropertyStr(void *property_value,
                                 size_t *property_value_size, const char *name,
                                 size_t len) {
  if (property_value != NULL) {
    if (property_value_size == NULL)
      return NXS_InvalidArgSize;
    else if (*property_value_size < len)
      return NXS_InvalidArgValue;
    std::strncpy((char *)property_value, name, len + 1);
  }
  if (property_value_size != NULL) {
    *property_value_size = len + 1;
  }
  return NXS_Success;
}

static nxs_status getPropertyStr(void *property_value,
                                 size_t *property_value_size,
                                 const std::string &value) {
  return getPropertyStr(property_value, property_value_size, value.c_str(),
                        value.size());
}

static nxs_status getPropertyInt(void *property_value,
                                 size_t *property_value_size, nxs_long value) {
  if (property_value != NULL) {
    if (property_value_size == NULL)
      return NXS_InvalidArgSize;
    else if (*property_value_size < sizeof(value))
      return NXS_InvalidArgValue;
    std::memcpy(property_value, &value, sizeof(value));
  }
  if (property_value_size != NULL) {
    *property_value_size = sizeof(value);
  }
  return NXS_Success;
}

static nxs_status getPropertyFlt(void *property_value,
                                 size_t *property_value_size,
                                 nxs_double value) {
  if (property_value != NULL) {
    if (property_value_size == NULL)
      return NXS_InvalidArgSize;
    else if (*property_value_size < sizeof(value))
      return NXS_InvalidArgValue;
    std::memcpy(property_value, &value, sizeof(value));
  }
  if (property_value_size != NULL) {
    *property_value_size = sizeof(value);
  }
  return NXS_Success;
}

template <typename T>
static nxs_status getPropertyVec(void *property_value,
                                 size_t *property_value_size, const T *values,
                                 size_t num_values) {
  auto size = num_values * sizeof(T);
  if (property_value != NULL) {
    if (property_value_size == NULL)
      return NXS_InvalidArgSize;
    else if (*property_value_size < size)
      return NXS_InvalidArgValue;
    std::memcpy(property_value, values, size);
  }
  if (property_value_size != NULL) {
    *property_value_size = size;
  }
  return NXS_Success;
}

// A string vector crosses the boundary as its strings laid end to end, each
// NUL-terminated, so the byte count is the sum of every length plus one. A
// caller that passes no value buffer is told that count and asks again; an
// empty vector is zero bytes. The core splits it back on the NULs.
static nxs_status getPropertyStrVec(void *property_value,
                                    size_t *property_value_size,
                                    const std::vector<std::string> &values) {
  size_t size = 0;
  for (const auto &v : values) size += v.size() + 1;
  if (property_value != NULL) {
    if (property_value_size == NULL)
      return NXS_InvalidArgSize;
    else if (*property_value_size < size)
      return NXS_InvalidArgValue;
    char *out = (char *)property_value;
    for (const auto &v : values) {
      std::memcpy(out, v.c_str(), v.size() + 1);
      out += v.size() + 1;
    }
  }
  if (property_value_size != NULL) {
    *property_value_size = size;
  }
  return NXS_Success;
}

///////////////////////////////////////////////////////////////////////////////
//  Dump debug
template <typename T>
static std::string print_value(const T &value) {
  std::stringstream ss;
  ss << " - 0x" << std::hex << (nxs_ulong)value;
  return ss.str();
}

static std::string print_value() { return ""; }

template <typename T, typename... Args>
static std::string print_value(T value, Args... args) {
  return print_value(value) + print_value(args...);
}

///////////////////////////////////////////////////////////////////////////////
//  Math utilities
static inline size_t cdiv(size_t a, size_t b) {
  return (a + b - 1) / b;
}

}  // namespace rt
}  // namespace nxs

#endif  // RT_UTILITIES_H
