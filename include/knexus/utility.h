#ifndef KNEXUS_UTILITY_H
#define KNEXUS_UTILITY_H

#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace knexus {

typedef std::function<void(const std::string &, const std::string &)>
    PathNameFn;
void iterateEnvPaths(const char *envVar, const char *envDefault,
                     const PathNameFn &func);

std::vector<uint8_t> base64Decode(const std::string_view &encoded,
                                  size_t decoded_size);

}  // namespace knexus

#endif  // KNEXUS_SYSTEM_H