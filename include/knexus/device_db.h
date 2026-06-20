#ifndef KNEXUS_DEVICE_DB_H
#define KNEXUS_DEVICE_DB_H

#include <knexus/info.h>

#include <optional>
#include <string>
#include <unordered_map>

namespace knexus {

typedef std::unordered_map<std::string, Info> DeviceInfoMap;

const DeviceInfoMap *getDeviceInfoDB();

Info lookupDeviceInfo(const std::string &archName);

}  // namespace knexus

#endif  // KNEXUS_DEVICE_DB_H