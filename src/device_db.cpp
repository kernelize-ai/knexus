#define KNEXUS_LOG_MODULE "device_info"
#include <knexus/log.h>

#include <dirent.h>
#include <knexus/device_db.h>
#include <knexus/utility.h>

#include <fstream>
#include <iostream>
#include <sstream>

using namespace knexus;

static bool initDeviceInfoDB(DeviceInfoMap &devs) {
  iterateEnvPaths("KNEXUS_DEVICE_PATH", "./device_lib",
                  [&](const std::string &path, const std::string &name) {
                    NXSLOG_INFO("File: {}", name);
                    std::string::size_type const p(name.find_last_of('.'));
                    std::string basename = name.substr(0, p);
                    devs.emplace(basename, path);
                  });
  return true;
}

const DeviceInfoMap *knexus::getDeviceInfoDB() {
  static DeviceInfoMap s_device_info_map;
  static bool init = initDeviceInfoDB(s_device_info_map);
  return &s_device_info_map;
}

Info knexus::lookupDeviceInfo(const std::string &archName) {
  const DeviceInfoMap *devmap = getDeviceInfoDB();
  auto ii = devmap->find(archName);
  if (ii != devmap->end()) return ii->second;
  return Info();
}
