#ifndef RT_CPU_HOST_H
#define RT_CPU_HOST_H

// The host as the kernel states it: sockets, cores, threads and NUMA nodes
// read once from sysfs and procfs under $KNEXUS_HOST_ROOT (default "/"), so a
// test can point the reader at a fixture tree. Linux only: elsewhere every
// vector is empty and the counts are zero, and the plugin states none of the
// host properties. Nothing here is measured; the names and units are those
// tabled in docs/Python_API.md under "Host and link properties".

#include <knexus-api.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace nxs {
namespace rt {

struct HostTopology {
  std::string name;                  // proc/cpuinfo "model name"
  nxs_long memory_bytes = -1;        // proc/meminfo MemTotal, bytes
  nxs_long sockets = 0, cores = 0, threads = 0, numa_nodes = 0;
  std::vector<nxs_long> thread_socket, thread_core, thread_numa_node;
  std::vector<nxs_long> numa_memory_bytes;
  std::vector<nxs_long> numa_distance;   // row-major numa_nodes^2
  bool found = false;                    // any cpu topology at all
};

namespace host_detail {

inline std::string readText(const std::filesystem::path &p) {
  std::ifstream in(p);
  if (!in) return std::string();
  std::stringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

inline bool readLong(const std::filesystem::path &p, nxs_long &out) {
  auto text = readText(p);
  if (text.empty()) return false;
  try {
    out = std::stoll(text);
  } catch (...) {
    return false;
  }
  return true;
}

// "0-3,8" -> {0,1,2,3,8}, the kernel's cpulist syntax
inline std::vector<nxs_long> cpulist(const std::string &text) {
  std::vector<nxs_long> out;
  std::stringstream ss(text);
  std::string part;
  while (std::getline(ss, part, ',')) {
    auto dash = part.find('-');
    try {
      if (dash == std::string::npos) {
        if (!part.empty() && part.find_first_not_of(" \t\n") != std::string::npos)
          out.push_back(std::stoll(part));
      } else {
        nxs_long lo = std::stoll(part.substr(0, dash));
        nxs_long hi = std::stoll(part.substr(dash + 1));
        for (nxs_long c = lo; c <= hi; ++c) out.push_back(c);
      }
    } catch (...) {
    }
  }
  return out;
}

// The value of `key` in a meminfo-shaped file, in bytes ("32296296 kB"); a
// NUMA node's line carries a "Node N" prefix.
inline nxs_long meminfoBytes(const std::filesystem::path &p, const std::string &key) {
  std::stringstream ss(readText(p));
  std::string line;
  while (std::getline(ss, line)) {
    auto colon = line.find(':');
    if (colon == std::string::npos) continue;
    std::string head = line.substr(0, colon);
    auto sp = head.find_last_of(" \t");
    std::string last = sp == std::string::npos ? head : head.substr(sp + 1);
    if (last != key) continue;
    std::stringstream val(line.substr(colon + 1));
    nxs_long n = -1;
    std::string unit;
    val >> n >> unit;
    if (n < 0) return -1;
    if (unit == "kB") return n * 1024;
    if (unit == "MB") return n * 1024 * 1024;
    return n;
  }
  return -1;
}

inline nxs_long numberSuffix(const std::string &name, size_t prefix) {
  if (name.size() <= prefix) return -1;
  for (size_t i = prefix; i < name.size(); ++i)
    if (!isdigit((unsigned char)name[i])) return -1;
  return std::stoll(name.substr(prefix));
}

}  // namespace host_detail

inline HostTopology readHostTopology(const std::filesystem::path &root) {
  using namespace host_detail;
  namespace fs = std::filesystem;
  HostTopology h;
  std::error_code ec;

  // proc/cpuinfo, proc/meminfo
  {
    std::stringstream ss(readText(root / "proc" / "cpuinfo"));
    std::string line;
    while (std::getline(ss, line)) {
      auto colon = line.find(':');
      if (colon == std::string::npos) continue;
      std::string head = line.substr(0, colon);
      head.erase(head.find_last_not_of(" \t") + 1);
      if (head == "model name") {
        std::string v = line.substr(colon + 1);
        v.erase(0, v.find_first_not_of(" \t"));
        v.erase(v.find_last_not_of(" \t\r\n") + 1);
        h.name = v;
        break;
      }
    }
    h.memory_bytes = meminfoBytes(root / "proc" / "meminfo", "MemTotal");
  }

  // sys/devices/system/cpu/cpu*/topology
  std::map<nxs_long, std::pair<nxs_long, nxs_long>> cpus;  // cpu -> (socket, core)
  fs::path cpu_dir = root / "sys" / "devices" / "system" / "cpu";
  if (fs::is_directory(cpu_dir, ec)) {
    for (auto &entry : fs::directory_iterator(cpu_dir, ec)) {
      nxs_long cpu = numberSuffix(entry.path().filename().string(), 3);
      if (cpu < 0 || entry.path().filename().string().rfind("cpu", 0) != 0) continue;
      nxs_long pkg = -1, core = -1;
      if (readLong(entry.path() / "topology" / "physical_package_id", pkg) &&
          readLong(entry.path() / "topology" / "core_id", core))
        cpus[cpu] = {pkg, core};
    }
  }
  if (cpus.empty()) return h;
  h.found = true;
  nxs_long max_cpu = cpus.rbegin()->first;
  h.thread_socket.assign(max_cpu + 1, -1);
  h.thread_core.assign(max_cpu + 1, -1);
  h.thread_numa_node.assign(max_cpu + 1, -1);
  std::set<nxs_long> sockets;
  std::set<std::pair<nxs_long, nxs_long>> cores;
  for (auto &[cpu, sc] : cpus) {
    h.thread_socket[cpu] = sc.first;
    h.thread_core[cpu] = sc.second;
    sockets.insert(sc.first);
    cores.insert(sc);
  }
  h.sockets = sockets.size();
  h.cores = cores.size();
  h.threads = cpus.size();

  // sys/devices/system/node/node*/{cpulist,meminfo,distance}
  std::map<nxs_long, fs::path> nodes;
  fs::path node_dir = root / "sys" / "devices" / "system" / "node";
  if (fs::is_directory(node_dir, ec)) {
    for (auto &entry : fs::directory_iterator(node_dir, ec)) {
      auto fname = entry.path().filename().string();
      nxs_long n = numberSuffix(fname, 4);
      if (n < 0 || fname.rfind("node", 0) != 0) continue;
      if (fs::is_regular_file(entry.path() / "cpulist", ec)) nodes[n] = entry.path();
    }
  }
  h.numa_nodes = nodes.size();
  std::vector<std::vector<nxs_long>> distances;
  for (auto &[n, path] : nodes) {
    for (nxs_long cpu : cpulist(readText(path / "cpulist")))
      if (cpu >= 0 && cpu <= max_cpu) h.thread_numa_node[cpu] = n;
    h.numa_memory_bytes.push_back(meminfoBytes(path / "meminfo", "MemTotal"));
    std::vector<nxs_long> row;
    std::stringstream ss(readText(path / "distance"));
    nxs_long d;
    while (ss >> d) row.push_back(d);
    distances.push_back(row);
  }
  h.numa_distance.assign(h.numa_nodes * h.numa_nodes, -1);
  for (nxs_long i = 0; i < h.numa_nodes; ++i)
    for (nxs_long j = 0; j < h.numa_nodes && j < (nxs_long)distances[i].size(); ++j)
      h.numa_distance[i * h.numa_nodes + j] = distances[i][j];
  return h;
}

// The process's host, read once from $KNEXUS_HOST_ROOT or "/".
inline const HostTopology &getHostTopology() {
  static HostTopology s_host = [] {
    const char *env = std::getenv("KNEXUS_HOST_ROOT");
    return readHostTopology(env && *env ? env : "/");
  }();
  return s_host;
}

}  // namespace rt
}  // namespace nxs

#endif  // RT_CPU_HOST_H
