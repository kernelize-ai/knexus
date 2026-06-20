#include <knexus.h>

#include <iostream>
#include <fstream>
#include <numeric>

std::vector<std::string_view> knexusArgs;

int main() {

  auto sys = knexus::getSystem();
  auto rt = sys.getRuntime(1);

  auto count = rt.getDeviceCount();

  std::cout << "RUNTIME: " << rt.getProperty<std::string>(NP_Name) << " - " << count << std::endl;

  for (int i = 0; i < count; ++i) {
    std::cout << "  Device: " << rt.getProperty<std::string>(i, NP_Name) << " - " << rt.getProperty<std::string>(i, NP_Architecture) << std::endl;
  }
  std::vector<char> data(1024, 1);

  auto dev0 = rt.getDevice(0);

  std::ifstream f("kernel.so", std::ios::binary);
  std::vector<char> soData;
  soData.insert(soData.begin(), std::istream_iterator<char>(f), std::istream_iterator<char>());
  
  auto nlib = dev0.createLibrary("kernel.so");

  auto buf = sys.createBuffer(data.size(), data.data());

  auto cpv = sys.copyBuffer(buf, dev0);
  std::cout << "    CopyBuffer: " << cpv << std::endl;

  auto queId = dev0.createCommandList();
  std::cout << "    CList: " << queId << std::endl;

  auto dev = knexus::lookupDevice("amd-gpu-gfx942");
  if (dev) {
    {
      const char *key = "name";
      auto pval = dev->getProperty<std::string>(key);
      std::cout << "PROP(" << key << "): ";
      if (pval)
        std::cout << *pval;
      else
        std::cout << "NOT FOUND";
      std::cout << std::endl;
    }
    {
      std::vector<std::string> prop_path = {"coreSubsystem", "maxPerUnit"};
      auto pval = dev->getProperty<int64_t>(prop_path);

      // make slash path
      std::string path = std::accumulate(std::begin(prop_path), std::end(prop_path), std::string(),
                                [](std::string &ss, std::string &s)
                                {
                                    return ss.empty() ? s : ss + "/" + s;
                                });
      std::cout << "PROP(" << path << "): ";
      if (pval)
        std::cout << *pval;
      else
        std::cout << "NOT FOUND";
      std::cout << std::endl;
    }

    std::vector<nxs_property> prop_epath = {NP_CoreSubsystem, NP_Count};
    auto eval = dev->getProperty<int64_t>(prop_epath);
  }
  return 0;
}
