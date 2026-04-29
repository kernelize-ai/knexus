#define NEXUS_LOG_MODULE "library"
#include <nexus/log.h>

#include "_info_impl.h"
#include "_kernel_impl.h"
#include "_library_impl.h"
#include "_runtime_impl.h"

using namespace nexus;
using namespace nexus::detail;

/// @brief Construct a Platform for the current system
LibraryImpl::LibraryImpl(Impl base) : Impl(base) {
  NXSLOG_TRACE("CTOR: {}", getId());
}

LibraryImpl::LibraryImpl(Impl base, Info info) : Impl(base), info(info) {
  // auto name = info.get<std::string_view>("Name");
  NXSLOG_TRACE("CTOR: {}", getId());
  // Iterate over all functions and kernels
  try {
    if (auto functions = info.getNode({"Functions"})) {
      for (auto &kernel : *functions) {
        auto name = kernel.at("Symbol").get<std::string>();
        Info::Node node(kernel);
        getKernel(name, Info(node));
      }
    }
  } catch (...) {
    NXSLOG_ERROR("LibraryImpl: ERROR loading functions");
  }
}

void LibraryImpl::releaseChildren() {
  kernels.clear();
}

nxs_status LibraryImpl::releaseAPI() {
  auto *rt = getParentOfType<RuntimeImpl>();
  if (!rt) return NXS_InvalidObject;
  return (nxs_status)rt->runAPIFunction<NF_nxsReleaseLibrary>(getId());
}

std::optional<Property> detail::LibraryImpl::getProperty(nxs_int prop) const {
  auto *rt = getParentOfType<RuntimeImpl>();
  return rt->getAPIProperty<NF_nxsGetLibraryProperty>(prop, getId());
}

Kernel LibraryImpl::getKernel(const std::string &kernelName, Info info) {
  NXSLOG_TRACE("getKernel: {}", kernelName);
  auto it = kernelMap.find(kernelName);
  if (it != kernelMap.end())
    return it->second;
  auto *rt = getParentOfType<RuntimeImpl>();
  nxs_int kid =
      rt->runAPIFunction<NF_nxsGetKernel>(getId(), kernelName.c_str());
  Kernel kern(Impl(this, kid), kernelName, info);
  if (nxs_failed(kid)) {
    NXSLOG_ERROR("getKernel: {} not found", kernelName);
  } else {
    kernels.add(kern);
    kernelMap[kernelName] = kern;
  }
  return kern;
}

///////////////////////////////////////////////////////////////////////////////
Library::Library(detail::Impl base) : Object(base) {}

Library::Library(detail::Impl base, Info info) : Object(base, info) {}

Info Library::getInfo() const { NEXUS_OBJ_MCALL(Info(), getInfo); }

std::optional<Property> Library::getProperty(nxs_int prop) const {
  NEXUS_OBJ_MCALL(std::nullopt, getProperty, prop);
}

Kernel Library::getKernel(const std::string &kernelName, Info info) {
  NEXUS_OBJ_MCALL(Kernel(), getKernel, kernelName, info);
}

Kernels Library::getKernels() const { NEXUS_OBJ_MCALL(Kernels(), getKernels); }