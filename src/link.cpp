#define KNEXUS_LOG_MODULE "link"
#include <knexus/log.h>

#include <knexus/link.h>

#include "_link_impl.h"

using namespace knexus;
using namespace knexus::detail;

LinkImpl::LinkImpl(Impl base, Device a, Device b, LinkFacts _facts)
    : Impl(base), facts(std::move(_facts)) {
  endpoints.add(a);
  endpoints.add(b);
}

void LinkImpl::releaseChildren() { endpoints.clear(); }

std::optional<Property> LinkImpl::getProperty(nxs_int prop) const {
  switch (prop) {
    case NP_Type:
      return Property(facts.type);
    case NP_LinkGenMax:
      return Property(facts.gen_max);
    case NP_LinkGenCurrent:
      return Property(facts.gen_cur);
    case NP_LinkWidthMax:
      return Property(facts.width_max);
    case NP_LinkWidthCurrent:
      return Property(facts.width_cur);
    case NP_LinkTopologyLevel:
      return Property(facts.level);
    case NP_Keys:
      // property ids, as every other object's NP_Keys
      return Property(PropIntVec{NP_Type, NP_LinkGenMax, NP_LinkGenCurrent, NP_LinkWidthMax,
                                 NP_LinkWidthCurrent, NP_LinkTopologyLevel});
    default:
      return std::nullopt;
  }
}

///////////////////////////////////////////////////////////////////////////////
/// Object wrapper - Link
///////////////////////////////////////////////////////////////////////////////
std::optional<Property> Link::getProperty(nxs_int prop) const {
  KNEXUS_OBJ_MCALL(std::nullopt, getProperty, prop);
}

Devices Link::getDevices() const { KNEXUS_OBJ_MCALL(Devices(), getDevices); }
