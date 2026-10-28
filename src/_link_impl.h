#ifndef _KNEXUS_LINK_IMPL_H
#define _KNEXUS_LINK_IMPL_H

#include <knexus/link.h>

#include <optional>
#include <string>

namespace knexus {
namespace detail {

// What a Link states: its kind, its two endpoints and the per-kind figures,
// -1 where the kind states none.
struct LinkFacts {
  std::string type;
  nxs_long gen_max = -1, gen_cur = -1;
  nxs_long width_max = -1, width_cur = -1;
  nxs_long level = -1;
};

/// @class LinkImpl
class LinkImpl : public Impl {
  Devices endpoints;
  LinkFacts facts;

 public:
  LinkImpl(Impl base, Device a, Device b, LinkFacts facts);

  void releaseChildren() override;

  std::optional<Property> getProperty(nxs_int prop) const;
  Devices getDevices() const { return endpoints; }
};

}  // namespace detail
}  // namespace knexus

#endif  // _KNEXUS_LINK_IMPL_H
