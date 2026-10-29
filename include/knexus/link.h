#ifndef KNEXUS_LINK_H
#define KNEXUS_LINK_H

#include <knexus-api.h>
#include <knexus/device.h>

#include <optional>

namespace knexus {

namespace detail {
class LinkImpl;
}  // namespace detail

// A link between two devices: a device's PCIe link to the host, the PCIe path
// between two PCI-attached devices, or a fabric link a plugin states. Owned by
// the System, because its endpoints may belong to different runtimes; the host
// endpoint is the cpu runtime's device. See docs/Core_API.md, Link.
class Link : public Object<detail::LinkImpl> {
 public:
  using Object::Object;

  // NP_Type ("PCIe", "NVLink", "Ethernet"), NP_LinkGenMax, NP_LinkGenCurrent,
  // NP_LinkWidthMax, NP_LinkWidthCurrent, NP_LinkTopologyLevel (scalars, -1
  // where the kind states no such figure) and NP_Keys.
  std::optional<Property> getProperty(nxs_int prop) const override;

  // Exactly two; an endpoint outside every loaded runtime is an empty Device.
  Devices getDevices() const;
};

typedef Objects<Link> Links;

}  // namespace knexus

#endif  // KNEXUS_LINK_H
