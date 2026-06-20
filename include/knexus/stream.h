#ifndef KNEXUS_STREAM_H
#define KNEXUS_STREAM_H

#include <knexus-api.h>
#include <knexus/object.h>

namespace knexus {

namespace detail {
class StreamImpl;
}  // namespace detail

// System class
class Stream : public Object<detail::StreamImpl> {
 public:
  Stream(detail::Impl base);
  using Object::Object;

  std::optional<Property> getProperty(nxs_int prop) const override;

};

typedef Objects<Stream> Streams;

}  // namespace knexus

#endif  // KNEXUS_STREAM_H