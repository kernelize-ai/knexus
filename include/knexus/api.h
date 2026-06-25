#ifndef _KNEXUS_API_H
#define _KNEXUS_API_H

namespace knexus {

enum NXS_Object_Type {
  NT_Buffer,
  NT_Command,
  NT_Device,

};

typedef enum NXS_Object_Type nxs_type;
}  // namespace knexus

#endif  // _KNEXUS_API_H
