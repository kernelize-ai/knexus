#ifndef NEXUS_API_LOG_H
#define NEXUS_API_LOG_H

#ifndef NEXUS_LOG_MODULE
#ifndef NXSAPI_LOG_MODULE
#define NEXUS_LOG_MODULE "nxs-api"
#else
#define NEXUS_LOG_MODULE "nxs-api:" NXSAPI_LOG_MODULE
#endif
#endif

#ifndef NEXUS_LOG_PADDING
#define NEXUS_LOG_PADDING "20"
#endif

#if defined(__cplusplus)
#include <nexus/log.h>
#endif

#endif  // NEXUS_API_LOG_H
