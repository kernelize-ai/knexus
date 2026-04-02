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
#define NEXUS_LOG_PADDING 20
#endif

// Per-plugin ANSI foreground for the module column (SGR prefix, e.g. "\033[32m"). Optional.
#ifndef NEXUS_LOG_MODULE_COLOR
#ifdef NXSAPI_LOG_MODULE_COLOR
#define NEXUS_LOG_MODULE_COLOR NXSAPI_LOG_MODULE_COLOR
#else
#define NEXUS_LOG_MODULE_COLOR ((const char*)0)
#endif
#endif

#if defined(__cplusplus)
#include <nexus/log.h>
#endif

#endif  // NEXUS_API_LOG_H
