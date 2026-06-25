#ifndef KNEXUS_API_LOG_H
#define KNEXUS_API_LOG_H

#ifndef KNEXUS_LOG_MODULE
#ifndef NXSAPI_LOG_MODULE
#define KNEXUS_LOG_MODULE "nxs-api"
#else
#define KNEXUS_LOG_MODULE "nxs-api:" NXSAPI_LOG_MODULE
#endif
#endif

#ifndef KNEXUS_LOG_PADDING
#define KNEXUS_LOG_PADDING 20
#endif

// Per-plugin ANSI foreground for the module column (SGR prefix, e.g. "\033[32m"). Optional.
#ifndef KNEXUS_LOG_MODULE_COLOR
#ifdef NXSAPI_LOG_MODULE_COLOR
#define KNEXUS_LOG_MODULE_COLOR NXSAPI_LOG_MODULE_COLOR
#else
#define KNEXUS_LOG_MODULE_COLOR ((const char*)0)
#endif
#endif

#if defined(__cplusplus)
#include <knexus/log.h>
#endif

#endif  // KNEXUS_API_LOG_H
