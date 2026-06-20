/*******************************************************************************
 ******************************************************************************/

#ifndef __NXS_VERSION_H
#define __NXS_VERSION_H

/* clang-format off */

/* Detect which version to target */
#if !defined(NXS_TARGET_KNEXUSAPI_VERSION)
/* #pragma message("nxs_version.h: NXS_TARGET_KNEXUSAPI_VERSION is not defined. Defaulting to 300 (OpenCL 3.0)") */
#define NXS_TARGET_KNEXUSAPI_VERSION 300
#endif
#if NXS_TARGET_KNEXUSAPI_VERSION != 100 && \
    NXS_TARGET_KNEXUSAPI_VERSION != 110 && \
    NXS_TARGET_KNEXUSAPI_VERSION != 120 && \
    NXS_TARGET_KNEXUSAPI_VERSION != 200 && \
    NXS_TARGET_KNEXUSAPI_VERSION != 210 && \
    NXS_TARGET_KNEXUSAPI_VERSION != 220 && \
    NXS_TARGET_KNEXUSAPI_VERSION != 300
#pragma message("nxs_version: NXS_TARGET_KNEXUSAPI_VERSION is not a valid value (100, 110, 120, 200, 210, 220, 300). Defaulting to 300 (OpenCL 3.0)")
#undef NXS_TARGET_KNEXUSAPI_VERSION
#define NXS_TARGET_KNEXUSAPI_VERSION 300
#endif


/* OpenCL Version */
#if NXS_TARGET_KNEXUSAPI_VERSION >= 300 && !defined(NXS_VERSION_3_0)
#define NXS_VERSION_3_0  1
#endif
#if NXS_TARGET_KNEXUSAPI_VERSION >= 220 && !defined(NXS_VERSION_2_2)
#define NXS_VERSION_2_2  1
#endif
#if NXS_TARGET_KNEXUSAPI_VERSION >= 210 && !defined(NXS_VERSION_2_1)
#define NXS_VERSION_2_1  1
#endif
#if NXS_TARGET_KNEXUSAPI_VERSION >= 200 && !defined(NXS_VERSION_2_0)
#define NXS_VERSION_2_0  1
#endif
#if NXS_TARGET_KNEXUSAPI_VERSION >= 120 && !defined(NXS_VERSION_1_2)
#define NXS_VERSION_1_2  1
#endif
#if NXS_TARGET_KNEXUSAPI_VERSION >= 110 && !defined(NXS_VERSION_1_1)
#define NXS_VERSION_1_1  1
#endif
#if NXS_TARGET_KNEXUSAPI_VERSION >= 100 && !defined(NXS_VERSION_1_0)
#define NXS_VERSION_1_0  1
#endif

/* Allow deprecated APIs for older OpenCL versions. */
#if NXS_TARGET_KNEXUSAPI_VERSION <= 220 && !defined(NXS_USE_DEPRECATED_KNEXUSAPI_2_2_APIS)
#define NXS_USE_DEPRECATED_KNEXUSAPI_2_2_APIS
#endif
#if NXS_TARGET_KNEXUSAPI_VERSION <= 210 && !defined(NXS_USE_DEPRECATED_KNEXUSAPI_2_1_APIS)
#define NXS_USE_DEPRECATED_KNEXUSAPI_2_1_APIS
#endif
#if NXS_TARGET_KNEXUSAPI_VERSION <= 200 && !defined(NXS_USE_DEPRECATED_KNEXUSAPI_2_0_APIS)
#define NXS_USE_DEPRECATED_KNEXUSAPI_2_0_APIS
#endif
#if NXS_TARGET_KNEXUSAPI_VERSION <= 120 && !defined(NXS_USE_DEPRECATED_KNEXUSAPI_1_2_APIS)
#define NXS_USE_DEPRECATED_KNEXUSAPI_1_2_APIS
#endif
#if NXS_TARGET_KNEXUSAPI_VERSION <= 110 && !defined(NXS_USE_DEPRECATED_KNEXUSAPI_1_1_APIS)
#define NXS_USE_DEPRECATED_KNEXUSAPI_1_1_APIS
#endif
#if NXS_TARGET_KNEXUSAPI_VERSION <= 100 && !defined(NXS_USE_DEPRECATED_KNEXUSAPI_1_0_APIS)
#define NXS_USE_DEPRECATED_KNEXUSAPI_1_0_APIS
#endif

#endif  /* __NXS_VERSION_H */
