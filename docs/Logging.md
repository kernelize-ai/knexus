# Nexus logging

Nexus uses [spdlog](https://github.com/gabime/spdlog) behind a small wrapper (`LogManager` in `api-src/log.cpp`). Call sites use the `NXSLOG_*` macros from `include/nexus/log.h` (or the API header chain that pulls in logging).

## Enabling output (runtime)

Logging is **off by default**: the logger uses a null sink until you opt in.

| Environment variable | Effect |
|----------------------|--------|
| **`NEXUS_LOG_ENABLE`** | If set to a non-zero integer, logging is enabled. If unset or `0`, logging stays disabled (null sink). |
| **`NEXUS_LOG_FILE`** | When logging is enabled: if this is set to a non-empty path, logs go to that file. Otherwise logs go to **stdout**. |
| **`NEXUS_LOG_LEVEL`** | When logging is enabled: sets the **minimum severity** spdlog will print. Lower-severity messages are dropped by the logger. See [Log level](#log-level-nexus_log_level) below. |
| **`NO_COLOR`** | If set to any non-empty value, disables **ANSI color on the module column** (level coloring follows spdlog’s sink behavior). Does not affect file logging. |

Initialization runs when the `LogManager` singleton is first constructed (typically early in process startup). Set these variables **before** starting the process (or `export` them in the shell before `python`, `./your_app`, etc.).

### Log level (`NEXUS_LOG_LEVEL`)

Only applies while **`NEXUS_LOG_ENABLE` is non-zero**. If unset or empty, the level defaults to **`trace`** (most verbose).

You can set the level in either form:

1. **Name** (case-insensitive), matching spdlog’s parser:  
   `trace`, `debug`, `info`, `warning` / `warn`, `error` / `err`, `critical`, `off`, `null` (same as `off`).

2. **Integer** `0`–`6` for spdlog’s `level_enum`:  
   `0` = trace … `6` = off.

Unknown names are treated as **`off`** (no log output; consistent with spdlog’s `from_str` behavior).

### Terminal colors

When output goes to a **TTY** (interactive terminal), Nexus uses spdlog’s **color** stdout sink so level names in the pattern can be colorized. Logs written to a **file** (`NEXUS_LOG_FILE`) are plain text with no ANSI colors. If stdout is **redirected or piped**, the terminal may not be detected as a TTY and colors may be disabled.

**Module column:** When logging to the color stdout sink (not to a file), the padded module name is wrapped in ANSI SGR codes. Plugins and translation units can set a **custom foreground**; otherwise Nexus falls back to defaults by module prefix.

| Mechanism | Purpose |
|-----------|---------|
| **`NXSAPI_LOG_MODULE_COLOR`** | (Before `#include <nexus-api/nxs_log.h>`.) String literal: ANSI SGR **prefix only** (e.g. `"\033[32m"`). Bundled plugins set distinct colors (CUDA green, CPU blue, HIP magenta, Metal bright magenta, Tenstorrent bright cyan). |
| **`NEXUS_LOG_MODULE_COLOR`** | (Before `#include <nexus/log.h>`.) Same as above for core or custom code; use `((const char*)0)` or omit for automatic behavior. |
| **Automatic** (color macro unset or null) | `NEXUS_LOG_MODULE` starting with **`nxs-api`** → cyan (`\033[36m`); otherwise → gray (`\033[90m`). |

Module colors are **disabled** when writing to a file, and also when **`NO_COLOR`** is set (any non-empty value), per [no-color.org](https://no-color.org/).

## Compile-time behavior

### `SPDLOG_ACTIVE_LEVEL`

Defined in `include/nexus/log_manager.h` (unless you define it earlier). It controls which **`NXSLOG_*` macros expand to real calls** versus `(void)0`. If a level is “compiled out,” **no** runtime environment variable can bring those statements back—they are not in the binary.

Default: **`SPDLOG_LEVEL_TRACE`** in non-`NDEBUG` builds, **`SPDLOG_LEVEL_INFO`** in `NDEBUG` builds.

Override before including `nexus/log.h` / `nexus/log_manager.h` if you need a project-wide cutoff (e.g. `-DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_WARN`).

### Module name: `NEXUS_LOG_MODULE`

Each translation unit can define **`NEXUS_LOG_MODULE`** to a string literal **before** the first include of `nexus/log.h` (or any header that pulls it in). That string appears as the module column in the log line.

When using **`#include <nexus-api/nxs_log.h>`** (typical for plugins and the C API layer):

- If you define **`NXSAPI_LOG_MODULE`** (e.g. `"cuda_plugin"`), Nexus expands the module to **`"nxs-api:" NXSAPI_LOG_MODULE`** unless you override **`NEXUS_LOG_MODULE`** yourself.
- Optionally define **`NXSAPI_LOG_MODULE_COLOR`** to a string literal (ANSI SGR open sequence) so that plugin’s module column uses that color.
- Default module for API code without `NXSAPI_LOG_MODULE` is **`"nxs-api"`**.

### Module column width: `NEXUS_LOG_PADDING`

Optional. Integer **minimum field width** for the module column (padding with spaces; longer names are truncated). Implemented in `LogManager::format_module_column`. Values less than 1 are treated as **10**. Defaults differ by header:

- `include/nexus/log.h`: **`10`**
- `include/nexus-api/nxs_log.h`: **`20`** (unless you define `NEXUS_LOG_PADDING` earlier)

## Macros (`NXSLOG_*`)

| Macro | Severity |
|-------|----------|
| `NXSLOG_TRACE` | Trace |
| `NXSLOG_DEBUG` | Debug |
| `NXSLOG_INFO` | Info |
| `NXSLOG_WARN` | Warning |
| `NXSLOG_ERROR` | Error |
| `NXSLOG_CRITICAL` | Critical |

Usage (fmt-style format string and arguments):

```cpp
#define NEXUS_LOG_MODULE "my_component"
#include <nexus/log.h>

NXSLOG_INFO("initialized device {}", id);
```

## CMake option `NEXUS_ENABLE_LOGGING`

The build defines **`NEXUS_ENABLE_LOGGING`** (default `ON` in the top-level `CMakeLists.txt`). It is **not** currently wired to disable logging code paths; **runtime** control uses **`NEXUS_LOG_ENABLE`** and related variables above.

## Related headers

| Header | Role |
|--------|------|
| `include/nexus/log_manager.h` | `LogManager`, spdlog includes, `SPDLOG_ACTIVE_LEVEL` default |
| `include/nexus/log.h` | `NXSLOG_*` macros, `NEXUS_LOG_MODULE` / `NEXUS_LOG_PADDING` defaults |
| `include/nexus-api/nxs_log.h` | API/plugin defaults for `NEXUS_LOG_MODULE` and padding; includes `log.h` in C++ |
