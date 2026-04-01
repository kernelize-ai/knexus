project(nxs-api)

set(CMAKE_CXX_VISIBILITY_PRESET default)

add_library(nxs-api STATIC
    nxs_api.cpp
    log.cpp
)

target_link_libraries(nxs-api PUBLIC spdlog::spdlog)
